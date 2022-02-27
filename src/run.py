from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob
import timeit

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer, BertTokenizer, ElectraTokenizer,
    get_linear_schedule_with_warmup,
)

from model import VConfig, VPLM, StrucDataset
from utils import (read_wrc_examples, convert_examples_to_features, RawResult, write_predictions)
from utils_evaluate import EvalOpts, main as evaluate_on_wrc

logger = logging.getLogger(__name__)


def set_seed(args):
    r"""
    Fix the random seed for reproduction.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    r"""
    Train the model
    """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    else:
        tb_writer = None

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'start_positions': batch[3],
                      'end_positions': batch[4]}
            if args.method == 'V-PLM':
                inputs.update({'visual_feature': batch[-1]})
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, prefix=str(global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    r"""
    Evaluate the model
    """
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]}
            feature_indices = batch[3]
            if args.method == 'V-PLM':
                inputs.update({'visual_feature': batch[-1]})
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)

    eval_time = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_tag_prediction_file = os.path.join(args.output_dir, "tag_predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    output_result_file = os.path.join(args.output_dir, "qas_eval_results_{}.json".format(prefix))
    output_file = os.path.join(args.output_dir, "eval_matrix_results_{}".format(prefix))

    write_predictions(examples, features, all_results, args.n_best_size, args.max_answer_length, args.do_lower_case,
                      output_prediction_file, output_tag_prediction_file, output_nbest_file, args.verbose_logging)

    # Evaluate
    evaluate_options = EvalOpts(data_file=args.predict_file,
                                root_dir=args.root_dir,
                                pred_file=output_prediction_file,
                                tag_pred_file=output_tag_prediction_file,
                                result_file=output_result_file,
                                out_file=output_file)
    results = evaluate_on_wrc(evaluate_options)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    r"""
    Load and process the raw data.
    """
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached', 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        'Text' if args.method == 'T-PLM' else 'HTML'))
    if not os.path.exists(os.path.dirname(cached_features_file)):
        os.makedirs(os.path.dirname(cached_features_file))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if output_examples or (args.method != 'T-PLM' and not evaluate):
            examples, tag_list = read_wrc_examples(input_file=input_file, root_dir=args.root_dir,
                                                     is_training=not evaluate, tokenizer=tokenizer, method=args.method,
                                                     simplify=True)
            if args.method != 'T-PLM' and not evaluate:
                tag_list = list(tag_list)
                tag_list.sort()
                tokenizer.add_tokens(tag_list)
        else:
            examples = None
    else:
        logger.info("Creating features from dataset file at %s", input_file)

        if args.method != 'T-PLM' and not evaluate:
            examples, tag_list = read_wrc_examples(input_file=input_file, root_dir=args.root_dir,
                                                     is_training=not evaluate, tokenizer=tokenizer, method=args.method,
                                                     simplify=True)
            tag_list = list(tag_list)
            tag_list.sort()
            tokenizer.add_tokens(tag_list)

        examples, _ = read_wrc_examples(input_file=input_file, root_dir=args.root_dir, is_training=not evaluate,
                                          tokenizer=tokenizer, method=args.method, simplify=False)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)
        if args.local_rank in [-1, 0] and args.save_features:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if args.method == "V-PLM":
        all_page_ids = [f.page_id for f in features]
        all_token_to_tag = torch.tensor([f.token_to_tag_index for f in features], dtype=torch.long)
    else:
        all_page_ids, all_token_to_tag = None, None

    if evaluate:
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_feature_index,
                               page_ids=all_page_ids, cnn_feature_dir=args.cnn_feature_dir,
                               token_to_tag=all_token_to_tag)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions,
                               page_ids=all_page_ids, cnn_feature_dir=args.cnn_feature_dir,
                               token_to_tag=all_token_to_tag)

    if output_examples:
        dataset = (dataset, examples, features)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--root_dir", default=None, type=str, required=True,
                        help="the root directory of the raw WebSRC dataset, which contains the HTML files.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Now support bert BERT and Electra models")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument('--method', type=str, required=True, choices=["T-PLM", "H-PLM", "V-PLM"],
                        help="choose the baseline models from T-PLM, H-PLM, and V-PLM")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument('--cnn_feature_dir', type=str, default=None,
                        help="the directory where the tag's cnn feature extracted from the screenshots are saved")
    parser.add_argument('--num_node_block', type=int, default=3,
                        help="the number of the visual information enhanced self-attention block in V-PLM")
    parser.add_argument('--cnn_feature_dim', type=int, default=1024,
                        help="the dimension of the cnn features")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal evaluation.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending "
                             "with step number")
    parser.add_argument('--eval_from_checkpoint', type=int, default=0,
                        help="Only evaluate the checkpoints with prefix larger than or equal to it, beside the final "
                             "checkpoint with no prefix")
    parser.add_argument('--eval_to_checkpoint', type=int, default=None,
                        help="Only evaluate the checkpoints with prefix smaller than it, beside the final checkpoint "
                             "with no prefix")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output "
                             "file.")

    parser.add_argument('--logging_steps', type=int, default=3000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=3000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--save_features', type=bool, default=True,
                        help="whether or not to save the processed features, default is True")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path,
                                                          from_tf=bool('.ckpt' in args.model_name_or_path),
                                                          config=config, cache_dir=args.cache_dir)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is
    # set. Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running
    # `--fp16_opt_level="O2"` will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        tokenizer.save_pretrained(args.output_dir)
        model.resize_token_embeddings(len(tokenizer))
        if args.method == 'V-PLM':
            html_config = VConfig(args, **config.__dict__)
            model = VPLM(model, html_config)
        model.to(args.device)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        if args.model_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        elif args.model_type == 'electra':
            tokenizer = ElectraTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        else:
            raise NotImplementedError()
        if args.method == 'V-PLM':
            bert_config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                                     cache_dir=args.cache_dir)
            bert_model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path,
                                                                       from_tf=bool('.ckpt' in args.model_name_or_path),
                                                                       config=bert_config, cache_dir=args.cache_dir)
            bert_model.resize_token_embeddings(len(tokenizer))
        else:
            bert_model = None

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            try:
                int(global_step)
            except ValueError:
                global_step = ""
            if global_step and int(global_step) < args.eval_from_checkpoint:
                continue
            if global_step and args.eval_to_checkpoint is not None and int(global_step) >= args.eval_to_checkpoint:
                continue
            if args.method == 'V-PLM':
                html_config = VConfig(args, **config.__dict__)
                model = VPLM(bert_model, html_config)
                model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))  # confirmed correct
            else:
                model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
