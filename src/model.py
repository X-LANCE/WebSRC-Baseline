import os

import torch
import torch.nn as nn
from transformers.modeling_bert import BertAttention, BertPreTrainedModel
from transformers import PretrainedConfig
from torch.utils.data import Dataset
import numpy as np


class StrucDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (*torch.Tensor): tensors that have the same size of the first dimension.
        page_ids (list): the corresponding page ids of the input features.
        cnn_feature_dir (str): the direction where the cnn features are stored.
        token_to_tag (torch.Tensor): the mapping from each token to its corresponding tag id.
    """

    def __init__(self, *tensors, page_ids=None, cnn_feature_dir=None, token_to_tag=None):
        tensors = tuple(tensor for tensor in tensors)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors
        self.page_ids = page_ids
        self.cnn_feature_dir = cnn_feature_dir
        self.token_to_tag = token_to_tag
        self._init_cnn_feature()

    def __getitem__(self, index):
        output = [tensor[index] for tensor in self.tensors]
        if self.cnn_feature is not None:
            page_id, ind = self.page_ids[index], self.token_to_tag[index]
            raw_cnn_feature = self.cnn_feature[page_id]
            assert ind.dim() == 1
            cnn_num, cnn_dim = raw_cnn_feature.size()
            ind[ind >= cnn_num] = cnn_num - 1
            ind = ind.unsqueeze(1).repeat([1, cnn_dim])
            cnn_feature = torch.gather(raw_cnn_feature, 0, ind)
            output.append(cnn_feature)
        return tuple(item for item in output)

    def __len__(self):
        return len(self.tensors[0])

    def _init_cnn_feature(self):
        if self.cnn_feature_dir is None:
            self.cnn_feature = None
            return
        self.cnn_feature = {}
        cnn_feature_dir = os.walk(self.cnn_feature_dir)
        for d, _, fs in cnn_feature_dir:
            for f in fs:
                if f.split('.')[-1] != 'npy':
                    continue
                domain = d.split('/')[-3][:2]
                page = f.split('.')[0]
                temp = torch.as_tensor(np.load(os.path.join(d, f)), dtype=torch.float)
                self.cnn_feature[domain + page] = torch.cat([temp, torch.zeros_like(temp[0]).unsqueeze(0)], dim=0)
        return


class VConfig(PretrainedConfig):
    r"""
    The configuration class to store the configuration of V-PLM

    Arguments:
        method (str): the name of the method in use, choice: ['T-PLM', 'H-PLM', 'V-PLM'].
        model_type (str): the model type of the backbone PLM, currently support BERT and Electra.
        num_node_block (int): the number of the visual information enhanced self-attention block in use.
        cnn_feature_dim (int): the dimension of the provided cnn features.
        kwargs (dict): the other configuration which the configuration of the PLM in use needs.
    """
    def __init__(self,
                 method,
                 model_type,
                 num_node_block,
                 cnn_feature_dim,
                 **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.model_type = model_type
        self.num_node_block = num_node_block
        self.cnn_feature_dim = cnn_feature_dim
        self.cat_hidden_size = self.hidden_size
        if self.method == 'V-PLM':
            self.cat_hidden_size += self.cnn_feature_dim


class VBlock(nn.Module):
    r"""
    the visual information enhanced self-attention block.
    """
    def __init__(self, config):
        super().__init__()
        self.method = config.method
        self.attention = BertAttention(config)
        self.dense = nn.Linear(config.cat_hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            inputs,
            visual_feature,
            attention_mask=None,
            head_mask=None
    ):
        if self.method == 'V-PLM':
            assert visual_feature.dim() == 3
            output = torch.cat([inputs, visual_feature], dim=2)
        else:
            output = inputs
        output = self.dense(output)
        output = self.dropout(output)
        output = self.LayerNorm(output + inputs)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dense.weight.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        output = self.attention(output, attention_mask=extended_attention_mask, head_mask=head_mask)[0]

        return output


class VPLM(BertPreTrainedModel):
    r"""
    the V-PLM model.

    Arguments:
        ptm: the Pretrained Language Model backbone in use, currently support BERT and Electra.
        config (VConfig): the configuration for V-PLM.
    """
    def __init__(self, ptm, config: VConfig):
        super(VPLM, self).__init__(config)
        self.base_type = config.model_type
        if config.model_type == 'bert':
            self.ptm = ptm.bert
        elif config.model_type == 'electra':
            self.ptm = ptm.electra
        else:
            raise NotImplementedError()
        self.struc = nn.ModuleList([VBlock(config) for _ in range(config.num_node_block)])
        self.qa_outputs = ptm.qa_outputs

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            visual_feature=None
    ):
        outputs = self.ptm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        for i, layer in enumerate(self.struc):
            sequence_output = layer(sequence_output, visual_feature, attention_mask=attention_mask, head_mask=head_mask)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
