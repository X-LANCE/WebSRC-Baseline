python -u -W ignore run.py                     \
  --train_file ../data/websrc1.0_train_.json   \
  --predict_file ../data/websrc1.0_dev_.json   \
  --root_dir ../data --do_train                \
  --model_type electra --method H-PLM          \
  --model_name_or_path google/electra-large-discriminator \
  --output_dir result/H-PLM_electra/           \
  --per_gpu_train_batch_size 2                 \
  --gradient_accumulation_steps 4              \
  --do_lower_case --num_train_epochs 2