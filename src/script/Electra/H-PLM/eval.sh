python -u -W ignore run.py                     \
  --train_file ../data/websrc1.0_train_.json   \
  --predict_file ../data/websrc1.0_dev_.json   \
  --root_dir ../data --do_eval --do_lower_case \
  --model_type electra --method H-PLM          \
  --model_name_or_path google/electra-large-discriminator \
  --output_dir result/H-PLM_electra/           \
  --eval_all_checkpoints