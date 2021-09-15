python -u -W ignore run.py                     \
  --train_file ../data/websrc1.0_train_.json   \
  --predict_file ../data/websrc1.0_dev_.json   \
  --root_dir ../data --do_eval --do_lower_case \
  --model_type electra --method T-PLM          \
  --model_name_or_path google/electra-large-discriminator \
  --output_dir result/T-PLM_electra/           \
  --eval_all_checkpoints