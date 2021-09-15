python -u -W ignore run.py                     \
  --train_file ../data/websrc1.0_train_.json   \
  --predict_file ../data/websrc1.0_dev_.json   \
  --root_dir ../data --do_train                \
  --model_type bert --method V-PLM             \
  --model_name_or_path bert-base-uncased       \
  --output_dir result/V-PLM_bert/              \
  --do_lower_case --num_train_epochs 5         \
  --num_node_block 3 --cnn_feature_dir ../data/visual-features