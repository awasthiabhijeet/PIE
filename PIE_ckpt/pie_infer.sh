cd ..


output_dir=scratch
data_dir=scratch
bert_cased_vocab=PIE_ckpt/vocab.txt
bert_config_file=PIE_ckpt/bert_config.json
path_multitoken_inserts=pickles/conll/common_multitoken_inserts.p 
path_inserts=pickles/conll/common_inserts.p 

python3.6 word_edit_model.py \
  --do_predict=True \
  --data_dir=$data_dir \
  --vocab_file=$bert_cased_vocab \
  --bert_config_file=$bert_config_file \
  --max_seq_length=128 \
  --predict_batch_size=16 \
  --output_dir=$output_dir \
  --do_lower_case=False \
  --path_inserts=$path_inserts \
  --path_multitoken_inserts=$path_multitoken_inserts \
  --predict_checkpoint=PIE_ckpt/pie_model.ckpt
  
  