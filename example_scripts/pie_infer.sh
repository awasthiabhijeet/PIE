cd ..


output_dir=scratch/output_1.0_ensemble_1
data_dir=scratch
bert_cased_vocab=PIE_ckpt/vocab.txt
bert_config_file=PIE_ckpt/bert_config.json
path_multitoken_inserts=$data_dir/pickles/common_multitoken_inserts.p 
path_inserts=$data_dir/pickles/common_inserts.p 


echo $tpu_name
python3.6 word_edit_model.py \
  --do_predict=True \
  --data_dir=$data_dir \
  --vocab_file=$bert_cased_vocab \
  --bert_config_file=$bert_config_file \
  --max_seq_length=128 \
  --predict_batch_size=128 \
  --output_dir=$output_dir \
  --do_lower_case=False \
  --use_tpu=True \
  --tpu_name=a3 \
  --tpu_zone=us-central1-a \
  --path_inserts=$path_inserts \
  --path_multitoken_inserts=$path_multitoken_inserts \
  
  