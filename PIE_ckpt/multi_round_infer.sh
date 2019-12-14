cur_dir=$PWD	
cd ..

export output_dir=scratch
export data_dir=scratch
export bert_cased_vocab=PIE_ckpt/vocab.txt
export bert_config_file=PIE_ckpt/bert_config.json

input_file=scratch/conll_test.txt

echo Running Round 0...

#tokenize input sentences into wordpiece tokens (output_tokens.txt)
python3.6 tokenize_input.py \
  --input=$input_file \
  --output_tokens=$data_dir/output_tokens.txt \
  --output_token_ids=$data_dir/test_incorr.txt \
  --vocab_path=$bert_cased_vocab \
  --do_spell_check=True 

#output_tokens is the wordpiece tokenized version of input (output_tokens.txt)
#test_incorr.txt has token_ids of wordpiece tokens

#PIE predicts edit ids for wordpiece tokenids (test_incorr.txt) 
cd $cur_dir
./pie_infer.sh
cd .. 
cp $output_dir/test_results.txt $data_dir/multi_round_0_test_results.txt
#test_results.txt contains edit ids inferred through PIE

#apply edits (test_results.txt) on the wordpiece tokens of input (output_tokens.txt) 
python3.6 apply_opcode.py \
  --vocab_path=$bert_cased_vocab \
  --input_tokens=$data_dir/output_tokens.txt \
  --edit_ids=$data_dir/multi_round_0_test_results.txt \
  --output_tokens=$data_dir/multi_round_0_test_predictions.txt \
  --infer_mode=conll \
  --path_common_inserts=pickles/conll/common_inserts.p \
  --path_common_multitoken_inserts=pickles/conll/common_multitoken_inserts.p \
  --path_common_deletes=pickles/conll/common_deletes.p \


#corrected_sentences: multi_round_0_test_predictions.txt

#iterate above for 3 more rounds and refine the corrected sentences further

for round_id in {1..3};
do
	echo Running Round $round_id ...
	python3.6 tokenize_input.py \
	  --input=$data_dir/multi_round_"$(( round_id - 1 ))"_test_predictions.txt \
	  --output_tokens=$data_dir/output_tokens.txt \
	  --output_token_ids=$data_dir/test_incorr.txt \
	  --vocab_path=$bert_cased_vocab 

	cd $cur_dir
	./pie_infer.sh 
	cd ..
	cp $output_dir/test_results.txt $data_dir/multi_round_"$round_id"_test_results.txt

	python3.6 apply_opcode.py \
	  --vocab_path=$bert_cased_vocab \
	  --input_tokens=$data_dir/output_tokens.txt \
	  --edit_ids=$data_dir/multi_round_"$round_id"_test_results.txt \
	  --output_tokens=$data_dir/multi_round_"$round_id"_test_predictions.txt \
	  --infer_mode=conll \
	  --path_common_inserts=pickles/conll/common_inserts.p \
	  --path_common_multitoken_inserts=pickles/conll/common_multitoken_inserts.p \
	  --path_common_deletes=pickles/conll/common_deletes.p
done
