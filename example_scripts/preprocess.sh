cd ..

path_to_cased_vocab_bert=PIE_ckpt/vocab.txt #bert's cased vocab path
data_dir=scratch

mkdir -p $data_dir/pickles

#preprocessing consists of 2 steps
# 1. extract the common inserts (\Sigma_a in the paper):
# Example usage:
python3.6 get_edit_vocab.py \
    --vocab_path="$path_to_cased_vocab_bert" \
    --incorr_sents=$data_dir/train_incorr_sentences.txt \
    --correct_sents=$data_dir/train_corr_sentences.txt \
    --common_inserts_dir=$data_dir/pickles \
    --size_insert_list=1000 \
    --size_delete_list=1000

# 2. extract edits from incorrect and correct sentence pairs
# Example usage:
python3.6 get_seq2edits.py \
    --vocab_path="$path_to_cased_vocab_bert" \
    --common_inserts_dir=$data_dir/pickles \
    --incorr_sents=$data_dir/train_incorr_sentences.txt \
    --correct_sents=$data_dir/train_corr_sentences.txt \
    --incorr_tokens=$data_dir/train_incorr_tokens.txt \
    --correct_tokens=$data_dir/train_corr_tokens.txt \
    --incorr_token_ids=$data_dir/train_incorr.txt \
    --edit_ids=$data_dir/train_labels.txt

# incorr_token_ids and edit_ids are used by word_edit_model.py for training GEC model

# NOTE: get_seq2edits.py can regect sentence pairs involving q-gram insertions where q > 2
