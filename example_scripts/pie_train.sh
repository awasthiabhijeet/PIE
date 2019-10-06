cd ..

if [ $# -ne 4 ]
 then
    echo "Illegal number of parameters"
    exit 1
fi

ensemble_id=$1
use_tpu=True
copy_weight=$2
random_seed=$3
tpu_name=$4

learning_rate=2e-5
mode=large

export data_dir=scratch
export path_inserts=$data_dir/pickles/common_inserts.p
export path_multitoken_inserts=$data_dir/pickles/common_multitoken_inserts.p

echo path_inserts: "$path_inserts"
echo path_multitoken_inserts: "$path_multitoken_inserts"


export CKPT_DIR=PIE_ckpt
export output_dir="$data_dir"/output_"$copy_weight"_ensemble_"$ensemble_id"

echo $output_dir


echo $tpu_name

python3.6 -u word_edit_model.py \
  --random_seed=$random_seed \
  --do_train=True \
  --data_dir=$data_dir \
  --vocab_file=$CKPT_DIR/vocab.txt \
  --bert_config_file=$CKPT_DIR/bert_config.json \
  --max_seq_length=64 \
  --train_batch_size=64 \
  --learning_rate=$learning_rate \
  --num_train_epochs=4 \
  --output_dir=$output_dir \
  --do_lower_case=False \
  --use_tpu=$use_tpu \
  --tpu_name=$tpu_name \
  --tpu_zone=us-central1-a \
  --save_checkpoints_steps=70000 \
  --iterations_per_loop=1000 \
  --num_tpu_cores=8 \
  --copy_weight=$copy_weight \
  --path_inserts=$path_inserts \
  --path_multitoken_inserts=$path_multitoken_inserts \
  --init_checkpoint=$CKPT_DIR/bert_model.ckpt
