# Description
end_to_end.sh is a sample script which uses a small GEC corpus
obtained from first 1000 sentences of [wi+locness](https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
) GEC dataset 
## this script does 
  * pre-processing to convert training data in the form of incorrect tokens and aligned edits (preprocess.sh)
    - This includes extracting common insertions (\Sigma_a set in the paper) from the train corpus
    - Common insertions define the various types Append and Replacement edits
    - Obtaining a parallel corpus of incorrect tokens and edits
  * trains a PIE model initialized with BERT checkpoint (pie_train.sh)
  * iteratively decodes the corrected output for [conll-14](https://www.comp.nus.edu.sg/~nlp/conll14st.html) test set (multi_round_infer.sh)
  * uses [m2scorer](https://github.com/nusnlp/m2scorer) to evaluate the corrected sentences in conll-14 test set (m2_eval.sh)

# Instructions
* data files are stored in the scratch directory outside example_scripts directory
  - train_incorr_sentences: (1000 incorrect sentences from (wi+locness)
  - train_corr_sentences: (1000 correct sentences from wi+locness)
  - conll_test.txt (test sentences in conll14)
  - official-2014.combined.m2 (m2 file provided by conll14)

* Download one of the pre-trained "cased" bert checkpoints:
  - https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
    or
    https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip
  - Unzip the checkpoints inside the PIE_ckpt directory outside example_scripts directory
  - The PIE_ckpt directory should contain following files:
      - bert_model.ckpt.data-00000-of-00001 (original bert checkpoint)
      - bert_model.ckpt.index
      - bert_model.ckpt.meta
      - bert_config.json
      - vocab.txt (vocab of cased bert)

* Run: ./end_to_end.sh from example_scripts directory

*  The final corrected sentences (multi_round_3_test_predictions.txt)
   are stored in scratch directory outside example_scripts directory
   and results of m2 scorer are displayed after successful run of end_to_end.sh.
   A successfuly trained model on these 1000 sentences should show a F_{0.5} score close to 26.6
   Note that these scripts train PIE model on a much smaller dataset for demonstartion purpose.

* Naming of generated files after "ith round" iterative inference:
  - multi_round_i_test_results.txt (edits inferred by PIE at ith round)
  - multi_round_i_test_predictions.txt ("corrected sentences at ith round")

* Follow comments in multi_round_infer.sh for further description of iterative inference.