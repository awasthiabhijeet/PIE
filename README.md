  # PIE: Parallel Iterative Edit Models for Local Sequence Transduction
  [WIP]Grammatical Error Correction using BERT  
  
  Code and Pre-trained models accompanying our paper "Parallel Iterative Edit Models for Local Sequence Transduction" (EMNLP-IJCNLP 2019)

  We present PIE, a BERT based architecture for local sequence transduction tasks like Grammatical Error Correction.
  Unlike existing approaches which model GEC as a task of translation from "incorrect" to "correct" language, we pose GEC as local sequence editing task. 
  Sequence editing happens in a sequence labelling set-up where parallel, non-autoregressive edits are made to the input sentence.
  PIE models for GEC are 5 to 15 times faster than existing state of the art architectures and still maintain a competitive accuracy.
  
  
 ## Datasets
 * All the public GEC datasets used in the paper can be obtained from [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data)
 
 ## Pretrained Models
 * [PIE as reported in the paper](https://storage.cloud.google.com/gecabhijeet/pie_model.zip?_ga=2.140659505.-2027708043.1555853547) 
   - trained on a Synethically created GEC dataset starting with BERT's initialization
   - finetuned further on Lang8, NUCLE and FCE datasets
   
 ## Code Description
 **An example usage of code in described in the directory "example_scripts".**
 
 * preprocess.sh
   - Extracts common insertions from a sample training data in the "scratch" directory
   - converts the training data in the form of incorrect tokens and aligned edits
 * pie_train.sh
   - trains a pie model using the converted training data
 * multi_round_infer.sh
   - uses a trained PIE model to obtain edits for incorrect sentences
   - does 4 rounds of iterative editing
   - uses conll-14 test sentences
 * m2_eval.sh
   - evaluates the final output using [m2scorer](https://github.com/nusnlp/m2scorer)
 * end_to_end.sh
   - describes the use of pre-processing, training, inference and evaluation scripts end to end.
 * More information in README.md inside "example_scripts"
  
 **Pre processing and Edits related**
 
 * seq2edits_utils.py
   - contains implementation of edit-distance algorithm.
   - cost for substitution modified as per section A.1 in the paper. 
   - Adapted from [belambert's implimentation](https://github.com/belambert/edit-distance/blob/master/edit_distance/code.py)
 * get_edit_vocab.py : Extracts common insertions (\Sigma_a set as described in paper) from a parallel corpus
 * get_seq2edits.py : Extracts edits aligned to input tokens 
 * tokenize_test.py : tokenize a file containing sentences to be corrected. token_ids obtained go as input to the model.
 * opcodes.py : A class where members are all possible edit operations
 * transform_suffixes.py: Contains logic for suffix transformations
 * tokenization.py : Similar to BERT's implementation, with some GEC specific changes
    
 **PIE model** (uses [tensorflow implementation of BERT](https://github.com/google-research/bert))
 
 * word_edit_model.py: Implementation of PIE for learning from a parallel corpous of incorrect tokens and aligned edits. 
   - logit factorization logic (Keep flag use_bert_more=True to enable logit factorization)
   - parallel sequence labeling   
 * modeling.py : Same as in BERT's implementation
 * modified_modeling.py
   - Rewires attention mask to obtain representations of candidate appends and replacements
   - Used for logit factorization.
 * optimization.py : Same as in BERT's implementation
 
 **Post processing**
 * apply_opcode.py
   - Applies inferred edits from the PIE model to the incorrect sentences. 
   - Handles punctuations and spacings as per requirements of a standard dataset (INFER_MODE).
   - Contains some obvious rules for captialization etc.
   