  # PIE: Parallel Iterative Edit Models for Local Sequence Transduction
 Fast Grammatical Error Correction using BERT  
  
  Code and Pre-trained models accompanying our paper "Parallel Iterative Edit Models for Local Sequence Transduction" (EMNLP-IJCNLP 2019)

We present PIE, a BERT based architecture for local sequence transduction tasks like Grammatical Error Correction. Unlike the standard approach of modeling GEC as a task of translation from "incorrect" to "correct" language, we pose GEC as local sequence editing task. We further reduce local sequence editing problem to a sequence labeling setup where we utilize BERT to non-autoregressively label input tokens with edits. We rewire the BERT architecture (without retraining) specifically for the task of sequence editing. We find that PIE models for GEC are 5 to 15 times faster than existing state of the art architectures and still maintain a competitive accuracy. For more details please see the paper.
  
  
 ## Datasets
 * All the public GEC datasets used in the paper can be obtained from [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data)
* [Synthetically created datasets](https://drive.google.com/open?id=1bl5reJ-XhPEfEaPjvO45M7w0yN-0XGOA) (perturbed version of 1 billion word corpus) divided into 5 parts to independently train 5 different ensembles. (all the ensembles are further finetuned using the public GEC datasets mentioned above.)
    
 
 ## Pretrained Models
 * [PIE as reported in the paper](https://storage.googleapis.com/gecabhijeet/pie_model.zip) 
   - trained on a Synethically created GEC dataset starting with BERT's initialization
   - finetuned further on Lang8, NUCLE and FCE datasets
 * **Inference using the pretrained PIE ckpt**
   - Copy the pretrained checkpoint files provided above to PIE_ckpt directory
   - Your PIE_ckpt directory should contain
      - bert_config.json
      - multi_round_infer.sh
      - pie_infer.sh
      - pie_model.ckpt.data-00000-of-00001
      - pie_model.ckpt.index
      - pie_model.ckpt.meta
      - vocab.txt
   - Run: `$ ./multi_round_infer.sh` from PIE_ckpt directory
   - NOTE: If you are using cloud-TPUs for inference, move the PIE_ckpt directory to the cloud bucket and change the paths in pie_infer.sh and multi_round_infer.sh accordingly
   
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
 * tokenize_input.py : tokenize a file containing sentences. token_ids obtained go as input to the model.
 * opcodes.py : A class where members are all possible edit operations
 * transform_suffixes.py: Contains logic for suffix transformations
 * tokenization.py : Similar to BERT's implementation, with some GEC specific changes
    
 **PIE model** (uses [implementation of BERT of bert in Tensorflow](https://github.com/google-research/bert))
 
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
   
 **Creating synthetic GEC dataset**
 * errorify directory contains the scripts we used for perturbing the one-billion-word corpus

## Citing this work
To cite this work, please cite our [EMNLP-IJCNLP 2019 paper](https://www.aclweb.org/anthology/D19-1435.pdf)
```
@inproceedings{awasthi-etal-2019-parallel,
    title = "Parallel Iterative Edit Models for Local Sequence Transduction",
    author = "Awasthi, Abhijeet  and
      Sarawagi, Sunita  and
      Goyal, Rasna  and
      Ghosh, Sabyasachi  and
      Piratla, Vihari",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1435",
    doi = "10.18653/v1/D19-1435",
    pages = "4259--4269",
}
```
## Acknowledgements
This research was partly sponsored by a Google India AI/ML Research Award and Google PhD Fellowship in Machine Learning. We gratefully acknowledge Google's TFRC program for providing us Cloud-TPUs. Thanks to [Varun Patil](https://github.com/pulsejet) for helping us improve the speed of pre-processing and synthetic-data generation pipelines.
