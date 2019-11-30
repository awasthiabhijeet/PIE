# Synthetic Data Generation Scripts

## Usage
	* python3 error.py $path_of_correct_file $output_path
	* Example
	  - python3 error.py ../scratch/train_corr_sentences.txt ../scratch
	  - Running above command will create two parallel files corr_sentences.txt and incorr_sentences.txt in ../scratch
	  - Note that the order of sentences in the newly created parallel files will not be same as the original file.

* morphs.txt was created by merging verbs, verbs.aux and noms from [here](https://github.com/ixa-ehu/matxin/blob/master/data/freeling/en/dictionary/verbs)
* [Synthetically created datasets](https://drive.google.com/open?id=1bl5reJ-XhPEfEaPjvO45M7w0yN-0XGOA)