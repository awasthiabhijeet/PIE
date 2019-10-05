import os
import argparse
import seq2edits_utils
from collections import defaultdict
import tokenization
import argparse
from utils import generator_based_read_file, do_pickle, pretty, custom_tokenize
from collections import Counter, defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.add_argument("--vocab_path", type=str, default=None, help="path to bert's cased vocab file")
  parser.add_argument("--incorr_sents", type=str, default=None, help="path to incorrect sentence file")
  parser.add_argument("--correct_sents", type=str, default=None, help="path to correct sentence file")
  parser.add_argument("--common_inserts_dir", type=str, default="pickles", help="path to store common inserts in pickles")
  parser.add_argument("--size_insert_list", type=int, default=500, help="size of common insertions list")
  parser.add_argument("--size_delete_list", type=int, default=500, help="size of common deletions list")
  # all the datasets can be obtained from here: https://www.cl.cam.ac.uk/research/nl/bea2019st/

parser = argparse.ArgumentParser()
add_arguments(parser)
FLAGS, unparsed = parser.parse_known_args()
wordpiece_tokenizer = tokenization.FullTokenizer(FLAGS.vocab_path, do_lower_case=False)

def merge_dicts(dicts):
    merged = defaultdict(int)
    for d in dicts:
        for elem in d:
            merged[elem] += d[elem]
    return merged

def update_dicts(insert_dict, delete_dict, rejected, processed):
    insert_dict = merge_dicts([p[0] for p in processed]+[insert_dict])
    delete_dict = merge_dicts([p[1] for p in processed]+[delete_dict])
    rejected +=  sum(p[2] for p in processed)
    return insert_dict, delete_dict, rejected

def get_ins_dels(incorr_line, correct_line):
    ins = defaultdict(int)
    dels = defaultdict(int)
    rejected = 0

    incorr_tokens = custom_tokenize(incorr_line, wordpiece_tokenizer, mode="train")
    correct_tokens = custom_tokenize(correct_line, wordpiece_tokenizer, mode="train")
    diffs = seq2edits_utils.ndiff(incorr_tokens, correct_tokens)

    for item in diffs:
        if item[0]=="+":
            if len(item[2:].split())>2:
                return defaultdict(int), defaultdict(int), 1
            ins[item[2:]]+=1
        elif item[0]=="-":
            dels[item[2:]]+=1

    return ins,dels,0

def segregate_insertions(insert_dict):
    #segregates unigram and bigram insetions
    #returns unigram and bigram list
    unigrams = []
    bigrams = []

    for item in insert_dict:
        if len(item.split())==2:
            bigrams.append(item)
        elif len(item.split())==1:
            unigrams.append(item)
        else:
            print("ERROR: we only support upto bigram insertions")

    return unigrams,bigrams

# Read raw data
pretty.pheader('Reading Input')
incorrect_lines_generator = generator_based_read_file(FLAGS.incorr_sents, 'incorrect lines')
correct_lines_generator = generator_based_read_file(FLAGS.correct_sents, 'correct lines')

insert_dict={}
delete_dict={}
rejected = 0 #number of sentences having more q-gram insertion where q>2

for incorrect_lines, correct_lines in zip(incorrect_lines_generator, correct_lines_generator):
    processed_dicts = Parallel(n_jobs=-1)(delayed(get_ins_dels)(*s) for s in tqdm(
        zip(incorrect_lines, correct_lines), total=len(incorrect_lines)))

    insert_dict,delete_dict, rejected=update_dicts(insert_dict, delete_dict, rejected, processed_dicts)

insert_dict=dict(Counter(insert_dict).most_common(FLAGS.size_insert_list))
delete_dict=dict(Counter(delete_dict).most_common(FLAGS.size_delete_list))

#insert_dict corresponds to \Sigma_a in the paper.
#Elements in \Sigma_a are considered for appends and replacements both
unigram_inserts, bigram_inserts = segregate_insertions(insert_dict)

do_pickle(unigram_inserts,os.path.join(FLAGS.common_inserts_dir,"common_inserts.p"))
do_pickle(bigram_inserts,os.path.join(FLAGS.common_inserts_dir,"common_multitoken_inserts.p"))
do_pickle(delete_dict,os.path.join(FLAGS.common_inserts_dir,"common_deletes.p"))