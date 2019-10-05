#tokenize input sentences using word piece tokenizer

import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
from utils import open_w, dump_text_to_list, pretty, read_file_lines, custom_tokenize
import tokenization
import argparse

def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument("--input", type=str, default=None, help="input file having possibly incorrect sentences")
  parser.add_argument("--output_tokens", type=str, default=None, help="tokenized version of input")
  parser.add_argument("--output_token_ids", type=str, default=None, help="token ids corresponding to output_tokens")
  parser.add_argument("--vocab_path", type=str, default=None, help="path to bert's cased vocab file")
  parser.add_argument("--do_spell_check", type="bool",default=False, help="wheter to spell check words")


parser = argparse.ArgumentParser()
add_arguments(parser)
FLAGS, unparsed = parser.parse_known_args()

if FLAGS.do_spell_check:
  print("\n\n******************* DOING SPELL CHECK while tokenizing input *******************\n\n")
else:
  print("\n\n********************* Skipping Spell Check while tokenizing input *******************\n\n")

wordpiece_tokenizer = tokenization.FullTokenizer(FLAGS.vocab_path, do_lower_case=False)
vocab_bert = wordpiece_tokenizer.vocab
vocab_words = [vocab_bert.keys()]


def get_tuple(line):
    
    if FLAGS.do_spell_check:
      line = line.strip().split()
      line = wordpiece_tokenizer.basic_tokenizer._run_spell_check(line)
      line = " ".join(line)    
    tokens = custom_tokenize(line, wordpiece_tokenizer)
    token_ids = wordpiece_tokenizer.convert_tokens_to_ids(tokens)
    #print(tokens)
    #print(token_ids)
    return tokens, token_ids

def write_output(raw_lines, tokens_file, token_ids_file):
    tuples = Parallel(n_jobs=1)(delayed(get_tuple)(
        raw_lines[i]) for i in tqdm(range(len(raw_lines))))

    for i in range(len(tuples)):
        tokens, token_ids = tuples[i]

        # Write text output
        tokens_file.write(' '.join(tokens))
        token_ids_file.write(' '.join(str(x) for x in token_ids))

        tokens_file.write('\n')
        token_ids_file.write('\n')
    return

if __name__=="__main__":
    incorrect_lines = read_file_lines(FLAGS.input, 'incorrect lines')
    with open_w(FLAGS.output_tokens) as tokens_file,\
        open_w(FLAGS.output_token_ids) as token_ids_file:

        pretty.pheader('Tokenizing Incorrect sentences')
        write_output(incorrect_lines, tokens_file, token_ids_file)
