import os
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import generator_based_read_file, do_pickle, pretty, custom_tokenize
from opcodes import Opcodes
from transform_suffixes import SuffixTransform
import tokenization
import argparse
import seq2edits_utils

def add_arguments(parser):
  """Build ArgumentParser."""
  parser.add_argument("--vocab_path", type=str, default=None, help="path to bert's cased vocab file")
  parser.add_argument("--common_inserts_dir", type=str, default="pickles", help="path to load common inserts")
  parser.add_argument("--incorr_sents", type=str, default=None, help="path to incorrect sentence file")
  parser.add_argument("--correct_sents", type=str, default=None, help="path to correct sentence file")
  parser.add_argument("--incorr_tokens", type=str, default=None, help="path to tokenized incorrect sentences")
  parser.add_argument("--correct_tokens", type=str, default=None, help="path to tokenized correct sentences")
  parser.add_argument("--incorr_token_ids", type=str, default=None, help="path to incorrect token ids of sentences")
  parser.add_argument("--edit_ids", type=str, default=None, help="path to edit ids for each sequence in incorr_token_ids")

parser = argparse.ArgumentParser()
add_arguments(parser)
FLAGS, unparsed = parser.parse_known_args()
wordpiece_tokenizer = tokenization.FullTokenizer(FLAGS.vocab_path, do_lower_case=False)

opcodes = Opcodes(
			path_common_inserts=os.path.join(FLAGS.common_inserts_dir,"common_inserts.p"),
    		path_common_multitoken_inserts=os.path.join(FLAGS.common_inserts_dir,"common_multitoken_inserts.p"),
        	use_transforms=True)

def seq2edits(incorr_line, correct_line):
	# Seq2Edits function (Described in Section 2.2 of the paper)
	# obtains edit ids from incorrect and correct tokens 
	# input: incorrect line and correct line
	# output: incorr_tokens, correct_tokens,  incorr token ids, edit ids
	
	#tokenize incorr_line and correct_line
	incorr_tokens = custom_tokenize(incorr_line, wordpiece_tokenizer, mode="train")
	correct_tokens = custom_tokenize(correct_line, wordpiece_tokenizer, mode="train")
	#generate diffs using modified edit distance algorith 
	# (Described in Appendix A.1 of the paper)
	diffs = seq2edits_utils.ndiff(incorr_tokens, correct_tokens)
	# align diffs to get edits
	edit_ids = diffs_to_edits(diffs)

	if not edit_ids:
		return None
	#get incorrect token ids
	incorr_tok_ids = wordpiece_tokenizer.convert_tokens_to_ids(incorr_tokens)
	return incorr_tokens, correct_tokens, incorr_tok_ids, edit_ids

def diffs_to_edits(diffs):
	#converts diffs to edit ids

	prev_edit = None
	edits = []
	for i,op in enumerate(diffs):
		# op has following forms:  " data" (no-edit) or "- data" (delete) or "+ data" (insert)
		# thus op[0] gives the operation and op[2:] gives the argument
		# (see ndiff function in diff_edit_distance)

		if op[0] == " ":
			edits.append(opcodes.CPY)
		elif op[0] == "-":
			edits.append(opcodes.DEL)
		elif op[0] == "+": #APPEND or REPLACE or SUFFIX TRANFORM
			assert len(edits)>0, "+ or - cannot occour in beginning since all sentences were\
								  were prefixed by a [CLS] token"

			q_gram = op[2:] #argument of on while op[0] operates

			if len(q_gram.split()) > 2: #reject q_gram if q>2
				return None

			if edits[-1] == opcodes.CPY: # CASE OF APPEND / APPEND BASED SUFFIX TRANSFROM (e.g. play -> played)
				if q_gram in opcodes.APPEND_SUFFIX: #priority to SUFFIX TRANSFORM
					edits[-1] = opcodes.APPEND_SUFFIX[q_gram]
				elif q_gram in opcodes.APPEND:
					edits[-1] = opcodes.APPEND[q_gram]
				else:
					# appending q_gram is not supported
					#we ignore the append and edits[-1] is retained as a COPY
					pass 

			elif edits[-1] == opcodes.DEL: # CASE of SUFFIX TRANSFORMATION / REPLACE EDIT
				del_token = diffs[i-1][2:] #replaced word				
				# check for transfomation match
				st = SuffixTransform(del_token, q_gram,opcodes).match()
				if st:
					edits[-1] = st
				else:
					#check for replace opration of transformation match failed
					if q_gram in opcodes.REP:
						edits[-1] = opcodes.REP[q_gram] 
					else:
						# replacement with q_gram is not supported
						# we ignore the replacement and UNDO delete by having edits[-1] as COPY
						edits[-1] = opcodes.CPY
			else:
				#since inserts are merged in diffs, edits[-1] is either a CPY or a DEL, if op[0] == "+"
				print("This should never occour")
				exit(1)
	return edits

pretty.pheader('Reading Input')
incorrect_lines_generator = generator_based_read_file(FLAGS.incorr_sents, 'incorrect lines')
correct_lines_generator = generator_based_read_file(FLAGS.correct_sents, 'correct lines')

with open(FLAGS.incorr_tokens,"w") as ic_toks, \
	 open(FLAGS.correct_tokens,"w") as c_toks, \
	 open(FLAGS.incorr_token_ids,"w") as ic_tok_ids, \
	 open(FLAGS.edit_ids,"w") as e_ids:
	for incorrect_lines, correct_lines in zip(incorrect_lines_generator, correct_lines_generator):
	    processed = Parallel(n_jobs=-1)(delayed(seq2edits)(*s) for s in tqdm(
	        zip(incorrect_lines, correct_lines), total=len(incorrect_lines)))

	    processed = [p for p in processed if p]
	    for p in processed:
	    	ic_toks.write(" ".join(p[0]) + "\n")
	    	c_toks.write(" ".join(p[1]) + "\n")
	    	ic_tok_ids.write(" ".join(map(str,p[2])) + "\n")
	    	e_ids.write(" ".join(map(str,p[3])) + "\n")


    


