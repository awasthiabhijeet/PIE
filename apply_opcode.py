"""Utility to apply opcodes to incorrect sentences."""

import pickle
import string
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
import opcodes
from utils import open_w, read_file_lines, pretty, bcolors
from transform_suffixes import apply_transform as apply_suffix_transform
from autocorrect import spell
import tokenization
import argparse

def add_arguments(parser):
  """Build ArgumentParser."""
  parser.add_argument("--vocab_path", type=str, default=None, help="path to bert's cased vocab file")
  parser.add_argument("--input_tokens", type=str, default=None, help="path to possibly incorrect token file")
  parser.add_argument("--edit_ids", type=str, default=None, help="path to edit ids to be applied on input_tokens")
  parser.add_argument("--output_tokens", type=str, default=None, help="path to edited (hopefully corrected) file")
  parser.add_argument("--infer_mode", type=str, default="conll", help="post processing mode bea or conll")
  parser.add_argument("--path_common_inserts",type=str,default=None,help="path of common unigram inserts")
  parser.add_argument("--path_common_multitoken_inserts",type=str,default=None,help="path of common bigram inserts")
  parser.add_argument("--path_common_deletes",type=str,default=None,help="path to common deletions observed in train data")

parser = argparse.ArgumentParser()
add_arguments(parser)
FLAGS, unparsed = parser.parse_known_args()

DO_PARALLEL = False
INFER_MODE=FLAGS.infer_mode

vocab = tokenization.load_vocab(FLAGS.vocab_path)
basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=False,vocab=vocab)
vocab_words = set(x for x in vocab)
common_deletes = pickle.load(open(FLAGS.path_common_deletes,"rb"))
path_common_inserts = FLAGS.path_common_inserts
path_common_multitoken_inserts = FLAGS.path_common_multitoken_inserts
opcodes = opcodes.Opcodes(path_common_inserts, path_common_multitoken_inserts)

if __name__ == '__main__':
    class config:
        INPUT_UNCORRECTED_WORDS = FLAGS.input_tokens
        INPUT_EDITS = FLAGS.edit_ids
        OUTPUT_CORRECTED_WORDS = FLAGS.output_tokens



def fix_apos_break(word, p_word, pp_word):
    #for l'optimse
    if p_word == "'" and pp_word not in ["i","a","s"] and len(pp_word) == 1 and pp_word.isalpha() and word.isalpha():
        return True
    else:
        return False



def apply_opcodes(words_uncorrected, ops_to_apply,
        join_wordpiece_subwords=True, remove_start_end_tokens=True, 
        do_spell_check=True, apply_only_first_edit=False,
        use_common_deletes=True):
    """Applies opcodes to an uncorrected token sequence and returns
    corrected token sequence."""
    # Initialize
    words_corrected = []

    # Loop over each word
    for i, word in enumerate(words_uncorrected):        
        if i >= len(ops_to_apply):
            words_corrected = words_corrected + words_uncorrected[i:]
            break


        op = ops_to_apply[i]

        # Skip if EOS is detected
        if op == opcodes.EOS:
            print("ERROR: EOS opcode:  This should not happen")
            exit(1)
            break

        elif op == opcodes.CPY:
            words_corrected.append(words_uncorrected[i])

        elif op == opcodes.DEL:
            if (words_uncorrected[i] in common_deletes)  or (not use_common_deletes):
                #and (i==len(words_uncorrected) or words_uncorrected[i+1][0:2]!="##")):
                continue
            else:
                words_corrected.append(words_uncorrected[i])

        elif op in opcodes.APPEND.values():
            words_corrected.append(words_uncorrected[i])
            insert_words = key_from_val(op, opcodes.APPEND).split()
            if i==0 and do_spell_check:
                insert_words[0] = insert_words[0].capitalize()
                if len(words_uncorrected) > 1:
                    words_uncorrected[i+1] = words_uncorrected[i+1].lower()
            words_corrected.extend(insert_words)

        elif op in opcodes.REP.values():
            replacement = key_from_val(op, opcodes.REP).split()
            words_corrected.extend(replacement)

        elif apply_suffix_transform(words_uncorrected, i, op, opcodes):
            replacement = apply_suffix_transform(words_uncorrected, i, op, opcodes)
            words_corrected.append(replacement)

        else:
            words_corrected.append(words_uncorrected[i])
            tqdm.write(bcolors.FAIL + 'ERROR: Copying illegal operation (failed transform?) at '
                    + str(words_uncorrected) + bcolors.ENDC)

        if apply_only_first_edit and (op != opcodes.CPY) and (i+1<len(words_uncorrected)):
            words_corrected = words_corrected + words_uncorrected[i+1:]
            break

    words_corrected = join_subwords(words_corrected)
    #print("Removing CLS and SEP")
    words_corrected = words_corrected[1:-1]

    if do_spell_check:
        words_corrected = basic_tokenizer._run_spell_check(words_corrected)

    return words_corrected

def key_from_val(val, entries):
    """Get key from a value in dict."""
    return list(entries.keys())[list(entries.values()).index(val)]

def join_subwords(word_list):
    global vocab
    tmp_word_list = []
    for i,word in enumerate(word_list):
        if i==0 or word_list[i][0:2]!="##":
            tmp_word_list.append(word_list[i])
        else:
            tmp_word_list[-1] = tmp_word_list[-1] + word_list[i][2:]

    result=[]

    for i,word in enumerate(tmp_word_list):
        if INFER_MODE == "bea":
            if i==0:
                result.append(word)
            elif word == 'i':
                result.append(word.capitalize())
            elif word == "-" and result[-1] == "-":
                result[-1] = result[-1] + "-"
            elif word == "'" and result[-1] == "'":
                result[-1] = "''"
            elif word in ["s","m","re","ve","d"] and result[-1] == "'":         #-----------------------> bea SPECIFIC
                result[-1] = "'{}".format(word)
            elif len(result) > 1 and word == "t" and result[-1]=="'" and result[-2][-1]=="n":
                result.pop()
                result[-1] = result[-1] + "'t"
            elif word == "ll" and result[-1] == "'":
                result[-1]="'ll"
            elif len(result) > 1 and fix_apos_break(word, result[-1], result[-2]):
                result.pop()
                result[-1] += "'" + word
            else:
                if len(result)==1:
                    if not tokenization.do_not_split(word):
                        word = word.capitalize()
                #elif (word != 'I') and (word[0].isupper()) and (result[-1] != '.') and (word.lower() in vocab_words):
                #    print("{} ----------------------------------->{}".format(word,word.lower()))
                #    word = word.lower()
                result.append(word)
        elif INFER_MODE=="conll":
            if i==0:
                result.append(word)
            elif word == 'i':
                result.append(word.capitalize())
            elif word=="-" or result[-1][-1] == "-":
                result[-1] = result[-1] + word
            elif word=="/" or result[-1][-1] == "/":
                result[-1] = result[-1] + word
            elif word == "'" and result[-1] == "'":
                result[-1] = "''"
            elif word in ["s","re"] and result[-1] == "'":
                result[-1] = "'{}".format(word)
            elif len(result) > 1 and word=="'" and len(result[-1])>1 and result[-2]=="'":
                main_word = result.pop()
                result[-1] = "'{}'".format(main_word)
            elif len(result) > 1 and len(word)==1 and result[-1]=="'" and len(result[-2])==1: #n't #I'm
                result.pop()
                result[-1]= result[-1] + "'" + word
            else:
                if len(result)==1:
                    if not tokenization.do_not_split(word):
                        word = word.capitalize()

                #if (gpv.use_spell_check) and (word not in vocab) and (spell(word) in vocab):
                #  print("{} --> {}".format(word, spell(word)))
                #  word = spell(word)
                result.append(word)
        else:
            print("wrong infer_mode")
            exit(1)


        if len(result) > 3 and result[-2]=="." and len(result[-3])>3:
            if not tokenization.do_not_split(result[-1]):
                result[-1]=result[-1].capitalize()
        
        #if len(result)>1 and result[-2] == "a" and result[-1].startswith(('a','e','i','o','u','A','E','I','O','U')):
        #    print("{} {}".format(result[-2],result[-1]))
        #    result[-2]="an"

        #if len(result)>1 and result[-2] == "an" and (not result[-1].startswith(('a','e','i','o','u','A','E','I','O','U'))):
        #    print("{} {}".format(result[-2],result[-1]))
        #    result[-2]="a"



    prev_word = None
    post_process_result = []
    for i, word in enumerate(result):
        if word != prev_word or word in {".", "!", "that", "?", "-", "had"}:
            post_process_result.append(word)


        prev_word = word

    return post_process_result


    '''
    elif len(result) > 1 and word=="t" and result[-1]=="'" and result[-2]=="n":
        result.pop()
        result[-1]="n't"
    '''
def split_and_convert_to_ints(words_uncorrected,edits):
    words_uncorrected = words_uncorrected.split(' ')
    edits = edits.split(' ')[0:len(words_uncorrected)]
    edits = list(map(int, edits))    
    return words_uncorrected, edits 


if __name__=="__main__":

    corrected = []

    pretty.pheader('Reading Input')
    edits = read_file_lines(config.INPUT_EDITS)
    #uncorrected = read_file_lines(config.INPUT_UNCORRECTED)
    words_uncorrected = read_file_lines(config.INPUT_UNCORRECTED_WORDS)

    if len(edits) != len(words_uncorrected):
        pretty.fail('FATAL ERROR: Lengths of edits and uncorrected files not equal')
        exit()

       
    pretty.pheader('Splitting and converting to integers')

    if not DO_PARALLEL:
        for i in tqdm(range(len(edits))):
            edits[i] = list(map(int, edits[i].split(' ')))
            #uncorrected[i] = list(map(int, uncorrected[i].split(' ')))
            words_uncorrected[i] = words_uncorrected[i].split(' ')
    else:
        result = Parallel(n_jobs=-1)(delayed(split_and_convert_to_ints)(*s) for s in tqdm(zip(words_uncorrected,edits), total=len(words_uncorrected)))
        words_uncorrected = [item[0] for item in result]
        edits = [item[1] for item in result]    
    
        #if(len(edits[i]) != len(uncorrected[i])):
            #print("edits: {}".format(edits[i]))
            #print("length uncorrected: {}".format(len(uncorrected[i])))
            #tqdm.write((bcolors.WARNING + "WARN: Unequal lengths at line {}".format(i + 1) + bcolors.ENDC))

    pretty.pheader('Applying opcodes')
    with open_w(config.OUTPUT_CORRECTED_WORDS) as out_file:

        #sentences_corrected_inplace = [] #contain copies, should be same length as uncorrected list of list
        #sentences_corrected_insert = [] #contains additional inserts, should be same length as uncorrected list of list

        if DO_PARALLEL:
            s_corrected = Parallel(n_jobs=-1)(delayed(apply_opcodes)(*s) for s in tqdm(zip(words_uncorrected,edits), total=len(words_uncorrected)))
            for line in s_corrected:
                out_file.write(" ".join(line)+"\n")
        else:
            for i in tqdm(range(len(edits))):
                #s_corrected = untokenize(apply_opcodes(words_uncorrected[i], uncorrected[i], edits[i]))
                #print(len(words_uncorrected[i]))
                s_corrected = apply_opcodes(words_uncorrected[i], edits[i])
                s_corrected = " ".join(s_corrected)

                out_file.write(s_corrected)
                out_file.write('\n')
