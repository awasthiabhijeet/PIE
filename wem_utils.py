#util functions for word_edit_model.py

import tensorflow as tf
import time

def list_to_ids(s_list, tokenizer):
	#converst list of strings to list of list of token ids
	result = []
	for item in s_list:
		tokens = item.split()
		ids = tokenizer.convert_tokens_to_ids(tokens)
		result.append(ids)

	return result

def list_embedding_lookup(embedding_table, input_ids,
 use_one_hot_embeddings, vocab_size, embedding_size):
  #input ids is a list of word ids
  #returns sum of word_embeddings corresponding to input ids
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.nn.embedding_lookup(embedding_table, input_ids)
  result = tf.reduce_sum(output,0,keepdims=True)
  #result = tf.expand_dims(result,0)
  print("********* shape of reduce_sum: {} ******".format(result))
  return result

def edit_embedding_loopkup(embedding_table, list_input_ids,
 use_one_hot_embeddings, vocab_size, embedding_size):
  #list_input_ids is a list of list of input ids
  #returns embedding for each list, this represents
  #this represents embedding of phrase represented by list
  list1 = [item[0] for item in list_input_ids]
  list2 = [item[1] for item in list_input_ids]

  if use_one_hot_embeddings:
    one_hot_list1 = tf.one_hot(list1, depth=vocab_size)
    one_hot_list2 = tf.one_hot(list2, depth=vocab_size)
    w1 = tf.matmul(one_hot_list1, embedding_table)
    w2 = tf.matmul(one_hot_list2, embedding_table)
  else:
    w1 = tf.nn.embedding_lookup(embedding_table, list1)
    w2 = tf.nn.embedding_lookup(embedding_table, list2)

  return w1+w2
  


def genealised_cross_entropy(probs, one_hot_labels,q=0.6, k=0):
	prob_mask = tf.to_float(tf.less_equal(probs,k))
	probs = prob_mask * k + (1-prob_mask)*probs
	probs = tf.pow(probs,q)
	probs = 1 - probs
	probs = probs / q
	loss = tf.reduce_sum(probs * one_hot_labels, axis=-1)
	return loss

def expand_embedding_matrix(embedding_matrix,batch_size):
  embedding_matrix = tf.expand_dims(embedding_matrix,0)
  embedding_matrix = tf.tile(embedding_matrix,[batch_size,1,1])
  return embedding_matrix

def timer(gen):
  while True:
    try:
      start_time = time.time()
      item = next(gen)
      elapsed_time = time.time() - start_time
      yield elapsed_time, item
    except StopIteration:
      break
#def expected_edit_embeddings(probs,embedding_matrix, batch_size):
  #probs: B x T x E [E = no. of edits]
  #embedding_matrix: B x E x D
  #output: B x T x D
 