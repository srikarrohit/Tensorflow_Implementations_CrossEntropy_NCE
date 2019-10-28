import os
import pickle
import numpy as np
import scipy.spatial.distance as sp

model_path = './model_python3/'
#loss_model = 'cross_entropy'
loss_model = 'nce'
model_filepath = os.path.join(model_path, 'word2vec_%s_final_skip2.model'%(loss_model))
dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))
input_file = open('word_analogy_dev.txt', 'r')
#output_file = open('./cross_outputs_local/cross_output_skip2chk.txt', 'w')
output_file = open('./nce_outputs_local/nce_output_skip2.txt', 'w')
result = ""

for l in input_file:
  cs = []
  lt_diff_arr = []
  rt_diff_arr = []
  left_set, right_set = l.strip().split("||")
  right_t = right_set.split(",")
  left_t = left_set.split(",")
    
  for lt in left_t:
    lt_word = lt[1:-1].split(":")
    lt_diff = embeddings[dictionary[lt_word[0]]] - embeddings[dictionary[lt_word[1]]]
    lt_diff_arr.append(lt_diff)
	
  for rt in right_t:
    rt_word = rt[1:-1].split(":")
    rt_diff = embeddings[dictionary[rt_word[0]]] - embeddings[dictionary[rt_word[1]]]
    rt_diff_arr.append(rt_diff)
    cs.append(1- sp.cosine(np.mean(lt_diff_arr, axis=0), rt_diff))#find cosine similarity and average
    
  most_illustrative = cs.index(max(cs))
  least_illustrative = cs.index(min(cs))
  
  result += right_set.replace(","," ")+" " + right_t[least_illustrative]+" "+right_t[most_illustrative]+"\n"

output_file.write(result)
output_file.close()