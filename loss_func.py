import tensorflow as tf
import numpy as np

def cross_entropy_loss(inputs, true_w):
  u_o = true_w
  v_c = inputs
  u_oTv_c = tf.tensordot(v_c, tf.transpose(u_o), 1)
  A = tf.diag_part(u_oTv_c)
  B = tf.log(tf.reduce_sum(tf.exp(u_oTv_c), 1))
  return tf.subtract(B, A)
    
def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):

  k = 2*len(sample)
  qw = inputs
  rw = tf.reshape(tf.gather(weights, labels), [-1, tf.shape(weights)[1]])
  bw = tf.reshape(tf.gather(biases, labels), [-1,1])
  s1_first_term = tf.reshape(tf.diag_part(tf.tensordot(qw,tf.transpose(rw),1)), [-1,1])
  s1 = tf.add(s1_first_term, bw)

  pn = tf.gather(np.float32(unigram_prob), labels)
  log_kpn = tf.log(tf.scalar_mul(k,pn)+1e-10)#Adding 1e-10
  delta = s1-log_kpn
  first_term =  tf.log(tf.sigmoid(delta)+1e-10)

  neg_rw = tf.gather(weights, sample)
  neg_bw = tf.reshape(tf.gather(biases, np.array(sample)), [-1,1])
  s2 = tf.add(tf.transpose(tf.tensordot(qw,tf.transpose(neg_rw),1)), neg_bw)

  neg_pn = tf.reshape(tf.gather(np.float32(unigram_prob), np.array(sample)),[-1,1])
  neg_log_kpn = tf.log(tf.scalar_mul(k,neg_pn)+1e-10)

  neg_delta = s2-neg_log_kpn
  second_term = tf.reduce_sum(tf.log(1.0-tf.sigmoid(tf.transpose(neg_delta))+1e-10), 1)
  expect_j = tf.negative(tf.add(first_term, second_term))
  return expect_j