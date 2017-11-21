import tensorflow as tf

slim = tf.contrib.slim
layers = tf.contrib.layers

def add_to_summaries():
  loss_summary_op = layers.summarize_collection(tf.GraphKeys.LOSSES)
