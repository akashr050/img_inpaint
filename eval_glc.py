import tensorflow as tf
import numpy as np
from models.generators import glc_gen
from input_generator import gen_inputs
import os
import time
# import matplotlib.pyplot as plt
# import io

flags = tf.app.flags
slim = tf.contrib.slim
layers = tf.contrib.layers

# TODO: ADD checkpoint saver
T_TRAIN, T_C, T_D = 100, 50, 40
flags.DEFINE_string('train_file', 'eval.txt', 'Path to train images')
flags.DEFINE_string('inp_dir', 'workspace', 'Path to input directory')
flags.DEFINE_integer('batch_size', 5, '')
flags.DEFINE_integer('epochs', 1000, '')
flags.DEFINE_integer('img_size', 160, 'Image height')
flags.DEFINE_integer('img_width', 160, 'image_width')
flags.DEFINE_integer('mask_min_size', 48, '')
flags.DEFINE_integer('mask_max_size', 96, '')
flags.DEFINE_float('mean_fill', 102.0/255.0, '')
flags.DEFINE_integer('num_channels', 3, '')
flags.DEFINE_integer('clip_gradient_norm', 4, '')
flags.DEFINE_string('tb_dir', 'tb_results', '')
flags.DEFINE_string('ckpt_dir', './checkpoints/', '')
flags.DEFINE_integer('eval_interval_secs', 600, '')
FLAGS = flags.FLAGS



def get_data(file_path):
  txt_file = open(file_path, 'r+').readlines()
  img_paths=[]
  for line in txt_file:
    img_path = os.path.join(FLAGS.inp_dir, 'raw_images', line[:-1])
    img_paths.append(img_path)
  img_paths = np.array(img_paths)
  return img_paths


# def viz_op(image, mask, gen_image):
#   plt.figure()
#   f, axarr = plt.subplots(3, 3)
#   for i in range(3):
#     axarr[i, 0] = plt.imshow(image[i])
#     axarr[i, 1] = plt.imshow(mask[i])
#     axarr[i, 2] = plt.imshow(gen_image[i])
#   buf = io.BytesIO()
#   plt.savefig(buf, format='png')
#   buf.seek(0)
#   return buf


def eval_glc():
  eval_img_paths = get_data(os.path.join(FLAGS.inp_dir, FLAGS.train_file))[:5]
  slim.get_or_create_global_step()
  inputs = gen_inputs(FLAGS)
  image = inputs['image_bch']
  mask = inputs['mask_bch']
  gen_output, generated_images = glc_gen.generator(image, mask, mean_fill=FLAGS.mean_fill)
  input_to_net = generated_images['glc_gen/masked_img']	* 255

  ############
  ## Viz op ##
  ############

  image = tf.cast(image*255, dtype=tf.uint8)
  mask = tf.cast(mask*255, dtype=tf.uint8)
  gen_image = tf.cast(gen_output*255, dtype=tf.uint8)
  # vizual_op = tf.py_func(viz_op, [image, mask, gen_image], tf.float32)
  # image_summary = tf.image.decode_png(vizual_op.getvalue(), channels=4)
  # image_summary = tf.expand_dims(image_summary, 0)
  # summary_op = tf.summary.image('image_summary', image_summary)
  tf.summary.image(name='input_images', tensor=image, max_outputs=5)
  tf.summary.image(name='mask', tensor=mask, max_outputs=5)
  tf.summary.image(name='gen_images', tensor=gen_image, max_outputs=5)
  tf.summary.image(name='masked_images', tensor=input_to_net, max_outputs=5)
  summary_op = tf.summary.merge_all()
  init_op = inputs['iterator'].initializer

  # slim.evaluation.evaluation_loop(
  #   '',
  #   FLAGS.ckpt_dir,
  #   os.path.join(FLAGS.tb_dir, 'eval'),
  #   num_evals=1,
  #   initial_op=init_op,
  #   initial_op_feed_dict={inputs['image_paths']: eval_img_paths},
  #   summary_op=summary_op,
  #   eval_interval_secs=FLAGS.eval_interval_secs)

  with tf.Session() as sess:
    tb_writer = tf.summary.FileWriter(FLAGS.tb_dir + '/eval')
    saver = tf.train.Saver()
    # sess.run(tf.global_variables_initializer())
    sess.run(inputs['iterator'].initializer, feed_dict={
      inputs['image_paths']: eval_img_paths})
    # while True:
    try:
      while True:
        saver.restore(sess, FLAGS.ckpt_dir)
        step = sess.run(slim.get_global_step())
        img_summary = sess.run(summary_op)
        tb_writer.add_summary(img_summary, step)
        time.sleep(600)
    except tf.errors.OutOfRangeError:
      pass
  return None

def main():
  with tf.device('/cpu:0'):
    eval_glc()

if __name__=='__main__':
  main()

