import tensorflow as tf
import numpy as np
import csv
from models.generators import glc_gen
from models.discriminators import glc_dis
from input_generator import gen_inputs
import utils
import loss

flags = tf.app.flags
slim = tf.contrib.slim
layers = tf.contrib.layers

# TODO: ADD checkpoint saver
T_TRAIN, T_C, T_D = 100, 50, 40
flags.DEFINE_string('eval_file', None, 'Path to evaluation csv')
flags.DEFINE_string('inp_dir', None, 'Path to input directory')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_integer('epochs', 1000, '')
flags.DEFINE_integer('img_size', 160, 'Image height')
flags.DEFINE_integer('img_width', 160, 'image_width')
flags.DEFINE_integer('mask_min_size', 48, '')
flags.DEFINE_integer('mask_max_size', 96, '')
flags.DEFINE_float('mean_fill', 102.0, '')
flags.DEFINE_integer('num_channels', 3, '')
flags.DEFINE_integer('clip_gradient_norm', 4, '')
flags.DEFINE_string('tb_dir', 'tb_results', '')
flags.DEFINE_string('ckpt_dir', 'checkpoints', '')
FLAGS = flags.FLAGS


def get_data(eval_file):
  csvreader = csv.reader(open(eval_file, 'r+'))
  csvreader.next()
  img_paths = []
  tag = []
  for index, line in enumerate(csvreader):
    img_paths.extend(line[0])
    tag.extend(line[1])
  img_paths = np.array(img_paths)
  tag = np.array(tag)
  img_train, img_val = img_paths[tag==0], img_paths[tag==1]
  return img_train, img_val



def train_glc():
  # train_img_paths = get_data(FLAGS.train_file)
  slim.get_or_create_global_step()
  inputs = gen_inputs(FLAGS)
  image = inputs['image_bch']
  mask = inputs['mask_bch']
  gen_output, _ = glc_gen.generator(image, mask, mean_fill=FLAGS.mean_fill)

  ##################
  ## Optimisation ##
  ##################

  # Discriminator loss
  dis_input = tf.concat([gen_output, image], axis=0)
  dis_mask = tf.concat([mask]*2, axis=0)
  dis_labels = tf.concat([tf.zeros(shape=(FLAGS.batch_size,)),
                          tf.ones(shape=FLAGS.batch_size,)], axis=0)
  pred_dis_labels, _ = glc_dis.discriminator(dis_input, dis_mask, FLAGS)
  discriminator_loss = loss.discriminator_minimax_loss(pred_dis_labels, dis_labels)

  # Generator loss
  gen_dis_input = gen_output
  gen_dis_masks = mask
  gen_dis_labels = tf.ones(shape=FLAGS.batch_size,)
  pred_gen_dis_labels, _ = glc_dis.discriminator(gen_dis_input, gen_dis_masks, FLAGS,
                                                 reuse=True)
  generator_dis_loss = loss.generator_minimax_loss(gen_dis_labels, pred_gen_dis_labels)
  generator_rec_loss = loss.reconstruction_loss(gen_output, mask, image)

  dis_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
  gen_rec_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
  gen_dis_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

  dis_train_op = utils.get_train_op_for_scope(discriminator_loss,
                                              dis_optimizer,
                                              ['glc_dis'],
                                              FLAGS.clip_gradient_norm)

  generator_rec_train_op = utils.get_train_op_for_scope(generator_rec_loss,
                                                        gen_rec_optimizer,
                                                        ['glc_gen'],
                                                        FLAGS.clip_gradient_norm)

  generator_dis_train_op = utils.get_train_op_for_scope(generator_dis_loss,
                                                        gen_dis_optimizer,
                                                        ['glc_gen'],
                                                        FLAGS.clip_gradient_norm)
  loss_summary_op = layers.summarize_collection(tf.GraphKeys.LOSSES)

  with tf.Session() as sess:
    tb_writer = tf.summary.FileWriter(FLAGS.tb_dir + '/train', sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(inputs['iterator'].initialize, feed_dict={
      inputs['image_paths']: train_img_paths})
    while True:
      try:
        for counter in range(T_TRAIN):
          step = sess.run(slim.get_global_step())
          if counter < T_C:
            _, loss_summaries = sess.run([generator_rec_train_op, loss_summary_op])
          elif counter < T_C+T_D:
            _, loss_summaries = sess.run([dis_train_op, loss_summary_op])
          else:
            _, loss_summaries = sess.run([generator_dis_train_op, loss_summary_op])
          tb_writer.add_summary(loss_summaries, step)
          print('Global_step: {}'.format(step))
          saver.save(sess, FLAGS.ckpt_dir)
      except tf.errors.OutOfRangeError:
        break
  return None

def main():
  train_glc()

if __name__=='__main__':
  main()

