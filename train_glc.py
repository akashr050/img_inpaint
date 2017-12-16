import tensorflow as tf
import numpy as np
from models.generators import glc_gen
from models.discriminators import glc_dis
from input_generator import gen_inputs
import utils
import loss
import os

flags = tf.app.flags
slim = tf.contrib.slim
layers = tf.contrib.layers

# TODO: ADD checkpoint saver
T_TRAIN, T_C, T_D = 30, 10, 10
flags.DEFINE_string('train_file', 'train.txt', 'Path to train images')
flags.DEFINE_string('inp_dir', 'workspace', 'Path to input directory')
flags.DEFINE_integer('batch_size', 5, '')
flags.DEFINE_integer('epochs', 50000, '')
flags.DEFINE_integer('img_size', 160, 'Image height')
flags.DEFINE_integer('img_width', 160, 'image_width')
flags.DEFINE_integer('mask_min_size', 48, '')
flags.DEFINE_integer('mask_max_size', 96, '')
flags.DEFINE_float('mean_fill', 102.0/255.0, '')
flags.DEFINE_integer('num_channels', 3, '')
flags.DEFINE_integer('clip_gradient_norm', 5, '')
flags.DEFINE_string('tb_dir', 'tb_results', '')
flags.DEFINE_string('ckpt_dir', 'checkpoints/', '')
FLAGS = flags.FLAGS


def get_data(file_path):
  txt_file = open(file_path, 'r+').readlines()
  img_paths=[]
  for line in txt_file:
    img_path = os.path.join(FLAGS.inp_dir, 'raw_images', line[:-1])
    img_paths.append(img_path)
  img_paths = np.array(img_paths)
  return img_paths


def train_glc():
  train_img_paths = get_data(os.path.join(FLAGS.inp_dir, FLAGS.train_file))
  slim.get_or_create_global_step()
  inputs = gen_inputs(FLAGS)

  image = inputs['image_bch']
  mask = inputs['mask_bch']
  #
  # image = np.ones((1,160,160,3))
  # mask = np.zeros((1,160,160,1))
  # mask[0,80:100,80:100,0] = 1
  # image = tf.convert_to_tensor(image, tf.float32)
  # mask = tf.convert_to_tensor(mask, tf.float32)

  # print("The image is ", image)
  # print("The mask is ",mask)


  gen_output, _ = glc_gen.generator(image, mask, mean_fill=FLAGS.mean_fill)
  # print(gen_output)
  # gen_output = image

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
  discriminator_loss_library = loss.tf_generator_minmax_disc_loss(tf.slice(pred_dis_labels, [0], [FLAGS.batch_size]),tf.slice(pred_dis_labels, [(FLAGS.batch_size)], [FLAGS.batch_size]))
  # Generator loss
  gen_dis_input = gen_output
  gen_dis_masks = mask
  gen_dis_labels = tf.zeros(shape=(FLAGS.batch_size,))
  pred_gen_dis_labels, _ = glc_dis.discriminator(gen_dis_input, gen_dis_masks, FLAGS,
                                                 reuse=True)
  generator_dis_loss = loss.generator_minimax_loss(pred_gen_dis_labels, gen_dis_labels)
  generator_dis_loss_library = loss.tf_generator_minmax_gen_loss(pred_gen_dis_labels)
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
  layers.summarize_collection(tf.GraphKeys.LOSSES)
  loss_summary_op = tf.summary.merge_all()
  with tf.Session() as sess:
    tb_writer = tf.summary.FileWriter(FLAGS.tb_dir + '/train', sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(inputs['iterator'].initializer, feed_dict={
      inputs['image_paths']: train_img_paths})
    while True:
      try:
        for counter in range(T_TRAIN):
          step = sess.run(slim.get_global_step())
          if counter < T_C:
            _, loss_summaries, aa = sess.run([generator_rec_train_op, loss_summary_op, generator_rec_loss])
          elif counter < T_C+T_D:
            _, loss_summaries, aa = sess.run([dis_train_op, loss_summary_op, discriminator_loss_library])
          else:
            _, loss_summaries, aa = sess.run([generator_dis_train_op, loss_summary_op, generator_dis_loss_library])
          tb_writer.add_summary(loss_summaries, step)
          print('Global_step: {}, Loss: {}'.format(step, aa))
          # print("The loss is ", aa)
          # saver.save(sess, FLAGS.ckpt_dir)
      except tf.errors.OutOfRangeError:
        break
  return None

def main():
  train_glc()

if __name__=='__main__':
  main()

