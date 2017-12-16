import tensorflow as tf
import numpy as np
from PIL import Image
import copy
from tensorflow.contrib.gan.python.losses.python import losses_impl as gan_loss
from models.generators import glc_gen
from models.discriminators import glc_dis

slim = tf.contrib.slim


def generator_minimax_loss(dis_pred, gt_vector):
  """
  This function calculates the minimax loss for generator based on the Ian goodfellow paper
  :param dis_pred:
  :param gt_vector:
  :return:
  """
  loss = tf.cast(-1* tf.divide(tf.reduce_sum(tf.multiply(tf.log((1 - dis_pred)), (1-gt_vector))),
                   tf.reduce_sum(1 - gt_vector),
                   name='generator_minimax_loss'), tf.float32)
  tf.losses.add_loss(loss)
  return loss


def discriminator_minimax_loss(dis_pred, gt_vector):
  loss_1 = tf.multiply(tf.log((1 - dis_pred)), (1-gt_vector))
  loss_2 = tf.multiply(tf.log(dis_pred), gt_vector)
  loss = - (loss_1 + loss_2)
  loss = tf.cast(tf.reduce_mean(loss, name='discriminator_minimax_loss'), tf.float32)
  tf.losses.add_loss(loss)
  return loss

def reconstruction_loss(gen_pred, mask, gt_image):
  if len(mask.get_shape()) == 3:
    mask = tf.expand_dims(mask,axis=3)
  loss = tf.losses.mean_squared_error(tf.multiply(gen_pred, mask),
                                      gt_image, weights=mask)
  return loss

def tf_generator_minmax_disc_loss(D_real, D_fake):
  return gan_loss.minimax_discriminator_loss(D_real,D_fake)

def tf_generator_minmax_gen_loss(D_fake):
  return gan_loss.minimax_generator_loss(D_fake)

def main():
  # input = tf.random_normal([1, 228, 228, 3], dtype= tf.float64)
  # gt = tf.random_normal([1, 228, 228, 3], dtype= tf.float64)
  ###########  input is real , gt is generated or fake
  # input = Image.open("./workspace/raw_images/000001.jpg")
  # input = np.asarray(input, dtype="float64")
  # input = input/255
  # gt = copy.deepcopy(input)
  # gt[:64, :24, :] = 1
  #
  # input = tf.convert_to_tensor(input, tf.float64)
  # input = tf.expand_dims(input,0)
  # gt = tf.convert_to_tensor(gt, tf.float64)
  # gt = tf.expand_dims(gt,0)
  #
  # mask = np.zeros([1, 218, 178], dtype= "float64")
  # mask[:, :124, :124] = 1
  # mask = tf.convert_to_tensor(mask)
  dis_pred_real = tf.constant([[0.9],[0.9],[0.9],[0.9],[0.9]], dtype = tf.float32)
  dis_pred_fake = tf.constant([[0.1],[0.1],[0.1],[0.1],[0.1]], dtype=tf.float32)
  dis_pred = tf.concat([dis_pred_real, dis_pred_fake], axis=0)
  gt_real = tf.ones([5,1], dtype = tf.float32)
  gt_fake = tf.zeros([5,1], dtype=tf.float32)
  gt_vector = tf.concat([gt_real, gt_fake], axis=0)

  tf_gan_loss = tf_generator_minmax_disc_loss(dis_pred_real, dis_pred_fake)
  other_loss = discriminator_minimax_loss(dis_pred, gt_vector)
  # disc_output = tf.random_normal([3,], dtype= tf.float64)
  # R_loss = reconstruction_loss(input, mask, gt)
  # print(R_loss)
  #
  # gt = np.zeros([3, ])
  # loss = generator_minimax_loss(disc_output, gt)
  # print(loss)
  # loss = discriminator_minimax_loss(disc_output, gt)
  # print(loss)
  # print(tf.losses.get_total_loss())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    m,l = sess.run([other_loss,tf_gan_loss])
    print("The tensorflow library gives ",l)
    print("Our function gives ",m)

if __name__=='__main__':
  main()