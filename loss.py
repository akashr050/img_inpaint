import tensorflow as tf
import numpy as np
from tensorflow.python.ops.losses import losses

slim = tf.contrib.slim


def generator_minimax_loss(dis_pred, gt_vector):
  """
  This function calculates the minimax loss for generator based on the Ian goodfellow paper
  :param dis_pred:
  :param gt_vector:
  :return:
  """
  loss = tf.cast(-1*tf.divide(tf.reduce_sum(tf.multiply(tf.log((1 - dis_pred)), (1-gt_vector))),
                   tf.reduce_sum(1 - gt_vector),
                   name='generator_minimax_loss'), tf.float32)
  tf.losses.add_loss(loss)
  return loss

def generator_minimax_loss_2(dis_pred, gt_vector):
  # loss = losses.sigmoid_cross_entropy((gt_vector),dis_pred, label_smoothing = 0.0, scope = 'generator_minimax_loss_2')
  loss = -tf.log(tf.multiply(dis_pred, (1-gt_vector)))
  loss = tf.cast(tf.reduce_mean(loss, name='generator_minimax_loss'), tf.float32)
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

def main():
  input = tf.random_normal([1, 228, 228, 3], dtype= tf.float64)
  gt = tf.random_normal([1, 228, 228, 3], dtype= tf.float64)
  mask = np.zeros([1, 228, 228], dtype= np.float64)
  mask[:, :124, :124] = 1
  mask = tf.convert_to_tensor(mask)
  disc_output = tf.random_normal([3,], dtype= tf.float64)
  R_loss = reconstruction_loss(input, mask, gt)
  print(R_loss)

  gt = np.zeros([3, ])
  loss = generator_minimax_loss(disc_output, gt)
  print(loss)
  loss = discriminator_minimax_loss(disc_output, gt)
  print(loss)
  print(tf.losses.get_total_loss())


if __name__=='__main__':
  main()