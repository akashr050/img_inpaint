import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def GAN_loss(out, case = 0):
    m = out.get_shape().as_list()[0]
    D_loss = -tf.reduce_mean(tf.log(tf.slice(out,[0,1],[m,1])) + tf.log(1. - tf.slice(out,[0,0],[m,1])))
    G_loss = -tf.reduce_mean(tf.log(1. - tf.slice(out,[0,0],[m,1])))

    # Alternative losses in case logits need to be calculated:
    # -------------------
    # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_real, tf.ones_like(D_logit_real)))
    # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake, tf.zeros_like(D_logit_fake)))
    # D_loss = D_loss_real + D_loss_fake
    # G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake, tf.ones_like(D_logit_fake)))
    if case == 0:
        return D_loss
    else:
        return G_loss

def Reconstruction_loss(input, mask, ground_truth):
    mask = tf.expand_dims(mask,axis=3)
    return tf.losses.mean_squared_error(tf.multiply(input,mask), tf.multiply(ground_truth,mask))


def main():
  input = tf.random_normal([1, 228, 228, 3], dtype= tf.float64)
  gt = tf.random_normal([1, 228, 228, 3], dtype= tf.float64)
  mask = np.zeros([1, 228, 228], dtype= np.float64)
  mask[:, :124, :124] = 1
  mask = tf.convert_to_tensor(mask)
  disc_output = tf.random_normal([3,2], dtype= tf.float64)
  R_loss = Reconstruction_loss(input, mask, gt)
  print(R_loss)

  loss = GAN_loss(disc_output)
  print(loss)



if __name__=='__main__':
  main()