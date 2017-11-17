import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def glc_gen(input, mask, mean_fill= 102, is_training=False, scope='glc_gen'):
  tf.assert_equal(input.get_shape().as_list()[:3], mask.get_shape().as_list()[:3])
  with tf.variable_scope(scope):
    end_points_collection = scope + '_endpoints'
    with slim.arg_scope([slim.conv2d,
                         slim.fully_connected,
                         slim.max_pool2d,
                         slim.conv2d_transpose],
                        outputs_collections=end_points_collection,
                        padding='SAME'):
      with slim.arg_scope([slim.conv2d_transpose], padding='SAME', biases_initializer=None):
        # net = tf.pad(input, [[0, 0], [100, 100], [100, 100], [0, 0]], mode="CONSTANT", constant_values=0.0)
        net = tf.add(input * (1-mask), mean_fill * mask, name='masked_img')
        tf.add_to_collection(end_points_collection, net)
        net = slim.conv2d(net, 64, [5, 5], scope='conv1')
        net = slim.conv2d(net, 128, [3, 3], stride=2, scope='conv2_1')
        net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv2_2')
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3_1')
        net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3_2')
        net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3_3')

        net = slim.conv2d(net, 256, [3, 3], rate=2, scope='conv3_4')
        net = slim.conv2d(net, 256, [3, 3], rate=4, scope='conv3_5')
        net = slim.conv2d(net, 256, [3, 3], rate=8, scope='conv3_6')
        net = slim.conv2d(net, 256, [3, 3], rate=16, scope='conv3_7')
        net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3_8')

        net = slim.conv2d_transpose(net, 128, [4, 4], 2, scope='conv4_1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv4_2')
        net = slim.conv2d_transpose(net, 64, [4, 4], 2, scope='conv5_1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv5_2')
        net = slim.conv2d(net, 3, [3, 3], activation_fn=tf.nn.sigmoid, scope='conv5_3')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points


def main():
  input = tf.random_normal([1, 228, 228, 3])
  mask = tf.random_normal([1, 228, 228, 1])
  a, b = glc_gen(input, mask)

if __name__=='__main__':
  main()
