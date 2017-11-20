import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def get_image_patch(input, mask):
  mask_index = tf.where(tf.equal(mask, 1))
  row_ind, col_ind = mask_index[:, 0], mask_index[:, 1]
  row_min, col_min = tf.reduce_min(row_ind), tf.reduce_min(col_ind)
  row_max, col_max = tf.reduce_max(row_ind), tf.reduce_max(col_ind)
  row_center = (row_min + row_max)/2
  col_center = (col_min + col_max)/2
  row_center = tf.cond(row_center <= 64, lambda: tf.cast(64.0, dtype=tf.float64), lambda: row_center)
  row_center = tf.cond(row_center >= 256 - 64, lambda: tf.cast(256 - 64, dtype=tf.float64), lambda: row_center)

  col_center = tf.cond(col_center <= 64, lambda: tf.cast(64.0, dtype=tf.float64), lambda: col_center)
  col_center = tf.cond(col_center >= 256 - 64, lambda: tf.cast(256 - 64, dtype=tf.float64), lambda: col_center)

  output = tf.slice(input, tf.cast([row_center-64, col_center-64,0], dtype=tf.int32), [128, 128, 3])
  return output


def glc_lc_dis(input, mask, output_collection, scope='glc_lc_dis'):
  tf.assert_equal(input.get_shape().as_list()[:3], mask.get_shape().as_list())
  batch_size = input.get_shape().as_list()[0]
  net_input = tf.Variable(initial_value=tf.zeros([batch_size, 128, 128, 3]))
  for i in range(input.get_shape().as_list()[0]):
    tf.scatter_update(net_input, [i], tf.expand_dims(get_image_patch(input[i], mask[i]), axis=0))

  with tf.variable_scope(scope):
    end_points_collection = output_collection
    with slim.arg_scope([slim.conv2d],
                        outputs_collections=end_points_collection,
                        padding='SAME',
                        stride=[2, 2],
                        kernel_size=[5, 5]):
      with slim.arg_scope([slim.fully_connected], outputs_collections=end_points_collection):
        tf.add_to_collection(end_points_collection, net_input)
        net = slim.conv2d(net_input, 64, scope='conv1')
        net = slim.stack(net, slim.conv2d, [128, 256, 512, 512], scope='conv2')
        net = slim.fully_connected(net, 1024)
      return net, output_collection


def glc_gb_dis(input, output_collection, scope='glc_gb_dis'):
  with tf.variable_scope(scope):
    end_points_collection = output_collection
    with slim.arg_scope([slim.conv2d],
                        outputs_collections=end_points_collection,
                        padding='SAME',
                        stride=[2, 2],
                        kernel_size=[5, 5]):
      with slim.arg_scope([slim.fully_connected], outputs_collections=end_points_collection):
        tf.add_to_collection(end_points_collection, input)
        net = slim.conv2d(input, 64, scope='conv1')
        net = slim.stack(net, slim.conv2d, [128, 256, 512, 512, 512], scope='conv2')
        net = slim.fully_connected(net, 1024)
      return net, output_collection


def glc_dis(input, mask, scope='glc_dis'):
  with tf.variable_scope(scope):
    end_points_collection = scope + '_endpoints'
    fc_lc, end_points_collection = glc_lc_dis(input, mask, end_points_collection)
    fc_gb, end_points_collection = glc_gb_dis(input, end_points_collection)
    fc_concat = tf.concat([fc_lc, fc_gb], axis=1, name='fc_output')
    output = slim.fully_connected(fc_concat, 2, activation_fn=tf.nn.sigmoid, scope='output')
    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return output, end_points


def main():
  input = tf.random_normal([1, 228, 228, 3])
  mask = np.zeros([1, 228, 228])
  mask[:, :124, :124] = 1
  mask = tf.convert_to_tensor(mask)
  a, b = glc_dis(input, mask)

if __name__=='__main__':
  main()
