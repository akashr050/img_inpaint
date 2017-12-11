import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator


def gen_inputs(params):
  ####Made changes to the following funcrion with now the mask is float32 as you are
  ####expecting a float value from py_func. Also converted the value of the mask from
  #### 1 to 1.0 for the float datatype.
  def generate_mask(batch_size):
    mask_temp = np.random.randint(low=params.mask_min_size, high=params.mask_max_size, size=batch_size)
    mask = np.zeros((batch_size, params.img_size, params.img_size, 1), dtype=np.float32)
    mask_offset_height = np.int32(np.random.uniform(low=10, high=params.img_size - mask_temp - 10, size=batch_size))
    mask_offset_width = np.int32(np.random.uniform(low=10, high=params.img_size - mask_temp - 10, size=batch_size))
    for i in range(batch_size):
      mask[i, mask_offset_height[i]:(mask_offset_height[i] + mask_temp[i]),
      mask_offset_width[i]:(mask_offset_width[i] + mask_temp[i]), :] = 1.0
    return mask

  def _input_parse_function(image_path, mask):
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_image_with_crop_or_pad(
      image_decoded, params.img_size, params.img_size)
    image = tf.to_float(image_resized)
    return image, mask

  image_paths = tf.placeholder(dtype=tf.string, shape=[None])
  masks = tf.py_func(generate_mask, [params.batch_size], tf.float32)
  print("The mask is ",dir(masks))
  data = Dataset.from_tensor_slices((image_paths, masks))
  data = data.map(_input_parse_function, num_parallel_calls=10).prefetch(params.batch_size * 10)
  data.shuffle(buffer_size=10000)
  data = data.repeat(params.epochs)
  data = data.batch(params.batch_size)
  iterator = data.make_initializable_iterator()
  image_bch, mask_bch = iterator.get_next()
  image_bch.set_shape((params.batch_size, params.img_size, params.img_size, 3))
  mask_bch.set_shape((params.batch_size, params.img_size, params.img_size, 1))

  inputs = dict()
  inputs['image_paths'], inputs['iterator'] = image_paths, iterator
  inputs['image_bch'], inputs['mask_bch'] = image_bch, mask_bch
  return inputs
