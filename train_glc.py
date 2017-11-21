import tensorflow as tf
import numpy as np
import csv
from tensorflow.contrib.data import Dataset, Iterator

import utils
from models import discriminators, generators

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('eval_file', None, 'Path to evaluation csv')
flags.DEFINE_string('inp_dir', None, 'Path to input directory')
flags.DEFINE_integer('img_height', 227, 'Image height')
flags.DEFINE_integer('img_width', 227, 'image_width')
flags.DEFINE_integer('mask_min_size', 90, '')
flags.DEFINE_integer('mask_max_size', 120, '')
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


def generate_mask(mask_size, params):
  mask_temp = np.random.uniform(low=params.mask_min_size, high=params.mask_max_size, size=mask_size)
  mask = np.zeros((mask_size, params.img_height, params.img_width), dtype=np.int32)
  mask_offset_height = np.random.uniform(low=10, high=params.img_height - mask_temp -10)
  mask_offset_width = np.random.uniform(low=10, high=params.)
  mask[:, mask_temp]


def train_glc():
  train_img_paths, val_img_paths = get_data(FLAGS.eval_file)

  images = tf.placeholder(dtype=tf.string, shape=[None])
  mask = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.img_height, FLAGS.img_width, 1])
  mask_dim = tf.random_uniform(FLAGS.batch_size, minval=FLAGS.mask_dim_size, maxval=FLAGS.mask_dim_size)
  tf.random_crop(im)

  return

def main():
  return

if __name__=='__main__':
  main()

