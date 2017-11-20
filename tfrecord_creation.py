import tensorflow as tf
import os
from PIL import Image
import numpy as np

image_path = "./workspace/CelebA/Img/img_align_celeba/img_align_celeba"

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = './workspace/celeba.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for file in os.listdir(image_path):
    if os.path.isfile(os.path.join(image_path,file)):
        img = np.array(Image.open(os.path.join(image_path,file)))
        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = img.shape[0]
        width = img.shape[1]

        # Put in the original images into array
        # Just for future check for correctness

        img_raw = img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw)
        }))
        writer.write(example.SerializeToString())

writer.close()
