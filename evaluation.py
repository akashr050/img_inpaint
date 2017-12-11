import tensorflow as tf

#Should we use the session inside the evaluation loop or does it inherit this
#from above.

def eval(sess,inputs,tb_writer):
    image = inputs['image_bch']
    mask = inputs['mask_bch']
    gen_output, _ = glc_gen.generator(image, mask, mean_fill=FLAGS.mean_fill)
    gen_out = sess.run([gen_output])
    generator_summary = tf.summary.image("Evaluation Images", gen_out)
    image_summary = tf.summary.image("Image Images", image)
    mask_summary = tf.summary.image("Mask Images", mask)

    tb_writer.add_summary([generator_summary, image_summary, mask_summary], step)
