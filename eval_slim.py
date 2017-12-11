import tensorflow as tf

slim = tf.contrib.slim

image = inputs['image_bch']
mask = inputs['mask_bch']
gen_output, _ = glc_gen.generator(image, mask, mean_fill=FLAGS.mean_fill)

tf.summary.image("Evaluation Images", gen_out)
tf.summary.image("Image Images", image)
tf.summary.image("Mask Images", mask)

checkpoint_dir = './checkpoint/'
log_dir = './logs/'

num_evals = 10

slim.evaluation_loop(
    '',
    checkpoint_dir,
    log_dir,
    num_evals = num_evals,
    summary_op = tf.contrib.deprecated.merge_summary(summary_ops),
    eval_interval_secs  = 6000)