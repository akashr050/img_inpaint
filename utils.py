import tensorflow as tf

slim = tf.contrib.slim
layers = tf.contrib.layers

def loss_summaries():
  loss_summary_op = layers.summarize_collection(tf.GraphKeys.LOSSES)
  return loss_summary_op


def get_train_op_for_scope(loss, optimizer, scopes, clip_gradient_norm):
  """Train operation function for the given scope used for training."""
  for var in tf.trainable_variables():
    if not (var in tf.model_variables()):
      tf.contrib.framework.add_model_variable(var)

  is_trainable = lambda x: x in tf.trainable_variables()

  var_list = []
  update_ops = []

  for scope in scopes:
    var_list.extend(
      filter(is_trainable, tf.contrib.framework.get_model_variables(scope)))
    update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))

    print('Trainable parameters %s' % tf.contrib.framework.get_model_variables(scope))
  return slim.learning.create_train_op(
    loss,
    optimizer,
    update_ops=update_ops,
    variables_to_train=var_list,
    clip_gradient_norm=clip_gradient_norm)
