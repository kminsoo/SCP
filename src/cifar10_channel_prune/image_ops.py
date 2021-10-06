import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from src.common_ops import create_weight
from src.common_ops import create_bias

CHANNEL_MASK_SUM="CHANNEL_MASK_SUM"
SCALE="SCALE"


def drop_path(x, keep_prob):
  """Drops out a whole example hiddenstate with the specified probability."""

  batch_size = tf.shape(x)[0]
  noise_shape = [batch_size, 1, 1, 1]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
  binary_tensor = tf.floor(random_tensor)
  x = tf.div(x, keep_prob) * binary_tensor

  return x


def conv(x, filter_size, out_filters, stride, name="conv", padding="SAME",
         data_format="NHWC", seed=None):
  """
  Args:
    stride: [h_stride, w_stride].
  """

  if data_format == "NHWC":
    actual_data_format = "channels_last"
  elif data_format == "NCHW":
    actual_data_format = "channels_first"
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))
  x = tf.layers.conv2d(
      x, out_filters, [filter_size, filter_size], stride, padding,
      data_format=actual_data_format,
      kernel_initializer=tf.contrib.keras.initializers.he_normal(seed=seed))

  return x


def fully_connected(x, out_size, name="fc", seed=None):
  in_size = x.get_shape()[-1].value
  with tf.variable_scope(name):
    w = create_weight("w", [in_size, out_size], seed=seed)
  x = tf.matmul(x, w)
  return x


def max_pool(x, k_size, stride, padding="SAME", data_format="NHWC",
             keep_size=False):
  """
  Args:
    k_size: two numbers [h_k_size, w_k_size].
    stride: two numbers [h_stride, w_stride].
  """

  if data_format == "NHWC":
    actual_data_format = "channels_last"
  elif data_format == "NCHW":
    actual_data_format = "channels_first"
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))
  out = tf.layers.max_pooling2d(x, k_size, stride, padding,
                                data_format=actual_data_format)

  if keep_size:
    if data_format == "NHWC":
      h_pad = (x.get_shape()[1].value - out.get_shape()[1].value) // 2
      w_pad = (x.get_shape()[2].value - out.get_shape()[2].value) // 2
      out = tf.pad(out, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]])
    elif data_format == "NCHW":
      h_pad = (x.get_shape()[2].value - out.get_shape()[2].value) // 2
      w_pad = (x.get_shape()[3].value - out.get_shape()[3].value) // 2
      out = tf.pad(out, [[0, 0], [0, 0], [h_pad, h_pad], [w_pad, w_pad]])
    else:
      raise NotImplementedError("Unknown data_format {}".format(data_format))
  return out


def global_avg_pool(x, data_format="NHWC"):
  if data_format == "NHWC":
    x = tf.reduce_mean(x, [1, 2])
  elif data_format == "NCHW":
    x = tf.reduce_mean(x, [2, 3])
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))
  return x


def batch_norm(x, is_training, name="bn", decay=0.9, epsilon=1e-5,
               data_format="NHWC",inference=False, scale_init=1.0):
  if data_format == "NHWC":
    shape = [x.get_shape()[3]]
  elif data_format == "NCHW":
    shape = [x.get_shape()[1]]
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))

  with tf.variable_scope(name, reuse=None if is_training else True):
    offset = tf.get_variable(
      "offset", shape,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    scale = tf.get_variable(
      "scale", shape,
      initializer=tf.constant_initializer(scale_init, dtype=tf.float32))
    moving_mean = tf.get_variable(
      "moving_mean", shape, trainable=False,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    moving_variance = tf.get_variable(
      "moving_variance", shape, trainable=False,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))

    if is_training:
      if not inference:
        x, mean, variance = tf.nn.fused_batch_norm(
        x, scale, offset, epsilon=epsilon, data_format=data_format,
        is_training=True)
        update_mean = moving_averages.assign_moving_average(
        moving_mean, mean, decay)
        update_variance = moving_averages.assign_moving_average(
        moving_variance, variance, decay)
        with tf.control_dependencies([update_mean, update_variance]):
          x = tf.identity(x)
      else:
        x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, epsilon=epsilon, data_format=data_format, is_training=True)
    else:
      x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                       variance=moving_variance,
                                       epsilon=epsilon, data_format=data_format,
                                       is_training=False)
  return x

def batch_norm_init(x, is_training, reader,mask=None, sess=None, name=None, decay=0.9, epsilon=1e-5,
               data_format="NHWC",inference=False):
  if data_format == "NHWC":
    shape = [x.get_shape()[3]]
  elif data_format == "NCHW":
    shape = [x.get_shape()[1]]
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))

  if name is None:
    offset_init = reader.get_tensor(tf.get_variable_scope().name + '/offset')
    scale_init = reader.get_tensor(tf.get_variable_scope().name+ '/scale')
    moving_mean_init = reader.get_tensor(tf.get_variable_scope().name + '/moving_mean')
    moving_variance_init = reader.get_tensor(tf.get_variable_scope().name + '/moving_variance')
  else:
    offset_init = reader.get_tensor(tf.get_variable_scope().name + '/%s/offset'%name)
    scale_init = reader.get_tensor(tf.get_variable_scope().name+ '/%s/scale'%name)
    moving_mean_init = reader.get_tensor(tf.get_variable_scope().name + '/%s/moving_mean'%name)
    moving_variance_init = reader.get_tensor(tf.get_variable_scope().name + '/%s/moving_variance'%name)
  if mask is not None:
      offset_init = np.take(offset_init, mask)
      scale_init = np.take(scale_init, mask)
      moving_mean_init = np.take(moving_mean_init, mask)
      moving_variance_init = np.take(moving_variance_init, mask)

  with tf.variable_scope(name):
    offset = tf.get_variable(
        "offset", shape,
        initializer=tf.constant_initializer(offset_init))
    scale = tf.get_variable(
        "scale", shape,
        initializer=tf.constant_initializer(scale_init))
    moving_mean = tf.get_variable(
        "moving_mean", shape, trainable=False,
        initializer=tf.constant_initializer(moving_mean_init))
    moving_variance = tf.get_variable(
        "moving_variance", shape, trainable=False,
        initializer=tf.constant_initializer(moving_variance_init))

    if is_training:
        if not inference:
          x, mean, variance = tf.nn.fused_batch_norm(
          x, scale, offset, epsilon=epsilon, data_format=data_format,
          is_training=True)
          update_mean = moving_averages.assign_moving_average(
          moving_mean, mean, decay)
          update_variance = moving_averages.assign_moving_average(
          moving_variance, variance, decay)
          with tf.control_dependencies([update_mean, update_variance]):
              x = tf.identity(x)
        else:
          x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, epsilon=epsilon, data_format=data_format, is_training=True)
    else:
        x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                        variance=moving_variance,
                                        epsilon=epsilon, data_format=data_format,
                                        is_training=False)
    return x

def relu(x, leaky=0.0):
  return tf.where(tf.greater(x, 0), x, x * leaky)
