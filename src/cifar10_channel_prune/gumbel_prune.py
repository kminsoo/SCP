import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from src.common_ops import create_weight
from src.common_ops import create_bias

CHANNEL_MASK_SUM="CHANNEL_MASK_SUM"
SCALE="SCALE"
OFFSET="OFFSET"
MASK_REGULARIZER = "MASK_REGULARIZER"
MASK_PARAMETER="MASK_PARAMETER"
CDF="CDF"
LOGISTIC_C="LOGISTIC_C"

def vgg_block_gumbel_prune(x, is_training, out_filters, strides=1,
                      name="bn", decay=0.9, epsilon=1e-5,
                      data_format="NHWC",inference=False,
                      **kwargs):

  sparse_threshold = kwargs["sparse_threshold"]
  logistic_k = kwargs["logistic_k"]
  logistic_c = kwargs["logistic_c"]
  temperature = kwargs["temperature"]
  gumbel_s = kwargs["s"]

  if data_format == "NHWC":
    inp_c = x.get_shape()[3]
  elif data_format == "NCHW":
    inp_c = x.get_shape()[1]
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))

  shape = [out_filters]
  with tf.variable_scope("gumbel_conv_bn_relu", reuse=None if is_training else True) as scope:
    w = create_weight("w", [3,3,inp_c, out_filters])
    offset = tf.get_variable(
      "offset", shape,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    scale = tf.get_variable(
      "scale", shape,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))
    moving_mean = tf.get_variable(
      "moving_mean", shape, trainable=False,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    moving_variance = tf.get_variable(
      "moving_variance", shape, trainable=False,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))


    cdf_value = 0.5 * (1.0 + tf.erf((sparse_threshold - offset)/((1e-8+tf.abs(scale)) * tf.sqrt(2.0))))
    mask_parameter = 1.0 / (1.0 + tf.exp(-logistic_k*(cdf_value-logistic_c)))
    # mask_parameter: bernoulli parameter for p(mask=0)
    mask_prob = [1.0-mask_parameter, mask_parameter]
    mask_prob = tf.stack(mask_prob, axis=1)
    # mask_prob: [channel_dimension,2 ]
    argmax_mask_prob = tf.argmax(mask_prob, axis=1, output_type=tf.int32)
    inference_mask_bool = tf.equal(argmax_mask_prob, 0)
    inference_mask = tf.to_float(inference_mask_bool)

    dist = tf.contrib.distributions.RelaxedOneHotCategorical(temperature, probs=mask_prob)
    mask = dist.sample(1)[0]
    # mask: [1,channel_dimension, 2] -> [channel_dimension, 2]
    argmax_mask = tf.argmax(mask, axis=1, output_type=tf.int32)
    #argmax_mask = tf.stop_gradient(argmax_mask)
    zero_mask_bool = tf.equal(argmax_mask, 0)
    zero_mask = tf.to_float(tf.equal(argmax_mask, 0))
    zero_mask_sum = tf.reduce_sum(zero_mask)
    collection_name = scope.name + "/scale/zero_mask"
    zero_mask_nonzero = tf.cond(tf.equal(zero_mask_sum, 0.0),
                                lambda: tf.zeros_like(zero_mask_sum),
                                lambda: tf.ones_like(zero_mask_sum))
    # argmax_mask: [channel_dimension], zero_mask: [channel_dimension]
    mask = tf.gather(mask, [0], axis=1)
    # mask: [channel_dimension, 1]

    if scale not in tf.get_collection(SCALE):
      tf.add_to_collection(SCALE, scale)
      tf.add_to_collection(OFFSET, offset)
      tf.summary.histogram(scope.name + "/cdf_value", cdf_value)
      tf.summary.histogram(scope.name + "/scale_value", scale)
      tf.summary.histogram(scope.name + "/offset_value", offset)
      tf.summary.scalar(scope.name + "/alive_channel_number", tf.reduce_sum(inference_mask))
      collection_name = scope.name + "/scale/zero_mask"
      tf.add_to_collection(collection_name, zero_mask)
      mask_grad_offset = tf.gradients(mask_parameter, offset)
      mask_grad_scale = tf.gradients(mask_parameter, scale)
      tf.summary.histogram(scope.name + "/gradients_offset", mask_grad_offset)
      tf.summary.histogram(scope.name + "/gradients_scale", mask_grad_scale)
      tf.summary.histogram(scope.name + "/bernoulli", mask_parameter)
      tf.add_to_collection(MASK_PARAMETER, mask_parameter)
      tf.summary.histogram(scope.name + "/bernoulli", mask_parameter)
      tf.add_to_collection(CDF, cdf_value)
      tf.summary.histogram(scope.name + "/cdf_value", cdf_value)
      tf.add_to_collection(MASK_REGULARIZER,offset + gumbel_s * tf.abs(scale))

    if strides == 1:
      pad = "SAME"
    else:
      pad_total = 3-1
      pad_beg = pad_total // 2
      pad_end = pad_total - pad_beg
      if data_format == "NHWC":
        x = tf.pad(x, [[0,0], [pad_beg, pad_end], [pad_beg, pad_end],[0,0]])
      elif data_format == "NCHW":
        x = tf.pad(x, [[0,0], [0,0], [pad_beg, pad_end], [pad_beg, pad_end]])
      pad = "VALID"
    if data_format == "NHWC":
      stride_format = [1,strides, strides, 1]
    elif data_format == "NCHW":
      stride_format = [1,1,strides,strides]
    x = tf.nn.conv2d(x, w, stride_format, pad, data_format=data_format)


    # Need to adjust below bn update
    # zero_mask: 1 -> update, zero_mask:0 -> no update
    if is_training:
      if not inference:
        x, mean, variance = tf.nn.fused_batch_norm(
                                             x, scale, offset, epsilon=epsilon,
                                             data_format=data_format,
                                             is_training=True)
        mean  = (1.0 - decay) * (moving_mean - mean)
        mean = mean * zero_mask # bn_stat update for when sample_mask = 1
        variance = (1.0 - decay) * (moving_variance - variance)
        variacne = variance * zero_mask # bn_stat update for when sample_mask = 1
        update_mean = moving_mean.assign_sub(mean)
        update_variance = moving_variance.assign_sub(variance)
        with tf.control_dependencies([update_mean, update_variance]):
          x = tf.identity(x)
      else:
        x, _, _ = tf.nn.fused_batch_norm(x, scale, offset,
                                                   epsilon=epsilon, data_format=data_format,
                                                   is_training=True)
    else:
      x, _, _ = tf.nn.fused_batch_norm(x, scale, offset,
                                         mean=moving_mean,
                                         variance=moving_variance,
                                         epsilon=epsilon, data_format=data_format,
                                         is_training=False)
  x = tf.nn.relu(x)
  if data_format == "NHWC":
    if is_training:
      reshape_mask = tf.reshape(mask, [1,1,1,-1])
      x = x * reshape_mask
    else:
      x = x * tf.reshape(inference_mask, [1,1,1,-1])
  elif data_format == "NCHW":
    if is_training:
      reshape_mask = tf.reshape(mask, [1,-1,1,1])
      x = x * reshape_mask
    else:
      x = x * tf.reshape(inference_mask, [1,-1,1,1])

  return x, inference_mask

def dense_block_gumbel_prune(x, is_training,
                      name="bn", decay=0.9, epsilon=1e-5,
                      data_format="NHWC",inference=False,
                      **kwargs):

  sparse_threshold = kwargs["sparse_threshold"]
  logistic_k = kwargs["logistic_k"]
  logistic_c = kwargs["logistic_c"]
  temperature = kwargs["temperature"]
  gumbel_s = kwargs["s"]

  inputs = x
  if data_format == "NHWC":
    shape = [x.get_shape()[3]]
  elif data_format == "NCHW":
    shape = [x.get_shape()[1]]
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))

  with tf.variable_scope("gumbel_bn_relu") as scope:
    offset = tf.get_variable(
      "offset", shape,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    scale = tf.get_variable(
      "scale", shape,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))
    moving_mean = tf.get_variable(
      "moving_mean", shape, trainable=False,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    moving_variance = tf.get_variable(
      "moving_variance", shape, trainable=False,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))


    cdf_value = 0.5 * (1.0 + tf.erf((sparse_threshold - offset)/((1e-8+tf.abs(scale)) * tf.sqrt(2.0))))
    mask_parameter = 1.0 / (1.0 + tf.exp(-logistic_k*(cdf_value-logistic_c)))

    # mask_parameter: bernoulli parameter for p(mask=0)
    mask_prob = [1.0-mask_parameter, mask_parameter]
    mask_prob = tf.stack(mask_prob, axis=1)
    # mask_prob: [channel_dimension,2 ]
    argmax_mask_prob = tf.argmax(mask_prob, axis=1, output_type=tf.int32)
    inference_mask = tf.to_float(tf.equal(argmax_mask_prob, 0))

    dist = tf.contrib.distributions.RelaxedOneHotCategorical(temperature, probs=mask_prob)
    mask = dist.sample(1)[0]
    # mask: [1,channel_dimension, 2] -> [channel_dimension, 2]
    argmax_mask = tf.argmax(mask, axis=1, output_type=tf.int32)
    #argmax_mask = tf.stop_gradient(argmax_mask)
    zero_mask = tf.to_float(tf.equal(argmax_mask, 0))
    zero_mask_sum = tf.reduce_sum(zero_mask)
    zero_mask_nonzero = tf.cond(tf.equal(zero_mask_sum, 0.0),
                                lambda: tf.zeros_like(zero_mask_sum),
                                lambda: tf.ones_like(zero_mask_sum))
    # argmax_mask: [channel_dimension], zero_mask: [channel_dimension]
    mask = tf.gather(mask, [0], axis=1)
    # mask: [channel_dimension, 1]

    mask_grad_offset = tf.gradients(mask_parameter, offset)
    mask_grad_scale = tf.gradients(mask_parameter, scale)

    if scale not in tf.get_collection(SCALE):
      tf.add_to_collection(SCALE, scale)
      tf.summary.histogram(scope.name + "/gradients_offset", mask_grad_offset)
      tf.summary.histogram(scope.name + "/gradients_scale", mask_grad_scale)
      tf.summary.scalar(scope.name + "/alive_channel_number", tf.reduce_sum(inference_mask))
      collection_name = scope.name + "/scale/zero_mask"
      tf.add_to_collection(collection_name, zero_mask)
      tf.summary.histogram(scope.name + "/scale_value", scale)
      tf.summary.histogram(scope.name + "/offset_value", offset)
      tf.add_to_collection(MASK_PARAMETER, mask_parameter)
      tf.summary.histogram(scope.name + "/bernoulli", mask_parameter)
      tf.add_to_collection(CDF, cdf_value)
      tf.summary.histogram(scope.name + "/cdf_value", cdf_value)
      tf.add_to_collection(MASK_REGULARIZER, offset + gumbel_s * tf.abs(scale))

    # Need to adjust below bn update
    # zero_mask: 1 -> update, zero_mask:0 -> no update
    if is_training:
      if not inference:
        x, mean, variance = tf.nn.fused_batch_norm(
                                             x, scale, offset, epsilon=epsilon,
                                             data_format=data_format,
                                             is_training=True)
        mean  = (1.0 - decay) * (moving_mean - mean)
        mean = mean * zero_mask
        variance = (1.0 - decay) * (moving_variance - variance)
        variance = variance * zero_mask
        update_mean = moving_mean.assign_sub(mean)
        update_variance = moving_variance.assign_sub(variance)
        with tf.control_dependencies([update_mean, update_variance]):
          x = tf.identity(x)
      else:
        x, _, _ = tf.nn.fused_batch_norm(x, scale, offset,
                                                   epsilon=epsilon, data_format=data_format,
                                                   is_training=True)
    else:
      x, _, _ = tf.nn.fused_batch_norm(x, scale, offset,
                                         mean=moving_mean,
                                         variance=moving_variance,
                                         epsilon=epsilon, data_format=data_format,
                                         is_training=False)
    x = tf.nn.relu(x)
    if data_format == "NHWC":
      if is_training:
        x = x * tf.reshape(mask, [1,1,1,-1])
      else:
        x = x * tf.reshape(inference_mask, [1,1,1,-1])
    elif data_format == "NCHW":
      if is_training:
        x = x * tf.reshape(mask, [1,-1,1,1])
      else:
        x = x * tf.reshape(inference_mask, [1,-1,1,1])
  return x, inference_mask
