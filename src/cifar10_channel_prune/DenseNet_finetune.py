import sys
import os
import time

import numpy as np
import pdb
import tensorflow as tf

from collections import OrderedDict
from src.cifar10_channel_prune.models import Model
from src.cifar10_channel_prune.image_ops import conv
from src.cifar10_channel_prune.image_ops import fully_connected
from src.cifar10_channel_prune.image_ops import batch_norm_init
from src.cifar10_channel_prune.gumbel_prune import dense_block_gumbel_prune
from src.cifar10_channel_prune.image_ops import relu
from src.cifar10_channel_prune.image_ops import max_pool
from src.cifar10_channel_prune.image_ops import global_avg_pool

from src.utils import count_model_params
from src.cifar10_channel_prune.utils import get_train_ops
from src.common_ops import create_weight
from src.common_ops import create_bias
CDF="CDF"
CDF2="CDF2"
CDF3="CDF3"
SCALE="SCALE"
OFFSET="OFFSET"
MASK_PARAMETER="MASK_PARAMETER"
MASK_PARAMETER2="MASK_PARAMETER2"
MASK_PARAMETER3="MASK_PARAMETER3"
sparse_threshold = 0.05
logistic_k = 20
logistic_c = 0.95
class densenet(Model):
  def __init__(self,
               images,
               labels,
               reader,
               sess,
               depth=40,
               growth_rate=12,
               compression_rate=1,
               ensemble_models=1,
               cutout_size=None,
               keep_prob=1.0,
               batch_size=128,
               clip_mode=None,
               grad_bound=None,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=10000,
               lr_dec_rate=0.1,
               lr_cosine=False,
               lr_max=None,
               lr_min=None,
               lr_T_0=None,
               lr_T_mul=None,
               optim_algo="momentum",
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               data_format="NCHW",
               name="DenseNet",
               child_lr_otf=True,
               *args,
               **kwargs
              ):
    """
    """

    super(self.__class__, self).__init__(
      images,
      labels,
      cutout_size=cutout_size,
      batch_size=batch_size,
      clip_mode=clip_mode,
      grad_bound=grad_bound,
      l2_reg=l2_reg,
      lr_init=lr_init,
      lr_dec_start=lr_dec_start,
      lr_dec_every=lr_dec_every,
      lr_dec_rate=lr_dec_rate,
      keep_prob=keep_prob,
      optim_algo=optim_algo,
      sync_replicas=sync_replicas,
      num_aggregate=num_aggregate,
      num_replicas=num_replicas,
      data_format=data_format,
      name=name)
    self.depth = depth
    self.growth_rate = growth_rate
    self.compression_rate = compression_rate
    self.lr_cosine = lr_cosine
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.lr_T_0 = lr_T_0
    self.lr_T_mul = lr_T_mul

    self.child_lr_otf = child_lr_otf
    self.ensemble_models = ensemble_models
    self.reader = reader
    self.sess = sess


    #self.assign_op_dict = OrderedDict()
    self._build_train()
    self._build_test()

  def _get_C(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return x.get_shape()[3].value
    elif self.data_format == "NCHW":
      return x.get_shape()[1].value
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

  def _get_HW(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    return x.get_shape()[2].value

  def _get_strides(self, stride):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return [1, stride, stride, 1]
    elif self.data_format == "NCHW":
      return [1, 1, stride, stride]
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))
  def add_layer(self,inputs, is_training):
    if self.data_format == "NHWC":
      concat_dim=3
    elif self.data_format == "NCHW":
      concat_dim=1
    scope = tf.get_variable_scope()
    scale = self.reader.get_tensor(scope.name + "/gumbel_bn_relu/scale")
    offset = self.reader.get_tensor(scope.name + "/gumbel_bn_relu/offset")
    cdf_value = 0.5 * (1.0 + tf.erf((sparse_threshold - offset)/((1e-5+tf.abs(scale)) * tf.sqrt(2.0))))
    cdf_value = tf.clip_by_value(cdf_value, clip_value_min=0.0, clip_value_max=1.0)
    mask_parameter = 1.0 / (1.0 + tf.exp(-logistic_k*(cdf_value-logistic_c)))
    mask_parameter = tf.clip_by_value(mask_parameter, clip_value_min=0.0, clip_value_max=1.0)
    mask_prob = [1.0-mask_parameter, mask_parameter]
    mask_prob = tf.stack(mask_prob, axis=1)
    argmax_mask_prob = tf.argmax(mask_prob, axis=1, output_type=tf.int32)
    inference_mask = tf.to_float(tf.equal(argmax_mask_prob, 0)).eval(session=self.sess)
    in_channel = tf.reduce_sum(inference_mask).eval(session=self.sess)
    argmax_mask = tf.where(inference_mask)
    argmax_mask = tf.reshape(argmax_mask, [-1]).eval(session=self.sess)

    outputs = batch_norm_init(inputs, is_training, reader=self.reader,name='gumbel_bn_relu', data_format=self.data_format)
    outputs = tf.nn.relu(outputs)
    if self.data_format == "NCHW":
      outputs = tf.gather(outputs, argmax_mask, axis=1)
    elif self.data_format == "NHWC":
      outputs = tf.gather(outputs, argmax_mask, axis=3)
    if is_training:
      self.base_params +=  2 * self.in_channel
      self.base_channels += in_channel
    scope = tf.get_variable_scope()
    w_init = self.reader.get_tensor(scope.name + '/w')
    w_init = tf.gather(w_init, argmax_mask, axis=2)
    print ('%s %d'%(scope.name,in_channel))
    w_init = tf.constant_initializer(w_init.eval(session=self.sess))
    w = create_weight("w", [3,3,in_channel, self.growth_rate],w_init)
    outputs = tf.nn.conv2d(outputs, w, [1,1,1,1], "SAME",data_format=self.data_format)

    if self.data_format == "NHWC":
      h, w = outputs.get_shape()[1].value, outputs.get_shape()[2].value
    elif self.data_format == "NCHW":
      h, w = outputs.get_shape()[2].value, outputs.get_shape()[3].value
    if is_training:
      self.base_params += (3*3*in_channel*self.growth_rate)
      self.base_flops += (2*h*w*(3*3*in_channel+1)*self.growth_rate)

    outputs = tf.concat([inputs, outputs],concat_dim)

    return outputs
  def add_transition(self, inputs, is_training):
    if self.data_format == "NHWC":
      concat_dim=3
    elif self.data_format == "NCHW":
      concat_dim=1
    stride_format=self._get_strides(2)

    scope = tf.get_variable_scope()
    scale = self.reader.get_tensor(scope.name + "/gumbel_bn_relu/scale")
    offset = self.reader.get_tensor(scope.name + "/gumbel_bn_relu/offset")
    cdf_value = 0.5 * (1.0 + tf.erf((sparse_threshold - offset)/((1e-5+tf.abs(scale)) * tf.sqrt(2.0))))
    cdf_value = tf.clip_by_value(cdf_value, clip_value_min=0.0, clip_value_max=1.0)
    mask_parameter = 1.0 / (1.0 + tf.exp(-logistic_k*(cdf_value-logistic_c)))
    mask_parameter = tf.clip_by_value(mask_parameter, clip_value_min=0.0, clip_value_max=1.0)
    mask_prob = [1.0-mask_parameter, mask_parameter]
    mask_prob = tf.stack(mask_prob, axis=1)
    argmax_mask_prob = tf.argmax(mask_prob, axis=1, output_type=tf.int32)
    inference_mask = tf.to_float(tf.equal(argmax_mask_prob, 0)).eval(session=self.sess)
    in_channel = tf.reduce_sum(inference_mask).eval(session=self.sess)
    argmax_mask = tf.where(inference_mask)
    argmax_mask = tf.reshape(argmax_mask, [-1]).eval(session=self.sess)

    outputs = batch_norm_init(inputs, is_training, reader=self.reader,name='gumbel_bn_relu', data_format=self.data_format)
    outputs = tf.nn.relu(outputs)

    if self.data_format == "NCHW":
      outputs = tf.gather(outputs, argmax_mask, axis=1)
    elif self.data_format == "NHWC":
      outputs = tf.gather(outputs, argmax_mask, axis=3)
    if is_training:
      self.base_params += 2 * self.in_channel # bn
      self.base_channels += in_channel

    out_channel = self.in_channel // self.compression_rate
    scope = tf.get_variable_scope()
    print ('%s %d'%(scope.name,in_channel))
    w_init = self.reader.get_tensor(scope.name + '/w')
    w_init = tf.gather(w_init, argmax_mask, axis=2)
    w_init = tf.constant_initializer(w_init.eval(session=self.sess))
    w = create_weight("w", [1,1,in_channel, out_channel], w_init)
    outputs = tf.nn.conv2d(outputs,w,[1,1,1,1], "SAME", data_format=self.data_format)

    if self.data_format == "NHWC":
      h, w = outputs.get_shape()[1].value, outputs.get_shape()[2].value
    elif self.data_format == "NCHW":
      h, w = outputs.get_shape()[2].value, outputs.get_shape()[3].value
    if is_training:
      self.base_params += (1*1*in_channel*out_channel)
      self.base_flops += (2*h*w*(1*1*in_channel+1)*out_channel)

    outputs = tf.nn.avg_pool(outputs, stride_format, stride_format, padding="VALID", data_format=self.data_format)
    return outputs
  def _model(self, images, is_training, reuse=False):
    assert (self.depth-4)%3==0
    N=(self.depth-4) // 3
    with tf.variable_scope(self.name, reuse=reuse) as scope:
      w_init = self.reader.get_tensor(scope.name + '/w')
      w_init = tf.constant_initializer(w_init)
      w=create_weight("w",[3,3,3,self.growth_rate*2], w_init)
      l=tf.nn.conv2d(images,w, [1,1,1,1], "SAME", data_format=self.data_format)
      self.in_channel = self.growth_rate * 2
      if is_training:
        self.base_params += (3*3*3*self.growth_rate*2)
        self.base_flops += (2*32*32*(3*3*3+1)*self.growth_rate*2)
      with tf.variable_scope('block1') as scope:
        for i in range(N):
          with tf.variable_scope('gumbel_dense_layer.{}'.format(i)) as scope:
            l = self.add_layer(l,is_training)
          self.in_channel += self.growth_rate
        with tf.variable_scope('gumbel_transition_layer') as scope:
          l = self.add_transition(l,is_training)
        self.in_channel = self.in_channel // self.compression_rate
      with tf.variable_scope('block2') as scope:
        for i in range(N):
          with tf.variable_scope('gumbel_dense_layer.{}'.format(i)) as scope:
            l = self.add_layer(l,is_training)
          self.in_channel += self.growth_rate
        with tf.variable_scope('gumbel_transition_layer') as scope:
          l = self.add_transition(l,is_training)
        self.in_channel = self.in_channel // self.compression_rate
      with tf.variable_scope('block3') as scope:
        for i in range(N):
          with tf.variable_scope('gumbel_dense_layer.{}'.format(i)) as scope:
            l = self.add_layer(l, is_training)
          self.in_channel += self.growth_rate


      with tf.variable_scope('fc') as scope:
        scale = self.reader.get_tensor(scope.name + "/gumbel_bn_relu/scale")
        offset = self.reader.get_tensor(scope.name + "/gumbel_bn_relu/offset")
        cdf_value = 0.5 * (1.0 + tf.erf((sparse_threshold - offset)/((1e-5+tf.abs(scale)) * tf.sqrt(2.0))))
        cdf_value = tf.clip_by_value(cdf_value, clip_value_min=0.0, clip_value_max=1.0)
        mask_parameter = 1.0 / (1.0 + tf.exp(-logistic_k*(cdf_value-logistic_c)))
        mask_parameter = tf.clip_by_value(mask_parameter, clip_value_min=0.0, clip_value_max=1.0)
        mask_prob = [1.0-mask_parameter, mask_parameter]
        mask_prob = tf.stack(mask_prob, axis=1)
        argmax_mask_prob = tf.argmax(mask_prob, axis=1, output_type=tf.int32)
        inference_mask = tf.to_float(tf.equal(argmax_mask_prob, 0)).eval(session=self.sess)
        in_channel = tf.reduce_sum(inference_mask).eval(session=self.sess)
        argmax_mask = tf.where(inference_mask)
        argmax_mask = tf.reshape(argmax_mask, [-1]).eval(session=self.sess)

        l = batch_norm_init(l, is_training, reader=self.reader,name='gumbel_bn_relu',
                            data_format=self.data_format)
        l = tf.nn.relu(l)
        if self.data_format == "NCHW":
          l = tf.gather(l, argmax_mask, axis=1)
        elif self.data_format == "NHWC":
          l = tf.gather(l, argmax_mask, axis=3)

        if is_training:
          self.base_params += 2 * self.in_channel # bn
          self.base_channels += in_channel

        l = global_avg_pool(l, data_format=self.data_format)

        w_init = self.reader.get_tensor(scope.name + '/w')
        w_init = tf.gather(w_init, argmax_mask)
        w_init = tf.constant_initializer(w_init.eval(session=self.sess))
        b_init = self.reader.get_tensor(scope.name + '/b')
        b_init = tf.constant_initializer(b_init)
        w=create_weight("w", [in_channel,10],w_init)
        #b=create_bias("b",[10],b_init )
        b=create_bias("offset",[10],b_init )

        x= tf.nn.bias_add(tf.matmul(l,w),b)
        if is_training:
          self.base_flops += (2*(in_channel-1)*10)
          self.base_params += (in_channel+1)*10
    return x
    # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    self.base_flops = 0
    self.base_params = 0
    self.base_channels = 0
    logits = self._model(self.x_train, is_training=True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=self.y_train)
    original_flops=
    original_params=
    original_channels=
    print ('Base_Flops: %d'%self.base_flops)
    print ('Base_Flops_pruned: %.2f'%(100.0*float(original_flops-self.base_flops)/float
                                      (original_flops)))
    print ('Base_Params: %d'%self.base_params)
    print ('Base_Params_pruned: %.2f'%(100.0*float(original_params-self.base_params)/float
                                       (original_params)))
    print ('Base_Channels: %d'%self.base_channels)
    print ('Base_Channels_pruned: %.2f'%(100.0*float(original_channels-self.base_channels)/float
                                         (original_channels)))


    self.loss = tf.reduce_mean(log_probs)




    self.train_preds = tf.argmax(logits, axis=1)
    self.train_preds = tf.to_int32(self.train_preds)
    self.train_acc = tf.equal(self.train_preds, self.y_train)
    self.train_acc = tf.to_int32(self.train_acc)
    self.train_acc = tf.reduce_sum(self.train_acc)

    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print("Model has {} params".format(self.num_vars))

    self.global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")
    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.global_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      lr_cosine=self.lr_cosine,
      lr_max=self.lr_max,
      lr_min=self.lr_min,
      lr_T_0=self.lr_T_0,
      lr_T_mul=self.lr_T_mul,
      num_train_batches=self.num_train_batches,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas,
      child_lr_otf=self.child_lr_otf)

  def _build_valid(self):
    if self.x_valid is not None:
      print("-" * 80)
      print("Build valid graph")
      logits = self._model(self.x_valid, False, reuse=True)
      self.valid_preds = tf.argmax(logits, axis=1)
      self.valid_preds = tf.to_int32(self.valid_preds)
      self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
      self.valid_acc = tf.to_int32(self.valid_acc)
      self.valid_acc = tf.reduce_sum(self.valid_acc)

  # override
  def _build_test(self):
    print("-" * 80)
    print("Build test graph")
    logits = self._model(self.x_test, False, reuse=True)
    self.test_preds = tf.argmax(logits, axis=1)
    self.test_preds = tf.to_int32(self.test_preds)
    self.test_acc = tf.equal(self.test_preds, self.y_test)
    self.test_acc = tf.to_int32(self.test_acc)
    self.test_acc = tf.reduce_sum(self.test_acc)

