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
from src.cifar10_channel_prune.image_ops import batch_norm
from src.cifar10_channel_prune.gumbel_prune import dense_block_gumbel_prune
from src.cifar10_channel_prune.image_ops import relu
from src.cifar10_channel_prune.image_ops import max_pool
from src.cifar10_channel_prune.image_ops import global_avg_pool

from src.utils import count_model_params
from src.cifar10_channel_prune.utils import get_train_ops
from src.common_ops import create_weight
from src.common_ops import create_bias
MASK_REGULARIZER = "MASK_REGULARIZER"
class densenet(Model):
  def __init__(self,
               images,
               labels,
               sparse_threshold,
               logistic_k=10000,
               logistic_c=0.9,
               temperature=0.1,
               sparse_bernoulli=0.001,
	       s=2.0,
               depth=40,
               growth_rate=12,
               compression_rate=1,
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
               name="ResNet",
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
    self.channel_mask = OrderedDict()
    self.sparse_threshold = sparse_threshold
    self.logistic_k=logistic_k
    self.logistic_c=logistic_c
    self.temperature=temperature
    self.s=s
    self.sparse_bernoulli=sparse_bernoulli
    self.gumbel_dict = {}
    self.gumbel_dict["sparse_threshold"] = self.sparse_threshold
    self.gumbel_dict["logistic_k"] = self.logistic_k
    self.gumbel_dict["logistic_c"] = self.logistic_c
    self.gumbel_dict["temperature"] = self.temperature
    self.gumbel_dict["s"] = self.s

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
      in_channel=inputs.get_shape()[3].value
      concat_dim=3
    elif self.data_format == "NCHW":
      in_channel=inputs.get_shape()[1].value
      concat_dim=1
    outputs, mask = dense_block_gumbel_prune(inputs, is_training,data_format=self.data_format,
                                             **self.gumbel_dict)

    scope = tf.get_variable_scope()
    if scope.name+"mask" not in self.channel_mask:
      mask_sum = tf.reduce_sum(mask)
      self.channel_mask[scope.name + "mask"] = mask_sum
    mask_sum_int = tf.cast(self.channel_mask[scope.name+"mask"], tf.int32)

    if is_training:
      self.base_params += 2 * in_channel # bn
      self.base_channels += in_channel
    else:
      self.dynamic_params_ += 2 * in_channel # bn
      self.dynamic_channels_ += mask_sum_int

    w = create_weight("w", [3,3,in_channel, self.growth_rate])
    outputs = tf.nn.conv2d(outputs, w, [1,1,1,1], "SAME",data_format=self.data_format)

    if self.data_format == "NHWC":
      h, w = outputs.get_shape()[1].value, outputs.get_shape()[2].value
    elif self.data_format == "NCHW":
      h, w = outputs.get_shape()[2].value, outputs.get_shape()[3].value
    if is_training:
      self.base_params += (3*3*in_channel*self.growth_rate)
      self.base_flops += (2*h*w*(3*3*in_channel+1)*self.growth_rate)
    else:
      self.dynamic_params_ += (3*3*mask_sum_int*self.growth_rate)
      self.dynamic_flops_ += (2*h*w*(3*3*mask_sum_int+1)*self.growth_rate)

    outputs = tf.concat([inputs, outputs],concat_dim)
    return outputs
  def add_transition(self, inputs, is_training):
    if self.data_format == "NHWC":
      in_channel=inputs.get_shape()[3].value
      concat_dim=3
    elif self.data_format == "NCHW":
      in_channel=inputs.get_shape()[1].value
      concat_dim=1
    stride_format=self._get_strides(2)
    outputs, mask = dense_block_gumbel_prune(inputs, is_training,data_format=self.data_format,
                                             **self.gumbel_dict)

    scope = tf.get_variable_scope()
    if scope.name+"mask" not in self.channel_mask:
      mask_sum = tf.reduce_sum(mask)
      self.channel_mask[scope.name + "mask"] = mask_sum
    mask_sum_int = tf.cast(self.channel_mask[scope.name+"mask"], tf.int32)

    if is_training:
      self.base_params += 2 * in_channel # bn
      self.base_channels += in_channel
    else:
      self.dynamic_params_ += 2 * in_channel # bn
      self.dynamic_channels_ += mask_sum_int

    out_channel = self.in_channel // self.compression_rate
    w = create_weight("w", [1,1,in_channel, out_channel])
    outputs = tf.nn.conv2d(outputs,w,[1,1,1,1], "SAME", data_format=self.data_format)

    if self.data_format == "NHWC":
      h, w = outputs.get_shape()[1].value, outputs.get_shape()[2].value
    elif self.data_format == "NCHW":
      h, w = outputs.get_shape()[2].value, outputs.get_shape()[3].value
    if is_training:
      self.base_params += (1*1*in_channel*out_channel)
      self.base_flops += (2*h*w*(1*1*in_channel+1)*out_channel)
    else:
      self.dynamic_params_ += (1*1*mask_sum_int*out_channel)
      self.dynamic_flops_ += (2*h*w*(1*1*mask_sum_int+1)*out_channel)

    outputs = tf.nn.avg_pool(outputs, stride_format, stride_format, padding="VALID", data_format=self.data_format)
    return outputs
  def _model(self, images, is_training, reuse=False):
    assert (self.depth-4)%3==0
    N=(self.depth-4) // 3
    with tf.variable_scope(self.name, reuse=reuse):
      w=create_weight("w",[3,3,3,self.growth_rate*2])
      l=tf.nn.conv2d(images,w, [1,1,1,1], "SAME", data_format=self.data_format)
      self.in_channel = self.growth_rate * 2
      if is_training:
        self.base_params += (3*3*3*self.growth_rate*2)
        self.base_flops += (2*32*32*(3*3*3+1)*self.growth_rate*2)
      else:
        self.dynamic_params_ += (3*3*3*self.growth_rate*2)
        self.dynamic_flops_ += (2*32*32*(3*3*3+1)*self.growth_rate*2)
      with tf.variable_scope('block1') as scope:
        for i in range(N):
          with tf.variable_scope('gumbel_dense_layer.{}'.format(i)) as scope:
            l = self.add_layer(l,is_training)
          print (l)
          self.in_channel += self.growth_rate
        with tf.variable_scope('gumbel_transition_layer') as scope:
          l = self.add_transition(l,is_training)
        print (l)
        self.in_channel = self.in_channel // self.compression_rate
      with tf.variable_scope('block2') as scope:
        for i in range(N):
          with tf.variable_scope('gumbel_dense_layer.{}'.format(i)) as scope:
            l = self.add_layer(l,is_training)
          print (l)
          self.in_channel += self.growth_rate
        with tf.variable_scope('gumbel_transition_layer') as scope:
          l = self.add_transition(l,is_training)
        print (l)
        self.in_channel = self.in_channel // self.compression_rate
      with tf.variable_scope('block3') as scope:
        for i in range(N):
          with tf.variable_scope('gumbel_dense_layer.{}'.format(i)) as scope:
            l = self.add_layer(l, is_training)
          print (l)
          self.in_channel += self.growth_rate


      with tf.variable_scope('fc') as scope:
        if self.data_format == "NHWC":
          in_channel=l.get_shape()[3].value
        elif self.data_format == "NCHW":
          in_channel=l.get_shape()[1].value
        l, mask = dense_block_gumbel_prune(l, is_training,data_format=self.data_format,
                                           **self.gumbel_dict)
        if scope.name+"mask" not in self.channel_mask:
          mask_sum = tf.reduce_sum(mask)
          self.channel_mask[scope.name + "mask"] = mask_sum
        mask_sum_int = tf.cast(self.channel_mask[scope.name+"mask"], tf.int32)

        if is_training:
          self.base_params += 2 * in_channel # bn
          self.base_channels += in_channel
        else:
          self.dynamic_params_ += 2 * in_channel # bn
          self.dynamic_channels_ += mask_sum_int

        l = global_avg_pool(l, data_format=self.data_format)
        w=create_weight("w", [in_channel,10])
        b=create_bias("offset",[10])

        x= tf.nn.bias_add(tf.matmul(l,w),b)
        if is_training:
          self.base_flops += (2*(in_channel-1)*10)
          self.base_params += (in_channel+1)*10
        else:
          self.dynamic_flops_ += (2*(mask_sum_int-1)*10)
          self.dynamic_params_ += (mask_sum_int+1)*10
        print (l)
    return x
    # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    self.base_flops = 0
    self.base_params = 0
    self.base_channels = 0
    self.dynamic_params = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
    self.dynamic_params_ = self.dynamic_params
    self.dynamic_flops = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
    self.dynamic_flops_ = self.dynamic_flops
    self.dynamic_channels = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
    self.dynamic_channels_ = self.dynamic_channels
    logits = self._model(self.x_train, is_training=True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=self.y_train)
    print ('Base_Flops: %d'%self.base_flops)
    print ('Base_Params: %d'%self.base_params)
    print ('Base_Channels: %d'%self.base_channels)
    self.loss = tf.reduce_mean(log_probs)
    if self.sparse_bernoulli > 0.0:
      self.sparse_bernoulli_loss1 = [tf.reduce_sum(var) for var in tf.get_collection(MASK_REGULARIZER)]
      self.sparse_bernoulli_loss = self.sparse_bernoulli * tf.add_n(self.sparse_bernoulli_loss1)
      self.loss += self.sparse_bernoulli_loss
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

