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
from src.cifar10_channel_prune.image_ops import relu
from src.cifar10_channel_prune.image_ops import max_pool
from src.cifar10_channel_prune.image_ops import global_avg_pool

from src.utils import count_model_params
from src.cifar10_channel_prune.utils import get_train_ops
from src.common_ops import create_weight
from src.common_ops import create_bias
class resnet(Model):
  def __init__(self,
               images,
               labels,
               num_residual_block=27,
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
    self.num_residual_block=num_residual_block
    self.lr_cosine = lr_cosine
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.lr_T_0 = lr_T_0
    self.lr_T_mul = lr_T_mul

    self.child_lr_otf = child_lr_otf
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
  def _basic_residual_block(self,inputs, is_training, strides, out_filters,filter_size=3,
                            first_block=False):
    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3].value
      stride_format = [1,strides, strides, 1]
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1].value
      stride_format = [1,1,strides,strides]
    with tf.variable_scope('residual_block') as scope:
      if first_block:
        with tf.variable_scope('bn_relu_conv_1'):
          x = batch_norm(inputs, is_training, data_format=self.data_format)
          x = tf.nn.relu(x)
          shortcut = x
          if strides == 1:
            pad = "SAME"
          else:
            pad_total = filter_size-1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            if self.data_format == "NHWC":
                x = tf.pad(x, [[0,0], [pad_beg, pad_end], [pad_beg, pad_end],[0,0]])
            elif self.data_format == "NCHW":
                x = tf.pad(x, [[0,0], [0,0], [pad_beg, pad_end], [pad_beg, pad_end]])
            pad = "VALID"
          w = create_weight("w", [filter_size,filter_size, inp_c, out_filters])
          x = tf.nn.conv2d(x, w, stride_format, pad, data_format=self.data_format)
        with tf.variable_scope('bn_relu_conv_2'):
          x = batch_norm(x, is_training, data_format=self.data_format)
          x = tf.nn.relu(x)
          w = create_weight("w", [filter_size,filter_size, out_filters, out_filters])
          x = tf.nn.conv2d(x, w, [1,1,1,1], "SAME", data_format = self.data_format)

        if strides > 1 or inp_c != out_filters:
          with tf.variable_scope('shortcut'):
            w = create_weight("w", [1,1,inp_c, out_filters])
            shortcut = tf.nn.conv2d(shortcut, w, stride_format, pad, data_format=self.data_format)
        out = x + shortcut
        return out
      else:
        if inp_c != out_filters:
          raise ValueError ('channel should be same with out_filters')
        shortcut = inputs
        with tf.variable_scope('bn_relu_conv_1'):
          x = batch_norm(inputs, is_training, data_format=self.data_format)
          x = tf.nn.relu(x)
          w = create_weight("w", [filter_size,filter_size, inp_c, out_filters])
          x = tf.nn.conv2d(x, w, [1,1,1,1], "SAME", data_format=self.data_format)
        with tf.variable_scope('bn_relu_conv_2'):
          x = batch_norm(x, is_training, data_format=self.data_format)
          x = tf.nn.relu(x)
          w = create_weight("w", [filter_size,filter_size, out_filters, out_filters])
          x = tf.nn.conv2d(x, w, [1,1,1,1], "SAME", data_format = self.data_format)

        out = x + shortcut
        return out
  def _model(self, images, is_training, reuse=False):
    assert self.num_residual_block%3==0
    each_block=self.num_residual_block//3
    filters = [16,32,64]
    strides=[1,2,2]
    with tf.variable_scope(self.name, reuse=reuse):
      w=create_weight("w",[3,3,3,16])
      x=tf.nn.conv2d(images,w, [1,1,1,1], "SAME", data_format=self.data_format)
      for i in range(self.num_residual_block):
        with tf.variable_scope('layer_{0}'.format(i)):
          first_block=True if i % each_block==0 else False
          x = self._basic_residual_block(x, is_training, strides[i//each_block],
                    filters[i//each_block], first_block=first_block)
          print (x)

      with tf.variable_scope('fc'):
        x = batch_norm(x, is_training, data_format=self.data_format)
        x = tf.nn.relu(x)
        x = global_avg_pool(x, data_format=self.data_format)
        if self.data_format == "NHWC":
          inp_c=x.get_shape()[3].value
        elif self.data_format == "NCHW":
          inp_c=x.get_shape()[1].value
        w=create_weight("w", [inp_c,10])
        b=create_bias("offset",[10])

        x= tf.nn.bias_add(tf.matmul(x,w),b)
    print (x)
    return x
    # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    logits = self._model(self.x_train, is_training=True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=self.y_train)

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

