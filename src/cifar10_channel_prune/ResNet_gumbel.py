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
class resnet(Model):
  def __init__(self,
               images,
               labels,
               sparse_threshold,
               logistic_k=10000,
               logistic_c=0.9,
               temperature=0.1,
               sparse_bernoulli=0.001,
               s=2.0,
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
    self.channel_mask = OrderedDict()
    self.sparse_threshold = sparse_threshold
    self.logistic_k=logistic_k
    self.logistic_c=logistic_c
    self.temperature=temperature
    self.sparse_bernoulli=sparse_bernoulli
    self.s=s
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
  def _basic_residual_block(self,inputs, is_training, strides, out_filters,filter_size=3, first_block=False):
    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3].value
      stride_format = [1,strides, strides, 1]
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1].value
      stride_format = [1,1,strides,strides]
    with tf.variable_scope('residual_block') as scope:
      if first_block:
        with tf.variable_scope('bn_relu_conv_1') as scope:
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
        with tf.variable_scope('bn_relu_conv_2') as scope:
          x, mask2 = dense_block_gumbel_prune(x, is_training, data_format=self.data_format,
                                             **self.gumbel_dict)
          if scope.name+"mask" not in self.channel_mask:
            mask2_sum = tf.reduce_sum(mask2)
            self.channel_mask[scope.name + "mask"] = mask2_sum
          mask2_sum_int = tf.cast(self.channel_mask[scope.name+"mask"], tf.int32)

          w = create_weight("w", [filter_size,filter_size, out_filters, out_filters])
          x = tf.nn.conv2d(x, w, [1,1,1,1], "SAME", data_format = self.data_format)
        non_zero = mask2_sum_int
        non_zero = tf.cond(tf.equal(non_zero, 0),
                           lambda: tf.zeros_like(non_zero),
                           lambda: tf.ones_like(non_zero))
        if self.data_format == "NHWC":
          h, w = x.get_shape()[1].value, x.get_shape()[2].value
        elif self.data_format == "NCHW":
          h, w = x.get_shape()[2].value, x.get_shape()[3].value

        if is_training:
          self.base_params += 2 * inp_c # bn
          self.base_channels += inp_c

          self.base_params += (filter_size*filter_size*inp_c*out_filters) # conv
          self.base_flops += (2*h*w*(filter_size*filter_size*inp_c+1)*out_filters)

          self.base_params += 2 * out_filters # bn
          self.base_channels += out_filters

          self.base_params += (filter_size*filter_size*out_filters*out_filters) # conv
          self.base_flops += (2*h*w*(filter_size*filter_size*out_filters+1)*out_filters)
        else:
          self.dynamic_params_ += 2 * inp_c * non_zero # bn
          self.dynamic_channels_ +=  inp_c * non_zero

          self.dynamic_params_ += (filter_size*filter_size*inp_c*out_filters) * non_zero # conv
          self.dynamic_flops_ += (2*h*w*(filter_size*filter_size*inp_c+1)*out_filters) * non_zero

          self.dynamic_params_ += 2 * mask2_sum_int * non_zero # bn
          self.dynamic_channels_ += mask2_sum_int * non_zero

          self.dynamic_params_ += (filter_size*filter_size*mask2_sum_int*out_filters) * non_zero # conv
          self.dynamic_flops_ += (2*h*w*(filter_size*filter_size*mask2_sum_int+1)*out_filters) * non_zero

        if strides > 1 or inp_c != out_filters:
          with tf.variable_scope('shortcut'):
            w = create_weight("w", [1,1,inp_c, out_filters])
            shortcut = tf.nn.conv2d(shortcut, w, stride_format, pad, data_format=self.data_format)
            if self.data_format == "NHWC":
              h, w = shortcut.get_shape()[1].value, shortcut.get_shape()[2].value
            elif self.data_format == "NCHW":
              h, w = shortcut.get_shape()[2].value, shortcut.get_shape()[3].value
            if not is_training:
              self.dynamic_params_ += (1*1*inp_c*out_filters)
              self.dynamic_flops_ += (2*h*w*(inp_c*1*1+1)*out_filters)
            else:
              self.base_params += (1*1*inp_c*out_filters)
              self.base_flops += (2*h*w*(inp_c*1*1+1)*out_filters)

        out = x + shortcut
        return out
      else:
        shortcut = inputs
        with tf.variable_scope('bn_relu_conv_1') as scope:
          x, mask = dense_block_gumbel_prune(inputs, is_training, data_format=self.data_format,
                                             **self.gumbel_dict)
          if scope.name+"mask" not in self.channel_mask:
            mask_sum = tf.reduce_sum(mask)
            self.channel_mask[scope.name + "mask"] = mask_sum
          mask_sum_int = tf.cast(self.channel_mask[scope.name+"mask"], tf.int32)

          w = create_weight("w", [filter_size,filter_size, inp_c, out_filters])
          x = tf.nn.conv2d(x, w, [1,1,1,1], "SAME", data_format=self.data_format)
        with tf.variable_scope('bn_relu_conv_2') as scope:
          x, mask2 = dense_block_gumbel_prune(x, is_training, data_format=self.data_format,
                                             **self.gumbel_dict)
          if scope.name+"mask" not in self.channel_mask:
            mask2_sum = tf.reduce_sum(mask2)
            self.channel_mask[scope.name + "mask"] = mask2_sum
          mask2_sum_int = tf.cast(self.channel_mask[scope.name+"mask"], tf.int32)

          w = create_weight("w", [filter_size,filter_size, out_filters, out_filters])
          x = tf.nn.conv2d(x, w, [1,1,1,1], "SAME", data_format = self.data_format)

        non_zero = mask_sum_int * mask2_sum_int
        non_zero = tf.cond(tf.equal(non_zero, 0),
                           lambda: tf.zeros_like(non_zero),
                           lambda: tf.ones_like(non_zero))
        if self.data_format == "NHWC":
          h, w = x.get_shape()[1].value, x.get_shape()[2].value
        elif self.data_format == "NCHW":
          h, w = x.get_shape()[2].value, x.get_shape()[3].value

        if is_training:
          self.base_params += 2 * inp_c # bn
          self.base_channels += inp_c

          self.base_params += (filter_size*filter_size*inp_c*out_filters) # conv
          self.base_flops += (2*h*w*(filter_size*filter_size*inp_c+1)*out_filters)

          self.base_params += 2 * out_filters # bn
          self.base_channels += out_filters

          self.base_params += (filter_size*filter_size*out_filters*out_filters) # conv
          self.base_flops += (2*h*w*(filter_size*filter_size*out_filters+1)*out_filters)
        else:
          self.dynamic_params_ += 2 * inp_c * non_zero # bn
          self.dynamic_channels_ += mask_sum_int * non_zero

          self.dynamic_params_ += (filter_size*filter_size*mask_sum_int*mask2_sum_int) * non_zero # conv
          self.dynamic_flops_ += (2*h*w*(filter_size*filter_size*mask_sum_int+1)*mask2_sum_int) * non_zero

          self.dynamic_params_ += 2 * mask2_sum_int * non_zero # bn
          self.dynamic_channels_ += mask2_sum_int * non_zero

          self.dynamic_params_ += (filter_size*filter_size*mask2_sum_int*out_filters) * non_zero # conv
          self.dynamic_flops_ += (2*h*w*(filter_size*filter_size*mask2_sum_int+1)*out_filters) * non_zero

        out = x + shortcut
        return out

  def _model(self, images, is_training, reuse=False):
    assert self.num_residual_block%3==0
    each_block=self.num_residual_block//3
    filters = [16,32,64]
    block = self._basic_residual_block
    expansion=1
    strides = [1,2,2]
    with tf.variable_scope(self.name, reuse=reuse):
      w=create_weight("w",[3,3,3,16])
      x=tf.nn.conv2d(images,w, [1,1,1,1], "SAME", data_format=self.data_format)

      if is_training:
        self.base_flops += (2*(3*3*3+1)*16*32*32)
        self.base_params += (3*3*3*16)
      else:
        self.dynamic_params_ = self.dynamic_params + 3*3*3*16
        self.dynamic_flops_ = self.dynamic_flops + 2*(3*3*3+1)*16*32*32

      self.dynamic_channels_ = self.dynamic_channels + 0
      in_planes = 16
      for i in range(self.num_residual_block):
        with tf.variable_scope('layer_{0}'.format(i)):
          first_block=True if i % each_block==0 else False
          x = self._basic_residual_block(x, is_training, strides[i//each_block],
                      filters[i//each_block], first_block=first_block)
          print (x)
      with tf.variable_scope('fc') as scope:
        x, mask = dense_block_gumbel_prune(x, is_training, data_format=self.data_format,
                                     **self.gumbel_dict)
        if scope.name + "mask" not in self.channel_mask:
          mask_sum = tf.reduce_sum(mask)
          self.channel_mask[scope.name + "mask"] = mask_sum
        mask_sum_int = tf.cast(self.channel_mask[scope.name + "mask"], tf.int32)
        if not is_training:
          self.dynamic_params_ += 2 * expansion * 64
          self.dynamic_channels_ += mask_sum_int
        else:
          self.base_params += 2 * expansion * 64
          self.base_channels += expansion * 64
        x = global_avg_pool(x, data_format=self.data_format)
        if self.data_format == "NHWC":
          inp_c=x.get_shape()[3].value
        elif self.data_format == "NCHW":
          inp_c=x.get_shape()[1].value
        w=create_weight("w", [inp_c,10])
        b=create_bias("offset",[10])
        x= tf.nn.bias_add(tf.matmul(x,w),b)
        if is_training:
          self.base_flops += (2*(inp_c-1) * 10)
          self.base_params += (inp_c+1) * 10
        else:
          self.dynamic_params_ += ((mask_sum_int+1) * 10)
          self.dynamic_flops_ += (2*(mask_sum_int-1) * 10)
    print (x)
    return x
    # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    self.base_flops = 0
    self.base_params = 0
    self.base_channels = 0
    self.dynamic_params = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
    self.dynamic_flops = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
    self.dynamic_channels = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
    logits = self._model(self.x_train, is_training=True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=self.y_train)
    print ('Base_Flops: %d'%self.base_flops)
    print ('Base_Params: %d'%self.base_params)
    print ('Base_Channels: %d'%self.base_channels)
    self.loss = tf.reduce_mean(log_probs)
    if self.sparse_bernoulli > 0.0:
      self.sparse_bernoulli_loss = [tf.reduce_sum(var)
                                     for var in tf.get_collection(MASK_REGULARIZER)]
      self.sparse_bernoulli_loss = self.sparse_bernoulli * tf.add_n(self.sparse_bernoulli_loss)

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

