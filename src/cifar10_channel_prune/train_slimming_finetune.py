from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pickle
import shutil
import sys
import time

import pdb
import numpy as np
import tensorflow as tf

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags

from src.cifar10_channel_prune.data_utils import read_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", True, "Delete output_dir if exists.")
DEFINE_string("data_path", "./data/cifar10", "")
DEFINE_string("network", "", "")
DEFINE_string("load_dir", "", "")
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCWH'")
DEFINE_float("lr_init",0.1, "")
DEFINE_float("lr_warmup_val",None, "")
DEFINE_float("percent",None, "")
DEFINE_integer("lr_warmup_steps",0, "")
DEFINE_integer("batch_size",64, "")
DEFINE_integer("num_epochs",160, "")
DEFINE_integer("log_every", 100, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

def get_ops(images, labels, reader,sess):
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """
  num_channels = 0
  scale_list = []

  if FLAGS.network == "VggNet19":
    from src.cifar10_channel_prune.Vgg_slimming_finetune import vggnet
    block = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
             512, 512, 512, 512, 'M', 512, 512, 512, 512]
    for i, filters in enumerate(block):
        if filters != 'M':
            layer_name = "VggNet/layer_{0}/vgg_block/".format(i)
            scale = reader.get_tensor(layer_name + "bn/scale")
            scale = np.absolute(scale)
            scale_list.append(scale)
            num_channels += np.shape(scale)[0]

    total_scale = np.concatenate(scale_list, axis =0)
    sort_total_scale = np.sort(total_scale, axis=None)
    threshold = sort_total_scale[num_channels - int((1.0-FLAGS.percent)*float(num_channels))]

    network = vggnet(images, labels,
                    threshold=threshold,
                    reader=reader,
                    depth=19,
                    sess=sess,
                    batch_size=FLAGS.batch_size,
                    name="VggNet",
                    lr_init=FLAGS.lr_init,
                    lr_warmup_val=FLAGS.lr_warmup_val, lr_warmup_steps=FLAGS.lr_warmup_steps)
  elif FLAGS.network == "VggNet16":
    from src.cifar10_channel_prune.Vgg_slimming_finetune import vggnet
    block = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    for i, filters in enumerate(block):
        if filters != 'M':
            layer_name = "VggNet/layer_{0}/vgg_block/".format(i)
            scale = reader.get_tensor(layer_name + "bn/scale")
            scale = np.absolute(scale)
            scale_list.append(scale)
            num_channels += np.shape(scale)[0]

    total_scale = np.concatenate(scale_list, axis =0)
    sort_total_scale = np.sort(total_scale, axis=None)
    threshold = sort_total_scale[num_channels - int((1.0-FLAGS.percent)*float(num_channels))]
    network = vggnet(images, labels,
                     threshold=threshold,
                     reader=reader,
                     depth=16,
                     sess=sess,
                     batch_size=FLAGS.batch_size,
                     name="VggNet",
                     lr_init=FLAGS.lr_init,
                     lr_warmup_val=FLAGS.lr_warmup_val, lr_warmup_steps=FLAGS.lr_warmup_steps)
  elif FLAGS.network == "DenseNet":
    from src.cifar10_channel_prune.DenseNet_slimming_finetune import densenet
    depth = 40
    depth_per_block = (40-4) // 3
    for i in range(depth_per_block):
        layer_name = "DenseNet/block1/dense_layer.{0}/".format(i)
        scale_name = layer_name + "bn/scale"
        scale = reader.get_tensor(scale_name)
        scale = np.absolute(scale)
        scale_list.append(scale)
        num_channels += np.shape(scale)[0]
    layer_name = "DenseNet/block1/transition_layer/"
    scale_name = layer_name + "bn/scale"
    scale = reader.get_tensor(scale_name)
    scale = np.absolute(scale)
    scale_list.append(scale)
    num_channels += np.shape(scale)[0]

    for i in range(depth_per_block):
        layer_name = "DenseNet/block2/dense_layer.{0}/".format(i)
        scale_name = layer_name + "bn/scale"
        scale = reader.get_tensor(scale_name)
        scale = np.absolute(scale)
        scale_list.append(scale)
        num_channels += np.shape(scale)[0]
    layer_name = "DenseNet/block2/transition_layer/"
    scale_name = layer_name + "bn/scale"
    scale = reader.get_tensor(scale_name)
    scale = np.absolute(scale)
    scale_list.append(scale)
    num_channels += np.shape(scale)[0]

    for i in range(depth_per_block):
        layer_name = "DenseNet/block3/dense_layer.{0}/".format(i)
        scale_name = layer_name + "bn/scale"
        scale = reader.get_tensor(scale_name)
        scale = np.absolute(scale)
        scale_list.append(scale)
        num_channels += np.shape(scale)[0]

    layer_name = "DenseNet/fc/"
    scale_name = layer_name + "bn/scale"
    scale = reader.get_tensor(scale_name)
    scale = np.absolute(scale)
    scale_list.append(scale)
    num_channels += np.shape(scale)[0]

    total_scale = np.concatenate(scale_list, axis =0)
    sort_total_scale = np.sort(total_scale, axis=None)
    threshold = sort_total_scale[num_channels - int((1.0-FLAGS.percent)*float(num_channels))]
    network = densenet(images, labels,
                    threshold=threshold,
                    reader=reader,
                    sess=sess,
                    batch_size=FLAGS.batch_size,
                    name="DenseNet",
                    lr_init=FLAGS.lr_init,
                    lr_warmup_val=FLAGS.lr_warmup_val, lr_warmup_steps=FLAGS.lr_warmup_steps)
  child_ops = {
    "global_step": network.global_step,
    "loss": network.loss,
    "train_op": network.train_op,
    "lr": network.lr,
    "grad_norm": network.grad_norm,
    "train_acc": network.train_acc,
    "num_train_batches": network.num_train_batches,
    }
  ops = {
    "child_list": child_ops,
    "eval_every": network.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func":  network.eval_once,
    "num_train_batches":network.num_train_batches,
  }

  return ops


def train():
  images, labels = read_data(FLAGS.data_path, num_valids=0)
  global best_test_acc
  global best_channel_mask
  global output_dir
  global load_dir
  global best_flops
  global best_params
  global best_channels
  global ckpt_state
  g = tf.Graph()
  with g.as_default():
    ckpt_state = tf.train.get_checkpoint_state(load_dir)
    reader = tf.train.NewCheckpointReader(ckpt_state.model_checkpoint_path)

    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    ops = get_ops(images, labels,reader, sess)
    child_ops = ops["child_list"]
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)

    tf.train.start_queue_runners(sess=sess)
    test_acc = ops["eval_func"](sess, "test")

    with tf.name_scope('Train'):
        start_time = time.time()
        while True:
          run_ops = [
            child_ops["loss"],
            child_ops["lr"],
            child_ops["grad_norm"],
            child_ops["train_acc"],
            child_ops["train_op"],
            ]
          receive_ops = sess.run(run_ops)
          global_step = sess.run(child_ops["global_step"])

          epoch = global_step // ops["num_train_batches"]
          curr_time = time.time()
          if global_step % FLAGS.log_every == 0:
            loss, lr, gn, tr_acc, _ = receive_ops[0:]
            log_string = ""
            log_string += "epoch={:<6d}".format(epoch)
            log_string += "ch_step={:<6d}".format(global_step)
            log_string += " loss={:<8.6f}".format(loss)
            log_string += " lr={:<8.4f}".format(lr)
            log_string += " |g|={:<8.4f}".format(gn)
            log_string += " tr_acc={:<3d}/{:>3d}".format(
            tr_acc, FLAGS.batch_size)
            log_string += " mins={:<10.2f}".format(
            float(curr_time - start_time) / 60)
            print(log_string)


          if global_step % ops["num_train_batches"] == 0:
            print("Epoch {}: Eval".format(epoch))
            test_acc = ops["eval_func"](sess, "test")
            if best_test_acc < test_acc:
              best_test_acc = test_acc
              checkpoint_path = os.path.join(output_dir, "model.ckpt")
              saver.save(sess, checkpoint_path, global_step=global_step)
            print ('best_test_acc: %.2f'%best_test_acc)
          if epoch >= FLAGS.num_epochs:
            print ('best_test_acc: %.2f'%best_test_acc)
            break

def main(_):
  print("-" * 80)
  global  best_test_acc
  global best_params
  global best_flops
  global best_channels
  global output_dir
  global load_dir
  if FLAGS.network == "VggNet19":
    load_dir='cifar10_VggNet19_slimming_network'
    output_dir = "cifar10_VggNet19_slimming_finetune_%s"%FLAGS.percent
  elif FLAGS.network == "VggNet16":
    load_dir='cifar10_VggNet16_slimming_network'
    output_dir = "cifar10_VggNet16_slimming_finetune_%s"%FLAGS.percent
  elif FLAGS.network == "DenseNet":
    load_dir='cifar10_DenseNet_slimming_network'
    output_dir = "cifar10_DenseNet_slimming_finetune_%s"%FLAGS.percent
  else:
    raise ValuedError("Network should be [preResNet, VggNet19, VggNet16, DenseNet]")
  if not os.path.isdir(output_dir):
    print("Path {} does not exist. Creating.".format(output_dir))
    os.makedirs(output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(output_dir))
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

  print("-" * 80)
  log_file = os.path.join(output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)
  train()
  utils.print_user_flags()

if __name__ == "__main__":
  best_test_acc, best_flops, best_params, best_channels = 0.0, 0, 0, 0
  best_channel_mask = {}
  tf.app.run()
