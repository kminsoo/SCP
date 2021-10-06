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
DEFINE_string("network", "VggNet19", "'ResNet', 'VggNet19' 'VggNet16' or 'DenseNet'" )
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCWH'")
DEFINE_float("lr_warmup_val",None, "")
DEFINE_float("sparse_threshold",0.05, "")
DEFINE_float("logistic_k",20.0, "")
DEFINE_float("logistic_c",None, "")
DEFINE_float("temperature",0.5, "")
DEFINE_float("sparse_bernoulli",0.001, "")
DEFINE_float("s",2.0, "")
DEFINE_integer("lr_warmup_steps",0, "")
DEFINE_integer("batch_size",64, "")
DEFINE_integer("num_epochs",160, "")
DEFINE_integer("log_every", 100, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

def get_ops(images, labels):
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """
  if FLAGS.network == "ResNet":
    from src.cifar10_channel_prune.ResNet_gumbel import resnet
    network = resnet(images, labels,
                     num_residual_block= 27,
                     batch_size=FLAGS.batch_size,
                     sparse_threshold=FLAGS.sparse_threshold,
                     logistic_k=FLAGS.logistic_k,
                     logistic_c=FLAGS.logistic_c,
                     temperature=FLAGS.temperature,
                     sparse_bernoulli=FLAGS.sparse_bernoulli,
                     s=FLAGS.s,
                     name="ResNet",
                     lr_warmup_val=FLAGS.lr_warmup_val, lr_warmup_steps=FLAGS.lr_warmup_steps)
  elif FLAGS.network == "VggNet19":
    from src.cifar10_channel_prune.Vgg_gumbel import vggnet
    network = vggnet(images, labels,
                     batch_size=FLAGS.batch_size,
                     sparse_threshold=FLAGS.sparse_threshold,
                     logistic_k=FLAGS.logistic_k,
                     logistic_c=FLAGS.logistic_c,
                     temperature=FLAGS.temperature,
                     sparse_bernoulli=FLAGS.sparse_bernoulli,
                     s=FLAGS.s,
                     depth=19,
                     name="VggNet",
                     lr_warmup_val=FLAGS.lr_warmup_val, lr_warmup_steps=FLAGS.lr_warmup_steps)
  elif FLAGS.network == "VggNet16":
    from src.cifar10_channel_prune.Vgg_gumbel import vggnet
    network = vggnet(images, labels,
                     batch_size=FLAGS.batch_size,
                     sparse_threshold=FLAGS.sparse_threshold,
                     logistic_k=FLAGS.logistic_k,
                     logistic_c=FLAGS.logistic_c,
                     temperature=FLAGS.temperature,
                     sparse_bernoulli=FLAGS.sparse_bernoulli,
                     s=FLAGS.s,
                     name="VggNet",
                     depth=16,
                     lr_warmup_val=FLAGS.lr_warmup_val, lr_warmup_steps=FLAGS.lr_warmup_steps)
  elif FLAGS.network == "DenseNet":
    from src.cifar10_channel_prune.DenseNet_gumbel import densenet
    network = densenet(images, labels,
                       batch_size=FLAGS.batch_size,
                       sparse_threshold=FLAGS.sparse_threshold,
                       logistic_k=FLAGS.logistic_k,
                       logistic_c=FLAGS.logistic_c,
                       temperature=FLAGS.temperature,
                       sparse_bernoulli=FLAGS.sparse_bernoulli,
                       s=FLAGS.s,
                       name="DenseNet",
                       lr_warmup_val=FLAGS.lr_warmup_val, lr_warmup_steps=FLAGS.lr_warmup_steps)
  child_ops = {
    "global_step": network.global_step,
    "loss": network.loss,
    "train_op": network.train_op,
    "lr": network.lr,
    "grad_norm": network.grad_norm,
    "train_acc": network.train_acc,
    "num_train_batches": network.num_train_batches,
    "channel_mask": network.channel_mask,
    "base_channels": network.base_channels,
    "base_flops": network.base_flops,
    "base_params": network.base_params,
    "params": network.dynamic_params_,
    "flops": network.dynamic_flops_,
    "channels": network.dynamic_channels_,
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
  global best_flops
  global best_params
  global best_channels
  g = tf.Graph()
  with g.as_default():
    ops = get_ops(images, labels)
    child_ops = ops["child_list"]

    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)
    tensorboard_summary_op = tf.summary.merge_all()
    tensorboard_dir = os.path.join('tensorboard')
    if FLAGS.network == "ResNet":
      tensorboard_dir = os.path.join(tensorboard_dir, "cifar10_ResNet56_prune")
    elif FLAGS.network == "VggNet19":
      tensorboard_dir = os.path.join(tensorboard_dir, "cifar10_VggNet19_prune")
    elif FLAGS.network == "VggNet16":
      tensorboard_dir = os.path.join(tensorboard_dir, "cifar10_VggNet16_prune")
    elif FLAGS.network == "DenseNet":
      tensorboard_dir = os.path.join(tensorboard_dir, "cifar10_DenseNet_prune")
    summary_writer = tf.summary.FileWriter(
        os.path.join(tensorboard_dir,FLAGS.network),graph=g)

    tf.train.start_queue_runners(sess=sess)
    step = 0
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
          if step % FLAGS.log_every == FLAGS.log_every-1:
            run_ops.append(tensorboard_summary_op)
          receive_ops = sess.run(run_ops)
          step += 1
          global_step = sess.run(child_ops["global_step"])

          epoch = global_step // ops["num_train_batches"]
          curr_time = time.time()
          if global_step % FLAGS.log_every == 0:
            loss, lr, gn, tr_acc, _, tensorboard_summary = receive_ops[0:]
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
            summary_writer.add_summary(tensorboard_summary,global_step)


          if global_step % ops["num_train_batches"] == 0:
            print("Epoch {}: Eval".format(epoch))
            test_acc = ops["eval_func"](sess, "test")
            channel_mask, flops, params, channels = sess.run([child_ops["channel_mask"],
                                                    child_ops["flops"],
                                                    child_ops["params"],
                                                    child_ops["channels"]])
            if best_test_acc < test_acc:
              best_channel_mask = channel_mask
              best_test_acc = test_acc
              best_params = params
              best_flops = flops
              best_channels = channels
              checkpoint_path = os.path.join(output_dir, "model.ckpt")
              saver.save(sess, checkpoint_path, global_step=global_step)
            print ('best_test_acc: %.2f'%best_test_acc)
            for mask_key in channel_mask:
              print ('%s: %d'%(mask_key, channel_mask[mask_key]))
            total_channels = child_ops["base_channels"]
            total_params = child_ops["base_params"]
            total_flops = child_ops["base_flops"]
            prune_params_rate = float(total_params-params)/float(total_params)*100.0
            prune_channels_rate = float(total_channels-channels)/float(total_channels)*100.0
            prune_flops_rate = float(total_flops-flops)/float(total_flops)*100.0
            print ('total_alive_params: %d'%params)
            print ('total_alive_flops: %d'%flops)
            print ('total_alive_channel: %d'%channels)
            print ('prune_params_rate: %.2f'%prune_params_rate)
            print ('prune_channels_rate: %.2f'%prune_channels_rate)
            print ('prune_flops_rate: %.2f'%prune_flops_rate)
          if epoch >= FLAGS.num_epochs:
            prune_channel = 0
            for mask_key in best_channel_mask:
              print ('%s: %d'%(mask_key, best_channel_mask[mask_key]))
            print ('best_test_acc: %.2f'%best_test_acc)
            print ('total_channel: %d'%total_channels)
            print ('total_alive_channel: %d'%best_channels)
            print ('total_alive_params: %d'%best_params)
            print ('total_alive_flops: %d'%best_flops)
            print ('prune_params_rate: %.2f'%(100.0*float(total_params-best_params)/(float(total_params))))
            print ('prune_channels_rate: %.2f'%(100.0*float(total_channels-best_channels)/(float(total_channels))))
            print ('prune_flops_rate: %.2f'%(100.0*float(total_flops-best_flops)/(float(total_flops))))
            break

def main(_):
  print("-" * 80)
  global  best_test_acc
  global best_params
  global best_flops
  global best_channels
  global output_dir
  output_dir = FLAGS.network
  output_dir=output_dir + "_sparse_threshold_%.2f"%FLAGS.sparse_threshold
  output_dir=output_dir + "_logistic_k_%.2f"%FLAGS.logistic_k
  output_dir=output_dir + "_logistic_c_%.2f"%FLAGS.logistic_c
  output_dir=output_dir + "_sparse_bernoulli_%.2f"%FLAGS.sparse_bernoulli
  if FLAGS.network == "ResNet":
    output_dir = os.path.join("cifar10_ResNet_gumbel_prune", output_dir)
  elif FLAGS.network == "VggNet16":
    output_dir = os.path.join("cifar10_VggNet16_gumbel_prune", output_dir)
  elif FLAGS.network == "VggNet19":
    output_dir = os.path.join("cifar10_VggNet19_gumbel_prune", output_dir)
  elif FLAGS.network == "DenseNet":
    output_dir = os.path.join("cifar10_DenseNet_gumbel_prune", output_dir)
  else:
    raise ValuedError("Network should be [ResNet, VggNet16, VggNet19, DenseNet]")
  if not os.path.isdir(output_dir):
    print("Path {} does not exist. Creating.".format(output_dir))
    os.makedirs(os.path.join(output_dir))
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
