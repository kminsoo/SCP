from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pickle
import shutil
import sys
import time

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
DEFINE_string("output_dir", "", "")
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCWH'")
DEFINE_string("network", "VggNet19", "'VggNet19' 'VggNet16' or 'DenseNet'" )
DEFINE_float("lr_warmup_val",None, "")
DEFINE_float("sparse_threshold",None, "")
DEFINE_float("cdf_threshold",None, "")
DEFINE_float("scale_init",0.5, "")
DEFINE_integer("lr_warmup_steps",0, "")
DEFINE_integer("batch_size",64, "")
DEFINE_integer("num_epochs", 160, "")
DEFINE_integer("log_every", 100, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

def get_ops(images, labels):
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """
  if FLAGS.network == "VggNet19":
    from src.cifar10_channel_prune.Vgg_slimming import vggnet
    network = vggnet(images, labels, slimming_weight=1e-4, depth=19, scale_init=FLAGS.scale_init,
                     batch_size=FLAGS.batch_size)
  elif FLAGS.network == "VggNet16":
    from src.cifar10_channel_prune.Vgg_slimming import vggnet
    network = vggnet(images, labels, slimming_weight=1e-4, depth=16, scale_init=FLAGS.scale_init,
                     batch_size=FLAGS.batch_size)
  elif FLAGS.network == "DenseNet":
    from src.cifar10_channel_prune.DenseNet_slimming import densenet
    network = densenet(images, labels, slimming_weight=1e-5, scale_init=FLAGS.scale_init,
                       batch_size=FLAGS.batch_size)
  else:
    raise ValuedError("Network should be [VggNet16, VggNet19, DenseNet]")

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
    tf.train.start_queue_runners(sess=sess)
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
              checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt")
              saver.save(sess, checkpoint_path, global_step=global_step)
            print ('best_test_acc: %.2f'%best_test_acc)
          if epoch >= FLAGS.num_epochs:
            break

def main(_):
  print("-" * 80)
  global  best_test_acc
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)
  train()
  utils.print_user_flags()

if __name__ == "__main__":
  best_test_acc = 0.0
  tf.app.run()
