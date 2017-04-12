'''
    Copyright 2017-2022 Department of Electrical and Computer Engineering
    University of Houston, TX/USA
    
    This file is part of UHVision Libraries.
    
    UH Vision libraries are free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    UH Vision Libraries are distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along 
    with licenses for all the 3rd party libraries used in this repository. 
    If not, see <http://www.gnu.org/licenses/>. 
    Please contact Hien Nguyen V for more info about licensing hvnguy35@central.uh.edu, 
    and other members of the UHVision Lab via github issues section.
    **********************************************************************************
    Author:   Ilker GURCAN
    Date:     4/8/17
    File:     alex_net_trainer
    Comments: Multi-GPU training of modified version of AlexNet
    **********************************************************************************
'''


import tensorflow as tf
from tflearn.config import init_training_mode
from tflearn.optimizers import Momentum

import re
import time
import os
from datetime import datetime

from utils import constants as consts
from model.modified_alex_net import ModifiedAlexNet
from data.parallel_data_feeder import ParallelDataFeeder

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_dir", './alexnet_train',
                           """"Directory in where logs and checkpoints are stored""")
tf.app.flags.DEFINE_integer("num_gpus", 1,
                            """"Number of GPUs to be used for training""")
tf.app.flags.DEFINE_boolean("log_device_placement", True,
                            """"Whether to log device placement""")
tf.app.flags.DEFINE_integer("num_epochs", 100,
                            """"How many times the whole training set has to be fed into network""")
tf.app.flags.DEFINE_integer("num_ex_per_epoch", 1024,
                            """"Number of samples in an epoch""")
tf.app.flags.DEFINE_integer("num_threads", 2,
                            """"Number of threads that will enqueue training samples from the sample queue""")
tf.app.flags.DEFINE_string("tf_record_file_name",
                           '/home/cougarnet.uh.edu/igurcan/Documents' 
                           '/phdStudies/UHVision/SelfDrivingCar/DriveXbox1'
                           '/gtav_training.tfrecords',
                           """"Native TF file where training samples are stored""")
tf.app.flags.DEFINE_float("moving_average_decay", 0.9999,
                          """"Decay rate for the past records in exponential moving average""")
tf.app.flags.DEFINE_integer("batch_size", 2,
                            """Number of samples in a batch""")
tf.app.flags.DEFINE_float("min_frac_ex_in_queue", 0.4,
                          """"Fraction of samples in a given epoch to be kept in queue for a nice shuffling""")

IM_W = 800
IM_H = 600
IM_D = 3


def tower_ops(data_feeder):
    '''
    Called per GPU. Creates a replica of AlexNet on the
     specified GPU.
    :param data_feeder: 
    :return: 
    '''

    # Fetch a batch of images and labels from sample source
    images, labels = data_feeder.inputs(FLAGS.batch_size,
                                        FLAGS.num_ex_per_epoch,
                                        FLAGS.min_frac_ex_in_queue,
                                        True)
    # Build the network and its loss functions
    # Each op name of any loss function starts with 'tower_i/*'
    alex_net = ModifiedAlexNet(batch_size=FLAGS.batch_size,
                               frame_size=(IM_H, IM_W),
                               num_channels=IM_D)
    alex_net.inference(False).loss().total_loss()
    # Get the actual and total losses
    act_loss, total_loss, _ = alex_net.get_all()

    # Add summary operations for each dimension of loss functions
    for l in [act_loss, total_loss]:
        loss_name = re.sub('%s[0-9]*/' % consts.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name+'_'+consts.STEERING_ANGLE, l[0])  # STEERING
        tf.summary.scalar(loss_name + '_' + consts.THROTTLE, l[1])  # THROTTLE
        tf.summary.scalar(loss_name + '_' + consts.BRAKE, l[2])  # BRAKE

    return total_loss


def average_gradients(tower_grads):
    '''
    
    :param tower_grads: 
    :return: 
    '''
    if FLAGS.num_gpus < 2:
        return tower_grads[0]
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        # ((grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add a leading dimension to the gradients to represent the tower
            expanded_grad = tf.expand_dims(g, 0)
            # Append on a 'GPU (tower)' dimension which will average over
            grads.append(expanded_grad)
        # Average over the 'tower' dimension
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        var = grad_and_vars[0][1]
        grad_and_var = (grad, var)
        average_grads.append(grad_and_var)
    return average_grads


def prepare_fields():

    types = [  # Order matters
        tf.string,
        tf.float32,
        tf.float32,
        tf.float32
    ]
    names = [  # Order matters
        consts.IMAGE_RAW,
        consts.STEERING_ANGLE,
        consts.THROTTLE,
        consts.BRAKE
    ]
    return zip(names, types)


def train():

    with tf.Graph().as_default(), tf.device(consts.CPU_NAME+'%d' % 0):
        # Initializations
        data_feeder = ParallelDataFeeder(FLAGS.tf_record_file_name,
                                         FLAGS.num_threads,
                                         FLAGS.num_epochs,
                                         prepare_fields(),
                                         [IM_H, IM_W, IM_D])
        init_training_mode()  # Necessary if you don't use Trainer from tflearn
        global_step = tf.get_variable(consts.GLOBAL_STEP,
                                      [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        tower_grads = []
        losses = []
        # A global optimizer is necessary for we compute gradients across several GPUs.
        # Although its compute_gradients is called on every GPU; reduction of these gradients
        # and applying them to update parameters are run on CPU
        opt = Momentum(learning_rate=0.001, momentum=0.9)
        opt = opt.get_tensor()
        # Building network replicas and their corresponding losses on specified # of GPUs
        with tf.variable_scope(tf.get_variable_scope()) as var_scope:
            for i in range(FLAGS.num_gpus):
                with tf.device(consts.GPU_NAME+'%d' % i):
                    with tf.name_scope(consts.TOWER_NAME+'%d' % i) as scope:
                        total_loss = tower_ops(data_feeder)
                        losses.append(total_loss)
                        # Share all parameters across all GPUs
                        var_scope.reuse_variables()
                        # Retain summaries from only the final tower
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        # Add operations to compute gradients on the current GPU
                        grads = opt.compute_gradients(total_loss)
                        # Keep track of the gradients across all GPUs
                        tower_grads.append(grads)
        # Synchronization point of all towers (GPUs).
        # Calculate the mean of each gradient
        grads = average_gradients(tower_grads)
        # Apply gradients to adjust the shared variables (weights & biases)
        apply_grad_op = opt.apply_gradients(grads, global_step=global_step)
        # Track moving averages of all variables. Evaluations that use
        # averaged parameters sometimes produce significantly better results
        # than the final trained values.
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        # Create the shadow variables and add ops to maintain moving averages
        apply_ema_op = ema.apply(tf.trainable_variables())
        # Group all updates into a single train op
        train_op = tf.group(apply_grad_op, apply_ema_op)
        # Create a saver for the trained model
        saver = tf.train.Saver(tf.global_variables())
        # Build the summary operation from the last tower summaries
        summary_op = tf.summary.merge(summaries)
        # Create a single variable initialization op
        # Finish creating all variables and building all tensors
        # before this point is reached!
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = FLAGS.log_device_placement
        with tf.Session(config=config) as sess:
            sess.run(init)
            # Start input enqueue threads for reading input data using 'parallel data feeder' mechanism
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            step = 1
            try:
                while not coord.should_stop():
                    start_time = time.time()
                    _, loss_values = sess.run([train_op]+losses)
                    duration = time.time()-start_time
                    if step % 10 == 0:
                        num_ex_per_step = FLAGS.batch_size*FLAGS.num_gpus
                        ex_per_sec = num_ex_per_step / duration
                        sec_per_batch = duration / FLAGS.num_gpus
                        format_str = ('%s: step %d (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(),
                                            step,
                                            ex_per_sec,
                                            sec_per_batch))
                    if step % 100 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                    # Save a checkpoint at the end of every epoch
                    if step % FLAGS.num_ex_per_epoch == 0:
                        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs %d steps.' % (FLAGS.num_epochs, step))
            finally:
                # When done, ask all threads to stop
                coord.request_stop()
            # Wait for threads to finish
            coord.join(threads)


def main(argv=None):

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
