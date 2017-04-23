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
    Date:     4/18/17
    File:     mann_cnn_trainer
    Comments: 
    **********************************************************************************
'''



import tensorflow as tf
from tflearn.optimizers import Momentum

import os
import re

from model.mann_cnn_model import MannCnnModel
from utils.parallel_net_runner import ConfigOptions
from utils.parallel_net_runner import ParallelNetRunner
from utils import constants as consts
from data.tf_record_feeder import TFRecordFeeder

FLAGS = tf.app.flags.FLAGS

# Configuration options for parallel network runner
tf.app.flags.DEFINE_string(ConfigOptions.TRAIN_DIR.value, './alexnet_train',
                           """"Directory in where logs and checkpoints are stored""")
tf.app.flags.DEFINE_integer(ConfigOptions.NUM_GPUS.value, 1,
                            """"Number of GPUs to be used for training""")
tf.app.flags.DEFINE_integer(ConfigOptions.NUM_EX_PER_EPOCH.value, 1024,
                            """"Number of samples in an epoch""")
tf.app.flags.DEFINE_float(ConfigOptions.MOVING_AVERAGE_DECAY.value, 0.9999,
                          """"Decay rate for the past records in exponential moving average""")
tf.app.flags.DEFINE_integer(ConfigOptions.BATCH_SIZE.value, 1,
                            """Number of samples in a batch""")
tf.app.flags.DEFINE_float(ConfigOptions.MIN_FRAC_EX_IN_QUEUE.value, 0.4,
                          """"Fraction of samples in a given epoch to be kept in queue for a nice shuffling""")
tf.app.flags.DEFINE_boolean(ConfigOptions.SHOULD_SHUFFLE.value, True,
                            """"Whether to shuffle samples.""")
# Configuration options for data feeder
tf.app.flags.DEFINE_integer("num_epochs", 100,
                            """"How many times the whole training set has to be fed into network""")
tf.app.flags.DEFINE_integer("num_threads", 1,
                            """"Number of threads that will enqueue training samples from the sample queue""")
tf.app.flags.DEFINE_string("tf_record_file_name",
                           '/home/ilithefallen/Documents/phdThesis'
                           '/UHVision/SelfDrivingCar/DriveXbox1'
                           '/gtav_training.tfrecords',
                           """"Native TF file where training samples are stored""")
# Configuration options for the session
tf.app.flags.DEFINE_boolean("log_device_placement", False,
                            """"Whether to log device placement""")

IM_W = 400
IM_H = 300
IM_D = 3


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


def _step(runner, summary_writer, summary_op):

    step = runner.step
    if step % (ConfigOptions.NUM_EX_PER_EPOCH.get_val()/
                   ConfigOptions.BATCH_SIZE.get_val()) == 0:
        print('Saving Summary & Checkpoint...')
        _, loss_values, summary_str = runner.sess.run([runner.train_op,
                                                       runner.loss,
                                                       summary_op])
        summary_writer.add_summary(summary_str, step)
        checkpoint_path = os.path.join(ConfigOptions.TRAIN_DIR.get_val(),
                                       'model.ckpt')
        runner.saver.save(runner.sess,
                          checkpoint_path,
                          global_step=step)
        print('Finished...')
    else:
        _, loss_values = runner.sess.run([runner.train_op, runner.loss])
    if step % 4 == 0:
        loss_format_str = ('step %d: '
                           '%s_loss: %.5f / '
                           '%s_loss: %.5f / '
                           '%s_loss: %.5f')
        print(loss_format_str % (step,
                                 consts.STEERING_ANGLE, loss_values[0],
                                 consts.THROTTLE, loss_values[1],
                                 consts.BRAKE, loss_values[2]))


def attach_summary_writers(runner):

    gpu_loss = runner.loss
    loss_name = re.sub('%s[0-9]*/' % consts.TOWER_NAME, '', gpu_loss.op.name)
    tf.summary.scalar(loss_name+'_'+consts.STEERING_ANGLE, gpu_loss[0])  # STEERING
    tf.summary.scalar(loss_name + '_' + consts.THROTTLE, gpu_loss[1])  # THROTTLE
    tf.summary.scalar(loss_name + '_' + consts.BRAKE, gpu_loss[2])  # BRAKE
    return tf.summary.merge_all()


def train():

    graph = tf.Graph()
    with graph.as_default():
        # All operations should be built into the same graph
        # Create data feeder
        # Data feeder has operations pushed into the graph
        data_feeder = TFRecordFeeder(FLAGS.num_threads,
                                     FLAGS.tf_record_file_name,
                                     FLAGS.num_epochs,
                                     prepare_fields(),
                                     [IM_H, IM_W, IM_D])
        # Specify session configuration options
        sess_config = tf.ConfigProto()
        sess_config.log_device_placement = FLAGS.log_device_placement
        # Network class' arguments (ModifiedAlexNet)
        net_kwargs = {
            'batch_size': ConfigOptions.BATCH_SIZE.get_val(),
            'seq_len' : 80,
            'num_channels': IM_D,
            'frame_size': (IM_H, IM_W),
            'num_classes': 3,  # Steering Angle, Throttle, and Brake
            'percentile': 0.3  # LSD will be applied to all losses
        }
        # Create an optimizer
        opt = Momentum(learning_rate=0.001, momentum=0.9)
        opt = opt.get_tensor()
        # Create Parallel Runner
        # If not specified, Parallel Runner creates
        # a graph and a session object when its 'build' method is called.
        # Use them if you need to. You may have access to these objects
        # through its properties list. Otherwise; specify graph and
        # session objects using its setter functions right after you
        # create an instance of it.
        runner = ParallelNetRunner(
            MannCnnModel,
            net_kwargs,
            data_feeder,
            opt,
            True,
            True
        )
        runner.config_proto = sess_config  # Specify custom session config options
        runner.graph = graph
        # Build the network.
        # Right after this call, graph and session objects are created/set
        runner.build()
        # Add summaries
        summary_writer = tf.summary.FileWriter(ConfigOptions.TRAIN_DIR.get_val(),
                                               runner.graph)
        # Don't add summary writers until you build the network.
        # All tensors that define graph operations are created after
        # 'build' method is called.
        summaries = attach_summary_writers(runner)
        # Start execution
        step_args = (summary_writer, summaries)  # Extra args passed to _step function
        runner.run(_step, step_args)


def main(argv=None):

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
