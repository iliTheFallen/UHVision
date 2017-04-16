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
    Date:     4/12/17
    File:     parallel_net_runner
    Comments: Runs training or testing procedure on the specified model.
    **********************************************************************************
'''

import tensorflow as tf
from tflearn.config import init_training_mode

from enum import Enum, unique

from utils import constants as consts

FLAGS = tf.flags.FLAGS  # Should already be set in the application


@unique
class ConfigOptions(Enum):
    # Common options
    NUM_GPUS = 'num_gpus'
    NUM_EX_PER_EPOCH = 'num_ex_per_epoch'
    SHOULD_SHUFFLE = 'should_shuffle'
    MIN_FRAC_EX_IN_QUEUE = 'min_frac_ex_in_queue'
    BATCH_SIZE = 'batch_size'
    MOVING_AVERAGE_DECAY = 'moving_average_decay'
    TRAIN_DIR = 'train_dir'  # Only for training
    CHECK_POINT_DIR = 'check_point_dir'  # Only for testing

    def get_val(self):
        return getattr(FLAGS, self.value)


class ParallelNetRunner(object):

    # Parameters
    __network_class = None
    __net_kwargs = None
    __optimizer = None
    __data_feeder = None
    __is_for_training = None
    __is_using_tflearn = None

    # Attributes
    __config_proto = None
    __graph = None
    __sess = None
    __step = None
    __train_op = None
    __test_op = None
    __loss = None
    __saver = None

    def __init__(self,
                 network_class,
                 net_kwargs,
                 data_feeder,
                 optimizer=None,
                 is_for_training=True,
                 is_using_tflearn=True):
        '''
        
        :param network_class: Model class to construct the model
        :param net_kwargs: Arguments to construct the model
        :param data_feeder: Multi-Threaded data feeder
        :param optimizer: One of optimizers defined in tensorflow. 
         Should be the subclass of tf.train.Optimizer
        :param is_for_training: Whether this runner is constructed for 
         training or testing
        :param is_using_tflearn: Is tflearn library used during construction 
         of the model?
        '''
        self.__network_class = network_class
        self.__net_kwargs = net_kwargs
        self.__optimizer = optimizer
        self.__data_feeder = data_feeder
        self.__is_for_training = is_for_training
        self.__is_using_tflearn = is_using_tflearn

        # Check whether the arguments and config options are in fine shape
        self._sanity_check()

    def _sanity_check(self):

        if not hasattr(self.__network_class, 'loss') \
                or not hasattr(self.__network_class, 'inference') \
                or not hasattr(self.__network_class, '__init__') \
                or not hasattr(self.__network_class, 'loss_func') \
                or not hasattr(self.__network_class, 'prepare_dict'):
            raise ValueError('Not a valid network object!')

        if not hasattr(self.__data_feeder, 'inputs'):
            raise ValueError('Not a valid ParallelDataFeeder object!')

        if self.__is_for_training:
            if not isinstance(self.__optimizer, tf.train.Optimizer):
                raise ValueError('Not a valid optimizer object!')
            if not hasattr(FLAGS, ConfigOptions.TRAIN_DIR.value):
                raise ValueError('Training mode is enabled; '
                                 'but training directory is not specified!')

    def _tower_ops(self):
        '''
        Called per GPU. Creates a replica of AlexNet on the
         specified GPU.
        :param data_feeder: 
        :return: 
        '''
        with tf.device(consts.CPU_NAME+"0"):
            # Fetch a batch of images and labels from sample source
            images, labels = self.__data_feeder.inputs(ConfigOptions.BATCH_SIZE.get_val(),
                                                       ConfigOptions.NUM_EX_PER_EPOCH.get_val(),
                                                       ConfigOptions.MIN_FRAC_EX_IN_QUEUE.get_val(),
                                                       ConfigOptions.SHOULD_SHUFFLE.get_val())
        # Build the network and its loss functions
        # Each op name of any loss function starts with 'tower_i/*'
        network = self.__network_class(images=images,
                                       labels=labels,
                                       **self.__net_kwargs)
        network.inference().loss_func()
        return network.loss

    @staticmethod
    def _average_gradients(tower_grads):
        '''

        :param tower_grads: 
        :return: 
        '''
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

    def _build_for_training(self):

        with self.__graph.as_default(), tf.device(consts.CPU_NAME+"0"):
            # Necessary if you don't use Trainer from tflearn
            if self.__is_using_tflearn:
                init_training_mode()
            global_step = tf.get_variable(consts.GLOBAL_STEP,
                                          [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)
            tower_grads = []
            loss = None
            # Building network replicas and their corresponding losses on specified # of GPUs
            with tf.variable_scope(tf.get_variable_scope()) as var_scope:
                for i in range(FLAGS.num_gpus):
                    with tf.device(consts.GPU_NAME + '%d' % i):
                        with tf.name_scope(consts.TOWER_NAME + '%d' % i) as scope:
                            print('Building network replica for GPU:%d...' % i)
                            loss = self._tower_ops()
                            # Share all parameters across all GPUs
                            var_scope.reuse_variables()
                            # Add operations to compute gradients on the current GPU
                            grads = self.__optimizer.compute_gradients(loss)
                            # Keep track of the gradients across all GPUs
                            tower_grads.append(grads)
            # Synchronization point of all towers (GPUs).
            # Calculate the mean of each gradient
            grads = ParallelNetRunner._average_gradients(tower_grads)
            # Apply gradients to adjust the shared variables (weights & biases)
            apply_grad_op = self.__optimizer.apply_gradients(grads, global_step=global_step)
            if hasattr(FLAGS, ConfigOptions.MOVING_AVERAGE_DECAY.value):
                # Track moving averages of all variables. Evaluations that use
                # averaged parameters sometimes produce significantly better results
                # than the final trained values.
                ema = tf.train.ExponentialMovingAverage(ConfigOptions.MOVING_AVERAGE_DECAY.get_val(),
                                                        global_step)
                # Create the shadow variables and add ops to maintain moving averages
                apply_ema_op = ema.apply(tf.trainable_variables())
                # Group all updates into a single train op
                self.__train_op = tf.group(apply_grad_op, apply_ema_op)
            else:
                self.__train_op = apply_grad_op
            self.__loss = loss
            # Create a saver for the trained model
            self.__saver = tf.train.Saver(tf.global_variables())

    def _build_for_testing(self):

        with self.__graph.as_default():
            images, _ = self.__data_feeder.inputs(ConfigOptions.BATCH_SIZE.get_val(),
                                                  ConfigOptions.NUM_EX_PER_EPOCH.get_val(),
                                                  ConfigOptions.MIN_FRAC_EX_IN_QUEUE.get_val(),
                                                  ConfigOptions.SHOULD_SHUFFLE.get_val())
            print('Building network on GPU:%d...' % 0)
            network = self.__network_class(images=images, **self.__net_kwargs)
            # Restore the moving average version of the learned variables for evaluation
            if hasattr(FLAGS, ConfigOptions.MOVING_AVERAGE_DECAY.value):
                variable_averages = tf.train.ExponentialMovingAverage(ConfigOptions.MOVING_AVERAGE_DECAY.get_val())
                variables_to_restore = variable_averages.variables_to_restore()
            else:
                variables_to_restore = tf.global_variables()
            saver = tf.train.Saver(variables_to_restore)
            with self.__sess:
                ckpt = tf.train.get_checkpoint_state(ConfigOptions.CHECK_POINT_DIR.get_val())
                if ckpt and ckpt.model_checkpoint_path:
                    # Restore from checkpoint file
                    saver.restore(self.__sess, ckpt.model_checkpoint_path)
                else:
                    raise ValueError('No checkpoint file is found!')
        self.__test_op = network
        self.__saver = saver

    def build(self):

        if not self.__graph:
            self.__graph = tf.Graph()
        if not self.__sess:
            if not self.__config_proto:
                self.__config_proto = tf.ConfigProto()
            self.__config_proto.gpu_options.allow_growth = True
            self.__config_proto.allow_soft_placement = True
            self.__sess = tf.Session(graph=self.__graph,
                                     config=self.__config_proto)
        if self.__is_for_training:
            self._build_for_training()
        else:
            if not hasattr(FLAGS, ConfigOptions.CHECK_POINT_DIR.value):
                raise ValueError('Folder where checkpoints are stored should be specified'
                                 'in the case of testing!')
            self._build_for_testing()

    def run(self, step_func, step_args):

        with self.__graph.as_default():
            with self.__sess:
                if self.__is_for_training:
                    # Create a single variable initialization op
                    # Finish creating all variables and building all tensors
                    # before this point is reached!
                    init = tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer())
                    self.__sess.run(init)
                # Create Coordinator for managing Queue Threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=self.__sess,
                                                       coord=coord)
                self.__step = 1
                try:
                    while not coord.should_stop():
                        # Run one step of training/testing process
                        if hasattr(self.__data_feeder, 'before_step'):
                            self.__data_feeder.before_step(self.__sess)
                        step_func(self, *step_args)
                        if hasattr(self.__data_feeder, 'after_step'):
                            self.__data_feeder.after_step(self.__sess)
                        self.__step += 1
                except tf.errors.OutOfRangeError as ex1:
                    print('Run out of Samples...')
                    coord.request_stop(ex1)
                except tf.errors.CancelledError as ex2:
                    print('Operation cancelled or Session is ended...')
                    coord.request_stop(ex2)
                except KeyboardInterrupt as ex3:
                    print('Interrupt signal is sent...')
                    coord.request_stop(ex3)
                finally:
                    coord.join(threads)

    # Common properties
    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph):
        self.__graph = graph

    @property
    def sess(self):
        return self.__sess

    @sess.setter
    def sess(self, sess):
        self.__sess = sess

    @property
    def config_proto(self):
        return self.__config_proto

    @config_proto.setter
    def config_proto(self, config_proto):
        self.__config_proto = config_proto

    @property
    def saver(self):
        return self.__saver

    @property
    def network_class(self):
        return self.__network_class

    @property
    def is_for_training(self):
        return self.__is_for_training

    @property
    def is_using_tflearn(self):
        return self.__is_using_tflearn

    # Properties valid for only training
    @property
    def step(self):
        return self.__step

    @property
    def train_op(self):
        return self.__train_op

    @property
    def loss(self):
        return self.__loss

    @property
    def optimizer(self):
        return self.__optimizer
