'''
    Copyright 2017-2022 Department of Electrical and Computer Engineering
    University of Houston, TX/USA
    
    This file is part of UHVision Libraries.
    
    UH Vision libraries are free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    UH Vison Libraries are distributed in the hope that it will be useful,
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
    Date:     3/7/17
    File:     main
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf
from tflearn.helpers.evaluator import Evaluator
from tflearn.metrics import Accuracy

import time as time
import numpy as np

from metalearning_tf.data.generators import OmniglotGenerator
import model.alex_net as alex

IM_W = 20
IM_H = 20
NUM_CLASSES = 5
NUM_SAMPLES_PER_CLASS = 10
SEQ_LENGTH = NUM_CLASSES * NUM_SAMPLES_PER_CLASS
BATCH_SIZE = 16
NUM_CHANNELS = 1


def preproc_data(input_, target):
    '''

    :param input_: BSxSLx(IMW*IMH)
    :param target: BSxSLx1
    :return:
    '''

    new_input_ = input_.reshape(BATCH_SIZE*SEQ_LENGTH, IM_H, IM_W, NUM_CHANNELS)
    new_target = np.zeros([BATCH_SIZE*SEQ_LENGTH, NUM_CLASSES], dtype=np.float32)
    new_target[np.arange(BATCH_SIZE*SEQ_LENGTH), target.reshape(-1)] = 1.0

    return new_input_, new_target


def create_placeholders():

    input_ph = tf.placeholder(tf.float32,
                                     (BATCH_SIZE * SEQ_LENGTH, IM_W, IM_H, NUM_CHANNELS),
                                     "input_ph")
    target_ph = tf.placeholder(tf.float32,
                               (BATCH_SIZE * SEQ_LENGTH, NUM_CLASSES),
                               "target_ph")
    return input_ph, target_ph


def main():

    # Load data as was defined in the original code in Theano
    # As in Alex's paper; data augmentation is provided via shifting/rotation
    sample_generator = OmniglotGenerator(data_folder='../MetaLearningTF/omni_samples/',
                                         batch_size=BATCH_SIZE,  # Each call to the next() operator fetches this many samples
                                         nb_samples=NUM_CLASSES,  # Actually, it is # of classes
                                         nb_samples_per_class=NUM_SAMPLES_PER_CLASS,  # # of samples per class
                                         max_rotation=0,
                                         max_shift=0,
                                         max_iter=None)
    # Use the first sample set as test data
    e, (test_input, test_target) = sample_generator.next()
    test_input, test_target = preproc_data(test_input, test_target)
    # Build the network
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        print('Building AlexNet...')
        acc_op = Accuracy(name="accuracy_alex_net")
        input_ph, target_ph = create_placeholders()
        alex_net = alex.build_alex_net([BATCH_SIZE*SEQ_LENGTH, IM_W, IM_H, NUM_CHANNELS],
                                       input_ph,
                                       sample_generator.nb_samples)
        loss_fn, train_op = alex.train(alex_net, target_ph)
        acc_op.build(alex_net, target_ph, input_ph)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            net_eval = Evaluator(alex_net, session=sess)
            print('Initializing variables...')
            sess.run(init)
            st = time.time()
            losses = []
            try:
                print('Training...')
                for e, (input_, target) in sample_generator:
                    # Testing...
                    if (e-1) > 0 and ((e-1) % 100 == 0):
                        print('Evaluating the model...')
                        feed_dict = {input_ph: test_input,
                                     target_ph: test_target}
                        acc = net_eval.evaluate(feed_dict,
                                                acc_op.get_tensor(),
                                                batch_size=BATCH_SIZE*SEQ_LENGTH)
                        print('Accuracy for episode %05d: %.6f' % (e, acc[0]))
                        print('Loss for %d episodes: %.6f' % (len(losses), np.mean(np.asarray(losses))))
                        print('Training...')
                        losses = []
                    # Either train or continue to train...
                    new_input_, new_target = preproc_data(input_, target)
                    feed_dict = {input_ph: new_input_,
                                 target_ph: new_target}
                    _, loss = sess.run([train_op, loss_fn], feed_dict=feed_dict)
                    losses.append(loss)
            except KeyboardInterrupt:
                print('Elapsed Time: %ld' %(time.time()-st))
                pass


if __name__ == "__main__":
    main()
