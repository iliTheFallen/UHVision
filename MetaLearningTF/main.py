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
    Date:     2/7/17
    File:     main
    Comments: Main entry point for the application. This is also where we parse user
      specified input on the command line.
    **********************************************************************************
'''

import time as time
import numpy as np
import tensorflow as tf

from model.nn_model import NNModel
from data.generators import OmniglotGenerator


def main():

    # Load data as was defined in the original code in Theano
    sample_generator = OmniglotGenerator(data_folder='./omni_samples/',
                                         batch_size=16,  # Each call to the next() operator fetches this many samples
                                         nb_samples=5,  # Actually, it is # of classes
                                         nb_samples_per_class=10,  # # of samples per class
                                         max_rotation=0,
                                         max_shift=0,
                                         max_iter=None)
    seq_len = sample_generator.nb_samples*sample_generator.nb_samples_per_class
    # Create Neural-Network which is composed of a sequence of controller units
    # and a single external memory unit
    nn = NNModel(batch_size=sample_generator.batch_size,
                 input_size=20*20,
                 num_classes=sample_generator.nb_samples,
                 controller_size=200,
                 memory_size=(128, 40),
                 num_read_heads=4,
                 learning_rate=0.001,
                 gamma=0.95)
    # Each time a sample is fetched from omniglot database, following outputs are
    # generated:
    #    input_ : (batch_size)x(nb_samples*nb_samples_per_class)x(input_size)
    #    target : (batch_size)x(nb_samples*nb_samples_per_class)
    with tf.Graph().as_default():
        # Store all scores (each score is a loss-per-episode)
        all_scores, scores = [], []
        # Build common tensors used throughout entire session
        nn.build(seq_len)
        # Generate inference and loss models
        (loss, train_op) = nn.generate_models()
        with tf.Session() as sess:
            st = time.time()
            # !!!!!!!!NEVER EVER EXECUTE the FOLLOWING ASSIGNMENT BEFORE BUILDING YOUR MODEL!!!!!!!
            init = tf.global_variables_initializer()
            try:
                # Initialize all variables (Note that! not operation tensors; but variable tensors)
                print('Initializing variables...')
                sess.run(init)
                print('Training starts...')
                for e, (input_, target) in sample_generator:
                    feed_dict = nn.prepare_dict(input_, target)
                    # Run one step of the model.  The return values are the activations
                    # from the `train_op` (which is discarded) and the `loss` Op.
                    _, score = sess.run([train_op, loss],
                                        feed_dict=feed_dict)
                    print("current score ", score)
                    all_scores.append(score)
                    scores.append(score)
                    # Asses your predictions against target
                    if e > 0 and not (e%100):
                        print('Episode %05d: %.6f' % (e, np.mean(scores).tolist()[0]))
                        scores.clear()
            except KeyboardInterrupt:
                print('Elapsed time: %ld' % (time.time()-st))
                pass

        # TODO Save the model for we can restore it in a completely different environment


if __name__ == "__main__":
    main()
