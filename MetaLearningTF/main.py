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


import tensorflow as tf

import time as time
import numpy as np

from metalearning_tf.data.generators import OmniglotGenerator
from metalearning_tf.model.nn_model import NNModel
import metalearning_tf.utils.summary_writers as sw

# Config options
IM_SIZE = 20
NUM_TRAINING_SEQ = None
SUMMARY_FREQ = 100
SUMMARY_SCOPE = 'mlTF'
TR_LOG_FOLDER = 'tensorboard_log'


def prepare_cat_summary(graph,
                        loss,
                        labels,
                        logits,
                        batch_size,
                        num_classes,
                        num_samples_per_class):
    # Create summary writers for both testing and training
    # Be cautious!!! This step must be done before any summary writer is attached
    # Add the graph to the event file for visualization
    print('Attaching summary writers for classification...')
    if tf.gfile.Exists(TR_LOG_FOLDER):
        tf.gfile.DeleteRecursively(TR_LOG_FOLDER)
    tf.gfile.MakeDirs(TR_LOG_FOLDER)
    writer = tf.summary.FileWriter(TR_LOG_FOLDER, graph)
    # Attach summary writers
    tf.summary.scalar('loss', loss)
    pre = tf.argmax(tf.reshape(logits,
                              [batch_size,
                               num_classes*num_samples_per_class,
                               num_classes]),
                    axis=2)
    tar = tf.reshape(labels,
                     [batch_size, num_classes*num_samples_per_class])
    accu = sw.attach_cat_pred_dist(pre,
                                   tar,
                                   num_classes,
                                   num_samples_per_class,
                                   batch_size,
                                   scope=SUMMARY_SCOPE)
    merged = tf.summary.merge_all()
    print()
    print()

    return writer, accu, merged


def main():

    # Load data as was defined in the original code in Theano
    sample_generator = OmniglotGenerator(data_folder='./omni_samples/',
                                         batch_size=16,  # Each call to the next() operator fetches this many samples
                                         nb_samples=5,  # Actually, it is # of classes
                                         nb_samples_per_class=10,  # # of samples per class
                                         max_rotation=0,
                                         max_shift=0,
                                         max_iter=NUM_TRAINING_SEQ)
    seq_len = sample_generator.nb_samples*sample_generator.nb_samples_per_class
    # Create Neural-Network which is composed of a sequence of controller units
    # and a single external memory unit
    nn = NNModel(batch_size=sample_generator.batch_size,
                 input_size=IM_SIZE*IM_SIZE,
                 num_classes=sample_generator.nb_samples,
                 controller_size=200,
                 memory_size=(128, 40),
                 num_read_heads=4,
                 learning_rate=1e-3,
                 gamma=0.95)
    with tf.Graph().as_default():
        # Build common tensors used throughout entire session
        nn.build(seq_len)
        # Construct the architecture
        labels, model, loss, train_op = nn\
            .generate_model()\
            .loss()\
            .train()\
            .get_all()
        with tf.Session() as sess:
            # Add required summaries for tensorboard
            writer, acc, merged = prepare_cat_summary(sess.graph,
                                                      loss,
                                                      labels,
                                                      model,
                                                      sample_generator.batch_size,
                                                      sample_generator.nb_samples,
                                                      sample_generator.nb_samples_per_class)
            st = time.time()
            try:
                scores, acc_list = [], np.zeros(sample_generator.nb_samples_per_class)
                # !!!!!!!!NEVER EVER EXECUTE the FOLLOWING ASSIGNMENT BEFORE BUILDING YOUR MODEL!!!!!!!
                init = tf.global_variables_initializer()
                # Initialize all variables (Note that! not operation tensors; but variable tensors)
                print('Initializing variables...')
                sess.run(init)
                print('Training the model...')
                # Each time a sample is fetched from omniglot database, following outputs are
                # generated:
                #    input_ : (batch_size)x(nb_samples*nb_samples_per_class)x(input_size)
                #    target : (batch_size)x(nb_samples*nb_samples_per_class)
                for e, (input_, target) in sample_generator:
                    feed_dict = nn.prepare_dict(input_, target)
                    _, score, accu = sess.run([train_op, loss, acc], feed_dict=feed_dict)
                    acc_list += accu
                    scores.append(score)
                    if e > 0 and e % SUMMARY_FREQ == 0:
                        print('*******************************Episode #%05d****************************' % e)
                        summary = sess.run(merged, feed_dict=feed_dict)
                        print('Writing summaries...')
                        writer.add_summary(summary, e)
                        writer.flush()
                        print('Average Accuracy for %d Episodes: %s' % (SUMMARY_FREQ, acc_list / float(SUMMARY_FREQ)))
                        print('Average Score for %d Episodes: %.3f' % (SUMMARY_FREQ, np.mean(scores)))
                        scores, acc_list = [], np.zeros(sample_generator.nb_samples_per_class)
                        print('*****************************************************************************')
            except KeyboardInterrupt:
                pass
            finally:
                writer.close()
                print('Elapsed time: %ld' % (time.time() - st))
        # TODO Save the model for we can restore it in a completely different environment


if __name__ == "__main__":

    main()

