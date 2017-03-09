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

import time as time
import numpy as np

from metalearning_tf.data.generators import OmniglotGenerator

import model.alex_net as alex

IM_W = 20
IM_H = 20


def preprocess_data(omniglot_generator):

    X = None
    y = None
    print('Preprocessing input data...')
    for e, (input_, target) in omniglot_generator:
        print('Current episode: %d' % e)
        # There is no sequence notion in this context. Our input_ will be of size:
        # (BS.SL)x(WxH). A 3D cube whose each slight is an input_
        in_shape = input_.shape
        input_ = input_.reshape(in_shape[0]*in_shape[1], in_shape[2])
        input_ = input_.reshape(in_shape[0]*in_shape[1], IM_W, IM_H, 1)
        tar_shape = target.shape
        target = target.reshape(tar_shape[0]*tar_shape[1], 1)
        # Stack them up vertically
        if X is None:
            X = input_
            y = target
        else:
            X = np.concatenate((X, input_), axis=0)
            y = np.concatenate((y, target), axis=0)
    return X, y


def main():

    # Load data as was defined in the original code in Theano
    # As in Alex's paper; data augmentation is provided via shifting/rotation
    sample_generator = OmniglotGenerator(data_folder='../MetaLearningTF/omni_samples/',
                                         batch_size=16,  # Each call to the next() operator fetches this many samples
                                         nb_samples=5,  # Actually, it is # of classes
                                         nb_samples_per_class=10,  # # of samples per class
                                         max_rotation=0,
                                         max_shift=0,
                                         max_iter=10)
    st = time.time()
    try:
        # Build the network
        network = alex.build_alex_net([IM_W, IM_H, 1], sample_generator.nb_samples)
        # Extract training data
        input_, labels = preprocess_data(sample_generator)
        # Train the network
        model = alex.train(network, input_, labels, 16, 'alexnet_model')
    except KeyboardInterrupt:
        print('Elapsed Time: %ld' %(time.time()-st))
        pass


if __name__ == "__main__":
    main()