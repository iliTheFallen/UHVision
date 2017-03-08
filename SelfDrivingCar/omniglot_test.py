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

from metalearning_tf.data.generators import OmniglotGenerator

import model.alex_net as alex


def main():

    # Load data as was defined in the original code in Theano
    # As in Alex's paper; data augmentation is provided via shifting/rotation
    sample_generator = OmniglotGenerator(data_folder='./omni_samples/',
                                         batch_size=16,  # Each call to the next() operator fetches this many samples
                                         nb_samples=5,  # Actually, it is # of classes
                                         nb_samples_per_class=10,  # # of samples per class
                                         max_rotation=0,
                                         max_shift=0,
                                         max_iter=None)
    st = time.time()
    try:
        # Build the network
        alex.build_alex_net([20, 20], sample_generator.nb_samples)
        # Extract training data

        # Train the network

    except KeyboardInterrupt:
        print('Elapsed Time: %ld' %(time.time()-st))


if __name__ == "__main__":
    main()