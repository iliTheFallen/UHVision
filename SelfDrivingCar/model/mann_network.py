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
    File:     mann_network
    Comments: 
    **********************************************************************************
'''


from metalearning_tf.model.nn_model import NNModel


class MANNNetwork(NNModel):

    __new_input_ph = None
    __new_target_ph = None

    def __init__(self,
                 input_ph=None,
                 target_ph=None,
                 batch_size=1,
                 input_size=800*600,
                 num_classes=3,
                 controller_size=200,
                 memory_size=(128, 40),
                 num_read_heads=4,
                 learning_rate=0.001,
                 gamma=0.95):

        self.__new_input_ph = input_ph
        self.__new_target_ph = target_ph

        super().__init__(batch_size=batch_size,
                         input_size=input_size,
                         num_classes=num_classes,
                         controller_size=controller_size,
                         memory_size=memory_size,
                         num_read_heads=num_read_heads,
                         learning_rate=learning_rate,
                         gamma=gamma)

    def _create_placeholders(self, seq_len):

        return self.__new_input_ph, self.__new_target_ph







