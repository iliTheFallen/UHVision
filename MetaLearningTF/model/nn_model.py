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
    Date:     2/20/17
    File:     nn_model
    Comments: This file contains the architecture for NTM which is based
     on the memory-access model defined in the paper called
     "Meta-Learning with Memory-Augmented Neural Networks".
    **********************************************************************************
'''

import model.controller_unit
import model.memory


class NNModel(object):

    __controller_size = 0  # How
    __num_classes = 0  # Number of classes/labels
    __memory_size = (0, 0)  # 2D Tuple specifying (numberOfElements)X(sizeOfEachElement)
    __num_read_heads = 0  # Number of read heads attached to controller

    def __init__(self,
                 controller_size,
                 num_classes,
                 memory_size,
                 num_read_heads):
        self.__controller_size = controller_size
        self.__num_classes = num_classes
        self.__memory_size = memory_size
        self.__num_read_heads = num_read_heads

