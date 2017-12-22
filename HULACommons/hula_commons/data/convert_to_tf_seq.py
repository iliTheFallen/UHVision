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
    Date:     5/6/17
    File:     convert_to_tf_seq
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf

from scipy.misc import imresize

from hula_commons.data.convert_to_tf_record import ConvertToTFRecord


class ConvertToTFSeq(ConvertToTFRecord):

    __seq_len = None

    def __init__(self,
                 reader,
                 out_folder,
                 out_file,
                 label_fields,
                 label_type,
                 seq_len):

        super(ConvertToTFSeq, self).__init__(reader,
                                             out_folder,
                                             out_file,
                                             label_fields,
                                             label_type)
        self.__seq_len = seq_len

    def _write_to_tf(self,
                     writer,
                     max_num_records,
                     im_size):

        num_tuples = 0
        num_fields = len(super(ConvertToTFSeq, self).fields)
        frames = []
        labels = [[] for _ in range(num_fields)]

        rec_idx = 0
        for (frame, label) in super(ConvertToTFSeq, self).reader:
            # Rescale the frame if necessary
            if im_size != frame.shape[0:2]:
                # Reconstruction sampling (sinc function first greater lobes interpolation)
                new_frame = imresize(frame, size=im_size, interp='lanczos')
            else:
                new_frame = frame
            # Save the record
            if rec_idx < self.__seq_len:
                frames.append(new_frame)
                for i in range(num_fields):
                    labels[i].append(label[i])
                # Advance to the next record
                rec_idx += 1
                if rec_idx == self.__seq_len:
                    features = super(ConvertToTFSeq, self)._prepare_features_map(frames, labels)
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())
                    num_tuples += 1
                    frames = None
                    labels = None
            else:
                # After the 1st record with size of 'seq_len' is saved,
                # all subsequent records have size of 1
                features = super(ConvertToTFSeq, self)._prepare_features_map(new_frame, label)
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
                num_tuples += 1
            # Stop saving if maximum # of records specified has been reached
            if max_num_records > 0 and num_tuples == max_num_records:
                break

        return num_tuples
