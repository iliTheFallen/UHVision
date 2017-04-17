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
    Date:     4/9/17
    File:     convert_to_tf_record
    Comments: Converts image+label data to TFRecords file format with Example protos.
    **********************************************************************************
'''

import tensorflow as tf

import os
from scipy.misc import imresize


from utils import constants as consts


class ConvertToTFRecord(object):

    __reader = None
    __out_folder = None
    __out_file = None
    __fields = None

    def __init__(self,
                 reader,
                 out_folder,
                 out_file,
                 fields):

        if not hasattr(reader, '__iter__') or not hasattr(reader, 'reset'):
            raise ValueError('Invalid Reader!')

        self.__reader = reader
        self.__out_folder = out_folder
        self.__out_file = out_file
        self.__fields = list(fields)  # Avoid exhaustion

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _prepare_features_map(self, frame, label):

        features = {}
        i = 0
        for (key, value) in self.__fields:
            if value == tf.uint8:
                features[key] = ConvertToTFRecord._bytes_feature(label[i])
            elif value == tf.int64:
                features[key] = ConvertToTFRecord._int64_feature(label[i])
            else:
                features[key] = ConvertToTFRecord._float_feature(label[i])
            i += 1
        # Finally append our raw image data to our feature map
        features[consts.IMAGE_RAW] = ConvertToTFRecord._bytes_feature(frame.tostring())
        return features

    def convert(self,
                max_num_records,
                im_size=(300, 400)):
        ''' Convert
        
        Converts samples provided by reader to native tensorflow format
        
        :param max_num_records:
        :param im_size:
        :return: None
        '''

        file_name = os.path.join(self.__out_folder,
                                 self.__out_file+'.tfrecords')
        # Remove if it exists
        if tf.gfile.Exists(file_name):
            tf.gfile.Remove(file_name)
        # Reset the source
        self.__reader.reset()
        print('Writing records into %s' % file_name)
        writer = tf.python_io.TFRecordWriter(file_name)
        num_tuples = 0
        for (frame, label) in self.__reader:
            if im_size != frame.shape[0:2]:
                # Reconstruction sampling (sinc function first greater lobes interpolation)
                new_frame = imresize(frame, size=im_size, interp='lanczos')
            else:
                new_frame = frame
            features = self._prepare_features_map(new_frame, label.tolist())
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            num_tuples += 1
            if max_num_records > 0 and num_tuples == max_num_records:
                break
        writer.close()
        print('Total number of tuples written: %d' % num_tuples)




