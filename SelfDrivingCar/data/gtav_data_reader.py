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
    Author:   Sidharth Sadani
    Date:     3/7/17
    File:     gtav_data_reader
    Comments: This module loads input data which consist of both video frames
     and associated dependent data such as steering angle, speed, out of road flag, etc.
    **********************************************************************************
'''

# import os
# import random

import numpy as np
from PIL import Image
import json

# print "Hello World"


class GTAVDataReader(object):

    def __init__ (self,
                  episodeSize=80,
                  max_iter=0,
                  isLast=False,
                  drive_folder='.',
                  jsonFolder='FData',
                  imgFolder='Frames',
                  filePrefix='frame'):
        self.episodeSize = episodeSize # Number of RNN Unrolling
        self.max_iter = max_iter
        self.num_iter = 0
        self.isLast = isLast # Have we seen the last frame of this drive
        self.driveFolder = drive_folder  # IG
        self.jsonFolder = jsonFolder
        self.imgFolder = imgFolder
        self.filePrefix = filePrefix
        # self.driveFolder = input("Enter Data Folder: ")
        self.jsonPrefix = None
        self.framePrefix = None
        self.bufSize = 0 # Current Size of Buffer
        self.lastFrameNo = 0 # Last Frame No Read Into Buffer
        # self.frameBuf = [] # Current List of Image Frames
        # self.dataBuf = [] # Current List of Data Frames
        self.frameBuf = None
        self.dataBuf = None
        self.fileDict = dict()
        self.parseListFile()
        self.fileList = list(self.fileDict.keys())
        self.maxDrive = len(self.fileList)
        print(self.maxDrive)
        self.currDrive = -1
        self.driveStartFrame = 0
        self.driveEndFrame = 0
        self.changeDrive()
        # for i in range(episodeSize):
        # self.loadSample()

        # print self.frameBuf

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # if self.num_iter == 0:
        #     self.num_iter += 1
        #     return self.frameBuf, self.dataBuf
        if not self.isLast:
            self.loadSample()
            self.num_iter += 1
            return self.frameBuf, self.dataBuf
        else:
            raise StopIteration()

    def parseListFile(self):
        listFile = self.driveFolder + '/listFile.txt'
        fHandle = open(listFile)
        for line in fHandle:
            if line.startswith("#"):
                continue
            lineLst = (line.strip()).split()
            self.fileDict[lineLst[0]] = (lineLst[1], lineLst[2]) # Start frame and end frame
        print(self.fileDict)


    def changeDrive(self):
        print("Changing Drives")
        # print("Drive: ", self.currDrive)
        # print("Drive Path: ", self.jsonPrefix)
        # print("Last Frame: ", self.lastFrameNo)
        self.currDrive += 1
        i = self.currDrive
        if(i==self.maxDrive):
            print("No More Data, All Drives Exhausted!")
            self.isLast = True
            raise StopIteration()
        self.jsonPrefix = self.driveFolder + '/' + self.fileList[i] + \
                '/' + self.jsonFolder + '/' + self.filePrefix
        self.framePrefix = self.driveFolder + '/' + self.fileList[i] + \
                '/' + self.imgFolder + '/' + self.filePrefix
        self.driveStartFrame = int(self.fileDict[self.fileList[i]][0])
        self.lastFrameNo = self.driveStartFrame - 1
        self.driveEndFrame = int(self.fileDict[self.fileList[i]][1])

    def loadSample(self):

        currFrameNo = self.lastFrameNo + 1
        if(currFrameNo > self.driveEndFrame):
            self.changeDrive()
            self.loadSample()
            return None
        jsonPath = self.jsonPrefix + str(currFrameNo) + '.json'
        imgPath = self.framePrefix + str(currFrameNo) + '.jpeg'
        try:
            jsonHandle = open(jsonPath)
        except:
            # self.isLast = True
            # This exception clause is to catch any frames that we manually remove from the dataset
            self.loadSample()
            return None
        jsonData = json.load(jsonHandle)
        jsonHandle.close()
        self.lastFrameNo += 1
        try:
            speed_exists = jsonData['n'][0]['speed']
        except:
            # print("Drive Path: ", self.jsonPrefix)
            # print("Last Frame: ", self.lastFrameNo)
            self.loadSample()
            return None

        ## Read JSON DATA:
        steer = jsonData['n'][0]['steer']
        throttle = jsonData['n'][0]['throttle']
        brake = jsonData['n'][0]['brake']

        controls = np.array([steer, throttle, brake], dtype=np.float)

        ## Read IMAGE DATA:
        img = Image.open(imgPath)
        img.load()
        imgData = np.asarray(img, dtype=np.uint8)
        # print imgPath
        # Update the frame and data buffers
        # if self.bufSize==self.episodeSize:
        #     self.frameBuf.pop(0)
        #     self.bufSize -= 1
        #     self.dataBuf.pop(0)
        # self.frameBuf.append(imgData)
        self.frameBuf = imgData
        # self.bufSize += 1
        # self.dataBuf.append(controls)
        self.dataBuf = controls
        # TODO: Check if next frame exists, if not set isLast = TRUE
        # nextFrameNo = currFrameNo + 1

    def reset(self):
        self.lastFrameNo = 0
        self.num_iter = 0
        self.frameBuf = None
        self.dataBuf = None


#################################
########## EXAMPLE USAGE: #######
# Comment Out For Actual Use
# GTAVDataReader(episodeSize, max_iter, isLast, folderContainingDrives)
#drivesFolder = '/Users/archer/NTM/Env/Data'
#x = GTAVDataReader(3, 0, False, drivesFolder)
#while(not x.isLast):
#    a,b = x.next()
#    print x.lastFrameNo
#    # print a
#    # print b
#    g = raw_input("Press Any Key To See Next Buffers, or Ctrl + C to Terminate")
#    print "Done"
