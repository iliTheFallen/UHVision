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
    File:     setup
    Comments: Anyone may use this script in order to install hula_commons
     project as a python distribution
    **********************************************************************************
'''

from setuptools import setup, find_packages

setup(
      name="hula_commons",
      version="0.0.1",
      author="Ilker GURCAN",
      author_email="igurcan@central.uh.edu",
      description="Common Software Packages used by HULA Lab",
      license="GNU",
      keywords="hula common python packages",
      url="https://github.com/iliTheFallen/UHVision/tree/master/MetaLearningTF",
      packages=find_packages(),
      requires=['tensorflow (>=1.0)', 'scipy', 'numpy']
)
