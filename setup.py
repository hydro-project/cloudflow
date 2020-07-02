#  Copyright 2019 U.C. Berkeley RISE Lab
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os

from distutils.core import setup
from setuptools.command.install import install
from setuptools import find_packages


class InstallWrapper(install):
    def run(self):
        # Run the standard PyPi copy
        install.run(self)

setup(
        name='flow',
        version='0.1.0',
        packages=find_packages(),
        license='Apache v2',
        long_description='The Hydrflow API',
        install_requires=['cloudburst'],
        cmdclass={'install': InstallWrapper}
)
