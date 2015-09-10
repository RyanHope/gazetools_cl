#===============================================================================
# This file is part of gazetools.
# Copyright (C) 2015 Ryan M. Hope <rmh3093@gmail.com>
#
# gazetools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gazetools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gazetools.  If not, see <http://www.gnu.org/licenses/>.
#===============================================================================

from distutils.core import setup

__version__ = '0.0.1'

setup(
    name='gazetools',
    version=__version__,

    packages=[
        'gazetools',
        'gazetools.resources',
        'gazetools.resources.cl',
        'gazetools.resources.images',
        'gazetools.resources.data'
    ],
    package_dir={
        'gazetools': 'python/gazetools',
        'gazetools.resources.cl': 'cl',
        'gazetools.resources.images': 'images',
        'gazetools.resources.data': 'data',
    },
    package_data={
        'gazetools.resources.cl': ['*.cl'],
        'gazetools.resources.images': ['*.png'],
        'gazetools.resources.data': ['*.csv']
    },

    description='OpenCL implementation of gazetools.',
    author='Ryan Hope',
    author_email='rmh3093@gmail.com',
    url='https://github.com/RyanHope/gazetools_cl',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
		'Programming Language :: Python :: 2',
		'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
		'Topic :: Utilities',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research'
    ],
	license='GPL-3',
	install_requires=[
        'pyopencl',
        'numpy'
	]
 )
