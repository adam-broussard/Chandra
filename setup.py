# pylint: skip-file
import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='galaxyteam',
    version='0.0.1',
    description=('Data operations and example analyses for working with the'
                 + 'Kaggle Chest X-ray dataset at https://www.kaggle.com/'
                 + 'datasets/paultimothymooney/chest-xray-pneumonia'),
    long_description=read('README.md'),
    url='https://github.com/adam-broussard/GalaxyTeam',
    author='Adam Broussard and Anthony Young',
    author_email='adamcbroussard@gmail.com',
    license='GNU GPLv3',
    packages=['galaxyteam'],
    install_requires=['numpy',
                      'scipy',
                      'kaggle',
                      'pandas',
                      'tqdm',
                      'tensorflow',
                      'keras',
                      'matplotlib'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3'
    ],
)
