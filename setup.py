import io
import os
import sys
import subprocess
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

PUBLISH_CMD = 'python setup.py register sdist upload'

if 'publish' in sys.argv:
    status = subprocess.call(PUBLISH_CMD, shell=True)
    sys.exit(status)

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('matchzoo/version.py').read()) 

short_description = 'Facilitating the design, comparison and sharing of deep text matching models. Based on MatchZoo'

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = [l.strip() for l in f if l.strip()]

extras_requires = {
    'tests': [
        'coverage >= 4.3.4',
        'codecov >= 2.0.15',
        'pytest >= 3.0.3',
        'pytest-cov >= 2.4.0',
        'flake8 >= 3.6.0',
        'flake8_docstrings >= 1.0.2'],
}


setup(
    name="matchzoo-lite",
    version=__version__,
    author="Sean Lee",
    author_email="xmlee@gmail.com",
    description=(short_description),
    license="Apache 2.0",
    keywords="text matching models based on https://github.com/NTMC-Community/MatchZoo",
    url="https://github.com/4AI/matchzoo-lite",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=install_requires,
    extras_require=extras_requires
)

