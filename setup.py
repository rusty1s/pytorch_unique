from os import path as osp

from setuptools import setup, find_packages

__version__ = '0.3.1'
url = 'https://github.com/rusty1s/pytorch_unique'

install_requires = ['cffi']
setup_requires = ['pytest-runner', 'cffi']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_unique',
    version=__version__,
    description='PyTorch Extension Library of Optimized Unique Operations',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=['pytorch', 'unique', 'deep-learning'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(exclude=['build']),
    ext_package='',
    cffi_modules=[osp.join(osp.dirname(__file__), 'build.py:ffi')],
)
