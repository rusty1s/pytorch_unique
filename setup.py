import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '1.0.2'
url = 'https://github.com/rusty1s/pytorch_unique'

install_requires = ['numpy']
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']
ext_modules = []
cmdclass = {}

if torch.cuda.is_available():
    ext_modules += [
        CUDAExtension('unique_cuda',
                      ['cuda/unique.cpp', 'cuda/unique_kernel.cu'])
    ]
    cmdclass['build_ext'] = BuildExtension

setup(
    name='torch_unique',
    version=__version__,
    description='Optimized PyTorch Unique Operation',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=['pytorch', 'unique', 'python'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
