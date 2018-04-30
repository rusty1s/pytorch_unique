import os.path as osp
import subprocess

import torch
from torch.utils.ffi import create_extension

headers = []
sources = []
include_dirs = ['torch_unique/src']
define_macros = []
extra_objects = []
extra_compile_args = ['-std=c99']
with_cuda = False

if torch.cuda.is_available():
    subprocess.call(['./build.sh', osp.dirname(torch.__file__)])

    headers += ['torch_unique/src/cuda.h']
    sources += ['torch_unique/src/cuda.c']
    include_dirs += ['torch_unique/kernel']
    define_macros += [('WITH_CUDA', None)]
    extra_objects += ['torch_unique/build/kernel.so']
    with_cuda = True

ffi = create_extension(
    name='torch_unique._ext.ffi',
    package=True,
    headers=headers,
    sources=sources,
    include_dirs=include_dirs,
    define_macros=define_macros,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args,
    with_cuda=with_cuda,
    relative_to=__file__)

if __name__ == '__main__':
    ffi.build()
