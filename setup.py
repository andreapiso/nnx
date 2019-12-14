import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


if __name__ == "__main__":
    setup(
        name='nnx',
        version='0.0.1',
        long_description=long_description,  
        long_description_content_type='text/markdown',
        url='https://github.com/andreapiso/nnx',
        
        packages = [
            "nnx",
            "nnx.classes"
        ],

        python_requires='>=3.6',

        install_requires=['numba >= 0.46']
    )