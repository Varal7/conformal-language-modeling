from setuptools import setup, find_packages

setup(
    name='clm',
    python_requires='>=3.7',
    packages=find_packages(exclude=('data', 'results')),
)
