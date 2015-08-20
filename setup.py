try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Bisecting K-means implementation for UFABC AI class',
    'author': '@MateusZitelli, @DaniLucarini',
    'author_email': 'zitellimateus@gmail.com',
    'version': '0.1',
    'install_requires': [],
    'packages': ['kmeans'],
    'scripts': [],
    'name': 'kmeans-ai'
}

setup(**config)
