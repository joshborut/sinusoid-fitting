import codecs
from setuptools import find_packages
from setuptools import setup


install_requires = [
    'torch',
    'numpy',
    'matplotlib',
]

setup(name='sinusoid-approximation',
      version='0.0.0',
      description='sinusoidal approximation with deep learning',
      long_description=codecs.open('README.md', 'r', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      author='Josh Borut',
      author_email='jcborut@gmail.com',
      license='MIT License',
      packages=find_packages(),
      install_requires=install_requires,
      test_requires=[]
      )
