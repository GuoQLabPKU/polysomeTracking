from setuptools import setup, find_packages 
import os, sys 
sys.path.insert(0, f'{os.path.dirname(__file__)}/nemotoc')
setup(     
      name = 'nemotoc',
      version = '1.0.0b0',
      description='neighboring molecule topology clustering(NEMO-TOC)',
      author='Wenhong Jiang',
      author_email='jiangwh@pku.edu.cn',
      url='https://github.com/GuoQLabPKU/polysomeTracking',
      packages=find_packages(),
      include_package_data = True,
     
      install_requires = [
              'numpy',
              'pandas',
              'matplotlib', 
              'cupy',
              'pytest',
              'networkx',
              'seaborn',
              'dill',
              'alive-progress',
              'psutil' ],
      
      python_requires = '>=3'
  
      )
