from setuptools import setup, find_packages 

setup(     
      name = 'nemotoc',
      version = '1.0.0-beta',
      description='neighboring molecule topology clustering(NEMO-TOC)',
      author='Wenhong Jiang',
      author_email='jiangwh@pku.edu.cn',
      url='https://github.com/GuoQLabPKU/polysomeTracking',
      packages=find_packages(),
      
      include_package_data = True,
      install_requires = [
              'python>=3'
              'numpy',
              'pandas',
              'matplotlib',
              'cupy',
              'cudatoolkit',
              'pytest',
              'networkx',
              'seaborn',
              'dill',
              'alive-progress',
              'psutil' ]
      
      
      
      )
