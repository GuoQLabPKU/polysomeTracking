from setuptools import setup, find_packages 

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
              'psutil' ]
      
      
  
      )
