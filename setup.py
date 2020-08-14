from setuptools import setup

setup(name='pad_classification',
      version='0.1',
      description='Deep learning experiments for the peripheral aritery disease (PAD)',
      url='http://github.com/AgenoDrei/pad_classification.git',
      author='AgenoDrei',
      author_email='s.mueller1995@gmail.com',
      license='MIT',
      packages=['pad_classificaiton'],
      install_requires=[
          'scikit-learn',
	  'torch',
	  'pandas',
	  'torchvision',
      ],
      zip_safe=False)
