from setuptools import setup


setup(name='nopg',
      version='0.1.1',
      description='Nonparametric Off-Policy Policy Gradient',
      url='https://github.com/jacarvalho/nopg',
      author='Joao Carvalho',
      author_email='joao@robot-learning.de',
      license='MIT',
      packages=['src'],
      zip_safe=False,
      install_requires=[
          'cloudpickle==1.2.2',
          'getch==1.0',
          'gym==0.15.4',
          'matplotlib==3.1.2',
          'numpy==1.18.1',
          'scikit-learn==0.22.1',
          'scipy==1.4.1',
          'sklearn==0.0',
          'torch==1.4.0',
          'jupyter==1.0.0',
          'Box2D-py==2.3.8',
          'seaborn'
      ],
      )
