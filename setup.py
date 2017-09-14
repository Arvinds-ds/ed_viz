from setuptools import setup

readme = open('README.md').read()

setup(
    name='ed_viz',
    version='0.0.2',
    description='Visualize entire tensorflow graph with Edward variables in ipython notebook using grpahviz',
    long_description=readme,
    author='Aravind S',
    author_email='arvindxxxx@gmail.com',
    url='https://github.com/arvinds-ds/ed_viz',
    py_modules=(
        'ed_viz','
    ),
    install_requires=(
        'graphviz',
    ),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Framework :: IPython',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
