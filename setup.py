from setuptools import setup, find_packages
setup(
    name='avici',
    version='1.0.0',
    description='Amortized Inference for Causal Structure Learning',
    author='Lars Lorch',
    author_email='lars.lorch@inf.ethz.ch',
    url="https://github.com/larslorch/avici",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        'jax>=0.3.17',
        'jaxlib>=0.3.14',
        'imageio',
        'jupyter',
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'igraph',
        'python-igraph',
        'scikit-learn',
        'sklearn',
        'tqdm>=4.64.1',
        'dm-haiku>=0.0.8',
        'psutil',
        'optax>=0.1.3',
        'pyarrow==7.0.0',
        'tensorflow-datasets~=4.3.0',
        'deepdiff',
        'seaborn',
        'torch',
    ]
)