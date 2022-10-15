from setuptools import setup, find_packages
setup(
    name='dibs',
    version='1.0.1',
    description='DiBS: Differentiable Bayesian Structure Learning',
    author='Lars Lorch',
    author_email='lars.lorch@inf.ethz.ch',
    url="https://github.com/larslorch/dibs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        'jax>=0.2.8',
        'jaxlib>=0.1.59',
        'numpy',
        'pandas',
        'python-igraph',
        'imageio',
        'jupyter',
        'tqdm',
        'matplotlib',
    ]
)
