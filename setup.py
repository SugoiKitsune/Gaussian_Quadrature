from setuptools import setup, find_packages

setup(
    name="gaussian_quadrature",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.16",
        "jaxlib>=0.4.16",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "plotly>=5.18.0",
        "sympy>=1.12",
        "pytest>=7.4.0",
    ],
    author="Gaussian Quadrature Contributors",
    author_email="",
    description="Advanced numerical integration techniques for financial derivatives pricing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
)