from setuptools import find_packages, setup

setup(
    name="citk",
    version="0.1.0",
    description=("computational intelligence toolkit"),
    author="Androsov Dmitri, Vladimir Sydorskyi",
    python_requires=">=3.6",
    install_requires=[
        "autograd==1.3",
        "numpy>=1.18.5",
        "tqdm>=4.48.0",
    ],
    packages=find_packages(),
)