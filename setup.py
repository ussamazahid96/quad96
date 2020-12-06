import setuptools

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Quad96-PYNQ",
    version="0.1",
    author="Ussama Zahid",
    author_email="ussamazahid96@gmail.com",
    description="Quadcopter Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ussamazahid96/quad96",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)