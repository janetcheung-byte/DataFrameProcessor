from setuptools import setup, find_packages

# Ensure UTF-8 encoding for README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="DataFrameProcessor",
    version="0.1.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="A Python package for processing DataFrames",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/janetcheung-byte/DataFrameProcessor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas",
        "numpy",
    ],
)
