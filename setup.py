from setuptools import setup, find_packages

setup(
    name="DataFrameProcessor",  # Replace with your package name
    version="0.1.0",  # Initial version
    author="Your Name",  # Replace with your name
    author_email="your_email@example.com",  # Replace with your email
    description="A Python package for processing DataFrames",  # Short description
    long_description=open("README.md").read(),  # Ensure you have a README.md file
    long_description_content_type="text/markdown",
    url="https://github.com/janetcheung-byte/DataFrameProcessor",  # Repository URL
    packages=find_packages(),  # Automatically find sub-packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
    install_requires=[
        "pandas",  # Add dependencies here
        "numpy",
    ],
)
