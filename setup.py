from setuptools import setup, find_packages

setup(
    name="dataframe-processor",
    version="0.1",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "matplotlib"],
    author="Your Name",
    author_email="your_email@example.com",
    description="A library for automating Pandas DataFrame preprocessing.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/DataFrameProcessor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
