from setuptools import setup, find_packages

setup(
    name="pocketml",
    author="Aditya",
    author_email="adityakuma0308@gmail.com",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "seaborn", "matplotlib"],
)