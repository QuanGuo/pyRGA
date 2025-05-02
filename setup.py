from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gwsolver",
    version="0.1.0",
    author="Quan Guo",
    author_email="qguo48@hotmail.com",
    description="A Python package for groundwater flow simulation and hydraulic tomography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QuanGuo/GWSolver",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Hydrogeology",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.8",
        "matplotlib>=3.5",
        "pyamg>=4.2",
        "mat73",
        "joblib",
        "scipy",
        "matplotlib",
        "pyamg",
        "mat73",
        "joblib"
    ]
) 