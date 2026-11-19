from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyrga-ht",
    version="0.2.0.dev0",
    author="Quan Guo",
    author_email="qguo48@hotmail.com",
    description="A Python package for groundwater flow simulation and hydraulic tomography using the Reformulated Geostatistical Approach (RGA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QuanGuo/pyRGA",
    packages=["pyrga"],
    package_dir={"pyrga": "src/pyRGA"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.8",
        "matplotlib>=3.5",
        "pyamg>=4.2",
        "mat73",
        "joblib",
        "sympy>=1.10",
        "pydantic>=2.7",
    ],
    extras_require={
        "plot": ["matplotlib>=3.5", "imageio>=2.0", "ipython>=8.0"],
        "test": ["pytest>=7.0", "httpx>=0.27"],
        "mcp": ["mcp>=1.20"],
        "web": ["fastapi>=0.115", "uvicorn>=0.30"],
    },
    entry_points={"console_scripts": [
        "pyrga-mcp=pyrga.mcp_server:main", "pyrga-api=pyrga.web_server:main"
    ]},
)
