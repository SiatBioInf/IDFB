from setuptools import find_packages, setup

setup(
    name="idfb",
    version="1.0.0",
    description="Integration of Data From Bulk RNA Sequencing Platforms",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "scipy==1.11.4",
        "torch>=2.2.0",
    ],
    entry_points={
        "console_scripts": [
            "idfb=idfb.cli:main",
        ],
    },
)
