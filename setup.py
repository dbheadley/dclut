from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'dclut: Dimensionally Clear Look Up Table'

with open("README.md", "r") as f:
    readme = f.read()

requirements = ['numpy>=1.26.4',
                'tqdm>=4.66.4',
                'xarray>=2024.5.0']
# Setting up
setup(
        name="dclut", 
        version=VERSION,
        author="Drew B. Headley",
        author_email="drewbheadley@gmail.com",
        description=DESCRIPTION,
        long_description=readme,
        packages=find_packages(where='dclut'),
        install_requires=requirements,
        keywords=['python', 'binary files', 'lookup table', 'dclut'],
        classifiers= [
            "Development Status :: Alpha",
            "Intended Audience :: Researchers",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
        license="MIT License",
        url="https://github.com/dbheadley/dclut",
        long_description_content_type="text/markdown"
)