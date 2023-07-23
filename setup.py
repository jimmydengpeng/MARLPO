import sys
from setuptools import setup, find_packages

# F-strings is requried
assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup(
    name='marlpo',
    version='0.0.1',
    author='JimmyDeng',
    author_email="jimmydengpeng@gmail.com",
    license="Apache 2.0",
    packages=find_packages(),
    exclude_package_data={
        '': ['exp_results/']
    }
)
