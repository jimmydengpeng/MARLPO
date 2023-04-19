import sys
from setuptools import setup

# F-strings is requried
assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup()

# setup(
#     name='colorlog',
#     version='0.0.1',
#     description='A simple logger supporting colorful terminal outputs.',
#     author='JimmyDeng',
#     author_email="jimmydengpeng@gmail.com",
#     license="Apache 2.0",
# )