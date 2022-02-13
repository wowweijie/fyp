from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'torch==1.9.0+cu111',
    'torchvision==0.10.0+cu111',
    'torchaudio==0.9.0',
    'stable-baselines3[extra]==1.4.0',
    'pyyaml',
    'gym']

setup(
    name='fyp',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package'
)