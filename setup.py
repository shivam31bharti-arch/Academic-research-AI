#able to build ml project as a package

from setuptools import find_packages,setup
from typing import List
import os

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

# Read long description safely
long_description = ''
try:
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            long_description = f.read()
except Exception:
    long_description = 'Automated Academic Research Paper Classification System using NLP and AutoML'

setup(
name='academic-research-ai',
version='1.0.0',
author='Shivam Bharti',
author_email='shivam.bharti@example.com',
description='Automated Academic Research Paper Classification System using NLP and AutoML',
long_description=long_description,
long_description_content_type='text/markdown',
url='https://github.com/shivam31bharti-arch/Academic-research-AI',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
python_requires='>=3.8',
classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
],
)
