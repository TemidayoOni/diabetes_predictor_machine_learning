from setuptools import setup, find_packages
from typing import List

# function to gather requirements from the requirements.txt file 

def get_requirements(file_to_path:str)->List[str]:

    """
    This function gets the requirments to setup, build and train the model 

    """
    hyphen_e = '-e .'

    requirements = []
    with open(file_to_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [r.replace("n/", "") for r in requirements]


        if hyphen_e in requirements:
            requirements.remove(hyphen_e)
    
    return requirements




# setup configuration

setup(
    name="diabetes_predictor",
    version= '0.0.1',
    author='Temidayo Oni',
    author_email='onitemidayo@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages=find_packages()
)


