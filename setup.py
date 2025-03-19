# This is responsible in creating my machine learning application as a package and even delpoy in pypy.

from setuptools import find_packages, setup
from typing import List

HYPTHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the list of requirements
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPTHEN_E_DOT in HYPTHEN_E_DOT:
            requirements.remove(HYPTHEN_E_DOT)

    return requirements


setup(
    name="cosmeticDefectsDetector",
    version="0.0.1",
    author="Siddharth, Arvind, Anjali, Aakash, Samrudhi",
    author_email="s4012307@student.rmit.edu.au",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
