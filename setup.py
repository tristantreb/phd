# https://www.datasciencelearner.com/importerror-attempted-relative-import-parent-package/#:~:text=Importerror%20attempted%20relative%20import%20with%20no%20known%20parent%20package%20error,for%20the%20package%20is%20undefined

from setuptools import find_packages, setup

setup(name="scr", packages=find_packages())
