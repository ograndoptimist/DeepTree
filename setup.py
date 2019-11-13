from setuptools import setup, find_packages


setup(
    name='deep-tree',
    version='0.1',
    packages=find_packages(),
    license=license,
    long_description_content_type="text/markdown",
    long_description="",
    author="Gabriel de Miranda ",
    url="https://github.com/ograndoptimist/DeepTree",
    description="A Pipeline for text data",
    setup_requires=['wheel', 'twine'],
    install_requires=['pandas', 'keras', 'numpy', 'treelib', 'tensorflow', 'sklearn', 'unidecode', 'nltk']
)
