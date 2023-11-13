from setuptools import setup, find_packages


setup(
    name='document_dataset',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "transformers",
        "Pillow",
    ],
    author='Victorgonl',
    author_email='victorgonl@outlook.com',
    description='A package for document images datasets.',
)
