import runpy

import setuptools

# Parse requirements
install_requires = [line.strip() for line in open("requirements.txt").readlines()]

# Get long description
with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

__version__ = runpy.run_path("face_autoencoder/_version.py")["__version__"]

# Setup package
setuptools.setup(
    name="face_autoencoder",
    version=__version__,
    author="Wiktor Łazarski & Aneta Jaśkiewicz",
    author_email="wjlazarski@gmail.com",
    description="Edge-Aware Autoencoder for human faces.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_project_link",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
)
