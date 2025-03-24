import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="rwm_dataset_tools",
    version="0.1.0",
    author="AU Aarhus University",
    author_email="msn@agro.au.dk",
    description="RoboWeedMaPS dataset extraction tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/heltechael/rwm_dataset_tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu 22.04",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'rwm-extract=run:main',
        ],
    },
)