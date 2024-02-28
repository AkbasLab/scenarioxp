import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scenarioxp",
    version="0.1.0",
    author="Quentin Goss",
    author_email="gossq@my.erau.edu",
    description="A toolkit for targeted scenario selection.",
    license="LICENSE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AkbasLab/scenarioxp",
    project_urls={
        "Bug Tracker": "https://github.com/AkbasLab/scenarioxp/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">= 3.9",
    install_requires=[
        "numpy >= 1.21.1",
        "scipy >= 1.7.1",
        "pandas >= 1.3.2",
        "sim_bug_tools >= 1.0.1"
    ],
)
