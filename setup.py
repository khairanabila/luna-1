from os import path as op
from setuptools import find_packages, setup
import io

with open("README.rst") as readme_file:
    readme = readme_file.read()

here = op.abspath(op.dirname(__file__))

with io.open(op.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [x.strip().replace("git+", "") for x in all_reqs if "+git" not in x]

setup_requires = []

setup(
    name="derune-luna",
    version="1.0.1",
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    author="arfy slowy",
    author_email="slowy.arfy@gmail.com",
    platform=["any"],
    install_requires=install_requires,
    dependency_links=dependency_links,
    long_description=readme,
    keywords="luna stable diffusion",
    setup_requires=setup_requires,
    url="https://github.com/De-Rune/luna",
    packages=find_packages("stable_diffusion_tensorflow"),
    package_dir={"": "stable_diffusion_tensorflow"},
)
