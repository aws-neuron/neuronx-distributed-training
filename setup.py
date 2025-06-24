from setuptools import PEP420PackageFinder, setup

exec(open("src/neuronx_distributed_training/_version.py").read())
setup(
    name="neuronx-distributed-training",
    version=__version__, #noqa F821
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="aws neuron",
    packages=PEP420PackageFinder.find(where="src"),
    package_data={"": []},
    install_requires=["neuronx_distributed"],
    python_requires=">=3.7",
    package_dir={"": "src"},
    require_wheel=True,  # Enable Python Wheel. Run 'brazil-build brazil_wheel'
    doc_command='amazon_doc_utils_build_sphinx'
)
