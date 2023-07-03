import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="quantum phase transition machine learning (qptml)",
    version="0.0.1",
    author="Hooman Karamnejad",
    packages=[],
    description="Quantum Phase Transition Machine Learning",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/hoomania/qptml.git",
    license='MIT',
    python_requires='>=3.8',
    install_requires=[]
)
