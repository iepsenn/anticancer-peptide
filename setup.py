from setuptools import setup, find_packages

setup(
    name="tacape",
    version='0.0.3',
    packages=find_packages(),
    url="https://github.com/omixlab/anticancer-peptide",
    author="Isadora Leitzke Guidotti, Frederico Schmitt Kremer",
    author_email="fred.s.kremer@gmail.com",
    description="TACaPe: Transformed-based Anti-Cancer Peptide Classification and Generation",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    keywords="bioinformatics machine-learning data science drug discovery QSAR",
    entry_points = {'console_scripts':[
        'tacape-train-classifier = tacape.train_classifier:main',
        'tacape-predict          = tacape.run_classifier:main',
        'tacape-train-generator  = tacape.train_generator:main',
        'tacape-generate         = tacape.run_generator:main'
        ]},
    install_requires = [
        requirement.strip('\n') for requirement in open("requirements.txt")
    ]
)