from setuptools import setup, find_packages

setup(
    name="intake-esgf",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "intake.drivers": [
            "esgf-catalog = intake_esgf.core:ESGFCatalog",
            "esgf-default-catalog = intake_esgf.core:ESGFDefaultCatalog",
        ],
        "console_scripts": [
            "query-facets = intake_esgf.tools:main",
        ],
    },
    package_data={
        "": ["*.yaml"],
    },
)
