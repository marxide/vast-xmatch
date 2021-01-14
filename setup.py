from setuptools import setup, find_packages
import vast_xmatch

setup(
    name="vast-xmatch",
    url="https://github.com/askap-vast/vast-xmatch/",
    author="Andrew O'Brien",
    author_email="obrienan@uwm.edu",
    packages=find_packages(),
    version=vast_xmatch.__version__,
    license="MIT",
    description="Python package for internal crossmatching of VAST catalogues.",
    install_requires=[
        "astropy==4.2",
        "click==7.1.2",
        "matplotlib==3.3.3",
        "numpy==1.19.4",
        "pandas==1.1.5",
        "peewee==3.14.0",
        "scipy==1.5.4",
        "seaborn==0.11.0",
        "uncertainties==3.1.5",
    ],
    entry_points="""
        [console_scripts]
        vast_xmatch_qc=vast_xmatch.cli:vast_xmatch_qc
        vast_xmatch_export=vast_xmatch.cli:vast_xmatch_export
    """,
)
