from setuptools import setup, find_packages
import sys, os

version = '0.1.3'

setup(name='natter',
      version=version,
      description="Natural Image Statistics Toolbox",
      long_description=""" """,
      install_requires=["Sphinx",
                        "numpy",
                        "scipy",
                        "mpmath",
                        "MDP"],
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='',
      author='Fabian Sinz, Sebastian Gerwinn, Lucas Theis, Philipp Lies',
      author_email='natter@bethgelab.org',
      url='',
      dependency_links = [
                          "http://sourceforge.net/projects/mdp-toolkit/files/mdp-toolkit/3.3/MDP-3.3.tar.gz/download"
                          ],
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
#      package_data={'':'LICENSE'},
      include_package_data=True,
      zip_safe=True,
      entry_points="""
      # -*- Entry points: -*-
      """,
      use_2to3 = True,
      )
