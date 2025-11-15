"""
Setup script for Genetic Algorithms Tutorial Series
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# Get version
__version__ = '1.0.0'

setup(
    name='genetic-algorithms-tutorial',
    version=__version__,
    author='GA Tutorial Team',
    author_email='your.email@example.com',
    description='Complete tutorial series on Genetic Algorithms from basics to industrial applications',
    long_description=read_file('README.md') if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/YOUR_USERNAME/GNETIC-ALGORITHMS',
    packages=find_packages(include=['ga_toolkit', 'ga_toolkit.*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.21.0,<2.0.0',
        'matplotlib>=3.4.0,<4.0.0',
    ],
    extras_require={
        'full': [
            'scipy>=1.7.0,<2.0.0',
            'scikit-learn>=1.0.0,<2.0.0',
            'plotly>=5.0.0,<6.0.0',
            'seaborn>=0.11.0,<1.0.0',
            'tqdm>=4.62.0,<5.0.0',
        ],
        'ml': [
            'scikit-learn>=1.0.0,<2.0.0',
            'scipy>=1.7.0,<2.0.0',
        ],
        'nas': [
            'tensorflow>=2.8.0,<3.0.0',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'notebook>=6.4.0',
            'ipywidgets>=7.6.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ga-visualize=ga_toolkit.visualization:main',
        ],
    },
    keywords='genetic-algorithms optimization evolutionary-computation machine-learning tutorial',
    project_urls={
        'Documentation': 'https://github.com/YOUR_USERNAME/GNETIC-ALGORITHMS/blob/main/README.md',
        'Source': 'https://github.com/YOUR_USERNAME/GNETIC-ALGORITHMS',
        'Tracker': 'https://github.com/YOUR_USERNAME/GNETIC-ALGORITHMS/issues',
    },
    include_package_data=True,
    zip_safe=False,
)
