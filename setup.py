from setuptools import setup

setup(
    name='GSTI',
    version='0.1',
    description='Geodetic and seismolgical time series inversion',
    author_email='simon.daout@ifg.uni-kiel.de',
    package_dir={'GSTI': 'src'},
    packages=[
        'GSTI'
    ],
    entry_points={
        'console_scripts':
            ['gsti = GSTI.optimize']
    }
)
