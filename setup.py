from setuptools import setup, find_packages

packs = find_packages()

print('packs', packs)


entry_points = {
    'console_scripts':[
    'run_deblending_pipeline = deblending_pipeline.command_line.run_deblending_pipeline:main']
}

setup(
    name='deblending_pipeline',
    version='0.0.0',
    packages=packs,
    url='',
    license='',
    author='andrea tramacere',
    author_email='andrea.tramacere@gmail.com',
    description='',
    install_requires=['numpy', 'pandas', 'asterism','astropy','simplejson'],
    zipsafe=False,
    entry_points=entry_points,
)

