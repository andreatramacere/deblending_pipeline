from setuptools import setup, find_packages

packs = find_packages()

print('packs', packs)


setup(
    name='deblending_pipeline',
    version='0.0.0',
    packages=packs,
    url='',
    license='',
    author='andrea tramacere',
    author_email='andrea.tramacere@gmail.com',
    description='',
    install_requires=['numpy', 'pandas', 'asterism','astropy'],
    zipsafe=False,
)

