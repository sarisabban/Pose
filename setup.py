import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
	name='pose',
	version='1.0',
	author='Sari Sabban',
	author_email='',
	description='A bare metal Python library for building and manipulating protein molecular structures',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/sarisabban/Pose',
	project_urls={'Bug Tracker':'https://github.com/sarisabban/Pose/issues'},
	license='GPL-2.0',
	packages=['pose'],
	include_package_data=True,
	package_data={'pose': ['*.json']},
	install_requires=['numpy'])
