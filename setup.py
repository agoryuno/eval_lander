from setuptools import setup

setup(
    name='eval_lander',
    version='0.1',
    packages=['eval_lander'],
    url='',
    license='',
    author='Alexander Goryunov',
    author_email='alex.goryunov@gmail.com',
    description='',
    requires_python='>=3.6',
    install_requires=[
        'gym',
        'numpy',
        'stable-baselines3',
        'Box2D'
    ]
)
