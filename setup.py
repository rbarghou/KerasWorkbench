from distutils.core import setup

setup(
    name='KerasWorkbench',
    version='0.1dev',
    author='Ramsey Barghouti',
    description="keras workbench for simple tooling that I would like to use across platforms",
    packages=[
        'kerasworkbench',
        'kerasworkbench.data',
        'kerasworkbench.data.audio',
        'kerasworkbench.generators',
    ],
    license='GPL-3.0',
    long_description=open('README.md').read(),
    install_requires=[
        "keras",
        "librosa",
        "numpy",
        "scipy",
        "tensorflow",
        "keras",
        "matplotlib",
        "pandas",
    ],
    data_files=[
        'kerasworkbench/data/audio/*.mp3',
    ]
)