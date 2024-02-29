from setuptools import setup

setup(
    name="deep_mono_depth",
    version="0.1",
    description="Framework for Deep Learning based monocular depth training",
    url="https://github.com/JEpp86/DeepMonoDepth",
    author="Jason Epp",
    author_email="jasonepp0@gmail.com",
    packages=["deep_mono_depth"],
    package_data={
        "deep_mono_depth": [
            "core\\*",
            "data\\*",
            "network\\*",
            "util\\*",
        ]
    },
    entry_points={
        "console_scripts": [
            "dmd_train=deep_mono_depth.Train:main",
            #"dmd_validate=deep_mono_depth.Validate:main",
            #"dmd_evaluate=deep_mono_depth.Evaluate:main",
            #"dmd_export=deep_mono_depth.Export:main",
        ]
    },
)
