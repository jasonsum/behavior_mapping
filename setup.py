import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="behavior_mapper",
    version="1.2.0.dev1",
    author="Jason Summer",
    author_email="jasummer92@gmail.com",
    description="Clusters channel activities or steps according to the transactions offered within a given organization's channel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jasonsum/behavior_mapping",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing"
    ],
    packages=['behavior_mapper'],
    python_requires='>=3.8',
    license='MIT',
    install_requires=['pandas','nltk','gensim','numpy','scikit-learn']
    #project_urls={'':'',},
)