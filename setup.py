import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="behavior_mapper",
    version="1.2.0.dev1",
    author="Jason Summer",
    author_email="jasummer92@gmail.com",
    description="Package clusters channel activities or steps according to the transactions offered within a given organization's channel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="#",
    classifiers=[
        "Development Status :: 3 - Alpha"
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing"
    ],
    package_dir={'','behavior_mapper'},
    #packages=setuptools.find_packages(),
    python_requires='>=3.8',
    license='MIT',
    install_requires=['pandas','nltk','gensim','numpy','sklearn']
    #project_urls={'':'',},
)