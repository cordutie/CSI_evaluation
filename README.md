# MIR Final Project: *Cover Identification using both Essentia and Libfmp*

**Esteban Gutiérrez**<sup>1</sup>, **Suvi Haeaerae**<sup>1</sup>, and **Isabelle Oktay**<sup>1</sup>

<sup>1</sup> *Department of Information and Communications Technologies, Universitat Pompeu Fabra* <br>

<div align="left">

## Introduction

Automatic cover detection presents a significant theoretical and practical challenge in Music
Information Retrieval (MIR). It plays a crucial role in maintaining comprehensive music
databases due to the frequent reinterpretation of songs by different artists and the release
of alternate versions by original artists. Besides aiding content organization for improved
music discovery, cover song identification (CSI) also facilitates copyright enforcement and
ensures proper attribution, addressing legal and ethical concerns in digital music distribution.
Despite successful implementations on small datasets, scalability remains an issue,
while scalable solutions often lack the required accuracy.

In this repository, two implementations of CSI algorithms are included so that the user can easily
apply them in local libraries.

## Requirements

In order to run the code in this repository, it is necessary to have the following packages installed:

- numpy==1.26.3
- librosa==0.10.1
- matplotlib==3.8.2
- pydub==0.25.1
- pytube==15.0.0
- essentia==2.1b6.dev1110
- ipython==7.34.0

Additionally, pydub might ask the user to install FFmpeg codecs. To install them on Linux, run:

```bash
sudo apt install ffmpeg
```

## Contents

The repository contains one folder called preamble, containing all the code necessary to run the repo, and two
ready-to-run scripts that use the code inside the folder preamble. For a brief tutorial on the use of each script, see
the following sections.

## Tutorial :)

In this brief tutorial we will see how to use this repository to both create a dataset of songs downloaded from Youtube
and then we will run the CSI algorithms to evaluate if the songs in the dataset can be classified as versions of each other.

## How to make a dataset

### Option 1: Youtube

The script main_dataset.py can be run to download an entire dataset from Youtube. To do that, it is
necessary to first create two csv files with the following structure:

```bash
link,name,genre
link,name,genre
...
```

One file should contain original songs and the other one covers. In this step, it is very important that the name of the covers
is the same as the originals. If you need to add extra information, you can do that in the genre.

Once this is done, simply run the code

```bash
python main_dataset.py
```

and give the paths to your csv files. This should download all the songs and save them in two folders in the directory of this repository.

### Option 2: Local samples

You can also build the dataset with local samples. In such cases, you just have to put all the originals and the covers in different folders,
making sure that the covers have the original name within their names, as in the following example:

- root
    - originals
        - me_haces_bien_jorge_drexler.wav
        - Martin_Oh_Can't_Leave_You_Behind.mp3
    - covers
        - drexler
            - me_haces_bien_jorge_drexler_cover_acustico.mp3
        - cover_Martin_Oh_Can't_Leave_You_Behind_metal_version.wav

Note that the inner folder organization does not really matter as long as the covers have the original name inside their names.

## How to run the algorithms

Once you have already set up the dataset, you can run the algorithms by

```bash
python main_tester.py
```

The scores should be plotted for both algorithms by default.

## How to display best/worst results

The function wrapper contained in the scripts main_tester.py and main_dataset.py can be accessed through the Jupyter notebook included in this repository. There we also added some functions to display the worst and best cases for all songs analyzed.

Note: In order to display the audio clip segments that libfmp identifies as similar, set the 'PLot' value as 1. 
