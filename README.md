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

In this repository two implementations of CSI algorithms are included so that the user can easily
apply them in local libraries.

## Requirements

In order to run the code in this repository it is necessary to have the following packages installed:

- numpy==1.26.3
- librosa==0.10.1
- matplotlib==3.8.2
- pydub==0.25.1
- pytube==15.0.0
- essentia==2.1b6.dev1110
- ipython==7.34.0

Additionally, pydub might ask the user to install FFmpeg codecs. In order to install them on Linux just run:

```bash
sudo apt install ffmpeg
```

## Contents

The repository contains one folder called preamble, containing all the code necessary to run the repo, and two 
ready to run scripts, that use the code inside the folder preamble. For a brief tutorial on the use of each script see
following sections.

## Tutorial :)

In this brief tutorial we will see how to use this repository to both create a dataset of songs downloaded from Youtube
and then we will run the CSI algorithms to evaluate if the songs in the dataset can be classified as versions of each other.

## How to make a dataset from Youtube



