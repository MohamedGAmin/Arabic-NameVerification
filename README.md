# Arabic Name Verification

<a href="https://hub.docker.com/r/mohamedgamin/digified"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Overview
This is an Arabic name verification project where Character-Level LSTM for real and fake name classifcation was used. 

The name given should be first, middle and last names written in arabic letters seprated by spaces. Then the model will detect if this name is a real name with high confidence, a real name with low confidence or a fake name. The model also consider the basic structure of the full name, which is middle and last name shouln't be a feminine name. So another gender classification model was used to make sure that the middle and the last names are masculine.

The project consists of the following parts:

1- Data Generation

2- Core Model

3- Inferance

4- Containerization

## 1- Data Generation
The raw data located `dataset/` which contains real arabic names for males and females. It was challanging to find much open source arabic names datasets. 

Some of the data was taken from this repository https://github.com/zakahmad/ArabicNameGenderFinder

### Data preprocessing

Three different ways were used to generate fake names out of the real names in the dataset located in `data_generation.py`:

1- Removing random letter

2- Swap any random 2 letters with each other

3- Repeat a random letter at a random position in the name.

### Dataset

The dataset after preprocessing that is used for training the model consists of:

1- 1971 females and 1877 males fake names

2- 1971 females and 1877 males real names

Total of 3848 fake and 3848 real names located in `generated_dataset/`

## 2- Core Model

The model used was LSTM for character level and below is the architecture of the model




