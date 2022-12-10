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

During training the data splited between 80% training dataset and 20% validation dataset

## 2- Core Model

The model used was LSTM for character level and below is the architecture of the model

```bash
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_12 (Bidirectio (None, 20, 1024)          2224128   
_________________________________________________________________
dropout_52 (Dropout)         (None, 20, 1024)          0         
_________________________________________________________________
bidirectional_13 (Bidirectio (None, 1024)              6295552   
_________________________________________________________________
dropout_53 (Dropout)         (None, 1024)              0         
_________________________________________________________________
dense_24 (Dense)             (None, 2)                 2050      
_________________________________________________________________
activation_22 (Activation)   (None, 2)                 0         
=================================================================
Total params: 8,521,730
Trainable params: 8,521,730
Non-trainable params: 0
_________________________________________________________________
```

The model was used two times, one time to classify if the name is real or fake and other time to classify the gender of the name to make sure that name is following the basic structure of the full name 

The weights of the model are located at `models/`

### Model Accuracy and Optimal threshold

The metric used was the accuracy for both models as the data is balanced between both classes.

Youden's J statistic was used  to obtain the optimal probability threshold and this method gives equal weights to both false positives and false negatives.

`J = Sensitivity + Specificity - 1`
                  
1- Real and Fake name Classifcation
  
  Training accuracy: 76% 
  
  Testing accuracy:  71%
  
  Optimal threshold= 0.55
 
2- Gender Classifcation

  Training accuracy: 85%
  
  Testing accuracy: 83%
  
  Optimal threshold= 0.58
  
## 3- Interface
Flask API was used to deploy the model on a simple web server. Using the saved models weights to create prediction and show the output for the user and the execution time.
<img width="655" alt="Screenshot 2022-12-11 at 2 29 09 AM" src="https://user-images.githubusercontent.com/54632431/206877672-54435dd2-4ea2-402b-bcf4-55b85d14c6f6.png">

