# Project Name :  Melanoma Skin Cancer Detection using CNN models
> To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

# The data set contains the following diseases:

1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion

## Table of Contents
## General Information for Project Pipeline
- Provide general information about your project here.
    To classify skin cancer using skin lesion images with higher accuracy and results, we have build a multiclass classification model using a custom convolutional neural network in TensorFlow.
    1. Data Reading/Data Understanding → Defining the path for train and test images
    2. Dataset Creation→ Create train & validation dataset from the train directory with a batch size of 32. Also, make sure you resize your images to 180*180.
    3. Dataset visualisation → Create a code to visualize one instance of all the nine classes present in the dataset 
    4. Model Building & training : 
        -   Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to  normalize pixel values between (0,1).
        -   Choose an appropriate optimiser and loss function for model training
        -   Train the model for ~20 epochs
        -   Check if there is any evidence of model overfit or underfit.
    5.  Chose an appropriate data augmentation strategy to resolve underfitting/overfitting 
    6.  Model Building & training on the augmented data :
        -   Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
        -   Choose an appropriate optimiser and loss function for model training
        -   Train the model for ~20 epochs
        -   Check if the earlier issue is resolved or not?
    7. Class distribution: Examine the current class distribution in the training dataset 
        -   Which class has the least number of samples?
        -   Which classes dominate the data in terms of the proportionate number of samples?
    8. Handling class imbalances: Rectify class imbalances present in the training dataset with Augmentor library.
    9. Model Building & training on the balanced data :
        -   Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
        -   Choose an appropriate optimiser and loss function for model training
        -   Train the model for ~30 epochs
        -   Check if the issues are resolved or not?

# What is the background of your project?

The goal of the project is to reduce the percentage of deaths caused by skin cancer. Main idea that drives this project is to use the advanced technology of Convolutional Neural Networks to identify the risk of cancer on the basis of the skin lesion images.

# What is the business probem that your project is trying to solve?
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

# What is the dataset that is being used?
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

## Models used and Conclusions
Steps to get the final CNN Model
1. Model 1 : Base Model
    We started with the following Base Model that consisted of:
    - Rescaling Layer
    - 2D Convolution Layer
    - Max Pooling Layer
    - 2D Convolution Layer
    - Max Pooling Layer
    - 2D Convolution Layer
    - Max Pooling Layer
    - Flattening layer
    - Dense Layer
    - Dense Layer
    Epochs = 20 and batch size=32
    Results : Training Accuracy - 92.2% Testing Accuracy - 54%
    Remarks : The model seems to overfits hence we can try implementing Regularisation. The main purpose of using dropouts is to reduce overfitting. Sometimes, a model trains on the training data set and its weights and biases converge to very specific values, values that are ideal for only the training data set. Adding a dropout layer to the neural network helps to break that specific combination of weights and biases.
2. Model 2 : Base Model + Dropout Layer 
    We changed the above model to include a Dropput layer before every Convolution Layer with Dropout ratio .2
    The final activation function was changed to softmax.
    The new model consisted of:
    - Rescaling Layer
    - 2D Convolution Layer
    - Max Pooling Layer
    - Drop Out Layer
    - 2D Convolution Layer
    - Max Pooling Layer
    - Drop Out Layer
    - 2D Convolution Layer
    - Max Pooling Layer
    - Drop Out Layer
    - Flattening layer
    - Dense Layer
    - Dense Layer
    Results : Training Accuracy - 74% Testing Accuracy - 47%
    Remarks : We see that it reduced the gap between training and validation accuracy but the validation accuracy was still low, hence we checked if using augmentation functions at the Keras Pre-processing layers.
3.  Model 3 : Model 2 + Image augmentation Real Time
    The new model consisted of:
    - Preprocessing layer for Image augmentation using functions (Real time)
    - Same as Model 2
     Results : Training Accuracy - 45.59% Testing Accuracy - 49.44%
     Remarks: That made huge difference, the model does not overfit anymore but the training accuracy is gone very low meaning it needs more data to train.
4.  Model 4: Model 3 + Augmentation using Image Data Generator (Creating different transformed dataset images)
    Image augmentation is a technique of preprocessing image data. It involves applying transformations (rotation, cropping, shearing, zooming etc.) on our existing images and adding these images to our database.
    The new model consisted of:
    1. Run Image Data generator on the training and validation dataset to create new transformed dataset
    2. Run the Model 3 on the above datasets
    Results : Training Accuracy - 19.65% Testing Accuracy - 13.56%
    Remarks: After visualizing the dataset, it seems that both the training and validation accuracy is gone too low, which isnt a good sign and means the data is underfitting. Hence we will check for class imbalance issue now.
5. Final model 5: Model 4 preceded by class imbalance rectification
    To rectify the class imbalance issue, we used a python package known as Augmentor to add more samples across all classes so that none of the classes have very few samples. Here we added 500 images to all the classes.
    Augmentor stored the augmented images in the output sub-directory of each of the sub-directories of skin cancer types.. Lets take a look at total count of augmented images.
    The new model consisted of same as Model 4 with more number of images added to the training dataset.
    Results : Training Accuracy : 98% and Testing Accuracy: 84%

    Hence , this model is better and shows good result (Not underfitting and not overfitting)

NOTE : The ImageDataGenerator accepts the original data, randomly transforms it, and returns only the new, transformed data. We call this “in-place” and “on-the-fly” data augmentation because this augmentation is done at training time (i.e., we are not generating these examples ahead of time/prior to training).

## Result Analysis:
- Problem of overfitting was reduced by adding dropout layers but not to a huge extent.
- Problem of overfitting was greatly reduced by real time augmentation. But the accuracy both training and validation went very low.
- To increase the training accuracy, Image data generator was used for Augmentation which didnt work.
- After checking if the data was imbalanced, it was found that there was class imbalance in the data and hence Augmentor was used to increase
the training dataset size for classes having less samples. This along with batch normalisation layers gave very good result and the training accuracy was seen as 98% while the validation accuracy was seen 84%

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- TensorFlow -  2.12.0
- Keras - 2.12.0
- matplotlib - 3.7.1
- numpy - 1.22.4
- pandas - 1.5.3

## Acknowledgements
- References if any...
    -   Upgrad Video Tutorials
    -   Live session on upgrad - Melanoma Assignment discussion
    -   For augmentation - https://pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
    -   https://www.tensorflow.org/api_docs/python/tf/keras/layers/

## Contact
Created by 
Vidhi Surana Datar [vidhi.s9@gmail.com] - feel free to contact me!
Yudishthir(dadhichboy2000@gmail.com)
Namrata (namrata.nanditina@gmail.com)


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->