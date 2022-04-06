# Image Classification
## Definition:-

The task of identifying what an image represents is called image classification. An image classification model is trained to recognize various classes of images.
An image classification model is fed images and their associated labels. Each label is the name of a distinct concept, or class, that the model will learn to recognize.
An image classification model can learn to predict whether new images belong to any of the classes it has been trained on.
Output is defined in terms of whether probabilities of image /label/classes are present or not.

## Software:-

1.  Google Colab:-
    

![](https://lh3.googleusercontent.com/kTBR_-DQWDtAzLbkivrynh1e0suQ9xY-UPFe7BGWyc2WmRf_zqiQx_oW8oZThhQdoz4UA-gQH6uLOuy6Mu8v_cacBSq_7sxuehBsiXcBZ3cvu4vwQQEf3F0-qJDnYvPo1tDYsqL7)

Colab is a free Jupyter notebook environment that runs entirely in the cloud. Most importantly, it does not require a setup and the notebooks that you create can be simultaneously edited by your team members - just the way you edit documents in Google Docs. Colab supports many popular machine learning libraries which can be easily loaded into your notebook

We will be preparing all of our models on colab only as if It is open source and it also gives us TPU as a processor to some extent that is what we need while doing training of the above models.

## FrameWorks:-
1. Tensorflow:-
![TensorFlow](https://www.tensorflow.org/images/tf_logo_social.png)

TensorFlow is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks.
In this project we are using tensor flow framework and its libraries for image classification.

## CIFAR Model:-
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.  
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.


### Workflow:-

This file shows how to classify images of cifar10 dataset. It creates an image classifier using a [`tf.keras.Sequential`]model, and loads data using [`tf.keras.utils.image_dataset_from_directory`].

### Basic machine learning workflow:-
1.  Examine and understand data:-
 > Import Tensor Flow and other libraries
 > Download and explore the dataset
 > Display images using pil.image.open command
 > Load data using a Keras utility
2.  Build an input pipeline
> Create a dataset
> Visualize the data
> Configure the dataset for performance
> Standardize the data
3.  Build the model
> Create the model
> Compile the model
> Check the summary
4.  Train the model
> Train the model by setting epochs
>  Visualize training results
5.  Test the model
> Visualize the testing result
6.  Improve the model and repeat the process.

The result of this is shown in the Google colab files.


