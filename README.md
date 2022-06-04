# Chandra

## Description

Chandra is a tool for classifying chest x-rays plates as exhibiting signs of pneumonia or not.  


<!-- MarkdownTOC autolink="true" autoanchor="true" -->

- [The Problem](#the-problem)
- [Methodology](#methodology)
    - [CNN Model Architecture and Training](#cnn-model-architecture-and-training)
    - [TL Model Architecture and Training](#tl-model-architecture-and-training)
- [Results](#results)
- [Using the Code](#using-the-code)
    - [Installation](#installation)
    - [Usage](#usage)

<!-- /MarkdownTOC -->



<a id="the-problem"></a>
## The Problem

Chest x-ray interpretation is a fundamental aspect of a radiologist’s training, however radiologists can undergo up to 13 years of training before being fully certified in a particular specialization.  In fact, Merritt Hawkins, the largest physician search and consulting firm in the United States, reports that radiology was the 10th most recruited specialty for physician searches in 2016, up from 19th the year before.  Further, the American College of Radiology tracks radiologist hiring trends, and saw a 14% increase in radiologist hires from 2016 to 2017.  This drastic increase in demand for trained radiologists leaves the 2025 predicted shortage of Radiologists in the U.S. in the tens of thousands, and has resulted in a commensurate increase in telehealth radiology due to the rising costs of employment, even after accounting for COVID-19 shutdowns.  The decentralization of patient care can lead to misdiagnosis of illnesses as well as delays in patient care.

<a id="methodology"></a>
## Methodology

We have tested several machine-learning algorithms to arrive at our final model with varying levels of complexity.  The first is a _k_-Nearest Neighbors algorithm, which simply tries to classify an image based on how similar its particular pixel values are relative to the images in the training set.  The second is a convolutional neural network (CNN), which is capable of recognizing composite patterns and shapes in images.  The third is also a CNN, but it has been trained using the pretrained RESNET-152 v2 (with some added output dense layers) and hence, we refer to it as the Transfer Learning (TL) model.  RESNET-152 v2 is a residual deep network designed to have many layers without running aground of the vanishing gradient problem.

We use the F1 Score as our evaluation metric for two reasons.  The first is because we want as few true cases of pneumonia to be misclassified by our models as possible (which can be gauged by the Recall/Sensitivity).  On the other hand, if this is to be a tool that allows radiologists to work much more efficiently by not needing to look at x-rays the model confidently classifies as normal, we want as few normal x-rays to be classified as pneumonia as possible (which can be measured using Precision).  Because the F1 Score balances these two metrics (while effectively providing more weight to the lower of the two), it provides an optimal measure of our model performance.

![image](https://user-images.githubusercontent.com/33520634/172018513-edc20ed9-869c-41d4-a235-2ab3a981d295.png)

<a id="cnn-model-architecture-and-training"></a>
### CNN Model Architecture and Training

Our CNN model consists of the following layers:

 - 2D Convolution (64 3x3 filters, ReLU activation)
 - 2D Max Pooling (4x4)
 - 2D Convolution (64 3x3 filters, ReLU activation)
 - 2D Max Pooliing (4x4)
 - Dense (64 neurons, ReLU activation)
 - Dropout (Probability=0.5)
 - Dense (16, ReLU activation)
 - Dropout (Probability=0.5)
 - Dense (1 neuron, Sigmoid activation)

The Convolution and Dense layers are the learning layers of the neural network.  The Max Pooling layers apply some regularization and reduce the model complexity for computational purposes, while the Dropout layers provide additional regularization to prevent overfitting.  We use early stopping on the validation dataset's F1 Score while training in addition to an exponential decay in learning rate corresponding to a 10% decrease with every epoch.


<a id="tl-model-architecture-and-training"></a>
### TL Model Architecture and Training

Our Transfer Learning model relies on the pretrained residual deep network RESNET-152 v2.  A residual deep network is composed of stacked "residual units", which are groupings of layers that are connected to one another, along with skip connections between residual units.  This architecture allows for a much larger number of layers without running aground of the vanishing gradient problem.  This model has been trained on the CIFAR-10 and CIFAR-100 image classification tasks and as a result, it is already capable of recognizing many common elements in images.  We make use of transfer learning in our TL model by freezing the weights of RESNET-152v2 and adding several trainable Dense layers after its outputs.  In total, our TL architecture consists of the following elements:

 - RESNET-152 v2 (Frozen)
 - Global Average Pooling
 - Dense (128 neurons, ReLU activation)
 - Dropout (Probability=0.1)
 - Dense (1 neuron, Sigmoid activation)

The global average pooling layer serves to greatly reduce the amount of information passed ot the following Dense layer and, as before, the Dropout layer provides regularization to prevent overfitting.  We employ early stopping on this model's validation F1 Score as well, as well as a plateau detector that reduces the learning rate when the F1 Score gains begin to decrease.


<a id="results"></a>
## Results

| Model  | F1 Score |
|:------:|---------:|
| _k_-NN | 0.944    |
| CNN    | 0.964    |
| TL     | 0.915    |

In the table above, we show the F1 Score for each of our models after training.  The CNN shows the best F1 score and, as a result, we choose it as our best-fit model.  We also examine the confusion matrices for each of the models below, where we find that the CNN model particularly shines at avoiding misclassifications of normal x-rays as containing pneumonia.

![image](https://user-images.githubusercontent.com/33520634/172019831-d0b753b9-6f54-43b9-8e1b-80d8b0475123.png)


Because our CNN model is the chosen best-fit model, we then make a final estimate of its performance on data it has never seen before by applying it to the test set, yielding an F1 Score of **0.966** and the following final confusion matrix:

![image](https://user-images.githubusercontent.com/33520634/172019861-c2235abd-9124-452c-b26e-59b3be210a40.png)

<a id="using-the-code"></a>
## Using the Code

If you would like to run our best fit model, just follow the instructions below:

<a id="installation"></a>
### Installation

You can install from source by running:

```
git clone git@github.com:adam-broussard/Chandra.git
cd Chandra
pip install .
```

<a id="usage"></a>
### Usage

Once the package has installed, if you would like to play around with retraining the models or use the same chest x-ray dataset, you'll want to start by downloading the files:
```
from chandra import dataset
dataset.download_dataset()
```

Finally, if you would simply like to read in our best fit model and apply it to new data, you can use:

```
from tensorflow.keras.models import load_model
from chandra.metrics import F1_Score
cnn = load_model('./notebooks/CNN_v1', custom_objects={'F1_Score':F1_Score}))
```
Happy classifying!
