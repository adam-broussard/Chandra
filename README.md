# Chandra

## Description

Chandra is a tool for classifying chest x-rays plates as exhibiting signs of pneumonia or not.  


<!-- MarkdownTOC autolink="true" autoanchor="true" -->

- [The Problem](#the-problem)
- [Methodology](#methodology)
    - [CNN Model Architecture and Training](#cnn-model-architecture-and-training)
    - [TL Model Architecture and Training](#tl-model-architecture-and-training)
- [Results](#results)
- [Applications](#applications)
- [Using the Code](#using-the-code)
    - [Installation](#installation)
    - [Usage](#usage)

<!-- /MarkdownTOC -->



<a id="the-problem"></a>
## The Problem

<a id="methodology"></a>
## Methodology

We have tested several machine-learning algorithms to arrive at our final model with varying levels of complexity.  The first is a _k_-Nearest Neighbors algorithm, which simply tries to classify an image based on how similar its particular pixel values are relative to the images in the training set.  The second is a convolutional neural network (CNN), which is capable of recognizing composite patterns and shapes in images.  The third is also a CNN, but it has been trained using the pretrained RESNET-152 v2 (with some added output dense layers) and hence, we refer to it as the Transfer Learning (TL) model.  RESNET-152 v2 is a residual deep network designed to have many layers without running aground of the vanishing gradient problem.

We use the F1 Score as our evaluation metric for two reasons.  The first is because we want as few true cases of pneumonia to be misclassified by our models as possible (which can be gauged by the Recall/Sensitivity).  On the other hand, if this is to be a tool that allows radiologists to work much more efficiently by not needing to look at x-rays the model confidently classifies as normal, we want as few normal x-rays to be classified as pneumonia as possible (which can be measured using Precision).  Because the F1 Score balances these two metrics (while effectively providing more weight to the lower of the two), it provides an optimal measure of our model performance.

![image](https://user-images.githubusercontent.com/33520634/172018513-edc20ed9-869c-41d4-a235-2ab3a981d295.png)

<a id="cnn-model-architecture-and-training"></a>
### CNN Model Architecture and Training

<a id="tl-model-architecture-and-training"></a>
### TL Model Architecture and Training

<a id="results"></a>
## Results

| Model  | F1 Score |
|:------:|---------:|
| _k_-NN | 0.947    |
| CNN    | 0.970    |
| TL     | 0.000    |

In the table above, we show the F1 Score for each of our models after training.  The CNN shows the best F1 score and, as a result, we choose it as our best-fit model.  We also examine the confusion matrices for each of the models below, where we find that the CNN model particularly shines at avoiding misclassifications of normal x-rays as containing pneumonia.

![image](https://user-images.githubusercontent.com/33520634/172019831-d0b753b9-6f54-43b9-8e1b-80d8b0475123.png)


Because our CNN model is the chosen best-fit model, we then make a final estimate of its performance on data it has never seen before by applying it to the test set, yielding an F1 Score of **0.966** and the following final confusion matrix:

![image](https://user-images.githubusercontent.com/33520634/172019861-c2235abd-9124-452c-b26e-59b3be210a40.png)


<a id="applications"></a>
## Applications

<a id="using-the-code"></a>
## Using the Code

<a id="installation"></a>
### Installation

You can install from source by running:

```
git clone https://github.com/adam-broussard/GalaxyTeam
cd GalaxyTeam
python setup.py build
python setup.py install [--user]
```

<a id="usage"></a>
### Usage
Once you have it installed, to do anything you'll need some data files which actually contain the X-ray scans. You can download them using:
```
>> from galaxyteam import dataset
>> dataset.download_dataset()
```
