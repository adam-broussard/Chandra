# Chandra

## Description

Chandra is a tool for classifying chest x-rays plates as exhibiting signs of pneumonia or not.  

## The Problem

## Methodology

We have tested several machine-learning algorithms to arrive at our final model with varying levels of complexity.  The first is a _k_-Nearest Neighbors algorithm, which simply tries to classify an image based on how similar its particular pixel values are relative to the images in the training set.  The second is a convolutional neural network (CNN), which is capable of recognizing composite patterns and shapes in images.  The third is also a CNN, but it has been trained using the pretrained RESNET-152 v2 (with some added output dense layers) and hence, we refer to it as the Transfer Learning (TL) model.  RESNET-152 v2 is a residual deep network designed to have many layers without running aground of the vanishing gradient problem.

We use the F1 Score as our evaluation metric for two reasons.  The first is because we want as few true cases of pneumonia to be misclassified by our models as possible (which can be gauged by the Recall/Sensitivity).  On the other hand, if this is to be a tool that allows radiologists to work much more efficiently by not needing to look at x-rays the model confidently classifies as normal, we want as few normal x-rays to be classified as pneumonia as possible (which can be measured using Precision).  Because the F1 Score balances these two metrics (while effectively providing more weight to the lower of the two), it provides an optimal measure of our model performance.

$ F1 = 2\frac{Precision \times Recall}{Precision+Recall} $

## Installation

You can install from source by running:

```
git clone https://github.com/adam-broussard/GalaxyTeam
cd GalaxyTeam
python setup.py build
python setup.py install [--user]
```

## Usage
Once you have it installed, to do anything you'll need some data files which actually contain the X-ray scans. You can download them using:
```
>> from galaxyteam import dataset
>> dataset.download_dataset()
```
