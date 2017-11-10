# Multiclass-classifier
Repo for a simple multiclass classifier in python using tensorflow. In this program, a dataset is read from a txt file and we apply a neural network algorithm with cross entropy to try to obtain a prevision.

## Requirements
To run this program the Numpy and Tensorflow library are required. 
To install Numpy, you can download it from the official website here:
* [Numpy](http://www.numpy.org/)

or you can use 

```
pip install numpy
```

To install Tensorflow, follow the instruction in the official website:
* [Tensorflow installing](https://www.tensorflow.org/install/)

## Binary classifier
This repository contains also one example of a binary classifier, that can be found in the file TA classifier.py. Differently from TA classifier tensorflow.py, this file does not contain any tensorflow command, so you just need Numpy to run it. 

## Dataset choice
The dataset I choose to show this simple multiclass classifier is a Teacher Assistants evaluations database. The database is open to the public, and it has been provided by the following source:

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

The database can be found online here:
* [Teaching Assistant Evaluation Data Set](https://archive.ics.uci.edu/ml/datasets/teaching+assistant+evaluation)

## Modification ##
Feel free to modify the structure and the main parameters of this linear classifier. If you want to use another database and you should have the file in the same folder for this to work. Also remember to change the dimension of the matrices consequently. If you use dataset of bigger sizes, I suggest to divide the dataset in bashes of around 100 elements.
