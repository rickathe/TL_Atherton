# TeuscherLab Multiplicaion Neural Network

Repo for a neural network built from numpy using Tariq Rasid's template for the base train/query design.    

Network can plot training/validation error over epochs or create bar graphs displaying the minimum mean error of a particular set of hyper-parameters for optimization.

To Run:
- runEpochs.py is the master file which the network is run from. Parameters are set here.
- nn#layer.py is the training/query network, replace # with the number of hidden layers desired.

Future updates:
- Remove hard coding of hidden layers, subsume into one neural network training/query file.
