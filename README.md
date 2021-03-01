# Convolutional Neural Network for energy disaggregation of the Non-Intrusive Load Monitoring REDD Data Set

This repository is the implementation of convolutional neural network (ConvNet) for energy disaggregation of the REDD data Set available at http://redd.csail.mit.edu. **This work is part of a final project for the course AM207 at Harvard University during Spring 2017**.  Other algorithms (Combinatorial Optimization method and a Factorial Hidden Markov Model) were tested on the data set using the NILMTK library (http://nilmtk.github.io). **The ConvNet implementation outperformed the other methods implemented in the NILMTK library (http://nilmtk.github.io).**

Using Non-Intrusive Load Monitoring data from one house not seen during training we inverted for the activations of the fridge and the microwave. The data set for the unseen house is approximately 6 months long. The long time series is splitted into windows of 85 seconds. We use binary classification as a metric. True positive means that we predict activation during a window while there is indeed an activation of the target appliance. The results of the various methods for different metrics is show here:

![](./figures/Scores.png)



# The Redd Data Set

In the context of this project we make use of the REDD data set, a public data set for energy disaggregation research. The REDD dataset is not readily accessible since access can only be granted by the authors at MIT. While REDD, is what made this project possible, it cannot be directly fed into the disaggregation algorithms developed/borrowed in this project. A data Pipeline is necessary to preprocess and feed the data for both training as well as testing purposes. While the training and disaggregation steps are significantly different in all of the methods that we use, the remaining steps are the same and are outlined in this figure.

![](./figures/Qualitative.png)

The redd data set consist of aggregated power from 6 houses with various sampling rates as well as  the power of a set of appliances per house. The recording is unfortunately not continuous in time and does not span the same time period for the all the houses. We downsample the data in order to align the appliances and the main meters time series. The preprocessing is specific to each algorithm used for prediction and is detailed later in this report. However the work flow common to each of the algorithm is shown here

![](./figures/NILM_Data_Pipeline.png)

# Convolutional Neural Network (ConvNet)

The implementation of the method presented in this section can be found in the notebook https://github.com/tperol/am207-NILM-project/blob/master/Report_convnet.ipynb. However the main codes are available in this separate repository (https://github.com/tperol/neuralnilm) to keep this final repository clean. Most of the preprocessing code are borrowed from Jack Kelly repository that was forked (https://github.com/JackKelly/neuralnilm). However the implementation of the python generator for the data augmentation on CPU, the ConvNet implementation (trained on GPU) and post processing for the metrics are our own implementation.

## 1- ConvNet introduction

Convolutional Neural Networks are similar to ordinary Neural Networks (multi-layer perceptrons). Each neuron receive an input, perform a dot product with its weights and follow this with a non-linearity (here we only use ReLu). The whole network has a loss function that is here the Root Mean Square (RMS) error (details later). The network implements the 'rectangle method'. From the input sequence we invert for the start time, the end time and the average power of only one appliance 

![](./figures/convnet_architecture.png)


Convolutional neural networks have revolutionized computer vision. From an image the convolutional layer learns through its weights low level features. In the case of an image the features detectors (filters) would be: horizontal lines, blobs etc. These filters are built using a small receptive field and share weights across the entire input, which makes them translation invariant. Similarly, in the case of time series, the filters extract low level feature in the time series. By experimenting we found that only using 16 of these filters gives a good predictive power to the ConvNet. This convolutional layer is then flatten and we use 2 hidden layers of 1024 and 512 neurons with ReLu activation function before the output layer of 3 neurons (start time, end time and average power).



## 2- Data pipeline

### 2.1 - Selecting appliances

We train each neural network per appliance. This is different from the CO and FHMM methods. For this report we only try to invert for the activation of the fridge and the microwave in the aggregated data. This two appliances have very different activation signatures (see Figure !!!).

### 2..2 - Selecting time sequences

We downsampled  the main meters and the submeters to 6 samples per seconds in order to have the aggregated and the submeter sequences properly aligned. We throw away any activation shorter than some threshold duration to avoid spurious spikes. For each sequence we use 512 samples (about 85 seconds of recording).

### 2.3 - Selecting houses

We choose to train the algorithm on house 1,2,3 and 6 and test the data on house 5.

### 2.4 - Dealing with unbalanced data: selecting aggregated data windows

We first extract using NILMTK libraries (http://nilmtk.github.io) the target appliance (fridge or microwave) activations in the time series. We concatenate the times series from house 1,2,3, and 6 for the training set and will test on house 5. We feed to our neural network algorithm (detailed later) balanced mini-batches of data sequences of aggregated data in which the fridge is activated and sequences in which it is not activated. This is a way to deal with unbalanced data -- there are more sequences where the fridge is not activated than sequences with the fridge activated. Most of the data pipeline used is borrowed from https://github.com/JackKelly/neuralnilm.

### 2.5 - Synthetic aggregated data

We use the method from Jack Kelly to create synthetic data (http://arxiv.org/abs/1507.06594)

We ran neural networks with and without synthetic aggregated data. We found that synthetic data acts as a regulizer, it improves the scores on useen house.

## 3 - Standardisation of the input data (aggregated data)

A typical step in the data pipeline of neural network is the standardization of data. For each sequences of 512 samples (= 85 seconds) we substract the mean to center the sequence. Furthermore every input sequence is divided by the standard deviation of a random sample in the training set. In this case we cannot divide each sequence by its own standard deviation because it would delete information about the scale of the signal.



## 4 - Output data (start time, end time and average power)

The output of the neural network is 3 neurons: start time, end time and average power. We rescale the time to the interval [0,1]. Therefore if the fridge starts in the middle of the input sequences the output of the first neuron is 0.5. If its stops after the end of the input window the ouput of the second neuron is set to 1. The third neuron is the average power during the activation period. Of course this is set to 0 when it is not activated during the input sequence. We also post process the data by setting any start time lower than 0 to 0 and end time higher than 1 to 1. We create a average power threshold set to 0.1 that indicates if the appliance was active or not (under the threshold the appliance is considered off, above it is considered on).
Here we show as an example the input data and the ouput calculated by a trained network. We compare this with the real appliance activation.

![](./figures/output_example.png)



As we can see here the network does a very good job at detecting the activation of the fridge. The red line is the aggregated data. In the flat region it would be impossible to detect the activation of the fridge with human eye. We would tend to put an activation in the step region. However the network does a very accurate prediction of the activation of the fridge !\



Because of the dimension of the ouput we choose classification score metrics. When the starting time and the ending time are both 0 we call this a negative. We also call negative if the power average is lower than threshold. Otherwise this is positive (the appliance is activated). We call TP true positive, TN true negative, FP false positive and FN false negative. The various metrics/scores used in this study are
$$
recall = \frac{TP}{TP + FN} \\
 precision = \frac{TP}{TP + FP} \\
 F1 = 2 * \frac{precision* recall}{precision + recall} \\
 accuracy = \frac{TP + TN}{P + N}
$$
where P is the number of positives and N the number of negatives.



## 5- Implementation strategy for real time data augmentation

While the neural network runs an NVIDIA GeForce GT 750M (GPU) we maintain the CPU busy doing the data augmentation in real time (load aggregated data, create the synthetic data, preprocess the mini-batch to be fed to the neural network). For this we create a python generator that creates a queue of 50 mini-batch and feed them successively to the GPU for training.

## 6 -Network architecture

We use a convolutional neural network (ConvNet) to take advantage of the translation invariance. We want the ConvNet to recognize target appliance activation anywhere in the sequence. For this project we have tried multiple architecture that are reported later on. These architecture all have a first convolutional layer of filter size 3 and stride 1. We have played with both the filter size and the number of output filters on the first layer. We have found that 16 filters is a reasonable number -- increasing the number of filters in the first layer did not improve significantly the scores.
The best neural network we found consist of

* Input layer: one channel and lenght of 512 samples


* 1D convolutional layer (filter size = 3, stride = 1 , number of filters = 16, activation function = relu, border mode = valid, weight initialization = normal distribution)


* Fully connected layer (N = 1024, activation function = relu, weight initialization = normal distribution)


* Fully connected layer (N = 512, activation function = relu, weight initialization = normal distribution)


* Fully connected layer (N= 3, activation function = relu)

The ouput has 3 neurons activated by a relu activation function since the output cannot be negative. We have tried other networks that are reported later in this notebook. However this is the layout of the best one we found

## 7 - Loss function

Since the output neurons spans the real axis there is no other choice than using a L2 norm for the loss function. This is (predicted start time - true start time)2 + (predicted end time - true end time)2 + (predicted average power - true average power)2. The total loss function is the sum of the loss function for all the sample in a mini-batch.

## 8 - Optimizer

We found by experimenting that the best optimizer is Adam (http://arxiv.org/pdf/1412.6980v8.pdf)



For the implementation of the ConvNet we use Keras (http://keras.io). This is a library implemented on top of Theano and Tensorflow (in this case we use Theano to take advantage of the GPU, GPU training is not yet available on Mac OS using TensorFlow).



