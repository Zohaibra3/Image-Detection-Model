# Image-Detection-Model
This model has been trained on a data set of images and it will find broken and flat surfaces in any given data to it.

The model is a Convolutional Neural Network (CNN) with the following architecture:

Conv2D Layer (32 filters, kernel size 3x3, ReLU activation)
MaxPooling2D Layer (2x2 pool size)
Conv2D Layer (64 filters, kernel size 3x3, ReLU activation)
MaxPooling2D Layer (2x2 pool size)
Flatten Layer
Dense Layer (128 neurons, ReLU activation)
Dropout Layer (50% dropout)
Dense Output Layer (1 neuron, sigmoid activation for binary classification)
