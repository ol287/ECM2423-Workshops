# Importing necessary libraries and functions
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras import utils

# Setting a seed for random number generation to ensure reproducibility
np.random.seed(1671)

# Defining constants for the model
NB_EPOCH = 200           # Number of epochs (iterations over the entire dataset)
BATCH_SIZE = 128         # Batch size for training
VERBOSE = 1              # Verbosity mode for printing during training
NB_CLASSES = 10          # Number of classes (digits 0-9)
OPTIMIZER = SGD()        # Optimizer for training the model
VALIDATION_SPLIT = 0.2   # Fraction of training data to use for validation

# Loading the MNIST dataset and splitting it into training and testing sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshaping the input data to flatten it into a 1D array
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)

# Converting the data type of input features to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizing the input features by scaling them to a range between 0 and 1
X_train /= 255
X_test /= 255

# Printing the number of samples in the training and testing sets
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Converting class vectors to binary class matrices (one-hot encoding)
Y_train = utils.to_categorical(y_train, NB_CLASSES)
Y_test = utils.to_categorical(y_test, NB_CLASSES)

# Defining the neural network model architecture using the Sequential API
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))  # Adding a dense layer with 10 units (one for each class)
model.add(Activation('softmax'))                       # Adding a softmax activation function for multiclass classification
model.summary()                                        # Printing a summary of the model architecture

# Compiling the model by specifying the loss function, optimizer, and evaluation metric
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# Training the model on the training data and validating it using a portion of the training data
history = model.fit(X_train, Y_train,
                    batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Evaluating the trained model on the test data
score = model.evaluate(X_test, Y_test)

# Printing the test score (loss) and test accuracy of the model
print("Test score:", score[0])
print('Test accuracy:', score[1])
