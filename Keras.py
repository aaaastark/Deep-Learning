#--------------------------------------------------------------------------------------------------#
# Loading the MNIST DataSets. Data Exploration
from tensorflow.keras.datasets import mnist
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
print(X_train.shape)
#output: (60000, 28, 28)
print(Y_train.shape)
#output: (60000,)
print(X_test.shape)
#output: (10000,28,28)
print(Y_test.shape)
#output: (10000,)
#--------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------------#
# Loading the MNIST DataSets. Data Exploration and Data Visualizing Digits(When you connected to internet)
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

sns.set(font_scale=2)
index_name = np.random.choice(np.arange(len(X_train)), 24, replace= False)
figure, axes = plt.subplots(nrows= 4, ncols= 6, figsize= (16,9))

for item_name in zip(axes.ravel(), X_train[index_name], Y_train[index_name]):
  axes, image, target = item_name
  axes.imshow(image, cmap=plt.cm.gray_r)
  axes.set_xticks([])
  axes.set_yticks([])
  axes.set_tile(target)
plt.tight_layout()
#--------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------------#
# Creating the Neural Networks and adding Layers to the Networks.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
NN = Sequential()
#--------------Then adding a convolution layer--------------------------------#
NN.add(Conv2D(filters=64, kernel_size=(3,3), activation= 'relu', input_shape=(28,28,1)))
#--------------Then adding a pooling layer------------------------------------#
NN.add(MaxPooling2D(pool_size=(2,2)))
#--------------Then adding flattening the results-----------------------------#
NN.add(Flatten())
#-------Then adding a dense layer to reduce the number of features-------#
NN.add(Dense(units=128, activation='relu'))
#-------Then adding a dense layer to produce the final output-----------#
NN.add(Dense(units=10, activation='softmax'))
#-------------------Printing the Model's Summary---------------------------#
print(NN.summary()) # show all data conclusion
#output: Model: "sequential_9"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_8 (Conv2D)            (None, 26, 26, 64)        640
# _________________________________________________________________
# max_pooling2d_5 (MaxPooling2 (None, 13, 13, 64)        0
# _________________________________________________________________
# flatten_3 (Flatten)          (None, 10816)             0
# _________________________________________________________________
# dense_2 (Dense)              (None, 128)               1384576
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                1290
# =================================================================
# Total params: 1,386,506
# Trainable params: 1,386,506
# Non-trainable params: 0
# ___________________________________________
#--------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------------#
# Creating the Neural Networks and adding Layers to the Networks.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import plot_model
from IPython.display import Image
NN = Sequential()
#--------------Then adding a convolution layer--------------------------------#
NN.add(Conv2D(filters=64, kernel_size=(3,3), activation= 'relu', input_shape=(28,28,1)))
#--------------Then adding a pooling layer------------------------------------#
NN.add(MaxPooling2D(pool_size=(2,2)))
#--------------Then adding flattening the results-----------------------------#
NN.add(Flatten())
#-------Then adding a dense layer to reduce the number of features-------#
NN.add(Dense(units=128, activation='relu'))
#-------Then adding a dense layer to produce the final output-----------#
NN.add(Dense(units=10, activation='softmax'))
#-------------------Visualizing a Model's Structure---------------------------#
plot_model(NN, to_file='convnet.png',show_shapes=True, show_layer_names=True)
Image(filename='convnet.png')  # picture name is convnet.png also bydefault bulit
#output: Online serach and then show model structure of pattern given
#--------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------------#
# Creating the Neural Networks and adding Layers to the Networks.
# Configuring Keras to Write the TensorBoard Log Files
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from tensorflow.keras.callbacks import TensorBoard
import time
NN = Sequential()
#--------------Then adding a convolution layer--------------------------------#
NN.add(Conv2D(filters=64, kernel_size=(3,3), activation= 'relu', input_shape=(28,28,1)))
#--------------Then adding a pooling layer------------------------------------#
NN.add(MaxPooling2D(pool_size=(2,2)))
#--------------Then adding flattening the results-----------------------------#
NN.add(Flatten())
#-------Then adding a dense layer to reduce the number of features-------#
NN.add(Dense(units=128, activation='relu'))
#-------Then adding a dense layer to produce the final output-----------#
NN.add(Dense(units=10, activation='softmax'))
#-------------------Then compiling the Model---------------------------#
NN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#-------------------Training and Evaluating the Model---------------------------#
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
#------------------TensorBoard Model creating --------------------------------#
tesn_bord = TensorBoard(log_dir = f'./logs/mnist{time.time()}',histogram_freq=1, write_graph=True)
#--------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------------#
# Creating the Neural Networks and adding Layers to the Networks.
# Recurrent Neural Networks for Sequences. Sentiment analysis with the IMDb Dataset
from tensorflow.keras.datasets import imdb
(X_train,Y_train),(X_test,Y_test) = imdb.load_data(num_words= 10000)
print(X_train.shape)
#output: (25000,)
print(X_test.shape)
#output: (25000,)
print(Y_train.shape)
#output: (25000,)
print(Y_test.shape)
#output: (25000,)
#--------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------------#
# Creating the Neural Networks and adding Layers to the Networks.
# Recurrent Neural Networks for Sequences. Sentiment analysis with the IMDb Dataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(X_train,Y_train),(X_test,Y_test) = imdb.load_data(num_words= 10000)
x_train = pad_sequences(X_train, maxlen= 200)
print(x_train.shape)
#output: (25000, 200)
x_test = pad_sequences(X_test, maxlen= 200)
print(x_test.shape)
#output: (25000, 200)

#--------------------------------------------------------------------------------------------------#
