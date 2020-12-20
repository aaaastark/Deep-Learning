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
