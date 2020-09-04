import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


# print(tf.__version__)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(f'training set shape: {train_images.shape}, test set shape {test_images.shape}')
print(f'training set length: {len(train_images)}, test set length {len(test_images)}')

# # plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# # plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# create the sequential neural network
# flatten make list of 28x28 flat the images to 28x28=784 pixels
# dense create a layer o neurons (first dense -> 128 neurons, second dense -> 10 neursons, output, prediction and logits)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
# compailing the model created with sequential function
model.compile(
    optimizer='adam', 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
# train the model with given training data set
model.fit(train_images, train_labels, epochs=10)
# evaluate the model on test data set
(test_loss, test_acc) = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# before make a predition, attach an other layer (softmax) to convert logits to probabilities (for easier reading purpose)
probability_model = keras.Sequential([ model, keras.layers.Softmax() ])
# make the prediction. the result is an array of array. 
# the deepest array rapresents the probability of the image to be of a certain class name (array langth = 10)
# the external array wrap all prediction (prability based)
predictions = probability_model.predict(test_images)
print(predictions[0])


# plt.figure()
# plt.imshow(test_images[0], cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.xlabel(class_names[np.argmax(predictions[0])])
# plt.show()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(predictions[i])])
plt.show()
