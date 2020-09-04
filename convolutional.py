from os import path
# import sys

import tensorflow as tf
from tensorflow import keras, train
from tensorflow.keras import datasets, layers, models, losses

import matplotlib.pyplot as plt
import numpy as np


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
# print(train_labels[0])
# sys.exit(0)

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# first parameters of Conv2D is number of filters to apply on eahc chunk of each image
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    # layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10)
])

# model.summary()

model.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# set checkpoint dir and path
checkpoint_path = 'convolutional_data/cp.ckpt'
checkpoint_dir = path.dirname(checkpoint_path)
# create callback to store checkpoint on model.fit (callback is passed to that method that is also the training model method)
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1    
)

# check if exist a checkpoint of train and use that checkpoint instead of re-train model (there is the possibility to save chackpoints base on period)
# SEE: Instead of realoding weight, model can be complitely saved, so a eventually wrapper can get the model without knowing the model architecture
# REF: https://www.tensorflow.org/tutorials/keras/save_and_load
lastest_cp = train.latest_checkpoint(checkpoint_dir)
history = None
if lastest_cp is not None:
    model.load_weights(lastest_cp)
    # history = model.fit(
    #     train_images,
    #     train_labels,
    #     epochs=2,
    #     validation_data=(test_images, test_labels),
    #     callbacks=[cp_callback]
    # )
else: 
    history = model.fit(
        train_images,
        train_labels,
        epochs=12,
        validation_data=(test_images, test_labels),
        callbacks=[cp_callback]
    )

    plt.plot(history.history['accuracy'], label='Train set accuracy')
    plt.plot(history.history['val_accuracy'], label='Test set accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Accuracy on test data set: {test_acc}')

probability_model = keras.Sequential([ model, layers.Softmax() ])
predictions = probability_model.predict(test_images)
# print(predictions)

plt.figure(figsize=(10,10))
start = 0
for i in range(start, start+25):
    plt.subplot(5,5,(i-start)+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f'{class_names[np.argmax(predictions[i])]} - {class_names[test_labels[i][0]]}')
plt.show()
