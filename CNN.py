
import numpy as np 
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook; tqdm.pandas()
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import csv
import random
from shutil import copyfile
from sklearn.model_selection import train_test_split


import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Set seed for reproducability
seed = 1234
np.random.seed(seed)


# Directory where images were saved
output_dir = "heatmap_data"

# Get all image file paths
image_paths = [os.path.join(output_dir, file) for file in os.listdir(output_dir) if file.endswith('.png')]

# Optional: Print the number of images
print(f"Total images: {len(image_paths)}")

# Extract labels from filenames (if applicable)
labels = []
for file in image_paths:
    # Assuming labels can be derived from the filename, e.g., axis_label
    # Modify this based on your filename structure
    #/Phase_diff_3.0_-1.0_heatmap.png
    angle = file.split('_')[2]  # Example: Extract "axis_label" from "axis_label_dist_angle_heatmap.png"
    dist = file.split('_')[3]  # Example: Extract "dist" from "axis_label_dist_angle_heatmap.png"
    labels.append((angle, dist))



train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=123)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.15, random_state=123)
     
input_shape= (20,345,1) # for MFCCs
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=input_shape))
model.add(layers.BatchNormalization()) 
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.BatchNormalization())  
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(128, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))
model.summary()
     

checkpoint_path = '/content/drive/MyDrive/models'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True)


learning_rate = 0.00002
optimizer = optimizers.Adam(learning_rate=learning_rate)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
     

batch_size = 80
epochs = 45
history = model.fit(train_paths, train_labels, epochs=epochs, batch_size = batch_size,
                    validation_data=(val_paths, val_labels), shuffle=False, callbacks=[early_stopping])
     

# Visualize loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_executed = early_stopping.stopped_epoch + 1
if epochs_executed != 1:
    rg = epochs_executed
else:
    rg = epochs 

plt.figure(figsize=(15,5))
plt.plot(range(rg), loss)
plt.plot(range(rg), val_loss)
plt.title('Loss over epochs', weight='bold', fontsize=22)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend(['Training loss', 'Validation loss'], fontsize=16)
plt.show()
     

# Visualize Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(15,5))
plt.plot(range(rg), acc)
plt.plot(range(rg), val_acc)
plt.title('Accuracy over epochs', weight='bold', fontsize=22)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(['Training accuracy', 'Validation accuracy'], fontsize=16)
plt.show()
     


test_loss, test_accuracy = model.evaluate(test_paths, test_labels)

predictions = model.predict(test_paths)

# Convert the predicted probabilities to class labels (0 or 1 in this case)
y_pred = (predictions > 0.5).astype(int)

# Compute the confusion matrix
cm = confusion_matrix(test_labels, y_pred)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Class 0', 'Class 1'])
plt.yticks(tick_marks, ['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
# Add text annotations within each cell
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.show()

#Printing the accuracy, the f1-score and the recall for each class
predicted_probabilities = predictions.flatten()
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

report = classification_report(test_labels, predicted_labels)
print(report)
     

#saving the model
model.save('/content/drive/MyDrive/models/CNN from scratch')
     


