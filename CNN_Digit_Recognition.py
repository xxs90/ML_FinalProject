# -*- coding: utf-8 -*-
"""
Description: This is a CNN digit recognition using Tensorflow
Created on Thu Apr  8 19:41:17 2021
@author: Yijia Shi
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


def alplot():
    #accurate loss plot method
    fig, ax = plt.subplots(1,2,figsize=(15,7))
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['train', 'validation'], loc='lower right')
    ax[0].grid()
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['train', 'validation'], loc='upper right')
    ax[1].grid()
    plt.show()


if __name__ == '__main__':
    train = pd.read_csv(r'C:\Users\97368\OneDrive - Pro\Desktop\2021Spring\ECE4424\FinalProject\digit-recognition/input/train.csv')
    test = pd.read_csv(r'C:\Users\97368\OneDrive - Pro\Desktop\2021Spring\ECE4424\FinalProject\digit-recognition/input/test.csv')
    print("Training data: " + str(train.shape))
    print("Test data: " + str(test.shape))
    x_train = train.iloc[:, 1:]
    y_train = train.iloc[:, 0]
    x_test = test.iloc[:, 1:]
    y_test = test.iloc[:, 0]
    
    sns.barplot(x=y_train.unique(), y=y_train.value_counts())
    plt.xlabel('Digits')
    plt.ylabel('Number of image samples')
    x_train = np.array(x_train).reshape(-1,28,28,1)
    # reshape the train datas
    # plot the number of samples of each digit
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        validation_split=0.2
    )
    # rescale the train data
    train_data = train_datagen.flow(x_train, y_train, batch_size=64, subset='training')
    validation_generator = train_datagen.flow(x_train, y_train, batch_size=64,subset='validation')
    # set train data and validation data for training.
    cnn_model = Sequential([    
        Input(shape=(28,28,1)),
        Conv2D(32, (3, 3), activation = 'relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation = 'relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation = 'relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')])
    # the component of the model, build the cnn model
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.summary()
    # summary of the model
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.0001)
    # avoid too many training without progress
    history=cnn_model.fit(train_data, validation_data = validation_generator, epochs=100, callbacks=[early_stopping])
    alplot()
    x_test = np.array(x_test).reshape(-1, 28, 28 , 1) / 255
    # reshape the test data for prediction
    
    preds = cnn_model.predict(x_test)
    labels = [np.argmax(x) for x in preds]
    ids = [x+1 for x in range(len(preds))]
    
    output = pd.DataFrame()
    output['ImageId'] = ids
    output['Label'] = labels
    
    output.to_csv('submission.csv', index=False)
    test_loss, test_acc = cnn_model.evaluate(x_test, y_test)