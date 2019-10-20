import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Activation, merge
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Dense
from keras.utils.np_utils import to_categorical
from keras import optimizers

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train_folder_0_path="/home/djkhai/PycharmProjects/HM_new/data/train/0/"
train_folder_1_path="/home/djkhai/PycharmProjects/HM_new/data/train/1/"

val_folder_0_path="/home/djkhai/PycharmProjects/HM_new/data/val/0/"
val_folder_1_path="/home/djkhai/PycharmProjects/HM_new/data/val/1/"

def mnist_model():
    model=Sequential()
    model.add(Dense(98, activation='sigmoid', name='fc1', input_dim=49))
    model.add(Dense(2, activation="softmax"))

    return model


def get_data(folder_0_path, folder_1_path):
    x=[]
    y=[]

    count=0
    for i in os.listdir(folder_0_path):
        if count<1000:
            label=0
            img = np.asarray(cv2.imread(folder_0_path + i, 0))
            img = cv2.resize(img, (20, 20))
            img= np.asarray(np.reshape(1- (img/255.0), 400))
            count=count+1

            x.append(img)
            y.append(label)

    count=0
    for i in os.listdir(folder_1_path):
        if count<1000:
            label=1
            img = np.asarray(cv2.imread(folder_1_path + i, 0))
            img = cv2.resize(img, (20, 20))
            img = np.asarray(np.reshape(1 - (img / 255.0), 400))
            count=count+1

            x.append(img)
            y.append(label)


    x=np.asarray(x)
    y=np.asarray(y)

    return x,y


train_x, train_y=get_data(train_folder_0_path, train_folder_1_path)
val_x, val_y= get_data(val_folder_0_path, val_folder_1_path)

#print(train_x.shape)

scaler = StandardScaler()
scaler.fit(train_x)

StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_train_x = scaler.transform(train_x)

pca=PCA(n_components=49)
pca.fit(scaled_train_x)

PCA(copy=True, iterated_power='auto', n_components=49, random_state=None,svd_solver='auto', tol=0.0, whiten=False)
pca_train_x = pca.transform(scaled_train_x)

'''
#plotting the values to find the required Pricipal Components.

per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)

labels = ['PC'+str(i) for i in range(1,len(per_var)+1)]

plt.subplots(figsize=(60,10))
plt.bar(x=range(1,len(per_var)+1), height=per_var,tick_label=labels)
plt.show()
'''

# Encoding labels to hot vectors
train_y_hot = to_categorical(train_y, num_classes = 2)
val_y_hot = to_categorical(val_y, num_classes = 2)
print(train_y_hot.shape)

model=mnist_model()
final_model_PATH="/home/djkhai/PycharmProjects/HM_new/mnist_0_OR_1_PCA_sigmoid.h5"


# compile the model
model.compile(loss = "binary_crossentropy", optimizer =optimizers.Adam(), metrics=["accuracy"])



max_epochs = 20  # too few
print("Starting training ")
model.fit(pca_train_x, train_y_hot, batch_size=200,epochs=max_epochs, verbose=1)
print("Training complete")

model.save(final_model_PATH)  # creates a HDF5 file 'my_model.h5'

# 4. evaluate model
# loss_acc = model.evaluate(val_x, val_y_hot, verbose=0)
# print("\nTest data loss = %0.4f  accuracy = %0.2f%%" % \
#   (loss_acc[0], loss_acc[1]*100) )





