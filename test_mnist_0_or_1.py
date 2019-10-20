import numpy as np
import os
import cv2

from keras.models import load_model

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

test_folder_0_path="/home/djkhai/PycharmProjects/HM_new/data/test/0/"
test_folder_1_path="/home/djkhai/PycharmProjects/HM_new/data/test/1/"

x=[]
y=[]

name=[]
for i in os.listdir(test_folder_0_path):
    label = 0
    img = np.asarray(cv2.imread(test_folder_0_path + i, 0))
    img = cv2.resize(img, (20, 20))
    img = np.asarray(np.reshape(1 - (img / 255.0), 400))
    n="0_"+i
    #count = count + 1

    x.append(img)
    y.append(label)
    name.append(n)

#count=0
for i in os.listdir(test_folder_1_path):
    label=1
    img = np.asarray(cv2.imread(test_folder_1_path + i, 0))
    img = cv2.resize(img, (20, 20))
    img = np.asarray(np.reshape(1 - (img / 255.0), 400))
    #count=count+1
    n = "1_" + i
    x.append(img)
    y.append(label)
    name.append(n)

test_x=np.asarray(x)
test_y=np.asarray(y)



scaler = StandardScaler()
scaler.fit(test_x)

StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_test_x = scaler.transform(test_x)

#train_x=StandardScaler().fit_transform(train_x)

pca=PCA(n_components=49)
pca.fit(scaled_test_x)

PCA(copy=True, iterated_power='auto', n_components=49, random_state=None,svd_solver='auto', tol=0.0, whiten=False)
pca_test_x = pca.transform(scaled_test_x)


model_path="/home/djkhai/PycharmProjects/HM_new/mnist_0_OR_1_PCA.h5"

model=load_model(model_path)

res=model.predict(pca_test_x)

for i in range(402):
    print(name[i]+"------",np.argmax(res[i]))