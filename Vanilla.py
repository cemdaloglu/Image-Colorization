# Libraries
import csv
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from skimage.measure import compare_ssim
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, InputLayer, UpSampling2D, Conv2DTranspose, LeakyReLU, AveragePooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Input, RepeatVector, Reshape, concatenate
import matplotlib.pyplot as plt
import numpy as np


images_gray = np.load("gray_scale.npy")
images_lab = np.load("ab1.npy")


# gray_scale
size_train = 4000
size_test = 100
size_test_begin = 0
change_pic = 0
X_train_l = images_gray[0+change_pic:change_pic+size_train]
X_train_l = X_train_l.reshape(size_train, 224, 224, 1)
X_test_l = images_gray[size_test_begin + size_train+change_pic:change_pic+size_train + size_test_begin + size_test]
X_test_l = X_test_l.reshape(size_test, 224, 224, 1)

# colored
X_train_lab = images_lab[0+change_pic:change_pic+size_train]
X_train_lab = X_train_lab.reshape(size_train, 224, 224, 2)
X_test_lab = images_lab[size_test_begin + size_train+change_pic:change_pic+size_train + size_test_begin + size_test]
X_test_lab = X_test_lab.reshape(size_test, 224, 224, 2)

X_gray_chick = np.zeros((1,224,224,2), dtype=float)
X_gray_chick = X_gray_chick.__add__(128)
check_test = 0
X_test_lab_temp = X_test_lab
X_test_l_temp = X_test_l
while check_test < size_test:
    if sum(sum(sum(sum(X_gray_chick == X_test_lab_temp[check_test]))))/100352 > 0.4:
        X_test_l = np.zeros((size_test-1, 224, 224, 1), dtype=float)
        X_test_l[0:check_test] = X_test_l_temp[0:check_test]
        X_test_l[check_test:] = X_test_l_temp[(check_test+1):]
        X_test_l_temp = X_test_l

        X_test_lab = np.zeros((size_test-1, 224, 224, 2), dtype=float)
        X_test_lab[0:check_test] = X_test_lab_temp[0:check_test]
        X_test_lab[check_test:] = X_test_lab_temp[check_test + 1:]
        X_test_lab_temp = X_test_lab
        size_test = size_test-1
        check_test = check_test - 1
    check_test = check_test + 1

check_train = 0
X_train_lab_temp = X_train_lab
X_train_l_temp = X_train_l
while check_train < size_train:
    if sum(sum(sum(sum(X_gray_chick == X_train_lab_temp[check_train]))))/100352 > 0.4:
        X_train_l = np.zeros((size_train-1, 224, 224, 1), dtype=float)
        X_train_l[0:check_train] = X_train_l_temp[0:check_train]
        X_train_l[check_train:] = X_train_l_temp[(check_train+1):]
        X_train_l_temp = X_train_l

        X_train_lab = np.zeros((size_train-1, 224, 224, 2), dtype=float)
        X_train_lab[0:check_train] = X_train_lab_temp[0:check_train]
        X_train_lab[check_train:] = X_train_lab_temp[check_train + 1:]
        X_train_lab_temp = X_train_lab
        size_train = size_train-1
        check_train = check_train - 1
    check_train = check_train + 1

X_test_l = X_test_l/255
X_train_l = X_train_l/255

X_train_l = tf.cast(X_train_l, tf.float32)
X_test_l = tf.cast(X_test_l, tf.float32)

# #Vanilla CNN Model 1
# model = Sequential()
# model.add(Conv2D(strides=1,filters=32,kernel_size=3,activation='relu',use_bias=True,padding='valid'))
# model.add(Conv2D(strides=1,filters=16,kernel_size=3,activation='relu',use_bias=True,padding='valid'))
# model.add(Conv2DTranspose(strides=1,filters=2,kernel_size=5,activation='relu',use_bias=True,padding='valid'))

#Vanilla CNN Model 2
model = Sequential()
model.add(Conv2D(strides=1,filters=32,kernel_size=3,activation='relu',use_bias=True,padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(strides=1,filters=16,kernel_size=3,activation='relu',use_bias=True,padding='valid'))
model.add(Conv2DTranspose(strides=2,filters=2,kernel_size=8,activation='relu',use_bias=True,padding='valid'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(x=X_train_l, y=X_train_lab, validation_split=0.2, epochs=250, batch_size=16)
model.summary()
xx = model.predict(X_test_l)
xx = xx.reshape(size_test, 224, 224, 2)

h = history
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Model Loss')
plt.legend(['loss', 'val_loss'])
plt.axis([0, 250, 0, 20000])
plt.show()
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Model Accuracy')
plt.show()
for k in range(size_test):
    # predicted image
    img = np.zeros((224, 224, 3))
    kor = X_test_l[k]*255

    kor = kor[:, :, 0]
    img[:, :, 1:] = xx[k]
    img[:, :, 0] = kor

    img = img.astype('uint8')
    img_ = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    plt.imshow(img_)
    plt.show()

    # actual image
    img_truth = np.zeros((224, 224, 3))
    kor_truth = X_test_l[k]*255
    kor_truth = kor_truth[:, :, 0]
    img_truth[:, :, 0] = kor_truth
    img_truth[:, :, 1:] = X_test_lab[k]

    img_truth = img_truth.astype('uint8')
    img_truth_ = cv2.cvtColor(img_truth, cv2.COLOR_LAB2RGB)
    plt.imshow(img_truth_)
    plt.show()
    (score, diff) = compare_ssim(img_truth, img_truth_, full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))
    print("PSNR: {}".format(cv2.PSNR(img_truth, img_truth_)))
    m = tf.keras.metrics.Accuracy()
    m.update_state(img_truth, img_truth_)
    print("Accuracy: ", m.result().numpy())
