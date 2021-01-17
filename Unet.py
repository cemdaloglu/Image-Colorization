# #Reference: The U-Net model is taken from the following link
# https://www.kaggle.com/rahuldshetty/image-colorization-with-unet-auto-encoders/notebook
# Libraries
import csv
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from skimage.measure import compare_ssim
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, InputLayer, UpSampling2D, Conv2DTranspose, LeakyReLU, AveragePooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Input, RepeatVector, Reshape, concatenate
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model


images_gray = np.load(r"C:\Users\CEM\Desktop\archive\l\gray_scale.npy")
images_lab = np.load(r"C:\Users\CEM\Desktop\archive\ab\ab\ab1.npy")


# gray_scale
size_train = 40
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

model = Sequential()
inputs = Input((224, 224, 1))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
conv10 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

history = model.fit(x=X_train_l, y=X_train_lab, validation_split=0.2, epochs=250, batch_size=16)
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
