from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np  

import cv2
import random


#constant
BATCH_SIZE = 128
NB_EPOCH = 10
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()
early_stopping=EarlyStopping(monitor='value_loss') 

(X_train, y_train), (X_test, y_test) = mnist.load_data()

for i in range(len(X_train)):
    ret, X_train[i] = cv2.threshold(X_train[i], 127,255, cv2.THRESH_BINARY)
for i in range(len(X_test)):
    ret, X_test[i] = cv2.threshold(X_test[i], 127,255, cv2.THRESH_BINARY)

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

print('преобразование данных закончено')

def get_neir():
    model = Sequential()
    model.add(Conv2D(64,(3, 3), activation = 'relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
      
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    model.compile(loss='MSE', optimizer=OPTIM, metrics=['accuracy'])
     
    return model
print('запуск обучения')
all_val_acc_histories =[]
all_val_loss_histories = []
all_acc_histories =[]
all_loss_histories = []
# k = 1
# for i in range(k):                            кросс-валидация, от которой в последствии отказался
#     print('провекра по к блокам №', i+1)
    # for j in range(len(X_train)):
    #     ran = random.randint(0, len(X_train) - 1)
    #     X_train[j], X_train[ran] = X_train[ran], X_train[j]
    #     Y_train[j], Y_train[ran] = Y_train[ran], Y_train[j]

model = get_neir()
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
                    validation_split=VALIDATION_SPLIT, verbose=1, 
                    callbacks=[early_stopping])

print('Testing...')

score = model.evaluate(X_test, Y_test, verbose=0)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])


#save model
model.save('My_NeirosetMY.h5')
print("Модель сохранена")

# визуализация анализа обучения по поколениям

loss_history = history.history['loss']
val_loss_history = history.history['val_loss']
acc_histories = history.history['accuracy']
vall_acc_histories = history.history['val_accuracy']
all_loss_histories.append(loss_history)
all_val_loss_histories.append(val_loss_history)
all_acc_histories.append(acc_histories)
all_val_acc_histories.append(vall_acc_histories)
    
r_all_loss_histories = [np.mean([x[i] for x in all_loss_histories]) for i in range(NB_EPOCH)]
r_all_acc_histories = [np.mean([x[i] for x in all_acc_histories]) for i in range(NB_EPOCH)]
r_all_val_loss_histories = [np.mean([x[i] for x in all_val_loss_histories]) for i in range(NB_EPOCH)]
r_all_val_acc_histories = [np.mean([x[i] for x in all_val_acc_histories]) for i in range(NB_EPOCH)]

    
plt.plot(range(len(r_all_loss_histories)), r_all_loss_histories,'bo', label='Потери на этапе обучения')
plt.plot(range(len(r_all_val_loss_histories)), r_all_val_loss_histories, 'b', label='Потери на этапе проверки')
plt.title('Потери на этапе обучения и проверки')
plt.xlabel('Поколения')
plt.ylabel('Потери')
plt.legend()
plt.show()

plt.clf()
plt.plot(range(len(r_all_acc_histories)), r_all_acc_histories,'bo', label='Точность на этопе обучения')
plt.plot(range(len(r_all_val_acc_histories)), r_all_val_acc_histories, 'b', label='Точность на этапе проверки')
plt.title('Точность на этапе обучения и проверки')
plt.xlabel('Поколения')
plt.ylabel('Точность')
plt.legend()
plt.show()

plt.clf
