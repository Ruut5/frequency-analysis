import keras
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

plt.imshow(test_data[9])

def vect(se, di=10000):
    results = np.zeros((len(se), di))
    #print(results)
    for i, se in enumerate(se):
        results[i, se] = 1.
    return results

x_train = vect(train_data)
x_test = vect(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

acc=20

history = model.fit(partial_x_train,partial_y_train, epochs=acc, batch_size=1024, validation_data=(x_val,y_val))
hi_dict = history.history
loss_values = hi_dict['loss']
val_loss_values = hi_dict['val_loss']

history2 = model.fit(x_train, y_train, epochs=4, batch_size=128)
results = model.evaluate(x_test, y_test)
hi_dict2 = history2.history
loss_values2 = hi_dict2['loss']

epochs = range(1,acc+1)
plt.plot(epochs, loss_values,'bo', label='Потери на этапе обучения')
plt.plot(epochs, val_loss_values, 'b', label='Потери на этапе проверки')
plt.title('Потери на этапе обучения и проверки')
plt.xlabel('Поколения')
plt.ylabel('Потери')
plt.legend()
plt.show()

plt.clf()
acc_val = hi_dict['acc']
val_acc_val = hi_dict['val_acc']
plt.plot(epochs, acc_val,'bo', label='Точность на этопе обучения')
plt.plot(epochs, val_acc_val, 'b', label='Точность на этапе проверки')
plt.title('Точность на этапе обучения и проверки')
plt.xlabel('Поколения')
plt.ylabel('Точность')
plt.legend()
plt.show()

plt.clf
epochs = range(1,5)
plt.plot(epochs, loss_values2,'b', label='Контрольные потери')
plt.title('Потере на контрольных данных')
plt.xlabel('Поколения')
plt.ylabel('Потери')
plt.legend()
plt.show()

plt.clf()
acc_val2 = hi_dict2['acc']
plt.plot(epochs, acc_val2,'b', label='Контрольная точность')
plt.title('Точность на контрольных данных')
plt.xlabel('Поколение')
plt.ylabel('Точность')
plt.legend()
plt.show()