from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data_banknote_authentication.csv")
dataset = df.values

attribs = dataset[:,0:4]
ev_binary = dataset[:,4]

min_max_scaler = preprocessing.MinMaxScaler()
attribs_scaler = min_max_scaler.fit_transform(attribs)

attribs_train, attribs_val_and_test, ev_binary_train, ev_binary_val_and_test = train_test_split(attribs_scaler, ev_binary, test_size=0.3)
attribs_val, attribs_test, ev_binary_val, ev_binary_test = train_test_split(attribs_val_and_test, ev_binary_val_and_test, test_size=0.5)

model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(attribs_train, ev_binary_train,
                 batch_size=32, epochs=100,
                 validation_data=(attribs_val,ev_binary_val))

model.evaluate(attribs_test,ev_binary_test)[1]

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


print(dataset)