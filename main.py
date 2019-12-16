from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# banknote auth set imp.
# df = pd.read_csv("data_banknote_authentication.csv")

# heart disease set imp.
df = pd.read_csv("heart.csv")

dataset = df.values

# Define the column number
column_num = dataset.shape.__getitem__(1)
string = 'Number of attributes: '
print("{}{}".format(string, column_num - 1))

# Specify the attributes and outputs
x = dataset[:, 0:column_num - 1]
y = dataset[:, column_num - 1]

# Compress the data between 0-1
min_max_scaler = preprocessing.MinMaxScaler()
x_scale = min_max_scaler.fit_transform(x)

# Split the set as train set, test set, validation set
x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(x_scale, y, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0.5)

# Define model architecture
model = Sequential([
    Dense(32, activation='relu', input_shape=(column_num - 1,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

# Define model features
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train,
                 batch_size=32, epochs=100,
                 validation_data=(x_val, y_val))

# Evaluate the model
model.evaluate(x_test, y_test)[1]

# Linear Regression
regressor = LinearRegression()
regressor.fit(x, y)
pickle.dump(regressor, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[57, 0, 0, 120, 354, 0, 1, 163, 1, 0.6, 2, 0, 2]]))

# Drawing plots
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
