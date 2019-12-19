from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import os

# Shows the dataset folder content
arr = os.listdir("datasets/")

# Prints all datasets in dataset folder
for i in arr:
    print(f'{arr.index(i)}{". "}{i}')

# Choose a dataset from folder
# Read .csv file and define it as dataframe.
while True:
    try:
        selected_dataset_index = int(input("Choose a dataset: "))
        print(f'{"Selected dataset: "}{arr[selected_dataset_index]}')
        df = pd.read_csv("datasets/" + arr[selected_dataset_index].format(str))
        break
    except IndexError:
        print("We knew that you'll try this M. Gökhan bey. :D")

# Creates an array and assign the dataframe to this array
dataset = df.values

print(dataset.shape)
print(len(dataset[0]) - 1)

# Takes inputs for prediction according to chosen dataset
inputs = []
for i in range(len(dataset[0]) - 1):
    inputs.append(float(input(f'{"Enter the "}{i}{". input: "}')))

# Prints the input you choose to console
for i in inputs:
    print(i, end=' ')

# Define the column number
column_num = dataset.shape.__getitem__(1)
print(f'{"Number of attributes: "}{column_num - 1}')

# Specify the input values and outputs
x = dataset[:, 0:column_num - 1]
y = dataset[:, column_num - 1]

# Compress the data between 0-1
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

# Split the set as train set, test set, validation set
x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(x_scaled, y, test_size=0.3)
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
# pickle.dump(regressor, open('model.pkl', 'wb'))
# model = pickle.load(open('model.pkl', 'rb'))

pickle.dump(regressor, open(f'{"models/"}{arr[selected_dataset_index]}{".pkl"}', 'wb'))
model = pickle.load(open(f'{"models/"}{arr[selected_dataset_index]}{".pkl"}', 'rb'))

# prints the prediction according to model
print(model.predict([inputs]))

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
