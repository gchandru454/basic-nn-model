### EXP NO: 01

### .

# <p align = "center"> Developing a Neural Network Regression Model </p>
## AIM
To develop a neural network regression model for the given dataset.

## <br><br><br><br><br><br><br><br>THEORY
The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a three neuron.Output layer  contains single neuron.First hidden layer contains four neurons and second hidden layer contains four neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model(single input neuron forms single output neuron).

## <br><br><br><br><br><br><br>Neural Network Model
![WhatsApp Image 2022-08-28 at 11 27 18 PM](https://user-images.githubusercontent.com/112341815/187089589-678a792a-89da-4b3e-bc5b-b01e61b367a8.jpeg)


## DESIGN STEPS
### STEP 1:
Loading the dataset
### STEP 2:
Split the dataset into training and testing
### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.
### STEP 4:
Build the Neural Network Model and compile the model.
### STEP 5:
Train the model with the training data.
### STEP 6:
Plot the performance plot
### STEP 7:
Evaluate the model with the testing data.

## PROGRAM
```python
# Developed By:CHANDRU.G
# Register Number:212220040029

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv("data.csv")
df.head()
x=df[['input']].values
x
y=df[['output']].values
y


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=40)
scaler=MinMaxScaler()
scaler.fit(xtrain)
scaler.fit(xtest)
xtrain1=scaler.transform(xtrain)
xtest1=scaler.transform(xtest)

model=Sequential([
    Dense(4,activation='relu'),
    Dense(4,activation='relu'),
    
    Dense(1)
])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(xtrain1,ytrain,epochs=3000)
lossmodel=pd.DataFrame(model.history.history)
lossmodel.plot()
model.evaluate(xtest1,ytest)

xn1=[[65]]
xn11=scaler.transform(xn1)
model.predict(xn11)
```

## Dataset Information
![Screenshot (70)](https://user-images.githubusercontent.com/112341815/187089255-80a94b92-8402-4a4c-a93d-0a03c7ab73ee.png)


## <br>OUTPUT
### Training Loss Vs Iteration Plot
![Screenshot (65)](https://user-images.githubusercontent.com/112341815/187089142-ebbcdf90-186d-4a17-b7ed-1ab230062626.png)

### Test Data Root Mean Squared Error
![Screenshot (67)](https://user-images.githubusercontent.com/112341815/187089174-cf43c30f-68e4-4e0c-933a-5508cf2be96e.png)

### New Sample Data Prediction
![Screenshot (68)](https://user-images.githubusercontent.com/112341815/187089194-2b8e2c6c-61f2-4b46-ae9c-f1a98ac5bfb0.png)


## RESULT
Thus,the neural network regression model for the given dataset is developed.
