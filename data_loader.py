import pandas as pd
import numpy as np
import random

def loadNclean_data(filepath):
    df = pd.read_csv(filepath)

    # Converting the result strings 'Iris-setosa','Iris-versicolor','Iris-virginica' to 0,1,2 respectively
    df.loc[df['Species']=='Iris-setosa','Species'] = 0
    df.loc[df['Species']=='Iris-versicolor','Species'] = 1
    df.loc[df['Species']=='Iris-virginica','Species'] = 2

    # Dropping the first column in the dataframe
    df = df.drop(['Id'],axis=1)

    # Converting the dataframe to numpy array
    data = df.values

    # Splitting the input and the output
    input = [np.reshape(x,(4,1)) for x in data[:,[0,1,2,3]]]
    output = [one_hot_output_vec(x,(3, 1)) for x in data[:,4]]

    # Combine input,output into one list
    data = list(zip(input,output))

    # Shuffle the data for randomness
    for x in range(10):
        random.shuffle(data)

    return (np.asarray(data[:100])),(np.asarray(data[100:len(data)]))

def one_hot_output_vec(output,shape,forTf=False):
    # Prepare a one hot vector for output values Ex: 2 ->[0,0,1] | 1 ->[0,1,0] | 0 ->[1,0,0]
    if forTf:
        vec = [0] * shape
        vec[output] = 1.0
    else:
        vec = np.zeros(shape)
        vec[output] = 1.0
    return vec

def loadNclean_dataForTf(filepath):
    df = pd.read_csv(filepath)

    # Converting the result strings 'Iris-setosa','Iris-versicolor','Iris-virginica' to 0,1,2 respectively
    df.loc[df['Species'] == 'Iris-setosa', 'Species'] = 0
    df.loc[df['Species'] == 'Iris-versicolor', 'Species'] = 1
    df.loc[df['Species'] == 'Iris-virginica', 'Species'] = 2

    # Dropping the first column in the dataframe
    df = df.drop(['Id'], axis=1)

    # Converting the dataframe to numpy array
    data = (df.values)
    # Randomize the data
    np.random.shuffle(data)
    input = data[:,[0,1,2,3]]
    out = data[:,4]
    output=[]
    for x in out:
        output.append(one_hot_output_vec(x,3,True))

    train_in = input[:120]
    train_out = output[:120]
    test_in = input[120:len(input)]
    test_out = output[120:len(output)]
    return  train_in,train_out,test_in,test_out
