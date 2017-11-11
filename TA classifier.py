#personal best: loss 25.8218001166 % lr: 0.1 3-8-5-3-1

import numpy as np

lr = 0.1 #our learning rate
X = np.array([[0,0,0]])
y = np.array([[0]])
input_file = open("tae_dataset.txt","r")

# there are 4 parameters in the dataset
# NATIVE SPEAKER: 1 = English Speaker, 2 = non-English speaker
# SUMMER: 1 = Summer, 2= Regular
# CLASS SIZE: (numerical)
# TA evaluation: 1 = Low, 2 = Medium, 3 = High
for line in input_file:
    native_speaker = int(line.split(',')[0])-1
    summer = int(line.split(',')[1])-1
    class_size = 1/(float(line.split(',')[2]))
    pre_eval = int(line.split(',')[3])
    if pre_eval == 1: evaluation = 0
    elif pre_eval == 2: evaluation = 0.5
    else: evaluation = 1
    X = np.append(X,[[native_speaker,summer,class_size]],axis=0)
    y = np.append(y,[[evaluation]],axis=0)

X = np.delete(X, 0, 0)
y = np.delete(y, 0, 0)
input_file.close()
np.random.seed(1)

# this neural network is going to have 3 hidden layers, defined by the following weight matrices
weights0 = 2*np.random.random((3,5)) - 1
weights1 = 2*np.random.random((5,8)) - 1
weights2 = 2*np.random.random((8,5)) - 1
weights3 = 2*np.random.random((5,1)) - 1

#sigmoid -activation function
def nonlin(x,deriv = False):
    if(deriv == True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

def test(X):
    layer0 = X
    layer1 = nonlin(np.dot(layer0,weights0))
    layer2 = nonlin(np.dot(layer1,weights1))
    layer3 = nonlin(np.dot(layer2,weights2))
    layer4 = nonlin(np.dot(layer3,weights3))
    print(layer4)

#train the network
for j in range(500000):
    
    #feed forward
    layer0 = X
    layer1 = nonlin(np.dot(layer0,weights0))
    layer2 = nonlin(np.dot(layer1,weights1))
    layer3 = nonlin(np.dot(layer2,weights2))
    layer4 = nonlin(np.dot(layer3,weights3))

    #calculate error
    layer4_error = y - layer4
    if(j %10000) == 0:
        print("Error:" + str(np.mean(np.abs(layer4_error))))

    #Back propagation of errors using the chain rule.
    layer4_delta = layer4_error*nonlin(layer4,deriv=True)
    layer3_error = layer4_delta.dot(weights3.T)
    layer3_delta = layer3_error * nonlin(layer3,deriv=True)
    layer2_error = layer3_delta.dot(weights2.T)
    layer2_delta = layer2_error * nonlin(layer2,deriv=True)
    layer1_error = layer2_delta.dot(weights1.T)
    layer1_delta = layer1_error * nonlin(layer1,deriv=True)

    #Using the deltas, we can use them to update the weights to
    #reduce the error rate with every iteration
    #This algorithm is called gradient descent
    weights3 +=layer3.T.dot(layer4_delta) * lr
    weights2 +=layer2.T.dot(layer3_delta) * lr
    weights1 +=layer1.T.dot(layer2_delta) * lr
    weights0 +=layer0.T.dot(layer1_delta) * lr
    
    
