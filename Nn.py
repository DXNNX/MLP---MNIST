import numpy as np
import os
import pickle

class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def prime(z):
        z[z < 0] = 0
        z[z > 0] = 1
        return z


class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))

class Softmax:
    @staticmethod
    def activation(X):
        y = X - np.expand_dims(np.max(X, axis = 1), 1)
        y = np.exp(y)
        ax_sum = np.expand_dims(np.sum(y, axis = 1), 1)
        p = y / ax_sum
        return p

    @staticmethod
    def prime(z):
        return Softmax.activation(z) * (1 - Softmax.activation(z))


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class MSE:
    def __init__(self, activation_fn=None):
        self.activation_fn = activation_fn

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)


class CrossEntropy:
    def __init__(self, activation_fn=None):
        self.activation_fn = activation_fn

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        return np.mean(np.sum(-np.multiply(y_true,np.log(y_pred)), axis=1))

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)


class Network:

    def save(self):
        pickle.dump(np.array([self.w,self.b,self.activations]), open( self.fileName, "wb" ) )
        
    def __init__(self, dimensions, activations,fileName):
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None
        self.fileName = fileName
        self.History = [[],[]]

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            self.w[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]

        if(os.path.isfile(self.fileName)):
            self.w,self.b,self.activations = pickle.load( open( fileName, "rb" ) )
            print("Wiii")

    
    def feed_forward(self, x):
        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.n_layers):
            z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
        return z, a

    def back_prop(self, z, a, y_true,masks=None):
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = np.dot(a[self.n_layers - 1].T, delta)

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        for i in reversed(range(2, self.n_layers)):
            delta = np.dot(delta, self.w[i].T) * self.activations[i].prime(z[i]) #* masks[i]
            dw = np.dot(a[i - 1].T, delta) 
            update_params[i - 1] = (dw, delta)

        for k, v in update_params.items():
            self.update_w_b(k, v[0], v[1])

    def update_w_b(self, index, dw, delta):
        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * np.mean(delta, 0)

    def fit(self, x_total, y_total, loss, epochs, batch_size, learning_rate=1e-3):
        if not x_total.shape[0] == y_total.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        # Initiate the loss object with the final activation function
        self.loss = loss(self.activations[self.n_layers])
        self.learning_rate = learning_rate
        x = x_total[:int(len(x_total)*0.8)]
        x_test = x_total[int(len(x_total)*0.8):]
        y_true = y_total[:int(len(x_total)*0.8)]
        y_true_test = y_total[int(len(x_total)*0.8):]
        
        for i in range(epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]
            
            if (i + 1) % 1 == 0:
                _, a = self.feed_forward(testx)
                L = self.loss.loss(testy, a[self.n_layers])
                accuracy = self.accuracy(a[self.n_layers],testy)
                self.History[0].append(L)
                self.History[1].append(accuracy)
                print("Loss:", L," / Accuaracy:",accuracy)

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self.feed_forward(x_[k:l])
                self.back_prop(z, a, y_[k:l])



    def predict(self, x):
        _, a = self.feed_forward(x)
        return a[self.n_layers]

    def accuracy(self,x,y_true):
        return (sum((np.argmax(x, axis=1) == np.argmax(y_true, axis=1)))/len(y_true))


def one_hot_vector(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e



from sklearn import datasets
import sklearn.metrics
import mnist
np.random.seed(1)

trainx = mnist.parse_idx('train-images-idx3-ubyte')
x = np.array([x.reshape(784) for x in trainx]) / 255
trainy = mnist.parse_idx('train-labels-idx1-ubyte')
y = np.array([one_hot_vector(j) for j in trainy])

testx = mnist.parse_idx('t10k-images-idx3-ubyte')
testx = Xs = [x.reshape(784) for x in testx]

testy = mnist.parse_idx('t10k-labels-idx1-ubyte')
testy = Ys = [one_hot_vector(j) for j in testy]

nn = Network((784,128,64,10), (Relu,Relu, Softmax), "1.bin")


nn.fit(x, y,
       loss=MSE,
       epochs=10,
       batch_size=32,
       learning_rate=0.0085,
       )

nn.save()

from PIL import Image
owny = np.array([one_hot_vector(i) for i in range(10)])
#Import own images
ownx = np.array([np.array(Image.open('own/'+str(i)+'.png').convert('L')).reshape((784,)) for i in range(10)])

def prod():   
    predictionO = nn.predict(ownx)
    accuracyO = nn.accuracy(predictionO,owny)
    print(accuracyO)

    prediction = nn.predict(testx)
    accuracy = nn.accuracy(prediction,testy)
    print(accuracy)
prod()




##y_true = []
##y_pred = []
##for i in range(len(prediction)):
##    y_pred.append(np.argmax(prediction[i]))
##    y_true.append(np.argmax(owny[i]))
##
##print(
##print(sklearn.metrics.classification_report(y_true, y_pred))
##
##
##
#import matplotlib.pyplot as plt
#plt.plot(range(len(nn.History[0])),nn.History[0])
#plt.plot(range(len(nn.History[1])),nn.History[1])
#plt.show()
