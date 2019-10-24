#!/usr/bin/env python3
import numpy as np
#import matplotlib.pyplot as plt
from io import StringIO

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = r'C:\Users\Patrick\Documents\Spring_2019\CSC_246\adult' #TODO: if doing development somewhere other than the cycle server, change this to the directory where a7a.train, a7a.dev, and a7a.test are


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

@np.vectorize
def sigmoidDeriv(z):
    return (sigmoid(z)*(1-sigmoid(z)))
               
#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y,0) #treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray([ys],dtype=np.float32).T, np.asarray(xs,dtype=np.float32).reshape(len(xs),NUM_FEATURES,1) #returns a tuple, first is an array of labels, second is an array of feature vectors

def init_model(args):
    w1 = None
    w2 = None
    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1,len(w2))
    else:
        #TODO (optional): If you want, you can experiment with a different random initialization. As-is, each weight is uniformly sampled from [-0.5,0.5).
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES) #bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1) #add bias column
    #At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.
    #TODO: Replace this with whatever you want to use to represent the network; you could use use a tuple of (w1,w2), make a class, etc.
    model = (w1,w2)
    return model

def eval_model(model, pointx):
    w1,w2 = extract_weights(model)
    a1 = np.matmul(w1, pointx)
    z1 = sigmoid(a1)

    #adding columns of 1's
    zrows, zcolumns = z1.shape
    z1Bias = np.ones((zrows+1, zcolumns))
    z1Bias[:-1, :] = z1

    a2 = np.matmul(w2, z1Bias)
    z2 = sigmoid(a2)
    if (z2>= 0.5):
        return 1
    else:
        return 0

def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args, test_ys, test_xs):
    w1, w2 = extract_weights(model)
    #rowsw1, columnsw1 = w1.shape
    #rowsw2, olumnsw2 = w2.shape
    #print("rowsw1,: columnsw1 ", rowsw1, columnsw1)
    #print("rowsw2,: olumnsw2 ", rowsw2, olumnsw2)
    graphiterations = args.iterations
    accuracyArray = np.zeros(graphiterations)
    accuracyArray2 = np.zeros(graphiterations)
    accuracyArray3 = np.zeros(graphiterations)
    maxAccuracy = 0
    count = np.zeros(graphiterations)

    
    for t in range (0, args.iterations):
        for num in range(0, train_ys.size):
            z0 = train_xs[num]
            #rowsz0, columnsz0 = z0.shape
            #print("rowsz0,: columnsz0 ", rowsz0, columnsz0)
            a1 = np.matmul(w1, train_xs[num])
            z1 = sigmoid(a1)
            #adding columns of 1's
            zrows, zcolumns = z1.shape
            #print("zrows,: zcolumns ", zrows, zcolumns)
            z1Bias = np.ones((zrows+1, zcolumns))
            z1Bias[:-1, :] = z1
            a2 = np.matmul(w2, z1Bias)
            z2 = sigmoid(a2)
            dEdy = (z2 - train_ys[num])
            gPrime2 = sigmoidDeriv(a2)
            delta2 = np.multiply(dEdy, gPrime2)
            dEdW2 = np.matmul(delta2, z1Bias.T)
            #rowsgPrime2, columnsgPrime2 = gPrime2.shape
            #print("gPrime2rows, gPrime2: ", rowsgPrime2, columnsgPrime2)
            #dEdW2rows, dEdW2columns = dEdW2.shape
            #print("dEdW2rows, dEdW2columns: ", dEdW2rows, dEdW2columns)
            gPrime1 = sigmoidDeriv(a1)
            rowsw2, columnsw2 = w2.shape
            #print("rowsW2, columnsW2: ", rowsw2, columnsw2)
            #rowsgPrime1, columnsgPrime1 = gPrime1.shape
            #print("gPrime1rows, gPrime1: ", rowsgPrime1, columnsgPrime1)
            #rowsdelta2, columnsdelta2 = delta2.shape
            #print("rowsdelta2, columnsdelta2: ", rowsdelta2, columnsdelta2)
            #gPrime10 = np.ones((rowsgPrime1+1,columnsgPrime1))
            #gPrime10[:-1, :] = gPrime1
            w2noBias = w2[:, :-1]
            w2delta=np.matmul(w2noBias.T, delta2)
            delta1 = np.multiply(w2delta, gPrime1)
            #rowsdelta1, columnsdelta1 = delta1.shape
            #print("rowsdelta1, columnsdelta1: ", rowsdelta1, columnsdelta1)
            #print("dEdW2:", dEdW2)
            dEdW1 = np.matmul(delta1, z0.T)
            w1 = w1 - args.lr * dEdW1
            w2 = w2 - args.lr * dEdW2
        #CODE FOR DEV DATA
        if not args.nodev:
            model1 = (w1,w2)
            accuracyArray[t] = test_accuracy(model1, dev_ys, dev_xs)
            accuracyArray2[t] = test_accuracy(model1, train_ys, train_xs)
            accuracyArray3[t] = test_accuracy(model1, test_ys, test_xs)
            if accuracyArray[t] > maxAccuracy:
                maxModel = (w1,w2)
                maxAccuracy = accuracyArray[t]
        count[t] = t
    ###CODE FOR PLOTTING
##    fig, ax = plt.subplots()
##    ax.plot(count, accuracyArray, label = 'Dev Data')
##    ax.plot(count, accuracyArray2, label = 'Training Data')
##    ax.plot(count, accuracyArray2, label = 'Test Data')
##
##
##    plt.legend()
##    ax.set(xlabel='Iterations', ylabel='Accuracy',
##           title='Iterations vs Accuracy')
##    ax.grid()
##
##    fig.savefig("Accuracy_vs_Iterations.png")
##    plt.show()
    
    if (args.nodev == True):
        model = (w1,w2)
    else:
        return maxModel
    return model



def test_accuracy(model, test_ys, test_xs):
    accuracy = 0.0
    w1, w2 = extract_weights(model)
    rows1, columns1 = w1.shape
    rows2, columns2 = w2.shape
    for num in range (0, test_ys.size):
        #print("prediction, actual: ", eval_model(model, test_xs[num]), test_ys[num])
        if(eval_model(model, test_xs[num]) ==  test_ys[num]):
            accuracy+=1
    return (accuracy/test_ys.size)
    #TODO: Implement accuracy computation of given model on the test data

def extract_weights(model):
    w1 = model[0]
    w2 = model[1]
    #TODO: Extract the two weight matrices from the model and return them (they should be the same type and shape as they were in init_model, but now they have been updated during training)
    return w1, w2

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')


    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)

    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args, test_ys, test_xs)
    accuracy = test_accuracy(model, test_ys, test_xs)
    
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))

if __name__ == '__main__':
    main()
