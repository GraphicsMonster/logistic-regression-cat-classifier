import numpy as np

def sigmoid(z):
    # returns the sigmoid of z
    return 1.0/(1.0+np.exp(-z))

def initialize_parameters(dim):
    # initialize the parameters W and b
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    # x = images, y = labels
    # w, b = parameters

    # forward propagation
    m = X.shape[1]
    A = sigmoid((np.dot(w.T, X.T) + b)) # A = y(hat) = probability vector

    # cost function
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) # mean loss over total number training examples

    # backward propagation

    # derivative of cost function with respect to w and b
    dw = (1/m) * np.dot(X.T, (A-Y).T)
    db = (1/m) * np.sum(A-Y)

    gradients = {"dw": dw, "db": db}
    return gradients, cost

def optimize(w, b, X, Y, num_iterations, learning_rate):

    costs = []

    for num in range(num_iterations):

        gradients, cost = propagate(w, b, X, Y)

        dw = gradients["dw"]
        db = gradients["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if(num % 10 == 0):
            costs.append(cost)
        
    parameters = {"w": w, "b": b}

    return parameters, costs

# the dataset X being passed should be a 2d array with the row being the number of images and the column being the number of features per image
# the weight vector w should be a 2d array with the row being the number of features per image and the column being 1
        