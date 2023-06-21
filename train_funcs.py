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
    m = X.shape[0]
    A = sigmoid((np.dot(X, w) + b)) # A = y(hat) = probability vector

    # cost function
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) # mean loss over total number training examples

    # backward propagation

    # derivative of cost function with respect to w and b
    dw = (1/m) * np.dot(X.T, (A-Y).T)
    db = (1/m) * np.sum(A-Y)

    gradients = {"dw": dw, "db": db}
    return gradients, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, batch_size):
    m = X.shape[1]
    costs = []

    for i in range(num_iterations):
        # Randomly shuffle the training examples
        permutation = np.random.permutation(m)
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        # Divide the dataset into mini-batches
        num_batches = m // batch_size
        for j in range(num_batches):
            # Select the mini-batch
            start = j * batch_size
            end = start + batch_size
            mini_batch_X = shuffled_X[:, start:end]
            mini_batch_Y = shuffled_Y[:, start:end]

            # Perform forward and backward propagation
            gradients, cost = propagate(w, b, mini_batch_X, mini_batch_Y)

            # Update parameters
            w = w - learning_rate * gradients["dw"]
            b = b - learning_rate * gradients["db"]

        # Compute cost and append to the list
            if i % 100 == 0:
                costs.append(cost)
                print("w = " + w)
                print("b = " + b)
                print("cost = " + cost)

    parameters = {"w": w, "b": b}
    return parameters, costs

        