from train_funcs import *

def predict(w, b, X):

    Y_prediction = sigmoid(np.dot(w.T, X.T) + b)
    Y_prediction = np.where(Y_prediction > 0.5, 1, 0)

    return Y_prediction