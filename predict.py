from train_funcs import *

def predict(w, b, X):

    m = X.shape[1]
    y_hat = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            y_hat[0, i] = 1
        else:
            y_hat[0, i] = 0

    return y_hat