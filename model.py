from predict import *
from train_funcs import *
from data_loader import *
from preprocessing import *

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):

    w, b = initialize_parameters(X_train.shape[1])

    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    print("w = " + str(parameters["w"]))
    print("b = " + str(parameters["b"]))
    print("costs = " + str(costs))

    Y_prediction = predict(parameters["w"], parameters["b"], X_test)

    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - Y_test)) * 100))

