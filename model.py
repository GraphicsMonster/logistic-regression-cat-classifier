from predict import *
from train_funcs import *
from data_loader import *
from preprocessing import *

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):

    w, b = initialize_parameters(X_train.shape[1])

    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    w = parameters["w"]
    b = parameters["b"]

    print("w: " + str(w))
    print("b: " + str(b))
    print("costs: " + str(costs))

    Y_test_prediction = predict(w, b, X_test)
    Y_train_prediction = predict(w, b, X_train)

    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_prediction - Y_test)) * 100))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_prediction - Y_train)) * 100))


path_training = "./dataset/training_set/training_set/cats"
X_train, Y_train = load_dataset(path_training, 1000)
X_train, Y_train = preprocess(X_train, Y_train)

path_testing = "./dataset/test_set/test_set/cats"
X_test, Y_test = load_dataset(path_testing, 500)
X_test, Y_test = preprocess(X_test, Y_test)

model(X_train, Y_train, X_test, Y_test, 1000, 0.005)

