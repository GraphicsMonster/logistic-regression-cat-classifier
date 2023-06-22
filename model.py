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

    w = parameters["w"]
    b = parameters["b"]

    Y_test_prediction = predict(w, b, X_test)
    Y_train_prediction = predict(w, b, X_train)

    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_prediction - Y_test)) * 100))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_prediction - Y_train)) * 100))


path = "./dataset/training_set/training_set/cats"
images, labels = load_dataset(path, num_images=50)
X_train, Y_train = preprocess(images, labels, num_images=30)

path = "./dataset/test_set/test_set/cats"
images, labels = load_dataset(path, num_images=30)
X_test, Y_test = preprocess(images, labels, num_images=20)

model(X_train, Y_train, X_test, Y_test, 1000, 0.005)


