import numpy as np
import argparse


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float) -> None:
    # TODO: Implement `train` using vectorization
    # train model based on user epoch input and stochastic gradient descent

    N, D = X.shape # N is number of examples, D is number of features (300)

    for epoch in range(num_epoch):
        for example in range(N):
            feat_vector = X[example, :]
            label_vector = y[example]

            # calculate by dot product between feat vector and empty theta or weights
            # apply sigmoid to result
            # subtract value from label to get gradient value to update theta or weights
            sum = np.dot(feat_vector, theta)
            value = sigmoid(sum)
            error = value - label_vector
            gradient_value = error * feat_vector
            theta -= learning_rate * gradient_value  # subtract since we are descending lol



def predict(theta : np.ndarray, X : np.ndarray) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    # trying @ method specified in hw question

    sum = X @ theta
    predict_values = sigmoid(sum)
    # threshold 0.5 return 0 or 1
    predict_values = np.where(predict_values >= 0.5, 1, 0)

    return predict_values

def compute_error(y_pred : np.ndarray, y : np.ndarray) -> float:
    # TODO: Implement `compute_error` using vectorization
    # just num of predicted wrong / total of example

    num_wrong = np.sum(y_pred != y)
    tot_examples = len(y)
    error = num_wrong / tot_examples

    return error

def unformat_data(filepath):
    # this method will read in dataset and parse through examples and their features 
    # and transform them to vectors or matrices
    # X : (N, D) feature matrix (feature by num of examples)
    # y : (N,) vector of labels

    X = []
    y = [] 

    # parse through file and break up by label and feature embeddings
    with open(filepath, "r") as file:
        for embedding in file:
            embedding = list(map(float, embedding.strip().split()))
            y.append(embedding[0])
            X.append(embedding[1:])

    return np.array(X), np.array(y)

def pretty_print_predictions(output_path, predictions):
    
    with open(output_path, 'w') as f:
        for label in predictions:
            f.write(f"{label}\n")

def print_metrics(output_path, train_metric, test_metric):

    with open(output_path, 'w') as f:
        f.write(f"error(train): {train_metric}\n")
        f.write(f"error(test): {test_metric}\n")

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()


    # load train & initialize theta vector as same size of num features in example with 0's
    X_train, y_train = unformat_data(args.train_input)
    D = X_train.shape[1]
    theta = np.zeros(D)

    # train model with train data
    train(theta, X_train, y_train, args.num_epoch, args.learning_rate)

    # predict on training data
    train_predict_values = predict(theta, X_train)

    # load test & output predictions
    X_test, y_test = unformat_data(args.test_input)
    test_predict_values = predict(theta, X_test)

    # error on train and test
    train_error = compute_error(train_predict_values, y_train)
    test_error = compute_error(test_predict_values, y_test)

    # pretty print error metrics, train and test predictions
    pretty_print_predictions(args.train_out, train_predict_values)
    pretty_print_predictions(args.test_out, test_predict_values)
    print_metrics(args.metrics_out, train_error, test_error)
