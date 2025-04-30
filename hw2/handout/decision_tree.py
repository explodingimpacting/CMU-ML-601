"""
Author: Roxxanne White
Date: 1/31/2025
File Name: decision_tree.py
Description: This file will implement a decision tree. Pulling from inspection.py in terms of entropy(), readfile() and 
other methods as well as calculating mutual information to grow our tree. It will then output various text files to print
metrics, labels and as well as a visual of our tree.

"""
import argparse
import numpy as np
import math

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self, left=None, right=None, attr=None, vote=None):
        self.left = left
        self.right = right
        self.attr = attr
        self.vote = vote

def readfile(input_file):
#this method reads in data sets & configures attributes(features) to data
#and last column (labels) to label; also list of feature names to print later

    data = np.genfromtxt(input_file, delimiter='\t', dtype=None, skip_header=1)
    labels = data[:, -1]
    attributes = data[:, :-1]
    feature_names = np.genfromtxt(input_file, delimiter='\t', dtype=str, max_rows=1)
    feature_names = feature_names[:-1]
   

    return attributes, labels, feature_names

def majorityvoteclassifier(train_label):
#this method will find majority vote lol; most common label
#we will take in lastcolumn and create dictionary with each value seen and count +1 for each occurence
#we will return most common label or majority voted label
    counts = {}

    for value in train_label: 
        if value in counts: 
            counts[value] += 1
        else:
            counts[value] = 1
#might have to fix this for tie breaker to return 1 instead of 0
    majority_vote = max(counts, key=counts.get)
#    print(f"Majority vote: {majority_vote}")  #added this to check output------
    return majority_vote

def entropy(train_label):
#to calculate entropy we must take tot instances for label
#keep count of each instance. Then apply entropy formula: h(x) = -(P(x) * log2(P(x)))
    tot_instances = len(train_label)
    counts = {}

    for value in train_label:
        if value in counts:
            counts[value] += 1
        else: 
            counts[value] = 1
#    print("what is counts @ 0?", counts.get(0))
#    print("what is count at 1?", counts.get(1))

    prob_0 = counts.get(0, 0) / tot_instances
    prob_1 = counts.get(1, 0) / tot_instances

#another error to stop log(0) I was getting; return 0 ultimatley
    if prob_0 == 0 or prob_1 == 0:
        return 0
    
    entropy_value = - ((prob_0 * math.log2(prob_0)) + prob_1 * math.log2(prob_1))

    return entropy_value

def mutual_info(attributes, attribute, label):
#we calculate mutual information: I(Y; X) = H(Y) - H(Y|X) for attributes
#difference b/w entropy of Y & wieghted conditional entropy H(Y|X)

    total = len(label)
    entropy_Y = entropy(label)

#split data based on attribute value 0 or 1
    left_labels = [label[i] for i in range(total) if attributes[i][attribute] == 0]
    right_labels = [label[i] for i in range(total) if attributes[i][attribute] == 1]

#had to add since kept getting NONE error; assuming split labels aren't empty
    if len(left_labels) == 0 or len(right_labels) == 0:
        return 0

#calculate H(Y|X = 0) and H(Y|X = 1); conditional entropy of Y given X;
# H(Y| X = 0) & H(Y| X = 1) entropy of labels when x is 0 and 1
    h_y_0 = entropy(left_labels)
    h_y_1 = entropy(right_labels)

#calculate the probablities of P(X = 0) and P(X = 1)
    prob_x_0 = len(left_labels) / total
    prob_x_1 = len(right_labels) / total

#finally plug and chug to mutual information formula
    mutual_i = entropy_Y - (prob_x_0 * h_y_0 + prob_x_1 * h_y_1)

    return mutual_i




def grow_tree(train_attributes, train_label, max_depth, current_depth):
    attributes = train_attributes
#    print("what are attributes????", attributes)
    label = train_label
    optimal_attr = None #kept getting nonaccess error except here ??

#base cases
    if len(np.unique(label)) == 1:
        node = Node(vote=label[0])
#        print(f"what is vote = labe[0]??: {label[0]}")
#        print(f"Created leaf node with vote: {node.vote}")
        return node

    if current_depth >= max_depth:
        return Node(vote = majorityvoteclassifier(train_label))
    
#now we split on features by calculating mutual info
#we must keep track of optimal features
    curr_mutual_i = -1

    for attribute in range(len(attributes[0])):
        mutual_i = mutual_info(attributes, attribute, label)
        if mutual_i > curr_mutual_i:
            curr_mutual_i = mutual_i
            optimal_attr = attribute
#            print("wht is optimal attr??", optimal_attr) #added to debug output ------

#if mutual info is less than 0 than no good split is found; just return mjvote
    if curr_mutual_i <= 0:
        return Node(vote=majorityvoteclassifier(train_label))
    
#maintain split data between left and right with lists
#    print("wht is optimal attr here?", optimal_attr) #added to debug output------
    left_data, right_data, left_labels, right_labels = [], [], [], []
    for i in range(len(attributes)):
        if train_attributes[i][optimal_attr] == 0:
            left_data.append(attributes[i])
            left_labels.append(label[i])
        else:
            right_data.append(attributes[i])
            right_labels.append(label[i])
#    print(f"Left data size: {len(left_data)}, Right data size: {len(right_data)}")


#recurse to grow L & R sub tees
    left_node = grow_tree(left_data, left_labels, max_depth, current_depth + 1)
    right_node = grow_tree(right_data, right_labels, max_depth, current_depth + 1)
     
#return current node or optimal node with optimal subtree
    return Node(attr=optimal_attr, left=left_node, right=right_node) 


def predict(input_tree, attributes):
#we traverse tree essentially based on feature values
#if 0 we go left if 1 we go right
#return a list of votes or our predictions based on traversal

    predictions = []

    for row in attributes: 
        node = input_tree

        while node.left is not None and node.right is not None:
#            print(f"At node {node.attr}, going {'left' if row[node.attr] == 0 else 'right'}") #added to debug output :l
            if row[node.attr] == 0:
                node = node.left
            else:
                node = node.right

        predictions.append(node.vote)

#    print(f"Reached leaf node with vote: {node.vote}") #added to debug output :LL
    return predictions

def error_rate(true_label, predictions):
#take list of predictions and last column of test dataset
#compares/counts predict value towards test true label and counts num of incorrect predictions 
#non matching values towards predict value and then divides that number towards overall tot num of instances in test dataset
#error rate = count of nonmataching values / total num of rows in test dataset, return value 
    tot_true = len(true_label)
    incorrect_matches = np.sum(np.array(predictions) != np.array(true_label))
    
    return incorrect_matches / tot_true

def write_out_metrics(metrics_path, train_error_rate, test_error_rate):

    with open(metrics_path, 'w') as f:
        f.write(f"error(train): {train_error_rate:.6f}\n")
        f.write(f"error(test): {test_error_rate:.6f}\n")

def write_out_labels(train_out_path, prediction_train, test_out_path, prediction_test):
# write out train predicted labels & test predicted labesl
    with open(train_out_path, 'w') as f:
        for value in prediction_train:
            f.write(f"{value}\n")

    with open(test_out_path, 'w') as f:
        for value in prediction_test:
           f.write(f"{value}\n")

def pretty_print_tree(node, current_depth, f, feature_names):
# pretty print the tree to txt file as stated
    if node is None:
        return
    
    #leaf node
    if node.left is None and node.right is None:
        f.write(f"{'|' * current_depth} {node.vote}: [0 0/0 1] \n")
    else:
    #print everything within
        f.write(f"{'|' * current_depth} {feature_names[node.attr]} = 0: [0 0/0 1] \n")
        pretty_print_tree(node.left, current_depth + 1, f, feature_names)
        pretty_print_tree(node.right, current_depth + 1, f, feature_names)

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()
    
    # depth tracker
    current_depth = 0

    # we read our file and grab features and label
    train_attributes, train_label, feature_names_train = readfile(args.train_input)
    test_attributes, test_label, feature_names_test = readfile(args.test_input)

    # we grow our tree
    dtree  = grow_tree(train_attributes, train_label, args.max_depth, current_depth)

    # we predict
    prediction_train = predict(dtree, train_attributes)
    prediction_test = predict(dtree, test_attributes)

    # we calculate error rate
    train_error_rate = error_rate(train_label, prediction_train)
    test_error_rate = error_rate(test_label, prediction_test)


    # we write out label files, metrics and tree as stated
    write_out_labels(args.train_out, prediction_train, args.test_out, prediction_test)
    write_out_metrics(args.metrics_out, train_error_rate, test_error_rate)


    #Here is a recommended way to print the tree to a file
    with open(args.print_out, "w") as file:
        pretty_print_tree(dtree, current_depth, file, feature_names_test)