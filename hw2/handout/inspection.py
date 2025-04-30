"""
Author: Roxxanne White
Date: 1/31/2025
File Name: inspection.py
Description: This file will implement a Majority Vote Classifier. It will calculate most common label
in a given dataset, predict upon that label towards in a dataset and then calculate error metrics. This 
file assumes all class labels are binary. It will also implement entropy calculation. It will then output
a text file of dubbed <name of training data>_inspect.txt

"""
import numpy as np
import sys
import math

def readfile(input_file):
#this method reads in test data sets
#we essentially only care about the last column and want to return that

    data = np.genfromtxt(input_file, delimiter='\t', dtype=None, skip_header= 1)
    labels = data[:, -1]  

    return labels
    
def majorityvoteclassifier(training_labels):
#this method will find majority vote lol; most common label
#we will take in lastcolumn and create dictionary with each value seen and count +1 for each occurence
#we will return most common label or majority voted label
    counts = {}

    for value in training_labels: 
        if value in counts: 
            counts[value] += 1
        else:
            counts[value] = 1

    #print("this is majority counts ", counts)
    majority_vote = max(counts, key=counts.get)
    #print("this is majority vote: ", majority_vote)
    
    return majority_vote

def entropy(train_labels):
#to calculate entropy we must take tot instances for label
#keep count of each instance. Then apply entropy formula: h(x) = -(P(x) * log2(P(x)))
    tot_instances = len(train_labels)
    counts = {}

    for value in train_labels:
        if value in counts:
            counts[value] += 1
        else: 
            counts[value] = 1
    
    prob_0 = counts.get(0) / tot_instances
    prob_1 = counts.get(1) / tot_instances
    entropy_value = - ((prob_0 * math.log2(prob_0)) + prob_1 * math.log2(prob_1))
     
    #print("this is value for key 0", prob_0)
    #print("this is keys for counts", counts.keys())
    #print("this is key values", counts.items()) """

    return entropy_value

def predict(majority_vote_label, input_label_set, output_path):
#this method will take in as parameters most common label & test dataset
#We will make a list of the same size of test dataset (num of rows) and fill it with common label value
#we will return a list with size of dataset we are testing against with filled common label value
#then print out to text file
    predictions = [majority_vote_label] * len(input_label_set)

    return predictions

def error_rate(true_labels, predictions):
#take list of predictions and last column of test dataset
#compares/counts predict value towards test true label and counts num of incorrect predictions 
#non matching values towards predict value and then divides that number towards overall tot num of instances in test dataset
#error rate = count of nonmataching values / total num of rows in test dataset, return value 
    tot_true = len(true_labels)
    incorrect_matches = np.sum(np.array(predictions) != np.array(true_labels))
    
    return incorrect_matches / tot_true

def write_out_metrics(outputfile1, train_error_rate, entropy_value):

    with open(outputfile1, 'w') as f:
         f.write(f"entropy: {entropy_value:.6f}\n")
         f.write(f"error: {train_error_rate:.6f}")



def main(train_input, outputfile1):
#we read our file & grab last column
    train_labels = readfile(train_input)

# we find most common label
    majority_vote_label = majorityvoteclassifier(train_labels)

# we predict
    predictions_train = predict(majority_vote_label, train_labels, outputfile1)

# we calculate error rate 
    train_error_rate = error_rate(train_labels, predictions_train)

# we calculate entropy
    entropy_value = entropy(train_labels)

# we write out last metrics.txt file
    write_out_metrics(outputfile1, train_error_rate, entropy_value)


if __name__ == '__main__':
    infile1 = sys.argv[1]
    outputfile1 = sys.argv[2]

    main(infile1, outputfile1)