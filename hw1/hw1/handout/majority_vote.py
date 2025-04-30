"""
Author: Roxxanne White
Date: 1/19/2025
File Name: majority_vote.py
Description: This file will implement a Majority Vote Classifier. It will calculate most common label
in a given dataset, predict upon that label towards in a dataset and then calculate error metrics. Gathered
calculations will be outputed to 3 txt files given respectively: train out, test out and metrics out. This 
file assumes all class labels are binary. 

"""
import numpy as np
import sys

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

def predict(majority_vote_label, input_label_set, output_path):
#this method will take in as parameters most common label & test dataset
#We will make a list of the same size of test dataset (num of rows) and fill it with common label value
#we will return a list with size of dataset we are testing against with filled common label value
#then print out to text file
    predictions = [majority_vote_label] * len(input_label_set)

    with open(output_path, 'w') as f:
        for value in predictions:
            f.write(f"{value}\n")

    return predictions

def error_rate(true_labels, predictions):
#take list of predictions and last column of test dataset
#compares/counts predict value towards test true label and counts num of incorrect predictions 
#non matching values towards predict value and then divides that number towards overall tot num of instances in test dataset
#error rate = count of nonmataching values / total num of rows in test dataset, return value 
    tot_true = len(true_labels)
    incorrect_matches = np.sum(np.array(predictions) != np.array(true_labels))
    
    return incorrect_matches / tot_true

def write_out_metrics(outputfile3, train_error_rate, test_error_rate):

    with open(outputfile3, 'w') as f:
         f.write(f"error(train): {train_error_rate:.6f}\n")
         f.write(f"error(test): {test_error_rate:.6f}")



def main(train_input, test_input):
#we read our files & grab last columns
    train_labels = readfile(train_input)
    test_labels = readfile(test_input)

# we find most common label
    majority_vote_label = majorityvoteclassifier(train_labels)

# we predict
    predictions_train = predict(majority_vote_label, train_labels, outputfile1)
    predictions_test = predict(majority_vote_label, test_labels, outputfile2)

# we calculate error rate against test and train datasets
    test_error_rate = error_rate(test_labels, predictions_test)
    train_error_rate = error_rate(train_labels, predictions_train)

# we write out last metrics.txt file
    write_out_metrics(outputfile3, train_error_rate, test_error_rate)


if __name__ == '__main__':
    infile1 = sys.argv[1]
    infile2 = sys.argv[2]
    outputfile1 = sys.argv[3]
    outputfile2 = sys.argv[4]
    outputfile3 = sys.argv[5]

    main(infile1, infile2)