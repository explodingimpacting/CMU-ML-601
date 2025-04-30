import csv
import numpy as np
import argparse
from collections import Counter

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def get_word_embedding(glove_map, review, label):
    # This method will take in glove embeddings and our dataset. 
    # Tokenize through review sentece, keep frequency count and check words in dict of words!
    # Calculate return a reduced sentence with associated calculated embeddings

    words = review.split() 
    frequency = Counter(words)
    word_embedding = []
    length = 0 #keep track of size of trimed sentenced

    for word, count in frequency.items():
        if word in glove_map:
            word_embedding.append(glove_map[word] * count)
            length += count


    #compute basic average by summing columns and then dividing my length of trimmed sentence
    sentence_embedding = np.array(word_embedding)
    sentence_embedding = np.sum(sentence_embedding, axis=0) 
    word_embedding_scaled = [
        embedding / length for embedding in sentence_embedding
    ]

    # just added label to front end
    word_embedding_scaled.insert(0, label)

    # kept getting point float printed, this fixed it
    word_embedding_scaled = [
        f"{embedding:.6f}" for embedding in word_embedding_scaled
    ]

    # return word_embedding, word_embedding_scaled, length
    return word_embedding_scaled

def format_data(dataset, glove_map, path):
    # This method will pretty print our sentence embeddings 
    with open(path, "w") as f:
        for i in range(len(dataset)):
            label, review = dataset[i]
            sentence_embedding = get_word_embedding(glove_map, review, label)
            f.write(f"{sentence_embedding[0]}\t" + "\t".join(map(str, sentence_embedding[1:])) + "\n")



if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()


    # grab our dicitonary of words
    glove_map = load_feature_dictionary(args.feature_dictionary_in)

    # train data
    train_dataset = load_tsv_dataset(args.train_input)
    # call train dataset and create output
    format_data(train_dataset, glove_map, args.train_out)

    # validation data
    validation_dataset = load_tsv_dataset(args.validation_input)
    # call validation dataset and create output
    format_data(validation_dataset, glove_map, args.validation_out)

    # test data
    test_dataset = load_tsv_dataset(args.test_input)
    # call test dataset and create output
    format_data(test_dataset, glove_map, args.test_out)





