from collections import Counter
import numpy

# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """

    pos_vocab = {}
    neg_vocab = {}
    for i in range(0, len(X)):
        counter = Counter(X[i])
        if y[i] == 1:
            for word in counter:
                pos_vocab[word] = counter.get(word) + pos_vocab.get(word, 0)
        else:
            for word in counter:
                neg_vocab[word] = counter.get(word) + neg_vocab.get(word, 0)

    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """

    pos_vocab = {}
    neg_vocab = {}

    for i in range(0, len(X)):
        email = X[i]
        if y[i] == 1:
            for j in range(0, len(email)):
                word = email[j]
                pos_vocab[word] = pos_vocab.get(word, 0) + 1
            for j in range(0, len(email) - 1):
                w1 = email[j]
                w2 = email[j + 1]
                pair = w1 + " " + w2
                pos_vocab[pair] = pos_vocab.get(pair, 0) + 1
        else:
            for j in range(0, len(email)):
                word = email[j]
                neg_vocab[word] = neg_vocab.get(word, 0) + 1
            for j in range(0, len(email) - 1):
                w1 = email[j]
                w2 = email[j + 1]
                pair = w1 + " " + w2
                neg_vocab[pair] = neg_vocab.get(pair, 0) + 1
    return dict(pos_vocab), dict(neg_vocab)



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template

    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels)
    dev_labels = []

    uniqueHamWords = len(pos_vocab.keys())
    uniqueSpamWords = len(neg_vocab.keys())
    totalHamWords = sum(pos_vocab.values())
    totalSpamWords = sum(neg_vocab.values())

    for email in dev_set:
        hamProb = 0
        spamProb = 0
        for word in email:
            hamProb += smoothing(pos_vocab.get(word, 0), totalHamWords, uniqueHamWords, laplace)
            spamProb += smoothing(neg_vocab.get(word, 0), totalSpamWords, uniqueSpamWords, laplace)

        spamProb += math.log(1 - pos_prior)
        hamProb += math.log(pos_prior)

        if hamProb > spamProb:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    print_paramter_vals(laplace, pos_prior)
    return dev_labels


def smoothing(freq, totalWords, uniqueWords, k):
    return math.log((freq + k) / (totalWords + k * (1 + uniqueWords)))


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.0009, bigram_laplace=0.006, bigram_lambda=0.51,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''

    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None
    pos_vocab, neg_vocab = create_word_maps_bi(train_set, train_labels, max_vocab_size)
    uniqueUniHam, totalUniHam, uniqueUniSpam, totalUniSpam, uniqueBiHam, totalBiHam, uniqueBiSpam, totalBiSpam = countWords(pos_vocab, neg_vocab)

    dev_labels = []
    for email in dev_set:
        uniHamProb = 0
        uniSpamProb = 0
        biHamProb = 0
        biSpamProb = 0
        for word in email:
            uniHamProb += smoothing(pos_vocab.get(word, 0), totalUniHam, uniqueUniHam, unigram_laplace)
            uniSpamProb += smoothing(neg_vocab.get(word, 0), totalUniSpam, uniqueUniSpam, unigram_laplace)
        for i in range(0, len(email) - 1):
            pair = email[i] + " " + email[i + 1]
            biHamProb += smoothing(pos_vocab.get(pair, 0), totalBiHam, uniqueBiHam, bigram_laplace)
            biSpamProb += smoothing(neg_vocab.get(pair, 0), totalBiSpam, uniqueBiSpam, bigram_laplace)

        totalSpamProb = (1 - bigram_lambda) * (math.log(pos_prior) + uniSpamProb) + bigram_lambda * (math.log(pos_prior) + biSpamProb)
        totalHamProb = (1 - bigram_lambda) * (math.log(pos_prior) + uniHamProb) + bigram_lambda * (math.log(pos_prior) + biHamProb)
        if totalHamProb > totalSpamProb:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels


def countWords(pos_vocab, neg_vocab):
    uniqueUniSpam = 0
    uniqueUniHam = 0

    totalUniSpam = 0
    totalUniHam = 0

    uniqueBiHam = 0
    uniqueBiSpam = 0

    totalBiHam = 0
    totalBiSpam = 0

    for item in pos_vocab:
        if ' ' in item:
            uniqueBiHam += 1
            totalBiHam += pos_vocab.get(item)
        else:
            uniqueUniHam += 1
            totalUniHam += pos_vocab.get(item)
    for item in neg_vocab:
        if ' ' in item:
            uniqueBiSpam += 1
            totalBiSpam += neg_vocab.get(item)
        else:
            uniqueUniSpam += 1
            totalUniSpam += neg_vocab.get(item)

    return uniqueUniHam, totalUniHam, uniqueUniSpam, totalUniSpam, uniqueBiHam, totalBiHam, uniqueBiSpam, totalBiSpam
