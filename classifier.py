import nltk
import numpy
import pandas as pd
import sklearn
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB


# read and return features and labels
def get_authors_comments(path):
    model_table = pd.read_table(path, sep=',')
    authors = model_table['Author']
    comments = model_table['Comment']
    return authors, comments


# receives the path for .csv file
def nltk_naive_bayes(path):
    authors, comments = get_authors_comments(path)
    features_sets = []
    for index in range(len(features)):
        features = [' '.join(item) for item in nltk.bigrams(word_tokenize(features[i]))]
        feat_dict = { i : True for i in features }
        features_sets.append((feat_dict, authors[i]))
    train_set = features_sets[len(features_sets)-1:]
    test_set = features_sets[:len(features_sets)]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))


# classifies based on Gaussian Naive Bayes in sklearn
# 
# analyser could be 'char_wb' for characters in word boundaries
# or it could be 'char' for words in text
# 
# ngram_range is tuple of what number of n is used, e.g, (2,2), (3,3) and so on
def gaussian_nb(path, set_analyzer, ngram_range):
    authors, comments = get_authors_comments(path)
    ngram_vectorizer = CountVectorizer(analyzer=set_analyzer, ngram_range=ngram_range)
    counts = ngram_vectorizer.fit_transform(comments)
    data = counts.toarray()
    gnb = GaussianNB()
    y_pred = gnb.fit(data, authors).predict(data)
    print("Number of mislabeled points out of a total %d points : %d" % (data.shape[0],(authors != y_pred).sum()))

gaussian_nb('/home/guilhermecasimiro/SI-Folder/reddit-experiment/AA Dataset/2_authors_sub_1058648_comm.csv',
            'char_wb', (1, 1))
gaussian_nb('/home/guilhermecasimiro/SI-Folder/reddit-experiment/AA Dataset/2_authors_sub_1058648_comm.csv',
            'char_wb', (2, 2))
gaussian_nb('/home/guilhermecasimiro/SI-Folder/reddit-experiment/AA Dataset/2_authors_sub_1058648_comm.csv',
            'char_wb', (3, 3))
gaussian_nb('/home/guilhermecasimiro/SI-Folder/reddit-experiment/AA Dataset/2_authors_sub_1058648_comm.csv',
            'char_wb', (4, 4))
gaussian_nb('/home/guilhermecasimiro/SI-Folder/reddit-experiment/AA Dataset/2_authors_sub_1058648_comm.csv',
            'char_wb', (5, 5))