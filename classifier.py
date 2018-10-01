from __future__ import print_function

import nltk
import numpy as np
import pandas as pd
import sklearn
import itertools
import matplotlib.pyplot as plt
from nltk import word_tokenize, pos_tag
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import *
from scipy import *


# Get the csv row length
def get_file_len(PATH):
    csv_len = 0
    for chunk in pd.read_csv(PATH, chunksize = 1000):
        csv_len += len(chunk)
    return csv_len


# Returns chunk iterator over csv
def get_chunk_iter(PATH, chunksize):
    return pd.read_csv(PATH, sep = ',', iterator = True, chunksize = chunksize)


# read and return features (comments) and labels (authors)
def get_authors_comments(model_table):
    authors = model_table['Author']
    comments = model_table['Comment']
    return authors, comments


# Maps Author to an integer starts with 0
def adjust_authors_dict(PATH):
    for chunk in pd.read_csv(PATH, chunksize = 1000):
        authors = chunk['Author'].unique()
        for author in authors:
            if not LABELS_DICT.__contains__(author):
                # add the new author to authors label dictionary
                if LABELS_DICT.__len__() == 0:
                    LABELS_DICT[author] = 0
                else:
                    LABELS_DICT[author] = list(LABELS_DICT.values())[-1] + 1


# Returns a new array mapping the authors name to an integer
def adjust_labels(authors):
    labels = authors.unique()
    new_authors = [LABELS_DICT[author] for author in authors]
    return np.array(new_authors) 


# It plots the confusion matrix
def plot_confusion_matrix(cm, classes, name, normalize=False, title='Confusion matrix'):
    cmap=plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('' + name + '.png')
    plt.close()


def evaluate(cls_name, cls, data, new_authors, kf):
    for train_index, test_index in kf.split(data, new_authors):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = new_authors[train_index], new_authors[test_index]
        cls.partial_fit(X_train, y_train, classes=ALL_CLASSES)
        y_pred = cls.predict(X_test)
        CLASSIFIERS_CONF_MATRIX[cls_name] += confusion_matrix(y_test, y_pred, list(LABELS_DICT.values()))


def classify_pos_tag(PATH, ngram, cv = 5):
    file_iterator = get_chunk_iter(PATH, CHUNKSIZE)
    chunk_num_iterations = np.ceil(get_file_len(PATH) / CHUNKSIZE)
    ngram_vectorizer = HashingVectorizer(ngram_range = ngram, analyzer = 'word')
    while True:
        try:
            rows = file_iterator.get_chunk()
            authors, comments = get_authors_comments(rows)
            new_authors = adjust_labels(authors)
            comm_pos_tags = []
            for comm in comments:
                pos_tag_tmp = np.array(pos_tag(word_tokenize(comm)))[:, 1]
                comm_pos_tags.append(' '.join(pos_tag_tmp))
            for cls_name, cls in PARTIAL_FIT_CLASSIFIERS.items():
                # pos_tag_comm = [pos_tag(word_tokenize(item)) for item in comments]
                data = ngram_vectorizer.transform(comm_pos_tags)
                kf = KFold(n_splits = cv, shuffle = True, random_state = 1)
                evaluate(cls_name, cls, data, new_authors, kf)
        except StopIteration:
            print("Finish")
            break
        except Exception as exc:
            print("Error")
            print(exc)
            raise
        name = 'POS_TAG' + '_' + str(ngram) + '_with_normalization'
        for cls_name, conf_matrix in CLASSIFIERS_CONF_MATRIX.items():
            conf_matrix = np.divide(conf_matrix, (chunk_num_iterations * cv))
            plot_confusion_matrix(conf_matrix, list(LABELS_DICT.keys()),
                              cls_name + name, True, cls_name)


# main method that run every configuration setted to all classifiers defined
def classify_and_plot(PATH, analyzer, ngram, cv = 5, pos = False):
    file_iterator = get_chunk_iter(PATH, CHUNKSIZE)
    chunk_num_iterations = np.ceil(get_file_len(PATH) / CHUNKSIZE)
    ngram_vectorizer = HashingVectorizer(ngram_range = ngram, analyzer = analyzer)
    # iterates every chunk of csv
    while True:
        try:
            rows = file_iterator.get_chunk()
            authors, comments = get_authors_comments(rows)
            new_authors = adjust_labels(authors)
            for cls_name, cls in PARTIAL_FIT_CLASSIFIERS.items():
                comments_transform = comments
                if pos:
                    comments_transform = []
                    for comm in comments:
                        pos_tag_tmp = np.array(pos_tag(word_tokenize(comm)))[:, 1]
                        comments_transform.append(' '.join(pos_tag_tmp))
                data = ngram_vectorizer.transform(comments_transform)
                kf = KFold(n_splits = cv, shuffle = True, random_state = 1)
                evaluate(cls_name, cls, data, new_authors, kf)
        except StopIteration:
            print("Finish")
            break
        except Exception as exc:
            print("Error")
            print(exc)
            raise
    name = '_' + analyzer + '_' + str(ngram) + '_with_normalization'
    for cls_name, conf_matrix in CLASSIFIERS_CONF_MATRIX.items():
        conf_matrix = np.divide(conf_matrix, (chunk_num_iterations * cv))
        plot_confusion_matrix(conf_matrix, list(LABELS_DICT.keys()),
                              cls_name + name, True, cls_name)


# receives the PATH for .csv file
def nltk_naive_bayes(PATH):
    authors, comments = get_authors_comments(PATH)
    features_sets = []
    for index in range(len(comments)):
        features = [' '.join(item) for item in nltk.bigrams(word_tokenize(comments[index]))]
        feat_dict = { i : True for i in features }
        features_sets.append((feat_dict, authors[index]))
    random.shuffle(features_sets)
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
# def gaussian_nb(authors, comments, set_analyzer, ngram_range):
#     labels, new_authors = adjust_labels(authors)
#     ngram_vectorizer = CountVectorizer(analyzer=set_analyzer, ngram_range=ngram_range)
#     counts = ngram_vectorizer.fit_transform(comments)
#     data = counts.toarray()
#     gnb = GaussianNB()
#     y_pred = cross_val_predict(gnb, data, new_authors, cv = 5)
#     title = 'Confusion Matrix - ' + 'Gaussian Naive Bayes'
#     plot_name = 'Gaussian_NB_' + set_analyzer + '_' + str(ngram_range) + '_'
#     plot_confusion_matrix(confusion_matrix(new_authors, y_pred), list(labels.keys()),
#                           plot_name + 'without_normalization', False, title)
#     plot_confusion_matrix(confusion_matrix(new_authors, y_pred), list(labels.keys()),
#                           plot_name + 'with_normalization', True, title)


# def support_machines_vectors(authors, comments, set_analyzer, ngram_range):
#     labels, new_authors = adjust_labels(authors)
#     ngram_vectorizer = CountVectorizer(analyzer=set_analyzer, ngram_range=ngram_range)
#     counts = ngram_vectorizer.fit_transform(comments)
#     data = counts.toarray()
#     clf = svm.SVC()
#     y_pred = cross_val_predict(clf, data, new_authors, cv = 5)
#     title = 'Confusion Matrix - ' + 'Support Machine Vector'
#     plot_name = 'SVM_' + set_analyzer + '_' + str(ngram_range) + '_'
#     plot_confusion_matrix(confusion_matrix(new_authors, y_pred), list(labels.keys()),
#                           plot_name + 'without_normalization', False, title)
#     plot_confusion_matrix(confusion_matrix(new_authors, y_pred), list(labels.keys()),
#                           plot_name + 'with_normalization', True, title)


### Classifiers setup

# list of classifiers it implements partial fit for online learning
LABELS_DICT = OrderedDict()
PATH = 'AA Dataset/2_authors_sub_1058648_comm.csv'
adjust_authors_dict(PATH)
AUTHORS_NUM = len(LABELS_DICT)
PARTIAL_FIT_CLASSIFIERS = {
    'SGD': SGDClassifier(),
    'Perceptron': Perceptron(),
    # 'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(),
}

CLASSIFIERS_CONF_MATRIX = {
    'SGD': np.zeros((AUTHORS_NUM, AUTHORS_NUM)),
    'Perceptron': np.zeros((AUTHORS_NUM, AUTHORS_NUM)),
    # 'NB Multinomial': np.array(([0, 0], [0, 0])),
    'Passive-Aggressive': np.zeros((AUTHORS_NUM, AUTHORS_NUM)),
}

KF_NUM = 5

ALL_CLASSES = np.array(range(0, AUTHORS_NUM))
CHUNKSIZE = 1000
# finish setup

# classify_and_plot(PATH, 'char', (1, 1), 5)
# classify_and_plot(PATH, 'char', (2, 2), 5)
# classify_and_plot(PATH, 'char', (3, 3), 5)
# classify_and_plot(PATH, 'char', (4, 4), 5)
# classify_and_plot(PATH, 'char', (5, 5), 5)


# classify_and_plot(PATH, 'char_wb', (1, 1), 5)
# classify_and_plot(PATH, 'char_wb', (2, 2), 5)
# classify_and_plot(PATH, 'char_wb', (3, 3), 5)
# classify_and_plot(PATH, 'char_wb', (4, 4), 5)
# classify_and_plot(PATH, 'char_wb', (5, 5), 5)


# classify_and_plot(PATH, 'word', (1, 1), 5)
# classify_and_plot(PATH, 'word', (2, 2), 5)
# classify_and_plot(PATH, 'word', (3, 3), 5)
# classify_and_plot(PATH, 'word', (4, 4), 5)
# classify_and_plot(PATH, 'word', (5, 5), 5)

classify_pos_tag(PATH, (1, 1), 5)
classify_pos_tag(PATH, (2, 2), 5)
classify_pos_tag(PATH, (3, 3), 5)