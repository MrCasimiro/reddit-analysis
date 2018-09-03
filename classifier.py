from __future__ import print_function

import nltk
import numpy as np
import pandas as pd
import sklearn
import itertools
import time
import matplotlib.pyplot as plt
from nltk import word_tokenize
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
from sklearn.naive_bayes import MultinomialNB


def get_file_len(path):
    csv_len = 0
    for chunk in pd.read_csv(path, chunksize = 1000):
        csv_len += len(chunk)
    return csv_len


def get_chunk_iter(path, chunksize):
    return pd.read_csv(path, sep = ',', iterator = True, chunksize = chunksize)


# read and return features and labels
def get_authors_comments(model_table):
    authors = model_table['Author']
    comments = model_table['Comment']
    return authors, comments


def adjust_labels(authors):
    labels = authors.unique()
    for author in labels:
        if not labels_dict.__contains__(author):
            # add the new author to authors label dictionary
            if labels_dict.__len__() == 0:
                labels_dict[author] = 0
            else:
                labels_dict[author] = list(labels_dict.values())[-1] + 1

    new_authors = [labels_dict[author] for author in authors]
    return labels_dict, np.array(new_authors) 


def plot_confusion_matrix(cm, classes, name, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
    plt.savefig('Results/Confusion Matrix/' + name + '.png')
    plt.close()

    
### Classifiers setup

# list of classifiers it implements partial fit for online learning
partial_fit_classifiers = {
    'SGD': SGDClassifier(),
    'Perceptron': Perceptron(),
    # 'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(),
}

classifiers_conf_matrix = {
    'SGD': np.array(([0, 0], [0, 0])),
    'Perceptron': np.array(([0, 0], [0, 0])),
    # 'NB Multinomial': np.array(([0, 0], [0, 0])),
    'Passive-Aggressive': np.array(([0, 0], [0, 0])),
}

types_analyser = ['char_wb', 'char', 'word']
types_ngram = [(1, 1), (2, 2), (3, 3)]
kf_number = 5

all_classes = np.array([0, 1])
path = 'AA Dataset/2_authors_sub_1058648_comm.csv'
chunksize = 2000
chunk_num_iterations = np.ceil(get_file_len(path) / chunksize)
labels_dict = OrderedDict()
# finish setup


file_iterator = get_chunk_iter(path, chunksize)
# iterates every chunk of csv
while True:
    try:
        rows = file_iterator.get_chunk()
        authors, comments = get_authors_comments(rows)
        labels, new_authors = adjust_labels(authors)
        ngram_vectorizer = HashingVectorizer(analyzer=types_analyser[0], ngram_range=types_ngram[0])

        for cls_name, cls in partial_fit_classifiers.items():
            data = ngram_vectorizer.transform(comments)
            kf = KFold(n_splits=kf_number, shuffle = True, random_state = 1)
            for train_index, test_index in kf.split(data, new_authors):
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = new_authors[train_index], new_authors[test_index]
                cls.partial_fit(X_train, y_train, classes=all_classes)
                y_pred = cls.predict(X_test)
                classifiers_conf_matrix[cls_name] += confusion_matrix(y_test, y_pred)
    except StopIteration:
        print("Finish")
        break
    except Exception as exc:
        print("Error")
        print(exc)
        raise        

for cls_name, conf_matrix in classifiers_conf_matrix.items():
    conf_matrix = np.divide(conf_matrix, (chunk_num_iterations * kf_number))
    plot_confusion_matrix(conf_matrix, list(labels.keys()),
                          cls_name + ' with_normalization', True, cls_name)

# receives the path for .csv file
def nltk_naive_bayes(path):
    authors, comments = get_authors_comments(path)
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


# path = 'AA Dataset/2_authors_sub_1058648_comm.csv'
# authors, comments = get_authors_comments(path)
# gaussian_nb(authors, comments,
#             'char_wb', (1, 1))
# gaussian_nb(authors, comments,
#             'char_wb', (2, 2))

# gaussian_nb(authors, comments,
#             'char', (1, 1))
# gaussian_nb(authors, comments,
#             'char', (2, 2))

# gaussian_nb(authors, comments,
#             'word', (1, 1))
# gaussian_nb(authors, comments,
#             'word', (2, 2))

# support_machines_vectors(authors, comments,
#             'char_wb', (1, 1))
# support_machines_vectors(authors, comments,
#             'char_wb', (2, 2))
# support_machines_vectors(authors, comments,
#             'char', (1, 1))
# support_machines_vectors(authors, comments,
#             'char', (2, 2))
# support_machines_vectors(authors, comments,
#             'word', (1, 1))
# support_machines_vectors(authors, comments,
#             'word', (2, 2))

