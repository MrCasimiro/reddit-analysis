import nltk
import numpy as np
import pandas as pd
import sklearn
import itertools
import matplotlib.pyplot as plt
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# read and return features and labels
def get_authors_comments(path):
    model_table = pd.read_table(path, sep=',')
    authors = model_table['Author']
    comments = model_table['Comment']
    return authors, comments


def adjust_labels(authors):
    labels = authors.unique()
    labels_dict = {}
    for i in range(len(labels)): labels_dict[labels[i]] = i
    new_authors = [labels_dict[author] for author in authors]
    return labels_dict, new_authors 


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
    labels, new_authors = adjust_labels(authors)
    ngram_vectorizer = CountVectorizer(analyzer=set_analyzer, ngram_range=ngram_range)
    counts = ngram_vectorizer.fit_transform(comments)
    data = counts.toarray()
    gnb = GaussianNB()
    y_pred = cross_val_predict(gnb, data, new_authors, cv = 5)
    title = 'Confusion Matrix - ' + 'Gaussian Naive Bayes'
    plot_name = 'Gaussian_NB_' + set_analyzer + '_' + str(ngram_range) + '_'
    plot_confusion_matrix(confusion_matrix(new_authors, y_pred), list(labels.keys()),
                          plot_name + 'without_normalization', False, title)
    plot_confusion_matrix(confusion_matrix(new_authors, y_pred), list(labels.keys()),
                          plot_name + 'with_normalization', True, title)


def support_machines_vectors(path, set_analyzer, ngram_range):
    authors, comments = get_authors_comments(path)
    labels, new_authors = adjust_labels(authors)
    ngram_vectorizer = CountVectorizer(analyzer=set_analyzer, ngram_range=ngram_range)
    counts = ngram_vectorizer.fit_transform(comments)
    data = counts.toarray()
    clf = svm.SVC()
    y_pred = cross_val_predict(clf, data, new_authors, cv = 5)
    title = 'Confusion Matrix - ' + 'Support Machine Vector'
    plot_name = 'SVM_' + set_analyzer + '_' + str(ngram_range) + '_'
    plot_confusion_matrix(confusion_matrix(new_authors, y_pred), list(labels.keys()),
                          plot_name + 'without_normalization', False, title)
    plot_confusion_matrix(confusion_matrix(new_authors, y_pred), list(labels.keys()),
                          plot_name + 'with_normalization', True, title)


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

# gaussian_nb('AA Dataset/2_authors_sub_1058648_comm.csv',
#             'char_wb', (1, 1))
# gaussian_nb('AA Dataset/2_authors_sub_1058648_comm.csv',
#             'char_wb', (2, 2))

# gaussian_nb('AA Dataset/2_authors_sub_1058648_comm.csv',
#             'char', (1, 1))
# gaussian_nb('AA Dataset/2_authors_sub_1058648_comm.csv',
#             'char', (2, 2))

# gaussian_nb('AA Dataset/2_authors_sub_1058648_comm.csv',
#             'word', (1, 1))
# gaussian_nb('AA Dataset/2_authors_sub_1058648_comm.csv',
#             'word', (2, 2))

support_machines_vectors('AA Dataset/2_authors_sub_1058648_comm.csv',
            'char_wb', (1, 1))
support_machines_vectors('AA Dataset/2_authors_sub_1058648_comm.csv',
            'char_wb', (2, 2))
support_machines_vectors('AA Dataset/2_authors_sub_1058648_comm.csv',
            'char', (1, 1))
support_machines_vectors('AA Dataset/2_authors_sub_1058648_comm.csv',
            'char', (2, 2))
support_machines_vectors('AA Dataset/2_authors_sub_1058648_comm.csv',
            'word', (1, 1))
# support_machines_vectors('AA Dataset/2_authors_sub_1058648_comm.csv',
#             'word', (2, 2))

