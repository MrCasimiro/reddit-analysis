from __future__ import print_function

import glob
import nltk
import numpy as np
import pandas as pd
import sklearn
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support as score
from nltk import word_tokenize, pos_tag
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
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


def adjust_one_x_all(auth):
    ONE_X_ALL_DICT = OrderedDict()
    ONE_X_ALL_DICT['Others'] = 0
    ONE_X_ALL_DICT[auth] = 1
    return ONE_X_ALL_DICT


def adjust_one_x_all_labels(author, authors):
    new_authors = []
    new_labels = []
    for auth in authors:
        if auth == author:
            new_authors.append(1)
        else:
            new_authors.append(0)
    return np.array(new_authors)


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


def return_classifiers():
    PARTIAL_FIT_CLASSIFIERS = {
        'SGD': SGDClassifier(),
        'Perceptron': Perceptron(),
        # 'NB Multinomial': MultinomialNB(alpha=0.01),
        'Passive-Aggressive': PassiveAggressiveClassifier(),
    }
    return PARTIAL_FIT_CLASSIFIERS


def return_conf_matrix(authors_num):
    CLASSIFIERS_CONF_MATRIX = {
        'SGD': np.zeros((authors_num, authors_num)),
        'Perceptron': np.zeros((authors_num, authors_num)),
        # 'NB Multinomial': np.array(([0, 0], [0, 0])),
        'Passive-Aggressive': np.zeros((authors_num, authors_num)),
    }
    return  CLASSIFIERS_CONF_MATRIX


def return_kf_hash():
    return {
        'kf-1': {
            'test': list(),
            'score': list()
        },
        'kf-2': {
            'test': list(),
            'score': list()
        },
        'kf-3': {
            'test': list(),
            'score': list()
        },
        'kf-4': {
            'test': list(),
            'score': list()
        },
        'kf-5': {
            'test': list(),
            'score': list()
        }
    }


def return_roc_data():
    ROC_VALUES = {
        'SGD': {
            'kf': return_kf_hash()
        },
        'Perceptron': {
            'kf': return_kf_hash()
        },
        'Passive-Aggressive': {
            'kf': return_kf_hash()
        }
    }
    return ROC_VALUES


# It plots the confusion matrix
def plot_confusion_matrix(cm, classes, name, normalize=False, title='Confusion matrix'):
    cmap=plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams["figure.figsize"] = (18,18)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60)
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


def save_confusion_matrix(file_name, confusion_matrix, class_names):
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    df_cm.to_csv(file_name)


def save_roc_data(file_name, cls_name, ROC_VALUES):
    data = list()
    for i in range(1,6):
        data.append(ROC_VALUES[cls_name]['kf']['kf-'+str(i)]['test'])
        data.append(ROC_VALUES[cls_name]['kf']['kf-'+str(i)]['score'])
    df_cm = pd.DataFrame(
       data 
    )
    df_cm.to_csv(file_name)


def print_confusion_matrix(confusion_matrix, class_names, name, figsize = (16,14), fontsize=13):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label',fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.savefig('' + name + '.png')
    plt.close()


def evaluate(CLASSIFIERS_CONF_MATRIX, cls_name, cls, data, new_authors, kf, ROC_VALUES):
    i = 1
    for train_index, test_index in kf.split(data, new_authors):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = new_authors[train_index], new_authors[test_index]
        cls.partial_fit(X_train, y_train, classes=ALL_CLASSES)
        y_pred = cls.predict(X_test)
        probas_ = cls.decision_function(X_test)
        ROC_VALUES[cls_name]['kf']['kf-'+ str(i)]['test'].extend(y_test)
        if(len(np.unique(new_authors)) == 2):
            ROC_VALUES[cls_name]['kf']['kf-'+ str(i)]['score'].extend(probas_)
        else:
            ROC_VALUES[cls_name]['kf']['kf-'+ str(i)]['score'].extend(y_pred)
        i += 1
        CLASSIFIERS_CONF_MATRIX[cls_name] += confusion_matrix(y_test, y_pred, list(LABELS_DICT.values()))


def classify_one_x_all(PATH, ten_authors, analyzer, ngram, cv = 5, pos = False):
    for author in ten_authors:
        ONE_X_ALL_DICT = adjust_one_x_all(author)
        file_iterator = get_chunk_iter(PATH, CHUNKSIZE)    
        chunk_num_iterations = np.ceil(get_file_len(PATH) / CHUNKSIZE)
        ngram_vectorizer = HashingVectorizer(ngram_range = ngram, analyzer = analyzer)
        PARTIAL_FIT_CLASSIFIERS = return_classifiers()
        CLASSIFIERS_CONF_MATRIX = return_conf_matrix(2)
        ROC_VALUES = return_roc_data()
        tmp = 0
        vl_error = 0
        while True:
            try:
                rows = file_iterator.get_chunk()
                authors, comments = get_authors_comments(rows)
                new_authors = adjust_one_x_all_labels(author, authors)
                for cls_name, cls in PARTIAL_FIT_CLASSIFIERS.items():
                    comments_transform = comments.values.astype('U')
                    if pos:
                        comments_transform = []
                        for comm in comments:
                            pos_tag_tmp = np.array(pos_tag(word_tokenize(comm)))[:, 1]
                            comments_transform.append(' '.join(pos_tag_tmp))
                    data = ngram_vectorizer.transform(comments_transform)
                    kf = KFold(n_splits = cv, shuffle = True, random_state = 1)
                    # print(new_authors)
                    i = 1
                    for train_index, test_index in kf.split(data, new_authors):
                        X_train, X_test = data[train_index], data[test_index]
                        y_train, y_test = new_authors[train_index], new_authors[test_index]
                        cls.partial_fit(X_train, y_train, classes=np.array(range(0, 2)))
                        y_pred = cls.predict(X_test)
                        probas_ = cls.decision_function(X_test)
                        ROC_VALUES[cls_name]['kf']['kf-'+ str(i)]['test'].extend(y_test)
                        ROC_VALUES[cls_name]['kf']['kf-'+ str(i)]['score'].extend(probas_)
                        i += 1
                        CLASSIFIERS_CONF_MATRIX[cls_name] += confusion_matrix(y_test, y_pred, list(ONE_X_ALL_DICT.values()))
            except ValueError:
                print('ValueError')
                vl_error += 1000
                next
            except StopIteration:
                print("Finish")
                break
            except Exception as exc:
                print("Error")
                print(exc)
                raise
        print('errro    ' + str(vl_error))
        print(tmp)
        for cls_name, conf_matrix in CLASSIFIERS_CONF_MATRIX.items():
            name = 'X_ALL_' + cls_name + '_' + analyzer + '_' + str(ngram)
            conf_matrix = np.divide(conf_matrix, (chunk_num_iterations * cv))
            # print_confusion_matrix(conf_matrix, list(ONE_X_ALL_DICT.keys()),
            #      author + '_' + name)
            save_confusion_matrix(author + '_' + name + '.csv', conf_matrix, list(ONE_X_ALL_DICT.keys()))
            save_roc_data('ROC_' + author + '_' + name + '.csv', cls_name, ROC_VALUES)


def classify_pos_tag(PATH, ngram, cv = 5):
    file_iterator = get_chunk_iter(PATH, CHUNKSIZE)
    chunk_num_iterations = np.ceil(get_file_len(PATH) / CHUNKSIZE)
    ngram_vectorizer = HashingVectorizer(ngram_range = ngram, analyzer = 'word')
    PARTIAL_FIT_CLASSIFIERS = return_classifiers()
    CLASSIFIERS_CONF_MATRIX = return_conf_matrix(AUTHORS_NUM)
    ROC_VALUES = return_roc_data()
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
                evaluate(CLASSIFIERS_CONF_MATRIX, cls_name, cls, data, new_authors, kf, ROC_VALUES)
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
        # print_confusion_matrix(conf_matrix, list(LABELS_DICT.keys()),
        #      cls_name + name)
        # save_confusion_matrix(cls_name + '_' + name + '.csv', conf_matrix, list(LABELS_DICT.keys()))
        save_roc_data('ROC_' + cls_name + '_' + name + '.csv', cls_name, ROC_VALUES)


# main method that run every configuration setted to all classifiers defined
def classify_and_plot(PATH, analyzer, ngram, cv = 5, pos = False):
    file_iterator = get_chunk_iter(PATH, CHUNKSIZE)
    chunk_num_iterations = np.ceil(get_file_len(PATH) / CHUNKSIZE)
    ngram_vectorizer = HashingVectorizer(ngram_range = ngram, analyzer = analyzer)
    PARTIAL_FIT_CLASSIFIERS = return_classifiers()
    CLASSIFIERS_CONF_MATRIX = return_conf_matrix(AUTHORS_NUM)
    ROC_VALUES = return_roc_data()
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
                evaluate(CLASSIFIERS_CONF_MATRIX, cls_name, cls, data, new_authors, kf, ROC_VALUES)
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
        # print_confusion_matrix(conf_matrix, list(LABELS_DICT.keys()),
        #      cls_name + name)
        # save_confusion_matrix(cls_name + '_' + name + '.csv', conf_matrix, list(LABELS_DICT.keys()))
        save_roc_data('ROC_' + cls_name + '_' + name + '.csv', cls_name, ROC_VALUES)


### Classifiers setup

# list of classifiers it implements partial fit for online learning
LABELS_DICT = OrderedDict()
PATH = 'AA Dataset/10_authors_sub_1058648_comm.csv'
adjust_authors_dict(PATH)

# adjust_all_10_authors()
AUTHORS_NUM = len(LABELS_DICT)

PATH_SUB = 'AA Dataset/authors_sub_1058648.csv'
PATH_ALL = 'AA Dataset/all_authors_sub_1058648_comm.csv'

ten_authors = pd.read_csv(PATH_SUB, sep=',', nrows=10)['Author'] 

KF_NUM = 5

ALL_CLASSES = np.array(range(0, AUTHORS_NUM))
CHUNKSIZE = 10000
# finish setup



# classify_one_x_all(PATH_ALL, ten_authors, 'char', (1, 1), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'char', (2, 2), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'char', (3, 3), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'char', (4, 4), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'char', (5, 5), 5)


classify_one_x_all(PATH_ALL, ten_authors, 'char_wb', (1, 1), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'char_wb', (2, 2), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'char_wb', (3, 3), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'char_wb', (4, 4), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'char_wb', (5, 5), 5)

classify_one_x_all(PATH_ALL, ten_authors, 'word', (1, 1), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'word', (2, 2), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'word', (3, 3), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'word', (4, 4), 5)
classify_one_x_all(PATH_ALL, ten_authors, 'word', (5, 5), 5)


# classify_and_plot(PATH, 'char', (1, 1), 5)
# classify_and_plot(PATH, 'char', (2, 2), 5)
# classify_and_plot(PATH, 'char', (3, 3), 5)
# classify_and_plot(PATH, 'char', (4, 4), 5)
# classify_and_plot(PATH, 'char', (5, 5), 5)
# classify_and_plot(PATH, 'char', (1, 5), 5)


# classify_and_plot(PATH, 'char_wb', (1, 1), 5)
# classify_and_plot(PATH, 'char_wb', (2, 2), 5)
# classify_and_plot(PATH, 'char_wb', (3, 3), 5)
# classify_and_plot(PATH, 'char_wb', (4, 4), 5)
# classify_and_plot(PATH, 'char_wb', (5, 5), 5)
# classify_and_plot(PATH, 'char_wb', (1, 5), 5)


# classify_and_plot(PATH, 'word', (1, 1), 5)
# classify_and_plot(PATH, 'word', (2, 2), 5)
# classify_and_plot(PATH, 'word', (3, 3), 5)
# classify_and_plot(PATH, 'word', (4, 4), 5)
# classify_and_plot(PATH, 'word', (5, 5), 5)
# classify_and_plot(PATH, 'word', (1, 5), 5)


# classify_pos_tag(PATH, (1, 1), 5)
# classify_pos_tag(PATH, (2, 2), 5)
# classify_pos_tag(PATH, (3, 3), 5)
# classify_pos_tag(PATH, (4, 4), 5)
# classify_pos_tag(PATH, (5, 5), 5)
# classify_pos_tag(PATH, (1, 5), 5)
