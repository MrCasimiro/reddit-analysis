import glob
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp


def accuracy(df):
    df = df.drop('Unnamed: 0', 1)
    diag = pd.Series(np.diag(df))
    trues = np.sum(np.array(diag))
    df_array = np.array(df)
    total = np.sum(df_array[:,:])
    return trues / total


def plot_2_authors(name, *n_grams):
    grams = list(n_grams)
    gram_labels = list(np.arange(1, len(grams) + 1))
    gram_labels[5] = '1-5'
    i = 0
    fig = plt.figure()
    ax = plt.subplot(111)
    for gram in grams:
        gram.sort()
        passive = pd.read_csv(gram[0])
        perc = pd.read_csv(gram[1])
        sgd = pd.read_csv(gram[2])
        acc = [accuracy(passive), accuracy(perc), accuracy(sgd)]
        plt.plot([1, 2, 3], acc, label = str(gram_labels[i]) + '-gram')
        i += 1
    plt.xticks([1,2,3], ['Passive-Agressive', 'Perceptron', 'SGD'])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.9))
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')
    plt.title(name)
    plt.show()


def two_authors_n_grams(name, file_name):
    one_gram = [ file for file in file_name if ('(1, 1)' in file)]
    bi_gram = [ file for file in file_name if ('(2, 2)' in file)]
    tri_gram = [ file for file in file_name if ('(3, 3)' in file)]
    four_gram = [ file for file in file_name if ('(4, 4)' in file)]
    five_gram = [ file for file in file_name if ('(5, 5)' in file)]
    one_five_gram = [ file for file in file_name if ('(1, 5)' in file)]
    plot_2_authors(name, one_gram, bi_gram, tri_gram, four_gram, five_gram, one_five_gram)


def plot_roc(df, gram_label, figure_num):
    plt.figure(figure_num) 
    matrix_tmp = np.matrix(df.values.tolist()) # return all columns and index column
    matrix_folds = matrix_tmp[:,1:].getA() # removes index columns and converts to an array
    tprs = []
    fprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in np.array([0, 2, 4, 6, 8]):
        fpr, tpr, _ = roc_curve(matrix_folds[i], matrix_folds[i+1])
        roc_auc = auc(fpr, tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label=r'%s-gram (AUC = %0.2f)' % (gram_label, mean_auc), lw=2, alpha=.8)

def roc(name, *n_grams):
    grams = list(n_grams)
    gram_labels = list(np.arange(1, len(grams) + 1))
    gram_labels[5] = '1-5'
    i = 0
    ax = plt.subplot(111)
    for gram in grams:
        gram.sort()
        passive = pd.read_csv(gram[0])
        perc = pd.read_csv(gram[1])
        sgd = pd.read_csv(gram[2])
        plot_roc(passive, gram_labels[i], 1)
        plot_roc(perc, gram_labels[i], 2)
        plot_roc(sgd, gram_labels[i], 3)
        i += 1
    plt.figure(1)
    plt.title('Mean ROC for Passive-Aggressive ' + name)
    plt.legend()
    plt.figure(2)
    plt.title('Mean ROC for Perceptron ' + name)
    plt.legend()
    plt.figure(3)
    plt.title('Mean ROC for SGD ' + name)
    plt.legend()
    plt.show()


def two_authors_roc(name, file_name):
    one_gram = [ file for file in file_name if ('(1, 1)' in file)]
    bi_gram = [ file for file in file_name if ('(2, 2)' in file)]
    tri_gram = [ file for file in file_name if ('(3, 3)' in file)]
    four_gram = [ file for file in file_name if ('(4, 4)' in file)]
    five_gram = [ file for file in file_name if ('(5, 5)' in file)]
    one_five_gram = [ file for file in file_name if ('(1, 5)' in file)]
    roc(name, one_gram, bi_gram, tri_gram, four_gram, five_gram, one_five_gram)


def mean_accuracy(name, *n_grams):
    grams = list(n_grams)
    acc = []
    for gram in grams:
        print(acc)
        passive = [file for file in gram if('Passive' in file)]
        perceptron = [file for file in gram if('Perceptron' in file)]
        sgd = [file for file in gram if('SGD' in file)] 
        acc.append(np.mean([accuracy(pd.read_csv(file)) for file in passive]))
        acc.append(np.mean([accuracy(pd.read_csv(file)) for file in perceptron]))
        acc.append(np.mean([accuracy(pd.read_csv(file)) for file in sgd]))
    return acc


def one_vs_rest(name, file_name):
    one_gram = [ file for file in file_name if ('(1, 1)' in file)]
    bi_gram = [ file for file in file_name if ('(2, 2)' in file)]
    tri_gram = [ file for file in file_name if ('(3, 3)' in file)]
    four_gram = [ file for file in file_name if ('(4, 4)' in file)]
    five_gram = [ file for file in file_name if ('(5, 5)' in file)]
    return mean_accuracy(name, one_gram, bi_gram, tri_gram, four_gram, five_gram)

two_authors_char = glob.glob('Results/Confusion Matrix/2 authors/char/*.csv')
two_authors_char_wb = glob.glob('Results/Confusion Matrix/2 authors/char_wb/*.csv')
two_authors_word = glob.glob('Results/Confusion Matrix/2 authors/word/*.csv')
two_authors_pos_tag = glob.glob('Results/Confusion Matrix/Pos Tag 2 authors/*.csv')
ten_authors_char = glob.glob('Results/Confusion Matrix/10 authors/char/*.csv')
ten_authors_char_wb = glob.glob('Results/Confusion Matrix/10 authors/char_wb/*.csv')
ten_authors_word = glob.glob('Results/Confusion Matrix/10 authors/word/*.csv')
ten_authors_pos_tag = glob.glob('Results/Confusion Matrix/Pos Tag 10 authors/*.csv')

all_authors_char = glob.glob('Results/Confusion Matrix/All authors - One vs All (Binary)/char/*.csv')
all_authors_char_wb = glob.glob('Results/Confusion Matrix/All authors - One vs All (Binary)/char_wb/*.csv')
all_authors_word = glob.glob('Results/Confusion Matrix/All authors - One vs All (Binary)/char/*.csv')

two_authors_char_roc = glob.glob('Results/ROC/2 authors/char/*.csv')
two_authors_char_wb_roc = glob.glob('Results/ROC/2 authors/char_wb/*.csv')
two_authors_word_roc = glob.glob('Results/ROC/2 authors/word/*.csv')
two_authors_pos_tag_roc = glob.glob('Results/ROC/Pos Tag 2 authors/*.csv')

# two_authors_roc('n-grams char between two authors', two_authors_char_roc)
# two_authors_roc('n-grams char word boundaries between two authors', two_authors_char_wb_roc)
# two_authors_roc('n-grams word between two authors', two_authors_word_roc)
# two_authors_roc('n-grams pos tag between two authors', two_authors_pos_tag_roc)

# ten_authors_char_roc = glob.glob('Results/ROC/10 authors/char/*.csv')
# ten_authors_char_wb_roc = glob.glob('Results/ROC/10 authors/char_wb/*.csv')
# ten_authors_word_roc = glob.glob('Results/ROC/10 authors/word/*.csv')
# ten_authors_pos_tag_roc = glob.glob('Results/ROC/Pos Tag 10 authors/*.csv')

# two_authors_n_grams('Accuracy between two authors n-grams char', two_authors_char)
# two_authors_n_grams('Accuracy between two authors n-grams char word boundaries', two_authors_char_wb)
# two_authors_n_grams('Accuracy between two authors n-grams word', two_authors_word)
# two_authors_n_grams('Accuracy between two authors n-grams pos tag', two_authors_pos_tag)
# two_authors_n_grams('Accuracy between ten authors n-grams char', ten_authors_char)
# two_authors_n_grams('Accuracy between ten authors n-grams char word boundaries', ten_authors_char_wb)
# two_authors_n_grams('Accuracy between ten authors n-grams word', ten_authors_word)
# two_authors_n_grams('Accuracy between ten authors n-grams pos tag', ten_authors_pos_tag)
one_vs_rest('Accuracy between ten authors vs all all n-grams char', all_authors_char)