from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import connect
import csv


def retrieve_comments(sub_id):
    """Returns a list of messages based on subreddit id."""
    comm_cur = CONN.cursor('retrieve_messages' + str(sub_id))
    comm_cur.itersize = 2000
    comm_cur.execute('SELECT body FROM comment WHERE subreddit_id = %s', (sub_id,))
    return comm_cur


def prepare_sent_analyzer():
    with open('subreddit_sentiment.csv', 'a', newline='') as csvfile:
            field_names = ['Name', 'Positive', 'Negative', 'Neutral', 'Compound']
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()


def write_sent_analyzer(sub_eval, name):
    with open('subreddit_sentiment.csv', 'a', newline='') as csvfile:
            field_names = ['Name', 'Positive', 'Negative', 'Neutral', 'Compound']
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerow({'Name': name,
                             'Positive': sub_eval['pos'],
                             'Negative': sub_eval['neg'],
                             'Neutral': sub_eval['neu'],
                             'Compound': sub_eval['compound']})


def sum_vader(dict_eval, dict_score):
    dict_eval['neg'] += dict_score['neg']
    dict_eval['pos'] += dict_score['pos']
    dict_eval['neu'] += dict_score['neu']
    dict_eval['compound'] += dict_score['compound']
    return dict_eval


def div_vader(dict_eval, length):
    dict_eval['neg'] /= length
    dict_eval['pos'] /= length
    dict_eval['neu'] /= length
    dict_eval['compound'] /= length
    return dict_eval

def give_score_vader(sentence_list):
    """Gives scores average for sentences in a comment"""
    sid = SentimentIntensityAnalyzer()
    sent_eval = {"neg": 0, "pos": 0, "neu": 0, 'compound': 0}
    for sentence in sentence_list:
        sid_score = sid.polarity_scores(sentence)
        sent_eval = sum_vader(sent_eval, sid_score)
    if len(sentence_list) == 0:
        print('zerado')
        print(sentence_list)
    sent_eval = div_vader(sent_eval, len(sentence_list))
    return sent_eval


def comm_sent_analyzer(sub_comments):
    """Analyses sentiment in comments per subreddit."""
    prepare_sent_analyzer()
    sub_eval = {}
    for sub in sub_comments:
        sub_eval[sub[1]] = {"neg": 0, "pos": 0, "neu": 0, 'compound': 0}
        sub_sentences = []
        print('Read comments')
        total_comm = 0
        for comment in retrieve_comments(sub[0]):
            # It divides comments in sentences
            sentence_list = tokenize.sent_tokenize(comment[0])
            if len(sentence_list) == 0:
                print('vazio')
                continue
            eval_comm = give_score_vader(sentence_list)
            sub_eval[sub[1]] = sum_vader(sub_eval[sub[1]], eval_comm)
            total_comm += 1
        print(total_comm)
        sub_eval[sub[1]] = div_vader(sub_eval[sub[1]], total_comm)
        write_sent_analyzer(sub_eval[sub[1]], sub[1])
    return sub_eval


CONN = connect.connect('reddit_test', 'guilhermecasimiro', 'grc_2018')
CUR = CONN.cursor()
CUR.execute("SELECT * from subreddit WHERE name = 'religion' OR name = 'science'")
result = comm_sent_analyzer(CUR.fetchall())
