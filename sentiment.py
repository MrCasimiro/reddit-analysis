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


def comm_sent_analyzer(sub_comments):
    """Analyses sentiment in comments per subreddit."""
    sid = SentimentIntensityAnalyzer()
    sub_eval = {}
    for sub in sub_comments:
        sub_eval[sub[1]] = {"neg": 0, "pos": 0, "neu": 0, 'compound': 0}
        sub_sentences = []
        print('Read comments')
        for comment in retrieve_comments(sub[0]):
            # It divides comments in sentences
            sentence_list = tokenize.sent_tokenize(comment[0])
            sub_sentences.extend(sentence_list)
        print('Give sentence scores')
        for sentence in sub_sentences:
            # It gives score for each sentence
            sid_score = sid.polarity_scores(sentence)
            sub_eval[sub[1]]['neg'] += sid_score['neg']
            sub_eval[sub[1]]['pos'] += sid_score['pos']
            sub_eval[sub[1]]['neu'] += sid_score['neu']
            sub_eval[sub[1]]['compound'] += sid_score['compound']
        sub_eval[sub[1]]['neg'] /= len(sub_sentences)
        sub_eval[sub[1]]['pos'] /= len(sub_sentences)
        sub_eval[sub[1]]['neu'] /= len(sub_sentences)
        sub_eval[sub[1]]['compound'] /= len(sub_sentences)
        print('Write ' + sub[1])
        with open('subreddit_sentiment.csv', 'a', newline='') as csvfile:
            field_names = ['Name', 'Positive', 'Negative', 'Neutral', 'Compound']
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerow({'Name': sub[1],
                             'Positive': sub_eval[sub[1]]['pos'],
                             'Negative': sub_eval[sub[1]]['neg'],
                             'Neutral': sub_eval[sub[1]]['neg'],
                             'Compound': sub_eval[sub[1]]['compound']})

    return sub_eval


CONN = connect.connect('reddit_test', 'postgres', 'postgres')
CUR = CONN.cursor()
CUR.execute("SELECT * FROM subreddit WHERE name = 'science' OR name = 'religion'")
result = comm_sent_analyzer(CUR.fetchall())
