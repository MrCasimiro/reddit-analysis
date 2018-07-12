import nltk
import connect
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Returns a list of messages based on subreddit id
def retrieve_messages(id):
  conn = connect.connect('reddit_test', 'postgres', 'postgres')
  cur = conn.cursor('retrieve_messages')
  cur.itersize = 2000
  cur.execute('SELECT * FROM comment WHERE subreddit_id = %s', (id,))
  
  
conn = connect.connect('reddit_test', 'postgres', 'postgres')
cur = conn.cursor()
cur.execute("SELECT * from subreddit WHERE name = 'science' OR name = 'religion'")
rows = cur.fetchall()
