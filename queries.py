import connect
import csv
import time


def averages(conn):
    cur = conn.cursor()
    cur.execute("select id from subreddit")
    rows = cur.fetchall()
    subreddit_ids = []
    for row in rows:
        subreddit_ids.append(int(row[0]))
        for id in subreddit_ids:
            cur.execute("select body from comment where subreddit_id=%d" % (id,))
            rows = cur.fetchall()
            count = 0
            total = 0
            for row in rows:
                count = count + 1
                total = total + len(row[0])
            line = "%d com média de tamanho de %s \n" % (id, str(total/count))
            with open("average.txt", "a") as myfile:
                myfile.write(line)


def comment_ranking(conn):
    cur = conn.cursor()
    cur.execute("""
                SELECT c.score, c.body, s.name
                FROM comment as c, subreddit as s
                WHERE c.subreddit_id = s.id
                ORDER BY c.score DESC; """)
    rows = cur.fetchall()
    with open('comment_ranking.csv', 'w', newline='') as csvfile:
        field_names = ['Score', 'Comment', 'Subreddit']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow({'Score': row[0], 'Comment': row[1],
                             'Subreddit': row[2]})


def subreddit_overall(conn):
    cur = conn.cursor()
    cur.execute("""
                SELECT s.name, COUNT(c.body), COUNT(DISTINCT c.link_id)
                FROM subreddit AS s, comment AS c
                WHERE s.id = c.subreddit_id GROUP BY s.id;""")
    rows = cur.fetchall()
    with open('subreddit_overall.csv', 'w', newline='') as csvfile:
        field_names = ['Subreddit', 'Number of comments', 'Number of posts',
                       'Comments per posts']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow({'Subreddit': row[0], 'Number of comments': row[1],
                             'Number of posts': row[2],
                             'Comments per posts': int(row[1])/int(row[2])})


def get_time(filename, method, conn):
    with open((filename + '_time'), 'w') as time_file:
        initial_time = time.time()
        method(conn)
        total_time = ''.join([str(time.time() - initial_time), ' seconds'])
        time_file.write(total_time)


conn = connect.connect('reddit_test', 'casimiro', 'danizinha15')
get_time('comment_ranking', comment_ranking, conn)
