import psycopg2

def connect(dbname, user, password):
    conn = None
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password)
    except Exception as ex:
        print(ex.args)
    return conn

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




conn = connect('reddit_test', 'casimiro', 'danizinha15')
averages(conn)
