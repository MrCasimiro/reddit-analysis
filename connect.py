import psycopg2
import json
import time


def connect(dbname, user, password):
    conn = None
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password)
    except Exception as ex:
        print(ex.args)
    return conn


def subreddit_id_exists(id, conn):
    # Verify if a subreddit was already insert
    cur = conn.cursor()
    cur.execute("SELECT * FROM subreddit where id=%d;" % (id,))
    if cur.fetchall():
        print("Subreddit de id %d já inserido" % id)
        return True
    return False


def create_tables(conn):
    cur = conn.cursor()
    cur.execute(""" CREATE TABLE subreddit (
                    ID      BIGINT PRIMARY KEY,
                    NAME    CHAR(50));""")
    cur.execute(""" CREATE TABLE comment (
                    ID              BIGINT          PRIMARY KEY,
                    AUTHOR          VARCHAR(50)     NOT NULL,
                    BODY            TEXT            NOT NULL,
                    GILDED          BIGINT,
                    SCORE           BIGINT,
                    SUBREDDIT_ID    BIGINT          REFERENCES SUBREDDIT(ID),
                    CREATED_UTC     TIMESTAMP WITHOUT TIME ZONE    NOT NULL);""")
    cur.close()
    conn.commit()


def populate(conn, filename):
    with open(filename, 'r') as data:
        line = data.readline()
        while line:
            cur = conn.cursor()
            try:
                comment_block = json.loads(line)

                # adjusting column variables to populate
                created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(
                                     int(comment_block['created_utc'])))
                subreddit_id = int(comment_block['subreddit_id'].split('_')[1],
                                   36)
                score = int(comment_block['score'])
                gilded = int(comment_block['gilded'])

                if not subreddit_id_exists(subreddit_id, conn):
                    cur.execute("""INSERT INTO subreddit (id, name) VALUES
                                (%s, %s)""", (subreddit_id,
                                              comment_block['subreddit'],))

                cur.execute("""INSERT INTO comment VALUES
                            (%s,%s,%s,%s,%s,%s,%s)""",
                            (int(comment_block['id'], 36),
                             comment_block['author'], comment_block['body'],
                             gilded, score, subreddit_id, created_time,))
            except Exception as ex:
                print(ex)
                conn.rollback()
                return
            conn.commit()
            line = data.readline()


conn = connect('reddit', 'casimiro', 'danizinha15')
# create_tables(conn)

initial_time = time.time()
populate(conn, 'data')
time_file = open('populate_time.txt', 'w')
total_time = ''.join([str(time.time() - initial_time), 'seconds'])
time_file.write(total_time)
conn.close()
