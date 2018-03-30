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
        return True
    return False


def retrieve_time(utc_time):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(utc_time)))


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
                    UPS             BIGINT,
                    DOWNS           BIGINT,
                    SUBREDDIT_ID    BIGINT          REFERENCES SUBREDDIT(ID),
                    CREATED_UTC     TIMESTAMP WITHOUT TIME ZONE    NOT NULL,
                    RETRIEVED_ON    TIMESTAMP WITHOUT TIME ZONE    NOT NULL,
                    LINK_ID         BIGINT          NOT NULL,
                    PARENT_ID       BIGINT          NOT NULL);""")
    cur.close()
    conn.commit()


def populate(conn, filename):
    with open(filename, 'r') as data:
        line = data.readline()
        deleted = 0
        valid = 0
        while line:
            cur = conn.cursor()
            try:
                comment_block = json.loads(line)

                if comment_block['body'] == '[deleted]':
                    deleted += 1
                else:
                    valid += 1
                    # adjusting column variables to populate
                    created_time = retrieve_time(comment_block['created_utc'])
                    retrieved_on = retrieve_time(comment_block['retrieved_on'])
                    subreddit_id = int(comment_block['subreddit_id'].split('_')[1],
                                       36)
                    score = int(comment_block['score'])
                    ups = int(comment_block['ups'])
                    downs = int(comment_block['downs'])
                    gilded = int(comment_block['gilded'])

                    if not subreddit_id_exists(subreddit_id, conn):
                        cur.execute("""INSERT INTO subreddit (id, name) VALUES
                                    (%s, %s)""", (subreddit_id,
                                                  comment_block['subreddit'],))

                    cur.execute("""INSERT INTO comment VALUES
                                (%s,%s,%s,%s,%s,%s,%s,%s,%s, %s, %s, %s)""",
                                (int(comment_block['id'], 36),
                                 comment_block['author'], comment_block['body'],
                                 gilded, score, ups, downs, subreddit_id,
                                 created_time, retrieved_on,
                                 int(comment_block['link_id'].split('_')[1],
                                     36),
                                 int(comment_block['parent_id'].split('_')[1],
                                     36),))
                    conn.commit()
            except Exception as ex:
                with open('populate_error', 'a') as error:
                    error.write(str(ex))
                    error.write(str(comment_block))
                    error.write("\n\n\n")
                conn.rollback()
            line = data.readline()
        with open('valid_comments', 'w') as valid_file:
            comment = """
                         Comentários válidos:   %d
                         Comentários inválidos: %d
                         Total:                 %d""" % (valid, deleted,
                                                         valid+deleted)
            valid_file.write(comment)
