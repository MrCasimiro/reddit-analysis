import psycopg2
import psycopg2.extras
import json
import time
import sys
import codecs


MEMORY_SIZE = 10000000 #in bytes


# It creates a connection with the db
def connect(dbname, user, password):
    conn = None
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password)
    except Exception as ex:
        print(ex.args)
    return conn


# It verifies if a subreddit already exists in database
def subreddit_id_exists(id, conn):
    # Verify if a subreddit was already insert
    cur = conn.cursor()
    cur.execute("SELECT * FROM subreddit where id=%d;" % (id,))
    if cur.fetchall():
        return True
    return False


# It retrieves the full time in gm
def retrieve_time(utc_time):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(utc_time)))


# It creates tables for the database
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


    # It customize the insert query
def create_insert(*args):
    values = list()
    for column in args:
        values.append(column) 
    return values


# write errors in a file
def write_error(filename, *args):
    with open(''.join(['populate_error_', filename]), 'a') as error_file:
        for error in args: error_file.write(str(error) + "\n")


# It insert a block of queries
def insert_block(conn, cur, query):
    try:
        sql = "INSERT INTO comment VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, %s, %s, %s) "
        psycopg2.extras.execute_batch(cur, sql, query, page_size=1000)
        conn.commit()
        query = None
        query = []
    except Exception as ex:
        write_error('ERRRO BD', ex, "\n\n\n")
        conn.rollback()
    return query


def is_comm_invalid(comm):
    if comm == '[deleted]' or comm == 'removed' or comm == 'deleted' or comm.strip() == '':
        return True
    return False


def is_author_not_pres(author):
    if author == '[deleted]':
        return True
    return False


# Populate db
def populate(name, conn, filename):
    with open(filename, "r") as data:
        line = data.readline()
        deleted = 0
        valid = 0
        query = []
        cur = conn.cursor()
        buffer_out_of_bounds = 0
        while line:
            try:
                comment_block = json.loads(line)

                if is_comm_invalid(comment_block['body']) or is_author_not_pres(comment_block['author']):
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
                    user_id = int(comment_block['id'], 36)
                    user = str(comment_block['author'])
                    msg = str(comment_block['body'])
                    link_id = int(comment_block['link_id'].split('_')[1],
                                     36)
                    parent_id = int(comment_block['parent_id'].split('_')[1],
                                     36)

                    if not subreddit_id_exists(subreddit_id, conn):
                        cur.execute("""INSERT INTO subreddit (id, name) VALUES
                                    (%s, %s)""", (subreddit_id,
                                                  comment_block['subreddit'],))

                    query.append(create_insert(user_id, user, msg, gilded, score,
                                               ups, downs, subreddit_id, created_time,
                                               retrieved_on, link_id, parent_id))
                    if sys.getsizeof(query) >= MEMORY_SIZE:
                        buffer_out_of_bounds += 1
                        print("Estouro de buffer nº" + str(buffer_out_of_bounds) + "\n\n\n\n")
                        query = insert_block(conn, cur, query)

            except Exception as ex:
                print('Erro JSON')
                print(comment_block)
                write_error(name, 'ERRO JSON', ex, comment_block, "\n\n\n")
            line = data.readline()
        insert_block(conn, cur, query)
                             
        with open(''.join(['valid_comments_', name]), 'a') as valid_file:
            comment = """
                         Comentários válidos:   %d
                         Comentários inválidos: %d
                         Total:                 %d""" % (valid, deleted,
                                                         valid+deleted)
            valid_file.write(comment)
