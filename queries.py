import connect
import csv
import time

# Execute and then process a query in batches
def get_result(conn, query, name):
    cur = conn.cursor(''.join([name, '-cursor']))
    cur.itersize = 2000
    cur.execute(query)
    return cur

# Generates a ranking of comments in a csv file
def comment_ranking(conn):
    query = """
            SELECT c.score, c.body, s.name
            FROM comment as c, subreddit as s
            WHERE c.subreddit_id = s.id;
            """

    with open('comment_ranking.csv', 'w', newline='') as csvfile:
        field_names = ['Score', 'Comment', 'Subreddit']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for row in get_result(conn, query, 'comment_ranking'):
            writer.writerow({'Score': row[0], 'Comment': row[1],
                             'Subreddit': row[2]})

# Generates a csv file containing an overall of a subreddit(name, number of comments,
# number of threads and comments per threads)
def subreddit_overall(conn):
    query = """
            SELECT s.name, COUNT(c.body), COUNT(DISTINCT c.link_id)
            FROM subreddit AS s, comment AS c
            WHERE s.id = c.subreddit_id GROUP BY s.id;"""
    
    with open('subreddit_overall.csv', 'w', newline='') as csvfile:
        field_names = ['Subreddit', 'Number of comments', 'Number of posts',
                        'Comments per posts']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for row in get_result(conn, query, 'subreddit_overall'):
            writer.writerow({'Subreddit': row[0], 'Number of comments': row[1],
                             'Number of posts': row[2],
                             'Comments per posts': int(row[1])/int(row[2])})


# Generates a csv file with comments over time (year, month, day of week, hours, minutes, seconds)
def comments_over_time(conn):
    query = """
                SELECT DISTINCT extract(second FROM created_utc), 
                 extract(minute FROM created_utc), extract(hour FROM created_utc),
                 extract(dow FROM created_utc), extract(month FROM created_utc), 
                 extract(year FROM created_utc), count(*) 
                FROM comment 
                GROUP BY extract(second FROM created_utc),
                 extract(minute FROM created_utc),
                 extract(hour FROM created_utc),
                 extract(dow FROM created_utc),
                 extract(month from created_utc),
                 extract(year from created_utc);
            """

    with open('comments_over_time.csv', 'w', newline='') as csvfile:
        field_names = ['Second', 'Minute', 'Hour', 'Day of Week', 'Month', 'Year', 'Number of comments']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        for row in get_result(conn, query, 'comments_over_time'):
            writer.writerow({'Second': row[0], 'Minute': row[1], 'Hour': row[2], 'Day of Week': row[3],
                             'Month': row[4], 'Number of comments': row[5]})


# Generates a csv file grouping the comments length and its frequency
def comments_length(conn):
    query = """
                SELECT LENGTH(body), COUNT(*)
                FROM COMMENT
                GROUP BY LENGTH(body)
                ORDER BY LENGTH(body) ASC;
            """
    with open('comments_length.csv', 'w', newline='') as csvfile:
        field_names = ['Comment length', 'Frequency']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        for row in get_result(conn, query, 'comments_length'):
            writer.writerow({'Comment length': row[0], 'Frequency': row[1]})


# Generates a csv file with user activities over time(year, month and day of week)
def users_activities(conn):
    query = """
            SELECT author, extract(dow FROM created_utc), 
             extract(month FROM created_utc), extract(year FROM created_utc)
            FROM comment
            GROUP BY author, extract(dow FROM created_utc), 
             extract(month FROM created_utc), extract(year FROM created_utc); """

    with open('users_activities.csv', 'w', newline='') as csvfile:
        field_names = ['Author', 'Day of Week', 'Month', 'Year']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for row in get_result(conn, query, 'users_activities'):
            writer.writerow({'Author': row[0], 'Day of Week': row[1], 'Month': row[2], 'Year': row[3]})    


def get_time(filename, method, *args):
    with open((filename + '_time'), 'w') as time_file:
        initial_time = time.time()
        method(*args)
        total_time = ''.join([str(time.time() - initial_time), ' seconds'])
        time_file.write(total_time)


conn = connect.connect('reddit', 'postgres', 'postgres')

#connect.create_tables(conn)
#get_time('populate', connect.populate, conn, 'data')

get_time('comments_length', comments_length, conn)
get_time('comment_ranking', comment_ranking, conn)
get_time('subreddit_overall', subreddit_overall, conn)
get_time('comments_over_time', comments_over_time, conn)
get_time('users_activities', users_activities, conn)