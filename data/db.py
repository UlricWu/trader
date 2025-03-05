import sqlite3

from utilts import logs
from datetime import datetime
# class DatabaseConnection:
#     def __init__(self, db_name):
#         self.db_name = db_name
#         self.connection = None
#
#     def __enter__(self):
#         self.connection = sqlite3.connect(self.db_name)
#         return self.connection
#
#     def __exit__(self, exc_type, exc_value, traceback):
#         if self.connection:
#             self.connection.close()
#         if exc_type:
#             logs.record_log(f"An error occurred: {exc_value}")


class Database:
    def __init__(self, db_name):
         self.db_name = db_name
         self.connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def connect(self):
        print("Connecting")

    def close(self):
        print("Closing connection")

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self as db:
                return func(db, *args, **kwargs)

        return wrapper

# class DB:
#
#     def __init__(self, database="tutorial.db"):
#         self.database = database
#
#     def _init(self):
#         return sqlite3.connect(self.database)
#
#     def insert(self, table, data):
#         if len(data):
#             return
#         con = self._init()
#
#         today = datetime.today().strftime('%Y%m%d')
#         temp_table = f"{table}_{today}"
#
#         data.to_sql(temp_table, con, if_exists='append', index=False)
#
#         insert_sql = f"""INSERT OR REPLACE INTO {table}  SELECT * FROM {temp_table} """  # formulate the query without specifying column names if two tables with the same columns
#         con.execute(insert_sql).fetchall()
#         con.execute(f"drop table if exists {temp_table}")
#
#         return self._return(con)
#
#     def _return(self, con):
#         con.commit()
#         return con.close()
#
#     def extract(self, table):
#         connection = self._init()
#         ans = connection.execute(f"select * from {table}").fetchall()
#
#         return ans

# def create_table(self, table, cols):
#     if not table:
#         return
#     if isinstance(cols, list):
#         cols = ",".join(cols)
#     if not isinstance(cols, str):
#         return
#
#     sql = f"""create table IF NOT EXISTS {table} ({cols})   """
#     connection = self._init()
#     cur = connection.cursor()
#     cur.execute(sql)
#     connection.close()
#
#     logs.record_log(f"SQL: {sql} has been operated")
#
# def insert(self, table, df):
#     if not (table and len(df)):
#         return
#     con = self._init()
#     df.to_sql(table, con, if_exists='replace', index=True)
#     con.cur.close()
#     # res = self.cur.execute(f"select * from {table}")
#
#     logs.record_log(f"SQL: insert {table} has been operated ")
#

