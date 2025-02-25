import sqlite3

from utilts import logs


class DB:

    def __init__(self, database="tutorial.db"):
        self.database = database

    def _init(self):
        return sqlite3.connect(self.database)

        # self.cur = con.cursor()

    def create_table(self, table, cols):
        if not table:
            return
        if isinstance(cols, list):
            cols = ",".join(cols)
        if not isinstance(cols, str):
            return

        sql = f"""create table IF NOT EXISTS {table} ({cols})   """
        connection = self._init()
        cur = connection.cursor()
        cur.execute(sql)
        connection.close()

        logs.record_log(f"SQL: {sql} has been operated")

    def insert(self, table, df):
        if not (table and len(df)):
            return
        con = self._init()
        df.to_sql(table, con, if_exists='replace', index=True)
        con.cur.close()
        # res = self.cur.execute(f"select * from {table}")

        logs.record_log(f"SQL: insert {table} has been operated ")

    def extract(self, table):
        connection = self._init()
        return connection.execute(f"select * from {table}").fetchall()
