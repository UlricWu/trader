import sqlite3

from utilts import logs
from datetime import datetime

import pandas as pd


class Database:
    def __init__(self, name="tutorial.db"):
        self._conn = sqlite3.connect(name)
        self._cursor = self._conn.cursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        if self.connection:
            self.close()
        if exc_type:
            logs.record_log(f"An error occurred: {exc_value}", 3)

    @property
    def connection(self):
        return self._conn

    @property
    def cursor(self):
        return self._cursor

    def commit(self):
        self.connection.commit()

    def close(self, commit=True):
        if commit:
            self.commit()
        self.connection.close()

    def execute(self, sql):
        # self.cursor.execute(sql, params )
        self.cursor.execute(sql)

    def executemany(self, sql):
        self.cursor.executescript(sql)

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()

    def query(self, sql):
        # self.cursor.execute(sql, params )
        self.cursor.execute(sql)
        return self.fetchall()

    # def __call__(self, func):
    #     def wrapper(*args, **kwargs):
    #         with self as db:
    #             return func(db, *args, **kwargs)
    #
    #     return wrapper


def update_daily(table, data):
    if check_table(table):
        create_table(table)

    today = datetime.today().strftime('%Y%m%d')

    temp_table = f"{table}_{today}"

    with Database() as db:
        data.to_sql(temp_table, db.connection, if_exists='append', index=False)

        insert_sql = f"""INSERT OR REPLACE INTO {table}  SELECT * FROM {temp_table} """  # formulate the query without specifying column names if two tables with the same columns
        db.execute(insert_sql)
        logs.record_log(f"SQL: insert {table} ts_day = {today} has been operated ")
        db.execute(f"drop table if exists {temp_table}")


def extract_table(table='daily', day=None, pandas=True):
    if not day:
        day = datetime.today().strftime('%Y%m%d')
    sql = f"select * from {table} where trade_date = '{day}'"

    with Database() as db:
        if pandas:
            return pd.read_sql_query(sql, db.connection)

        return db.query(sql)


def check_table(table_name='daily'):
    sql = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}' "

    with Database() as db:
        return len(db.query(sql)) == 0


def create_table_daily():
    file = '/home/wsw/trader/configs/daily.sql'
    with open(file, 'r') as sql_file:
        sql_script = sql_file.read()
    logs.record_log(sql_script)

    with Database() as db:
        db.executemany(sql_script)


def create_table(table):
    if table == 'daily':
        return create_table_daily()

    #     if not table:
    #         return
    #     if isinstance(cols, list):
    #         cols = ",".join(cols)
    #     if not isinstance(cols, str):
    #         return
    #
    #     sql = f"""create table IF NOT EXISTS {table} ({cols})   """
    # logs.record_log(f"SQL: {sql} has been operated")
