import sqlite3

from utilts import logs
from datetime import datetime

import pandas as pd


class Database:
    def __init__(self, name="tutorial.db"):

        self._conn = sqlite3.connect(name)
        self._cursor = self._conn.cursor()

        logs.record_log(f'loading database={name}')

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


# from config.config import SchemaConfig
from pydantic import BaseModel
from typing import Dict


class SchemaConfig(BaseModel):
    ts_code: str
    trade_date: str
    open: str
    high: str
    low: str
    close: str
    vol: str

    @classmethod
    def from_dict(cls, raw: Dict[str, str]) -> "SchemaConfig":
        return cls(**raw)


def load_and_normalize_data(df: pd.DataFrame, adjustment=None) -> pd.DataFrame:
    rename_map = {
        "trade_date": "date",
        "vol": "volume",
        "ts_code": "symbol"
    }

    # schema = SchemaConfig().from_dict(s)
    # rename_map = {
    #     schema.trade_date: "date",
    #     schema.ts_code: "symbol",
    #     schema.open: "open",
    #     schema.high: "high",
    #     schema.low: "low",
    #     schema.close: "close",
    #     schema.vol: "volume",
    # }
    df = df.rename(columns=rename_map)
    df.sort_values(["date", "symbol"], inplace=True)

    if adjustment is None:
        return df


def extract_table(database="tutorial.db", table='daily', end_day=None, start_day=None, pandas=True, ts_code=None):
    if not end_day:
        end_day = datetime.today().strftime('%Y%m%d')
    if not start_day:
        start_day = end_day

    if int(start_day) > int(end_day):
        raise ValueError(f"start_day={start_day} >day={end_day}")

    sql = f"select * from {table} where trade_date >= '{start_day}' and trade_date <= '{end_day}' "

    if ts_code:
        id_sql = "', '".join(str(x) for x in ts_code)
        sql += f"""and ts_code IN ('{id_sql}') """

    with Database(database) as db:
        if not check_table(table=table, database=database):
            logs.record_log(f"Table {table} does not exist")

        # print(db.query('select * from daily limit 1'))
        if pandas:
            # return pd.read_sql_query(sql, db.connection,parse_dates=['trade_date'])
            df = pd.read_sql_query(sql, db.connection)
            df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
            return df

        return db.query(sql)


def check_table(table='daily', database="tutorial.db"):
    sql = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}' "

    with Database(database) as db:
        return len(db.query(sql)) == 0


def create_table_daily():
    file = '/home/wsw/trader/config/daily.sql'
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


def back_db():
    # source = sqlite3.connect('existing_db.db')
    # dest = sqlite3.connect(':memory:')
    day = datetime.today().strftime('%Y%m%d')
    path = f'db/backup_{day}.db'
    with Database() as source:
        with Database(path) as dest:
            source.connection.backup(dest.connection)

    logs.record_log(f"Backup complete for {day} in {path}")
