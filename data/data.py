import tushare as ts
from datetime import datetime


class Data:
    def __init__(self, token):
        self.token = token
        self.pro = ts.pro_api(token)

    def get_oneday(self, trade_date='20180810'):
        return self.pro.daily(trade_date=trade_date)

    def get_daily(self, trade_date):
        if not trade_date:
            return

        return self.pro.daily(trade_date=trade_date)

    def get_table(self, table_name, today=None):
        # if isinstance(ts_code, list):
        #     ts_code = ",".join(ts_code)

        # return self.pro.query('daily', ts_code=ts_code, start_date=start_date, end_date=end_date)

        if today is None:
            today = datetime.today().strftime('%Y%m%d')

        return self.pro.query(table_name, trade_date=today)

    def get_realtime_tick(self, ts_code='600000.SH'):
        return ts.realtime_tick(ts_code=ts_code)

    # def update(self, table,start_date=None, end_date=None):
    #     if not table:
    #         return
    #     return self.pro.index_daily(ts_code='000001.SH', start_date=start_date, end_date=end_date)
