import tushare as ts


class Data:
    def __init__(self, token):
        self.token = token
        self.pro = ts.pro_api(token)

    def get_oneday(self, trade_date='20180810'):
        return self.pro.daily(trade_date=trade_date)

    def get_daily(self, ts_code, start_date='20180701', end_date='20180718'):
        if not ts_code:
            return
        if isinstance(ts_code, list):
            ts_code = ",".join(ts_code)

        return self.pro.query('daily', ts_code=ts_code, start_date=start_date, end_date=end_date)

    def get_realtime_tick(self, ts_code='600000.SH'):
        return ts.realtime_tick(ts_code=ts_code)

    # def update(self, table,start_date=None, end_date=None):
    #     if not table:
    #         return
    #     return self.pro.index_daily(ts_code='000001.SH', start_date=start_date, end_date=end_date)
