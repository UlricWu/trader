DROP TABLE  IF EXISTS daily;
CREATE TABLE  IF NOT EXISTS daily
(
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_code    char(7)  NOT NULL,-- '股票代码' ,
    trade_date int(8)  not NULL, -- '交易日期',
    open      decimal(18, 4) DEFAULT NULL, -- '开盘价',
    high      decimal(18, 4) DEFAULT NULL, -- '最高价',
    low       decimal(18, 4) DEFAULT NULL, -- '开盘价',
    close     decimal(18, 4) DEFAULT NULL, -- '收盘价',
    pre_close decimal(18, 4) DEFAULT NULL, -- '昨收价【除权价，前复权】',
    change    decimal(18, 4) DEFAULT NULL, -- '涨跌额',
    pct_chg    decimal(18, 4) DEFAULT NULL, -- '今收-除权昨收）/除权昨收 ',
    vol        decimal(18, 4) DEFAULT NULL, -- '成交量 （手）',
    amount    decimal(18, 4) DEFAULT NULL, -- '成交额 （千元）'
--     notes json DEFAULT NULL,
     PRIMARY KEY ( ts_code, trade_date)
) ;
