# trader
知识储备
实操
    数据
    ｜ 量化交易的前提是要有海量的可靠数据。
        获取
            机构
                东方财富Choice、Wind资讯金融终端，数据源（库）供应商主要有Wind资讯、恒生聚源、朝阳永续、天软、巨潮、天相、巨灵、国泰安、通联

            个人
                tsshare
                观察者
                akshare
                jqdatasdk


        频率
            1、5、15、30min
            日
            周
            月

        处理
            清洗
                空数据
                异常数据
                四舍五入方法
                    四舍五入出错的后果：对于2元以下的低价股，四舍五入可能带来1分钱的误差，相当于至少0.5个百分点！
                    影响到价格自由度，影响到委买委卖的分布
                    ｜ http://centerforpbbefr.rutgers.edu/2005/Paper%202005/PBFEA29.pdf

                合并
                    后复权
                    前复权

                投资者人数、指数的PE值等数据

            计算指标
                均线、BIAS和BOLL
                ma5、ma10、ma20
                RSI指标信号
                20日均线

            转换

        储存
            本地
            服务器


    交易文件
        策略
        ｜ https://github.com/UFund-Me/Qbot/tree/main/qbot/strategies
            均线和BIAS指标信号
            股价低于BOLL线底
            K线上穿D线
            RSI指标信号
            价格大于20日均线1.02倍，则买入
            ｜ 价格低于20日均线0.98倍，则卖出

        计算收益率
        ｜ # 根据信号生成交易指令，并计算收益率
        ｜ orders = np.zeros_like(signals)
        ｜ orders[signals > 0] = 1   # 买入信号
        ｜ orders[signals < 0] = -1  # 卖出信号
        ｜ returns = np.diff(data['close']) * orders[:-1]
        ｜ 
            买入信号
            卖出信号


    可视化
    监控
    量化交易
    ｜ backtrader、easytrader 和 easyquotation 的量化交易框架
        回测
            时间和策略
            

        交易

    配置
        账号密码

    webserver

云服务
