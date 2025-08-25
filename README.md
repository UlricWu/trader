# trader


## data flow
Now FeatureEngineer → FeatureEvent → Strategy → SignalEvent → Portfolio.

(todo)
[DataHandler]  
---> MarketEvent 
(raw_close, raw_volume, adj_factor_pre, adj_factor_post, …)
---> [AnalyticsEngine]
1) Adjustment stage (policy = RAW | HFQ | DYPRE | QFQ_STATIC)
2) Rolling window update (per symbol)
3) Indicator/feature computation (MA, returns, vol, etc.)
4) Publish AnalyticsEvent (both raw_* and adj_* fields)
5) Write to FeatureStore (for downstream reuse)
                                   |
                                   v
                           AnalyticsEvent 
   (统一生成所有特征（价格、指标、信号等）。)
                                   |
            +----------------------+------------------+
            |                      |                  |
     [FeatureEngineer]     [MLPipeline]        [RuleStrategy]
              (直接消费 AnalyticsEvent，进行训练和预测。)
            |                      |
    FeatureEvent             PredictionEvent
            |                      |
           ...                  [Portfolio]

实际好处

模块化：行情 → 分析 → 特征 → 策略 → 风控 → 组合 → 回测，层次清晰。

灵活性：换模型、加指标，只动 AnalyticsEngine，不影响 Strategy。

可追溯性：FeatureStore 是完整的特征历史，可用于回测和训练。

可扩展性：你可以很容易在 AnalyticsEngine 里增加新特征（如 MACD、RSI）