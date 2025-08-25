#!filepath: trader/analytics/adjustment.py
from enum import Enum
import pandas as pd
from dataclasses import dataclass

class AdjustmentPolicy(Enum):
    RAW = "raw"
    HFQ = "hfq"          # 后复权（安全）
    DYPRE = "dypre"      # 动态前复权（到“当前时点”为止）

@dataclass
class Adjuster:
    policy: AdjustmentPolicy = AdjustmentPolicy.HFQ

    def adjust_close(
        self,
        date: pd.Timestamp,
        raw_close: float,
        factor_series: pd.Series,   # index=date, value=cumulative factor F(t)
        window_start: pd.Timestamp,
        now: pd.Timestamp
    ) -> float:
        if self.policy == AdjustmentPolicy.RAW or factor_series is None:
            return raw_close

        F_t = float(factor_series.loc[date])

        if self.policy == AdjustmentPolicy.HFQ:
            F_t0 = float(factor_series.loc[window_start])
            return raw_close * (F_t / F_t0)

        if self.policy == AdjustmentPolicy.DYPRE:
            F_now = float(factor_series.loc[now])
            return raw_close * (F_now / F_t)

        return raw_close



    # def apply_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
    #     adj_factors = tushare_api.get_adj_factors(...)
    #     df["adj_factor"] = df["date"].map(adj_factors)
    #     df[["open", "high", "low", "close"]] *= df["adj_factor"]
    #     return df

    # def _apply_adjustment(self, data):
    #     # adjust_type = MODE_TO_ADJUST.get(RUN_MODE, "none")
    #     if adjust_type == "qfq":
    #         return self._forward_adjust(data)
    #     elif adjust_type == "hfq":
    #         return self._backward_adjust(data)
    #     else:
    #         return data

