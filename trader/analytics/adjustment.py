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
