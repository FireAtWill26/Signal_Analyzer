import numpy as np
# 隔日收益率计算方程
# close为保存股票收盘价的np.array,可以从"/dfs/dataset/230-172466165521/data/dataprod/IntervalFull/IntervalFull.close.N49,5860f"中提取,形状为(n_days,n_stocks,n_intv);
# adjFactor为保存股票复权因子的np.array,可以从"/dfs/dataset/230-172466165521/data/dataprod/CAX/adjfactor.N,5860f"中提取,形状为(n_days,n_stocks);
# nStock为whole universe中股票的数量
# didx为日期索引,从0开始;
# tidx为intv的索引,从0开始;

def computeDailyStockReturnRatio(close, adjFactor, nStock, didx, tidx):
    basePrice = close[didx-1, 0:nStock, tidx].copy()
    basePrice *= adjFactor[didx, 0:nStock]
    return_ratio = close[didx, 0:nStock, tidx] / basePrice - 1
    return return_ratio

# 实际收益及收益率基于隔日收益率数组和持仓数据获得
# prev_hold为didx-1日的持仓值,根据我们的持仓方式设计,满足prev_hold.sum()=20e6
# hold为didx当日的持仓值,根据我们的持仓方式设计,满足hold.sum()=20e6
def computeActualDailyReturnRatio(prev_hold, hold, close, adjFactor, nStock, didx, tidx):
    new_hold = prev_hold * (computeDailyStockReturnRatio(close, adjFactor, nStock, didx, tidx) + 1)
    ret = new_hold - hold
    daily_return = ret.sum()
    daily_return_ratio = daily_return / hold.sum()
    return daily_return, daily_return_ratio
