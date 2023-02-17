"""
MIT License

Copyright (c) 2021 RomFR57 rom.fr57@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from math import fabs

import numpy as np
from numba import jit
from numba.extending import overload


def calculate_indicators(mean_, close_, open_, high_, low_, volume_):
    indicators = {}

    indicators['close'] = close_
    indicators['open'] = open_
    indicators['high'] = high_
    indicators['low'] = low_
    indicators['volume'] = volume_

    short_wma = wma(data=mean_, period=20)
    indicators['short_wma'] = short_wma
    medium_wma = wma(data=mean_, period=50)
    indicators['wma'] = medium_wma
    long_wma = wma(data=mean_, period=100)
    indicators['long_wma'] = long_wma

    ewma_ = ewma(data=mean_, period=15, alpha=0.97)
    indicators['ewma'] = ewma_
    tema_ = trix(data=mean_, period=30)
    indicators['tema'] = tema_
    """MacD"""
    macd_ = macd(data=mean_, fast=12, slow=26)
    indicators['macd'] = macd_
    """Stoch"""
    stoch1, stoch2 = stoch(close_, high_, low_, period_k=5, period_d=3)
    indicators['stoch1'] = stoch1
    indicators['stoch2'] = stoch2
    """William %R"""
    wpr_ = wpr(close_, high_, low_, 14)
    indicators['wpr'] = wpr_
    """RSI"""
    rsi_experts = rsi(mean_, period=5)
    indicators['rsi_experts'] = rsi_experts
    rsi_ = rsi(mean_, period=14)
    indicators['rsi'] = rsi_
    srsi_ = srsi(mean_, period=14)
    indicators['srsi'] = srsi_
    """Bollinger Bands"""
    _, bolinger_up, bolinger_down, _ = bollinger_bands(mean_, period=20)
    indicators['bolinger_up'] = bolinger_up
    indicators['bolinger_down'] = bolinger_down
    """Keltner Channel"""
    _, kc_up, kc_down, _ = keltner_channel(close_, open_, high_, low_, period=20)
    indicators['kc_up'] = kc_up
    indicators['kc_down'] = kc_down
    """Ichimoku"""
    tenkansen, kinjunsen, chikou, senkou_a, senkou_b = ichimoku(mean_, tenkansen=9, kinjunsen=26, senkou_b=52, shift=26)
    indicators['tenkansen'] = tenkansen
    indicators['kinjunsen'] = kinjunsen
    indicators['chikou'] = chikou
    indicators['senkou_a'] = senkou_a
    indicators['senkou_b'] = senkou_b
    # """Volume Profile"""
    # vol_profile_count, vol_profile_price = volume_profile(close_, volume_)
    """True Range"""
    atr_ = atr(open_, high_, low_, period=14)
    indicators['atr'] = atr_
    """Momentum"""
    momentum_ = momentum(mean_, period=40)
    indicators['momentum_'] = momentum_
    """Rate Of Change"""
    roc_ = roc(mean_, period=12)
    indicators['roc'] = roc_
    """Volatility Index"""
    vix_ = vix(close_, low_, period=30)
    indicators['vix'] = vix_
    # """Supertrend"""
    # supertrend_up, supertrend_down = super_trend(close_, open_, high_, low_)
    """Chopiness Index"""
    chop_ = chop(close_, open_, high_, low_)
    indicators['chop'] = chop_
    """Center Of Gravity"""
    cog_ = cog(mean_)
    indicators['cog'] = cog_

    return indicators


def add_indicators_to_dataset(indicators, indicators_names, dates, mean_):
    new_data = []
    for i in range(len(indicators_names)):
        item = indicators_names[i]
        # print(item, indicators[item].shape)
        # print(99, indicators[item], i)
        new_data.append(indicators[item])

        # print(new_data.shape)
    # new_data = np.stack((new_data, mean_), axis=0)
    new_data.append(mean_)
    indicators_names.append('mean')
    new_data = np.array(new_data)
    new_data = np.swapaxes(new_data, 0, 1)
    new_data = new_data[100:, :]
    new_dates = dates[100:]
    return new_data, new_dates


@overload(np.clip)
def np_clip(a, a_min, a_max, out=None):
    """
    Numba Overload of np.clip
    :type a: np.ndarray
    :type a_min: int
    :type a_max: int
    :type out: np.ndarray
    :rtype: np.ndarray
    """
    if out is None:
        out = np.empty_like(a)
    for i in range(len(a)):
        if a[i] < a_min:
            out[i] = a_min
        elif a[i] > a_max:
            out[i] = a_max
        else:
            out[i] = a[i]
    return out


@jit(nopython=True)
def convolve(data, kernel):
    """
    Convolution 1D Array
    :type data: np.ndarray
    :type kernel: np.ndarray
    :rtype: np.ndarray
    """
    size_data = len(data)
    size_kernel = len(kernel)
    size_out = size_data - size_kernel + 1
    out = np.array([np.nan] * size_out)
    kernel = np.flip(kernel)
    for i in range(size_out):
        window = data[i:i + size_kernel]
        out[i] = sum([window[j] * kernel[j] for j in range(size_kernel)])
    return out


@jit(nopython=True)
def sma(data, period):
    """
    Simple Moving Average
    :type data: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    size = len(data)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        window = data[i - period + 1:i + 1]
        out[i] = np.mean(window)
    return out


@jit(nopython=True)
def wma(data, period):
    """
    Weighted Moving Average
    :type data: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    weights = np.arange(period, 0, -1)
    weights = weights / weights.sum()
    out = convolve(data, weights)
    return np.concatenate((np.array([np.nan] * (len(data) - len(out))), out))


@jit(nopython=True)
def cma(data):
    """
    Cumulative Moving Average
    :type data: np.ndarray
    :rtype: np.ndarray
    """
    size = len(data)
    out = np.array([np.nan] * size)
    last_sum = np.array([np.nan] * size)
    last_sum[1] = sum(data[:2])
    for i in range(2, size):
        last_sum[i] = last_sum[i - 1] + data[i]
        out[i] = last_sum[i] / (i + 1)
    return out


@jit(nopython=True)
def ema(data, period, smoothing=2.0):
    """
    Exponential Moving Average
    :type data: np.ndarray
    :type period: int
    :type smoothing: float
    :rtype: np.ndarray
    """
    size = len(data)
    weight = smoothing / (period + 1)
    out = np.array([np.nan] * size)
    out[0] = data[0]
    for i in range(1, size):
        out[i] = (data[i] * weight) + (out[i - 1] * (1 - weight))
    out[:period - 1] = np.nan
    return out


@jit(nopython=True)
def ewma(data, period, alpha=1.0):
    """
    Exponential Weighted Moving Average
    :type data: np.ndarray
    :type period: int
    :type alpha: float
    :rtype: np.ndarray
    """
    weights = (1 - alpha) ** np.arange(period)
    weights /= np.sum(weights)
    out = convolve(data, weights)
    return np.concatenate((np.array([np.nan] * (len(data) - len(out))), out))


@jit(nopython=True)
def dema(data, period, smoothing=2.0):
    """
    Double Exponential Moving Average
    :type data: np.ndarray
    :type period: int
    :type smoothing: float
    :rtype: np.ndarray
    """
    return (2 * ema(data, period, smoothing)) - ema(ema(data, period, smoothing), period, smoothing)


@jit(nopython=True)
def trix(data, period, smoothing=2.0):
    """
    Triple Exponential Moving Average
    :type data: np.ndarray
    :type period: int
    :type smoothing: float
    :rtype: np.ndarray
    """
    return ((3 * ema(data, period, smoothing) - (3 * ema(ema(data, period, smoothing), period, smoothing))) +
            ema(ema(ema(data, period, smoothing), period, smoothing), period, smoothing))


@jit(nopython=True)
def macd(data, fast, slow, smoothing=2.0):
    """
    Moving Average Convergence Divergence
    :type data: np.ndarray
    :type fast: int
    :type slow: int
    :type smoothing: float
    :rtype: np.ndarray
    """
    return ema(data, fast, smoothing) - ema(data, slow, smoothing)


@jit(nopython=True)
def stoch(c_close, c_high, c_low, period_k, period_d):
    """
    Stochastic
    :type c_close: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type period_k: int
    :type period_d: int
    :rtype: (np.ndarray, np.ndarray)
    """
    size = len(c_close)
    k = np.array([np.nan] * size)
    for i in range(period_k - 1, size):
        e = i + 1
        s = e - period_k
        ml = np.min(c_low[s:e])
        if ml == np.max(c_high[s:e]):
            ml = ml - 0.1
        k[i] = ((c_close[i] - ml) / (np.max(c_high[s:e]) - ml)) * 100
    return k, sma(k, period_d)


@jit(nopython=True)
def kdj(c_close, c_high, c_low, period_rsv=9, period_k=3, period_d=3, weight_k=3, weight_d=2):
    """
    KDJ
    :type c_close: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type period_rsv: int
    :type period_k: int
    :type period_d: int
    :type weight_k: int
    :type weight_d: int
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """
    size = len(c_close)
    rsv = np.array([np.nan] * size)
    for i in range(period_k - 1, size):
        e = i + 1
        s = e - period_k
        ml = np.min(c_low[s:e])
        rsv[i] = ((c_close[i] - ml) / (np.max(c_high[s:e]) - ml)) * 100
    k = sma(rsv, period_rsv)
    d = sma(k, period_d)
    return k, d, (weight_k * k) - (weight_d * d)


@jit(nopython=True)
def wpr(c_close, c_high, c_low, period):
    """
    William %R
    :type c_close: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type period: int
    :rtype: (np.ndarray, np.ndarray)
    """
    size = len(c_close)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        e = i + 1
        s = e - period
        mh = np.max(c_high[s:e])
        out[i] = ((mh - c_close[i]) / (mh - np.min(c_low[s:e]))) * -100
    return out


@jit(nopython=True)
def rsi(data, period, smoothing=2.0, f_sma=True, f_clip=True, f_abs=True):
    """
    Relative Strengh Index
    :type data: np.ndarray
    :type period: int
    :type smoothing: float
    :type f_sma: bool
    :type f_clip: bool
    :type f_abs: bool
    :rtype: np.ndarray
    """
    size = len(data)
    delta = np.array([np.nan] * size)
    up = np.array([np.nan] * size)
    down = np.array([np.nan] * size)
    delta = np.diff(data)
    if f_clip:
        up, down = np.clip(delta, a_min=0, a_max=np.max(delta)), np.clip(delta, a_min=np.min(delta), a_max=0)
    else:
        up, down = delta.copy(), delta.copy()
        up[delta < 0] = 0.0
        down[delta > 0] = 0.0
    if f_abs:
        for i, x in enumerate(down):
            down[i] = fabs(x)
    else:
        down = np.abs(down)
    rs = sma(up, period) / sma(down, period) if f_sma else ema(up, period - 1, smoothing) / ema(
        down, period - 1, smoothing)
    out = np.array([np.nan] * size)
    out[1:] = (100 - 100 / (1 + rs))
    return out


@jit(nopython=True)
def srsi(data, period, smoothing=2.0, f_sma=True, f_clip=True, f_abs=True):
    """
    Stochastic Relative Strengh Index
    :type data: np.ndarray
    :type period: int
    :type smoothing: float
    :type f_sma: bool
    :type f_clip: bool
    :type f_abs: bool
    :rtype: np.ndarray
    """
    r = rsi(data, period, smoothing, f_sma, f_clip, f_abs)[period:]
    s = np.array([np.nan] * len(r))
    for i in range(period - 1, len(r)):
        window = r[i + 1 - period:i + 1]
        mw = np.min(window)
        s[i] = ((r[i] - mw) / (np.max(window) - mw)) * 100
    return np.concatenate((np.array([np.nan] * (len(data) - len(s))), s))


@jit(nopython=True)
def cmo(c_close, period, f_sma=True, f_clip=True, f_abs=True):
    """
    `
    :type c_close: np.ndarray
    :type period: int
    :type f_sma: bool
    :type f_clip: bool
    :type f_abs: bool
    :rtype: np.ndarray
    """
    size = len(c_close)
    deltas = np.array([np.nan] * size)
    sums_up = np.array([np.nan] * size)
    sums_down = np.array([np.nan] * size)
    for i in range(period - 1, size):
        window = c_close[i + 1 - period:i + 1]
        d = np.diff(window)
        if f_clip:
            up, down = np.clip(d, a_min=0, a_max=np.max(d)), np.clip(d, a_min=np.min(d), a_max=0)
        else:
            up, down = d.copy(), d.copy()
            up[d < 0] = 0.0
            down[d > 0] = 0.0
        if f_abs:
            for j, x in enumerate(down):
                down[j] = fabs(x)
        else:
            down = np.abs(down)
        sums_up[i] = sum(up)
        sums_down[i] = sum(down)
    return 100 * ((sums_up - sums_down) / (sums_up + sums_down))


@jit(nopython=True)
def bollinger_bands(data, period, dev_up=2.0, dev_down=2.0):
    """
    Bollinger Bands
    :type data: np.ndarray
    :type period: int
    :type dev_up: float
    :type dev_down: float
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    :return: middle, up, down, width
    """
    size = len(data)
    bb_up = np.array([np.nan] * size)
    bb_down = np.array([np.nan] * size)
    bb_width = np.array([np.nan] * size)
    bb_mid = sma(data, period)
    for i in range(period - 1, size):
        std_dev = np.std(data[i - period + 1:i + 1])
        mid = bb_mid[i]
        bb_up[i] = mid + (std_dev * dev_up)
        bb_down[i] = mid - (std_dev * dev_down)
        bb_width[i] = bb_up[i] - bb_down[i]
    return bb_mid, bb_up, bb_down, bb_width


@jit(nopython=True)
def keltner_channel(c_close, c_open, c_high, c_low, period, smoothing=2.0):
    """
    Keltner Channel
    :type c_close: np.ndarray
    :type c_open: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type period: int
    :type smoothing: float
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    :return: middle, up, down, width
    """
    e = ema(c_close, period, smoothing)
    aa = 2 * atr(c_open, c_high, c_low, period)
    up = e + aa
    down = e - aa
    return e, up, down, up - down


@jit(nopython=True)
def donchian_channel(c_high, c_low, period):
    """
    Donchian Channel
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type period: int
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    :return: middle, up, down, width
    """
    size = len(c_high)
    out_up = np.array([np.nan] * size)
    out_down = np.array([np.nan] * size)
    for i in range(period - 1, size):
        e = i + 1
        s = e - period
        out_up[i] = np.max(c_high[s:e])
        out_down[i] = np.min(c_low[s:e])
    return (out_up + out_down) / 2, out_up, out_down, out_up - out_down


@jit(nopython=True)
def heiken_ashi(c_open, c_high, c_low, c_close):
    """
    Heiken Ashi
    :type c_open: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type c_close: np.ndarray
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    :return: open, high, low, close
    """
    ha_close = (c_open + c_high + c_low + c_close) / 4
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (c_open[0] + c_close[0]) / 2
    for i in range(1, len(c_close)):
        ha_open[i] = (c_open[i - 1] + c_close[i - 1]) / 2
    return \
        ha_open, np.maximum(np.maximum(ha_open, ha_close), c_high), np.minimum(np.minimum(ha_open, ha_close), c_low), \
        ha_close


@jit(nopython=True)
def ichimoku(data, tenkansen=9, kinjunsen=26, senkou_b=52, shift=26):
    """
    Ichimoku
    :type data: np.ndarray
    :type tenkansen: int
    :type kinjunsen: int
    :type senkou_b: int
    :type shift: int
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    :return: tenkansen, kinjunsen, chikou, senkou a, senkou b
    """
    size = len(data)
    n_tenkansen = np.array([np.nan] * size)
    n_kinjunsen = np.array([np.nan] * size)
    n_senkou_b = np.array([np.nan] * (size + shift))
    for i in range(tenkansen - 1, size):
        window = data[i + 1 - tenkansen:i + 1]
        n_tenkansen[i] = (np.max(window) + np.min(window)) / 2
    for i in range(kinjunsen - 1, size):
        window = data[i + 1 - kinjunsen:i + 1]
        n_kinjunsen[i] = (np.max(window) + np.min(window)) / 2
    for i in range(senkou_b - 1, size):
        window = data[i + 1 - senkou_b:i + 1]
        n_senkou_b[i + shift] = (np.max(window) + np.min(window)) / 2
    return \
        n_tenkansen, n_kinjunsen, np.concatenate(((data[shift:]), (np.array([np.nan] * (size - shift))))), \
        np.concatenate((np.array([np.nan] * shift), ((n_tenkansen + n_kinjunsen) / 2))), n_senkou_b


@jit(nopython=True)
def volume_profile(c_close, c_volume, bins=10):
    """
    Volume Profile
    :type c_close: np.ndarray
    :type c_volume: np.ndarray
    :type bins: int
    :rtype: (np.ndarray, np.ndarray)
    :return: count, price
    """
    min_close = np.min(c_close)
    max_close = np.max(c_close)
    norm = 1.0 / (max_close - min_close)
    sum_h = np.array([0.0] * bins)
    for i in range(len(c_close)):
        sum_h[int((c_close[i] - min_close) * bins * norm)] += c_volume[i] ** 2
    sq = np.sqrt(sum_h)
    return sq / sum(sq), np.linspace(min_close, max_close, bins)


@jit(nopython=True)
def tr(c_open, c_high, c_low):
    """
    True Range
    :type c_open: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :rtype: np.ndarray
    """
    return np.maximum(np.maximum(c_open - c_low, np.abs(c_high - c_open)), np.abs(c_low - c_open))


@jit(nopython=True)
def atr(c_open, c_high, c_low, period):
    """
    Average True Range
    :type c_open: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    return sma(tr(c_open, c_high, c_low), period)


@jit(nopython=True)
def adx(c_open, c_high, c_low, period_adx, period_dm, smoothing=2.0):
    """
    Average Directionnal Index
    :type c_open: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type period_adx: int
    :type period_dm: int
    :type smoothing: float
    :rtype: np.ndarray
    """
    up = np.concatenate((np.array([np.nan]), c_high[1:] - c_high[:-1]))
    down = np.concatenate((np.array([np.nan]), c_low[:-1] - c_low[1:]))
    dm_up = np.array([0] * len(up))
    up_ids = up > down
    dm_up[up_ids] = up[up_ids]
    dm_up[dm_up < 0] = 0
    dm_down = np.array([0] * len(down))
    down_ids = down > up
    dm_down[down_ids] = down[down_ids]
    dm_down[dm_down < 0] = 0
    avg_tr = atr(c_open, c_high, c_low, period_dm)
    dm_up_avg = 100 * ema(dm_up, period_dm, smoothing) / avg_tr
    dm_down_avg = 100 * ema(dm_down, period_dm, smoothing) / avg_tr
    return ema(100 * np.abs(dm_up_avg - dm_down_avg) / (dm_up_avg + dm_down_avg), period_adx, smoothing)


@jit(nopython=True)
def obv(c_close, c_volume):
    """
    On Balance Volume
    :type c_close: np.ndarray
    :type c_volume: np.ndarray
    :rtype: np.ndarray
    """
    size = len(c_close)
    out = np.array([np.nan] * size)
    out[0] = 1
    for i in range(1, size):
        if c_close[i] > c_close[i - 1]:
            out[i] = out[i - 1] + c_volume[i]
        elif c_close[i] < c_close[i - 1]:
            out[i] = out[i - 1] - c_volume[i]
        else:
            out[i] = out[i - 1]
    return out


@jit(nopython=True)
def momentum(data, period):
    """
    Momentum
    :type data: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    size = len(data)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        out[i] = data[i] - data[i - period + 1]
    return out


@jit(nopython=True)
def roc(data, period):
    """
    Rate Of Change
    :type data: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    size = len(data)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        p = data[i - period + 1]
        out[i] = ((data[i] - p) / p) * 100
    return out


@jit(nopython=True)
def aroon(data, period):
    """
    Aroon
    :type data: np.ndarray
    :type period: int
    :rtype: (np.ndarray, np.ndarray)
    :return: up, down
    """
    size = len(data)
    out_up = np.array([np.nan] * size)
    out_down = np.array([np.nan] * size)
    for i in range(period - 1, size):
        window = np.flip(data[i + 1 - period:i + 1])
        out_up[i] = ((period - window.argmax()) / period) * 100
        out_down[i] = ((period - window.argmin()) / period) * 100
    return out_up, out_down


@jit(nopython=True)
def cmf(c_close, c_high, c_low, c_volume, period):
    """
    Chaikin Money Flow
    :type c_close: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type c_volume: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    size = len(c_close)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        e = i + 1
        s = e - period
        w_close = c_close[s:e]
        w_high = c_high[s:e]
        w_low = c_low[s:e]
        w_vol = c_volume[s:e]
        out[i] = sum((((w_close - w_low) - (w_high - w_close)) / (w_high - w_low)) * w_vol) / sum(w_vol)
    return out


@jit(nopython=True)
def vix(c_close, c_low, period):
    """
    Volatility Index
    :type c_close: np.ndarray
    :type c_low: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    size = len(c_close)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        hc = np.max(c_close[i + 1 - period:i + 1])
        out[i] = ((hc - c_low[i]) / hc) * 100
    return out


@jit(nopython=True)
def fdi(c_close, period):
    """
    Fractal Dimension Index
    :type c_close: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    size = len(c_close)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        window = c_close[i + 1 - period:i + 1]
        pdiff = 0
        length = 0
        hc = np.max(window)
        lc = np.min(window)
        for j in (range(1, period - 1)):
            if hc > lc:
                diff = (window[-j] - lc) / (hc - lc)
                length += np.sqrt(((diff - pdiff) + (1 / (period ** 2))) ** 2) if j > 1 else 0
                pdiff = diff
        out[i] = (1 + (np.log(length) + np.log(2)) / np.log(2 * period)) if length > 0 else 0
    return out


@jit(nopython=True)
def entropy(c_close, c_volume, period, bins=2):
    """
    Entropy (Experimental)
    :type c_close: np.ndarray
    :type c_volume: np.ndarray
    :type period: int
    :type bins: int
    :rtype: np.ndarray
    """
    size = len(c_close)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        e = i + 1
        s = e - period
        close_w = c_close[s:e]
        volume_w = c_volume[s:e]
        min_w = np.min(close_w)
        norm = 1.0 / (np.max(close_w) - min_w)
        sum_h = np.array([0.0] * bins)
        for j in range(period):
            sum_h[int((close_w[j] - min_w) * bins * norm)] += volume_w[j] ** 2
        count = np.sqrt(sum_h)
        count = count / sum(count)
        count = count[np.nonzero(count)]
        out[i] = -sum(count * np.log2(count))
    return out


@jit(nopython=True)
def poly_fit_extra(data, deg=1, extra=0):
    """
    Polynomial Fit Extrapolation
    :type data: np.ndarray
    :type deg: int
    :type extra: int
    :rtype: np.ndarray
    """
    size = len(data)
    x = np.arange(0, size, 1)
    m = np.ones((x.shape[0], deg + 1))
    m[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            m[:, n] = m[:, n - 1] * x
    scale = np.empty((deg + 1,))
    for n in range(0, deg + 1):
        norm = np.linalg.norm(m[:, n])
        scale[n] = norm
        m[:, n] /= norm
    lsf = (np.linalg.lstsq(m, data, rcond=-1)[0] / scale)[::-1]
    lx = np.arange(0, size + extra, 1)
    out = np.zeros(lx.shape)
    for i, v in enumerate(lsf):
        out *= lx
        out += v
    return out


@jit(nopython=True)
def fourier_fit_extra(data, harmonic, extra=0):
    """
    Fourier Transform Fit Extrapolation
    :type data: np.ndarray
    :type harmonic: int
    :type extra: int
    :rtype: np.ndarray
    """
    size = len(data)
    x = np.arange(0, size, 1)
    m = np.ones((x.shape[0], 2))
    m[:, 1] = x
    scale = np.empty((2,))
    for n in range(0, 2):
        norm = np.linalg.norm(m[:, n])
        scale[n] = norm
        m[:, n] /= norm
    lsf = (np.linalg.lstsq(m, data, rcond=-1)[0] / scale)[::-1]
    lsd = data - lsf[0] * x
    size_lsd = len(lsd)
    four = np.zeros(size_lsd, dtype=np.complex128)
    for i in range(size_lsd):
        sum_f = 0
        for n in range(size_lsd):
            sum_f += lsd[n] * np.exp(-2j * np.pi * i * n * (1 / size_lsd))
        four[i] = sum_f
    freq = np.empty(size)
    mi = (size - 1) // 2 + 1
    freq[:mi] = np.arange(0, mi)
    freq[mi:] = np.arange(-(size // 2), 0)
    freq *= 1.0 / size
    lx = np.arange(0, size + extra)
    out = np.zeros(lx.shape)
    index = [v for _, v in sorted([(np.absolute(four[v]), v) for v in list(range(size))])][::-1]
    for i in index[:1 + harmonic * 2]:
        out += (abs(four[i]) / size) * np.cos(2 * np.pi * freq[i] * lx + np.angle(four[i]))
    return out + lsf[0] * lx


@jit(nopython=True)
def super_trend(c_close, c_open, c_high, c_low, period_atr=10, multi=3):
    """
    Supertrend
    :type c_close: np.ndarray
    :type c_open: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type period_atr: int
    :type multi: int
    :rtype: (np.ndarray, np.ndarray)
    :return: up, down
    """
    size = len(c_close)
    avg_tr = atr(c_open, c_high, c_low, period_atr)
    hl2 = (c_high + c_low) / 2
    b_up = hl2 + (multi * avg_tr)
    b_down = hl2 - (multi * avg_tr)
    st = np.array([np.nan] * size)
    for i in range(1, size):
        j = i - 1
        if c_close[i] > b_up[j]:
            st[i] = 1
        elif c_close[i] < b_down[j]:
            st[i] = 0
        else:
            st[i] = st[j]
            if st[i] == 1 and b_down[i] < b_down[j]:
                b_down[i] = b_down[j]
            if st[i] == 0 and b_up[i] > b_up[j]:
                b_up[i] = b_up[j]
    return np.where(st == 1, b_down, np.nan), np.where(st == 0, b_up, np.nan)


@jit(nopython=True)
def chop(c_close, c_open, c_high, c_low, period=14):
    """
    Chopiness Index
    :type c_close: np.ndarray
    :type c_open: np.ndarray
    :type c_high: np.ndarray
    :type c_low: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    size = len(c_close)
    out = np.array([np.nan] * size)
    a_tr = atr(c_open, c_high, c_low, period)
    for i in range(period - 1, size):
        e = i + 1
        s = e - period
        out[i] = (100 * np.log10(np.sum(a_tr[s:e]) / (np.max(c_high[s:e]) - np.min(c_low[s:e])))) / np.log10(period)
    return out


@jit(nopython=True)
def cog(data, period=10):
    """
    Center Of Gravity
    :type data: np.ndarray
    :type period: int
    :rtype: np.ndarray
    """
    size = len(data)
    out = np.array([np.nan] * size)
    for i in range(period - 1, size):
        e = i + 1
        s = e - period
        window = data[s:e]
        den = np.sum(window)
        num = 0
        for j in range(0, period):
            num += window[j] * (period - j)
        out[i] = - num / den
    return out
