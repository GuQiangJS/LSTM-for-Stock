"""数据处理器"""

import datetime
import socket
from LSTM_for_Stock import indicators
import QUANTAXIS as QA
import pandas as pd
from sklearn.preprocessing import normalize
import logging
from QUANTAXIS.QAUtil import QA_util_datetime_to_strdate as datetostr
from QUANTAXIS.QAFetch.QAQuery import QA_fetch_stock_to_market_date
from QUANTAXIS.QAFetch.QAQuery_Advance import QA_fetch_stock_block_adv


class Wrapper(object):
    """數據包裝器"""

    def build(self, df):  # pylint: disable=C0103
        """數據處理

        Args:
            df (pd.DataFrame): 待處理的數據

        Returns:
            pd.DataFrame : 處理后的數據
        """
        return df


class Wrapper_fillna(Wrapper):
    def build(self, df):
        """數據處理。
        1. **向前**填充 stock 的数据。
        2. 删除 stock 中依然为空的数据。（上市时间在benchmark之后的情况。）

        Args:
            df (pd.DataFrame): 待處理的數據

        Returns:
            pd.DataFrame : 處理后的數據
        """
        result = df.copy()
        result = result.fillna(method='ffill')
        return result.dropna()


class Wrapper_default(Wrapper):
    def build(self, df):
        result = df.copy()
        result = result.fillna(method='ffill')
        result = result.drop(columns=['up_count', 'down_count'])
        return result.dropna()


class Wrapper_remove_benchmark(Wrapper_default):
    def build(self, df):
        result = super(Wrapper_remove_benchmark, self).build(df)
        result = result.drop(
            columns=[f for f in result.columns if 'bench' in f])
        return result.dropna()


class Wrapper_append_features(Wrapper):
    def build(self, df):
        result = df.copy()
        result = result.fillna(method='ffill')
        result = result.drop(columns=['up_count', 'down_count'])

        result['EMA_5'] = QA.QA_indicator_EMA(result, 5)
        result['CCI_5'] = QA.talib_indicators.CCI(result, 5)
        result['RSI_5'] = QA.QA_indicator_RSI(result, 5, 5, 5)['RSI1']
        result['MOM_5'] = indicators.talib_MOM(result, 5)
        result[[
            'BB_SMA_LOWER_5', 'BB_SMA_MIDDLE_5',
            'BB_SMA_UPPER_5']] = indicators.talib_BBANDS(
            result, 5)
        # result[['AROON_DOWN_5', 'AROON_UP_5']] = QA.talib_indicators.AROON(
        #     result, 5)
        # result['AROONOSC_5'] = QA.talib_indicators.AROONOSC(result,5)

        return result.dropna()


class DataLoader(object):
    """數據提供器"""

    @staticmethod
    def today():
        return datetime.datetime.today().strftime('%Y-%m-%d')


class DataLoaderStock(DataLoader):
    """股票數據提供器

    Args:
        stock_code (str): 股票代碼
        benchmark_code (str): 指數代碼
        fq (str, optional): Defaults to 'qfq'. 是否取復權數據。
            * `qfq` - 前復權
            * `hfq` - 后復權
            * `bfq` or `None` - 不復權
        online (bool, optional): Defaults to False. 是否獲取在線數據
        start (str, optional): Defaults to '1990-01-01'. 開始日期
        end (str, optional): Defaults to DataLoader.today(). 結束日期
    """

    def __init__(self,
                 stock_code,
                 benchmark_code='399300',
                 fq='qfq',
                 online=False,
                 start='1990-01-01',
                 end=DataLoader.today(),
                 wrapper=Wrapper(),
                 *args, **kwargs):
        """股票數據提供器
        
        Args:
            stock_code (str): 股票代碼
            benchmark_code (str, optional): Defaults to '399300'. 指數代碼
            fq (str, optional): Defaults to 'qfq'. 是否取復權數據。
                * `qfq` - 前復權
                * `hfq` - 后復權
                * `bfq` or `None` - 不復權
            online (bool, optional): Defaults to False. 是否獲取在線數據
            start (str, optional): Defaults to '1990-01-01'. 開始日期
            end (str, optional): Defaults to DataLoader.today(). 結束日期
            wrapper (Wrapper, optional): Defaults to Wrapper(). `DataFrame`包装器。
            appends (str): 待附加的股票代码列表。默认为空。
        """
        self.__stock_code = stock_code
        self.__benchmark_code = benchmark_code
        self.__online = online
        self.__start = start
        self.__end = end
        self.__fq = fq
        self.__wrapper = wrapper
        self.__data_raw = pd.DataFrame()
        self.__data = pd.DataFrame()
        self.__loaded = False
        self.__appends = kwargs.pop('appends', [])

    def load(self) -> pd.DataFrame:
        """读取数据。拼接 stock 和 benchmark 的数据。
        *以 benchmark 的数据作为左侧数据源， stock 的数据作为右侧数据源*。
        **合并后会调用 `self.__wrapper.build` 方法对数据源进行包装。
        
        Returns:
            pd.DataFrame: 合并后的数据。返回的数据与 `self.data` 一致。
        """
        if self.__online:
            bench_df = self.__fetch_index_day_online()
        else:
            bench_df = self.__fetch_index_day()
        stock_df = self.__fetch_stock_day_core(self.__stock_code)
        self.__data_raw = stock_df.join(bench_df, rsuffix='_bench')
        for c in self.__appends:
            code_df = self.__fetch_stock_day_core(c,
                                                  start=datetostr(
                                                      bench_df.index[0]))
            if not code_df.empty:
                self.__data_raw = self.__data_raw.join(code_df, rsuffix='_' + c)

        self.__data = self.__data_raw.copy()
        if self.__wrapper:
            self.__data = self.__wrapper.build(self.__data)
        self.__loaded = True
        return self.__data

    def __fetch_stock_day_core(self, code: str, start=None, end=None):
        if not start:
            start = self.__start
        if not end:
            end = self.__end
        if self.__online:
            return self.__fetch_stock_day_online(code)
        else:
            return self.__fetch_stock_day(code, start, end)

    @property
    def loaded(self) -> bool:
        """是否已经读取过数据"""
        return self.__loaded

    @property
    def data_raw(self) -> pd.DataFrame:
        """原始数据"""
        if not self.loaded:
            self.load()
        return self.__data_raw

    @property
    def data(self) -> pd.DataFrame:
        """处理后的数据"""
        if not self.loaded:
            self.load()
        return self.__data

    @property
    def stock_code(self) -> str:
        """股票代码"""
        return self.__stock_code

    @property
    def benchmark_code(self) -> str:
        """指数代码"""
        return self.__benchmark_code

    @property
    def start(self) -> str:
        """数据开始日期"""
        return self.__start

    @property
    def end(self) -> str:
        """数据结束日期"""
        return self.__end

    @property
    def online(self) -> bool:
        """是否在线获取数据"""
        return self.__online

    @property
    def fq(self) -> str:
        """复权处理"""
        return self.__fq

    def __fetch_stock_day(self, code, start, end) -> pd.DataFrame:
        """獲取本地的股票日線數據。會丟棄 `code` 列，并根據 `self.__fq` 來確定是否取復權數據。
        返回列名：_stock_columns
        """
        d = QA.QA_fetch_stock_day_adv(code, start=start, end=end)
        if not d:
            return pd.DataFrame()
        if self.__fq == 'qfq':
            df = d.to_qfq()
        elif self.__fq == 'hfq':
            df = d.to_hfq()
        else:
            df = d.data
        # 原始列名['open', 'high', 'low', 'close', 'volume', 'amount', 'preclose', 'adj']
        df = df.reset_index().drop(columns=['code']).set_index('date')
        df = df.astype('float32')
        return df[self._stock_columns]

    def __fetch_index_day(self) -> pd.DataFrame:
        """獲取本地的指數日線數據。會丟棄 `code` 列。
        返回列名：_index_columns"""
        d = QA.QA_fetch_index_day_adv(
            self.__benchmark_code, start=self.__start, end=self.__end)
        df = d.data.reset_index().drop(columns=['code']).set_index('date')
        df = df.astype('float32')
        # 原始列名['open', 'high', 'low', 'close', 'up_count', 'down_count', 'volume','amount'],
        return df[self._index_columns]

    @property
    def _index_columns(self) -> [str]:
        return [
            'open', 'high', 'low', 'close', 'volume', 'amount', 'up_count',
            'down_count'
        ]

    @property
    def _stock_columns(self) -> [str]:
        return ['open', 'high', 'low', 'close', 'volume', 'amount']

    def __fetch_stock_day_online(self, code, start, end,
                                 times=5) -> pd.DataFrame:
        """讀取股票在線日線數據。
            times (int, optional): Defaults to 5. 重試次數
        返回列名：_stock_columns。
        """
        retries = 0
        while True:
            try:
                df = QA.QAFetch.QATdx.QA_fetch_get_stock_day(code,
                                                             start,
                                                             end)
                df = df.astype('float32')
                # 原始列名['open', 'close', 'high', 'low', 'vol', 'amount', 'code', 'date', 'date_stamp'] pylint: disable=C0301
                return df.rename(columns={'vol': 'volume'})[self._stock_columns]
            except socket.timeout:
                if retries < times:
                    retries = retries + 1
                    continue
                raise

    def __fetch_index_day_online(self, times=5) -> pd.DataFrame:
        """讀取指數在線日線數據。
            times (int, optional): Defaults to 5. 重試次數
        返回列名：_index_columns。
        """
        retries = 0
        while True:
            try:
                df = QA.QAFetch.QATdx.QA_fetch_get_index_day(
                    self.__benchmark_code, self.__start, self.__end)
                df = df.astype('float32')
                # 原始列名['open', 'close', 'high', 'low', 'vol', 'amount', 'up_count', 'down_count', 'date', 'code', 'date_stamp'] pylint: disable=C0301
                return df.rename(columns={'vol': 'volume'})[self._index_columns]
            except socket.timeout:
                if retries < times:
                    retries = retries + 1
                    continue
                raise


class Normalize(object):
    """数据标准化器"""

    def __init__(self, *args, **kwargs):
        pass

    def build(self, df):
        """执行数据标准化。**数据归一化**。

        Args:
            df (pd.DataFrame 或 pd.Series): 待处理的数据。

        Returns:
            pd.DataFrame 或 pd.Series: 与传入类型一致。
        """
        return df / df.iloc[0]


class Normalize_Empty(Normalize):
    """空白的数据标准化器"""

    def build(self, df):
        """执行数据标准化。返回原始数据。"""
        return df


# class Normalize_append_features(object):
#     """数据标准化器"""
#
#     def __init__(self, *args, **kwargs):
#         pass
#
#     def build(self, df):
#         """执行数据标准化。**数据归一化**。
#
#         Args:
#             df (pd.DataFrame 或 pd.Series): 待处理的数据。
#
#         Returns:
#             pd.DataFrame 或 pd.Series: 与传入类型一致。
#         """
#         tmp = df.copy()
#         for col in tmp.columns:
#             tmp[col] = normalize([tmp[col]])[0]
#             tmp[col + '_N1'] = tmp[col] / tmp.iloc[0][col]
#             # tmp[col + '_N3'] = tmp[col].pct_change().fillna(0)
#             # if col in ['CCI_5', 'RSI_5','AROON_UP_5','AROON_DOWN_5']:
#             #     tmp[col] = sklearn.preprocessing.normalize([tmp[col]])[0]
#             # elif col in ['MOM_5']:
#             #     continue
#             # tmp[col] = tmp[col] / tmp.iloc[0][col]
#         return tmp


class DataHelper(object):
    @staticmethod
    def train_test_split(df,
                         batch_size,
                         train_size=0.85,
                         start=None,
                         end=None):
        """拆分训练集和测试集。
        
        Args:
            df (pd.DataFrame): 数据源
            batch_size (int): 每一批次的数据量。一般来说是window+days。
                window: 窗口期日期长度。拆分后的训练集/测试集中每一个单一项会包含多少个日期的数据。
                days: 结果日期长度。拆分后的结果集中每一个单一项会包含多少个日期的数据。
            train_size (float, optional): Defaults to 0.85. 训练集比率。应该在 0.0 ~ 1.0 之间。
            start (int, optional): Defaults to None. 从 `self.data` 中取值的开始下标。
            end (int, optional): Defaults to None. 从 `self.data` 中取值的结束下标。
            norm (func, optional): Defaults to None. 对每一批次的数据（X+Y）执行标准化的方法。

        Returns:
            [[pd.DataFrame],[pd.DataFrame]]: 训练集X+Y,测试集X+Y
        """
        if train_size is None:
            train_size = 0.85

        if df.empty:
            raise ValueError('df is empty')

        if not start and not end:
            # 开始/结束的下标均为空，表示取所有
            df_tmp = df.copy()
        elif not start:
            df_tmp = df.copy().iloc[start:]
        elif not end:
            df_tmp = df.copy().iloc[:end]
        else:
            df_tmp = df.copy().iloc[start:end]
        X = []
        for i in range(df_tmp.shape[0]):
            if i + batch_size > df_tmp.shape[0]:
                break
            X.append(df_tmp[i:i + batch_size])  # 当前取出需要分割为X，Y的批次数据

        train_end_index = round(train_size * len(X))  # 训练集结束的位置
        return X[:train_end_index], X[train_end_index:]

    @staticmethod
    def xy_split_1(dfs, window, days, col_name='close', norm=Normalize()):
        """拆分 `train_test_split` 返回的 `train` 和 `test` 结果。

        Args:
            dfs ([pd.DataFrame]): `train` 和 `test` 结果
            norm: 数据构造器。实现了 `build` 方法的类均可。
                如果为空则会自动为 `Normalize_Empty` 。
                默认为 `Normalize`。
            window (int): 窗口期
            days (int): 结果期
            col_name (str): 结果期取值的列名

        Returns:
            [[pd.DataFrame],[pd.Series]]: 按照 window,days 拆分后的集合。
                X中包含所有列。Y中只包含 `col_name` 指定的列。
                **Y 返回的结果是相对于返回集合的前一条数据做的`norm`处理。**

        See Also:
            * :py:func:`DataHelper.xy_split_2`

        Examples:

            >>> arr = [i for i in range(2, 8)]
            >>> window = len(arr) - 2
            >>> days = 2
            >>> window, days
            4, 2
            >>> x, y = DataHelper.xy_split_1([pd.DataFrame(arr, columns=['c'])],
            ...                              window, days, col_name='c')
            >>> x
            [      c
                0  1.0
                1  1.5
                2  2.0
                3  2.5]

            y 的结果为 先取 [5, 6, 7] ，然后返回 [6/5.7/5]

            >>> y
            [4    1.2
             5    1.4
             Name: c, dtype: float64]
            >>> type(x[0])
            <class 'pandas.core.frame.DataFrame'>
            >>> type(y[0])
            <class 'pandas.core.series.Series'>

        """
        X = []
        Y = []
        if norm is None:
            norm = Normalize()
        for df in dfs:
            df_tmp = df.copy()
            X.append(norm.build(df_tmp)[:window])
            Y.append(norm.build(df_tmp[-1 - days:])[col_name][1:1 + days])
        return X, Y

    @staticmethod
    def xy_split_2(dfs, window, days, col_name='close', norm=Normalize()):
        """拆分 `train_test_split` 返回的 `train` 和 `test` 结果。

        Args:
            dfs ([pd.DataFrame]): `train` 和 `test` 结果
            norm: 数据构造器。实现了 `build` 方法的类均可。
                如果为空则会自动为 `Normalize_Empty` 。
                默认为 `Normalize`。
            window (int): 窗口期
            days (int): 结果期
            col_name (str): 结果期取值的列名

        Returns:
            [[pd.DataFrame],[pd.Series]]: 按照 window,days 拆分后的集合。
                X中包含所有列。Y中只包含 `col_name` 指定的列。
                **Y 返回的结果是相对于返回集合的前一条数据做的`norm`处理。**

        See Also:
            * :py:func:`LSTM_for_Stock.DataHelper.xy_split_1`

        Examples:

            >>> arr = [i for i in range(2, 8)]
            >>> window = len(arr) - 2
            >>> days = 2
            >>> window, days
            4, 2
            >>> x, y = DataHelper.xy_split_2([pd.DataFrame(arr, columns=['c'])],
            ...                              window, days, col_name='c')
            >>> x
            [      c
                0  1.0
                1  1.5
                2  2.0
                3  2.5]

            **与 :func:`~DataHelper.xy_split_1` 不同的是 对整个集合取一次 `norm` 操作。然后直接取值。
            也就是 `y` 的值是相对于整个集合的第一条的。**

            >>> y
            [4    3.0
            5    3.5
            Name: c, dtype: float64]
            >>> type(x[0])
            <class 'pandas.core.frame.DataFrame'>
            >>> type(y[0])
            <class 'pandas.core.series.Series'>

        """
        X = []
        Y = []
        if norm is None:
            norm = Normalize()
        for df in dfs:
            df_tmp = norm.build(df.copy())
            X.append(df_tmp[:window])
            Y.append(df_tmp[-1 - days:][col_name][1:1 + days])
        return X, Y


def get_ipo_date(code):
    """获取上市日期

    Args:
        code (str): 股票代码

    Returns:
        datetime: 如果获取失败或者有异常会返回None
    """
    result = _get_tdx_ipo_date(code)
    if not result:
        result = _get_tushare_ipo_date(code)
    return result


def _get_tushare_ipo_date(code):
    try:
        return datetime.datetime.strptime(QA_fetch_stock_to_market_date(code),
                                          '%Y-%m-%d')
    except:
        return None


def _get_tdx_ipo_date(code):
    try:
        info = QA.QA_fetch_stock_info(code)
        if not info.empty:
            return datetime.datetime.strptime(str(info.loc[code]['ipo_date']),
                                              '%Y%m%d')
    except:
        return None


def get_block_code(code, type='zjhhy') -> (str):
    """按照证监会行业分类，获取指定股票的同分类所有股票代码"""
    df = QA_fetch_stock_block_adv()
    k = df.get_code(code).data
    name = k[k['type'] == type].reset_index()['blockname'].values[0]
    return [c for c in df.get_block(name).code if c != code]
