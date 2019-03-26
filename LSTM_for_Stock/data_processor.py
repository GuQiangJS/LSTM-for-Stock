import datetime
import pandas as pd
import QUANTAXIS as QA
import socket


class Wrapper(object):
    """數據包裝器"""

    def __init__(self, kwargs: dict = {}):
        self.kwargs = kwargs

    def build(self, df):
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
                 wrapper=Wrapper()):
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
        self.__loaded=False

    def load(self) -> pd.DataFrame:
        """读取数据。拼接 stock 和 benchmark 的数据。
        *以 benchmark 的数据作为左侧数据源， stock 的数据作为右侧数据源*。
        **合并后会调用 `self.__wrapper.build` 方法对数据源进行包装。
        
        Returns:
            pd.DataFrame: 合并后的数据。返回的数据与 `self.data` 一致。
        """
        if self.__online:
            stock_df = self.__fetch_stock_day_online()
            bench_df = self.__fetch_index_day_online()
        else:
            stock_df = self.__fetch_stock_day()
            bench_df = self.__fetch_index_day()
        self.__data_raw = bench_df.join(stock_df, lsuffix='_bench')
        self.__data = self.__data_raw.copy()
        if self.__wrapper:
            self.__data = self.__wrapper.build(self.__data)
        self.__loaded=True
        return self.__data

    @property
    def loaded(self)->bool:
        """是否已经读取过数据"""
        return self.__loaded


    @property
    def data_raw(self)->pd.DataFrame:
        if not self.loaded:
            self.load()
        return self.__data_raw

    @property
    def data(self) -> pd.DataFrame:
        if not self.loaded:
            self.load()
        return self.__data

    @property
    def stock_code(self) -> str:
        return self.__stock_code

    @property
    def benchmark_code(self) -> str:
        return self.__benchmark_code

    @property
    def start(self) -> str:
        return self.__start

    @property
    def end(self) -> str:
        return self.__end

    @property
    def online(self) -> bool:
        return self.__online

    @property
    def fq(self) -> str:
        return self.__fq

    def __fetch_stock_day(self) -> pd.DataFrame:
        """獲取本地的股票日線數據。會丟棄 `code` 列，并根據 `self.__fq` 來確定是否取復權數據。
        返回列名：_stock_columns
        """
        d = QA.QA_fetch_stock_day_adv(
            self.__stock_code, start=self.__start, end=self.__end)
        if self.__fq == 'qfq':
            df = d.to_qfq()
        elif self.__fq == 'hfq':
            df = d.to_hfq()
        else:
            df = d.data
        #原始列名['open', 'high', 'low', 'close', 'volume', 'amount', 'preclose', 'adj']
        df = df.reset_index().drop(columns=['code']).set_index('date')
        return df[self._stock_columns]

    def __fetch_index_day(self) -> pd.DataFrame:
        """獲取本地的指數日線數據。會丟棄 `code` 列。
        返回列名：_index_columns"""
        d = QA.QA_fetch_index_day_adv(
            self.__benchmark_code, start=self.__start, end=self.__end)
        df = d.data.reset_index().drop(columns=['code']).set_index('date')
        #原始列名['open', 'high', 'low', 'close', 'up_count', 'down_count', 'volume','amount'],
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

    def __fetch_stock_day_online(self, times=5) -> pd.DataFrame:
        """讀取股票在線日線數據。
            times (int, optional): Defaults to 5. 重試次數
        返回列名：_stock_columns。
        """
        retries = 0
        while True:
            try:
                d = QA.QAFetch.QATdx.QA_fetch_get_stock_day(
                    self.__stock_code, self.__start, self.__end)
                #原始列名['open', 'close', 'high', 'low', 'vol', 'amount', 'code', 'date', 'date_stamp']
                return d.rename(columns={'vol': 'volume'})[self._stock_columns]
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
                d = QA.QAFetch.QATdx.QA_fetch_get_index_day(
                    self.__benchmark_code, self.__start, self.__end)
                #原始列名['open', 'close', 'high', 'low', 'vol', 'amount', 'up_count', 'down_count', 'date', 'code', 'date_stamp']
                return d.rename(columns={'vol': 'volume'})[self._index_columns]
            except socket.timeout:
                if retries < times:
                    retries = retries + 1
                    continue
                raise


class DataHelper(object):
    @staticmethod
    def train_test_split(df,
                         col='close',
                         train_size=0.85,
                         window=10,
                         days=3,
                         start=None,
                         end=None):
        """拆分训练集和测试集。
        
        Args:
            df (pd.DataFrame): 数据源
            col (str): Defaults to close. 结果集中取的列名
            train_size (float, optional): Defaults to 0.85. 训练集比率。应该在 0.0 ~ 1.0 之间。
            window (int, optional): Defaults to 10. 窗口期日期长度。拆分后的训练集/测试集中每一个单一项会包含多少个日期的数据。
            days (int, optional): Defaults to 3. 结果日期长度。拆分后的结果集中每一个单一项会包含多少个日期的数据。
            start (int, optional): Defaults to None. 从 `self.data` 中取值的开始下标。
            end (int, optional): Defaults to None. 从 `self.data` 中取值的结束下标。

        Returns:
            [[pd.DataFrame],[pd.Series],[pd.DataFrame],[pd.Series]]: 训练集X,训练集Y,测试集X,测试集Y
        """
        if train_size is None:
            train_size = 0.85

        if df.empty:
            raise ValueError('df is empty')
        if not col:
            col = 'close'
        if col not in df.columns:
            raise ValueError('{0} not in df.columns'.format(col))

        if not start and not end:
            #开始/结束的下标均为空，表示取所有
            df_tmp = df.copy()
        elif not start:
            df_tmp = df.copy().iloc[start:]
        elif not end:
            df_tmp = df.copy().iloc[:end]
        else:
            df_tmp = df.copy().iloc[start:end]
        X = []
        Y = []
        X_columns = [c for c in df_tmp.columns if c != col]
        for i in range(df_tmp.shape[0]):
            if i + window + days > df_tmp.shape[0]:
                break
            X.append(df_tmp.iloc[i:i + window][X_columns])
            Y.append(df_tmp.iloc[i + window:i + window + days][col])

        train_end_index = round(train_size * len(X))  #训练集结束的位置
        return (X[:train_end_index], Y[:train_end_index], X[train_end_index:],
                Y[train_end_index:])
