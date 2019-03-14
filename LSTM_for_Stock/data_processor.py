import datetime

import numpy as np
from QUANTAXIS import QA_fetch_index_day_adv as Fetch_index_adv
from QUANTAXIS import QA_fetch_stock_day_adv as Fetch_stock_adv
from sklearn.preprocessing import Normalizer


class DataLoader(object):
    """數據讀取器

    Args:
        stock_code (str): 股票代碼
        benchmark_code (str): 指數代碼
        split (float): 訓練集+驗證集/測試集的拆分比率。（0<x<1）。默認為0.1。
                       Example: 0.1 意味著保留10%的數據作為驗證集
        columns (list): 獲取數據時保留的列名。**會取第一列為結果集**。
        fillna (str): 填充Nan值所用的method。參考 `pandas.DataFrame.fillna`_。
                      如果不需要填充則傳入None。
        fq (str): 是否採用復權數據。默認為 前復權。如果不需要復權則傳入 `None`。
        dropna : 是否在填充Nan值之後丟棄剩餘的Nan值。
        start (str): 數據開始日期。數據格式(`%Y-%m-%d`)。默認值 `1990-01-01`。
        end (str): 數據結束日期。數據格式(`%Y-%m-%d`)。默認值 `當天`。

    .. _pandas.DataFrame.fillna:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
    """

    def __init__(self, stock_code,
                 benchmark_code,
                 split=0.1,
                 columns=['close', 'open', 'high', 'low', 'volume'],
                 fillna='ffill',
                 dropna=True,
                 fq='qfq',
                 start='1990-01-01',
                 end=datetime.datetime.today().strftime('%Y-%m-%d')):
        self._stock_df = self._fetch_stock_adv(stock_code, start, end, fq=fq)
        self._benchmark_df = self._fetch_index_adv(benchmark_code, start, end)
        self.stock_code = stock_code
        self.benchmark_code = benchmark_code
        # 完整數據
        self.data = self._stock_df[columns].join(self._benchmark_df[columns],
                                                 rsuffix='_benchmark')
        if fillna:
            self.data.fillna(method=fillna, inplace=True)
        if dropna:
            self.data.dropna(inplace=True, subset=[self.data.columns[0]])
        len_split = int(len(self.data) * (1 - split))
        # # 丟棄了測試用列的數據源
        # d = self.data.drop(columns=[val_column_name])
        # 訓練集+测试集可用數據
        self._np_train = self.data.values[:len_split]
        self.len_train = len(self._np_train)  # 訓練集+测试集大小
        # 保留的验证集可用數據
        self._np_valid = self.data.values[len_split:]
        self.len_valid = len(self._np_valid)  # 保留的验证集大小

    def get_valid_data(self, window_size, test_size, normalize):
        """按照給定的窗口大小獲取验证集的X,Y。
        X的大小為窗口大小，Y為窗口期最後一項。
        Args:
            test_size: 測試數據大小
            normalize: 是否執行正則化
            window_size: 窗口大小

        Returns:
            np.array,np.array: x:三維數組，第一維尺寸是 總拆分的數量；
                                           第二維尺寸是 window_size；
                                           第三維尺寸是 總共的測試特性數據量；
                               y:二維數組，第一維尺寸與 x 一致。
                                           第二維尺寸是 test_size。
        """
        data_x = []
        data_y = []
        for i in range(self.len_valid - window_size - test_size):
            x, y = self._next_window(self._np_valid,
                                     i,
                                     window_size,
                                     test_size,
                                     normalize)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_train_data(self, window_size, test_size, normalize):
        """按照給定的窗口大小獲取訓練集+测试集的X,Y。
        X的大小為窗口大小，Y為窗口期最後一項。
        Args:
            test_size: 測試數據大小
            normalize: 是否執行正則化
            window_size: 窗口大小

        Returns:
            np.array,np.array: x:三維數組，第一維尺寸是 總拆分的數量；
                                           第二維尺寸是 window_size；
                                           第三維尺寸是 總共的測試特性數據量；
                               y:二維數組，第一維尺寸與 x 一致。
                                           第二維尺寸是 test_size。
        """
        data_x = []
        data_y = []
        for i in range(self.len_train - window_size - test_size):
            x, y = self._next_window(self._np_train,
                                     i,
                                     window_size,
                                     test_size,
                                     normalize)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def _next_window(self, arr, start, window_size, test_size, normalize):
        """按照指定起始位置和長度構建X,Y
        Args:
            arr (np.array): 待拆分的数据集
            test_size (int): 結果列長度
            normalize (bool): 是否執行規則化
            start (int): 起始位置
            window_size (int): 窗口長度
        Return:
            np.array,np.array: x:二維數組，第一維是 window_size 大小；
                                           第二維是 總共的測試特性數據量；
                               y:一維數組，test_size 大小。
                               **為 i+window_size開始往後的 test_size 的值。**
        """

        '''
        y取值時會去 self.data 中取值，因為如果 start+window_size+test_size
        的值超過了 _np_train 的取值範圍時會報錯。
        但是當 start+window_size+test_size 的值過大，超過了 data 的尺寸時，
        依然會報錯。這裡就涉及到了構造函數傳入的 split 值，如果過小或者
        數據集本身就很小而window_size或者test_size過大就可能造成此問題。
       '''
        # 首先從i開始取出window_size+test_size長度的數據
        window = np.copy(arr[start:start + window_size + test_size])
        if normalize:
            # 分別對每一列做標準化
            for j in range(window.shape[1]):
                window[:, j] = Normalizer().fit_transform([window[:, j]])[0]
        # 取數據的前一部分為x
        x = window[:-test_size]
        # 取數據的最後 test_size 部分的第一列為y。
        # 第一列被標記在構造函數的columns參數中。
        y = window[-test_size:, [0]][:, 0]
        return x, y

    @property
    def start(self):
        """獲取當前數據的開始日期。如果沒有數據則返回None。"""
        return self.data.index[0] if not self.data.empty else None

    @property
    def end(self):
        """獲取當前數據的結束日期。如果沒有數據則返回None。"""
        return self.data.index[-1] if not self.data.empty else None

    def _fetch_stock_adv(self, code, start, end, fq, drop_code=True):
        """讀取股票日線。
        讀取時會使用 `QA_DataStruct_Stock_day` 類型 填充 `self._stock_data`

        Args:
            code: 代碼
            start: 開始日期。%Y-%m-%d
            end: 結束日期。%Y-%m-%d
            fq: 是否採用復權數據。默認為 前復權。如果不需要復權則傳入 `None`。
            drop_code: 是否丟棄index中的code列

        Returns:
            pd.DataFrame: 日線數據表
        """
        # QA_DataStruct_Stock_day類型
        self._stock_data = Fetch_stock_adv(code, start, end)
        if fq == 'qfq':
            dataframe = self._stock_data.to_qfq()
        elif fq == 'hfq':
            dataframe = self._stock_data.to_hfq()
        else:
            dataframe = self._stock_data.data

        if drop_code:
            dataframe = dataframe.reset_index()
            return dataframe.drop(columns=['code']).set_index('date')
        else:
            return dataframe

    def _fetch_index_adv(self, code, start, end, drop_code=True):
        """讀取指數日線
        讀取時會使用 `QA_DataStruct_Stock_day` 類型 填充 `self._benchmark_data`

        Args:
            code: 代碼
            start: 開始日期。%Y-%m-%d
            end: 結束日期。%Y-%m-%d
            drop_code: 是否丟棄index中的code列

        Returns:
            pd.DataFrame: 日線數據表
        """
        # QA_DataStruct_Stock_day類型
        self._benchmark_data = Fetch_index_adv(code, start, end)
        if drop_code:
            return self._benchmark_data.data.reset_index() \
                .drop(columns=['code']) \
                .set_index('date')
        else:
            return self._benchmark_data.data
