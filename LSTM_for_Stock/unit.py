from inspect import signature

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def __get_param(cls, con_name):
    sig = signature(cls)
    return sig.parameters.get(con_name)


def get_param_default_value(cls, con_name):
    param = __get_param(cls, con_name)
    if param:
        return param.default
    return None


class PlotHelper(object):
    """繪圖幫助類"""
    sns.set()
    matplotlib.rcParams['font.family'] = 'sans-serif',
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    @staticmethod
    def plot_history(history, *args, **kwargs):  # pylint: disable=W0613
        """绘制训练 & 验证的准确率值

        Args:
            fig_size (float, float):
            window (int): 窗口期。
            benchmarkcode (str): 指数代码
            stockcode (str): 股票代码
            days (int): 计算期
            history:
            save_path (str): 保存路徑。如果不為空則自動保存。
            show (bool): 是否打印圖片。默認為 True。

        Returns:
            ::py:class:matplotlib.pyplot

        """

        subtitle = ''

        window = kwargs.pop('window', None)
        fig_size = kwargs.pop('fig_size', None)
        benchmarkcode = kwargs.pop('benchmarkcode', None)
        days = kwargs.pop('days', None)
        stockcode = kwargs.pop('stockcode', None)
        save_path=kwargs.pop('save_path',None)
        show=kwargs.pop('show',True)

        if stockcode:
            subtitle = subtitle + '股票:{} '.format(stockcode)
        if benchmarkcode:
            subtitle = subtitle + '指數:{} '.format(benchmarkcode)
        if window:
            subtitle = subtitle + '窗口期:{} '.format(window)
        if days:
            subtitle = subtitle + '計算期:{} '.format(days)

        if fig_size:
            plt.figure(figsize=fig_size)

        legend = []
        if 'acc' in history.history:
            plt.plot(history.history['acc'])
            legend.append('訓練集')
        if 'val_acc' in history.history:
            plt.plot(history.history['val_acc'])
            legend.append('測試集')
        plt.title('模型準確性')
        if subtitle:
            plt.suptitle(subtitle)
        plt.ylabel('準確性')
        plt.xlabel('代數/輪次')
        plt.legend(legend, loc='upper left')
        if show:
            plt.show()

        legend = []
        # 绘制训练 & 验证的损失值
        if 'loss' in history.history:
            plt.plot(history.history['loss'])
            legend.append('訓練集')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'])
            legend.append('測試集')
        plt.title('模型損失值')
        if subtitle:
            plt.suptitle(subtitle)
        plt.ylabel('損失值')
        plt.xlabel('代數/輪次')
        plt.legend(legend, loc='upper left')
        if save_path:
            plt.savefig(save_path, format="svg")
        if show:
            plt.show()
        return plt
