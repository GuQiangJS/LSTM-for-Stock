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
    sns.set()
    matplotlib.rcParams['font.family'] = 'sans-serif',
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    def plot_history(history):
        """绘制训练 & 验证的准确率值

        Args:
            history:

        Returns:

        """
        legend = []
        if 'acc' in history.history:
            plt.plot(history.history['acc'])
            legend.append('訓練集')
        if 'val_acc' in history.history:
            plt.plot(history.history['val_acc'])
            legend.append('測試集')
        plt.title('模型準確性')
        plt.ylabel('準確性')
        plt.xlabel('代數/輪次')
        plt.legend(legend, loc='upper left')
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
        plt.ylabel('損失值')
        plt.xlabel('代數/輪次')
        plt.legend(legend, loc='upper left')
        plt.show()
