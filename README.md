# LSTM-for-Stock

---

## 安装说明

### For Python3.5

> 如果是 TensorFlow为仅支持 CPU 的版本的话，从第四步开始。
>
> TensorFlow GPU 版本的支持说明：<https://www.tensorflow.org/install/gpu>

1. 安装 CUDA 及相关Patch

    <https://developer.nvidia.com/cuda-toolkit>。可以通过这里 <https://developer.nvidia.com/cuda-toolkit-archive> 选择需要的版本。我选的是 `9.0`。

2. 安装 cuDNN

    1. <https://developer.nvidia.com/rdp/cudnn-archive>。根据第一步CUDA的版本选择对应的版本。之前选择了 `9.0` 所以这里选择 `Download cuDNN XXXX for CUDA 9.0`

    > 对于 `Windows2012R2` 版本来说选择 `cuDNN Library for Windows 7` 即可。

    2. 将压缩包中的内容复制至`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64`中。
> 以上步骤安装完成后，对于Windows系统来说，还需要设置系统变量。参考 <https://www.tensorflow.org/install/gpu#windows_setup>

3. 安装  Microsoft Visual C++ 2015 Redistributable 更新 3 （Windows下安装TensorFlow的前置条件）<https://www.tensorflow.org/install/pip>

   1.  转到 Visual Studio 下载页面，
   2.  选择“可再发行组件和生成工具”，
   3.  下载并安装 Microsoft Visual C++ 2015 Redistributable 更新 3。

4. 安装 Anaconda

    > 中科大Conda镜像站 <https://mirrors.ustc.edu.cn/anaconda/>
    >
    > conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
    >
    > conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
    >
    > conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
    >
    > conda config --set show_channel_urls yes
    >
    > ---
    >
    > 清华大学Conda镜像站 https://mirror.tuna.tsinghua.edu.cn/help/anaconda/
    >
    > conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    >
    > conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    >
    > conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
    >
    > conda config --set show_channel_urls yes

5. 创建 conda 工作区

    `conda create -n finance35 python=3.5`

6. 激活工作区

    `conda activate finance35`

7. 安装 TensorFlow

    * CPU版本 ~~`pip install --upgrade tensorflow`~~

    * GPU版本 ~~`pip install --upgrade tensorflow-gpu`~~

    * CPU版本 `pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.12.0-cp35-cp35m-win_amd64.whl)`

    * GPU版本 `pip install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.12.0-cp35-cp35m-win_amd64.whl`

    > **这里注意看 <https://www.tensorflow.org/install/pip> 页面的最下方。对于不同的Python版本会有不同的包。**
    >
    >  
    >
    > 验证是否安装成功：
    >
    > `python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"`

8. 安装 CNTK（微软的深度学习包）

    * CPU版本`pip install cntk`
    * GPU版本`pip install cntk-gpu`

    > ```python
    > import cntk
    > cntk.__version__
    > ```

    > keras 中强制使用CNTK
    >
    > ```
    > import os
    > os.environ['KERAS_BACKEND']='cntk'
    > ```

9. 安装 Pandas

    `pip --proxy=127.0.0.1:1080 install pandas`

10. 安装 QUANTAXIS

    ~~`pip install quantaxis`~~ 改为使用 `pip install git+https://github.com/GuQiangJS/QUANTAXIS.git --upgrade` 或者下载代码至本地安装

    ```
    $ git clone https://github.com/GuQiangJS/QUANTAXIS.git
    $ cd QUANTAXIS
    $ pip install
    ```

    > 从我个人Fork的分支安装只是解决了我个人遇到的不支持Python3.5以上版本的情况，不代表所有代码均支持。

    > 坑爹的代码 (**`async`**) 不支持 Python3.5
    >
    > <PEP 530 -- Asynchronous Comprehensions>
    >
    > ```python
    > try:
    >     res = pd.DataFrame([item async for item in cursor])
    > except SyntaxError:
    >     print('THIS PYTHON VERSION NOT SUPPORT "async for" function')
    >     pass
    > ```

    ---

    > git代理设置方法解决
    >
    > `git config --global http.proxy http://127.0.0.1:1080`
    >
    > `git config --global https.proxy https://127.0.0.1:1080`
    >
    > `git config --global --unset http.proxy`
    >
    > `git config --global --unset https.proxy`

11. 重新安装 pytdx

      ```
      pip uninstall pytdx
      pip install pytdx
      ```

12. 安装 pyecharts_snapshot

      `pip install pyecharts_snapshot`

13. 安装 talib

      `conda install -c quantopian ta-lib`

14. 安装 nb_conda_kernels 并设置对应关系（用于jupyter中可选Python运行环境）

      `conda install ipykernel`， `python -m ipykernel install --user --name finance35 --display-name finance_py_35`

15. 安装 python.app

      `conda install -c conda-forge python.app`

      > 对于在Mac下运行来说，需要安装这个。**并且以 pythonw 方式运行**。否则会出现以下错误： ImportError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information. <https://matplotlib.org/faq/osx_framework.html#conda>

---

其他可能用到的包：

* 谷歌风格文档插件 [sphinxcontrib-napoleon](https://github.com/sphinx-contrib/napoleon)

  `pip install sphinxcontrib-napoleon`

* `conda install -c quantopian zipline`
* `conda install -c conda-forge cvxopt`

* jupyter 插件 <https://github.com/ipython-contrib/jupyter_contrib_nbextensions>

```
conda install -c conda-forge jupyter_contrib_nbextensions
# Install nbextension files, and edits nbconvert config files
jupyter contrib nbextension install --user
# Install yapf for code prettify
pip install yapf
# Install autopep8
pip install autopep8
# Jupyter extensions configurator 
pip install jupyter_nbextensions_configurator
# Enable nbextensions_configurator
jupyter nbextensions_configurator enable --user
```

[Jupyter Notebook 小贴士](http://blog.leanote.com/post/carlking5019/Jupyter-Notebook-Tips)

> 如果选择了 `autopep8` ，还需要安装 `pip install autopep8`

查看插件是否启动 `http://localhost:8888/nbextensions`

* pylint <https://www.pylint.org/#install>

[如何使用Pylint 来规范Python 代码风格 - IBM](https://www.ibm.com/developerworks/cn/linux/l-cn-pylint/index.html)

`pip install pylint`