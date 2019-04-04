# LSTM-for-Stock

---

## 安装说明

### For Python3.5

> 如果是 TensorFlow为仅支持 CPU 的版本的话，从第四步开始。
>
> TensorFlow GPU 版本的支持说明：[https://www.tensorflow.org/install/gpu]()

1. 安装 CUDA

    [https://developer.nvidia.com/cuda-toolkit]()。可以通过这里 [https://developer.nvidia.com/cuda-toolkit-archive]() 选择需要的版本。我选的是 `9.0`。

2. 安装 cuDNN

    [https://developer.nvidia.com/rdp/cudnn-archive]()。根据第一步CUDA的版本选择对应的版本。之前选择了 `9.0` 所以这里选择 `Download cuDNN XXXX for CUDA 9.0`
    > 对于 `Windows2012R2` 版本来说选择 `cuDNN Library for Windows 7` 即可。
    
> 以上步骤安装完成后，对于Windows系统来说，还需要设置系统变量。参考 [https://www.tensorflow.org/install/gpu#windows_setup]()

3. 安装  Microsoft Visual C++ 2015 Redistributable 更新 3 （Windows下安装TensorFlow的前置条件）[https://www.tensorflow.org/install/pip]()

   1.  转到 Visual Studio 下载页面，
   2.  选择“可再发行组件和生成工具”，
   3.  下载并安装 Microsoft Visual C++ 2015 Redistributable 更新 3。

4. 安装 Anaconda

    > 清华大学Conda镜像站 [https://mirror.tuna.tsinghua.edu.cn/help/anaconda/]()
    > 
    > conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    > 
    > conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    > 
    > conda config --set show_channel_urls yes

4. 创建 conda 工作区

    `conda create -n finance35 python=3.5`

5. 激活工作区

    `conda activate finance35`

6. 安装 TensorFlow

    * CPU版本 ~~`pip install --upgrade tensorflow`~~

    * GPU版本 ~~`pip install --upgrade tensorflow-gpu`~~

    * CPU版本 `pip install [--upgrade tensorflow](https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.12.0-cp35-cp35m-win_amd64.whl)`

    * GPU版本 `pip install [--upgrade tensorflow-gpu](https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.12.0-cp35-cp35m-win_amd64.whl)`

    > **这里注意看 [https://www.tensorflow.org/install/pip]() 页面的最下方。对于不同的Python版本会有不同的包。**

7. 安装 Pandas

    `pip --proxy=127.0.0.1:1080 install pandas`

8. 安装 QUANTAXIS

    ~~`pip install quantaxis`~~ 改为使用 `pip install git+https://github.com/GuQiangJS/QUANTAXIS.git --upgrade` 或者下载代码至本地安装

    ```
    $ git clone https://github.com/GuQiangJS/QUANTAXIS.git
    $ cd QUANTAXIS
    $ pip install
    ```

    > 从我个人Fork的分支安装只是解决了我个人遇到的不支持Python3.5以上版本的情况，不代表所有代码均支持。

    > 坑爹的代码 (**`async`**) 不支持 Python3.5
    >
    > [PEP 530 -- Asynchronous Comprehensions](https://www.python.org/dev/peps/pep-0530/)
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

9. 安装 pyecharts_snapshot

    `pip install pyecharts_snapshot`
---

其他可能用到的包：

* `conda install -c quantopian zipline`
* `conda install -c conda-forge cvxopt`
* `conda install -c quantopian ta-lib`
