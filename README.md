## 文件说明

- 具体流程写在了**基本流程.md**之中

- input/csv主要存放CSV文件，是读取的数据

> 目前主要使用的是[A1,A26,B124,B590_1,B590_2,C1]进行训练，来预测[B1]

* model文件夹下是模型文件，full_model是被调用的完整模型

- result/figure文件夹下存放输出的图像

- result/logs存放logs记录

- result/model存放训练好的模型

- tool存放损失函数等

- autorun.sh是测试时候写的脚本文件

## 参数说明
主要对Parser的参数进行说明，文件里也存有注释
[full_system.py]内 -ttotrain = 1表示训练，ttotrain = 0 表示测试

模型结构部分：

- --use_attention 使用Encoder与Decoder之间使用注意力机制
- --use_static_embedding 使用静态特征嵌入来表示初始h0、c0

> 以上两个参数设置均以1表示使用，0表示不使用

动静态特征融合部分：

- --only_dynamic 只用动态特征
- --only_static_concat_dynamic" 静态拼接动态降维
- --only_static_plus_dynamic 静态和动态直接相加

> 三个参数全设置为0就代表使用动静态特征融合模块，代码内存在检测保证三个参数只有一个为1或者全为0

运行脚本说明
------------
run_seq_train.sh

run_seq_test.sh


分别为预设了训练、测试参数的脚本


其他说明
--------
如果要对产量进行预测的话：搭建的CSV需要有油嘴和油压数据，以及前对应天数的产量信息之后预测的数据都需要油嘴和油压数据。现在算的误差都是用去归一化后数据计算[MRE平均相对误差和R^2相关系数]

Only_LSTM 使用单LSTM去预测，直接将最后的输出隐藏层h送入线性层，调整为Predict_Day
Seq2Seq 可以任意调整输入输出时间

更换尝试了多种损失函数，发现损失函数为RMSLE，效果最佳，能够提升精度
