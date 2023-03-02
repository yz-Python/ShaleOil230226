此处存放模型，除了主要模型以外部分都放到了not_use_for_now里
Seq2Seq开头文件存放模型的组件，包括Encoder Decoder Attention等结构
Full_model开头文件为完整模型，也就是输入输出实际进行调用的
------------------------------------
only_lstm为单lstm结构，只能用于预测一天产量
Seq2Seq_model为最初版本的结构，Decoder没有输入要预测天的产量以外的动态信息（油嘴、油压）
目前这模型部分主要使用带有changeDe的文件，表示Decoder部分输入动态信息
------------------------------------
origin表示注意力机制使用相关性注意力
qkv表示使用cross-Attention
qkv_v1表示使用cross-Attention，但是Decoder部分堆叠了两层LSTM，是对第一层LSTM的输出结果计算注意力机制
visualsentinel为视觉哨兵注意力
这些均是尝试使用过的注意力机制，
目前效果最佳为qkv_v1
---------------------------
