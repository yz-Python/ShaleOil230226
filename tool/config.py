import os
import time
class Config:
    # 数据参数
    feature_columns = list([0,1,3])    # 要作为feature的列，按原数据从0开始计算，也可以用list 如 [2,4,6,8] 设置
    label_columns = [3]                  # 要预测的列，按原数据从0开始计算, 如：将来可以同时预测第日产油量和日产气量
    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)  
    static_nature_column = list(range(0,4))  ## 区分静态特征的两类 前四个属于地质参数和流体物性参数	后面的属于工艺参数			
    dynamic_length = len(feature_columns)-len(label_columns)
    static_human_column = list(range(4,13))

    input_size = len(feature_columns)
    output_size = len(label_columns)

    naturesize=len(static_nature_column)
    humansize=len(static_human_column)

    name = 'name'
    roll_predict_day = 145       # 迭代预测未来天数
    # 网络参数

    
    embedding_size = 256       #送入LSTM前升维
    hidden_size = 256        # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2             # LSTM的堆叠层数
    dropout_rate = 0.2          # dropout概率
    time_step = 0           # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数
    predict_day = 0            # 预测未来几天
    # 训练参数
    data_selected = []
    do_train = True
    do_predict = True
    do_predict_roll = True
    add_train = True          # 是否载入已有模型参数进行增量训练
    shuffle_train_data = False  # 是否对训练数据做shuffle #如果把上一batch的hc传入下一batch，shuffle必须为False   
    use_cuda = True            #  是否使用GPU训练 
    cudadevice = 'cuda:1'      #如果不使用gpu就改成 cudadevice='cpu',不改也行，模型在gpu没法用的情况下自动使用cpu
    traindata = 'traindata'
    train_data_rate = 0.67      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.15      # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 0   
    learning_rate = 0.00005
    epoch = 200                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5           # 训练多少epoch，验证集没提升就停掉
    random_seed = 42           # 随机种子，保证可复现

    # 框架参数
    used_frame = "pytorch"  # 选择的深度学习框架，不同的框架模型保存后缀不一样
    model_postfix = {"pytorch": ".pth"}
    model_name = model_postfix[used_frame]

    # 路径参数
    static_data_path ='input/csv/static/All_static.csv'
    dynamic_data_root = 'input/csv/dynamic'
    model_save_path = "result/model/"
    figure_save_path = "result/figure/"
    log_save_path = "result/logs/"
    do_log_print_to_screen = True
    do_log_save_to_file = True                  # 是否将config和训练过程记录到log
    do_figure_save = True
    do_train_visualized = False          # 训练loss可视化，pytorch用visdom,留着下次使用
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)
