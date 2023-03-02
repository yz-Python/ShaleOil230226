import os
import time
import pandas as pd
import numpy as np


class Config:  # 设定一些基本不会经常变化的参数
    # 数据参数
    feature_columns = list([0, 1, 3])  # 要作为feature的列，按原数据从0开始计算，也可以用list 如 [2,4,6,8] 设置
    label_columns = [3]  # 要预测的列，按原数据从0开始计算, 如：将来可以同时预测第日产油量和日产气量
    label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(feature_columns, label_columns)
    static_nature_column = list(range(0, 4))  ## 区分静态特征的两类 前四个属于地质参数和流体物性参数	后面的属于工艺参数
    dynamic_length = len(feature_columns) - len(label_columns)
    static_human_column = list(range(4, 13))

    input_size = len(feature_columns)
    output_size = len(label_columns)

    naturesize = len(static_nature_column)
    humansize = len(static_human_column)

    name = 'name'
    roll_predict_day = 145  # 迭代预测未来天数
    # 网络参数

    save_data_cache = True  # 是否保存数据采样的结果
    embedding_size = 256  # 送入LSTM前升维
    hidden_size = 256  # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2  # LSTM的堆叠层数
    dropout_rate = 0.2  # dropout概率
    time_step = 0  # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数
    predict_day = 0  # 预测未来几天
    # 训练参数
    data_selected = []
    do_train = True
    do_predict = True
    do_predict_roll = True
    add_train = True  # 是否载入已有模型参数进行增量训练
    shuffle_train_data = False  # 是否对训练数据做shuffle #如果把上一batch的hc传入下一batch，shuffle必须为False   
    use_cuda = True  # 是否使用GPU训练
    cudadevice = 'cuda:1'  # 如果不使用gpu就改成 cudadevice='cpu',不改也行，模型在gpu没法用的情况下自动使用cpu
    traindata = 'traindata'
    train_data_rate = 0.67  # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.15  # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 0
    learning_rate = 0.00005
    epoch = 200  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5  # 训练多少epoch，验证集没提升就停掉
    random_seed = 42  # 随机种子，保证可复现

    # 框架参数
    used_frame = "pytorch"  # 选择的深度学习框架，不同的框架模型保存后缀不一样
    model_postfix = {"pytorch": ".pth"}
    model_name = model_postfix[used_frame]

    # 路径参数
    data_cache_path = 'cache'
    static_data_path = 'input/csv/static/All_static.csv'
    dynamic_data_root = 'input/csv/dynamic'
    model_save_path = "result/model/"
    figure_save_path = "result/figure/"
    log_save_path = "result/logs/"
    do_log_print_to_screen = True
    do_log_save_to_file = True  # 是否将config和训练过程记录到log
    do_figure_save = True
    do_train_visualized = False  # 训练loss可视化，pytorch用visdom,留着下次使用
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)  # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)


# ------------- 数据处理部分 ------------------------#
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, config):
        self.config = config
        self.full_static_data_path = self.normalize(config.static_data_path)  # 归一化静态数据，并获得静态数据地址用于自动选择

        self.data, self.staticdata_human, self.staticdata_nature, self.data_column_name = self.read_data()

        self.norm_human_data = self.staticdata_human
        self.norm_nature_data = self.staticdata_nature
        # self.data=self.data.astype('float')
        self.train_num = []
        # self.test_start_num = int(self.data_num * self.config.train_data_rate)
        self.norm_data = []

        self.std = []
        self.mean = []
        for i, data in enumerate(self.data):
            self.data_num = data.shape[0]
            train_num = int(self.data_num * self.config.train_data_rate)

            mean = np.mean(data, axis=0)  # .reshape(self.data_num,1)
            # 数据的均值和方差 axis =0在列上归一，axis=1在行上归一
            std = np.std(data, axis=0)  # .reshape(self.data_num,1)

            norm_data = (data - mean) / std  # 归一化，去量纲
            self.norm_data.append(norm_data)
            self.train_num.append(train_num)
            self.mean.append(mean)
            self.std.append(std)

        # self.norm_nature_data.append(self.staticdata_nature[i].reshape(1,config.naturesize))

        # self.norm_human_data.append(self.staticdata_human[i].reshape(1,config.humansize))

    def normalize(self, static_path):
        Data = pd.read_csv(static_path)
        cate = Data.columns.tolist()
        Data = Data.values
        feature = np.array(Data[:, :-1], dtype=float)
        number = Data[:, -1:]
        mean = np.mean(feature, axis=0)
        std = np.std(feature, axis=0)
        norm = (feature - mean) / std

        norm_data = np.concatenate([norm, number], axis=1)
        # print(norm_data)
        df1 = pd.DataFrame(data=norm_data,
                           columns=cate)
        norm_static_file = static_path.replace('.csv', '_norm.csv')
        df1.to_csv(norm_static_file, index=False)
        return norm_static_file

    def read_data(self):  # 读取初始数据
        init_data = []
        data_column_name = []
        all_data_name = []
        for name in os.listdir(self.config.dynamic_data_root):
            if ".csv" in name:
                all_data_name.append(name)  # 带有csv后缀,遍历所有数据集的名称
        for name in self.config.data_selected:  # 从设定的井编号读取数据，数据形式为['All']或者['A1','C1']
            if name == 'All':  # name是自己输入决定使用的井，data_name是遍历得到的所有数据集的csv名称列表
                assert len(self.config.data_selected) == 1  # 保证输入只有All，不能All 和A1，B1一起写
                for data_name in all_data_name:  # 读取全部井训练数据
                    data = pd.read_csv(os.path.join(self.config.dynamic_data_root, data_name),
                                       usecols=self.config.feature_columns)
                    init_data.append(data.values)
                    data_column_name.append(data.columns.tolist())
            else:
                name_csv = name + ".csv"  # A1 → A1.csv
                try:
                    data = pd.read_csv(os.path.join(self.config.dynamic_data_root, name_csv),
                                       usecols=self.config.feature_columns)  # 读取name井训练数据
                    init_data.append(data.values)
                    data_column_name.append(data.columns.tolist())
                except ValueError:
                    print('选取的{}井与读取到的数据集井不匹配，检查数据集或检查data_selected参数'.format(name))
        print('动态数据加载完成')
        data_selected = self.config.data_selected
        if data_selected[0] == 'All':
            data_selected = []
            for name in all_data_name:
                data_selected.append(name.replace('.csv', ''))
        init_static_data = np.array(
            self.select_static_data(dynamic_name_list=data_selected, norm_static_file=self.full_static_data_path))
        init_static_human_data = init_static_data[:, :,
                                 self.config.static_human_column].tolist()  # self.select_static_data(dynamic_name_list=data_selected,norm_static_file=self.full_static_data_path,
        # cols=self.config.static_human_column)
        init_static_nature_data = init_static_data[:, :, self.config.static_nature_column].tolist()
        return init_data, init_static_human_data, init_static_nature_data, data_column_name
        # .columns.tolist() 是获取列名

    def select_static_data(self, dynamic_name_list, norm_static_file):  # 动态数据输入形式：['A1','B1','C1']
        static_data = []
        # print(norm_static_file)
        Data = pd.read_csv(norm_static_file)
        Data = Data.values
        feature = np.array(Data[:, :-1], dtype=float)
        Name = Data[:, -1:].tolist()
        try:
            for dynamic_name in dynamic_name_list:
                index = Name.index([dynamic_name])
                static_data.append(feature[index:index + 1, :])
            # print(static_data)
        except ValueError:
            print('{}井不匹配静态数据，检查是否确实静态数据或者井编号不对应'.format(dynamic_name))
            os.kill()
        print('静态数据加载完成')
        return (static_data)

    def get_train_and_valid_data(self):
        full_train_data, full_valid_data, full_train_label, full_valid_label = [], [], [], []
        for j in range(len(self.norm_data)):
            data_ind = self.norm_data[j]
            print(data_ind.shape)
            feature_data = data_ind[:self.train_num[j]]
            label_data = data_ind[:self.train_num[j],
                         self.config.label_in_feature_index]
            norm_nature_data = self.norm_nature_data[j]
            norm_human_data = self.norm_human_data[j]

            static_nature_data = np.array([norm_nature_data for i in range(self.config.time_step)])
            static_nature_data = static_nature_data.reshape(static_nature_data.shape[0], static_nature_data.shape[2])
            static_human_data = np.array(
                [norm_human_data for i in range(self.config.time_step)])  # 静态数据复制5遍，因为要和5天预测3天维度的encoder输入数据对应上
            static_human_data = static_human_data.reshape(static_human_data.shape[0],
                                                          static_human_data.shape[2])  # 这里实现的是静态数据添加时间信息

            embed = np.eye(self.config.time_step)  # [1 0 0]
            static_nature_data = np.append(static_nature_data, embed, axis=1)  # [0 1 0]
            static_human_data = np.append(static_human_data, embed, axis=1)  # [0 0 1]  拼接进去作为One-Hot编码 表示时间顺序

            length = self.config.time_step + self.config.predict_day  # 5+3=8
            # print(length)
            feature = [feature_data[start_index + i * (length): start_index + (i + 1) * (length)]
                       # 训练部分按照错位进行采样得到的形式,包含油嘴油压产量
                       for start_index in range(length)  # 这里是动态特征部分1-5天预测6-8,2-6天预测7-9这样一致往后推
                       for i in range((self.train_num[j] - start_index) // (length))]  # 取得数据是1-8天，2-9天这样顺序，这样堆叠下去
            label = [label_data[start_index + i * length: start_index + (i + 1) * length]  # 标签部分处理方法相同，仅有产量
                     for start_index in range(length)
                     for i in range((self.train_num[j] - start_index) // length)]
            feature, label = np.array(feature), np.array(label)
            print(feature.shape, label.shape)

            train_data, valid_data, train_label, valid_label = train_test_split(feature, label,
                                                                                test_size=self.config.valid_data_rate,
                                                                                random_state=self.config.random_seed,
                                                                                shuffle=self.config.shuffle_train_data)  # 划分训练和验证集
            s_h_train = np.array([static_human_data for k in range(len(train_data))])  # s表示静态数据，h表示人工参数
            s_n_train = np.array([static_nature_data for k in range(len(train_data))])  # ，n表示自然（地质）参数
            s_h_valid = np.array([static_human_data for k in range(len(valid_data))])
            s_n_valid = np.array([static_nature_data for k in range(len(valid_data))])
            if j == 0:
                full_train_data, full_valid_data, full_train_label, full_valid_label = train_data, valid_data, train_label, valid_label
                full_s_h_train, full_s_n_train, full_s_h_valid, full_s_n_valid = s_h_train, s_n_train, s_h_valid, s_n_valid
            else:
                full_train_data, full_valid_data = np.concatenate([full_train_data, train_data],
                                                                  axis=0), np.concatenate([full_valid_data, valid_data],
                                                                                          axis=0)
                full_train_label, full_valid_label = np.concatenate([full_train_label, train_label],
                                                                    axis=0), np.concatenate(
                    [full_valid_label, valid_label], axis=0)
                full_s_h_train = np.concatenate([full_s_h_train, s_h_train], axis=0)
                full_s_n_train = np.concatenate([full_s_n_train, s_n_train], axis=0)
                full_s_h_valid = np.concatenate([full_s_h_valid, s_h_valid], axis=0)
                full_s_n_valid = np.concatenate([full_s_n_valid, s_n_valid], axis=0)  # 这里是将多井数据拼在一起，用于一起训练

        # 此处分别对训练集和测试集进行对应输入采样
        # full_train_data是一个[xxx,8,3]的数据 full_train_label是一个[xxx，8,1]的数据
        encoder_train = full_train_data[:, :self.config.time_step, :]  # 取1-5天的动态数据和产量，油嘴油压产量
        decoder_train = full_train_label[:, self.config.time_step - 1:length - 1,
                        :]  # 取5-7天的产量，最初是为了用于teacherforcing的进行训练，后来没有用teacherforcing，在实际的验证集没有用到这部分数据
        label_train = full_train_label[:, self.config.time_step:length, :]  # 取6-8天的产量，作为标签
        # dynamic_train = full_train_data[:,self.config.time_step:length,:self.config.dynamic_length] #取5-8天的动态数据，油嘴油压

        encoder_valid = full_valid_data[:, :self.config.time_step, :]  # 验证集，同上
        decoder_valid = full_valid_label[:, self.config.time_step - 1:length - 1, :]
        label_valid = full_valid_label[:, self.config.time_step:length, :]
        # dynamic_valid = full_valid_data[:,self.config.time_step:length,:self.config.dynamic_length]
        if self.config.save_data_cache == True:
            self.save_cache(encoder_train, label_train, category='train')
            self.save_cache(encoder_valid, label_valid, category='valid')  # 保存中间数据采样结果，注意数据都是经过归一化的

        return encoder_train, decoder_train, label_train, \
               encoder_valid, decoder_valid, label_valid, \
               full_s_h_train, full_s_n_train, full_s_h_valid, full_s_n_valid

    def get_test_data(self, roll, return_label_data=False):
        norm_data = self.norm_data[0]  # 测试只考虑单井测试情况。
        train_num = self.train_num[0]
        feature_data = norm_data[train_num:]  # 测试样本train_num需要被设定为0
        time_step = self.config.time_step  # 防止time_step大于测试集数量
        step_for_no_roll = ((feature_data.shape[0] - time_step) // self.config.predict_day) + 1
        print(step_for_no_roll)
        norm_nature_data = self.norm_nature_data[0]
        norm_human_data = self.norm_human_data[0]
        static_nature_data = np.array([norm_nature_data for i in range(self.config.time_step)])
        static_nature_data = static_nature_data.reshape(static_nature_data.shape[0], static_nature_data.shape[2])
        static_human_data = np.array([norm_human_data for i in range(self.config.time_step)])  # 数据形式有点奇怪，不知道为什么，反正调一下
        static_human_data = static_human_data.reshape(static_human_data.shape[0], static_human_data.shape[2])

        embed = np.eye(self.config.time_step)  # [1 0 0]
        static_nature_data = np.append(static_nature_data, embed, axis=1)  # [0 1 0]
        static_human_data = np.append(static_human_data, embed, axis=1)  # [0 0 1]  拼接进去作为One-Hot编码

        # 在滚动测试数据中，采样方式按照Predict_day连续进行错位采样 1-5天预测6-8天，4-8天预测9-11天，与训练数据采样方式不同
        # 迭代测试数据仅采样1-5天数据，预测天数按照roll_predict_day决定
        if roll:  # roll表示迭代预测
            encoder_test = [feature_data[: time_step]]
            decoder_test = [feature_data[time_step:]]
            # decoder_test= np.array(decoder_test)
            # decoder_test = decoder_test[:,:,:self.config.dynamic_length]
            return np.array(encoder_test), static_human_data, static_nature_data

        if not roll:  # 滚动预测
            new_encoder_test = []
            if self.config.time_step >= self.config.predict_day:  # 采样方式按照Predict_day连续进行错位采样 1-5天预测6-8天，4-8天预测9-11天，与训练数据采样方式不同
                encoder_test = [feature_data[i * self.config.predict_day:  time_step + i * self.config.predict_day]  #
                                for i in range(step_for_no_roll)]  # 按照1-5天，4-8天。。。这样的顺序采样encoder输入样本

            if self.config.time_step < self.config.predict_day:  # 基本用不上这种情况，不会出现预测天数大于输入天数的滚动预测
                middle_day = self.config.predict_day
                encoder_test = [feature_data[i * middle_day:  time_step + i * middle_day]
                                for i in range(step_for_no_roll)]  #

            for i in encoder_test:
                if len(i) == self.config.time_step:
                    new_encoder_test.append(i)  # 去除编码器输入数据不够的样本

        #     new_decoder_test =[]
        #     decoder_test = [feature_data[  time_step+i*self.config.predict_day :  time_step+(i+1)*self.config.predict_day]
        #         for i in range(step_for_no_roll)] #按照6-8天，9-11天。。。这样的顺序采样decoder输入样本

        #     for i in decoder_test :
        #         if len(i) ==self.config.predict_day:
        #             new_decoder_test.append(i)

        # new_decoder_test =np.array(new_decoder_test)
        # new_decoder_test = new_decoder_test[:,:,:self.config.dynamic_length] #decoder只保留油嘴油压数据
        encoder_test = np.array(new_encoder_test)
        if self.config.save_data_cache == True:
            self.save_cache(encoder_test, None, category='test')
        # decoder_test =new_decoder_test

        # final_len =min(len(encoder_test),len(decoder_test))   
        # encoder_test = encoder_test[:final_len,:,:]   #保证测试样本数目对应
        # decoder_test = decoder_test[:final_len,:,:]

        return encoder_test, static_human_data, static_nature_data

    def save_cache(self, encoder, label, category):
        import shutil
        assert category in ['train', 'valid', 'test']
        cache_path = os.path.join(self.config.data_cache_path, category)  # 获得根目录
        encoder_path = os.path.join(cache_path, 'encoder_input')
        shutil.rmtree(encoder_path, ignore_errors=True)  # 清空目录
        os.makedirs(encoder_path)  # 生成目录

        for number in range(encoder.shape[0]):  # 如果只是测试集,就只需要保存encoder
            data_encoder = encoder[number]
            np.savetxt(os.path.join(encoder_path, str(number) + ".csv"), data_encoder, delimiter=',')

        if category in ['train', 'valid']:
            label_path = os.path.join(cache_path, 'label')  # 如果训练过程，还需要保存验证集
            shutil.rmtree(label_path, ignore_errors=True)
            os.makedirs(label_path)
            for number_label in range(encoder.shape[0]):
                data_label = label[number_label]
                np.savetxt(os.path.join(label_path, str(number_label) + ".csv"), data_label, delimiter=',')


# ------------- 定义基本的模型框架 -------------------#
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from xpinyin import Pinyin
from model.full_model_no_dynamic import Net
from tool.loss.RMSLEloss import RMSLEloss
import setproctitle

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tool.evaluate.mean_relative_error import mean_relative_error


def train(config, logger, train_and_valid_data):
    device = torch.device(config.cudadevice if config.use_cuda and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU

    if config.do_train_visualized:
        import visdom
        vis = visdom.Visdom(env='model_pytorch')
    encoder_train, decoder_train, label_train, encoder_valid, decoder_valid, label_valid, \
    full_s_h_train, full_s_n_train, full_s_h_valid, full_s_n_valid = train_and_valid_data
    # decoder_train是在teacherforcing下送入Decoder的t-1时刻产量，dynamic_train是t时刻影响产量的动态数据
    full_s_h_train, full_s_n_train = torch.from_numpy(full_s_h_train).float(), torch.from_numpy(full_s_n_train).float()
    full_s_h_valid, full_s_n_valid = torch.from_numpy(full_s_h_valid).float(), torch.from_numpy(full_s_n_valid).float()
    encoder_train, decoder_train, label_train = torch.from_numpy(encoder_train).float(), torch.from_numpy(
        decoder_train).float(), torch.from_numpy(label_train).float()  # 先转为Tensor
    print(encoder_train.shape, decoder_train.shape, label_train.shape)
    train_loader = DataLoader(TensorDataset(encoder_train, decoder_train, label_train, full_s_h_train, full_s_n_train),
                              batch_size=config.batch_size)  # DataLoader可自动生成可训练的batch数据

    encoder_valid, decoder_valid, label_valid = torch.from_numpy(encoder_valid).float(), torch.from_numpy(
        decoder_valid).float(), torch.from_numpy(label_valid).float()
    valid_loader = DataLoader(TensorDataset(encoder_valid, decoder_valid, label_valid, full_s_h_valid, full_s_n_valid),
                              batch_size=config.batch_size)
    print(encoder_valid.shape, decoder_valid.shape, label_valid.shape)

    model = Net(config, teacherforcing=False).to(device)  # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中
    if config.add_train:  # 如果是增量训练，会先加载原模型参数
        model.load_state_dict(torch.load(config.model_save_path + config.name + config.model_name))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = RMSLEloss()  # 这两句是定义优化器和loss
    # criterion = torch.nn.MSELoss()
    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train()  # pytorch中，训练时要转换成训练模式
        train_loss_array = []

        h_0, c_0 = None, None
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, _data in loop:
            setproctitle.setproctitle("zys:" + str(epoch) + "/" + "{}".format(config.epoch))
            _encoder_train, _decoder_train, label_train, human, nature = _data[0].to(device), _data[1].to(device), \
                                                                         _data[2].to(device), _data[3].to(device), \
                                                                         _data[4].to(device)
            optimizer.zero_grad()  # 训练前要将梯度信息置 0
            # print(_encoder_train.shape,_decoder_train.shape, label_train.shape)  #torch.Size([B, 5, 3]) torch.Size([B, 3, 1]) torch.Size([B, 3, 1])
            _decoder_train = _decoder_train.permute(1, 0, 2)
            label_train = label_train.permute(1, 0, 2)  # 成为[3,1(batch_size),1]
            # dynamic_train = dynamic_train.permute(1,0,2) #成为[3,1(batch_size),2]

            pred_Y = model(_encoder_train, _decoder_train, human, nature, h_0, c_0)

            loss = criterion(pred_Y, label_train)  # 计算loss
            loss.backward()  # 将loss反向传播
            optimizer.step()
            # 用优化器更新参数
            train_loss_array.append(loss.item())
            global_step += 1
            if config.do_train_visualized and global_step % 100 == 0:  # 每一百步显示一次
                vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
                         update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))
        # 以下为验证集和早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合

        model.eval()  # pytorch中，预测时要转换成预测模式
        model.teacherforcing = False
        valid_loss_array = []
        h_0, c_0 = None, None
        loopv = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for i, _data in loopv:
            encoder_valid, decoder_valid, label_valid, human, nature = _data[0].to(device), _data[1].to(device), _data[
                2].to(device), _data[3].to(device), _data[4].to(device)
            decoder_valid = decoder_valid.permute(1, 0, 2)
            label_valid = label_valid.permute(1, 0, 2)
            # dynamic_valid = dynamic_valid.permute(1,0,2)
            # _valid_X, _encoder_valid = _valid_X.to(device), _encoder_valid.to(device)
            pred_Y = model(encoder_valid, decoder_valid, human, nature, h_0, c_0)

            loss = criterion(pred_Y, label_valid)  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
                    "The valid loss is {:.6f}.".format(valid_loss_cur))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.name + config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:  # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                logger.info(" The training stops early in epoch {}".format(epoch))
                break


def predict(config, encoder_test, human, nature):
    # 获取测试数据
    device = torch.device(config.cudadevice if config.use_cuda and torch.cuda.is_available() else "cpu")
    encoder_test = encoder_test
    encoder_test = torch.from_numpy(encoder_test).float()
    # dynamic_test =dynamic_test
    # dynamic_test = torch.from_numpy(dynamic_test).float()
    print(encoder_test.shape)
    test_loader = DataLoader(TensorDataset(encoder_test), batch_size=1)

    human, nature = torch.from_numpy(human).float(), torch.from_numpy(nature).float()
    bathsize = config.batch_size

    human = human.repeat(bathsize, 1, 1).to(device)
    nature = nature.repeat(bathsize, 1, 1).to(device)

    # 加载模型

    model = Net(config, teacherforcing=False).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.name + config.model_name))  # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)
    predict_day = config.predict_day
    # 预测过程
    model.eval()
    h_0, c_0 = None, None
    print(encoder_test.shape)
    for _data in test_loader:
        data_X = _data[0].to(device)
        decoder_input0 = data_X[:, -1:, -1:].permute(1, 0, 2)  # 取输入特征的最后一维作为Decoder0的初始输入
        # 因为Predict所以这么设计
        predictTensor = torch.zeros(predict_day - 1, 1, 1).to(device)
        decoder_input = torch.cat((decoder_input0, predictTensor), dim=0)
        # dynamic_input = dynamic_test.permute(1,0,2)

        pred_X = model(data_X, decoder_input, human, nature, h_0, c_0)
        cur_pred = torch.squeeze(pred_X, dim=2)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()  # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据


def predict_roll(config, encoder_test, human, nature):
    # 获取测试数据
    device = torch.device(config.cudadevice if config.use_cuda and torch.cuda.is_available() else "cpu")

    encoder_test = encoder_test
    encoder_test = torch.from_numpy(encoder_test).float()
    # dynamic_test =dynamic_test
    # dynamic_test = torch.from_numpy(dynamic_test).float()
    print(encoder_test.shape)

    roll_predict_day = config.roll_predict_day

    test_loader = DataLoader(TensorDataset(encoder_test), batch_size=1)

    human, nature = torch.from_numpy(human).float(), torch.from_numpy(nature).float()
    bathsize = config.batch_size

    human = human.repeat(bathsize, 1, 1).to(device)
    nature = nature.repeat(bathsize, 1, 1).to(device)

    # 加载模型

    model = Net(config, teacherforcing=False).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.name + config.model_name))  # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()
    h_0, c_0 = None, None

    for _data in test_loader:
        data_X = _data[0].to(device)
        decoder_input0 = data_X[:, -1:, -1:].permute(1, 0, 2)
        # print(decoder_input0.shape)#
        predictTensor = torch.zeros(roll_predict_day - 1, 1, 1).to(device)
        decoder_input = torch.cat((decoder_input0, predictTensor), dim=0)
        # dynamic_input = dynamic_test.permute(1,0,2)        
        pred_X = model(data_X, decoder_input, human, nature, h_0, c_0)
        cur_pred = torch.squeeze(pred_X, dim=2)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()  # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据


# ---------------------这一部分是应用的模块 --------------------#

import logging
import sys
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt

frame = "pytorch"


def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger


def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray, name: str):
    # label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : 
    #                     origin_data.train_num + origin_data.start_num_in_test +config.roll_predict_day,
    #                                         config.label_in_feature_index
    label_len = len(predict_norm_data)
    calculate_label_data = origin_data.data[0][origin_data.train_num[0] + config.time_step:
                                               origin_data.train_num[0] + label_len + config.time_step,
                           config.label_in_feature_index]
    label_data = origin_data.data[0][origin_data.train_num[0]:, config.label_in_feature_index]
    actual_len = len(calculate_label_data)
    print(label_data.shape)

    predict_data = predict_norm_data * origin_data.std[0][config.label_in_feature_index] + \
                   origin_data.mean[0][config.label_in_feature_index]  # 通过保存的均值和方差还原数据, 这个很重要

    label_name = [origin_data.data_column_name[0][i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)
    p = Pinyin()
    label_name_pinyin = []
    for i in range(len(label_name)):
        label_name_pinyin.append(p.get_pinyin(label_name[i]))
    # label 和 predict 是错开config.predict_day天的数据的
    # 下面是两种norm后的loss的计算方式，结果是一样的，可以简单手推一下
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of {} is ".format(label_name) + str(loss_norm))
    # mean_squared_error ,mean_absolute_error ,r2_score
    mse = mean_squared_error(calculate_label_data[:], predict_data[:actual_len])
    mae = mean_absolute_error(calculate_label_data[:], predict_data[:actual_len])
    r2 = r2_score(calculate_label_data[:], predict_data[:actual_len])
    mre = mean_relative_error(calculate_label_data[:], predict_data[:actual_len])
    # loss = np.sum((calculate_label_data[:] - predict_data[:] ) ** 2)/len(predict_data)
    # loss_norm = loss#/(origin_data.std[0][config.label_in_feature_index] ** 2)
    logger.info("The mean relative error of {} is ".format(label_name_pinyin) + str(mre))
    logger.info("The mean squared error of {} is ".format(label_name_pinyin) + str(mse))
    logger.info("The mean average error of {} is ".format(label_name_pinyin) + str(mae))
    logger.info("The R^2 of {} is ".format(label_name_pinyin) + str(r2))

    # label_train = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    label_train = range(len(label_data))
    predict_X = range(len(predict_data))
    predict_X = [x + config.time_step for x in predict_X]  # 错开5天画图
    '''
    这里需要修改代码，保存预测值和标注值
    '''
    plt.cla()
    plt.clf()
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    if True:  # not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
        for i in range(label_column_num):
            plt.figure(i + 1)
            # plt.rcParams["font.sans-serif"] = ["SimHei"]
            # 预测数据绘制
            # 这里是中文的设置, 没有这个设置"日产油量无法显示"
            plt.plot(label_train, label_data[:, i], label='label')
            plt.plot(predict_X, predict_data[:, i],
                     label='predict')  # plt.title("Predict {}  with {}".format(label_name[i], config.used_frame))
            plt.title(config.train_name + "预测" + config.val_name)
            plt.legend()
            logger.info(
                "The predicted {} for the next {} day(s) is: ".format(label_name_pinyin[i], config.predict_day) +
                str(np.squeeze(predict_data[-config.predict_day:, i])))
            if config.do_figure_save:
                # print(config.figure_save_path+"add_stastic_{}predict_{}_with_{}.png".format(config.continue_flag, label_name_pinyin[i], config.used_frame))
                plt.savefig(config.figure_save_path + "{}.png".format(config.val_name + '_val_' + config.name + name))
        plt.show()
    df1 = pd.DataFrame(data=predict_data, columns=[name])
    df1.to_csv('result.csv', index=False)


def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data_gainer = Data(config)

        if config.do_train:
            encoder_train, decoder_train, label_train, encoder_valid, decoder_valid, label_valid, \
            full_s_h_train, full_s_n_train, full_s_h_valid, full_s_n_valid = data_gainer.get_train_and_valid_data()
            # 以5天预测3天为例子，输入的数据分别是：
            # 1-5天动态特征和产量，5-7天产量，6-8天产量，6-8天动态数据 这部分是训练集内容，验证集同理
            # 后面是人为静态参数和自然（地质）静态参数
            train(config, logger, [encoder_train, decoder_train, label_train, encoder_valid, decoder_valid,
                                   label_valid, full_s_h_train, full_s_n_train, full_s_h_valid, full_s_n_valid])

        if config.do_predict:
            config.batch_size = 1
            encoder_test, human, nature = data_gainer.get_test_data(roll=False, return_label_data=False)
            pred_result = predict(config, encoder_test, human, nature)  # 这里输出的是未还原的归一化预测数据
            draw(config, data_gainer, logger, pred_result, '_no_roll')
        if config.do_predict_roll:
            config.batch_size = 1
            encoder_test, human, nature = data_gainer.get_test_data(roll=True, return_label_data=False)
            pred_result = predict_roll(config, encoder_test, human, nature)  # 这里输出的是未还原的归一化预测数据
            draw(config, data_gainer, logger, pred_result, '_roll')
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__ == "__main__":
    import argparse

    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    # 如果在linux下面, 可以方便采用下面的参数输入
    # 在Windows环境下，可以单独修改参数，直接运行
    parser.add_argument("-b", "--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("-e", "--epoch", default=100, type=int, help="epochs num")
    parser.add_argument("--ttotrain", default=0, type=int, help="1代表训练,0代表测试")
    parser.add_argument("--use_attention", default=1, type=int, help="seq2seq_attention")
    parser.add_argument("--use_static_embedding", default=1, type=int, help="Inite,h0_c0")

    parser.add_argument("--only_dynamic", default=0, type=int, help="只用动态嵌入")
    parser.add_argument("--only_static_concat_dynamic", default=0, type=int, help="静态拼接动态降维")
    parser.add_argument("--only_static_plus_dynamic", default=0, type=int, help="静态和动态直接相加")

    parser.add_argument("--roll_predict_day", default=125, type=int, help="迭代预测天数")  # 相当于让Decoder迭代多少次

    parser.add_argument("--predict_day", default=3, type=int, help="Decoder预测天数")
    parser.add_argument("--time_step", default=5, type=int, help="Encoder读取天数")

    parser.add_argument("--name", default='A1', type=str, help="填写训练预设名称")  # 可以填写'All' 或者'A1,B1,B590_1,B123'这种形式
    parser.add_argument("--val_name", default='C1', type=str, help="填写测试预设名称")  # 填写'C1'这种形式，暂时只支持测试1口井
    # only_dynamic only_static_concat_dynamic only_static_plus_dynamic
    args = parser.parse_args()

    con = Config()
    for key in dir(args):  # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):  # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))  # 将属性值赋给Config
    # 建议window下，采用下面, 便于调试

    assert con.only_dynamic + con.only_static_concat_dynamic + con.only_static_plus_dynamic <= 1
    # 这几个参数做消融实验时候只能有同时存才1个或者都不存在，就代表动静态融合模块
    sign = True if con.ttotrain == 1 else False
    use_attention = '_No_Attention'
    use_static_embedding = '_hc_False'
    dynamic_fusion = '_ds_fusion'
    if con.only_dynamic == 1:
        dynamic_fusion = '_dynamic_only'
    if con.only_static_concat_dynamic == 1:
        dynamic_fusion = '_concat_only'
    if con.only_static_plus_dynamic == 1:
        dynamic_fusion = '_plus_only'
    if con.use_attention == 1:
        con.use_attention = True
        use_attention = '_Attention'
    else:
        con.use_attention = False
    if con.use_static_embedding == 1:
        con.use_static_embedding = True
        use_static_embedding = '_hc_True'
    else:
        con.use_static_embedding = False
    con.train_name = con.name
    end_sign = '_RMLSE' + str(con.batch_size)
    con.name = con.name + use_attention + use_static_embedding + end_sign + dynamic_fusion + str(con.time_step) + str(
        con.predict_day)
    # 设定名称，用于指示使用方法以及数据
    print('Train:' + str(sign))

    if sign == True:  # 表示训练
        if ',' in con.train_name:
            con.data_selected = con.train_name.split(',')
            print('已选择{}井训练'.format(con.data_selected))
        elif con.train_name == 'All':
            con.data_selected = ['All']
            print('已选择[{}]路径下所有井训练'.format(con.dynamic_data_root))
        else:
            con.data_selected = [con.train_name]  # 只输入单井情况
            print('已选择{}井训练'.format(con.data_selected))
        if not con.train_name:
            assert ('井名输入错误')

        con.train_data_rate = 0.95  # 设定数据用于训练的百分比
        con.add_train = False  # 不加载权重进行训练
        con.do_train = True
        con.do_predict = False
        con.do_predict_roll = False
    else:
        con.data_selected = [con.val_name]
        if not con.train_name:
            assert ('井名输入错误')
        con.train_data_rate = 0.1  # 取井剩下1-train_data_rate的百分比用于测试
        con.do_train = False
        con.do_predict = True
        con.do_predict_roll = False

    main(con)
