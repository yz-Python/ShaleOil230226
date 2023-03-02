#------------- 定义基本的模型框架 -------------------#
import torch
from torch.utils.data import DataLoader, TensorDataset
from xpinyin import Pinyin
from tqdm import tqdm
from model.full_model_changeDe_qkv import Net
from sklearn.metrics import mean_squared_error ,mean_absolute_error ,r2_score
from tool.mean_relative_error import mean_relative_error
from tool.MMDloss import MMD_loss
from tool.RMSLEloss import RMSLEloss
import setproctitle

def train(config, logger, train_and_valid_data):
    
    device = torch.device(config.cudadevice if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU训练还是GPU
   
    if config.do_train_visualized:
        import visdom
        vis = visdom.Visdom(env='model_pytorch')
    train_X, decoder_X, label_X, dynamic_X,valid_Y, decoder_Y, label_Y,dynamic_Y,full_s_h_train,full_s_n_train,full_s_h_valid,full_s_n_valid= train_and_valid_data
    #train_X, train_static, train_Y, valid_X, valid_Y = train_and_valid_data
    full_s_h_train,full_s_n_train=torch.from_numpy(full_s_h_train).float(), torch.from_numpy(full_s_n_train).float()
    full_s_h_valid,full_s_n_valid=torch.from_numpy(full_s_h_valid).float(), torch.from_numpy(full_s_n_valid).float()
    train_X, decoder_X, label_X, dynamic_X= torch.from_numpy(train_X).float(), torch.from_numpy(decoder_X).float(), torch.from_numpy(label_X).float(),torch.from_numpy(dynamic_X).float()    # 先转为Tensor
    print(train_X.shape,decoder_X.shape,label_X.shape,dynamic_X.shape)
    train_loader = DataLoader(TensorDataset(train_X, decoder_X,label_X,dynamic_X,full_s_h_train,full_s_n_train), batch_size=config.batch_size)    # DataLoader可自动生成可训练的batch数据

    valid_Y, decoder_Y, label_Y, dynamic_Y= torch.from_numpy(valid_Y).float(), torch.from_numpy(decoder_Y).float(), torch.from_numpy(label_Y).float() ,torch.from_numpy(dynamic_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_Y, decoder_Y, label_Y,dynamic_Y,full_s_h_valid,full_s_n_valid), batch_size=config.batch_size)
    print(valid_Y.shape,decoder_Y.shape,label_Y.shape,dynamic_Y.shape)
    
    model_teacher = Net(config,teacherforcing= False,transfer=True).to(device) 
    model_student = Net(config,teacherforcing= False,transfer=True).to(device)      # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中
    model_teacher.load_state_dict(torch.load(config.model_save_path + config.transfer_name+config.model_name))
    student_weight = torch.load(config.model_save_path + config.transfer_name+config.model_name)
    ###########################################################################
    #尝试调整部分
    # del_key=[]
    # for key,_ in student_weight.items():
    #     if 'ECA' not in key:
    #         del_key.append(key)
    # for key in del_key:
    #     del student_weight [key]
    # model_student.load_state_dict(student_weight,strict=False)
    
    # for name,para in model_student.named_parameters():
    #     if 'ECA' in name:
    #         para.requires_grad_=False #冻结ECA权重
    ###################################################################### 
    model_student.load_state_dict(student_weight,strict=False)
    for name,para in model_teacher.named_parameters():
        para.requires_grad_=False #冻结Teacher权重
    train_layer = [p for p in model_student.parameters() if p.requires_grad == True]
    optimizer = torch.optim.Adam(train_layer, lr=config.learning_rate)
    criterion_label = RMSLEloss()      # 这两句是定义优化器和loss
    criterion_MMD = MMD_loss()
    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model_student.train()                   # pytorch中，训练时要转换成训练模式
        model_teacher.train()
        train_loss_array = []
        h_0,c_0 = None,None
        loop = tqdm(enumerate(train_loader),total =len(train_loader))
        for i, _data in loop:
            setproctitle.setproctitle("zys:" + str(epoch) + "/" + "{}".format(config.epoch))
            _train_X, _decoder_X, label_X,dynamic_X,human,nature= _data[0].to(device),_data[1].to(device),_data[2].to(device),_data[3].to(device),_data[4].to(device),_data[5].to(device)
            optimizer.zero_grad()               # 训练前要将梯度信息置 0
            #print(_train_X.shape,_decoder_X.shape, label_X.shape)  #torch.Size([1, 5, 4]) torch.Size([1, 3, 1]) torch.Size([1, 3, 1])
            _decoder_X = _decoder_X.permute(1,0,2)
            label_X = label_X.permute(1,0,2)     #成为[3,1(batch_size),1]
            dynamic_X = dynamic_X.permute(1,0,2)
            #print(dynamic_X.shape,_decoder_X.shape)
            #for idx in mark_t:
               # human_in.append(human[int(idx.item())])
                #nature_in.append(nature[int(idx.item())])
           #human_in = torch.from_numpy(np.array(human_in)).float().to(device)     #废弃代码
           # nature_in =torch.from_numpy(np.array(nature_in)).float().to(device)
            pred_Y, h_out,c_out,h_transfer_student= model_student(_train_X,_decoder_X,dynamic_X,human, nature, h_0,c_0)    # 这里走的就是前向计算forward函数
            _,_, _,h_transfer_teacher = model_teacher(_train_X,_decoder_X,dynamic_X,human, nature, h_0,c_0)
            
            #h_0, c_0 = h_out,c_out
            #h_0.detach_(), c_0.detach_()    # 去掉梯度信息 已废弃 ，采用静态嵌入已经和后续传入的hc0无关了
                #hidden_train = (h_0, c_0)
            loss1 = criterion_label(pred_Y, label_X)  # 计算loss

            h_t = h_transfer_teacher[0].permute(1,0,2).reshape(h_transfer_teacher[0].shape[1],-1)
            h_s = h_transfer_student[0].permute(1,0,2).reshape(h_transfer_student[0].shape[1],-1)  #encoder输出的h
            # h_t = student_embedding_h.permute(1,0,2).reshape(student_embedding_h.shape[1],-1)
            # h_s = teacher_embedding_h.permute(1,0,2).reshape(teacher_embedding_h.shape[1],-1)  #hc0
            
            loss2 = criterion_MMD(h_t,h_s)
            alpha = config.loss_alpha    #用来限制MMD对模型的影响
            loss = loss1+loss2*alpha
            loss.backward()                     # 将loss反向传播
            optimizer.step()                    # 用优化器更新参数
            train_loss_array.append(loss.item())
            global_step += 1
            if config.do_train_visualized and global_step % 100 == 0:   # 每一百步显示一次
                vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
                         update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))

        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        
        model_student.eval()                    # pytorch中，预测时要转换成预测模式
        model_student.teacherforcing = False
        valid_loss_array = []
        h_0,c_0 = None,None
        loopv = tqdm(enumerate(valid_loader),total =len(valid_loader))
        for i, _data in loopv:
            
            valid_Y, decoder_Y, label_Y ,dynamic_Y,human,nature= _data[0].to(device),_data[1].to(device),_data[2].to(device),_data[3].to(device),_data[4].to(device),_data[5].to(device)
            decoder_Y = decoder_Y.permute(1,0,2).contiguous()
            label_Y = label_Y.permute(1,0,2).contiguous()
            dynamic_Y = dynamic_Y.permute(1,0,2).contiguous()
            #_valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            
            pred_Y, h_out,c_out, _ = model_student(valid_Y, decoder_Y, dynamic_Y,human, nature, h_0,c_0)
            #pred_Y, hidden_valid = model(_valid_X, human, nature, hidden_valid)
            #if not config.do_continue_train: hidden_valid = None
            loss = criterion_label(pred_Y,  label_Y)  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
              "The valid loss is {:.6f}.".format(valid_loss_cur))
        # torch.save(model_student.state_dict(), config.model_save_path + config.name+ config.model_name)
        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model_student.state_dict(),config.model_save_path +str(config.loss_alpha)+'_pretrain_'+config.trainname+config.name+ config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:    # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                logger.info(" The training stops early in epoch {}".format(epoch))
                break


def predict(config, test_X,dynamic_x,human, nature):
    # 获取测试数据
    device = torch.device(config.cudadevice if config.use_cuda and torch.cuda.is_available() else "cpu")
    test_X = test_X
    test_X = torch.from_numpy(test_X).float()
    dynamic_x =dynamic_x
    dynamic_x = torch.from_numpy(dynamic_x).float()
    print(test_X.shape,dynamic_x.shape)
    test_loader = DataLoader(TensorDataset(test_X, dynamic_x),batch_size=1)

    human, nature = torch.from_numpy(human).float(), torch.from_numpy(nature).float()
    bathsize = config.batch_size
    
    human = human.repeat(bathsize,1,1).to(device)
    nature = nature.repeat(bathsize,1,1).to(device)

    # 加载模型
    
    model = Net(config,teacherforcing= False).to(device)
    model.load_state_dict(torch.load(config.model_save_path +str(config.loss_alpha)+'_pretrain_'+config.trainname+config.name+ config.model_name))   # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)
    predict_day = config.predict_day
    # 预测过程
    model.eval()
    hidden_predict = None
    h_0,c_0 = None,None
    print(test_X.shape)
    for _data in test_loader:
        
        data_X ,dynamic_x= _data[0].to(device), _data[1].to(device)

        decoder_input0 = data_X[:,-1:,-1:].permute(1,0,2)#取输入特征的最后一维作为Decoder0的初始输入
                                                         #因为只在Predict所以这么设计
        #print(decoder_input0.shape)#
        predictTensor = torch.zeros(predict_day-1,1,1).to(device)
        decoder_input = torch.cat((decoder_input0,predictTensor),dim = 0)

        dynamic_input = dynamic_x.permute(1,0,2)
        #print(decoder_input0.shape)#
        #predictTensor = torch.zeros(predict_day-1,1,1).to(device)
        #decoder_input = torch.cat((decoder_input0,predictTensor),dim = 0)
        
        pred_X, h_out,c_out = model(data_X,decoder_input,dynamic_input,human, nature, h_0,c_0)
        # if not config.do_continue_train: hidden_predict = None    # 废弃，实验发现无论是否是连续训练模式，把上一个time_step的hidden传入下一个效果都更好
        cur_pred = torch.squeeze(pred_X, dim=2)
        #print(cur_pred.shape)
        result = torch.cat((result, cur_pred), dim=0)
        #print(result.shape)

    return result.detach().cpu().numpy()    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据
def predict_roll(config, test_X,dynamic_x, human, nature):
    # 获取测试数据
    device = torch.device(config.cudadevice if config.use_cuda and torch.cuda.is_available() else "cpu")

    test_X = test_X
    test_X = torch.from_numpy(test_X).float()
    dynamic_x =dynamic_x
    dynamic_x = torch.from_numpy(dynamic_x).float()
    print(test_X.shape,dynamic_x.shape)

    roll_predict_day = config.roll_predict_day
   
    test_loader = DataLoader(TensorDataset(test_X, dynamic_x), batch_size=1)

    human, nature = torch.from_numpy(human).float(), torch.from_numpy(nature).float()
    bathsize = config.batch_size
    
    human = human.repeat(bathsize,1,1).to(device)
    nature = nature.repeat(bathsize,1,1).to(device)

    # 加载模型
    
    model = Net(config,teacherforcing= False).to(device)
    model.load_state_dict(torch.load(config.model_save_path +str(config.loss_alpha)+'_pretrain_'+config.trainname+config.name+ config.model_name))   # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()
    h_0,c_0 = None,None

    for _data in test_loader:
        data_X ,dynamic_x= _data[0].to(device), _data[1].to(device)
        decoder_input0 = data_X[:,-1:,-1:].permute(1,0,2)
        #print(decoder_input0.shape)#
        predictTensor = torch.zeros(roll_predict_day-1,1,1).to(device)
        decoder_input = torch.cat((decoder_input0,predictTensor),dim = 0)
        dynamic_input = dynamic_x.permute(1,0,2)        
        pred_X, h_out,c_out = model(data_X,decoder_input,dynamic_input,human, nature, h_0,c_0)
        #pred_X, hidden_predict = model(data_X, human, nature, hidden_predict)
        # if not config.do_continue_train: hidden_predict = None    # 实验发现无论是否是连续训练模式，把上一个time_step的hidden传入下一个效果都更好
        #print(pred_X.shape)
        cur_pred = torch.squeeze(pred_X, dim=2)
        #print(cur_pred.shape)
        result = torch.cat((result, cur_pred), dim=0)
        #print(result.shape)

    return result.detach().cpu().numpy()    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据

#---------------------这一部分是应用的模块 --------------------#

import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tool.config import Config 
from tool.data import Data
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

def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray,name:str):
    # label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : 
    #                     origin_data.train_num + origin_data.start_num_in_test +config.roll_predict_day,
    #                                         config.label_in_feature_index
    label_len=len(predict_norm_data)
    calculate_label_data = origin_data.data[0][origin_data.train_num[0] + origin_data.start_num_in_test :
        origin_data.train_num[0] + origin_data.start_num_in_test+label_len,
                                            config.label_in_feature_index]
    label_data =origin_data.data[0][origin_data.train_num[0] + origin_data.start_num_in_test :,config.label_in_feature_index]
    print(label_data.shape)
    
    predict_data = predict_norm_data * origin_data.std[0][config.label_in_feature_index] + \
                   origin_data.mean[0][config.label_in_feature_index]   # 通过保存的均值和方差还原数据, 这个很重要
    
    length=min(label_data.shape[0] , predict_data.shape[0])
    #label_data=label_data[:length,:]
    #predict_data = predict_data[:length,:]
    #assert label_data.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[0][i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)
    p = Pinyin()
    label_name_pinyin =[]
    for i in range(len(label_name)):
        label_name_pinyin.append(p.get_pinyin(label_name[i]))
    # label 和 predict 是错开config.predict_day天的数据的
    # 下面是两种norm后的loss的计算方式，结果是一样的，可以简单手推一下
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of {} is ".format(label_name) + str(loss_norm))
    #mean_squared_error ,mean_absolute_error ,r2_score
    mse = mean_squared_error(calculate_label_data[:] , predict_data[:])
    mae = mean_absolute_error(calculate_label_data[:] , predict_data[:])
    r2  = r2_score(calculate_label_data[:] , predict_data[:])
    mre = mean_relative_error(calculate_label_data[:] , predict_data[:])
    #loss = np.sum((calculate_label_data[:] - predict_data[:] ) ** 2)/len(predict_data)
    #loss_norm = loss#/(origin_data.std[0][config.label_in_feature_index] ** 2)
    logger.info("The mean relative error of {} is ".format(label_name_pinyin) + str(mre))
    logger.info("The mean squared error of {} is ".format(label_name_pinyin) + str(mse))
    logger.info("The mean average error of {} is ".format(label_name_pinyin) + str(mae))
    logger.info("The R^2 of {} is ".format(label_name_pinyin) + str(r2))

    label_X = range(len(label_data))
    predict_X =range(len(predict_data))
    predict_X = [ x + config.time_step for x in predict_X]
    plt.cla()
    plt.clf()
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    if True:#not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
        for i in range(label_column_num):
            plt.figure(i+1)      
            #plt.rcParams["font.sans-serif"] = ["SimHei"]
               # 预测数据绘制
           #这里是中文的设置, 没有这个设置"日产油量无法显示"
            plt.plot(label_X, label_data[ :,i], label='label')
            plt.plot(predict_X, predict_data[:,i], label='predict')            #plt.title("Predict {}  with {}".format(label_name[i], config.used_frame))
            plt.title(config.trainname+"迁移"+config.valname)
            plt.legend()
            logger.info("The predicted {} for the next {} day(s) is: ".format(label_name_pinyin[i], config.predict_day) +  
                str(np.squeeze(predict_data[-config.predict_day:, i])))
            if config.do_figure_save:
                #print(config.figure_save_path+"add_stastic_{}predict_{}_with_{}.png".format(config.continue_flag, label_name_pinyin[i], config.used_frame))
                #plt.savefig(config.figure_save_path+"{}.png".format('alpha'+config.loss_alpha+'_'+config.trainname+'val'+config.name+name))
                plt.savefig(config.figure_save_path+"{}.png".format('alpha'+str(config.loss_alpha)+'_'+config.trainname+'_pretrain_'+config.name+name))
        plt.show()

def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data_gainer = Data(config)

        if config.do_train:
            if config.finetune==True:
                train_x, decoder_x, label_x,dynamic_x, train_y, decoder_y, label_y,dynamic_y, full_s_h_train,full_s_n_train,full_s_h_valid,full_s_n_valid= data_gainer.get_train_and_valid_data()
            #print(train_x.shape,decoder_x.shape, label_x.shape, train_y.shape, decoder_y.shape, label_y.shape)
                train(config, logger, [train_x, decoder_x, label_x,dynamic_x, train_y, decoder_y, label_y,dynamic_y,full_s_h_train,full_s_n_train,full_s_h_valid,full_s_n_valid])
            if config.finetune==False:
            #不做迁移训练，直接测试
                device = torch.device(config.cudadevice if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU训练还是GPU
                model_student = Net(config,teacherforcing= False,transfer=True).to(device)
                model_student.load_state_dict(torch.load(config.model_save_path + config.transfer_name+config.model_name))
                torch.save(model_student.state_dict(), config.model_save_path + config.name+ config.model_name) 
        if config.do_predict:
            config.batch_size = 1
            test_X, dynamic_x, human, nature = data_gainer.get_test_data(roll=False,return_label_data=False)
            pred_result = predict(config, test_X,dynamic_x, human, nature)       # 这里输出的是未还原的归一化预测数据
            draw(config, data_gainer, logger, pred_result,'_no_roll')
        if config.do_predict_roll:
            config.batch_size = 1
            test_X, dynamic_x,human, nature = data_gainer.get_test_data(roll= True,return_label_data=False)
            pred_result = predict_roll(config, test_X,dynamic_x, human, nature)       # 这里输出的是未还原的归一化预测数据
            draw(config, data_gainer, logger, pred_result,'_roll')
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__=="__main__":
    import argparse
    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    #如果在linux下面, 可以方便采用下面的参数输入
    #parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    #parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--epoch", default=10, type=int, help="epochs num")
    parser.add_argument("--ttotrain", default=1, type=int, help="sign==train means train")
    parser.add_argument("--use_attention",default=1, type=int, help="seq2seq_attention")
    parser.add_argument("--use_static_embedding",default=1, type=int, help="Inite,h0_c0")
    parser.add_argument("--only_dynamic",default=0, type=int, help="只用动态嵌入")
    parser.add_argument("--only_static_concat_dynamic",default=0, type=int, help="静态拼接动态降维")
    parser.add_argument("--only_static_plus_dynamic",default=0, type=int, help="静态和动态直接相加")
    parser.add_argument("--roll_predict_day",default=110, type=int, help="滚动预测天数")
    
    parser.add_argument("--predict_day", default=1, type=int, help="Decoder预测天数")
    parser.add_argument("--time_step", default=10, type=int, help="Encoder读取天数")
    
    parser.add_argument("--transfer_name", default='A1C1', type=str, help="source_name")  #加载Teacher预训练的权重
    parser.add_argument("--name", default='B1', type=str, help="name")
    
    parser.add_argument("--loss_alpha", default=0, type=float, help="迁移Loss影响")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config

    con.predict_day= 3
    con.time_step = 5
    sign = con.ttotrain
    use_attention = '_No_Attention'
    use_static_embedding ='_hc_False'
    end_sign = '_seq2seq_RMSLE'
    dynamic_fusion='_ds_fusion'
    if con.only_dynamic == 1:
        dynamic_fusion = '_dynamic_only' 
    if con.only_static_concat_dynamic == 1:
        dynamic_fusion = '_concat_only' 
    if con.only_static_plus_dynamic == 1:
        dynamic_fusion = '_plus_only' 
    if con.use_attention == 1:
        con.use_attention = True
        use_attention = '_Attention'
    else :
        con.use_attention =False
    if con.use_static_embedding ==1:
        con.use_static_embedding =True
        use_static_embedding ='_hc_True'
    else :
        con.use_static_embedding =False
    con.valname = con.name   
    con.name = con.name + use_attention + use_static_embedding + end_sign + dynamic_fusion + str(con.time_step) +str(con.predict_day)
    #con.transfer_name = con.transfer_name + use_attention + use_static_embedding + end_sign + dynamic_fusion + str(con.time_step) +str(con.predict_day)
    con.trainname = con.transfer_name
    con.transfer_name = 'Pretrain_'+con.transfer_name + use_attention + use_static_embedding + end_sign + dynamic_fusion + str(con.time_step) +str(con.predict_day)
    print('Train:'+str(sign))
    
    if con.valname=='A1':
        con.train_data_path = ['input/csv/A1.csv']
        con.train_static_data_path = 'input/csv/A1_static.csv'
    elif con.valname=='B1':
        con.train_data_path = ['input/csv/B1_no_wrongpoint.csv']
        con.train_static_data_path = 'input/csv/B1_static.csv'
    elif con.valname=='C1':
        con.train_data_path = ['input/csv/C1.csv']
        con.train_static_data_path = 'input/csv/C1_static.csv'
    else:
        assert('没有这个井编号')
    if sign == 1:
        con.train_data_rate = 0.2   #使用20%的数据进行训练，80%数据进行测试，滚动预测需要修改roll_predict_day，设置大于测试数据的长度就会报错
        #con.finetune = False
        con.epoch=40
        con.patience =5
        con.finetune = True
        con.do_train = True
        con.do_predict = True      #进行真值预测 图像名称以no_roll结尾
        con.do_predict_roll = True #进行滚动预测 图像名称以roll结尾
    if sign == 0:
        con.batch_size=1
        con.train_data_rate = 0
        con.do_train = True
        con.finetune = False  #不做迁移
        con.roll_predict_day = 124
        con.do_predict = True
        con.do_predict_roll = True
    
    main(con)
