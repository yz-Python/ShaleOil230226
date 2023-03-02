import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
class Data:
    def __init__(self, config):
        self.config = config
        self.full_static_data_path = self.normalize(config.static_data_path) #归一化静态数据，并获得静态数据地址用于自动选择

        self.data, self.staticdata_human,self.staticdata_nature,self.data_column_name= self.read_data()

        self.norm_human_data =self.staticdata_human
        self.norm_nature_data =self.staticdata_nature
        #self.data=self.data.astype('float')
        self.train_num =[]
        #self.test_start_num = int(self.data_num * self.config.train_data_rate)
        self.norm_data=[]
        
        self.std =[]
        self.mean =[]
        for i,data in enumerate(self.data):
            self.data_num = data.shape[0]
            train_num = int(self.data_num * self.config.train_data_rate)
            
            mean = np.mean(data, axis=0)#.reshape(self.data_num,1)
                     # 数据的均值和方差 axis =0在列上归一，axis=1在行上归一
            std = np.std(data, axis=0)#.reshape(self.data_num,1)                
        
            norm_data = (data - mean)/std   # 归一化，去量纲
            self.norm_data.append(norm_data)
            self.train_num.append(train_num)
            self.mean.append(mean)
            self.std.append(std)
   
        #self.norm_nature_data.append(self.staticdata_nature[i].reshape(1,config.naturesize)) 
              
        #self.norm_human_data.append(self.staticdata_human[i].reshape(1,config.humansize))  

    def normalize(self,static_path):   
        Data = pd.read_csv(static_path)
        cate = Data.columns.tolist()
        Data = Data.values
        feature = np.array(Data[:,:-1],dtype=float)
        number = Data[:,-1:]
        mean = np.mean(feature,axis=0)
        std = np.std(feature,axis=0)
        norm = (feature-mean)/std

        norm_data = np.concatenate([norm,number],axis=1)
        #print(norm_data)
        df1 = pd.DataFrame(data=norm_data,
                            columns=cate)
        norm_static_file = static_path.replace('.csv','_norm.csv')
        df1.to_csv(norm_static_file,index=False) 
        return norm_static_file
        
    def read_data(self):                # 读取初始数据
        init_data=[]
        data_column_name = []
        all_data_name= []
        for name in os.listdir(self.config.dynamic_data_root):
            if ".csv" in name:
                all_data_name.append(name)   #带有csv后缀,遍历所有数据集的名称
        for name in self.config.data_selected:  #从设定的井编号读取数据，数据形式为['All']或者['A1','C1']
            if name == 'All':  #name是自己输入决定使用的井，data_name是遍历得到的所有数据集的csv名称列表
                assert len(self.config.data_selected)==1 #保证输入只有All，不能All 和A1，B1一起写
                for data_name in all_data_name:  #读取全部井训练数据
                    data = pd.read_csv(os.path.join(self.config.dynamic_data_root,data_name), usecols=self.config.feature_columns)
                    init_data.append(data.values)
                    data_column_name.append(data.columns.tolist())
            else:
                name_csv = name+".csv" # A1 → A1.csv
                try:
                    data =pd.read_csv(os.path.join(self.config.dynamic_data_root,name_csv), usecols=self.config.feature_columns)#读取name井训练数据
                    init_data.append(data.values)
                    data_column_name.append(data.columns.tolist())
                except ValueError:
                    print('选取的{}井与读取到的数据集井不匹配，检查数据集或检查data_selected参数'.format(name))
        print('动态数据加载完成')
        data_selected = self.config.data_selected
        if data_selected[0] == 'All' :
            data_selected=[]
            for name in all_data_name:
                data_selected.append(name.replace('.csv',''))
        # init_static_human_data = pd.read_csv(self.config.train_static_data_path, usecols=self.config.static_human_column) #根须tool/config文件下静态参数列的位置选择性读取
        # init_static_nature_data = pd.read_csv(self.config.train_static_data_path, usecols=self.config.static_nature_column)
        init_static_data = np.array(self.select_static_data(dynamic_name_list=data_selected,norm_static_file=self.full_static_data_path))
        init_static_human_data =init_static_data[:,:,self.config.static_human_column].tolist() #self.select_static_data(dynamic_name_list=data_selected,norm_static_file=self.full_static_data_path,
                                                            #cols=self.config.static_human_column)
        init_static_nature_data = init_static_data[:,:,self.config.static_nature_column].tolist()
        #print(len(init_static_nature_data ))#self.select_static_data(dynamic_name_list=data_selected,norm_static_file=self.full_static_data_path,
                                                            #cols=self.config.static_nature_column)
        return init_data, init_static_human_data,init_static_nature_data ,data_column_name
                     # .columns.tolist() 是获取列名
    def select_static_data(self,dynamic_name_list,norm_static_file):#动态数据输入形式：['A1','B1','C1']
        static_data = []
        #print(norm_static_file)
        Data = pd.read_csv(norm_static_file)
        Data = Data.values
        feature = np.array(Data[:,:-1],dtype=float)
        Name = Data[:,-1:].tolist()
        try:
            for dynamic_name in dynamic_name_list:
                index = Name.index([dynamic_name])
                static_data.append(feature[index:index+1,:])
            #print(static_data)
        except ValueError:
            print('{}井不匹配静态数据，检查是否确实静态数据或者井编号不对应'.format(dynamic_name))
            os.kill() 
        print('静态数据加载完成')
        return(static_data)

    def get_train_and_valid_data(self):
        full_train_data, full_valid_data, full_train_label, full_valid_label=[],[],[],[]
        for j in range(len(self.norm_data)):
            data_ind = self.norm_data[j]
            print(data_ind.shape)
            feature_data = data_ind[:self.train_num[j]]
            label_data = data_ind[:self.train_num[j],
                                    self.config.label_in_feature_index]   
            norm_nature_data = self.norm_nature_data[j]
            norm_human_data = self.norm_human_data[j]
            
            static_nature_data = np.array([norm_nature_data for i in range(self.config.time_step)])
            static_nature_data = static_nature_data.reshape(static_nature_data.shape[0],static_nature_data.shape[2])
            static_human_data = np.array([norm_human_data for i in range(self.config.time_step)]) #静态数据复制5遍，因为要和5天预测3天维度的encoder输入数据对应上
            static_human_data = static_human_data.reshape(static_human_data.shape[0],static_human_data.shape[2]) #这里实现的是静态数据添加时间信息

            embed = np.eye(self.config.time_step)                                # [1 0 0]
            static_nature_data=np.append(static_nature_data,embed,axis=1)        # [0 1 0]
            static_human_data=np.append(static_human_data,embed,axis=1)          # [0 0 1]  拼接进去作为One-Hot编码 表示时间顺序

            length = self.config.time_step+self.config.predict_day  # 5+3=8
            #print(length)
            feature = [feature_data[start_index + i*(length) : start_index + (i+1)*(length)]  #训练部分按照错位进行采样得到的形式,包含油嘴油压产量
                    for start_index in range(length)                                          #这里是动态特征部分1-5天预测6-8,2-6天预测7-9这样一致往后推
                    for i in range((self.train_num[j] - start_index) // (length))]            #取得数据是1-8天，2-9天这样顺序，这样堆叠下去
            label = [label_data[start_index + i*length : start_index + (i+1)*length]        #标签部分处理方法相同，仅有产量
                    for start_index in range(length)
                    for i in range((self.train_num[j] - start_index) // length)]
            feature, label = np.array(feature), np.array(label)
            print(feature.shape,label.shape)
           
            train_data, valid_data, train_label, valid_label = train_test_split(feature, label, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)   # 划分训练和验证集
            s_h_train = np.array([static_human_data for k in range(len(train_data))])#s表示静态数据，h表示人工参数
            s_n_train = np.array([static_nature_data for k in range(len(train_data))])           #，n表示自然（地质）参数 
            s_h_valid = np.array([static_human_data for k in range(len(valid_data))])
            s_n_valid = np.array([static_nature_data for k in range(len(valid_data))])                  
            if j ==0:
                full_train_data, full_valid_data, full_train_label, full_valid_label =train_data, valid_data, train_label, valid_label
                full_s_h_train,full_s_n_train,full_s_h_valid,full_s_n_valid =s_h_train,s_n_train,s_h_valid,s_n_valid
            else:
                full_train_data, full_valid_data =np.concatenate([full_train_data,train_data],axis=0),np.concatenate([full_valid_data,valid_data],axis=0)
                full_train_label, full_valid_label =np.concatenate([full_train_label,train_label],axis=0),np.concatenate([full_valid_label,valid_label],axis=0)
                full_s_h_train = np.concatenate([full_s_h_train,s_h_train],axis=0)
                full_s_n_train = np.concatenate([full_s_n_train,s_n_train],axis=0)
                full_s_h_valid = np.concatenate([full_s_h_valid,s_h_valid],axis=0)
                full_s_n_valid = np.concatenate([full_s_n_valid,s_n_valid],axis=0)  #这里是将多井数据拼在一起，用于一起训练

        #此处分别对训练集和测试集进行对应输入采样
        #full_train_data是一个[xxx,8,3]的数据 full_train_label是一个[xxx，8,1]的数据
        encoder_train = full_train_data[:,:self.config.time_step,:]                                #取1-5天的动态数据和产量，油嘴油压产量
        decoder_train = full_train_label[:,self.config.time_step-1:length-1,:]                   #取5-7天的产量，最初是为了用于teacherforcing的进行训练，后来没有用teacherforcing，在实际的验证集没有用到这部分数据
        label_train = full_train_label[:,self.config.time_step:length,:]                         #取6-8天的产量，作为标签
        dynamic_train = full_train_data[:,self.config.time_step:length,:self.config.dynamic_length] #取5-8天的动态数据，油嘴油压

        encoder_valid = full_valid_data[:,:self.config.time_step,:]                              #验证集，同上
        decoder_valid = full_valid_label[:,self.config.time_step-1:length-1,:]
        label_valid = full_valid_label[:,self.config.time_step:length,:]
        dynamic_valid = full_valid_data[:,self.config.time_step:length,:self.config.dynamic_length]
       
        return encoder_train, decoder_train, label_train, dynamic_train,        \
               encoder_valid, decoder_valid, label_valid, dynamic_valid,        \
               full_s_h_train, full_s_n_train, full_s_h_valid, full_s_n_valid

    def get_test_data(self, roll ,return_label_data=False):
        norm_data = self.norm_data[0] #测试只考虑单井测试情况。
        train_num = self.train_num[0]
        feature_data = norm_data[train_num:] #测试样本train_num需要被设定为0
        time_step = self.config.time_step     # 防止time_step大于测试集数量
        step_for_no_roll = ((feature_data.shape[0]- time_step) // self.config.predict_day)+1
        print(step_for_no_roll)
        norm_nature_data = self.norm_nature_data[0]
        norm_human_data = self.norm_human_data[0]
        static_nature_data = np.array([norm_nature_data for i in range(self.config.time_step)])
        static_nature_data = static_nature_data.reshape(static_nature_data.shape[0],static_nature_data.shape[2])
        static_human_data = np.array([norm_human_data for i in range(self.config.time_step)]) #数据形式有点奇怪，不知道为什么，反正调一下
        static_human_data = static_human_data.reshape(static_human_data.shape[0],static_human_data.shape[2])

        embed = np.eye(self.config.time_step)                                  # [1 0 0]
        static_nature_data=np.append(static_nature_data,embed,axis=1)          # [0 1 0]
        static_human_data=np.append(static_human_data,embed,axis=1)            # [0 0 1]  拼接进去作为One-Hot编码

        # 在滚动测试数据中，采样方式按照Predict_day连续进行错位采样 1-5天预测6-8天，4-8天预测9-11天，与训练数据采样方式不同
        # 迭代测试数据仅采样1-5天数据，预测天数按照roll_predict_day决定
        if roll: #roll表示迭代预测
            encoder_test = [feature_data[ : time_step]]
            decoder_test = [feature_data[ time_step: ]]
            decoder_test= np.array(decoder_test)
            decoder_test = decoder_test[:,:,:self.config.dynamic_length]
            return np.array(new_encoder_test),decoder_test,static_human_data,static_nature_data   

        if not roll: #滚动预测 
            new_encoder_test = []
            if self.config.time_step >=self.config.predict_day: #采样方式按照Predict_day连续进行错位采样 1-5天预测6-8天，4-8天预测9-11天，与训练数据采样方式不同
                encoder_test = [feature_data[ i*self.config.predict_day :  time_step +i*self.config.predict_day] #
                   for i in range(step_for_no_roll)] #按照1-5天，4-8天。。。这样的顺序采样encoder输入样本

            if self.config.time_step <self.config.predict_day:   #基本用不上这种情况，不会出现预测天数大于输入天数的滚动预测
                middle_day = self.config.predict_day 
                encoder_test = [feature_data[ i*middle_day :  time_step +i*middle_day]
                   for i in range(step_for_no_roll)] #
    
            for i in encoder_test :
                if len(i) ==self.config.time_step:
                    new_encoder_test.append(i) #去除编码器输入数据不够的样本
                    
            new_decoder_test =[]
            decoder_test = [feature_data[  time_step+i*self.config.predict_day :  time_step+(i+1)*self.config.predict_day]
                for i in range(step_for_no_roll)] #按照6-8天，9-11天。。。这样的顺序采样decoder输入样本
            
            for i in decoder_test :
                if len(i) ==self.config.predict_day:
                    new_decoder_test.append(i)

        new_decoder_test =np.array(new_decoder_test)
        new_decoder_test = new_decoder_test[:,:,:self.config.dynamic_length] #decoder只保留油嘴油压数据
        encoder_test = np.array(new_encoder_test)
        decoder_test =new_decoder_test

        final_len =min(len(encoder_test),len(decoder_test))   
        encoder_test = encoder_test[:final_len,:,:]   #保证测试样本数目对应
        decoder_test = decoder_test[:final_len,:,:]

        return encoder_test,decoder_test,static_human_data,static_nature_data