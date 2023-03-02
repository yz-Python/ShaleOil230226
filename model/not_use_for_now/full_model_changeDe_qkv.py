import torch
from torch.nn import Module, LSTM, Linear , Conv1d , Softmax , ReLU , BatchNorm2d,LayerNorm,ModuleDict
from model.Seq2Seq_model_changeDe_qkv import Encoder
import numpy as np
class ECA(Module):
    def __init__(self, config):
        super(ECA, self).__init__()

        x_size = config.input_size   #3
        self.x_size =x_size
        self.dynamic_size = x_size-1
        humansize = config.humansize   #9
        naturesize =config.naturesize  #4 
        embeddingsize = config.embedding_size

        self.full_range = np.arange(embeddingsize*3)
        np.random.shuffle(self.full_range)
        self.shuffle_list = self.full_range #shuffle过后

        
        self.humanlinear = Linear(humansize+config.time_step,embeddingsize)       #256 
        self.naturelinear = Linear(naturesize+config.time_step,embeddingsize)    #变256
        self.dynamiclinear = Linear(self.dynamic_size,embeddingsize)
        self.xlinear = Linear(x_size,embeddingsize)
        self.valuelinear =Linear(1,embeddingsize)
        self.concat_downsample =Linear(embeddingsize*3,embeddingsize)
        self.conv = Conv1d(in_channels = config.time_step,
                            out_channels= config.time_step,
                            kernel_size = 5 ,
                            padding= 3,
                            groups= config.time_step)
        self.relu = ReLU()
        self.softmax = Softmax(dim = 2)
        self.config =config
        self.value_down = Linear(embeddingsize*2,embeddingsize)
    def forward(self,x,human,nature):  #[[1,3,9]
        #print(nature.shape)  
        dynamic = x[:,:,:self.dynamic_size]
        value = x[:,:,-1].unsqueeze(-1)
        x = self.xlinear(x)
        value = self.valuelinear(value)
        dynamic =self.dynamiclinear(dynamic)
        if self.config.only_dynamic == 1:
            return x    
        res = dynamic                          #only_dynamic only_static_concat_dynamic only_static_plus_dynamic
        batch,_,xc = x.shape
        #print(human.shape)      #动静态特征融合
        human = self.humanlinear(human)
        human = human[:batch,:,:]
        nature = self.naturelinear(nature)
        nature = nature[:batch,:,:]
        if self.config.only_static_plus_dynamic ==1:
            return x+human+nature
        hn = torch.cat([human,nature],2)
        hn = hn[:batch,:,:]
        xhn = torch.cat([dynamic,hn],2)
        if self.config.only_static_concat_dynamic ==1:
            return self.concat_downsample(xhn)
        b,n,c = xhn.shape
        Mask = torch.zeros(b,n,c).to(device=x.device)
        xx = torch.zeros(b,n,c).to(device=x.device)

        for i,j in enumerate(self.shuffle_list):
            Mask[:,:,i]=xhn[:,:,j]              #shuffle

        Mask = self.conv(Mask)  #Shuffle后进行卷积

        for i,j in enumerate(self.shuffle_list):
            xx[:,:,j] =Mask[:,:,i] 
        xx =self.relu(xx)
        x_select = xx[:,:,:xc]
        
        #x_res =torch.mul(res,self.softmax(res))
        x_select = self.softmax(x_select)
        dynamic = torch.mul(dynamic,x_select)
        dynamic = (dynamic+res)/2
        value =torch.cat([dynamic,value],2)
        value = self.value_down(value)
        #x = x+res
        return(value)

class Get_h_c(Module):
    def __init__(self,config):
        super().__init__()
        inlength = config.humansize + config.naturesize
        embedding = config.embedding_size

        self.fc1= Linear(inlength , embedding)
        self.relu = ReLU()
        self.LN1 = LayerNorm(embedding)
        self.fc2= Linear(embedding , 3*embedding)
        self.LN2 = LayerNorm(3*embedding)
        
        self.linearh0 = Linear(3*embedding, embedding)
        self.linearc0 = Linear(3*embedding, embedding)
        self.config =config
    def forward(self,human,nature):
        human = human[:,:self.config.lstm_layers,:self.config.humansize]
        nature = nature[:,:self.config.lstm_layers,:self.config.naturesize]
        #print(human.shape)
        static = torch.cat([human,nature],dim=2)
        out = self.fc1(static)
        out = self.LN1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.LN2(out)
        out = self.relu(out)
        h_0 = self.linearh0(out)
        c_0 = self.linearc0(out)
        return h_0,c_0
class Net(Module):
    '''
    pytorch预测模型，包括LSTM时序预测层和Linear回归输出层
    可以根据自己的情况增加模型结构
    '''
    def __init__(self, config,teacherforcing = True,transfer = False):
        super(Net, self).__init__()
        #humansize = 4
        #naturesize = 9
        # self.lstm = LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size,
        #                  num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        self.encoder = Encoder(input_dim=config.embedding_size, hidden_dim=config.hidden_size,
                         num_layers=config.lstm_layers,  dropout_rate=config.dropout_rate)
        if config.use_attention ==True:
            from model.Seq2Seq_model_changeDe_qkv import ATT_Decoder,Attention
            self.decoder = ATT_Decoder(enc_hidden_size=config.hidden_size, dec_hidden_size = config.hidden_size, num_layers = config.lstm_layers,
                                   attention =Attention(config.hidden_size, config.hidden_size) , 
                                   embedding_dim = config.embedding_size,dropout_rate=config.dropout_rate,dynamicDe_num =config.dynamic_length)
        else:
            from model.Seq2Seq_model_changeDe_no_Attention import ATT_Decoder
            self.decoder = ATT_Decoder(enc_hidden_size=config.hidden_size, dec_hidden_size = config.hidden_size, num_layers = config.lstm_layers,
                                       attention=None,
                                        embedding_dim = config.embedding_size,dropout_rate=config.dropout_rate,dynamicDe_num =config.dynamic_length)
        #self.seq2seq = Seq2Seq(encoder = self.encoder, decoder = self.decoder,teacherforcing = teacherforcing)

        self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)
        self.ECA = ECA(config)
        self.teacherforcing = teacherforcing
        self.Gethc = Get_h_c(config)
        self.hc = config.use_static_embedding
        self.transfer = transfer
        self.config=config
    def forward(self, x,decoder_input,dyanmic_dec_input,human,nature,h_0,c_0,first_one=True):
        
        #  h_0和c_0用静态数据升维度。
        #human,nature分别表示静态特征   x:[1,3,4]   human:[1,3,4+3=7]  nature:[1,3,9+3=12]
        #print(human.shape,nature.shape)
        if first_one==True:
            if self.hc:
                h_0,c_0 =self.Gethc(human,nature)
                h_0 = h_0.permute(1,0,2).contiguous()
                c_0 = c_0.permute(1,0,2).contiguous()
                embedding_h = h_0
        elif first_one==False:
            h_0,c_0 = h_0.contiguous(),c_0.contiguous()
        encoder_input = self.ECA(x,human,nature)
        #print(decoder_input.shape)
        batch_size = encoder_input.shape[0]    
        decoder_future_num = decoder_input.shape[0] #预测的时间长度
        enc_outputs, h, c = self.encoder(encoder_input,h_0,c_0)
        #获得Encoder特征矩阵  torch.Size([5, 2, 256]) torch.Size([4, 2, 256]) torch.Size([4, 2, 256]) 四层LSTM
        encoder_h = h
        h_for_next_batch = enc_outputs[self.config.predict_day-1:self.config.predict_day,:,:]
        h_for_next_batch = h_for_next_batch.repeat(self.config.lstm_layers,1,1)
        c_for_next_batch = torch.zeros(self.config.lstm_layers, batch_size, self.config.embedding_size).to(h_for_next_batch.device)
        #创建outputs张量存储decoder输出
        #outs = torch.zeros(decoder_future_num, batch_size, 1).to(device)
        outs = torch.zeros(decoder_future_num, batch_size, 1).to(encoder_input.device)     ####！！！注意在使用的时候加device放到gpu上！！！！！！！ 此处仅是测试代码
        dec_input = decoder_input[0,:,:].unsqueeze(0)
        dyna_dec_in = dyanmic_dec_input[0,:,:].unsqueeze(0)
        #用于存放预测结果
        for t in range(0, decoder_future_num):
            #dec_input = decoder_input[t,:,:]

            out, h, c = self.decoder(dec_input,dyna_dec_in, h, c,enc_outputs)   
            outs[t] = out                
            
            dyna_dec_in = dyanmic_dec_input[t,:,:].unsqueeze(0)
            #可取真实值作为下一时刻的输入，也可取这一次的预测值作为下一时刻的输hc入,关键在于teacherforcing
            dec_input = decoder_input[t,:,:].unsqueeze(0) if self.teacherforcing else out
        if self.transfer == True:
            
            return outs,h_for_next_batch,c_for_next_batch,(encoder_h,embedding_h)
        else:
            return outs,h_for_next_batch,c_for_next_batch
        
if __name__ =='__main__':
    from config_test import Config
    from collections import OrderedDict
    from torchvision.models._utils import IntermediateLayerGetter
    model =Net(Config)
    model_teacher =IntermediateLayerGetter(model,{'encoder':'out'})
    xin =torch.randn(1,5,3)
    decoder_input=(1,3,1)
    dyanmic_dec_input=(1,3,2)
    human=torch.randn(1,5,14)
    nature=torch.randn(1,5,9)
    h=None
    c=None
    out =model_teacher([xin,decoder_input,dyanmic_dec_input,human,nature,h,c])
    x,h,c = out.items
    print()
        
