import torch
from torch.nn import Module, LSTM, Linear, Conv1d, Softmax, ReLU, BatchNorm2d, LayerNorm, ModuleDict
import numpy as np
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout_rate)

    def forward(self, src, h_0, c_0):
        batch, _, _ = src.shape
        if h_0 == None:
            outputs, (h, c) = self.LSTM(src, )
        else:
            h_0 = h_0[:, :batch, :].contiguous()
            c_0 = c_0[:, :batch, :].contiguous()
            outputs, (h, c) = self.LSTM(src, (h_0, c_0))

        return outputs.transpose(0, 1), h, c

    '''
    output:最后一层LSTM的每个隐藏状态h        [batch_size,5,256]
    h:每一层最后一个隐藏状态ht    [2,batch_size,256] 一共2层LSTM
    c:每一层最后一个细胞状态ct
    模型输入1-5天的动态数据，输出5天的编码隐藏层和最后一天的隐藏层h，细胞层c
    '''


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, bidirectional=False):  # 即encoder和decoder的输出维度
        super(Attention, self).__init__()
        # 双向的话，enc_hid_dim要乘2

        self.q = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)  # 不要偏置，做一个线性变换
        self.k = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)  # 不要偏置，做一个线性变换
        self.softmax = nn.Softmax(dim=2)
        self.down = nn.Linear(dec_hid_dim * 2, dec_hid_dim, bias=False)

    def forward(self, h, enc_out):
        # torch.Size([2, 8, 256]) torch.Size([5, 8, 256]) 2是层数，8是batch,5是Encoder天数
        h = h.permute(1, 0, 2)  # 取h化为[8,2,256]
        enc_out = enc_out.permute(1, 0, 2)  # 化为[8,5,256]
        q = self.q(h)
        k = self.k(enc_out)
        v = self.v(enc_out)
        qk = torch.einsum('bij,bkj->bik', q, k)  # [8,2,10]
        att = self.softmax(qk)
        att_out = torch.einsum('bij,bjk->bik', att, v)  # [8,2,256]
        att_out = torch.cat([h, att_out], dim=2)
        out = self.down(att_out)
        out = out.permute(1, 0, 2).contiguous()

        return out

    '''
    输入h为[2,batch_size,256]，为Decoder部分送入的一日隐藏状态，2是LSTM层数
    对Encoder5天每一天的隐藏层进行加权求和，得到注意力结果，输出维度不变
    注意力机制输入上一时刻的隐藏层h和encoder5天的编码隐藏层
    输出为经过注意力加权的上一时刻隐藏层隐藏层h
    '''


class ATT_Decoder(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, num_layers, attention, config, embedding_dim, dropout_rate,
                 dynamicDe_num, bidirectional=False):
        super(ATT_Decoder, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        # self.attention1 = Attention(config.hidden_size, config.hidden_size)
        self.attention2 = Attention(config.hidden_size, config.hidden_size)
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.fc11 = nn.Linear(1, embedding_dim)
        # self.fc12 = nn.Linear(dynamicDe_num,embedding_dim)
        # self.fc1 = nn.Linear(2*embedding_dim,embedding_dim)
        # self.fc2 = nn.Linear(self.enc_hidden_size+self.dec_hidden_size+self.embedding_dim,1)
        self.fc3 = nn.Linear(embedding_dim, 1)

        self.lstm1 = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.dec_hidden_size,
                             num_layers=self.num_layers,
                             batch_first=True, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.dec_hidden_size,
                             num_layers=self.num_layers,
                             batch_first=True, dropout=dropout_rate)

    def forward(self, dec_input, h1, c1, h2, c2, enc_output):
        # dec_input：torch.Size([1, B, 1])  dyanmic_dec_input:torch.Size([1, B, 2]) 
        # h1:torch.Size([2, B, 256]) c1:torch.Size([2, B, 256])
        # h2：torch.Size([2, B, 256])  c2：torch.Size([2, B, 256]) 
        # enc_output: torch.Size([5, B, 256])
        #

        embedded = self.fc11(dec_input.transpose(0, 1))  # [batchsize，seqlen，embeddingsize]
        # Decoderinput是上一时刻的产量
        # embedded2 = self.fc12(dyanmic_dec_input.transpose(0,1))     #是当前时刻的动态影响因素，需要进行特征拼接
        # embedded = torch.cat([embedded1,embedded2],2)
        # embedded = self.fc1(embedded)   #变成：# embedded = [ batch_size, 1, emb_dim]
        # att_h1 = self.attention1(h1, enc_output)
        embedded, (h1, c1) = self.lstm1(embedded, (h1, c1))  # 先进行一次Decoder编码，再去做attention

        # att_h2 = self.attention2(h2, enc_output)  
        # lstm_input = embedded
        # dec_output, (dec_h, dec_c) = self.lstm2(lstm_input, (att_h2, c2))

        embedded = embedded.permute(1, 0, 2)
        att_embedded = self.attention2(embedded, enc_output)
        # 用decoder的当前状态编码后的h和encoder的所有时刻的状态enc_output的隐藏侧h计算注意力权重。
        att_embedded = att_embedded.permute(1, 0, 2)
        dec_output, (h2, c2) = self.lstm2(att_embedded, (h2, c2))

        pred = self.fc3(dec_output)
        pred = pred.transpose(0, 1)
        return pred, h1, c1, h2, c2

    '''
    输入t-1时刻的产量信息，输入t时刻的动态信息（油嘴油压），以及t-1时刻的隐藏层h1,h2和细胞层
    输出t时刻的产量预测值
    '''


class No_ATT_Decoder(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, num_layers, attention, embedding_dim, dropout_rate,
                 dynamicDe_num, bidirectional=False):
        super(No_ATT_Decoder, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.fc11 = nn.Linear(1, embedding_dim)
        # self.fc12 = nn.Linear(dynamicDe_num,embedding_dim)
        # self.fc1 = nn.Linear(2*embedding_dim,embedding_dim)
        # self.fc2 = nn.Linear(self.enc_hidden_size+self.dec_hidden_size+self.embedding_dim,1)
        self.fc3 = nn.Linear(embedding_dim, 1)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.dec_hidden_size, num_layers=self.num_layers,
                            dropout=dropout_rate)

    def forward(self, dec_input, h, c, enc_output):
        embedded = self.fc11(dec_input.transpose(0, 1))  # 应该是[batchsize，seqlen，embeddingsize]
        # embedded2 = self.fc12(dyanmic_dec_input.transpose(0,1))
        # embedded = torch.cat([embedded1,embedded2],2)
        # embedded = self.fc1(embedded)
        enc_output = enc_output.transpose(0, 1)  # enc_output = [batch_size, src_len, enc_hid_dim]
        lstm_input = embedded.transpose(0,
                                        1)  # torch.cat((embedded,att_c), dim=2)   #lstm_input = [1, batch_size, enc_hid_dim + emb_dim]
        dec_output, (dec_h, dec_c) = self.lstm(lstm_input, (h, c))
        pred = self.fc3(dec_output)
        return pred, dec_h, dec_c


class ECA(Module):  # 动静态特征融合模块
    def __init__(self, config):
        super(ECA, self).__init__()

        x_size = config.input_size  # 3
        self.x_size = x_size
        self.dynamic_size = x_size - 1  # 动态因素，和产量分开
        humansize = config.humansize  # 9
        naturesize = config.naturesize  # 4
        embeddingsize = config.embedding_size

        self.full_range = np.arange(embeddingsize * 3)
        np.random.shuffle(self.full_range)
        self.shuffle_list = self.full_range  # shuffle过后

        self.humanlinear = Linear(humansize + config.time_step, embeddingsize)  # 256
        self.naturelinear = Linear(naturesize + config.time_step, embeddingsize)  # 256
        self.dynamiclinear = Linear(self.dynamic_size, embeddingsize)
        self.xlinear = Linear(x_size, embeddingsize)
        self.valuelinear = Linear(1, embeddingsize)
        self.concat_downsample = Linear(embeddingsize * 3, embeddingsize)
        self.conv = Conv1d(in_channels=config.time_step,
                           out_channels=config.time_step,
                           kernel_size=5,
                           padding=3,
                           groups=config.time_step)
        self.relu = ReLU()
        self.softmax = Softmax(dim=2)
        self.config = config
        self.value_down = Linear(embeddingsize * 2, embeddingsize)

    def forward(self, x, human, nature):
        # X：[B,5,3]  Human:[B,5,4+5]  Nature:[B,5,9+5]
        # print(nature.shape)   x是送入Encoder的数据，包括油嘴油压和产量
        dynamic = x[:, :, :self.dynamic_size]  # dynamic只取油嘴油压
        value = x[:, :, -1].unsqueeze(-1)  # value只取产量
        x = self.xlinear(x)  # 这里分别处理主要是为了消融实验效果
        value = self.valuelinear(value)
        dynamic = self.dynamiclinear(dynamic)
        if self.config.only_dynamic == 1:
            return x  # 只用动态数据
        res = dynamic  # only_dynamic only_static_concat_dynamic only_static_plus_dynamic
        batch, _, xc = x.shape
        # print(human.shape)      #动静态特征融合
        human = self.humanlinear(human)
        human = human[:batch, :, :]
        nature = self.naturelinear(nature)
        nature = nature[:batch, :, :]
        if self.config.only_static_plus_dynamic == 1:
            return x + human + nature  # 动静态相加
        hn = torch.cat([human, nature], 2)  # 拼接静态特征
        hn = hn[:batch, :, :]
        xhn = torch.cat([dynamic, hn], 2)  # 动静态特征拼接
        if self.config.only_static_concat_dynamic == 1:
            return self.concat_downsample(xhn)  # 动静态拼接c
        b, n, c = xhn.shape
        Mask = torch.zeros(b, n, c).to(device=x.device)
        xx = torch.zeros(b, n, c).to(device=x.device)
        # 以一下部分为动静态融合
        for i, j in enumerate(self.shuffle_list):
            Mask[:, :, i] = xhn[:, :, j]  # 动静态特征shuffle

        Mask = self.conv(Mask)  # Shuffle后进行卷积

        for i, j in enumerate(self.shuffle_list):
            xx[:, :, j] = Mask[:, :, i]  # 反shuffle
        xx = self.relu(xx)
        x_select = xx[:, :, :xc]  # 提取动态特征对应index部分的融合特征

        # x_res =torch.mul(res,self.softmax(res))
        x_select = self.softmax(x_select)  # 对融合特征计算注意力权重
        dynamic = torch.mul(dynamic, x_select)
        dynamic = (dynamic + res) / 2  # 保留残差结构
        value = torch.cat([dynamic, value], 2)  # 将动态特征和产量信息拼接
        value = self.value_down(value)
        # x = x+res
        return (value)

    '''
    动静态特征融合模块，输入为静态数据和5天的动态数据
    输出按照parser的选择输出不同融合方式的动态数据
    '''


class Get_h_c(Module):  # 静态嵌入模块
    def __init__(self, config):
        super().__init__()
        inlength = config.humansize + config.naturesize
        embedding = config.embedding_size

        self.fc1 = Linear(inlength, embedding)
        self.relu = ReLU()
        self.LN1 = LayerNorm(embedding)
        self.fc2 = Linear(embedding, 3 * embedding)
        self.LN2 = LayerNorm(3 * embedding)

        self.linearh0 = Linear(3 * embedding, embedding)
        self.linearc0 = Linear(3 * embedding, embedding)
        self.config = config

    def forward(self, human, nature):
        # human(b,5,4+5) nature(b,5,9+5)这里的静态信息已经在Data类中提前处理了，添加了时间信息
        human = human[:, :self.config.lstm_layers, :self.config.humansize]  # 这里剥离时间信息，只保留原始特征信息，并且保证维度和lstm层数相同
        nature = nature[:, :self.config.lstm_layers, :self.config.naturesize]
        # print(human.shape)
        static = torch.cat([human, nature], dim=2)
        out = self.fc1(static)
        out = self.LN1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.LN2(out)
        out = self.relu(out)
        h_0 = self.linearh0(out)
        c_0 = self.linearc0(out)
        return h_0, c_0  # 大小为[B,2,256]

    '''
    静态嵌入模块，输入为静态数据，
    输出为初始隐藏层h和细胞状态c
    '''


class Net(Module):  # 基于Seq2Seq的大结构
    def __init__(self, config, teacherforcing=False, transfer=False):
        super(Net, self).__init__()
        # humansize = 4
        # naturesize = 9
        if False:
            self.learn_weight = torch.Tensor(1, 1, config.humansize + config.naturesize)
            torch.nn.init.normal_(self.learn_weight, mean=0.0, std=0.1)
            self.dynamic_weight = torch.Tensor(1, 1, config.input_size)
            torch.nn.init.normal_(self.dynamic_weight, mean=0.0, std=0.1)

        else:
            self.learn_weight = torch.ones(1, 1, config.humansize + config.naturesize)
            self.dynamic_weight = torch.ones(1, 1, config.input_size)
        self.fix_weight = torch.ones(1, 1, config.time_step, requires_grad=False)
        self.encoder = Encoder(input_dim=config.embedding_size, hidden_dim=config.hidden_size,
                               num_layers=config.lstm_layers, dropout_rate=config.dropout_rate)
        if config.use_attention == True:
            self.decoder = ATT_Decoder(enc_hidden_size=config.hidden_size, dec_hidden_size=config.hidden_size,
                                       num_layers=config.lstm_layers,
                                       attention=True, config=config,
                                       embedding_dim=config.embedding_size, dropout_rate=config.dropout_rate,
                                       dynamicDe_num=config.dynamic_length)
        else:
            self.decoder = No_ATT_Decoder(enc_hidden_size=config.hidden_size, dec_hidden_size=config.hidden_size,
                                          num_layers=config.lstm_layers,
                                          attention=None,
                                          embedding_dim=config.embedding_size, dropout_rate=config.dropout_rate,
                                          dynamicDe_num=config.dynamic_length)

        self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)
        self.ECA = ECA(config)
        self.teacherforcing = teacherforcing
        self.Gethc = Get_h_c(config)
        self.hc_initialize = config.use_static_embedding
        self.transfer = transfer
        self.config = config

    def forward(self, encoder_input, decoder_input, human, nature, h_0, c_0):
        #  h_0和c_0用静态数据升维度的特征矩阵。 encoder_input：[B,5,3] decoder_input:[3,B,1] dyanmic_dec_input:[3,B,2]
        # human,nature分别表示静态特征     human:[b,5,4+5=9]  nature:[b,5,9+5=14]
        # print(human.shape,nature.shape)
        self.learn_weight = self.learn_weight.to(encoder_input.device)
        self.dynamic_weight = self.dynamic_weight.to(encoder_input.device)
        self.fix_weight = self.fix_weight.to(encoder_input.device)
        weight = torch.sigmoid(self.learn_weight)
        human_weight = weight[:, :, :self.config.humansize]
        nature_weight = weight[:, :, self.config.humansize:]
        full_weight = human_weight
        for i in [self.fix_weight, nature_weight, self.fix_weight]:
            full_weight = torch.cat([full_weight, i], 2)
        full_weight = full_weight.repeat(encoder_input.shape[0], encoder_input.shape[1], 1)
        h_cat_n = torch.cat([human, nature], 2)
        after_weighted = h_cat_n * full_weight

        dynamic_weight = torch.sigmoid(self.dynamic_weight)
        dynamic_weight = dynamic_weight.repeat(encoder_input.shape[0], encoder_input.shape[1], 1)
        encoder_input = encoder_input * dynamic_weight

        human = after_weighted[:, :, :self.config.humansize + self.config.time_step]
        nature = after_weighted[:, :, self.config.humansize + self.config.time_step:]

        if self.hc_initialize:
            h_0, c_0 = self.Gethc(human, nature)
            h_0 = h_0.permute(1, 0, 2).contiguous()
            c_0 = c_0.permute(1, 0, 2).contiguous()
            embedding_h = h_0

        encoder_input = self.ECA(encoder_input, human, nature)  # 动静态特征融合

        batch_size = encoder_input.shape[0]
        predict_future_num = decoder_input.shape[0]  # 预测的时间长度

        enc_outputs, h1, c1 = self.encoder(encoder_input, h_0, c_0)  # 编码器对5天数据进行编码
        h2, c2 = h1.clone(), c1.clone()
        # 获得Encoder特征矩阵  torch.Size([B, 5, 256]) torch.Size([2, B, 256]) torch.Size([2, B, 256]) 2层LSTM
        encoder_transfer_h = h1
        outs = torch.zeros(predict_future_num, batch_size, 1).to(
            encoder_input.device)  # 创建outputs张量存储decoder输出，用于存放预测结果
        dec_input = decoder_input[0, :, :].unsqueeze(0)
        # dyna_dec_in = dyanmic_dec_input[0,:,:].unsqueeze(0)

        for t in range(0, predict_future_num):  # 对于设定未来预测天数进行迭代
            # dec_input = decoder_input[t,:,:]
            if self.config.use_attention == True:
                out, h1, c1, h2, c2 = self.decoder(dec_input, h1, c1, h2, c2, enc_outputs)
            else:
                out, h1, c1 = self.decoder(dec_input, h1, c1, enc_outputs)
            outs[t] = out

            # dyna_dec_in = dyanmic_dec_input[t,:,:].unsqueeze(0)
            # 可取真实产量值作为下一时刻的输入，也可取这一次的预测值作为下一时刻的hc输入
            # 在于teacherforcing训练方式的与否（目前不采用Teacherforcing的训练方式）
            # 在测试时候Teacherforcing 必须设置为False
            dec_input = decoder_input[t, :, :].unsqueeze(0) if self.teacherforcing else out
        if self.transfer == True:

            return outs, (encoder_transfer_h, embedding_h)
        else:
            return outs

    '''
    整体模型
    '''


'''
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
    测试代码
'''
