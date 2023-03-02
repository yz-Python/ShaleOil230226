import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(Encoder, self).__init__()
        # 设置输入参数
        self.input_dim = input_dim
        #self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        # 根据input和embbed的的维度，初始化embedding层
        #self.embedding = nn.Embedding(input_dim, self.embbed_dim)
        # 初始化GRU，获取embbed的输入维度，输出隐层的维度，设置GRU层的参数
        #self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                         num_layers=num_layers, batch_first=True, dropout=dropout_rate)

    def forward(self, src, h_0,c_0):
        if h_0 ==None:
            outputs, (h,c) = self.LSTM(src,)
        else:
            outputs, (h,c) = self.LSTM(src,(h_0,c_0))

        return outputs.transpose(0,1), h , c
        '''
        output, (hn, cn) = rnn(input, (h0, c0))
        output:最后一层LSTM的每个隐藏状态h        [1,5,256]
        h:每一层最后一个隐藏状态ht    [5,1,256] 一共5层
        c:每一层最后一个细胞状态ct
        '''
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, bidirectional=False):   #即encoder和decoder的输出维度
        super(Attention, self).__init__()
        #双向的话，enc_hid_dim要乘2
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.w = nn.Linear((enc_hid_dim*2)+dec_hid_dim, dec_hid_dim,bias=False)   #不要偏置，做一个线性变换
        else:
            self.w = nn.Linear(enc_hid_dim+dec_hid_dim, dec_hid_dim,bias=False)   #不要偏置，做一个线性变换
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)   #不要偏置，做一个线性变换
        
    def forward(self, h, enc_out):
        # h = [1,batch_size, dec_hid_dim]
        # enc_out = [src_len, batch_size, enc_hid_dim]
        h = h.squeeze(0)
        src_len = enc_out.shape[0] 
        #因为要将当前时刻的h和所有时刻的h即enc_out（一个src_len个时刻）的h拼接送入全连接，所以先将h复制src_len次，然后拼接
        #s = [batch_size, src_len, dec_hid_dim]
        
        if len(h.shape)==3:  h = h[0] #torch.Size([2, 16, 512])   #当lstm多层时，只取最后一层的h。

        h = h.unsqueeze(1).repeat(1,src_len,1) 

        #交换维度和h一样，方便concat，变成：enc_out = [batch_size, src_len, enc_hid_dim]
        enc_out = enc_out.transpose(0,1) 
        
        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.w(torch.cat((h,enc_out), dim=2)))
        
        #因为经过self.v后维度变为：[batch_size, src_len, 1]，所以squeeze掉最后一个维度
        attention = self.v(energy).squeeze(2)  
        
        #用softmax在src_len维度归一化一波，就得到了attention权重
        return F.softmax(attention, dim=1)
        
class ATT_Decoder(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, num_layers, attention, embedding_dim, dropout_rate,bidirectional=False):
        super(ATT_Decoder, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(1,embedding_dim)
        self.fc2 = nn.Linear(self.enc_hidden_size+self.dec_hidden_size+self.embedding_dim,1)
        self.fc3 = nn.Linear(embedding_dim,1)
        #!!!!decoder的lstm不需要双向，因为他要生成正确的顺序的序列。只是encoder中的lstm要双向
        #时序任务我认为最好不要考虑encoder双向
        self.lstm = nn.LSTM(input_size=self.enc_hidden_size+self.embedding_dim, hidden_size=self.dec_hidden_size, num_layers=self.num_layers,
                            dropout=dropout_rate)
        '''
        if self.bidirectional:
            self.lstm = nn.LSTM(input_size=(self.enc_hidden_size*2)+self.embedding_dim, hidden_size=self.dec_hidden_size, num_layers=self.num_layers)
            self.fc = nn.Linear((self.enc_hidden_size*2)+self.dec_hidden_size+self.embedding_dim, self.vocab_size)
        else:
            self.lstm = nn.LSTM(input_size=self.enc_hidden_size+self.embedding_dim, hidden_size=self.dec_hidden_size, num_layers=self.num_layers)
            self.fc = nn.Linear(self.enc_hidden_size+self.dec_hidden_size+self.embedding_dim, self.vocab_size)
        '''
    def forward(self, dec_input, h, c, enc_output):
        # dec_input = [batch_size]，直接用产量表示
        # h = [batch_size, dec_hid_dim], c = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim]
        #dec_input = dec_input.unsqueeze(1)    #dec_input = [batch_size, 1]
        embedded = self.fc1(dec_input) #应该是[batchsize，seqlen，embeddingsize]

        #embedded = self.embedding(dec_input).transpose(0,1)   #交换一下维度，变成：# embedded = [1, batch_size, emb_dim]

        #用decoder的当前状态h和encoder的所有时刻的状态enc_output计算注意力权重。
        att = self.attention(h, enc_output).unsqueeze(1)   #多增加一个维度，变成：a = [batch_size, 1, src_len] 
        
        enc_output = enc_output.transpose(0,1)    # enc_output = [batch_size, src_len, enc_hid_dim]
        
        #att和enc_output矩阵相乘相当于维度变为:batch_size  x  (1 x src_len)  x  (src_len x enc_hid_dim) 变为[batch_size, 1, enc_hid_dim]
        att_c = torch.bmm(att, enc_output).transpose(0,1)   #转置一下，变为：[1, batch_size, enc_hid_dim]
        #print(embedded.shape,att_c.shape)      #torch.Size([1, 3, 256]) torch.Size([1, 1, 256])
        lstm_input = torch.cat((embedded,att_c), dim=2)   #lstm_input = [1, batch_size, enc_hid_dim + emb_dim]

        dec_output, (dec_h, dec_c) = self.lstm(lstm_input, (h, c))   
        # print(dec_output.shape)
        # print(att_c.shape)
        # print(embedded.shape)
        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # att_c = [batch_size, enc_hid_dim]
        # embedded = embedded
        # dec_output = dec_output
        # att_c = att_c
        # 把这3者拼接，然后送入最后的全连接层拟合预测结果，pred = [sequencelen，batch_size, output_dim]

        #pred = self.fc2(torch.cat((dec_output, att_c, embedded), dim = 2))
        pred = self.fc3(dec_output)
        return pred, dec_h,dec_c
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,totrain = True):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.totrain = totrain
    def forward(self, encoder_input, decoder_input,h,c):
        # encoder input = torch.randn(2,5,256)  batch_size 在0  5是用来预测的天数
        # deocder input = torch.randn(3,2,1)    batch_size 在1  3是未来预测的天数   采用产量数据
        
        #目前想法从x0-x4为encoder输入
        #x4,x5,x6的产量为decoder的输入，之后预测x5，x6，x7的产量，这是训练时的操作
        #验证的时候只需要给decoder输入x4的产量，用预测的x5产量与x6产量
        #print(decoder_input.shape)
        batch_size = encoder_input.shape[0]    
        decoder_future_num = decoder_input.shape[0] #预测的时间长度

        enc_outputs, h, c = self.encoder(encoder_input,h,c)
        #获得Encoder特征矩阵  torch.Size([5, 2, 256]) torch.Size([4, 2, 256]) torch.Size([4, 2, 256]) 四层LSTM
        
        #创建outputs张量存储decoder输出
        #outs = torch.zeros(decoder_future_num, batch_size, 1).to(device)
        outs = torch.zeros(decoder_future_num, batch_size, 1)     ####！！！注意在使用的时候加device放到gpu上！！！！！！！ 此处仅是测试代码
        dec_input = decoder_input[0,:,:].unsqueeze(0)
        #print("decoder_input",dec_input.shape)
        #用于存放预测结果
        for t in range(0, decoder_future_num):
            #dec_input = decoder_input[t,:,:]
            out, h, c = self.decoder(dec_input, h, c,enc_outputs)   
            outs[t] = out                
            
            
            #可能取真实值作为下一时刻的输入，也有可能取这一次的预测值作为下一时刻的输入
            dec_input = decoder_input[t,:,:].unsqueeze(0) if self.totrain else out
            
        return outs

if __name__ == "__main__":
    encoder = Encoder(256,256,4,0.1) 
    decoder = ATT_Decoder(256, 256, 4, Attention(256,256),256,0.1)
    # input = torch.randn(2,5,256)
    # enout , eh , ec = encoder(input)
    # decoder_input = torch.randn(1,2,1)    #2是batch_size
    # print(enout.shape, eh.shape, ec.shape)
    # deout , dh , dc = decoder(decoder_input,eh,ec,enout)
    # print(deout.shape)
    seq2seq = Seq2Seq(encoder,decoder)
    enin = torch.randn(2,5,256)
    dein = torch.randn(3,2,1)
    print(seq2seq)
    h,c = None,None
    t = dein[0,:,:].unsqueeze(0)
    # print(t.shape)

    seqouts =seq2seq(enin,dein,h,c)
    
    # print(seqouts.shape)