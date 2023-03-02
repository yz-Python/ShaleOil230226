import numpy as np
def normalize(x,time_delay,predict_day,paras = False,predict = False):
    if x.dtype != 'float32' :
         x= x.astype(np.float32)
    if predict == False:
        assert time_delay+predict_day == x.shape[1]
        encoder = x[:,:time_delay,:]
        decoder = x[:,time_delay:time_delay+predict_day,:]
        #print(encoder)
        B,leng,feat = encoder.shape 
        encoder = encoder.reshape(-1,feat)
        en_mean = np.mean(encoder,axis=0)
        en_std = np.std(encoder,axis=0)#.reshape(B,1,feat).repeat(leng,axis = 1)
        encoder = (encoder-en_mean)/en_std
        encoder = encoder.reshape(B,leng,feat)
        #print(en_mean)
        #print('std:',en_std)
        B,leng,feat = decoder.shape
        decoder = decoder.reshape(-1,feat)
        if predict_day !=1:   
            de_mean = np.mean(decoder,axis=0)#.reshape(B,1,feat).repeat(leng,axis = 1)
            de_std = np.std(decoder,axis=0)#.reshape(B,1,feat).repeat(leng,axis = 1)
            decoder = (decoder-de_mean)/de_std
            decoder = decoder.reshape(B,leng,feat)
        elif predict_day ==1:
            de_mean = en_mean
            de_std = en_std
            decoder = (decoder-de_mean)/de_std
            decoder = decoder.reshape(B,leng,feat)
            
        x[:,:time_delay,:] = encoder
        x[:,time_delay:time_delay+predict_day,:] = decoder
        if paras == True:
            mean = en_mean
            std = en_std
            return(x,mean,std)
        return(x)
    if predict == True:
        #assert time_delay == x.shape[1]
        encoder = x 
        B,leng,feat = encoder.shape 
        
        encoder = encoder.reshape(-1,feat)
        en_mean = np.mean(encoder,axis=0)
        en_std = np.std(encoder,axis=0)#.reshape(B,1,feat).repeat(leng,axis = 1)
        encoder = (encoder-en_mean)/en_std
        x = encoder.reshape(B,leng,feat)
        if paras == True:
            mean = en_mean
            std = en_std
            
            return(x,mean,std)
        return(x)
    
if __name__ == '__main__':
    time_delay = 3
    predict_day =1
    #x = np.randn(100,8,3)
    y = np.array([[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[0.,1.,2.]],[[7.,8.,9.],[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]])
    #y = y.reshape(-1,3)
    #print(y)
    #y=y.reshape(2,4,3)
    #print(y)
    # mean = np.mean(y,axis=1).reshape(2,1,3)
    # mean = mean.repeat(4,axis = 1)
    # print(mean)
    # std = np.std(y,axis=1)
    # y = (y-mean)/std
    # print(mean.shape)
    # print(y.shape)
    # print(1)
    x,mean,std = normalize(y,time_delay,predict_day,True)
    print(x,'\n',mean,'\n',std)