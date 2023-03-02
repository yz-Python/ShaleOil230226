#!/usr/bin/env python
# encoding: utf-8
import numpy as np
def mean_relative_error(predict, target):
    error = target-predict
    for i,j in  enumerate(error):  
        error[i] =abs(j)
    mean_error = sum(error/target)/len(error)
    return mean_error

if __name__ =='__main__':
    test=np.array([0,1,2,3,4])
    label=np.array([1,1,1,1,1])
    print(mean_relative_error(test,label))
   # torch.Size([4, 2, 256])是Encoder输出的h以及c，看看试试拉进这两个的距离，一般而言C是隐藏的，H就可以当做输出
   #还是说拉进Encoder的最后一层输出特征矩阵，也就是X的输出是[5,2,256] 5是time_delay
   #