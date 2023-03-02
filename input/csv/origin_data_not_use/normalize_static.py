import numpy as np
import pandas as pd
import os
def normalize(static_path):   
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
def select_static_data(dynamic_name_list,norm_static_file):
    static_data = []
    #print(norm_static_file)
    Data = pd.read_csv(norm_static_file)
    Data = Data.values
    feature = np.array(Data[:,:-1],dtype=float)
    name = Data[:,-1:].tolist()
    #print(name)
    try:
        for dynamic_name in dynamic_name_list:
            index = name.index([dynamic_name])
            static_data.append(feature[index:index+1,:])
        #print(static_data)
    except ValueError:
        print('{}井不匹配静态数据，检查是否确实静态数据或者井编号不对应'.format(dynamic_name))
    print('静态数据加载完成')
    return(static_data)


if __name__ =="__main__":
    a=normalize('input/csv/static/All_static.csv')
    select_static_data(['A1'],a)