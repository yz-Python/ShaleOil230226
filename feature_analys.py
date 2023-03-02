import pandas as pd
'''
输入文件：excel表格
输出文件：各个特征与日产油量的关系图
'''
print('ssssss')
def extractData(path):
    f = pd.ExcelFile(path)
    sheets = f.sheet_names
    rst = []
    columns = []
    for name in sheets:
        data = pd.read_excel(path, name)
        count = 0
        for i in data.index:
            if count == 0:
                columns.extend(list(data.loc[i]))
                count = count + 1
            else:
                rst.append(list(data.loc[i]))
                count = count + 1
    data = pd.DataFrame(rst, columns=columns)
    data2 = data.loc[:, ['油嘴', '油压', '井口温度', '日产液量']]
    data2 = data2.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    data2 = data2.iloc[::-1]
    return data2

def showDirs(sourceDir, desDir, path):
    import matplotlib.pyplot as plt
    data = extractData(sourceDir + path)
    X = data.shape[0]
    X = list(range(X))
    Y1 = data['油嘴'].values
    Y2 = data['油压'].values
    Y3 = data['井口温度'].values
    Y4 = data['日产液量'].values

    plt.plot(X, Y1, color="b", linestyle ='--', label="nozzle")
    plt.plot(X, Y2, color="g", linestyle ='--', label="pressure")
    plt.plot(X, Y3, color="y", linestyle ='--', label="temperature")
    plt.plot(X, Y4, color="r", linestyle ='-', label="output")
    plt.xlabel("day")
    plt.ylabel("value")
    (filename, extension) = os.path.splitext(file)
    plt.title(filename + " feature analysis")
    print(desDir, filename)
    plt.legend()
    despath = os.path.join(desDir, filename + ".jpg")
    plt.savefig(despath) # 保存图片
    plt.close()


def show(data):
    import matplotlib.pyplot as plt
    X = data.shape[0]
    X = list(range(X))
    Y1 = data['油嘴'].values
    Y2 = data['油压'].values
    Y3 = data['井口温度'].values
    Y4 = data['日产液量'].values

    plt.plot(X, Y1, color="b", linestyle ='--', label="nozzle")
    plt.plot(X, Y2, color="g", linestyle ='--', label="pressure")
    plt.plot(X, Y3, color="y", linestyle ='--', label="temperature")
    plt.plot(X, Y4, color="r", linestyle ='-', label="output")
    plt.xlabel("day")
    plt.ylabel("value")
    plt.title("%s feature analysis"%())
    plt.show()

if __name__ == "__main__":
    import os
    sourceDir = "../input/excel/"
    desDir = "../input/feature/pic/"
    infors = list(os.walk(sourceDir))
    root = infors[0][0]
    files = infors[0][-1]
    for file in files:
        showDirs(sourceDir, desDir, file)
