from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.externals import joblib
import numpy as np
import os
import sys

def read_data(src):
    files = os.listdir(src)
    labels = []
    datas = []
    for i, file in enumerate(files):
        temp = os.path.join(src, file)        
        data = np.load(temp)        
        label = -1
        if file.find('cat') != -1:
            label = 0
        else:
            label = 1
        datas.append(data[0])
        labels.append(label)
    return datas, labels

if __name__ == '__main__':
    src = sys.argv[1]
    datas, labels = read_data(src)
    knn = KNeighborsClassifier(n_neighbors=7, algorithm="kd_tree").fit(datas, labels)
    save_path = "model/knn.joblib"
    joblib.dump(knn, save_path)
