"""
author: dcy@nankai.edu.cn
"""
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import pickle
import warnings
import dpkt

warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # sklearn.svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# protocol_dict = {"tcp": 1, "udp": 2, "icmp": 3}
# service_dict = {"http": 1, "ecr_i": 2, "private": 3}
# flag_dict = {"SF": 1, "S0": 2, "REJ": 3}
class NetWorkMultiModelDetection:

    def __init__(self, already_trained=False):
        """
        初始化网络数据检测类
        :param if_trained:
        """
        # self._MODELLIST = ["CBLOF", "HBOS", "PCA", "MCD", "IForest", "LODA", "LOF", "KNN"]
        self._MODELLIST = ["KNN", "SVM", "DT", "RF", "ADA", "NB", "MLP"]
        # self._DATAPATH = "dataset/"
        # self._MODELPATH = "netdata/model/"
        self._DATAPATH = "feature_dataset/"
        self._MODELPATH = "netdata/my_model/"

        # self.standarscaler = StandardScaler()
        self.KNN_clf = KNeighborsClassifier()
        self.SVM_clf = SVC(probability=True)
        self.GP_clf = GaussianProcessClassifier()
        self.DT_clf = DecisionTreeClassifier()
        self.RF_clf = RandomForestClassifier()
        self.ADA_clf = AdaBoostClassifier()
        self.NB_clf = GaussianNB()
        self.MLP_clf = MLPClassifier()

        if already_trained:
            self._load_models()

    def _load_models(self):
        """
        将模型文件和归一化尺度读取到内存中
        :return:
        """
        # self.standarscaler = pickle.load(open(r"{}standardscalar.pkl".format(self._MODELPATH), 'rb'))

        self.KNN_clf = pickle.load(open("{}KNN_model.pkl".format(self._MODELPATH), 'rb'))
        self.SVM_clf = pickle.load(open("{}SVM_model.pkl".format(self._MODELPATH), 'rb'))
        self.DT_clf = pickle.load(open("{}DT_model.pkl".format(self._MODELPATH), 'rb'))
        self.RF_clf = pickle.load(open("{}RF_model.pkl".format(self._MODELPATH), 'rb'))
        self.ADA_clf = pickle.load(open("{}ADA_model.pkl".format(self._MODELPATH), 'rb'))
        self.NB_clf = pickle.load(open("{}NB_model.pkl".format(self._MODELPATH), 'rb'))
        self.MLP_clf = pickle.load(open("{}MLP_model.pkl".format(self._MODELPATH), 'rb'))

    def multimodel_train(self):
        # total_data = pd.read_csv(self._DATAPATH + "trainset.csv")
        total_data = pd.read_csv(self._DATAPATH + "train.csv")
        x_train = total_data.drop(columns=["label"], axis=1)  # 训练去除无用列名
        y_train = total_data["label"]
        # self.standarscaler.fit(x_train.values)
        # x_train = pd.DataFrame(self.standarscaler.transform(x_train), index=x_train.index, columns=x_train.columns)
        # pickle.dump(self.standarscaler, open(r"{}standardscalar.pkl".format(self._MODELPATH), 'wb'))

        for model in self._MODELLIST:
            print("{} begin!".format(model))

            clf = eval("self." + model + "_clf").fit(x_train, y_train.values)
            y_train_scores = np.array(clf.predict_proba(x_train))[:, 1]  # ???
            y_train_scores = sorted(y_train_scores)
            np.save(r"{}{}_train_scores".format(self._MODELPATH, model), y_train_scores)
            pickle.dump(clf, open("{}{}_model.pkl".format(self._MODELPATH, model), 'wb'))

            print("{} finished!".format(model))

    def multimodel_predict(self, networkdata):
        # testdata = self.standarscaler.transform(pd.DataFrame(networkdata))
        testdata = pd.DataFrame(networkdata)
        results = dict()
        for model in self._MODELLIST:
            results[model] = eval('self.' + model + '_clf').predict(testdata)
        df_rs = pd.DataFrame(results)
        df_rs["result"] = df_rs.apply(lambda x: 0 if x.sum() <= 3 else 1, axis=1)
        return df_rs


# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    # test = pd.read_csv("dataset/testset.csv")
    # test_x = test.drop(columns = ["label", "label_number"], axis=1)
    # test_y = test["label_number"]

    # 用于测试集判断效果
    test = pd.read_csv("feature_dataset/test.csv")
    test_x = test.drop(columns=["label"], axis=1)
    test_y = test["label"]
    # md = NetWorkMultiModelDetection()
    # md.multimodel_train()
    md = NetWorkMultiModelDetection(already_trained=True)
    rs = md.multimodel_predict(test_x)["result"]

    '''
    # 实际网络环境NKU下的检测
    test = pd.read_csv("feature_dataset/real_test.csv")

    md = NetWorkMultiModelDetection(already_trained=True)
    rs = md.multimodel_predict(test)["result"]
    print(rs)
    '''

    print("accuracy:\t", accuracy_score(test_y, rs))
    print("precision:\t", precision_score(test_y, rs))
    print("recall:\t", recall_score(test_y, rs))
    print("f1_score:\t", f1_score(test_y, rs))

    # total_data = pd.read_csv("dataset/train.csv")
    # total_data['label_number'] = total_data.label.apply(lambda x: 0 if x == "normal." else 1)
    # total_data['protocol_type'] = total_data.protocol_type.apply(lambda x: protocol_dict[x] if x in protocol_dict.keys() else 0)
    # total_data['service'] = total_data.service.apply(lambda x: service_dict[x] if x in service_dict.keys() else 0)
    # total_data['flag'] = total_data.flag.apply(lambda x: flag_dict[x] if x in flag_dict.keys() else 0)
    #
    # total_data.to_csv("dataset/trainset.csv", index=None)
