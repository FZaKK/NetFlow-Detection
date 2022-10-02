from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math

# 设置dataframe中的显示最大长度
pd.set_option('max_colwidth', 1000)

# log归一化（标准化）好像得自己写
minmax_standard = preprocessing.MinMaxScaler()
Zscore_standard = preprocessing.StandardScaler()


# min-max标准化，运用于无离群值
def minmax_scaler(data_col):
    temp = minmax_standard.fit_transform(data_col.values.reshape(-1, 1))
    return temp


# Z-score标准化
def Zscore_scaler(data_col):
    temp = Zscore_standard.fit_transform(data_col.values.reshape(-1, 1))
    return temp


# 对于数据值域差距过大的，使用log标准化的方式，log(value+1)
def log_scaler(data_col):
    temp = np.log(data_col.values + 1)
    return temp


def data_concat(malware_path, normal_path):
    # 以下部分用于将正常netflow和恶意netflow的csv放到一起
    malware_to_train = pd.read_csv(malware_path)
    normal_to_train = pd.read_csv(normal_path)

    # 打标签0，1
    malware_to_train['label'] = 1
    normal_to_train['label'] = 0
    trainset_df = pd.concat([malware_to_train, normal_to_train], axis=0)

    print(trainset_df)
    # trainset_df.to_csv('feature_dataset/data.csv', index=None)


def feature_data_preprocessing(data_path):
    # 对于FlowNo.特征的处理，因为这个是index标志，指示的是No序号,不进行处理
    # 先丢掉不需要的信息，如FlowNo和(srcip,srcport,des,desport)
    data = pd.read_csv(data_path)
    data = data.drop(columns=['FlowNo.', 'Destination', 'Destination Port', 'Source', 'Source Port'], axis=1)

    # 对于Answer RRS特征的处理，经过观察发现只有DNS，MDNS，LLMNR等格式拥有这个特征值
    # 填充Answer RRs字段，对于空值部分使用数据-1进行填充
    # 填充tcp segment len和udp len应该可以填充0，填充IP Flags为NAN
    encode_label = preprocessing.LabelEncoder()  # 对IP进行label编码，如果为空设置为-1
    encode_onehot = preprocessing.OneHotEncoder(sparse=False)

    data['Answer RRs'] = data['Answer RRs'].fillna(value=-1)
    data['IP_Flags'] = data['IP_Flags'].fillna(value='NAN')
    data['Next sequence number'] = data['Next sequence number'].fillna(value=0)
    data['Sequence number'] = data['Sequence number'].fillna(value=0)
    data['udp_Length'] = data['udp_Length'].fillna(value=0)
    data['TCP Segment Len'] = data['TCP Segment Len'].fillna(value=0)

    data['IP_Flags_Encoded'] = encode_label.fit_transform(data['IP_Flags'])
    data = data.drop(columns=['IP_Flags'], axis=1)
    data['Protocols_in_frame_Encoded'] = encode_label.fit_transform(data['Protocols in frame'])
    data = data.drop(columns=['Protocols in frame'], axis=1)

    # 处理protocal的独热编码
    proto = data['Protocol']
    proto_pre = encode_label.fit_transform(proto)
    proto_dict = {}
    for item in zip(proto, proto_pre):
        proto_dict[item[1]] = item[0]
    proto_array = encode_onehot.fit_transform(data[['Protocol']])
    proto_df = pd.DataFrame(proto_array)
    proto_df.columns = [proto_dict[i] for i in proto_df.columns]
    data = pd.concat([data, proto_df], axis=1)  # 两个dataframe连接
    data = data.drop(columns=['Protocol'], axis=1)

    label = data.pop('label')
    data.insert(loc=data.shape[1], column='label', value=label)

    # print(proto_df)
    # data.to_csv('feature_dataset/Normal_to_train.csv', index=None)
    # print(data['TCP Segment Len'])

    return data


def data_normalize(data_to_normalize_frame):
    # org_data = pd.read_csv('feature_dataset/data.csv')
    org_data = pd.DataFrame(data_to_normalize_frame)

    # 逐列进行数据处理
    org_data['Answer RRs'] = minmax_scaler(org_data['Answer RRs'])
    org_data['BytesEx'] = Zscore_scaler(org_data['BytesEx'])
    org_data['Duration'] = log_scaler(org_data['Duration'])
    org_data['FPL'] = Zscore_scaler(org_data['FPL'])
    org_data['Length'] = Zscore_scaler(org_data['Length'])
    org_data['Next sequence number'] = Zscore_scaler(org_data['Next sequence number'])
    org_data['No.'] = minmax_scaler(org_data['No.'])
    org_data['NumPackets'] = Zscore_scaler(org_data['NumPackets'])
    org_data['SameLenPktRatio'] = Zscore_scaler(org_data['SameLenPktRatio'])
    org_data['Sequence number'] = Zscore_scaler(org_data['Sequence number'])
    org_data['StdDevLen'] = Zscore_scaler(org_data['StdDevLen'])
    org_data['IAT'] = minmax_scaler(org_data['IAT'])
    org_data['reconnects'] = minmax_scaler(org_data['reconnects'])
    org_data['APL'] = Zscore_scaler(org_data['APL'])
    org_data['BitsPerSec'] = log_scaler(org_data['BitsPerSec'])
    org_data['AvgPktPerSec'] = Zscore_scaler(org_data['AvgPktPerSec'])
    org_data['udp_Length'] = Zscore_scaler(org_data['udp_Length'])
    org_data['tcp_Flags'] = Zscore_scaler(org_data['tcp_Flags'])
    org_data['Time'] = log_scaler(org_data['Time'])
    org_data['TCP Segment Len'] = Zscore_scaler(org_data['TCP Segment Len'])
    org_data['IOPR'] = Zscore_scaler(org_data['IOPR'])
    org_data['NumForward'] = Zscore_scaler(org_data['NumForward'])
    org_data['IP_Flags_Encoded'] = minmax_scaler(org_data['IP_Flags_Encoded'])
    org_data['Protocols_in_frame_Encoded'] = minmax_scaler(org_data['Protocols_in_frame_Encoded'])  # 这里可以想想到底用啥

    # print(org_data[0: 1].to_dict())
    # print(org_data)
    # org_data.to_csv('feature_dataset/dataset.csv', index=None)

    return org_data  # 返回Dataframe


def dataset_split(dataset):
    # 以下部分用于切分出具体的测试集和训练集
    # dataset = pd.read_csv('feature_dataset/dataset.csv')
    x_train, x_test = train_test_split(dataset, train_size=0.8)

    print(x_train)
    # x_train.to_csv('feature_dataset/train.csv', index=None)
    # x_test.to_csv('feature_dataset/test.csv', index=None)
    # print(type(x_train))


if __name__ == '__main__':
    # data_concat('feature_dataset/Data_Malware.csv', 'feature_dataset/Data_Normal.csv')
    data_to_normalize = feature_data_preprocessing('feature_dataset/Bidirectional_Botnet_all_features.csv')
    data_normalized = data_normalize(data_to_normalize)
    dataset_split(data_normalized)
    print('zzekun')

    # print(data_normalized[0: 1].to_dict())
    # print(data_to_normalize['Answer RRs'])

# 对于BytesEx的特征的处理，data数据包中的数量级是10-1000，目前使用的是z-socre,此时的z-score有正负数还有较大值如(4)
# 可以转移使用的z-score方法
# 如果出现缺值的情形，使用平均值填充BytesEx
# BytesEx_mean = data['BytesEx'].mean()
# data['BytesEx']= data['BytesEx'].fillna(BytesEx_mean)
# BytesEx_var = data['BytesEx'].var()  # 方差
# data['BytesEx'] = data['BytesEx'].apply(lambda x: (float(x) - BytesEx_mean) / math.sqrt(BytesEx_var)) # 需要除以标准差

# 将pandas里的某一属性列和index转化为dict形式
# my_dict = data[0: 1]['url'].to_dict()
# str_to_produce = my_dict[0]
# print(data)
