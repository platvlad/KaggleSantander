import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from sklearn.metrics import roc_auc_score


def separate_labels(df):
    df_without_labels = df.loc[:, (df.columns != 'target') & (df.columns != 'ID_code')]
    df_targets = df.loc[:, df.columns == 'target']
    df_data = np.array(df_without_labels.values)
    df_targets = np.array(df_targets.values)
    if len(df_targets[0]):
        df_targets = df_targets[..., 0]
    return df_data, df_targets


dataset = pd.read_csv('train.csv')
data, targets = separate_labels(dataset)

test_dataset = pd.read_csv('test.csv')
test_data, test_targets = separate_labels(test_dataset)

# x_train = data[:160000]
# y_train = targets[:160000]
# x_test = data[160000:]
# y_test = targets[160000:]

x_train = data
y_train = targets
x_test = test_data
y_test = test_targets

positive = np.where(y_train == 1)
negative = np.where(y_train == 0)
x_positive = x_train[positive]
x_negative = x_train[negative]
for i in range(len(x_positive[0])):
    np.random.shuffle(x_positive[..., i])
    np.random.shuffle(x_negative[..., 1])
x_train = np.concatenate((x_positive, x_negative))
y_train = np.concatenate((np.ones(shape=(len(x_positive),)), np.zeros(shape=(len(x_negative),))))

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

d_train = lgb.Dataset(x_train_scaled, label=y_train)

params = {}
params['learning_rate'] = 0.044
params['num_threads'] = 4
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
params['boost_from_average'] = False
params['sub_feature'] = 0.5
params['num_leaves'] = 3
params['min_data'] = 80
# params['min_sum_hessian_in_leaf'] = 10.0
params['max_depth'] = 2
params['feature_fraction'] = 0.041
params['bagging_freq'] = 5
params['bagging_fraction'] = 0.331
params['seed'] = 44000

clf = lgb.train(params, d_train, 32000)

y_test_score = clf.predict(x_test_scaled)

np.save('with_random_shuffle', y_test_score)

# roc_auc = roc_auc_score(y_test, y_test_score)
# print(roc_auc)
