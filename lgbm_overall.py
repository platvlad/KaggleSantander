import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


def separate_labels(df):
    df_without_labels = df.loc[:, (df.columns != 'target') & (df.columns != 'ID_code')]
    df_targets = df.loc[:, df.columns == 'target']
    df_data = np.array(df_without_labels.values)
    df_targets = np.array(df_targets.values)
    if len(df_targets[0]):
        df_targets = df_targets[..., 0]
    return df_data, df_targets


train = pd.read_csv('train.csv')
x_train, y_train = separate_labels(train)

test = pd.read_csv('test.csv')
x_test, y_test = separate_labels(test)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)
d_train = lgb.Dataset(x_train_scaled, label=y_train)

params = {}
params['learning_rate'] = 0.0006
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
params['boost_from_average'] = False
params['num_threads'] = 4
params['sub_feature'] = 0.5
params['num_leaves'] = 13
params['min_data'] = 80
params['min_sum_hessian_in_leaf'] = 10.0
params['max_depth'] = -1
params['feature_fraction'] = 0.041
params['bagging_freq'] = 5
params['bagging_fraction'] = 0.331
params['seed'] = 44000
clf = lgb.train(params, d_train, 128000)
y_test_score = clf.predict(x_test_scaled)

np.save('y_lgmb', y_test_score)

# id_codes = test.loc[:, (test.columns == 'ID_code')]
# output_df = pd.DataFrame(y_test_score, columns=['target'])
# df_to_write = pd.concat([id_codes, output_df], axis=1, sort=False)
# df_to_write.to_csv('submission.csv', index=False)

