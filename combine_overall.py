import numpy as np
import pandas as pd

from_lgbm = np.load('y_lgmb.npy')

with_random_shuffle = np.load('with_random_shuffle.npy')

mixed = np.zeros(shape=np.shape(from_lgbm))

for i in range(len(from_lgbm)):
    mixed[i] = (46 * from_lgbm[i] + 9 * with_random_shuffle[i]) / 55

test = pd.read_csv('test.csv')

id_codes = test.loc[:, (test.columns == 'ID_code')]
output_df = pd.DataFrame(mixed, columns=['target'])
df_to_write = pd.concat([id_codes, output_df], axis=1, sort=False)
df_to_write.to_csv('submission4.csv', index=False)
