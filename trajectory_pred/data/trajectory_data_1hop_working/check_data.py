import numpy as np
import pandas as pd

a = np.load('trajectory_data_1hop_working/B1.npy')
print(a.shape)
DF = pd.DataFrame(np.squeeze(a))
print(DF)
# # # save the dataframe as a csv file
DF.to_csv("trajectory_data_1hop_working/a.csv")