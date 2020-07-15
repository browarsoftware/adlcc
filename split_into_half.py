from scipy.special._precompute.gammainc_asy import header
from sklearn.decomposition import PCA
import pandas as pd
#file_name = "30-16faces.data.attr.csv"
file_name = "same_twarze_faces_data_network_OK.txt"

all_data = pd.read_csv(file_name, header=None)
all_data = all_data.sample(frac=1).reset_index(drop=True)

x = all_data.iloc[0:100902, :]
y = all_data.iloc[100902:all_data.shape[0], :]
x.to_csv("PCA_half/Random_First_Half_" + file_name)
y.to_csv("PCA_half/Random_Second_Half_" + file_name)
