from sklearn.decomposition import PCA
import pandas as pd
#file_name = "30-16faces.data.attr.csv"
#file_name = "faces.data.2020-06-18.csv"

file_name = "Random_First_Half_same_twarze_faces_data_network_OK.txt"
all_data = pd.read_csv("PCA_half/Random_First_Half_same_twarze_faces_data_network_OK.txt")
#all_data = pd.read_csv("24-3636faces.data.attr.csv")
#all_data = pd.read_csv("29-2128faces.data.attr.csv")

x = all_data.iloc[:, 3:150]
#x = all_data_to_cluster.iloc[0:100902, :]
all_data2 = pd.read_csv("PCA_half/Random_Second_Half_same_twarze_faces_data_network_OK.txt")
y = all_data2.iloc[:, 3:150]

pca = PCA(n_components=128, whiten = False)
principalComponents = pca.fit_transform(x)
print(x.shape)

print(principalComponents.shape)
print(pca.components_.shape)

principalComponents_pd = pd.DataFrame(principalComponents)
principalComponents_pd.insert(0,'Identity',all_data.iloc[:, 2])
principalComponents_pd.insert(0,'FileName',all_data.iloc[:, 1])

principalComponents_pd.to_csv("PCA_half/PCA_principalComponents_" + file_name)
components_pd = pd.DataFrame(pca.components_)
components_pd.to_csv("PCA_half/PCA_components_" + file_name)
singular_values_pd = pd.DataFrame(pca.singular_values_)
singular_values_pd.to_csv("PCA_half/PCA_singular_values_" + file_name)
mean_pd = pd.DataFrame(pca.mean_)
mean_pd.to_csv("PCA_half/PCA_mean_" + file_name)
explained_variance_ratio_pd = pd.DataFrame(pca.explained_variance_ratio_)
explained_variance_ratio_pd.to_csv("PCA_half/PCA_explained_variance_ratio_" + file_name)

vv = pca.explained_variance_ratio_
for a in range(1,len(vv)):
    vv[a] = vv[a - 1] + vv[a]

cummulative_explained_variance_ratio_pd = pd.DataFrame(vv)
cummulative_explained_variance_ratio_pd.to_csv("PCA_half/PCA_cummulative_explained_variance_ratio_pd_" + file_name)

yy = pca.transform(y)
yy_df = pd.DataFrame(yy)
#print(all_data.iloc[100902:all_data_to_cluster.shape[0], 1].shape)
#print(yy_df.shape)
#print(all_data_to_cluster.shape[0])
#print(all_data.iloc[100902:all_data_to_cluster.shape[0], 1])

yy_df.insert(0,'Identity',all_data2.iloc[:, 2])
yy_df.insert(0,'FileName',all_data2.iloc[:, 1])
'''
for a in range(100902):
    yy_df.iloc[a, 0] = all_data.iloc[100902 + a, 0]
    yy_df.iloc[a, 1] = all_data.iloc[100902 + a, 1]
'''
#yy_df.iloc[:,0] = all_data.iloc[100902:all_data_to_cluster.shape[0], 0]
#yy_df.iloc[:,1] = all_data.iloc[100902:all_data_to_cluster.shape[0], 1]

#for a in range(yy_df.shape[0]):


yy_df.to_csv("PCA_half/PCA_Transformed_Half_" + file_name)

'''
#0.8805904217932227
import hdbscan
from sklearn.metrics.cluster import homogeneity_score
#xx = all_data.iloc[:, 2:130]
xx = principalComponents_pd.iloc[:, 0:10]
clusterer = hdbscan.HDBSCAN(min_cluster_size = 2)
clusterin_labels = clusterer.fit_predict(xx)
hs = homogeneity_score(all_data.iloc[:, 1],clusterin_labels)
print(hs)

'''
#explained_variance_ratio_pd = pd.DataFrame(pca.explained_variance_ratio_)
#mean_pd.to_csv("PCA_mean_" + file_name)

'''
pd.DataFrame()
pd.to_csv
'''








'''
vv = pca.explained_variance_ratio_
vv1 = pca.explained_variance_ratio_
print(len(vv))
for a in range(1,len(vv)):
    vv1[a] = vv1[a - 1] + vv1[a]
    print(str(a) + " " + str(vv1[a]))
print(vv1)
'''
#principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])