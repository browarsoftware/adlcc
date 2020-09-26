#accuracy calculation
ff = read.csv('d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_face_recognition\\Clustering_OK\\Supervised\\NN_1024Random_First_Half_same_twarze_faces_data_network_OK.txt_.csv', sep=',', header = FALSE)
#ff = read.csv('d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_face_recognition\\Clustering_OK\\Supervised\\SVM_Random_First_Half_same_twarze_faces_data_network_OK.txt_.csv', sep=',', header = FALSE)
#ff = read.csv('d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_face_recognition\\Clustering_OK\\Supervised\\KNN_Random_First_Half_same_twarze_faces_data_network_OK.txt_.csv', sep=',', header = FALSE)


#ff = read.csv('d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_face_recognition\\Clustering_OK\\Supervised\\NN_PCA_1024PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt_.csv', sep=',', header = FALSE)
#ff = read.csv('d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_face_recognition\\Clustering_OK\\Supervised\\KNN_PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt_.csv', sep=',', header = FALSE)
#ff = read.csv('d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_face_recognition\\Clustering_OK\\Supervised\\SVM_PCAPCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt_.csv', sep=',', header = FALSE)

cc = 0
for (a in 1:length(ff$V2))
{
  if (ff$V2[a] == ff$V3[a])
    cc = cc + 1
}
cc / length(ff$V2)