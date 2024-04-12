from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("CC_GENERAL.csv")
df = df.fillna(0)


x = df.iloc[:, 1:17]
y = df.iloc[:, -1]
print(x.shape)
print(x.head)
#normalize data
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)

#adjust data with principle components
pca = PCA(2)
principalComponents = pca.fit_transform(x_scale)
principalDf = pd.DataFrame(principalComponents, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['CUST_ID']]], axis=1)


#make prediction with origional data set
km = KMeans(3)
km.fit(x)

y_cluster_kmeans = km.predict(x)
s_score = metrics.silhouette_score(x, y_cluster_kmeans)
print('Score for our Training dataset without PCA is: %.4f ' % s_score)


#make prediction with pca data set
final_x = finalDf.iloc[:, 0:2]

print(final_x.head())


km_wpca = KMeans(3)
km_wpca.fit(final_x)

y_cluster_kmeans2 = km_wpca.predict(final_x)

s_score = metrics.silhouette_score(final_x, y_cluster_kmeans2)
print('Score for our Training dataset with PCA is: %.4f ' % s_score)




