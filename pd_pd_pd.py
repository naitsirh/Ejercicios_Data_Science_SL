'''
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


scale = StandardScaler()
X = np.array([[0, 0], [2, 2]], dtype='float64')
scale.fit(X)
X_scaled = scale.transform(X)
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X_scaled)


first = np.array([[0, 0]])
second = np.array([[2, 2]])


for i in range(int(input())):
	lst = [input().split()]
	X_new = np.array(lst, dtype='float64')
	X_new_scaled = scale.transform(X_new)
	p = kmeans.predict(X_new_scaled)
	if 0 in p:
		first = np.concatenate((first, X_new), axis=0)
	elif 1 in p:
		second = np.concatenate((second, X_new), axis=0)


if len(first) == 1:
	print('None')
else:
	F = first[1:,].mean(axis=0)
	print(np.around(F, 2))


if len(second) == 1:
	print('None')
else:
	S = second[1:,].mean(axis=0)
	print(np.around(S, 2))
'''



'''
import numpy as np
from sklearn.cluster import KMeans


n = int(input())
list1 = []
list2 = []
matrix = []

for i in range(n):
	matrix.append(input().split())


X = np.array([[0, 0], [2, 2]], dtype='float64')
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
j = 0


for i in kmeans.predict(matrix):
	if i == 0:
		list1.append(matrix[j])
	if i == 1:
		list2.append(matrix[j])
	j+=1


res1 = np.round(np.array(list1).astype(float), 2)
res2 = np.round(np.array(list2).astype(float), 2)  


if len(list1) == 0:
	print('None')
else: 
	kmeans1 = KMeans(n_clusters=1, random_state=0).fit(res1)
	print(np.round(np.array(kmeans1.cluster_centers_[0]).astype(float), 2))

if len(list2) == 0:
	print('None')
else:
	kmeans2 = KMeans(n_clusters=1, random_state=0).fit(res2)
	print(np.round(np.array(kmeans2.cluster_centers_[0]).astype(float), 2))
'''




import numpy as np


first = np.array([[0., 0.]])
second = np.array([[2., 2.]])
n = int(input())

data = []

for i in range(n):
	data.append([float(i) for i in input().split()])


data = np.array(data).reshape((-1,2))


for i in range(n):
	dist1 = np.sqrt(((data[i]-first[0])**2).sum())
	dist2 = np.sqrt(((data[i]-second[0])**2).sum())

	if (dist1) <= (dist2):
		first = np.vstack((first,data[i]))
	else:
		second = np.vstack((second,data[i]))


if first.size > 2:
	mean1 = np.mean(first[1:], axis=0)
	print(np.around(mean1, decimals=2))
else:
	print(None)


if second.size > 2:
	mean2 = np.mean(second[1:], axis=0)
	print(np.around(mean2, decimals=2))
else:
	print(None)