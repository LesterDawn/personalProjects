import random
import pandas as pd

import numpy as np

data = [[1, 2, 3, 4, 'i'], [2, 3, 4, 5, 'i'], [3, 4, 5, 6, 'i'],
        [4, 5, 6, 7, 'i'], [5, 6, 7, 8, 'i']]


def Euclidean_dist(data1, data2):
    result = 0
    for i in range(0, 4):
        result = result + ((data1[i] - data2[i]) ** 2)
    return np.sqrt(result)


def select_centroid(dataset, k):
    return random.sample(dataset, k)


def dataframe_construct(dataset):
    return pd.DataFrame([data.append(-1) for data in dataset])


def k_means(dataset, k):
    centroids = select_centroid(dataset, k)
    changed_or_not = True  # Flag that indicates the centroid changed or not
    data_cluster = dataframe_construct(dataset)

    while changed_or_not:
        changed_or_not = False
        # Assign each sample to the cluster with the nearest mean point
        for i in range(len(dataset)):
            minDist = 0
            clusterID = -1
            for j in range(k):
                distance = Euclidean_dist(dataset[i], centroids[j])
                if distance < minDist:
                    minDist = distance
                    clusterID = j
            if data_cluster[i][1] != clusterID:
                changed_or_not = True
            data_cluster[i][1] = clusterID

        # Compute mean points of the clusters
        for i in range(k):
            cluster = df[df[1] == i][0].tolist()
            for lst in cluster:
                lst.pop()
            centroids[i] = np.mean(cluster, axis=0)

    return centroids, data_cluster


# df = dataframe_construct(data)
# df[1].iloc[0] = 1
# df[1].iloc[1] = 2
# df[1].iloc[2] = 3
# df[1].iloc[3] = 1
# df[1].iloc[4] = 1
# centroids = select_centroid(data, 3)


# for i in range(3):
#     cluster = df[df[1] == i][0].tolist()
#     for lst in cluster:
#         lst.pop()
#     centroids[i] = np.mean(cluster, axis=0)
# print(centroids)
def get_attribute(lst):
    temp = []
    for ele in lst:
        temp.append([ele[0], ele[1]])
    return np.array(temp)


def centroid_cal(cluster):
    temp1 = []
    temp2 = []
    for lst in cluster:
        for i in range(len(lst) - 1):
            temp1.append(lst[i])
        temp2.append(temp1)
        temp1 = []
    return np.mean(temp2, axis=0)


print(dataframe_construct(data))
narr = np.zeros(shape=(1, 3))
