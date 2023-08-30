import numpy as np
import pickle
def jaccard_distance(label1, label2):
    intersection = np.logical_and(label1, label2).sum()
    union = np.logical_or(label1, label2).sum()
    jaccard = 1 - intersection / union
    return jaccard

def greedy_label_merge(labels):
    num_labels = labels.shape[1]
    merged_labels = labels.copy()  # 复制标签列表，避免修改原始标签
    merge_history = []
    while num_labels > 1:

        for i in range(num_labels):
            min_distance = float('inf')
            merge_indices = None
            for j in range(i + 1, num_labels):
                distance = jaccard_distance(labels[i], labels[j])
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)
                if j==num_labels-1:
                    merge_history.append(merge_indices)

        if merge_indices is not None:
            label1, label2 = merge_indices
            merged_labels[label1] = np.logical_or(labels[label1], labels[label2])
            merged_labels = np.delete(merged_labels, label2, axis=0)
            num_labels -= 1

    return merged_labels

# 示例数据
label = np.load('../../data/EUR11585/fre_train_y.npy')
num_labels = label.shape[1]

# 使用第一个标签作为初始合并标签
# merged_labels = greedy_label_merge(label)
#
# print("Final merged labels:")
# print(merged_labels)
# 计算两两标签之间的Jaccard距离
from sklearn.cluster import AgglomerativeClustering
distances = np.zeros((num_labels, num_labels))
for i in range(num_labels):
    for j in range(i + 1, num_labels):
        distance = jaccard_distance(label[i], label[j])
        distances[i][j] = distance


# 创建聚类器并进行聚类
# , distance_threshold=0.65
clustering = AgglomerativeClustering(n_clusters=200, affinity='precomputed', linkage='average', distance_threshold=None)
clustering.fit(distances)

# 获取聚类结果
labels = clustering.labels_
num_clusters = len(np.unique(labels))

# 查看每个簇中的标签
clusters = {}
for i in range(num_clusters):
    clusters[i] = []

for i, label_idx in enumerate(labels):
    clusters[label_idx].append(i)

merge_id = []
# 打印每个簇中的标签
for cluster_id, cluster_labels in clusters.items():
    # print("Cluster", cluster_id)
    # print("Labels:", cluster_labels)
    merge_id.append(cluster_labels)


n_rows = label.shape[0]
n_cols = len(merge_id)

merge_train_y = np.zeros((n_rows, n_cols))

for idx, pairs in enumerate(merge_id):
    for row in range(label.shape[0]):
        for pair in pairs:
            if label[row, pair] == 1:
                merge_train_y[row, idx] = 1
                break

label_tst = np.load('../../data/EUR11585/fre_test_y.npy')
n_rows = label_tst.shape[0]
n_cols = len(merge_id)

merge_test_y = np.zeros((n_rows, n_cols))
for idx, pairs in enumerate(merge_id):
    for row in range(label_tst.shape[0]):
        for pair in pairs:
            if label_tst[row, pair] == 1:
                merge_test_y[row, idx] = 1
                break
np.save('merge_test_y.npy', merge_test_y)

# print(merge_train_y)
# with open('merge_id.pickle', 'wb') as f:
#     pickle.dump(merge_id, f)
# np.save('merge_train_y.npy', merge_train_y)
# with open('merge_id.pickle', 'rb') as file:
#     label_set = pickle.load(file)
# x = np.load('merge_train_y.npy')
print('done')

# import numpy as np
#
# def jaccard_distance(label1, label2):
#     intersection = np.logical_and(label1, label2).sum()
#     union = np.logical_or(label1, label2).sum()
#     jaccard = 1 - intersection / union
#     return jaccard
#
# # label = np.array([[1, 1, 0, 0, 0],
# #                   [1, 1, 0, 1, 0],
# #                   [1, 0, 0, 0, 1],
# #                   [0, 1, 0, 1, 0],
# #                   [0, 0, 1, 1, 0],
# #                   [1, 1, 0, 0, 1]])
# #
# count = 0
# label = np.load('train_y.npy')
# # for i in range(746):
# #     if i == 745:
# #         break
# #     label0 = label[i]
# #     label1 = label[i+1]
# #     jaccard = jaccard_distance(label0, label1)
# #     if jaccard!=1:
# #         count+=1
# #     print("Jaccard Distance between label0 and label1:", jaccard)
# # print(count)
# for i in range(746):
#
#     for j in range(i + 1, 746):
#         label1 = label[i]
#         label2 = label[j]
#         jaccard = jaccard_distance(label1, label2)
#         if jaccard!=1:
#             count+=1
#         print(f"Jaccard Distance between label {i} and label {j}: {jaccard}")
# print(count)


# import pickle
# import numpy as np
# with open('../../data/EUR/label_set.pickle', 'rb') as file:
#     label_set = pickle.load(file)
# with open('../../data/EUR/text_set.pickle', 'rb') as file:
#     text_set = pickle.load(file)
#
# train_doc = text_set['train']
# test_doc = text_set['test']
#
# train_y = label_set['train']
# test_y = label_set['test']
#
# #
# num_labels = train_y.shape[1]
#
#
#
# # 统计每个标签的频率
# label_frequencies = np.sum(train_y, axis=0)
# sorted_indices = np.argsort(label_frequencies)[::-1]
# sorted_frequencies = label_frequencies[sorted_indices]
# sorted_labels = train_y[:, sorted_indices]
#
# count = 0
# frequent_index = []
# # 打印每个标签的频率
# for i in range(num_labels):
#     # print(f"标签 {sorted_indices[i]+1} 的频率：{sorted_frequencies[i]}")
#     count+=1
#     if count<=746:
#         frequent_index.append(sorted_indices[i])
