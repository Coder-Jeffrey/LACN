import numpy as np
import pickle
def jaccard_distance(label1, label2):
    intersection = np.logical_and(label1, label2).sum()
    union = np.logical_or(label1, label2).sum()
    jaccard = intersection / union
    # if jaccard>=0.25:
    #     print('--------------------')
    #     print(intersection)
    #     print(union)
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
        distance = jaccard_distance(label[:,i], label[:,j])
        distances[i][j] = distance
        distances[j][i] = distance

# np.save('Jaccard_sim.npy', distances)
d = distances[0][6]
with open('../../data/EUR11585/frequent_index.pickle', 'rb') as file:
    fre_index = pickle.load(file)


# def merge_labels(distances, fre_index):
#     merged_labels = []  # 用于记录合并标签的下标
#     record = []
#     for label in range(len(fre_index)):
#         if label in record:
#             continue
#         min_distance = float('inf')
#         min_distance_label = None
#         temp = []
#         temp.append(label)
#         for i in range(len(fre_index)):
#             if i in record:
#                 continue
#             if i != label:
#                 distance = distances[label][i]
#                 if distance==1.0:
#                     continue
#                 if distance < min_distance and distance<=0.75:
#                     min_distance = distance
#                     min_distance_label = i
#         if min_distance_label is not None:
#             temp.append(min_distance_label)
#         for v in temp:
#             record.append(v)
#         merged_labels.append(temp)
#
#     return merged_labels
def merge_labels(distances, fre_index):
    merged_labels = []  # 用于记录合并标签的下标
    record = []
    for label in range(len(fre_index)):
        if label in record:
            continue
        min_distance = float('-inf')
        min_distance_label = None
        temp = []
        temp.append(label)
        for i in range(len(fre_index)):
            if i in record:
                continue
            if i != label:
                distance = distances[label][i]
                if distance==1.0:
                    continue
                if distance > min_distance and distance>=0.1:
                    min_distance = distance
                    min_distance_label = i
        if min_distance_label is not None:
            temp.append(min_distance_label)
        for v in temp:
            record.append(v)
        merged_labels.append(temp)

    return merged_labels
merge_id = merge_labels(distances, fre_index)
label_trn = np.load('../../data/EUR11585/fre_train_y.npy')
n_rows = label_trn.shape[0]
n_cols = len(merge_id)
merge_train_y = np.zeros((n_rows, n_cols))
for idx, pairs in enumerate(merge_id):
    for row in range(label_trn.shape[0]):
        for pair in pairs:
            if label_trn[row, pair] == 1:
                merge_train_y[row, idx] = 1
                break
np.save('merge_train_y_greedy_01.npy', merge_train_y)

merge_id = merge_labels(distances, fre_index)
# print(merge_id)
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
np.save('merge_test_y_greedy_01.npy', merge_test_y)
with open('merge_id_greedy_01.pickle', 'wb') as f:
    pickle.dump(merge_id, f)

print('d')

# testtttt
# merge_id = [[1,3],[2,7],[0],[4],[5],[6]]
# label_tst = np.array([[0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,0]])
# n_rows = label_tst.shape[0]
# n_cols = len(merge_id)
# merge_test_y = np.zeros((n_rows, n_cols))
# for idx, pairs in enumerate(merge_id):
#     for row in range(label_tst.shape[0]):
#         for pair in pairs:
#             if label_tst[row, pair] == 1:
#                 merge_test_y[row, idx] = 1
#                 break
# # np.save('merge_test_y_greedy_251.npy', merge_test_y)
# print(merge_test_y)
# print('')


# label_tst = np.load('../../data/EUR11585/fre_test_y.npy')
# n_rows = label_tst.shape[0]
# n_cols = len(merge_id)
#
# merge_test_y = np.zeros((n_rows, n_cols))
