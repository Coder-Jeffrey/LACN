import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules,fpgrowth
import pickle

# with open('../../data/EUR11585/frequent_index.pickle', 'rb') as file:
#     fre_index = pickle.load(file)

# label = np.load('./EUR/train_label.npy')
label = np.load('../../data/EUR11585/Y_trn.npy')

# 获取高频标签的下标
label_frequencies = np.sum(label, axis=0)
sorted_indices = np.argsort(label_frequencies)[::-1]
count = 0
frequent_index = []
# 打印每个标签的频率
for i in range(label.shape[1]):
    # print(f"标签 {sorted_indices[i]+1} 的频率：{sorted_frequencies[i]}")
    count += 1
    if count <= 293:
        frequent_index.append(sorted_indices[i])

# 挖掘标签联系
label_df = pd.DataFrame(label)
frequent_itemsets = fpgrowth(label_df, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
# 根据"LHS"和"RHS"列创建一个唯一的标识符，然后使用集合(set)来去除重复规则
rules['unique_rule'] = rules.apply(lambda row: frozenset([row['antecedents'], row['consequents']]), axis=1)
unique_rules = rules.drop_duplicates(subset='unique_rule')

# 移除辅助列
unique_rules = unique_rules.drop(columns=['unique_rule'])

# 打印处理后的关联规则
print(unique_rules)
# fset = rules['antecedents'][24]
# for element in fset:
#     print(element)

# # 过滤，只要前提是高频标签的规则
# filtered_rules = rules[
#     rules['antecedents'].apply(lambda x: all(item in frequent_index for item in x))
# ]
# print(filtered_rules['antecedents'])
x = rules['antecedents']
y = rules['consequents']
z = rules['zhangs_metric']

confidence = []

l = len(z)
for i in range(len(z)):
    confidence.append(z[i])
    # print(z[i])
# with open('confidence_s0.0044.pickle', 'wb') as f:
#     pickle.dump(cc, f)

# l = len(x)

antecedents = []
consequents = []

for i in range(len(x)):
    temp = []
    for e in x[i]:
        temp.append(e)
    antecedents.append(temp)
for i in range(len(y)):
    temp = []
    for e in y[i]:
        temp.append(e)
    consequents.append(temp)

count = 0
filter_antecedent = []
filter_consequent = []
filter_confidence = []
for i in range(len(consequents)):
    if set(consequents[i]).issubset(frequent_index):
        # print(consequent)
        count+=1
    else:
        temp = []
        for v in consequents[i]:
            if v not in frequent_index:
                temp.append(v)
        filter_antecedent.append(antecedents[i])
        filter_consequent.append(temp)
        filter_confidence.append(confidence[i])


unique_data = [list(sublist) for sublist in set(tuple(sublist) for sublist in filter_consequent)]

print(unique_data)
print(len(unique_data))

flat_list = [item for sublist in unique_data for item in sublist]
# 使用集合(set)去除重复元素
result_list = list(set(flat_list))
print(result_list)

new_antecedents = []
new_consequents = []
new_confidences = []

for item in result_list:
    max_confidence = -1
    max_index = -1

    for i, consequent_list in enumerate(filter_consequent):
        if item in consequent_list:
            confidence = filter_confidence[i]
            if confidence > max_confidence:
                max_confidence = confidence
                max_index = i

    if max_index != -1 and max_confidence==filter_confidence[max_index]:
        new_antecedents.append(filter_antecedent[max_index])
        new_consequents.append(item)
        new_confidences.append(filter_confidence[max_index])

# for v in new_consequents:
#     if v in frequent_index:
#         print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
print('')
# print(count)
# print(frequent_index)
with open('antecedents_s0.0015.pickle', 'wb') as f:
    pickle.dump(new_antecedents, f)

with open('consequents_s0.0015.pickle', 'wb') as f:
    pickle.dump(new_consequents, f)

with open('confidences_s0.0015.pickle', 'wb') as f:
    pickle.dump(new_confidences, f)
# with open('confidences_s0.00001.pickle', 'wb') as f:
#     pickle.dump(new_confidences, f)

print('done')
# with open('antecedents_s0.0044.pickle', 'wb') as f:
#     pickle.dump(antecedents, f)
# with open('consequents_s0.0044.pickle', 'wb') as f:
#     pickle.dump(consequents, f)
# print(next(iter(x[0])))

# out = np.array([0., 0., 1., ..., 1., 0.])
#
# for antecedent, consequent in zip(antecedents, consequents):
#     # 检查antecedent中的值是否都在out的下标中，并且对应的out的值都为1.0
#     if all(out[value] == 1.0 for value in antecedent):
#         # 更新out中consequent对应下标的值为1.0
#         out[consequent] = 1.0
#
# print(out)

