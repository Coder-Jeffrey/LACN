import re
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from functools import partial


from typing import Union, Optional, List, Iterable, Hashable
TPredict = np.ndarray
TTarget = Union[Iterable[Iterable[Hashable]], csr_matrix]
TMlb = Optional[MultiLabelBinarizer]
TClass = Optional[List[Hashable]]

def get_mlb(classes: TClass = None, mlb: TMlb = None, targets: TTarget = None):
    if classes is not None:
        mlb = MultiLabelBinarizer(classes, sparse_output=True)
    if mlb is None and targets is not None:
        if isinstance(targets, csr_matrix):
            mlb = MultiLabelBinarizer(range(targets.shape[1]), sparse_output=True)
            mlb.fit(None)
        else:
            mlb = MultiLabelBinarizer(sparse_output=True)
            mlb.fit(targets)
    return mlb

def get_inv_propensity(train_y: csr_matrix, a=0.55, b=1.5):
    n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)


def get_psp(prediction: TPredict, targets: TTarget, inv_w: np.ndarray, mlb: TMlb = None,
            classes: TClass = None, top=5):
    mlb = get_mlb(classes, mlb, targets)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top])

    prediction = prediction.multiply(inv_w)
    num = prediction.multiply(targets).sum()
    t, den = csr_matrix(targets.multiply(inv_w)), 0
    for i in range(t.shape[0]):
        den += np.sum(np.sort(t.getrow(i).data)[-top:])
    return num / den


get_psp_1 = partial(get_psp, top=1)
get_psp_3 = partial(get_psp, top=3)
get_psp_5 = partial(get_psp, top=5)
get_psp_10 = partial(get_psp, top=10)

def clean_str(string):
    # remove stopwords
    # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for n in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(n + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        #         print("mat",mat)
        num = np.sum(mat, axis=1)
        p[n] = np.mean(num / (n + 1))
    return np.around(p, decimals=4)


def Ndcg_k(true_mat, score_mat, k):
    res = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    label_count = np.sum(true_mat, axis=1)

    for m in range(k):
        y_mat = np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m + 1):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)

# def propensity_scored_precision_at_k(true_mat, score_mat, k):
#     # true_mat: binary matrix of ground truth labels (shape [num_samples, num_classes])
#     # score_mat: real-valued matrix of predicted scores (shape [num_samples, num_classes])
#     # k: number of predicted labels to consider
#     psp = np.zeros((k, 1)) # initialize propensity-scored precision array
#     rank_mat = np.argsort(score_mat)[:, ::-1] # indices of predicted labels sorted by descending score
#     backup = np.copy(score_mat) # make a backup copy of score_mat
#     for i in range(k):
#         score_mat = np.copy(backup)
#         score_mat[rank_mat[:, :(i+1)]] *= (i+1) / np.sum(1 / np.arange(1, i+2)) # update propensity scores
#         precision = np.mean(np.sum(score_mat * true_mat, axis=1) / np.sum(score_mat, axis=1)) # compute precision at k
#         psp[i] = precision
#     return np.around(psp, decimals=4) # round propensity-scored precision values to 4 decimal places

def propensity_scored_precision_at_k(true_mat, score_mat, k):
    ps = 1 / true_mat.sum(axis=0, dtype=np.float32)
    psp = torch.zeros((k, ), dtype=torch.float32)
    rank_mat = torch.argsort(score_mat, dim=1, descending=True)
    for i in range(k):
        indices = rank_mat[:, :i+1]
        ps_prop = (i+1) / torch.arange(1, i+2, dtype=torch.float32)
        ps_prop = ps_prop.view(1, -1).repeat(indices.size(0), 1)
        ps_weight = ps[indices]
        ps_weight = torch.where(ps_weight > 0, ps_weight, torch.tensor([1e-7], dtype=torch.float32))
        psp[i] = torch.mean((ps_weight * true_mat)[indices] @ ps_prop / torch.sum(ps_prop))
    return psp.cpu().numpy()

def get_factor(label_count,k):
    res=[]
    for i in range(len(label_count)):
        n=int(min(label_count[i],k))
        f=0.0
        for j in range(1,n+1):
            f+=1/np.log(j+1)
        res.append(f)
    return np.array(res)


def get_fre(y, y_pred, num_label):
    result = np.zeros(num_label, dtype=float)
    for i in range(len(y)):
        for j in range(num_label):
            if y[i][j] == 1:
                result[j] = result[j]+1
    return y, y_pred

def get_metrics(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    # label_fre = np.load('./Data/AAPD/label_fre.npy')
    # del_label = []
    # for i in range(len(label_fre)):
    #     if label_fre[i]<0:
    #         del_label.append(i)
    # y = np.delete(y,del_label,1)
    # y_pred = np.delete(y_pred,del_label,1)
    # 处理预测值
    zeros = np.zeros_like(y_pred)
    ones = np.ones_like(y_pred)
    y_pred = np.where(y_pred >= 0.5, ones, zeros)
    hamming_loss = metrics.hamming_loss(y, y_pred)
    micro_f1 = metrics.f1_score(y, y_pred, average='micro',zero_division=1)
    micro_precision = metrics.precision_score(y, y_pred, average='micro',zero_division=1)
    micro_recall = metrics.recall_score(y, y_pred, average='micro',zero_division=1)
    macro_f1 = metrics.f1_score(y, y_pred, average='macro',zero_division=1)
    macro_precision = metrics.precision_score(y, y_pred, average='macro',zero_division=1)
    macro_recall = metrics.recall_score(y, y_pred, average='macro',zero_division=1)
    # instance_f1 = metrics.f1_score(y, y_pred, average='samples')
    # instance_precision = metrics.precision_score(y, y_pred, average='samples')
    # instance_recall = metrics.recall_score(y, y_pred, average='samples')
    return [hamming_loss, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1]


def generate_adj(label, num_classes):
    num = np.zeros(num_classes, dtype=float)
    adj = np.zeros((num_classes, num_classes), dtype=float)
    label = label.tolist()
    for i in tqdm(range(len(label))):
        real_label = label[i]
        label_index = [k for k in range(num_classes) if real_label[k] == 1]
        n = len(label_index)
        for j in range(n):
            num[label_index[j]] += 1
            s = j + 1
            while s <= n - 1:
                adj[label_index[j]][label_index[s]] += 1.
                adj[label_index[s]][label_index[j]] += 1.
                s = s + 1
    return {'nums': num, 'adj': adj}


# def gen_A(num_classes, result):
#     _adj = result['adj']
#     _nums = result['nums']
#     # _nums[101] = 1.0
#     # _nums[102] = 1.0
#     _nums = _nums[:, np.newaxis]
#     _adj = _adj / _nums
#     # _adj[_adj < t] = 0
#     # _adj[_adj >= t] = 1
#     # s = _adj.sum(0, keepdims=True) + 1e-6
#     # _adj = _adj * p / s
#     _adj = _adj + np.identity(num_classes, np.float64)
#     return _adj
def gen_A(num_classes, result):
    _adj = result['adj']
    _nums = result['nums']
    _nums[_nums == 0] = 1e-8  # 将零值替换为一个较小的非零值
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj = _adj + np.identity(num_classes, np.float64)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj
# def gen_adj(A):
#     nan_mask = torch.isnan(A)
#     nan_indices = torch.nonzero(nan_mask)
#     if nan_indices.numel() > 0:
#         print("A matrix contains NaN values at indices:", nan_indices)
#     row_sum = A.sum(1)
#     D = torch.where(row_sum > 0, torch.pow(row_sum.float(), -0.5), torch.tensor(1e-8))
#     D = torch.diag(D)
#     adj = torch.matmul(torch.matmul(A, D).t(), D)
#     return adj

def get_inv_propensity(train_y, a=0.55, b=1.5):

    n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)

def get_psp(true_mat, score_mat, inv_w, k):
    res = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        # score_mat = np.ceil(score_mat)
        pre_mat = np.multiply(score_mat, inv_w)
        mat = np.multiply(pre_mat, true_mat)
        num = np.sum(mat, axis=1)
        pres = np.mean(num/(k+1))
        t, den = csr_matrix(np.multiply(true_mat, inv_w)), 0
        for i in range(t.shape[0]):
            den += np.sum(np.sort(t.getrow(i).data)[-(k+1):])
        res[k] = pres / den
    return np.around(res, decimals=4)
