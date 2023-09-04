from tqdm import tqdm
# from utils import precision_k, Ndcg_k, get_metrics
from utils import precision_k, Ndcg_k, get_metrics,get_psp_1,get_psp_3,get_psp_5,get_inv_propensity
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sp

import numpy as np
import torch

def train_model(train_loader, device, model, optimizer,gama):
    model.train()
    run_loss = 0.0
    prec_k = []
    ndcg_k = []
    psp_k = []

    real_labels = []
    preds = []
    iftrain = 1
    from tqdm import auto
    # for batch_idx, data in enumerate(auto.tqdm(train_loader, desc='Training')):
    for batch_idx, data in enumerate(tqdm(train_loader, desc='Training')):
        # optimizer.zero_grad()
        # input_ids = tokenizer.encode(batch, add_special_tokens=True)
        # input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        input_ids, input_mask, token_type_ids, labels, ground_label, doc_position, label_position = data[0].long(), data[1].long(), data[2].long(), data[3].float(), data[4].float(),\
                                                                                      data[5].type(torch.int64), data[6].type(torch.int64)
        input_ids, input_mask, token_type_ids, labels, ground_label, doc_position, label_position = input_ids.to(device),input_mask.to(device), \
                                                                                      token_type_ids.to(device), labels.to(device), ground_label.to(device),\
                                                                           doc_position.to(device), label_position.to(device)

        # bert_out = model(input_ids, attention_mask)
        # out_put = bert_out[0]

        output, loss, label_index = model(optimizer,input_ids, input_mask, token_type_ids, doc_position, label_position,labels, ground_label, iftrain,gama)
        run_loss += loss

        # preds_cpu = pre_all_labels(output.data.cpu())
        #preds_cpu = pre_all_labels(output.data.cpu())
        # labels_cpu = labels.data.cpu().float()
        labels_cpu = ground_label.data.cpu().float()
        real_labels.extend(labels_cpu.tolist())
        preds_cpu = output.data.cpu()
        # preds_cpu = output.data
        # preds_cpu = np.array(preds_cpu)
        # zeros = np.zeros_like(preds_cpu.tolist())
        # ones = np.ones_like(preds_cpu.tolist())
        # y_pred = np.where(np.array(preds_cpu.tolist()) >= 0.505, ones, zeros)
        # from sklearn import metrics
        # micro_f1 = metrics.f1_score(labels_cpu.tolist(), y_pred, average='micro')
        preds.extend(preds_cpu.tolist())


        prec = precision_k(labels_cpu.numpy(), preds_cpu, 5)
        prec_k.append(prec)
        ndcg = Ndcg_k(labels_cpu.numpy(), preds_cpu, 5)
        ndcg_k.append(ndcg)
        # psp = propensity_scored_precision_at_k(labels_cpu.numpy(), preds_cpu.numpy(), 5)
        # psp_k.append(psp)


    metrics = get_metrics(real_labels, preds)
    epoch_prec = np.array(prec_k).mean(axis=0)
    epoch_ndcg = np.array(ndcg_k).mean(axis=0)
    # epoch_psp = np.array(psp_k).mean(axis=0)

    loss = run_loss / (batch_idx+1)

    return loss, epoch_prec, epoch_ndcg, metrics

def test_model(test_loader, device, model, optimizer):
    model.eval()
    gama = None
    run_loss = 0.0
    prec_k = []
    ndcg_k = []
    psp_k = []

    all_psp_labels = []

    real_labels = []
    preds = []
    iftrain = 0
    for batch_idx, data in enumerate(tqdm(test_loader, desc='Testing')):
        input_ids, input_mask, token_type_ids, labels, ground_label, doc_position, label_position = data[0].long(), data[1].long(), data[2].long(), data[3].float(), data[4].float(),\
                                                                                      data[5].type(torch.int64), data[6].type(torch.int64)
        input_ids, input_mask, token_type_ids, labels, ground_label, doc_position, label_position = input_ids.to(device),input_mask.to(device), \
                                                                                      token_type_ids.to(device), labels.to(device), ground_label.to(device),\
                                                                           doc_position.to(device), label_position.to(device)
        with torch.no_grad():
            output, loss, label_index = model(optimizer,input_ids, input_mask, token_type_ids, doc_position, label_position,labels, ground_label, iftrain, gama)

        run_loss += loss

        # labels_cpu = labels.data.cpu().float()
        labels_cpu = ground_label.data.cpu().float()
        real_labels.extend(labels_cpu.tolist())
        preds_cpu = output.data.cpu()
        # preds_cpu = output.data
        preds.extend(preds_cpu.tolist())
        prec = precision_k(labels_cpu.numpy(), preds_cpu, 5)
        prec_k.append(prec)
        ndcg = Ndcg_k(labels_cpu.numpy(), preds_cpu, 5)
        ndcg_k.append(ndcg)
        # psp = propensity_scored_precision_at_k(labels_cpu.numpy(), preds_cpu.numpy(), 5)
        # psp_k.append(psp)
        for i in range(len(label_index.data.cpu())):
            all_psp_labels.append(label_index.data.cpu()[i].tolist())

    mlb = MultiLabelBinarizer(sparse_output=True)
    targets = sp.csr_matrix(real_labels,shape=(len(real_labels),3956))
    # all_label = np.load("./Data/AAPD/train_label.npy")
    all_label = np.load("../data/EUR11585/Y_trn.npy")

    inv_w = get_inv_propensity(sp.csr_matrix(all_label))
    pred = np.array(all_psp_labels)
    mlb.fit(pred)
    psp_5 = (get_psp_5(pred, targets, inv_w, mlb))
    psp_3 = (get_psp_3(pred, targets, inv_w, mlb))
    psp_1 = (get_psp_1(pred, targets, inv_w, mlb))

    metrics = get_metrics(real_labels, preds)
    epoch_prec = np.array(prec_k).mean(axis=0)
    epoch_ndcg = np.array(ndcg_k).mean(axis=0)
    epoch_psp = np.array(psp_k).mean(axis=0)
    loss = run_loss / (batch_idx+1)
    return loss, epoch_prec, epoch_ndcg, metrics, psp_1, psp_3, psp_5
