from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import pickle
import numpy as np
import math
import torch
from collections import deque

def cosine_similarity(x, y):
    '''
    Cosine Similarity of two tensors
    Args:
        x: torch.Tensor, m x d
        y: torch.Tensor, n x d
    Returns:
        result, m x n
    '''

    assert x.size(1) == y.size(1)
    x = torch.nn.functional.normalize(x, dim=1)
    y = torch.nn.functional.normalize(y, dim=1)
    return x @ y.transpose(0, 1)

class LE(nn.Module):
    def __init__(self, num_feature, num_classes, hidden_dim=128):
        super(LE, self).__init__()
        self.fe1 = nn.Sequential(
            nn.Linear(num_feature, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.fe2 = nn.Linear(hidden_dim, hidden_dim)
        self.le1 = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.le2 = nn.Linear(hidden_dim, hidden_dim)
        self.de1 = nn.Sequential(
            nn.Linear(2 * hidden_dim, num_classes),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_classes),
        )
        self.de2 = nn.Linear(num_classes, num_classes)

    def forward(self, x, y):
        x = self.fe1(x) + self.fe2(self.fe1(x))
        y = self.le1(y) + self.le2(self.le1(y))
        d = torch.cat([x, y], dim=-1)
        d = self.de1(d) + self.de2(self.de1(d))
        return d
class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class ClassifyModel(nn.Module):
    def __init__(self, num_labels, batch_size, label_adj, label_embed, loss_function, pretrained_model='./roberta-base'):
        super(ClassifyModel, self).__init__()
        self.dim = 768
        self.m = 0.9
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.label_adj = label_adj.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        # self.g = torch.nn.DataParallel(self.g)
        self.GCN3 = GraphConvolution(self.dim, self.dim)
        self.relu1 = torch.nn.LeakyReLU(0.2)
        self.GCN4 = GraphConvolution(self.dim,self.dim)
        self.GCN6 = GraphConvolution(self.dim, self.dim)
        self.relu2 = torch.nn.LeakyReLU(0.2)
        self.GCN5 = GraphConvolution(self.dim, self.dim)
        self.conv = torch.nn.Conv2d(1, num_labels, kernel_size=[9, 9], padding=4)
        self.label_embed = self.load_labelembedd(label_embed)
        # self.linear_adj = nn.Linear(300, 300)
        self.linear1 = nn.Linear(self.dim, num_labels)
        self.linear = nn.Linear(self.dim*2, self.dim)
        self.linear2 = nn.Linear(self.dim, 1)
        self.linear_label1 = nn.Linear(self.dim, self.dim)
        self.linear_label2 = nn.Linear(self.dim, self.dim)
        self.w_att = torch.nn.LeakyReLU(0.2)

        params_decay = (p for name, p in self.named_parameters() if 'bias' not in name)
        params_no_decay = (p for name, p in self.named_parameters() if 'bias' in name)
        self.alpha = 0.01
        self.beta = 0.01
        self.threshold = 0
        self.loss_func = loss_function
        self.method = 'threshold'
        self.le = LE(768, 200)
        self.device = 'cuda:0'
        self.dropout = torch.nn.Dropout(0.5)
        self.gradient_clip_value, self.gradient_norm_queue = 5.08, deque([np.inf], maxlen=5)

    def classifier(self, optimizer, input_ids, input_mask, labels, iftrain):
        if iftrain!=0:
            optimizer.zero_grad()
        bert_output = self.bert_model(input_ids, input_mask)
        doc_embedding = bert_output[0]
        doc_embedding = self.dropout(doc_embedding)

        # #for test
        # H_enc = torch.sum(doc_embedding, dim=1)
        # label_output = torch.sigmoid(self.linear1(H_enc))

        # label
        x = []
        for i in range(self.batch_size):
            temp = self.label_embed.weight.data
            x.append(temp)
        label_embedding = torch.stack(x)

        label_embedding = self.GCN3(label_embedding, self.label_adj)
        label_embedding = self.relu1(label_embedding)
        # label_embedding = self.GCN4(label_embedding, self.label_adj)
        # dual GCN update adj
        A = torch.sigmoid(torch.bmm(self.linear_label1(label_embedding),self.linear_label2(label_embedding).transpose(1,2)))
        A = self.normalize_adj(A)
        d_label_embedding = self.GCN5(label_embedding, A)
        d_label_embedding = self.relu2(d_label_embedding)
        # d_label_embedding = self.GCN6(d_label_embedding, A)
        label_embedding = torch.cat((label_embedding,d_label_embedding),-1)
        label_embedding = self.linear(label_embedding).squeeze(-1)

        label_embedding = torch.nn.functional.normalize(label_embedding, p=1, dim=-2)
        doc_embedding = torch.nn.functional.normalize(doc_embedding, p=1, dim=-2)

        word_label_att = torch.bmm(doc_embedding, label_embedding.transpose(1, 2))


        word_label_att = word_label_att.unsqueeze(1)

        Att_v = self.conv(word_label_att)

        Att_v = torch.max(Att_v, dim=1)[0]

        Att_v = torch.max(Att_v, keepdim=True, dim=-1)[0]
        Att_v_tanh = torch.tanh(Att_v)
        H_enc = Att_v_tanh * doc_embedding
        # H_enc = self.dropout(H_enc)
        H_enc = torch.sum(H_enc, dim=1)


        #GCN + Bert
        #text_output = torch.sigmoid(self.linear1(text_embedding))
        label_output = torch.sigmoid(self.linear1(H_enc))
        # print('-------------------------')
        # print(label_output)

        # if iftrain == 1:
        #     le = self.le(text_embedding.detach(), labels)
        #     p_le = torch.sigmoid(le.unsqueeze(2)).squeeze()
        #     label_output = self.m * label_output + (1-self.m) * p_le
        # output = self.m * label_output + (1 - self.m) * text_output

        # # LSAN
        # doc_embedding = torch.layer_norm(doc_embedding, [2])
        # label_embedding = torch.layer_norm(label_embedding, [2])
        # G = torch.softmax(torch.matmul(doc_embedding, label_embedding.transpose(1, 2)), dim=-1)
        # doc_label_embedding = torch.bmm(G.transpose(1, 2), doc_embedding)
        #
        # weight1 = torch.sigmoid(self.weight1(doc_label_embedding))
        # weight2 = torch.sigmoid(self.weight2(label_embedding))
        # weight1 = weight1 / (weight1 + weight2)
        # weight2 = 1 - weight1
        #
        # label_embedding = weight1 * doc_label_embedding + weight2 * label_embedding

        if iftrain!=0:
            # datastore = self.datastore()
            # knn_result = self.knn(label_output,text_embedding,datastore,5)
            # label_output = label_output*0.75 + knn_result*0.25
            # closs = self.compute_contrasitve_loss(label_output, text_embedding)

            # loss = self.loss_func(label_output,labels) + 0.1*closs
            loss = self.loss_func(label_output, labels)
            # print(loss.data)
            # path = str(iftrain) + 'datastore_.txt'
            # for i in range(len(H_enc)):
            #     h = H_enc.detach().cpu().numpy()[i]
            #     l = labels.cpu().numpy()[i]
            #     v = {'h': h.tolist(), 'l': l.tolist()}
            #     v = str(v)
            #     with open(path, 'a+') as f:
            #         f.write(v + '\n')
            #datastore = self.datastore()
            #self.knn(labels,H_enc,datastore)
            loss.backward()
            optimizer.step()
        else:
            #datastore = self.datastore()
            #print("datastore load finish")
            #knn_result = self.knn(label_output,text_embedding,datastore,5)
            loss = self.loss_func(label_output, labels)
        return label_output, loss.data

    def forward(self, optimizer,input_ids, input_mask, labels, iftrain):
        label_output, loss = self.classifier(optimizer, input_ids, input_mask, labels, iftrain)
        return label_output, loss

    def load_labelembedd(self, label_embed):
        """Load the embeddings based on flag"""
        embed = torch.nn.Embedding(label_embed.size(0), label_embed.size(1))
        embed.weight = torch.nn.Parameter(label_embed)
        return embed

    def normalize_adj(self, adj):
        # row_sum = torch.tensor(adj.sum(1))
        row_sum = adj.sum(1).clone().detach().requires_grad_(True)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj_normalized = torch.bmm(torch.bmm(adj, d_mat_inv_sqrt).transpose(1,2),d_mat_inv_sqrt)
        return adj_normalized

class ClassifyModel11585(nn.Module):
    def __init__(self, num_labels, batch_size, label_adj, merge_id, frequent_index, frequent_frequency, antecedents, consequents, confidence, pretrained_model='./pretrain/bert_base_uncase'):
        super(ClassifyModel11585, self).__init__()
        self.merge_id = merge_id
        self.frequent_index = frequent_index
        self.frequency = frequent_frequency
        self.antecedents = antecedents
        self.consequents = consequents
        self.confidence = confidence

        self.dim = 768
        self.m = 0.9
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.label_adj = label_adj.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        # self.g = torch.nn.DataParallel(self.g)
        self.GCN3 = GraphConvolution(self.dim, self.dim)
        self.relu1 = torch.nn.LeakyReLU(0.2)
        self.GCN4 = GraphConvolution(self.dim,self.dim)
        self.GCN6 = GraphConvolution(self.dim, self.dim)
        self.relu2 = torch.nn.LeakyReLU(0.2)
        self.GCN5 = GraphConvolution(self.dim, self.dim)
        self.conv = torch.nn.Conv2d(1, num_labels, kernel_size=[9, 9], padding=4)
        # self.label_embed = self.load_labelembedd(label_embed)
        # self.linear_adj = nn.Linear(300, 300)
        self.linear1 = nn.Linear(self.dim, num_labels)
        self.linear = nn.Linear(self.dim*2, self.dim)
        self.linear2 = nn.Linear(self.dim, 1)
        self.linear_label1 = nn.Linear(self.dim, self.dim)
        self.linear_label2 = nn.Linear(self.dim, self.dim)
        self.w_att = torch.nn.LeakyReLU(0.2)

        params_decay = (p for name, p in self.named_parameters() if 'bias' not in name)
        params_no_decay = (p for name, p in self.named_parameters() if 'bias' in name)
        self.alpha = 0.01
        self.beta = 0.01
        self.threshold = 0
        self.loss_func = torch.nn.BCELoss()
        self.method = 'threshold'
        self.le = LE(768, 200)
        self.device = 'cuda:0'
        self.dropout = torch.nn.Dropout(0.2)
        self.gradient_clip_value, self.gradient_norm_queue = 5.08, deque([np.inf], maxlen=5)

    def classifier(self,optimizer,input_ids, input_mask, token_type_ids, fit_docspace_position, fit_labelspace_position, labels, ground_label, iftrain, gama):
        if iftrain!=0:
            optimizer.zero_grad()
        bert_output = self.bert_model(input_ids, input_mask, token_type_ids)
        output = bert_output[0]
        # output = self.dropout(output)

        text_embedding = bert_output[1]

        token_hidden_size = output.shape[-1]
        doc_seq_length = output.shape[-2] - self.num_labels - 3
        doc_embedding = self.gather_indexes(output, fit_docspace_position)
        doc_embedding = torch.reshape(doc_embedding, [-1, doc_seq_length, token_hidden_size])


        label_embedding = self.gather_indexes(output, fit_labelspace_position)
        label_embedding = torch.reshape(label_embedding, [-1, self.num_labels, token_hidden_size])
        
        # label update
        label_embedding = self.GCN3(label_embedding, self.label_adj)
        label_embedding = self.relu1(label_embedding)


        A = torch.sigmoid(torch.bmm(self.linear_label1(label_embedding),self.linear_label2(label_embedding).transpose(1,2)))
        A = self.normalize_adj(A)
        d_label_embedding = self.GCN5(label_embedding, A)
        d_label_embedding = self.relu2(d_label_embedding)

        label_embedding = torch.cat((label_embedding,d_label_embedding),-1)
        label_embedding = self.linear(label_embedding).squeeze(-1)

        label_embedding = torch.nn.functional.normalize(label_embedding, p=1, dim=-2)
        doc_embedding = torch.nn.functional.normalize(doc_embedding, p=1, dim=-2)

        word_label_att = torch.bmm(doc_embedding, label_embedding.transpose(1, 2))

        word_label_att = word_label_att.unsqueeze(1)

        Att_v = self.conv(word_label_att)

        Att_v = torch.max(Att_v, dim=1)[0]

        Att_v = torch.max(Att_v, keepdim=True, dim=-1)[0]
        Att_v_tanh = torch.tanh(Att_v)
        H_enc = Att_v_tanh * doc_embedding
        # H_enc = self.dropout(H_enc)
        H_enc = torch.sum(H_enc, dim=1)


        #GCN + Bert
        # label_output = torch.sigmoid(self.linear1(text_embedding))
        label_output = torch.sigmoid(self.linear1(H_enc))
        temp = label_output
        pred_3956 = self.get_origin_label(temp, ground_label)
        # from utils import precision_k
        # labels_temp = labels
        # labels_temp = labels_temp.data.cpu().float().detach().numpy()
        # temp = temp.cpu().detach()
        # preLoss = precision_k(labels_temp, temp, 5)
        # p_a5 = preLoss[4][0]
        # p_a5loss = 1.0 - p_a5
        # print('-------------------------')
        # print(label_output)

        # if iftrain == 1:
        #     le = self.le(text_embedding.detach(), labels)
        #     p_le = torch.sigmoid(le.unsqueeze(2)).squeeze()
        #     label_output = self.m * label_output + (1-self.m) * p_le
        # output = self.m * label_output + (1 - self.m) * text_output

        # # LSAN
        # doc_embedding = torch.layer_norm(doc_embedding, [2])
        # label_embedding = torch.layer_norm(label_embedding, [2])
        # G = torch.softmax(torch.matmul(doc_embedding, label_embedding.transpose(1, 2)), dim=-1)
        # doc_label_embedding = torch.bmm(G.transpose(1, 2), doc_embedding)
        #
        # weight1 = torch.sigmoid(self.weight1(doc_label_embedding))
        # weight2 = torch.sigmoid(self.weight2(label_embedding))
        # weight1 = weight1 / (weight1 + weight2)
        # weight2 = 1 - weight1
        #
        # label_embedding = weight1 * doc_label_embedding + weight2 * label_embedding

        if iftrain!=0:
            # datastore = self.datastore()
            # knn_result = self.knn(label_output,text_embedding,datastore,5)
            # label_output = label_output*0.75 + knn_result*0.25
            closs = self.compute_contrasitve_loss(label_output, text_embedding)

            loss = self.loss_func(label_output,labels) + gama*closs
            # loss = self.loss_func(label_output, labels)
            # print(loss.data)
            # path = str(iftrain) + 'datastore_.txt'
            # for i in range(len(H_enc)):
            #     h = H_enc.detach().cpu().numpy()[i]
            #     l = labels.cpu().numpy()[i]
            #     v = {'h': h.tolist(), 'l': l.tolist()}
            #     v = str(v)
            #     with open(path, 'a+') as f:
            #         f.write(v + '\n')
            #datastore = self.datastore()
            #self.knn(labels,H_enc,datastore)
            loss.backward()
            optimizer.step()
            labels_index = []
        else:
            #datastore = self.datastore()
            #print("datastore load finish")
            #knn_result = self.knn(label_output,text_embedding,datastore,5)
            # loss = self.loss_func(label_output, labels)
            loss = torch.tensor(1.0)
            scores, labels_index = torch.topk(pred_3956, 3956)
        return pred_3956, loss.data, labels_index

    def forward(self, optimizer,input_ids, input_mask, token_type_ids, fit_docspace_position, fit_labelspace_position,labels, ground_label, iftrain, gama):
        label_output,loss,labels_index = self.classifier(optimizer,input_ids, input_mask, token_type_ids, fit_docspace_position, fit_labelspace_position,labels, ground_label, iftrain, gama)
        return label_output, loss,labels_index

    def load_labelembedd(self, label_embed):
        """Load the embeddings based on flag"""
        embed = torch.nn.Embedding(label_embed.size(0), label_embed.size(1))
        embed.weight = torch.nn.Parameter(label_embed)
        return embed

    def normalize_adj(self, adj):
        # row_sum = torch.tensor(adj.sum(1))
        row_sum = adj.sum(1).clone().detach().requires_grad_(True)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj_normalized = torch.bmm(torch.bmm(adj, d_mat_inv_sqrt).transpose(1,2),d_mat_inv_sqrt)
        return adj_normalized

    def gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = sequence_tensor.shape
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = torch.reshape(
            torch.range(0, batch_size-1, dtype=torch.int64) * seq_length, [-1, 1]).type(torch.int64).cuda()
        positions = positions.type(torch.int64)
        flat_positions = torch.reshape(positions + flat_offsets, [-1, 1])
        flat_positions = flat_positions.expand([flat_positions.shape[0], width])
        flat_sequence_tensor = torch.reshape(sequence_tensor, [batch_size * seq_length, width])
        output_tensor = torch.gather(flat_sequence_tensor, 0, flat_positions)
        return output_tensor

    def get_origin_label_yuan(self, label_out, ground_labels):
        label_out = label_out.cpu().detach().numpy()
        label_out = np.array(label_out)
        # label_out200 = np.where(label_out200 >= 0.5, 1., 0.)
        # label293 = np.zeros((self.batch_size, 293))
        # sim = np.load('association11585/Jaccard_sim.npy')
        # for i in range(self.batch_size):
        #     for j in range(len(self.merge_id)):
        #         # for k in self.merge_id[j]:
        #         #     label293[i][k] = label_out[i][j]
        #         if len(self.merge_id[j])!=1:
        #             label1 = self.merge_id[j][0]
        #             label2 = self.merge_id[j][1]
        #             a = self.frequency[label1]
        #             b = self.frequency[label2]
        #             # cc = label_out[i][j]
        #             if sim[label1][label2]>=0.5:
        #                     label293[i][label1] = label_out[i][j]
        #                     label293[i][label2] = label_out[i][j]
        #             elif a>b:
        #                     label293[i][label1] = label_out[i][j]
        #                     label293[i][label2] = label_out[i][j] * (b/a)
        #             else:
        #                 label293[i][label1] = label_out[i][j] * (a / b)
        #                 label293[i][label2] = label_out[i][j]
        #         else:
        #             label293[i][self.merge_id[j][0]] = label_out[i][j]
                    # label293[i][label1] = label_out[i][j]
                    # label293[i][label2] = label_out[i][j]

                # else:
                #     label293[i][self.merge_id[j][0]] = label_out[i][j]

        # ground_labels293 = ground_labels293.cpu()
        # label293 = torch.from_numpy(label293).float()

        # print(label293)

        label3956 = np.zeros((self.batch_size, 3956))
        # 遍历label_raw
        ind = self.frequent_index
        for i in range(self.batch_size):
            for j in range(len(self.frequent_index)):
                label3956[i][self.frequent_index[j]] = label_out[i][j]

        # 遍历antecedents和consequents
        # a = self.antecedents
        # b = self.consequents
        # c = self.confidence
        # with open('./association11585/antecedents_s0.001.pickle', 'rb') as file:
        #     antecedents = pickle.load(file)
        # with open('./association11585/consequents_s0.001.pickle', 'rb') as file:
        #     consequents = pickle.load(file)
        # with open('./association11585/confidences_s0.001.pickle', 'rb') as file:
        #     confidences = pickle.load(file)
    

        # for i in range(self.batch_size):
        #     for index, (antecedent, consequent) in enumerate(zip(antecedents, consequents)):
        #         proSum = 0.0
        #         for value in antecedent:
        #             if label3956[i][value]<0.5:
        #                 proSum = 0.0
        #                 break
        #             proSum += label3956[i][value]
        #         predicted_prob = proSum*confidences[index]
        #         # if set(consequent).issubset(self.frequent_index):
        #         #     continue
        #         # if predicted_prob>1.0:
        #         #     predicted_prob=1.0
        #         label3956[i][consequent] = predicted_prob

        pred_3956 = torch.from_numpy(label3956).float()
        # raw_loss = self.loss_func(pred_3956, ground_label)
        return pred_3956

    def get_origin_label(self, label_out, ground_labels293):
        label_out = label_out.cpu().detach().numpy()
        label_out = np.array(label_out)
        # label_out200 = np.where(label_out200 >= 0.5, 1., 0.)
        label293 = np.zeros((self.batch_size, 293))
        sim = np.load('association/Jaccard_sim.npy')
        for i in range(self.batch_size):
            for j in range(len(self.merge_id)):
                # for k in self.merge_id[j]:
                #     label293[i][k] = label_out[i][j]
                if len(self.merge_id[j])!=1:
                    label1 = self.merge_id[j][0]
                    label2 = self.merge_id[j][1]
                    a = self.frequency[label1]
                    b = self.frequency[label2]
                    # cc = label_out[i][j]
                    # if a>b :
                    #     label293[i][label1] = label_out[i][j]
                    #     label293[i][label2] = label_out[i][j]*(b/a)
                    # else:
                    #     label293[i][label1] = label_out[i][j]*(a/b)
                    #     label293[i][label2] = label_out[i][j]
                    if sim[label1][label2]>=0.5:
                        # print(sim[label1][label2])
                        label293[i][label1] = label_out[i][j]
                        label293[i][label2] = label_out[i][j]
                    elif a>b:
                        label293[i][label1] = label_out[i][j]
                        label293[i][label2] = label_out[i][j] * (b/a)
                    else:
                        label293[i][label1] = label_out[i][j] * (a / b)
                        label293[i][label2] = label_out[i][j]
                else:
                    label293[i][self.merge_id[j][0]] = label_out[i][j]

        ground_labels293 = ground_labels293.cpu()
        label293 = torch.from_numpy(label293).float()
        # raw293_loss = self.loss_func(label293, ground_labels293)

        # print(label293)
        label3956 = np.zeros((self.batch_size, 3956))
        # 遍历label_raw
        ind = self.frequent_index
        for i in range(self.batch_size):
            for j in range(len(self.frequent_index)):
                label3956[i][self.frequent_index[j]] = label293[i][j]

        with open('./association/antecedents_s0.001.pickle', 'rb') as file:
            antecedents = pickle.load(file)
        with open('./association/consequents_s0.001.pickle', 'rb') as file:
            consequents = pickle.load(file)
        with open('./association/confidences_s0.001.pickle', 'rb') as file:
            confidences = pickle.load(file)

        # 遍历antecedents和consequents
        for i in range(self.batch_size):
            for index, (antecedent, consequent) in enumerate(zip(antecedents, consequents)):
                # print(index)
                proSum = 0.0
                for value in antecedent:
                    if label3956[i][value]<0.5:
                        proSum = 0.0
                        break
                    proSum += label3956[i][value]
                predicted_prob = proSum*confidences[index]
                if predicted_prob>1.0:
                    predicted_prob=1.0
                # if set(consequent).issubset(self.frequent_index):
                #     continue
                label3956[i][consequent] = predicted_prob

        # loss
        # gl = ground_label.numpy()
        # from sklearn import metrics   
        # zeros = np.zeros_like(label3956)
        # ones = np.ones_like(label3956)
        # y_pred = np.where(label3956 >= 0.5, ones, zeros)
        # micro_f1 = metrics.f1_score(gl, y_pred, average='micro', zero_division=1)
        # raw_loss = -np.log(micro_f1)        

        pred_3956 = torch.from_numpy(label3956).float()
        return pred_3956
        


    def compute_contrasitve_loss(self, label_vec, text_embedding):
        C_ij = cosine_similarity(label_vec, label_vec)
        # C_ij = torch.mm(label_vec.T,label_vec)
        B_ij = torch.zeros(len(text_embedding), len(text_embedding))
        for k in range(len(text_embedding)):
            for q in range(len(text_embedding)):
                B_ij[k][q] = C_ij[k][q] / (C_ij.sum(1)[k] - C_ij[k][q])
        text_embedding = torch.nn.functional.normalize(text_embedding, dim=1)
        τ = 10
        text_dist = torch.zeros((len(text_embedding), len(text_embedding)))
        exp_dist = torch.zeros(text_dist.shape[0], text_dist.shape[0])
        exp_dist_sum = torch.zeros(text_dist.shape[0])

        for i in range(len(text_embedding)):
            for j in range(len(text_embedding)):
                text_dist[i][j] = torch.sqrt(torch.sum((text_embedding[i] - text_embedding[j]) ** 2)) / τ

                exp_dist[i][j] = math.exp(-text_dist[i][j])

                if i != j:
                    exp_dist_sum[i] = exp_dist[i][j] + exp_dist_sum[i]

        L_con = torch.zeros(text_dist.shape[0], text_dist.shape[0])

        for m in range(text_dist.shape[0]):
            for n in range(text_dist.shape[0]):
                if m != n:
                    L_con[m][n] = -B_ij[m][n] * math.log(exp_dist[m][n] / exp_dist_sum[m])
        result = L_con.mean()
        return result.cuda()
        
    # def get_origin_label(self, label_out200, ground_label):
    #     label_out200 = label_out200.cpu().detach().numpy()
    #     label_out200 = np.array(label_out200)
    #     print()
    #     label_out200 = np.where(label_out200 >= 0.5, 1., 0.)
    #     label293 = np.zeros((self.batch_size, 293))
    #     for i in range(self.batch_size):
    #         for j in range(label_out200.shape[1]):
    #             if label_out200[i][j] == 1.:
    #                 for k in self.merge_id[j]:
    #                     label293[i][k] = 1.
    #     # print(label293)
    #     label3956 = np.zeros((self.batch_size, 3956))
    #     # 遍历label_raw
    #     for i in range(self.batch_size):
    #         for j in range(len(self.frequent_index)):
    #             if label293[i][j] == 1.:
    #                 label3956[i][self.frequent_index[j]] = 1.

    #     # 遍历antecedents和consequents
    #     for i in range(self.batch_size):
    #         for antecedent, consequent in zip(self.antecedents, self.consequents):
    #             # 检查antecedent中的值是否都在out的下标中，并且对应的out的值都为1.0
    #             if all(label3956[i][value] == 1.0 for value in antecedent):
    #                 # 更新out中consequent对应下标的值为1.0
    #                 label3956[i][consequent] = 1.0

    #     from sklearn import metrics
    #     ground_label = ground_label.cpu().numpy()
    #     hamming_loss = metrics.hamming_loss(ground_label, label3956)
    #     pred_3956 = label3956
    #     return pred_3956, hamming_loss


