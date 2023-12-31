import torch
import torch.utils.data as data_utils
import numpy as np
from utils import generate_adj, gen_A, gen_adj

def load_data(batch_size, ind):
    path = 'BertInputG0{}'.format(ind)
    # path = 'subBertInput293'
    train_input_ids = np.load('../data/EUR/{}/train_input_ids.npy'.format(path))
    train_input_mask = np.load('../data/EUR/{}/train_input_mask.npy'.format(path))
    train_token_type_ids = np.load('../data/EUR/{}/train_token_type_ids.npy'.format(path))
    train_label = np.load('../data/EUR/{}/train_label.npy'.format(path))
    train_fit_doc_position = np.load('../data/EUR/{}/train_fit_doc_position.npy'.format(path))
    train_fit_label_position = np.load('../data/EUR/{}/train_fit_label_position.npy'.format(path))
    test_input_ids = np.load('../data/EUR/{}/test_input_ids.npy'.format(path))
    test_input_mask = np.load('../data/EUR/{}/test_input_mask.npy'.format(path))
    test_token_type_ids = np.load('../data/EUR/{}/test_token_type_ids.npy'.format(path))
    # test_label = np.load('./Data/AAPD/test_label.npy')
    test_fit_doc_position = np.load('../data/EUR/{}/test_fit_doc_position.npy'.format(path))
    test_fit_label_position = np.load('../data/EUR/{}/test_fit_label_position.npy'.format(path))
    # text_mask = np.load('../data/EUR/BertInput/text_mask.npy')
    # train_adj = np.load('../data/EUR/BertInput/train_adj.npy')
    # test_adj = np.load('../data/EUR/BertInput/test_adj.npy')

    num_labels = train_label.shape[1]
    result = generate_adj(train_label, num_classes=num_labels)
    adj = torch.from_numpy(gen_A(num_labels, result)).float()
    label_adj = gen_adj(adj)

    # test_label = np.load('./association/merge_test_y_greedy_260.npy')
    test_label = np.random.rand(2306, 293) # Just for data alignment
    # test_label = np.random.rand(3621, 251)
    # test_label = np.load('../data/EUR/fre_test_y_369.npy')

    ground_train_label = np.load('../data/EUR/raw_label.npy')
    # ground_test_label = np.load('../data/EUR/raw_label_3621.npy')
    ground_test_label = np.load('../data/EUR/testSub_y.npy')
    # ground_train_label = np.load('../data/EUR/raw_label_11129_369.npy')
    # ground_test_label = np.load('../data/EUR/raw_label_3701_369.npy')

    # train_label293 = np.load('../data/EUR/fre_train_y.npy')
    # test_label293 = np.load('../data/EUR/fre_test_y.npy')


    train_data = data_utils.TensorDataset(torch.from_numpy(train_input_ids), torch.from_numpy(train_input_mask),
                                          torch.from_numpy(train_token_type_ids), torch.from_numpy(train_label), torch.from_numpy(ground_train_label),
                                          torch.from_numpy(train_fit_doc_position), torch.from_numpy(train_fit_label_position))
    test_data = data_utils.TensorDataset(torch.from_numpy(test_input_ids), torch.from_numpy(test_input_mask),
                                         torch.from_numpy(test_token_type_ids), torch.from_numpy(test_label), torch.from_numpy(ground_test_label),
                                         torch.from_numpy(test_fit_doc_position), torch.from_numpy(test_fit_label_position))

    train_dataloader = data_utils.DataLoader(train_data, batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_utils.DataLoader(test_data, batch_size, shuffle=False, drop_last=True)
    return train_dataloader, test_dataloader, label_adj

if __name__ == '__main__':
    load_data(32)
