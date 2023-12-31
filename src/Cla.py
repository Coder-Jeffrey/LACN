import torch
from Cmodel import ClassifyModel11585
from Dataloader import load_data
from Ctrain import train_model, test_model
import warnings
import pickle
import torch.nn as nn
warnings.filterwarnings('ignore')

print('start')

def classification(epochs, batch_size, learning_rate, weight_decay):
    jst=[4]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for i in jst:
        ind = i
        print('当前参数G0.{}'.format(ind))
        train_dataloader, test_dataloader, label_adj = load_data(batch_size=batch_size, ind=ind)
        path = './association/merge_id_greedy_0{}.pickle'.format(ind)
        with open(path, 'rb') as file:
            merge_id = pickle.load(file)
        # with open('./association/merge_id_greedy_02.pickle', 'rb') as file:
        #     merge_id = pickle.load(file)
        with open('../data/EUR/frequent_index.pickle', 'rb') as file:
            frequent_index = pickle.load(file)
        with open('../data/EUR/frequent_frequency.pickle', 'rb') as file:
            frequent_frequency = pickle.load(file)
        with open('./association/antecedents_s0.001.pickle', 'rb') as file:
            antecedents = pickle.load(file)
        with open('./association/consequents_s0.001.pickle', 'rb') as file:
            consequents = pickle.load(file)
        with open('./association/confidences_s0.001.pickle', 'rb') as file:
            confidence = pickle.load(file)

        num_labels = label_adj.size()[0]
        # model = EncoderModel(54, label_adj)
        # # model = nn.DataParallel(model)
        # # model = BertEncoderModel(54, label_adj)
        # model = model.to(device)
        model = ClassifyModel11585(num_labels, batch_size, label_adj=label_adj, merge_id=merge_id, frequent_index=frequent_index, frequent_frequency=frequent_frequency, antecedents=antecedents, consequents=consequents, confidence=confidence)
        model = model.to(device)
        criterion = torch.nn.BCELoss()
        # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99))
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        # Gama = [0.009]

        gama = 0.009
        s = 'yuanG0'+str(ind) + 'checkpoint.pt'
        # for gama in Gama:
        # print(f"the value of gama:{gama}")
        for epoch in range(epochs):
            train_loss, train_prec, train_ndcg, train_metrics = train_model(train_dataloader, device, model, optimizer,gama)
            step_scheduler.step(train_loss)
            test_loss, test_prec, test_ndcg, test_metrics,psp_1,psp_3,psp_5 = test_model(test_dataloader, device, model, optimizer)

            # s = str(epoch) + 'checkpoint.pt'
            # torch.save(model, s)

            print("Epoch={:02d}".format(epoch + 1))
            print("train_loss={:.4f}".format(train_loss.item()),
                "p@1:%.4f" % (train_prec[0]), "p@3:%.4f" % (train_prec[2]), "p@5:%.4f" % (train_prec[4]),
                "n@3:%.4f" % (train_ndcg[2]), "n@5:%.4f" % (train_ndcg[4]))
            print("train_HL:%.4f" % (train_metrics[0]), "Mi-P:%.4f" % (train_metrics[1]), "Mi-R:%.4f" % (train_metrics[2]), "Mi-F1:%.4f" % (train_metrics[3]),
                "Ma-P:%.4f" % (train_metrics[4]), "Ma-R:%.4f" % (train_metrics[5]), "Ma-F1:%.4f" % (train_metrics[6]))
            print("**test_loss={:.4f}".format(test_loss.item()),
                "p@1:%.4f" % (test_prec[0]), "p@3:%.4f" % (test_prec[2]), "p@5:%.4f" % (test_prec[4]),
                "n@3:%.4f" % (test_ndcg[2]), "n@5:%.4f" % (test_ndcg[4]))
            print("**test_HL:%.4f" % (test_metrics[0]), "Mi-P:%.4f" % (test_metrics[1]), "Mi-R:%.4f" % (test_metrics[2]), "Mi-F1:%.4f" % (test_metrics[3]),
                "Ma-P:%.4f" % (test_metrics[4]), "Ma-R:%.4f" % (test_metrics[5]), "Ma-F1:%.4f" % (test_metrics[6]),
                "PSP1:%.4f" % (psp_1), "PSP3:%.4f" % (psp_3), "PSP5:%.4f" % (psp_5))

if __name__ == '__main__':
    epochs = 30
    learning_rate = 3e-5
    batch_size = 16
    weight_decay = 0.0
    classification(epochs, batch_size, learning_rate, weight_decay)

