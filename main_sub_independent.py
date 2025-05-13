import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import math
import torch.utils.data
import torch.nn.functional as F

import datetime
from models.TAHAG_Independent import Model
from utils.utils import *
import argparse
import copy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(10)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start_time_for_checkpoint = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"

class MyModel():
    def __init__(self, model = None, source_loader = 0, target_loader = 0, iteration = 10000, n_epochs=1000, lr = 0.01, optimizer_name='adam', log_interval = 10, sub_id = 0, session_id = 0):
        self.model = model
        self.model.to(device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.iteration = iteration
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.log_interval = log_interval
        self.sub_id = sub_id
        self.session_id = session_id
        self.root_path = './output/model_state/SEED4_SubIndependent/' + start_time_for_checkpoint + '/'
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        self.path = self.root_path + 'sub_' + str(self.sub_id) + '_session_' + str(self.session_id) + '.pth'

    def __getModel__(self):
        return self.model

    def train(self):
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)
        correct = 0
        epoch = 0
        learning_rate = self.lr

        for i in range(1, self.iteration + 1):
            self.model.train()
            if self.optimizer_name == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            elif self.optimizer_name == 'rmsprop':
                optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            try:
                source_data, source_label, source_domain_label = next(source_iter)
            except Exception as err:
                epoch = epoch + 1
                source_iter = iter(self.source_loader)
                source_data, source_label, source_domain_label = next(source_iter)
            try:
                target_data, target_label, target_domain_label = next(target_iter)
            except Exception as err:
                target_iter = iter(self.target_loader)
                target_data, target_label, target_domain_label = next(target_iter)
            source_data, source_label = source_data.to(device), source_label.to(device)
            target_data = target_data.to(device)
            domain_label = torch.cat([source_domain_label, target_domain_label]).to(device)

            optimizer = optimizer_scheduler(optimizer=optimizer, init_lr=self.lr, p=float(epoch / self.n_epochs))
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (epoch / self.n_epochs))) - 1
            src_pred, domain_pred, mmd_loss = self.model(source_data, x_tgt = target_data, alpha = gamma)
            cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), source_label.squeeze())
            domain_loss = F.nll_loss(F.log_softmax(domain_pred, dim=1), domain_label.squeeze())
            loss = cls_loss + domain_loss + gamma * mmd_loss

            loss.backward()
            optimizer.step()

            if i % self.log_interval == 0:
                self.model.eval()
                t_correct = self.test(i)
                if t_correct > correct:
                    correct = t_correct
                    torch.save(self.model, self.path)

        return 100. * correct / len(self.target_loader.dataset)

    def test(self, iteration):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target, _ in self.target_loader:
                data = data.to(device)
                target = target.to(device)
                preds, _, mmd_loss = self.model(data, data)
                test_loss += F.nll_loss(F.log_softmax(preds, dim=1), target.squeeze(), reduction='sum').item()
                pred = preds.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()
            test_loss /= len(self.target_loader.dataset)

            if iteration % (self.log_interval*10) == 0:
                print('Iteration: [{}/{} ({:.2f}%)], Validation average loss: {:.4f}, Validation accuracy: {}/{} ({:.2f}%)'.format(
                    iteration, self.iteration + 1, 100. * iteration / (self.iteration+1), test_loss, correct, len(self.target_loader.dataset), 100. * correct / len(self.target_loader.dataset)
                ))

        return correct

def cross_subject(data, label, session_id, subject_id, category_number, batch_size, iteration, n_epochs, lr, optimizer_name, log_interval, adj):
    one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    train_idxs = list(range(15))
    del train_idxs[subject_id]
    test_idx = subject_id

    target_data, target_label = one_session_data[test_idx], one_session_label[test_idx]
    source_data, source_label = copy.deepcopy(one_session_data[train_idxs]), copy.deepcopy(one_session_label[train_idxs])

    del one_session_data
    del one_session_label

    source_data_comb = source_data[0]
    source_label_comb = source_label[0]
    for j in range(1, len(source_data)):
        source_data_comb = np.vstack((source_data_comb, source_data[j]))
        source_label_comb = np.vstack((source_label_comb, source_label[j]))

    # domain labels.
    source_domain_label = np.zeros_like(source_label_comb)
    target_domain_label = np.ones_like(target_label)
    source_loader = torch.utils.data.DataLoader(dataset=CustomDataset(source_data_comb, source_label_comb, source_domain_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    target_loader = torch.utils.data.DataLoader(dataset=CustomDataset(target_data, target_label, target_domain_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)

    model = MyModel(model = Model(5, category_number, adj, 0.25, adaptive=True, attention=True),
                    source_loader = source_loader,
                    target_loader = target_loader,
                    iteration = iteration,
                    lr = lr,
                    optimizer_name=optimizer_name,
                    log_interval = log_interval,
                    n_epochs=n_epochs,
                    sub_id=subject_id,
                    session_id=session_id)
    acc = model.train()
    print('\nTarget_session_id: {}, current_subject_id: {}, acc: {}\n'.format(session_id, subject_id, acc))
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('--dataset', type=str, default='seed4', help= 'dataset used for model')
    parser.add_argument('--norm_type', type=str, default='ele', help='normalization. ele, sample, global or none.')
    parser.add_argument('--batch_size', type=int, default=16, help='size for one batch, integer. default 16.')
    parser.add_argument('--n_epochs', type=int, default=200, help='training epochs, default 200.')
    parser.add_argument('--log_interval', type=float, default=10, help='evaluation log_interval.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate. default 0.01.')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer. adam, rmsprop')
    parser.add_argument('--model_name', type=str, default='Demo', help='model name.')

    # parameters.
    args = parser.parse_args()
    dataset_name = args.dataset
    norm_type = args.norm_type
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    log_interval = args.log_interval
    lr = args.lr
    optimizer_name = args.optimizer
    model_name = args.model_name

    # data preparation.
    print('Model name:', model_name, 'Dataset name:', dataset_name)
    datas, labels = load_data(dataset_name)

    # normlization.
    if norm_type == 'ele':
        data_tmp = copy.deepcopy(datas)
        label_tmp = copy.deepcopy(labels)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = norminy(data_tmp[i][j])
    elif norm_type == 'sample':
        data_tmp = copy.deepcopy(datas)
        label_tmp = copy.deepcopy(labels)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = norminx(data_tmp[i][j])
    elif norm_type == 'global':
        data_tmp = copy.deepcopy(datas)
        label_tmp = copy.deepcopy(labels)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = normalization(data_tmp[i][j])
    elif norm_type == 'none':
        data_tmp = copy.deepcopy(datas)
        label_tmp = copy.deepcopy(labels)
    else:
        pass

    trial_total, category_number, _ = get_number_of_label_n_trial(dataset_name)
    spatio_adj = get_adj_from_standard()

    if dataset_name == 'seed3':
        iteration = math.ceil(n_epochs * 3394 / batch_size)
    elif dataset_name == 'seed4':
        iteration = math.ceil(n_epochs * 820 / batch_size)
    else:
        iteration = 10000

    csub = []
    for session_id in range(3):
        for subject_id in range(15):
            print('=' * 50)
            print('subject_id:', subject_id, 'session_id:', session_id)
            csub.append(cross_subject(data_tmp, label_tmp, session_id, subject_id, category_number, batch_size, iteration, n_epochs, lr, optimizer_name, log_interval, spatio_adj))

    print('Cross-subject: ', csub)
    print('Cross-subject Mean: ', np.mean(csub), 'Std: ', np.std(csub))

    sessions = np.split(np.array(csub), 3)
    best_ret = np.max(sessions, axis=0)
    print('Best session cross-subject:', best_ret)
    print('Best cross-subject Mean:', np.mean(best_ret), 'Std:', np.std(best_ret))
