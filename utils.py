import torch.nn.init as init
import torch
import numpy as np


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False

    def update(self, it, acc):
        self.acc_list += [acc]
        self.iter_list += [it]
        self.save_model = False
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
            self.save_model = True

    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.iter_list += [it]
        self.save_model = False
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U_T = acc_u
            self.best_acc_S_T = acc_s
            self.save_model = True


def log_print(s, log):
    print(s)
    with open(log, 'a') as f:
        f.write(s + '\n')


def synthesize_feature(flow, sm, dataset, opt):
    gen_feat = torch.FloatTensor(dataset.ntest_class * opt.nSample, opt.X_dim)
    gen_label = np.zeros([0])
    with torch.no_grad():
        for i in range(dataset.ntest_class):
            text_feat = np.tile(dataset.test_att[i].astype('float32'), (opt.nSample, 1))
            text_feat = torch.from_numpy(text_feat).cuda()
            sr = sm(text_feat)
            z = torch.randn(opt.nSample, opt.X_dim).cuda()
            # z = z*z.norm(dim=-1, keepdim=True)
            G_sample = flow.reverse_sample(z, sr)
            gen_feat[i*opt.nSample:(i+1)*opt.nSample] = G_sample
            gen_label = np.hstack((gen_label, np.ones([opt.nSample])*i))
    return gen_feat, torch.from_numpy(gen_label.astype(int))



def save_model(it, flow, gs, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_1': flow.state_dict(),
        'state_dict_2': gs.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)
