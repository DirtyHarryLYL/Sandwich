import torch
import torch.nn as nn
import numpy as np
import pickle

class flag_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ref = nn.Parameter(torch.from_numpy(pickle.load(open('./models/pairwise_cosine_sim_new.pkl', 'rb'))).float(), requires_grad=False) # C, C, C 
        self.sample_size=100
    def forward(self, scores, targets):
        '''
        scores:  N, C, C
        targets: N, C
        '''
        N, C  = targets.shape
        index = torch.where(targets == 1)

        if len(index[0]) > self.sample_size:
            select_idx = np.random.choice(np.array(range(len(index[0]))), size=self.sample_size, replace=False)
            index = (index[0][select_idx], index[1][select_idx])

        logits = scores[index[0], index[1], :]
        relative_dist = logits[:, :, None].expand(-1, -1, C) - logits[:, None, :].expand(-1, C, -1) # N, C, C
        ref_labels = self.ref[index[1]]
        
        loss_pos = 0
        pos = relative_dist[ref_labels > 0]
        if torch.sum(pos < 0) > 0:
            loss_pos += -pos[pos < 0].mean()
        return loss_pos
if __name__ == '__main__':
    model = flag_loss().cuda()
    a = torch.rand(10, 290, 290).cuda()
    b = (torch.rand(10, 290) > 0.99).long().cuda()
    print(b.sum())
    print(model(a, b))