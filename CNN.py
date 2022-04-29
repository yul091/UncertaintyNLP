import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    def __init__(self, args, tokenizer, num_labels):
        super().__init__()
        self.args = args
        self.class_num = num_labels
        self.kernel_num = args.kernel_num
        if isinstance(args.kernel_sizes, list):
            self.kernel_sizes = args.kernel_sizes # list of kernel sizes
        else: 
            self.kernel_sizes = [int(x) for x in args.kernel_sizes.split(',')] 
        self.in_channels = 1
        self.metric_fc_dim = 50

        # Load pre-trained embedding
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(tokenizer.embs_npa).float(), 
                                                  padding_idx=tokenizer.pad_token_id, freeze=False) 

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.in_channels, self.kernel_num, (K, tokenizer.embs_npa.shape[1])) for K in self.kernel_sizes]
        )
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)

        if args.ensemble:
            self.embed2 = nn.Embedding.from_pretrained(torch.from_numpy(tokenizer.embs_npa).float(), 
                                                       padding_idx=tokenizer.pad_token_id, freeze=False) 

            self.convs2 = nn.ModuleList(
                [nn.Conv2d(self.in_channels, self.kernel_num, (K, tokenizer.embs_npa.shape[1])) for K in self.kernel_sizes]
            )
            self.dropout2 = nn.Dropout(args.dropout)
            self.fc2 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x1 = self.embed(x)  # (N, W, D)
        if self.args.static:
            x1 = Variable(x1)

        x1 = x1.unsqueeze(1)  # (N, Ci, W, D)
        x1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x1]  # [(N, Co), ...]*len(Ks)
        x1 = torch.cat(x1, 1) # (N, len(Ks)*Co)
        x1 = self.dropout(x1)  # (B, len(Ks)*Co)
        logit1 = self.fc(x1)  # (B, V)

        if self.args.ensemble:
            x2 = self.embed2(x)  # (N, W, D)
            if self.args.static:
                x2 = Variable(x2)

            x2 = x2.unsqueeze(1)  # (N, Ci, W, D)
            x2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]  # [(N, Co, W), ...]*len(Ks)
            x2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x2]  # [(N, Co), ...]*len(Ks)
            x2 = torch.cat(x2, 1) # (N, len(Ks)*Co)
            x2 = self.dropout2(x2)  # (B, len(Ks)*Co)
            logit2 = self.fc2(x2)  # (B, V)

            return x1, logit1, x2, logit2 
            
        return x1, logit1