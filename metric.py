import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPRegressor


# Function to calculate the accuracy of our predictions vs labels
def accurate_nb(preds, labels):
    pred_flat = np.argmax(preds, axis=-1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


# Define ECE
class ECE(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def distance(hidden1, hidden2):
    """
    hidden1/hidden2 (H / B X H) -> (float / B)
    """
    dim = hidden1.shape[-1]
    if hidden1.dim() == 1 and hidden1.dim() == hidden2.dim():
        # H -> float
        return (hidden1 - hidden2).norm(p=2).square() / dim

    elif hidden1.dim() == 2 and hidden1.dim() == hidden2.dim(): 
        # B X H -> B
        return (hidden1 - hidden2).norm(p=2, dim=1).square() / dim 
 
    else: 
        raise ValueError("Dimension should be smaller than 2!")



def multiclass_metric_loss(hidden, target, args, num_labels):
    dim = hidden.shape[1] # H
    indices = []
    # Record all indices for each class
    for class_idx in range(num_labels):
        indice_i = torch.where(target == class_idx)[0].tolist()
        indices.append(indice_i)

    loss_metric = Variable(torch.FloatTensor([0])).squeeze().to(args.device)
    num_trips = 0
    for class_idx in range(num_labels): # for each class_idx
        # intra class loss
        intra_ids = indices[class_idx] # all the indices for this class
        num_intra_examples = len(intra_ids) # the number of instances
        for anchor in range(num_intra_examples):
            # calculate L2 distance of h_a to any other h_n -> B
            neg_dists = (hidden[intra_ids[anchor]] - hidden).norm(p=2, dim=1).square() / dim
            for pos in range(anchor + 1, num_intra_examples):
                # calculate L2 distance of (h_a, h_p) -> float num
                pos_dist = distance(hidden[intra_ids[anchor]], hidden[intra_ids[pos]])
                # random sample negative samples
                neg_dists[intra_ids] = np.NaN # mask the intra examples
                inter_ids = torch.where(neg_dists-pos_dist < args.alpha)[0] # hard negative examples 

                if inter_ids.shape[0] > 0:
                    n_idx = random.choice(inter_ids.tolist()) # random choose one negative instance
                    loss_metric += torch.clamp(args.alpha + pos_dist - neg_dists[n_idx], min=0)
                    num_trips += 1

    if num_trips > 0:
        loss_metric = loss_metric / num_trips

    return loss_metric



# Define distinctiveness score
class Mahalanobis:
    def __init__(self, train_dataloader, model, num_labels, args):
        super().__init__()
        self.hidden_num = 1
        self.model = model
        self.class_num = num_labels
        self.device = args.device
        self.args = args
        self.train_dataloader = train_dataloader
        # Get train data distribution
        self.u_list, self.std_inverse, self.train_fx, self.train_gt = self.get_distribution() 
        # project scores to [0, 1]
        self.lr = self.train_logic()

    def train_logic(self):
        train_scores = self.mahalanobis_dist(self.train_dataloader, is_train=True) # N
        train_scores = train_scores.reshape(-1, self.hidden_num) # N X 1
        lr = LogisticRegression(C=1.0, penalty='l2', tol=0.01)
        lr.fit(train_scores, self.train_gt)
        print(lr.score(train_scores, self.train_gt))
        return lr

    def get_distribution(self):
        fx, y, gt = self.get_feature(self.train_dataloader) # N X H, N
        u_list, std_list = [], []
        for target in range(self.class_num): # for class i
            fx_tar = fx[torch.where(y == target)] # all features in class i -> B' X H
            mean_val = torch.mean(fx_tar.float(), dim = 0) # mu i -> H
            std_val = (fx_tar - mean_val).transpose(-2,-1) @ (fx_tar - mean_val) # sigma i -> H X H
            u_list.append(mean_val)
            std_list.append(std_val)

        std_inverse = torch.inverse(sum(std_list) / len(y)) # H X H
        return u_list, std_inverse, fx, gt.numpy()

    def get_feature(self, data_loader):
        fx, y_list, gt_list = [], [], []
        self.model.eval()
        for i, batch in enumerate(data_loader):
            batch = tuple(t.to(self.device) for t in batch) 
            b_input_ids, b_input_mask, b_labels = batch # B X T, B X T, B

            # Telling the model not to compute or store gradients
            with torch.no_grad():
                if self.args.model != 'bert':
                    res = self.model(b_input_ids) 
                    hiddens, logits = res[0], res[1] # B X H, B X V
                else:
                    hiddens = self.model.model1.bert(b_input_ids, attention_mask=b_input_mask)[1]
                    hiddens = self.model.model1.dropout(hiddens) # B X H
                    logits = self.model.model1.classifier(hiddens) # B X V

                # Calculate ground truth
                preds = torch.max(logits, dim=1)[1] # B
                gt = (preds == b_labels).int().detach().cpu() # B
                
            # Move logits and labels to CPU
            features = hiddens.detach().cpu()
            label_ids = b_labels.cpu()
            fx.append(features)
            y_list.append(label_ids)
            gt_list.append(gt)
        
        return torch.cat(fx, dim=0), torch.cat(y_list, dim=0), torch.cat(gt_list, dim=0)


    def mahalanobis_dist(self, data_loader, is_train=False):
        """
        Calculate mahalanobis distance: max_{c}{(f(x)-mu_c) sigma (f(x)-mu_c)},
        between test sample x and the closest class-conditional Gaussian distribution.
        """
        if is_train:
            fx = self.train_fx
        else:
            fx, _ = self.get_feature(data_loader) # N X H
        score = []
        for target in range(self.class_num): # for each class
            tmp = (fx - self.u_list[target]) @ self.std_inverse @ (fx - self.u_list[target]).transpose(-2, -1) # N X N
            tmp = tmp.diagonal().reshape([-1, 1]) # N X 1
            score.append(-tmp)

        score = torch.cat(score, dim=1) # N X V
        score = torch.max(score, dim=1)[0].detach().cpu().numpy() # N
        return score

    def _uncertainty_calculate(self, fx):
        """
        fx (B X H / B X T X H): feature representations of RNN / transformers.
        """
        score = []
        for target in range(self.class_num): # for each class
            tmp = (fx - self.u_list[target]) @ self.std_inverse @ (fx - self.u_list[target]).transpose(-2, -1) # N X N
            tmp = tmp.diagonal().reshape([-1, 1]) # N X 1
            score.append(-tmp)

        score = torch.cat(score, dim=1) # N X V
        score = torch.max(score, dim=1)[0].detach().cpu().numpy() # N
        score = score.reshape([-1, self.hidden_num]) # N X 1
        pred_score = self.lr.predict_proba(score) # N X 2
        # pred = self.lr.predict(score) # N X 2
        return torch.from_numpy(pred_score[:,1]) # confidence # N
