import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from metric import accurate_nb, ECE, multiclass_metric_loss


def evaluator(dataloader, model, args, Mahala=None, use_transform=False):
    # Validation
    ece_criterion = ECE().to(args.device)
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    # Tracking variables 
    eval_accurate_nb = 0
    nb_eval_examples = 0
    logits_list = []
    labels_list = []

    # Evaluate data for one epoch
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(args.device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch # B X T, B X T, B
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
        # Forward pass, calculate logit predictions
            if args.model != 'bert':
                res = model(b_input_ids) 
                hiddens, logits = res[0], res[1] # B X H, B X V
            else:
                hiddens = model.model1.bert(b_input_ids, attention_mask=b_input_mask)[1]
                hiddens = model.model1.dropout(hiddens) # B X H
                logits = model.model1.classifier(hiddens) # B X V

            # Transform representations (mahalanobis dist + winning score)
            if use_transform and Mahala is not None:
                # print("Generating transformed feature representation ...")
                score_list, x_list = [], []
                # Caculate winning score
                softmax = F.softmax(logits, dim=1).detach().cpu() # B X V
                winning_scores = torch.max(softmax, dim=1)[0] # B
                for sample_idx in range(args.n_samples):
                    x_init = (hiddens + torch.zeros_like(hiddens).uniform_(args.epsilon_x, args.epsilon_x)).detach().cpu() # B X H 
                    scores = Mahala._uncertainty_calculate(x_init) + winning_scores # B
                    x_list.append(x_init) # list of (B X H)
                    score_list.append(scores.unsqueeze(0)) # list of (1 X B)

                # Select representations that produce minimum uncertainty
                score_list = torch.cat(score_list, dim=0) # n_samples X B
                max_idx = torch.max(score_list, dim=0)[1] # B
                trans_hiddens = [x_list[idx][i].unsqueeze(0) for i, idx in enumerate(max_idx)] # list of (1 X H)
                trans_hiddens = torch.cat(trans_hiddens, dim=0).to(args.device) # B X H
                if args.model != 'bert':
                    logits = model.fc(trans_hiddens) # B X V
                else:
                    logits = model.model1.classifier(trans_hiddens) # B X V


        logits_list.append(logits)
        labels_list.append(b_labels)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        # Calculate predicting accuracy
        tmp_eval_nb = accurate_nb(logits, label_ids)
        eval_accurate_nb += tmp_eval_nb
        nb_eval_examples += label_ids.shape[0]
        
    eval_accuracy = eval_accurate_nb/nb_eval_examples
    logits_ece = torch.cat(logits_list)
    labels_ece = torch.cat(labels_list)
    ece = ece_criterion(logits_ece, labels_ece).item()

    return eval_accuracy, ece




def Trainer(train_dataloader, 
            validation_dataloader, 
            prediction_dataloader, 
            model, 
            args, 
            dirname, 
            num_labels):
    """
    dataset (torch.TensorDataset): 
        train_inputs (T), train_masks (T), train_labels (1)
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

    # Define saved model name
    if not args.ensemble and args.coeff == 0:
        model_save = 'model.pt'
    elif args.ensemble and args.coeff == 0:
        model_save = 'model-ensemble{}.pt'.format(args.cross_rate)
    elif not args.ensemble and args.coeff != 0:
        model_save = 'model-metric{}.pt'.format(args.coeff)
    else:
        model_save = 'model-ensemble{}-metric{}.pt'.format(args.cross_rate, args.coeff)

    # Store our loss and accuracy for plotting
    best_val = -np.inf
    # trange is a tqdm wrapper around the normal python range
    for epoch in range(args.epochs): 
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        # Tracking variables
        tr_loss =  0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()

        # Train the data for one epoch
        for step, batch in tqdm(enumerate(train_dataloader)):
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch # B X T, B X T, B
            optimizer.zero_grad()

            # Compute hiddens and CE loss
            if args.model != 'bert':
                # target_onehot = F.one_hot(b_labels, num_classes=num_labels) # B X V
                res = model(b_input_ids) 
                hiddens, logits = res[0], res[1] # B X H, B X V
                loss_ce = F.cross_entropy(logits, b_labels)
                if args.ensemble:
                    hiddens2, logits2 = res[2], res[3] # B X H, B X V
                    loss_ce2 = F.cross_entropy(logits2, b_labels)
            else:
                if not args.ensemble:
                    res1 = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                else:
                    res1, res2 = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    loss_ce2, logits2 = res2[0], res2[1] # B X H, B X V
                    hiddens2 = model.model2.bert(b_input_ids, attention_mask=b_input_mask)[1]
                    hiddens2 = model.model2.dropout(hiddens2) # B X H

                loss_ce, logits = res1[0], res1[1] # B X H, B X V
                hiddens = model.model1.bert(b_input_ids, attention_mask=b_input_mask)[1]
                hiddens = model.model1.dropout(hiddens) # B X H

            if torch.cuda.device_count() > 1:
                loss_ce = loss_ce.mean()

            loss = loss_ce
            if args.coeff != 0 :
                # Compute metric loss
                loss_metric = multiclass_metric_loss(hiddens, b_labels, args, num_labels)
                loss += args.coeff * loss_metric

            if args.ensemble:
                # Combine all losses
                loss += loss_ce2

                if args.coeff != 0:
                    loss_metric2 = multiclass_metric_loss(hiddens2, b_labels, args, num_labels)
                    loss += args.coeff * loss_metric2
                if args.cross_rate != 0:
                    # Compute ensemble loss
                    loss_between = F.mse_loss(logits, logits2)
                    loss += args.cross_rate * loss_between

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("train CE loss: {}".format(tr_loss/nb_tr_steps))

        # Evaluate on validation data
        eval_accuracy, ece = evaluator(validation_dataloader, model, args)
        scheduler.step(eval_accuracy) # Adjust learning rate if a metric has stopped improving
        print("val acc: {}, val ECE: {}".format(eval_accuracy, ece))

        if eval_accuracy > best_val: # Store model if best performance
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            print("Saving model to %s" % dirname)
            
            if args.model != 'bert':
                torch.save(model.state_dict(), os.path.join(dirname, model_save))
            else:
                if not args.ensemble and args.coeff == 0:
                    dirname = os.path.join(dirname, 'basic')
                elif args.ensemble and args.coeff == 0:
                    dirname = os.path.join(dirname, 'ensemble')
                elif not args.ensemble and args.coeff != 0:
                    dirname = os.path.join(dirname, 'metric')
                else:
                    dirname = os.path.join(dirname, 'metric-ensemble')

                model_to_save = model.module if hasattr(model, 'module') else model 
                model_to_save.save_pretrained(dirname)   

            best_val = eval_accuracy

        # Test model on test data
        eval_accuracy, ece = evaluator(prediction_dataloader, model, args)
        print("test acc: {}, test ECE: {}".format(eval_accuracy, ece))