import os
import torch
import torch.nn as nn
import argparse
from utils import set_seed
from metric import Mahalanobis
from preprocess import process_data
from RNN import BiGRUForSequenceClassification
from bert import BertEnsembleModel
from trainer import Trainer, evaluator
from transformers import AutoConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs for training.")
    parser.add_argument("--seed", default=0, type=int, help="Number of epochs for training.")
    parser.add_argument("--dataset", default='20news', type=str, help="dataset")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay if we apply some.")
    # parser.add_argument("--beta_on", default=1., type=float, help="Weight of on manifold reg")
    # parser.add_argument("--beta_off", default=1., type=float, help="Weight of off manifold reg")
    # parser.add_argument("--eps_in", default=1e-4, type=float, help="Perturbation size of on-manifold regularizer")
    # parser.add_argument("--eps_y", default=0.1, type=float, help="Perturbation size of label")
    # parser.add_argument('--eps_out', default=0.001, type=float, help="Perturbation size of out-of-domain adversarial training")
    parser.add_argument('--saved_dataset', type=str, default='n', help='Whether save the preprocessed pt file of the dataset')
    parser.add_argument('--model', default='bert', choices=['bert', 'rnn'], type=str, help="Model architecture for training.")
    parser.add_argument('--dict_path', default='glove.6B.200d.txt', type=str, help="File path of GloVe pre-trained embedding.")
    parser.add_argument('--from_scratch', action='store_true', default=False, help="Wether to train from scratch or load checkpoint.")
    parser.add_argument('--ensemble', action='store_true', default=False, help="Wether to use self-ensembling for training.")
    parser.add_argument('--coeff', default=0.1, type=float, help="Weight for metric loss.")
    parser.add_argument('--alpha', default=0.5, type=float, help="Margin enforced between positive and negative pairs.")
    parser.add_argument('--cross_rate', default=1.0, type=float, help="Weight for ensemble loss.")
    parser.add_argument('--n_samples', default=100, type=int, help="How many samples to generate to transform each test input.")
    parser.add_argument('--epsilon_x', default=1.0, type=float, help="The radius of the sphare for generating transformed inputs.")
    parser.add_argument('--num_layers', default=2, type=int, help="Number of layers of RNN model.")
    parser.add_argument('--hidden_size', default=256, type=int, help="Hidden dimension of RNN model.")
    parser.add_argument('--dropout', default=0.2, type=float, help="Dropout rate of the dropout layer in RNN.")


    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)


    # Load dataset and preprocess dataset
    if args.dataset == '20news' or args.dataset == '20news-15':
        MAX_LEN = 150
    else:
        MAX_LEN = 256

    tokenizer, train_data, validation_data, prediction_data, train_dataloader, validation_dataloader, prediction_dataloader = process_data(args, MAX_LEN)

    if args.saved_dataset == 'n':
        dataset_dir = 'dataset/{}'.format(args.dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        torch.save(train_data, dataset_dir+'/train.pt')
        torch.save(validation_data, dataset_dir+'/val.pt')
        torch.save(prediction_data, dataset_dir+'/test.pt')

    else:
        dataset_dir = 'dataset/{}'.format(args.dataset)
        train_data = torch.load(dataset_dir+'/train.pt')
        validation_data = torch.load(dataset_dir+'/val.pt')
        prediction_data = torch.load(dataset_dir+'/test.pt')

    if args.dataset == '20news':
        num_labels = 20
    elif args.dataset == '20news-15':
        num_labels = 15
    elif args.dataset == 'wos-100':
        num_labels = 100
    elif args.dataset == 'wos':
        num_labels = 134

    print("number of labels: {}".format(num_labels))

    # Define model
    if args.model == 'bert':
        # Load pre-trained model
        config = AutoConfig.from_pretrained(
            "bert-base-uncased", 
            num_labels=num_labels, 
            output_hidden_states=True,
        )
        model = BertEnsembleModel(
            "bert-base-uncased",  
            config=config, 
            args=args,
        )
    else:
        model = BiGRUForSequenceClassification(
            tokenizer=tokenizer, 
            args=args,
            num_labels=num_labels, 
        )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # Train
    output_dir = 'checkpoint/{}/{}-{}'.format(args.dataset, args.model, args.seed) # save dir 
    if args.from_scratch or not os.path.exists(output_dir):
        # Start training
        Trainer(
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            prediction_dataloader=prediction_dataloader,
            model=model,
            args=args,
            dirname=output_dir,
            num_labels=num_labels,
        )
    
    # Load checkpoint
    if args.model == 'rnn':
        # model_save = os.listdir(output_dir)[-1]
        # Define saved model name
        if not args.ensemble and args.coeff == 0:
            model_save = 'model.pt'
        elif args.ensemble and args.coeff == 0:
            model_save = 'model-ensemble1.0.pt'
        elif not args.ensemble and args.coeff != 0:
            model_save = 'model-metric.pt'
        else:
            model_save = 'model-ensemble-metric.pt'

        ckpt_path = os.path.join(output_dir, model_save)
        model.load_state_dict(torch.load(ckpt_path))
    else:
        model = BertEnsembleModel(
            output_dir, # checkpoint dir
            config=config, 
            args=args,
        )
        model.to(device)
        
    
    # Evaluate
    eval_accuracy, ece = evaluator(validation_dataloader, model, args)
    print("(normal) val acc: {:.4f}, val ECE: {:.4f}".format(eval_accuracy, ece))
    eval_accuracy, ece = evaluator(prediction_dataloader, model, args)
    print("(normal) test acc: {:.4f}, test ECE: {:.4f}".format(eval_accuracy, ece))

    # Test transform
    MD = Mahalanobis(train_dataloader, model, num_labels, args)
    eval_accuracy, ece = evaluator(validation_dataloader, model, args, Mahala=MD, use_transform=True)
    print("(transformed) val acc: {:.4f}, val ECE: {:.4f}".format(eval_accuracy, ece))
    eval_accuracy, ece = evaluator(prediction_dataloader, model, args, Mahala=MD, use_transform=True)
    print("(transformed) test acc: {:.4f}, test ECE: {:.4f}".format(eval_accuracy, ece))



if __name__ == "__main__":
    main()