import pandas as pd
import random
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def load_dataset(dataset):
    print("Loading {} dataset ...".format(dataset))
    if dataset == 'sst':
        df_train = pd.read_csv("./dataset/sst/SST-2/train.tsv", delimiter='\t', header=0)
        df_val = pd.read_csv("./dataset/sst/SST-2/dev.tsv", delimiter='\t', header=0)
        df_test = pd.read_csv("./dataset/sst/SST-2/sst-test.tsv", delimiter='\t', header=None, names=['sentence', 'label'])

        train_sentences = df_train.sentence.values
        val_sentences = df_val.sentence.values
        test_sentences = df_test.sentence.values
        train_labels = df_train.label.values
        val_labels = df_val.label.values
        test_labels = df_test.label.values   
    

    if dataset == '20news':
        VALIDATION_SPLIT = 0.8
        newsgroups_train  = fetch_20newsgroups(subset='train',  shuffle=True, random_state=0)
        print(newsgroups_train.target_names)
        # print(len(newsgroups_train.data))
        newsgroups_test  = fetch_20newsgroups(subset='test',  shuffle=False)
        # print(len(newsgroups_test.data))
        train_len = int(VALIDATION_SPLIT * len(newsgroups_train.data))
        train_sentences = newsgroups_train.data[:train_len]
        val_sentences = newsgroups_train.data[train_len:]
        test_sentences = newsgroups_test.data
        train_labels = newsgroups_train.target[:train_len]
        val_labels = newsgroups_train.target[train_len:]
        test_labels = newsgroups_test.target


    if dataset == '20news-15':
        VALIDATION_SPLIT = 0.8
        cats = ['alt.atheism',
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'comp.windows.x',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'misc.forsale',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space']
        newsgroups_train  = fetch_20newsgroups('dataset/20news', subset='train',  shuffle=True, categories=cats, random_state=0)
        print(newsgroups_train.target_names)
        # print(len(newsgroups_train.data))
        newsgroups_test  = fetch_20newsgroups('dataset/20news', subset='test',  shuffle=False, categories=cats)
        # print(len(newsgroups_test.data))
        train_len = int(VALIDATION_SPLIT * len(newsgroups_train.data))
        train_sentences = newsgroups_train.data[:train_len]
        val_sentences = newsgroups_train.data[train_len:]
        test_sentences = newsgroups_test.data
        train_labels = newsgroups_train.target[:train_len]
        val_labels = newsgroups_train.target[train_len:]
        test_labels = newsgroups_test.target


    if dataset == '20news-5':
        cats = [
        'soc.religion.christian',
        'talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc',
        'talk.religion.misc']
              
        newsgroups_test  = fetch_20newsgroups('dataset/20news', subset='test',  shuffle=False, categories=cats)
        print(newsgroups_test.target_names)
        # print(len(newsgroups_test.data))
        train_sentences = None
        val_sentences = None
        test_sentences = newsgroups_test.data
        train_labels = None
        val_labels = None
        test_labels = newsgroups_test.target


    if dataset == 'wos':
        TESTING_SPLIT = 0.6
        VALIDATION_SPLIT = 0.8
        file_path = './dataset/WebOfScience/WOS46985/X.txt'
        with open(file_path, 'r') as read_file:
            x_temp = read_file.readlines()
            x_all = []
            for x in x_temp:
                x_all.append(str(x))

        # print(len(x_all))
        file_path = './dataset/WebOfScience/WOS46985/Y.txt'
        with open(file_path, 'r') as read_file:
            y_temp= read_file.readlines()
            y_all = []
            for y in y_temp:
                y_all.append(int(y))
        # print(len(y_all))
        # print(max(y_all), min(y_all))

        x_in = []
        y_in = []
        for i in range(len(x_all)):
            x_in.append(x_all[i])
            y_in.append(y_all[i])

        train_val_len = int(TESTING_SPLIT * len(x_in))
        train_len = int(VALIDATION_SPLIT * train_val_len)

        train_sentences = x_in[:train_len]
        val_sentences = x_in[train_len:train_val_len]
        test_sentences = x_in[train_val_len:]

        train_labels = y_in[:train_len]
        val_labels = y_in[train_len:train_val_len]
        test_labels = y_in[train_val_len:]

        # print(len(train_labels))
        # print(len(val_labels))
        # print(len(test_labels))


    if dataset == 'wos-100':
        TESTING_SPLIT = 0.6
        VALIDATION_SPLIT = 0.8
        file_path = './dataset/WebOfScience/WOS46985/X.txt'
        with open(file_path, 'r') as read_file:
            x_temp = read_file.readlines()
            x_all = []
            for x in x_temp:
                x_all.append(str(x))

        # print(len(x_all))
        file_path = './dataset/WebOfScience/WOS46985/Y.txt'
        with open(file_path, 'r') as read_file:
            y_temp= read_file.readlines()
            y_all = []
            for y in y_temp:
                y_all.append(int(y))
        # print(len(y_all))
        # print(max(y_all), min(y_all))

        x_in = []
        y_in = []
        for i in range(len(x_all)):
            if y_all[i] in range(100):
                x_in.append(x_all[i])
                y_in.append(y_all[i])

        for i in range(133):
            num = 0
            for y in y_in:
                if y == i:
                    num = num + 1
            # print(num)

        train_val_len = int(TESTING_SPLIT * len(x_in))
        train_len = int(VALIDATION_SPLIT * train_val_len)

        train_sentences = x_in[:train_len]
        val_sentences = x_in[train_len:train_val_len]
        test_sentences = x_in[train_val_len:]

        train_labels = y_in[:train_len]
        val_labels = y_in[train_len:train_val_len]
        test_labels = y_in[train_val_len:]

        # print(len(train_labels))
        # print(len(val_labels))
        # print(len(test_labels))


    if dataset == 'wos-34':
        TESTING_SPLIT = 0.6
        VALIDATION_SPLIT = 0.8
        file_path = './dataset/WebOfScience/WOS46985/X.txt'
        with open(file_path, 'r') as read_file:
            x_temp = read_file.readlines()
            x_all = []
            for x in x_temp:
                x_all.append(str(x))

        # print(len(x_all))
        file_path = './dataset/WebOfScience/WOS46985/Y.txt'
        with open(file_path, 'r') as read_file:
            y_temp= read_file.readlines()
            y_all = []
            for y in y_temp:
                y_all.append(int(y))
        # print(len(y_all))
        # print(max(y_all), min(y_all))

        x_in = []
        y_in = []
        for i in range(len(x_all)):
            if (y_all[i] in range(100)) != True:
                x_in.append(x_all[i])
                y_in.append(y_all[i])

        for i in range(133):
            num = 0
            for y in y_in:
                if y == i:
                    num = num + 1
            # print(num)

        train_val_len = int(TESTING_SPLIT * len(x_in))
        train_len = int(VALIDATION_SPLIT * train_val_len)

        train_sentences = None
        val_sentences = None
        test_sentences = x_in[train_val_len:]
        
        train_labels = None
        val_labels = None
        test_labels = y_in[train_val_len:]
        # print(len(test_labels))
        

    if dataset == 'agnews':
        VALIDATION_SPLIT = 0.8
        labels_in_domain = [1, 2]

        train_df = pd.read_csv('./dataset/agnews/train.csv', header=None)
        train_df.rename(columns={0: 'label',1: 'title', 2:'sentence'}, inplace=True)
        # train_df = pd.concat([train_df, pd.get_dummies(train_df['label'],prefix='label')], axis=1)
        print(train_df.dtypes)
        train_in_df_sentence = []
        train_in_df_label = []
        
        for i in range(len(train_df.sentence.values)):
            sentence_temp = ''.join(str(train_df.sentence.values[i]))
            train_in_df_sentence.append(sentence_temp)
            train_in_df_label.append(train_df.label.values[i]-1)

        test_df = pd.read_csv('./dataset/agnews/test.csv', header=None)
        test_df.rename(columns={0: 'label',1: 'title', 2:'sentence'}, inplace=True)
        # test_df = pd.concat([test_df, pd.get_dummies(test_df['label'],prefix='label')], axis=1)
        test_in_df_sentence = []
        test_in_df_label = []
        for i in range(len(test_df.sentence.values)):
            test_in_df_sentence.append(str(test_df.sentence.values[i]))
            test_in_df_label.append(test_df.label.values[i]-1)

        train_len = int(VALIDATION_SPLIT * len(train_in_df_sentence))

        train_sentences = train_in_df_sentence[:train_len]
        val_sentences = train_in_df_sentence[train_len:]
        test_sentences = test_in_df_sentence
        train_labels = train_in_df_label[:train_len]
        val_labels = train_in_df_label[train_len:]
        test_labels = test_in_df_label
        # print(len(train_sentences))
        # print(len(val_sentences))
        # print(len(test_sentences))

    return train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels