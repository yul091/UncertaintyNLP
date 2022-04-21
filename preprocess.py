import os
import re
import torch
import numpy as np
from tqdm import tqdm
from utils import load_dataset
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# Parse the vocabulary and embeddings
class GloVe_tokenizer:
    def __init__(self, dict_path='glove.6B.100d.txt', max_len=512):
        self.max_len = max_len
        if not os.path.exists(dict_path):
            raise ValueError("No dictionary exists! Please download from http://nlp.stanford.edu/data/glove.6B.zip")
        else:
            vocab, embeddings = [], []
            with open(dict_path,'rt') as fi:
                full_content = fi.read().strip().split('\n')
            for i in range(len(full_content)):
                i_word = full_content[i].split(' ')[0]
                i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
                vocab.append(i_word) # a list of words, which will be useful in the text to token ids conversion step
                embeddings.append(i_embeddings) # a list of embeddings, this will be used to initialise the embeddings layer

            # Convert the vocabulary and the embeddings into numpy arrays
            vocab_npa = np.array(vocab)
            embs_npa = np.array(embeddings)

            # Add the padding and unknown tokens to the vocab and embeddings arrays
            # insert '<PAD>' and '<UNK>' tokens at start of vocab_npa.
            self.pad_token, self.unk_token = '<PAD>', '<UNK>'
            self.pad_token_id, self.unk_token_id = 0, 1
            vocab_npa = np.insert(vocab_npa, self.pad_token_id, self.pad_token)
            vocab_npa = np.insert(vocab_npa, self.unk_token_id, self.unk_token)
            pad_emb_npa = np.zeros((1, embs_npa.shape[1])) # embedding for '<PAD>' token.
            unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True) # embedding for '<UNK>' token.
            # insert embeddings for PAD and UNK tokens at top of embs_npa.
            self.embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa)) # Vocab X E
            self.word2idx = {word:idx for idx, word in enumerate(vocab_npa)}
            self.idx2word = {idx:word for idx, word in enumerate(vocab_npa)}
                
    def encode(self, word):
        if word not in self.word2idx:
            return self.unk_token_id
        else:
          return self.word2idx[word]

    def decode(self, idx):
        if idx not in self.idx2word:
            return '<UNK>'
        else:
          return self.idx2word[idx]

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def encode_sentence(self, string):
        cleaned_sent = self.clean_str(string)
        words = cleaned_sent.split()[:self.max_len]
        return [self.encode(word) for word in words]



def process_data(args, MAX_LEN):
    train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels = load_dataset(args.dataset)
    
    # Preprocess using pre-trained tokenizer
    train_input_ids, val_input_ids, test_input_ids = [], [], []
    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        for sent in tqdm(train_sentences):
            # Encode: 
            # (1) Tokenize the sentence; 
            # (2) Prepend the '[CLS]' token; 
            # (3) Append the '[SEP]' token; 
            # (4) Map tokens to IDs.
            encoded_sent = tokenizer.encode(
                sent, # Sentence to be encoded
                add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                max_length=MAX_LEN, # Truncate all sentences
            )
            # Add the encoded input_ids to the list
            train_input_ids.append(encoded_sent)

        for sent in val_sentences:
            encoded_sent = tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=MAX_LEN,
            )
            val_input_ids.append(encoded_sent)

        for sent in test_sentences:
            encoded_sent = tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=MAX_LEN,
            )
            test_input_ids.append(encoded_sent)
    else: 
        # Preprocess using GloVe embedding
        tokenizer = GloVe_tokenizer(dict_path=args.dict_path, max_len=MAX_LEN)

        for sent in tqdm(train_sentences): 
            # Encode: 
            # (1) Tokenize the sentence;  
            # (2) Map tokens to IDs.
            encoded_sent = tokenizer.encode_sentence(sent)
            # Add the encoded input_ids to the list
            train_input_ids.append(encoded_sent)

        for sent in val_sentences:
            encoded_sent = tokenizer.encode_sentence(sent)
            val_input_ids.append(encoded_sent)

        for sent in test_sentences:
            encoded_sent = tokenizer.encode_sentence(sent)
            test_input_ids.append(encoded_sent)

    print("train data size {}, val data size {}, test data size {}".format(
        len(train_input_ids), len(val_input_ids), len(test_input_ids)
    ))
    # Pad input_ids
    train_input_ids = pad_sequences(
        train_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", 
        padding="post", value=tokenizer.pad_token_id,
    )
    val_input_ids = pad_sequences(
        val_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", 
        padding="post", value=tokenizer.pad_token_id,
    )
    test_input_ids = pad_sequences(
        test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", 
        padding="post", value=tokenizer.pad_token_id,
    )

    # Create attention masks [1 for token and 0 for padding]
    train_attention_masks = []
    val_attention_masks = []
    test_attention_masks = []

    for seq in train_input_ids:
        seq_mask = [float(i>0) for i in seq]
        train_attention_masks.append(seq_mask)
    for seq in val_input_ids:
        seq_mask = [float(i>0) for i in seq]
        val_attention_masks.append(seq_mask)
    for seq in test_input_ids:
        seq_mask = [float(i>0) for i in seq]
        test_attention_masks.append(seq_mask)

    # Convert all input_ids into torch tensors
    train_inputs = torch.tensor(train_input_ids)
    validation_inputs = torch.tensor(val_input_ids)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(val_labels)
    train_masks = torch.tensor(train_attention_masks)
    validation_masks = torch.tensor(val_attention_masks)
    test_inputs = torch.tensor(test_input_ids)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_attention_masks)

    # Create an iterator of our data with torch DataLoader. 
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    prediction_data = TensorDataset(test_inputs, test_masks, test_labels)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.eval_batch_size)

    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=args.eval_batch_size)

    return tokenizer, train_data, validation_data, prediction_data, train_dataloader, validation_dataloader, prediction_dataloader
