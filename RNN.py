import torch
import torch.nn as nn


class BiGRUForSequenceClassification(nn.Module):
    def __init__(self, tokenizer, args, num_labels):
        super().__init__()
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.ensemble = args.ensemble

        # Load pre-trained embedding
        self.tok_embed = nn.Embedding.from_pretrained(torch.from_numpy(tokenizer.embs_npa).float(), 
                                                      padding_idx=tokenizer.pad_token_id, freeze=False) 
        # Bi-directional GRU
        self.rnn = nn.GRU(input_size=tokenizer.embs_npa.shape[1], hidden_size=args.hidden_size,
                          num_layers=args.num_layers, bidirectional=True, batch_first=True) 
        
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(in_features=args.hidden_size, out_features=num_labels)

        # Self-ensembling
        if args.ensemble:
            self.tok_embed2 = nn.Embedding.from_pretrained(torch.from_numpy(tokenizer.embs_npa).float(), 
                                                           padding_idx=tokenizer.pad_token_id, freeze=False) 
            
            self.rnn2 = nn.GRU(input_size=tokenizer.embs_npa.shape[1], hidden_size=args.hidden_size,
                               num_layers=args.num_layers, bidirectional=True, batch_first=True) 
            
            self.dropout2 = nn.Dropout(args.dropout)
            self.fc2 = nn.Linear(in_features=args.hidden_size, out_features=num_labels)
            

    def forward(self, inputs_ids, inputs_embeds=None):
        """
        inputs_ids (Tensor): B X T
        """
        if inputs_embeds is None: 
            # Embedding
            inputs_embeds = self.tok_embed(inputs_ids) # B X T X E
        # Sequence Modeling with dropout
        _, h = self.rnn(inputs_embeds) # 2*layers X B X H
        h = h.view(self.num_layers, 2, inputs_ids.size(0), self.hidden_size) # layers X 2 X B X H
        enc = h[-1, :, :, :].mean(dim=0) # Average the bi-direction -> B X H
        # Dropout
        enc = self.dropout(enc)
        # FC projection
        logits = self.fc(enc) # B X V

        if self.ensemble:
            if inputs_embeds is None: 
                inputs_embeds = self.tok_embed2(inputs_ids) # B X T X E
            
            _, h2 = self.rnn2(inputs_embeds) # 2*layers X B X H
            h2 = h2.view(self.num_layers, 2, inputs_ids.size(0), self.hidden_size) # layers X 2 X B X H
            enc2 = h2[-1, :, :, :].mean(dim=0) # B X H
            enc2 = self.dropout2(enc2)
            logits2 = self.fc2(enc2) # B X V

            return enc, logits, enc2, logits2
        
        return enc, logits