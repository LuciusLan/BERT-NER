"""BERT NER Inference."""

from __future__ import absolute_import, division, print_function

import json
import os

import torch
from torch import nn
import torch.nn.functional as F
from nltk import word_tokenize
from transformers import AutoModel, AutoConfig, AutoTokenizer


LABEL_2_ID = {'PAD':0, 'O': 1, 'MISC': 2, 'PER': 3,
              'ORG': 4, 'LOC': 5, 'SP': 6}
LABEL_BIO = {'<PAD>': 0, 'O': 1, 'B-MISC': 2, 'I-MISC': 3, 'B-PER': 4, 'I-PER': 5, 'B-ORG': 6, 'I-ORG': 7, 
             'B-LOC': 8, 'I-LOC': 9, '[CLS]': 10, '[SEP]': 11} #, '<START>': 18, '<STOP>': 19}


MAX_LEN = 128
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
NUM_EPOCH = 8
LSTM_HIDDEN = 100
BIAFFINE_DROPOUT = 0.3
BASELINE = False
LONGBERT = False # True when using Longformer / Bigbird
MODEL_CACHE_DIR = 'D:\\Dev\\bert-base-cased'
MAX_SPAN_LEN = 6

class TModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.baseline = BASELINE
        if LONGBERT:
            self.transformer = AutoModel.from_pretrained(
                pretrained_model_name_or_path="allenai/longformer-base-4096", cache_dir=MODEL_CACHE_DIR, config=config)
        else:
            self.transformer = AutoModel.from_pretrained(
                pretrained_model_name_or_path="bert-base-cased", cache_dir=MODEL_CACHE_DIR, config=config, add_pooling_layer=False)
        self.dropout = nn.Dropout(BIAFFINE_DROPOUT)
        self.relu = nn.ReLU(True)
        if BASELINE:
            #self.ner = nn.LSTM(bidirectional=True, input_size=config.hidden_size, hidden_size=LSTM_HIDDEN, batch_first=True)
            self.ner = nn.Linear(config.hidden_size, len(LABEL_BIO))
        else:
            self.boundary_encoder = nn.LSTM(bidirectional=True, input_size=config.hidden_size, hidden_size=LSTM_HIDDEN, batch_first=True)
            self.boundary_decoder = nn.LSTM(bidirectional=False, input_size=LSTM_HIDDEN*2, hidden_size=LSTM_HIDDEN, batch_first=True)
            self.boundary_biaffine = BoundaryBiaffine(LSTM_HIDDEN, LSTM_HIDDEN*2, 1)
            self.boundary_seg = BoundarySeg()
            self.boundary_final0 = nn.Linear(config.hidden_size, LSTM_HIDDEN*2)
            self.boundary_final1 = nn.Linear(LSTM_HIDDEN*2, LSTM_HIDDEN*2)
            self.boundary_fc = nn.Linear(LSTM_HIDDEN*2, 2)

            # No *2 since the boundary decoder can only be unidirectioal
            self.seg_final0 = nn.Linear(config.hidden_size, LSTM_HIDDEN*4)
            self.seg_final1 = nn.Linear(LSTM_HIDDEN*4, LSTM_HIDDEN*4)
            self.type_lstm = nn.LSTM(bidirectional=True, input_size=config.hidden_size, hidden_size=LSTM_HIDDEN, batch_first=True)
            self.type_final0 = nn.Linear(config.hidden_size, LSTM_HIDDEN*2)
            self.type_final1 = nn.Linear(LSTM_HIDDEN*2, LSTM_HIDDEN*2)
            self.type_fc = nn.Linear(LSTM_HIDDEN*2, len(LABEL_2_ID))
            self.ner_final = nn.Linear(LSTM_HIDDEN*8+config.hidden_size, len(LABEL_BIO))
        self.get_trigram = nn.Conv1d(LSTM_HIDDEN*2, LSTM_HIDDEN*2, 3, padding=1, bias=False)
        self.get_trigram.weight = torch.nn.Parameter(torch.ones([LSTM_HIDDEN*2, LSTM_HIDDEN*2, 3]), requires_grad=False)
        self.get_trigram.requires_grad_ = False
        
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if BASELINE:
            ner_result = self.dropout(self.ner(sequence_output))
            return ner_result
        else:
            boundary_hidden = self.boundary_encoder(sequence_output)[0]
            boundary_hidden = self.dropout(boundary_hidden)

            #eq 8
            seg_result = self.get_trigram(boundary_hidden.transpose(1,2)).transpose(1,2)
            seg_result = self.boundary_decoder(seg_result)[0]
            seg_result = self.dropout(seg_result)

            #eq 9 with softmax normalization
            seg_result = F.softmax(self.boundary_biaffine(seg_result, boundary_hidden), dim=2)
            #eq 10
            seg_result = self.boundary_seg(seg_result, boundary_hidden)
            
            type_hidden = self.type_lstm(sequence_output)[0]
            type_hidden = self.dropout(type_hidden)

            #eq 3
            boundary_result = F.logsigmoid(self.boundary_final0(sequence_output)+self.boundary_final1(boundary_hidden)).mul(boundary_hidden)
            type_result = F.logsigmoid(self.type_final0(sequence_output)+self.type_final1(type_hidden)).mul(type_hidden)
            seg_result = F.logsigmoid(self.seg_final0(sequence_output)+self.seg_final1(seg_result)).mul(seg_result)
            #eq 4
            ner_result = self.ner_final(torch.cat([sequence_output, boundary_result, type_result, seg_result], dim=-1))
            #del seg_result, boundary_result, type_result
            #torch.cuda.empty_cache()
            return ner_result, self.boundary_fc(boundary_hidden), self.type_fc(type_hidden)


class BoundarySeg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, span_adjacency, bound_hidden):
        temp = []
        max_len = bound_hidden.size(1)
        for j in range(max_len):
            j_sum = []
            for i in range(j, min(j+MAX_SPAN_LEN, max_len)):
                # eq 10
                result = torch.cat([bound_hidden[:, i], bound_hidden[:, j]], 1)
                #result = torch.cat([bound_hidden[:, i], bound_hidden[:, j], (bound_hidden[:, i]-bound_hidden[:, j]),
                #                    bound_hidden[:, i]*bound_hidden[:, j]], 1)
                result = result * span_adjacency[:, j, i]
                j_sum.append(result)
            temp.append(torch.sum(torch.stack(j_sum, dim=0), dim=0))
        final = torch.stack(temp, 1)
        return final


class PairwiseBilinear(nn.Module):
    """ A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)"""
    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.zeros(input1_size, input2_size, output_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=True) if bias else 0

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        intermediate = torch.mm(input1.contiguous().view(-1, input1_size[-1]), self.weight.view(-1, self.input2_size * self.output_size))
        # (N x L2 x D2) -> (N x D2 x L2)
        input2 = input2.transpose(1, 2)
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3).contiguous()
        # (N x L1 x L2 x O) + (O) -> (N x L1 x L2 x O)
        output = output + self.bias

        return output

class BoundaryBiaffine(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = PairwiseBilinear(input1_size, input2_size, output_size)
        self.U = nn.Linear(input1_size, output_size, bias=False)
        self.V = nn.Linear(input2_size, output_size, bias=False)

    def forward(self, input1, input2):
        # Changed from original pairwise biaffine, U is only on input1 (d_j) V is only on input2 (h_i^Bdy)
        return self.W_bilin(input1, input2).add(self.U(input1).unsqueeze(2)).add(self.V(input2).unsqueeze(1))


class Ner:

    def __init__(self,model_dir: str):
        self.model , self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k):v for k,v in self.label_map.items()}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = AutoConfig.from_pretrained('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = TModel(model_config)
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0,"[CLS]")
        valid_positions.insert(0,1)
        ## insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids,input_mask,segment_ids,valid_positions

    def predict(self, text: str):
        input_ids,input_mask,segment_ids,valid_ids = self.preprocess(text)
        input_ids = torch.tensor([input_ids],dtype=torch.long,device=self.device)
        input_mask = torch.tensor([input_mask],dtype=torch.long,device=self.device)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long,device=self.device)
        valid_ids = torch.tensor([valid_ids],dtype=torch.long,device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]

        logits = []
        pos = 0
        for index,mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos],logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label],confidence) for label,confidence in logits]
        words = word_tokenize(text)
        assert len(labels) == len(words)
        output = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]
        return output

