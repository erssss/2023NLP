# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch


class DataPrecessForSentence(Dataset):
    def __init__(self, bert_tokenizer, df, max_seq_len=50):
        super(DataPrecessForSentence, self).__init__()
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_seq_len
        (
            self.input_ids,
            self.attention_mask,
            self.token_type_ids,
            self.labels,
        ) = self.get_input(df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.labels[idx],
        )

    def get_input(self, df):
        sentences = df["s1"].values
        labels = df["similarity"].values

        tokens_seq = list(map(self.bert_tokenizer.tokenize, sentences))
        result = list(map(self.trunate_and_pad, tokens_seq))

        input_ids = [i[0] for i in result]
        attention_mask = [i[1] for i in result]
        token_type_ids = [i[2] for i in result]

        return (
            torch.Tensor(input_ids).type(torch.long),
            torch.Tensor(attention_mask).type(torch.long),
            torch.Tensor(token_type_ids).type(torch.long),
            torch.Tensor(labels).type(torch.long),
        )

    def trunate_and_pad(self, tokens_seq):
        tokens_seq = ["[CLS]"] + tokens_seq
        if len(tokens_seq) > self.max_seq_len:
            tokens_seq = tokens_seq[0 : self.max_seq_len]
        padding = [0] * (self.max_seq_len - len(tokens_seq))
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens_seq)
        input_ids += padding
        attention_mask = [1] * len(tokens_seq) + padding
        token_type_ids = [0] * (self.max_seq_len)

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len

        return input_ids, attention_mask, token_type_ids
