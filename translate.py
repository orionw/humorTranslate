import math
import copy
import time
import os
import typing
import random
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data, datasets
import torchtext
from tqdm import tqdm

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import spacy
import gc


from transformer import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load spacy tokenizers.
spacy_en = spacy.load('en')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                 eos_token = EOS_WORD, pad_token=BLANK_WORD)

print("Loading Dataset")
full = pd.read_csv(os.path.join("data", "full_unique.csv"))
english_lines = list(full["edited_version"])
spanish_lines = list(full["original_clean"])
print("### There are {} lines of data ####".format(len(english_lines)))

fields = (["src", SRC], ["trg", TGT])
examples = [torchtext.data.Example.fromlist((spanish_lines[i], english_lines[i]), fields ) for i in range(len(english_lines))]
MAX_LEN = 200
train, val = torchtext.data.Dataset(examples, fields=fields, filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
        len(vars(x)['trg']) <= MAX_LEN).split()

list_of_val = []
for example in val.examples:
    new_dict = {"original": " ".join(example.src), "edited": " ".join(example.trg)}
    list_of_val.append(new_dict)

MIN_FREQ = 1
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
gc.collect()

val_df = pd.DataFrame(list_of_val)
val_df.to_csv("val_data.csv")

pad_idx = TGT.vocab.stoi["<blank>"]

model = TransformerModel(len(SRC.vocab), len(TGT.vocab), N=2).cuda()
device = torch.device('cuda')

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def eval_text(valid_iter, model, n_samples: int = 2):
    samples = []
    for i, batch in enumerate(valid_iter):
        new_sample = {}
        if i >= n_samples:
            return samples
        src = batch.src.transpose(0, 1)[:1].cuda()
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2).cuda()
        out = greedy_decode(model, src, src_mask, 
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        instance = ""
        for i in range(0, src.size(1)):
            sym = SRC.vocab.itos[src[0, i]]
            if sym == "</s>": break
            instance += sym + " "
        new_sample["original"] = instance
        instance = ""
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            instance += sym + " "
        new_sample["translated"] = instance
        instance = ""
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            instance += sym + " "
        new_sample["ground_truth"] = instance
        samples.append(new_sample)
    return samples


def eval_all_text(valid_iter, model, n_samples: int = 2):
    samples = []
    count = 0
    for i, batch in enumerate(valid_iter):
        print("On batch ", i)
        if i >= n_samples:
            return samples
        print("Eval batch with size", batch.src.shape)
        for sentence_idx in range(batch.src.transpose(0, 1).shape[0]):
            new_sample = {}
            count += 1
            src = batch.src.transpose(0, 1)[sentence_idx:sentence_idx+1].cuda()
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2).cuda()
            out = greedy_decode(model, src, src_mask, 
                                max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
            instance = ""
            for i in range(0, src.size(1)):
                sym = SRC.vocab.itos[src[0, i]]
                if sym == "</s>": break
                instance += sym + " "
            new_sample["original"] = instance
            instance = ""
            for i in range(1, out.size(1)):
                sym = TGT.vocab.itos[out[0, i]]
                if sym == "</s>": break
                instance += sym + " "
            new_sample["translated"] = instance
            instance = ""
            for i in range(1, batch.trg.size(0)):
                sym = TGT.vocab.itos[batch.trg.data[i, sentence_idx]]
                if sym == "</s>": break
                instance += sym + " "
            new_sample["edited"] = instance
            samples.append(new_sample)
    print("There were new sents", count)
    assert count == len(samples), "did not match up"
    return samples


def add_examples(text: typing.List[str], MAX_LEN=200, BATCH_SIZE=1000):
    examples = [torchtext.data.Example.fromlist((text[i], ""), fields ) for i in range(len(text))]
    data = torchtext.data.Dataset(examples, fields=fields, filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
        len(vars(x)['trg']) <= MAX_LEN)
    new_iter = DataIterator(data, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    return new_iter
    

def scope(args: argparse.Namespace):
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 1000
    train_iter = DataIterator(train, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = DataIterator(val, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

    len_iter_train = len(train) // BATCH_SIZE

    model_opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    loop = tqdm(total=args.n_epochs * len_iter_train, position=0, leave=True)
    loss_list = []
    for epoch in range(args.n_epochs):
        model.train()
        loss, loop = run_epoch((rebatch(pad_idx, b) for b in train_iter), 
                  model, 
                  LossFunction(model.generator, criterion, model_opt), loop, epoch)
        loss_list.append(loss)
        model.eval()
        samples = eval_text(valid_iter, model)
        for sample in samples:
            print(sample)
        
        # outside sources
        text = []
        if args.interactive:
            print("Input the number of examples first, then each example")
            new_examples = int(input())
            for i in range(new_examples):
                text.append(str(input()))
            new_data_iter = add_examples(text)
            samples = eval_text(new_data_iter, model)
            for sample in samples:
                print(sample)
        else:
            new_examples = 1 #int(input())
            for i in range(new_examples):
                # text.append(str(input()))
                text.append(str("Trump tweeted as she was testifying: Was it witness tampering?"))
            new_data_iter = add_examples(text)
            samples = eval_text(new_data_iter, model)
            for sample in samples:
                print(sample)


        if epoch and epoch % 10 == 0:
            if not os.path.isdir("models"):
                os.makedirs("models")
            torch.save(model.state_dict(), os.path.join("models", "{}-{}-model.pt".format(epoch, int(time.time()))))
    plt.plot(list(range(len(loss_list))), loss_list)
    plt.savefig("loss_plot.png")


def evaluate(args: argparse.Namespace):
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 1000
    valid_iter = DataIterator(val, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

    model_opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    samples = eval_all_text(valid_iter, model, n_samples=len(val))
    if args.verbose:
        for sample in samples:
            print(sample)
    
    if args.save_output:
        checkpoint_name = args.checkpoint.split("/")[-1][:-2]
        pd.DataFrame(samples).to_csv(f"eval_data_only_{checkpoint_name}.csv")

    # outside sources
    text = []
    if args.interactive:
        print("Input the number of examples first, then each example")
        new_examples = int(input())
        for i in range(new_examples):
            text.append(str(input()))
        new_data_iter = add_examples(text)
        samples = eval_text(new_data_iter, model)
        for sample in samples:
            print(sample)
    elif args.verbose:
        new_examples = 1 
        for i in range(new_examples):
            text.append(str("Trump tweeted as she was testifying: Was it witness tampering?"))
        new_data_iter = add_examples(text)
        samples = eval_text(new_data_iter, model)
        for sample in samples:
            print(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="whether to print or not", default=False)
    parser.add_argument("-s", "--save_output", action="store_true", help="save output of evaluation", default=False)
    parser.add_argument("-i", "--interactive", action="store_true", help="interact with the model", default=False)
    parser.add_argument("-e", "--evaluate", action="store_true", help="interact with the model", default=False)
    parser.add_argument("-all", "--eval_all", action="store_true", help="evall all checkpoints", default=False)
    parser.add_argument("-n", "--n_epochs", type=int, help="number of epochs to run", default=500)
    parser.add_argument("-c", "--checkpoint", type=str, help="the location of the checkpoint to run", default="models/90-1574381105-model.pt") 
    parser.add_argument("-d", "--data_folder", type=str, help="the location of where the data is", default="data/full_unique.csv")  # TODO implement this
    args = parser.parse_args()
    if not args.evaluate:
        scope(args)
    elif args.evaluate and not args.eval_all:
        print("Evaluating one")
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        evaluate(args)
    elif args.eval_all:
        print("Evaluating all")
        for checkpoint_path in glob.glob("models/*.pt"):
            args.checkpoint = checkpoint_path
            model.load_state_dict(torch.load(args.checkpoint))
            model.eval()
            evaluate(args)
    else:
        raise NotImplementedError()

