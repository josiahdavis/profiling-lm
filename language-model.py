# coding: utf-8
import argparse
import time
import math
import os
from io import open

# import data
# import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.profiler as profiler
import torch.optim as optim
from apex import pyprof

# pyprof.nvtx.init()


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, seq_len, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.position_embeddings = nn.Embedding(seq_len, ninp)
        self.word_embeddings = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()
        position = torch.arange(0, seq_len).unsqueeze(1)
        self.register_buffer("position", position)
        self.dropout = nn.Dropout(p=dropout)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.position_embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        word_embeddings = self.word_embeddings(src) * math.sqrt(self.ninp)
        position = self.position[: src.size(0), :]
        position_embeddings = self.position_embeddings(position)
        src = self.dropout(word_embeddings + position_embeddings)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, args):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def evaluate(model, data_source, corpus, criterion, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


parser = argparse.ArgumentParser(
    description="PyTorch Wikitext-2 Transformer Language Model"
)
parser.add_argument(
    "--data", type=str, default="./data/wikitext-2", help="location of the data corpus"
)
parser.add_argument(
    "--emsize", type=int, default=200, help="size of word embeddings (also, d_model)",
)
parser.add_argument(
    "--nhid",
    type=int,
    default=200,
    help="number of hidden units per layer (e.g., dim_feedforward)",
)
parser.add_argument("--nlayers", type=int, default=1, help="number of layers")
parser.add_argument("--lr", type=float, default=20, help="initial learning rate")
parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
parser.add_argument("--epochs", type=int, default=1, help="upper epoch limit")
parser.add_argument(
    "--batch_size", type=int, default=20, metavar="N", help="batch size"
)
parser.add_argument("--bptt", type=int, default=35, help="sequence length")
parser.add_argument(
    "--dropout",
    type=float,
    default=0.2,
    help="dropout applied to layers (0 = no dropout)",
)
parser.add_argument(
    "--tied", action="store_true", help="tie the word embedding and softmax weights"
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--log-interval", type=int, default=200, metavar="N", help="report interval"
)
parser.add_argument(
    "--save", type=str, default="model.pt", help="path to save the final model"
)


parser.add_argument(
    "--nhead",
    type=int,
    default=1,
    help="the number of heads in the encoder of the transformer model",
)


def main():

    args = parser.parse_args()
    pyprof.nvtx.init()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device {device}")

    corpus = Corpus(args.data)

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)
    print(f"train_data.shape={train_data.shape}")
    print(f"val_data.shape={val_data.shape}")
    print(f"test_data.shape={test_data.shape}")

    ntokens = len(corpus.dictionary)
    print(f"ntokens={ntokens}")
    # model = model.TransformerModel(
    model = TransformerModel(
        ntokens,
        args.emsize,
        args.nhead,
        args.nhid,
        args.nlayers,
        args.bptt,
        args.dropout,
    ).cuda()
    # ).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    print(criterion)
    print(f"Using tokens={ntokens}, emsize={args.emsize}, nhid={args.emsize}")
    print(
        f"""ntokens={ntokens}, emsize={args.emsize}, 
    nhead={args.nhead}, nhid={args.nhid}, nlayers={args.nlayers}, 
    bpttt={args.bptt}, dropout={args.dropout}
    """
    )

    iter_to_capture = 1

    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.

    with torch.autograd.profiler.emit_nvtx():
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            model.train()
            total_loss = 0.0
            start_time = time.time()
            ntokens = len(corpus.dictionary)
            for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
                data, targets = get_batch(train_data, i, args)
                # TODO: Use language modelling abstraction with torchtext
                model.zero_grad()
                if (epoch == 1) and (batch == iter_to_capture):
                    profiler.start()
                output = model(data)
                loss = criterion(output.view(-1, ntokens), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                for p in model.parameters():
                    p.data.add_(-lr, p.grad.data)
                # TODO: Use an optimizer
                if (epoch == 1) and (batch == iter_to_capture):
                    profiler.stop()
                total_loss += loss.item()
                if batch % args.log_interval == 0 and batch > 0:
                    cur_loss = total_loss / args.log_interval
                    elapsed = time.time() - start_time
                    print(
                        "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                        "loss {:5.2f} | ppl {:8.2f}".format(
                            epoch,
                            batch,
                            len(train_data) // args.bptt,
                            lr,
                            elapsed * 1000 / args.log_interval,
                            cur_loss,
                            math.exp(cur_loss),
                        )
                    )
                    total_loss = 0
                    start_time = time.time()
            val_loss = evaluate(model, val_data, corpus, criterion, args)
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss,
                    math.exp(val_loss),
                )
            )
            print("-" * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0

    # Run on test data.
    test_loss = evaluate(model, test_data, corpus, criterion, args)
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
    )
    print("=" * 89)


if __name__ == "__main__":
    main()
