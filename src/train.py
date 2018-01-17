import os
import math
import argparse
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset

from translate import model_translate
import time

def parse_arguments():
    
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=80,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=5.0,
                   help='initial learning rate')
    p.add_argument('-teacher_forcing_ratio', type=float, default=5.0,
                   help='teacher forcing ratio')   
    p.add_argument('-external_valid_script', type=str, default='./validate_by_bleu.sh')    

    return p.parse_args()


def evaluate(model, val_iter, vocab_size, Lang1, Lang2):
    model.eval()
    pad = Lang2.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src = Variable(src.data.cuda(), volatile=True)
        trg = Variable(trg.data.cuda(), volatile=True)
        output = model(src, trg)
        loss = F.cross_entropy(output.view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        total_loss += loss.data[0]
    return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, Lang1, Lang2):
    model.train()
    total_loss = 0
    pad = Lang2.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        # print(output.size())
        # print(trg.size())
        loss = F.cross_entropy(output.view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data[0]

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0

def main():
    args = parse_arguments()
    hidden_size = 256
    embed_size = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, Lang1, Lang2 = load_dataset(args.batch_size)
    de_size, en_size = len(Lang1.vocab), len(Lang2.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[Lang1_vocab]:%d [Lang2_vocab]:%d" % (de_size, en_size))

    print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=1, dropout=0.2)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.2)
    
    ## seq2seq = Seq2Seq(encoder, decoder).cuda()
     
    seq2seq = Seq2Seq(encoder, decoder)
    if torch.cuda.device_count() > 1:
        print("Total", torch.cuda.device_count(), "GPUs!")
        seq2seq = nn.DataParallel(seq2seq)
    seq2seq.cuda()
    
    optimizer = optim.Adam(seq2seq.parameters(), lr=0.001)
    # optimizer = optim.Adadelta(seq2seq.parameters(), rho=0.95)
    ## optimizer = optim.SGD(seq2seq.parameters(), lr=1.0) 
    ## scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # seq2seq.load_state_dict(torch.load("save/seq2seq_8.pt"))
    print(seq2seq)

    # model_translate(seq2seq, "save/seq2seq_6.pt", "data/valid.ch.1664", "eval/mt06.out", Lang1, Lang2, args.external_valid_script, beam_size=12, max_len=120)
    # exit(-1)    

    best_val_loss = None
    ## scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    for e in range(1, args.epochs+1):
        # scheduler.step()
        start = time.time()
        train(e, seq2seq, optimizer, train_iter,
              en_size, args.grad_clip, Lang1, Lang2)
        val_loss = evaluate(seq2seq, val_iter, en_size, Lang1, Lang2)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))
        end = time.time()
        print("[Total Time] %.2fs" % (end - start))
        # Save the model if the validation loss is the best we've seen so far.
        # if not best_val_loss or val_loss < best_val_loss:
        print("[!] saving model...")
        if not os.path.isdir("save"):
            os.makedirs("save")
        torch.save(seq2seq.state_dict(), './save/seq2seq_%d.pt' % (e))
        # best_val_loss = val_loss
        model_path = "save/seq2seq_"+str(e)+".pt"
        model_translate(seq2seq, model_path, "data/valid.ch.1664", "eval/mt06.out", Lang1, Lang2, args.external_valid_script, beam_size=12, max_len=120)
    test_loss = evaluate(seq2seq, test_iter, en_size, Lang1, Lang2)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
