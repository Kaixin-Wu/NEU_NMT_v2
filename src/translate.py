##  Author          :   Wu Kaixin 
##  Date            :   2018/1/3
##  Email           :   wukaixin_neu@163.com
##  Last Modified in:   NEU NLP Lab., shenyang

import subprocess
import torch
from torch.autograd import Variable

def model_translate(model,
             model_path, 
	     input_file, 
             output_file, 
             Lang1,
             Lang2,
             external_valid_script,
             beam_size=12,
             max_len=120):

    infile = open(input_file)
    outfile = open(output_file, "w")

    model.eval()
    model.load_state_dict(torch.load(model_path))

    src_SOS_id = Lang1.vocab.stoi['<sos>']
    src_EOS_id = Lang1.vocab.stoi['<eos>']
    trg_SOS_id = Lang2.vocab.stoi['<sos>']
    trg_EOS_id = Lang2.vocab.stoi['<eos>']

    for line in infile:
        wordList = line.strip().split()
        word_ids = [Lang1.vocab.stoi[word] for word in wordList]
        ids = [[src_SOS_id] + word_ids + [src_EOS_id]]

        src = torch.LongTensor(ids)
        src = src.transpose(0, 1)
        src = Variable(src.cuda(), volatile=True)

        trg = torch.LongTensor([[trg_SOS_id]*max_len])
        trg = trg.transpose(0, 1)
        trg = Variable(trg.cuda(), volatile=True)

        xx, yy, zz = model.translate(src, trg, beam_size, Lang2)
       
        out = []
        for index in reversed(zz):
            if index == int(trg_EOS_id):
                break

            out.append(Lang2.vocab.itos[index])

        outfile.write(' '.join(out)+'\n')

        ## out = [Lang2.vocab.itos[int(index)] for index in zz]
        ## outfile.write(' '.join(reversed(out))+'\n')
    
    infile.close()
    outfile.close()

    status, output = subprocess.getstatusoutput(external_valid_script + " " + output_file)
    lines = output.strip().split("\n")

    BLEU_score = 0
    for line in lines:
        if line.find("BLEU score") != -1:
            tokenList = line.strip().split()
            for i in range(len(tokenList)):
                if (tokenList[i] == "BLEU") and (tokenList[i+1] == "score"):
                    BLEU_score = float(tokenList[i+3])
                    break

    print("[BLEU: %f]" % BLEU_score)
