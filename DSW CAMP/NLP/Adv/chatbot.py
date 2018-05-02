# ChatBot with Deep NLP

import numpy as np
import pandas as pd
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim


################################ SET HYPER PARAMS

class HParams():
    def __init__(self):
        self.n_layers = 2
        self.hidden_size = 512
        self.fc_size = 512
        self.dropout = 0.9
        self.batch_size = 20
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 5.
        self.cuda = True
        self.num_epoch = 200
        self.max_length = 10

hp = HParams()

################################# READ DATA

data = pd.DataFrame.from_csv('dataset/indo.tsv', sep='\t')

# Reading and decoding files

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    #s = unicode_to_ascii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

data['input'] = data['input'].apply(normalize_string)
data['target'] = data['target'].apply(normalize_string)

NUM_LINES = len(data)
print(NUM_LINES)

# Indexing words

SOS_token = 0
EOS_token = 1

class Voc:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


VOC = Voc()
pairs = []
for input,target in zip(data['input'],data['target']):
    VOC.index_words(input)
    VOC.index_words(target)
    if len(input.split(' '))<hp.max_length and len(target.split(' '))<hp.max_length:
        pairs.append([input,target])

print(len(pairs))
print(random.choice(pairs))
print(VOC.n_words)

######################################## PYTORCH UTILS.
print("data ready, now let's go into pytorch")

def lr_decay(optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr']>hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer

def indexes_from_sentence(sentence):
    sentence_split = sentence.split(' ')
    length = len(sentence_split)
    vec = [EOS_token]*hp.max_length
    vec[:length] = [VOC.word2index[word] for word in sentence_split]
    return vec, length

def make_batch(index=None):
    if index is None:
        batch_idx = np.random.choice(len(pairs),hp.batch_size)
    else:
        batch_idx = [idx%len(pairs) for idx in range(index, index+hp.batch_size)]
    batch_pairs = [pairs[idx] for idx in batch_idx]
    inputs = []
    outputs = []
    lengths_inputs = []
    lengths_outputs = []
    for pair in batch_pairs:
        input, length_input = indexes_from_sentence(pair[0])
        output, length_output = indexes_from_sentence(pair[1])
        inputs.append(input)
        outputs.append(output)
        lengths_inputs.append(length_input)
        lengths_outputs.append(length_output)

    inputs = [torch.Tensor(input) for input in inputs]
    outputs = [torch.Tensor(output) for output in outputs]

    input_batch_lengths = dict(zip(inputs,lengths_inputs))
    output_batch_lengths = dict(zip(outputs,lengths_outputs))

    sorted_inputs = sorted(input_batch_lengths, key=lambda x: x[1], reverse=True)
    sorted_outputs = sorted(output_batch_lengths, key=lambda x: x[1], reverse=True)

    if hp.cuda:
        batch_inputs = Variable(torch.stack(sorted_inputs,1).cuda())
        batch_outputs = Variable(torch.stack(sorted_outputs,1).cuda())
    else:
        batch_inputs = Variable(torch.stack(sorted_inputs,1))
        batch_outputs = Variable(torch.stack(sorted_outputs,1))

    lengths_inputs.sort(reverse=True)
    lengths_outputs.sort(reverse=True)

    # batch_input =  length*batch
    return batch_inputs, batch_outputs, lengths_inputs, lengths_outputs

def variable_from_sentence(sentence):
    vec, length = indexes_from_sentence(sentence)
    inputs = [vec]
    lengths_inputs = [length]
    if hp.cuda:
        batch_inputs = Variable(torch.stack(torch.Tensor(inputs),1).cuda())
    else:
        batch_inputs = Variable(torch.stack(torch.Tensor(inputs),1))
    return batch_inputs, lengths_inputs

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

########################################### CREATE MODELS

class EncoderRNN(nn.Module):
    def __init__(self, voc_size, hidden_size, n_layers=2):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=hp.dropout)

    def forward(self, inputs, lengths, hidden_cell=None, batch_size=hp.batch_size):
        if hidden_cell is None:
            if hp.cuda:
                hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
                cell = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
            else:
                hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
                cell = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
            hidden_cell = (hidden, cell)
        embedded = self.embedding(inputs.long()) # (hp.max_length*batch*emb)
        input_pack = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        output_pack, hidden_cell = self.lstm(input_pack, hidden_cell)
        #output = nn.utils.rnn.pad_packed_sequence(output_pack)
        return output_pack, hidden_cell

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, fc_size, voc_size, n_layers=2):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=hp.dropout)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, voc_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden_cell):
        # input = 1(length)*batch
        embedded = self.embedding(input)
        output, hidden_cell = self.lstm(embedded, hidden_cell)
        # output = 1(length)*batch*emb
        output = self.fc1(output.squeeze())
        output = self.fc2(output)
        output = self.softmax(output)
        return output, hidden_cell

######################################## ONE TRAINING ITERATION

def train(batch_inputs, batch_targets, lengths_inputs, lengths_targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder.train()
    decoder.train()

    loss = 0

    # what we call context:
    _, encoder_hidden_cell = encoder(batch_inputs, lengths_inputs)
    decoder_hidden_cell = encoder_hidden_cell

    ammorce = Variable(torch.LongTensor([[SOS_token]*hp.batch_size]))
    ammorce = ammorce.cuda() if hp.cuda else ammorce

    if np.random.rand()>0.5: # teacher forcing
        decoder_input = torch.cat([ammorce,batch_targets.long()[:-1,:]]) #
        decoder_output, decoder_hidden_cell = decoder(
                decoder_input, decoder_hidden_cell)
        loss += criterion(decoder_output.view(-1,VOC.n_words), batch_targets.long().view(-1))

    else: # no teacher forcing
        decoder_input = ammorce
        for di in range(hp.max_length):
            decoder_output, decoder_hidden_cell = decoder(
                decoder_input, decoder_hidden_cell)
            # decoder_output contains the probabilities for each word IDs.
            # The index of the probability is the word ID.
            _, topi = torch.topk(decoder_output,1)
            # topi = (batch*1)
            decoder_input = torch.transpose(topi,0,1)
            # target is (batch*length)
            loss += criterion(decoder_output, batch_targets[di,:].long())

    loss.backward()
    nn.utils.clip_grad_norm(encoder.parameters(), hp.grad_clip)
    nn.utils.clip_grad_norm(decoder.parameters(), hp.grad_clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / float(hp.max_length)

###################################### EVALUATE ITERATION

def evaluate(encoder, decoder, sentence):

    encoder.train(False)
    decoder.train(False)

    batch_inputs, lengths_inputs = variable_from_sentence(sentence)
    _, encoder_hidden_cell = encoder(batch_inputs, lengths_inputs, batch_size=1)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if hp.cuda else decoder_input
    decoder_hidden_cell = encoder_hidden_cell

    decoded_words = []

    for di in range(hp.max_length):
        decoder_output, decoder_hidden_cell = decoder(
            decoder_input, decoder_hidden_cell)
        # decoder_output contains the probabilities for each word IDs.
        # The index of the probability is the word ID.
        _, topi = torch.topk(decoder_output,1)
        # topi = (batch*1) = 1*1
        decoder_input = topi.unsqueeze(0)
        ni = topi.data[0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(VOC.index2word[ni])

    return decoded_words

def evaluate_test(encoder, decoder):
    s1 = random.choice(pairs)[0]
    s2 = "halo ."
    s3 = "apa kabarmu ?"
    sentences = [s1,s2,s3]
    f = open('results', 'w+')
    for s in sentences:
        f.write('>'+s+'\n')
        output_words = evaluate(encoder, decoder, s)
        output_sentence = ' '.join(output_words)
        f.write('<'+output_sentence+'\n')
        f.write(''+'\n')
    f.close()

###################################### RUN ITERATIONS

def trainIters(encoder, decoder, print_every=1000, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=hp.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=hp.lr)

    criterion = nn.NLLLoss()
    iter_per_epoch = len(pairs)/hp.batch_size + 1
    n_iters = len(pairs)*hp.num_epoch
    sel = np.random.rand()

    for epoch in range(hp.num_epoch):
        print("EPOCH # ",epoch)
        for iter in range(1, int(iter_per_epoch)+1):
            batch_inputs,batch_targets,lengths_inputs,lengths_targets = make_batch((iter-1)*hp.batch_size)

            loss = train(batch_inputs, batch_targets, lengths_inputs, lengths_targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            encoder_optimizer = lr_decay(encoder_optimizer)
            decoder_optimizer = lr_decay(decoder_optimizer)

            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                total_iterations = hp.batch_size*(epoch*iter_per_epoch+iter)
                print('%s (%d iters) %.4f' % (timeSince(start, total_iterations/float(n_iters)), total_iterations, print_loss_avg))

            if iter % 1000 == 0:
                evaluate_test(encoder,decoder)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # save epoch:
        torch.save(encoder.state_dict(),'model/encoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))
        torch.save(decoder.state_dict(),'model/decoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))

######################################## MAIN SCRIPT

print("constructing models...")
encoder1 = EncoderRNN(VOC.n_words, hp.hidden_size, n_layers=hp.n_layers)
decoder1 = DecoderRNN(hp.hidden_size, hp.fc_size, VOC.n_words,
                               n_layers=hp.n_layers)

print("moving to GPU...")
if hp.cuda:
    encoder1 = encoder1.cuda()
    decoder1 = decoder1.cuda()

print("starting iterations...")
trainIters(encoder1, decoder1, print_every=10)
