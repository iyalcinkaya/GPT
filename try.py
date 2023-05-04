import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

with open('input.txt', 'r' , encoding = 'utf-8') as f:
    corpus = f.read()

chars = sorted(list(set(corpus)))
vocab_size = len(chars)

print('Vocabulary: ', chars)
print('Vocabulary size: ', vocab_size)
print('-------------------')

ch2in = {ch:i for i, ch in enumerate(chars)} # convert char to index
in2ch = {i:ch for i, ch in enumerate(chars)} # convert index to char

encode = lambda s: [ch2in[c] for c in s] 
decode = lambda l: ''.join([in2ch[c] for c in l])

data = torch.tensor(encode(corpus), dtype=torch.long)

n = int(len(data) * 0.9) 
train_data = data[:n] # 90% of data for training
val_data = data[n:] # 10% of data for validation

batch_size = 4 # number of independent sequences to be processed in parallel
block_size = 8 # maximum content length for predictions
print(train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb , yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-------------------')


# -----------------------------------------------
# print("Length of dataset in characters: ", len(corpus))
# print(text[:100])
# print(list(enumerate(chars)))
# print(''.join(chars))
# print(encode('hello'))
# print(decode(encode('hello')))
# print(data.shape, data.dtype)
# print(data[:100])
# create 5x5 tensor
# x = torch.arange(25).reshape(5,5)
# create a 5x5 tensor 2 dimensional
# y = torch.arange(25).view(5,5)
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"When input is {context} then target is {target}")