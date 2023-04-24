with open('input.txt', 'r' , encoding = 'utf-8') as f:
    text = f.read()

print("Length of dataset in characters: ", len(text))
# print(text[:100])
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print("Vocabulary size: ", vocab_size)

print(list(enumerate(chars)))

chin = {ch:i for i, ch in enumerate(chars)}
inch = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [chin[c] for c in s]
decode = lambda l: ''.join([inch[c] for c in l])

print(encode('hello'))
print(decode(encode('hello')))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

n = int(len(data) * 0.9) 
train_data = data[:n] # 90% of data for training
val_data = data[n:] # 10% of data for validation

block_size = 8
print(train_data[:block_size+1])


x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context} then target is {target}")


torch.manual_seed(1337)
batch_size = 4 # number of independent sequences to be processed in parallel
block_size = 8 # maximum content length for predictions

