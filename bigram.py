import torch
import torch.nn as nn
from torch.nn import functional as F

# parameters  
batch_size = 32         # number of independent sequences to be processed in parallel
block_size = 8          # maximum context length for predictions
max_iters = 3000        # maximum number of training iterations
eval_interval = 100     # interval for evaluation of the model
learning_rate = 1e-2    # learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu' # device for the tensors
eval_iters = 200        # number of iterations for evaluation
#------------------

torch.manual_seed(1337) # for reproducibility

with open('input.txt', 'r' , encoding = 'utf-8') as f:
    corpus = f.read()

# Create a set of unique characters in the corpus and sort them
    # set() is an unordered collection of unique items.
    # list() is a built-in function that creates a list from an iterable object.
    # sorted() takes a list or set and returns a new sorted list.
chars = sorted(list(set(corpus)))
vocab_size = len(chars)
# Create a dictionary that maps each character to an integer
ctoi = { c:i for i, c in enumerate(chars) }          # creates dictionary char to integer
itoc = { i:c for i, c in enumerate(chars) }          # creats dictionary integer to char
encode = lambda s: [ctoi[c] for c in s]              # creates a function that converts a string to a list of integers
decode = lambda l: ''.join([itoc[i] for i in l])     # creates a function that converts a list of integers to a string

# Train and test data
data = torch.tensor(encode(corpus), dtype=torch.long) # converts the corpus to a tensor of integers
n = int(len(data) * 0.9)                              # 90% of data for training
train_data = data[:n]                                 # 90% of data for training
val_data   = data[n:]                                 # 10% of data for validation

# Data loader
def get_batch(split):                                         # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data       # Choose the data to be used
    ix = torch.randint(len(data) - block_size, (batch_size,)) # The first arg -> upper bound, the second arg -> the shape of the output tensor (it must be a tuple)
    x = torch.stack([data[i:i+block_size] for i in ix])       # stack() concatenates a sequence of tensors along a new dimension
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x.to(device), y.to(device)                         # returns the input and target tensors (if cuda is available, it will be used)

#------------------
# Something will be here
#------------------

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # creates a table of embeddings (creates the weights matrix)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C) # (B*T, C)
            targets = targets.view(B*T)   # (B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate (self, idx, max_new_tokens) :
        # idx is (B, T) array of indices in the current context
        for in range (max_new_tokens) :
        # get the predictions
        logits, loss = self(idx)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
probs = F.softmax(logits, dim=-1) # (B, C)
# sample from the distribution
id_next = torch. multinomial (probs, num_samples=1) # (B, 1)
# append sampled index to the running sequence
idx = torch. cat((idx, idx_next), dim=1) # (B, T+1)
    
xb , yb = get_batch('train')
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)






  









    

