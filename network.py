import torch
import torch.nn.functional as F
import hyperparams as hp
import random
import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        super(Network, self).__init__()
        self.X_training, self.Y_training = None, None
        self.X_dev, self.Y_dev = None, None
        self.X_test, self.Y_test = None, None
        self.words, self.char_size = None, None
        self.int_to_string, self.string_to_int = None, None
        self.C, self.W1, self.b1, self.W2, self.b2 = None, None, None, None, None
        self.__initialize_params()

    def __initialize_params(self):
        self.C = torch.randn((hp.CHAR_SIZE, hp.EMBEDDING_DIM))
        self.W1 = torch.randn((hp.BLOCK_SIZE * hp.EMBEDDING_DIM, hp.HIDDEN_LAYER_SIZE))
        self.b1 = torch.randn(hp.HIDDEN_LAYER_SIZE)
        self.W2 = torch.randn((hp.HIDDEN_LAYER_SIZE, hp.CHAR_SIZE))
        self.b2 = torch.randn(hp.CHAR_SIZE)
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True
        self.step_idx = []
        self.epoch_counter = 0
        self.loss_idx = []

    def load_data(self, filename='polish_names.txt'):
        self.words = [w.lower() for w in open(filename, 'r', encoding="utf8").read().splitlines()]
        chars = sorted(list(set(''.join(self.words))))
        self.string_to_int = {s: i + 1 for i, s in enumerate(chars)}
        self.string_to_int['.'] = 0
        self.int_to_string = {i: s for s, i in self.string_to_int.items()}
        self.char_size = len(self.string_to_int)
        random.shuffle(self.words)
        n1 = int(0.8 * len(self.words))
        n2 = int(0.9 * len(self.words))

        self.X_training, self.Y_training = self.__build_dataset(self.words[:n1])
        self.X_dev, self.Y_dev = self.__build_dataset(self.words[n1:n2])
        self.X_test, self.Y_test = self.__build_dataset(self.words[n2:])

    def __build_dataset(self, words):
        X, Y = [], []
        for w in words:
            context = [0] * hp.BLOCK_SIZE
            for ch in w + '.':
                ix = self.string_to_int[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def train(self, n_epochs=1000, lr=0.1):
        for epoch in range(n_epochs):
            batch_indexes = torch.randint(0, self.X_training.shape[0], (32,))
            embeddings = self.C[self.X_training[batch_indexes]]
            hidden = torch.tanh(embeddings.view(-1, hp.BLOCK_SIZE * hp.EMBEDDING_DIM) @ self.W1 + self.b1)
            logits = hidden @ self.W2 + self.b2
            loss = F.cross_entropy(logits, self.Y_training[batch_indexes])
            print(loss.item())
            for p in self.parameters:
                p.grad = None
            loss.backward()
            for p in self.parameters:
                p.data -= lr * p.grad
            self.step_idx.append(self.epoch_counter)
            self.epoch_counter += 1
            self.loss_idx.append(loss.item())
        plt.plot(self.step_idx, self.loss_idx)
        plt.show()

    def sample(self, n=10):
        out = []
        context = [0] * hp.BLOCK_SIZE
        res = ""
        for _ in range(n):
            embeddings = self.C[torch.tensor(context).view(1, -1)]
            hidden = torch.tanh(embeddings.view(-1, hp.BLOCK_SIZE * hp.EMBEDDING_DIM) @ self.W1 + self.b1)
            logits = hidden @ self.W2 + self.b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            res += self.int_to_string[ix]
            context = context[1:] + [ix]
            out.append(ix)
        return res