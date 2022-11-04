import torch, numpy as np
from tqdm import tqdm
from .common import arr_size, lenA, batch

class Study:
    def __init__(self, model, read, diff, p):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
        self.model, self.p = model, p
        self.data, self.teach, self.div = read
        self.diff = np.array([len(self.teach)-diff, diff])/batch

    def train(self):
        l, self.p.test, d = [], False, int(self.diff[0])
        print('train')

        self.model.train()
        for i in tqdm(range(1, d+1)):
            train, teach = self.create_randrange()

            pred = self.model(train)
            loss = self.loss_fn(pred, teach)
            l.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i % (3000/batch) == 0 or i == 1) and self.p.execute:
                self.p.saveimg(self.model, teach, i)
        self.p.saveloss(l)

    def test(self):
        self.p.test, d = True, int(self.diff[1])
        self.test_loss, self.co = 0, 0
        psum, ans = [torch.zeros(lenA) for _ in range(2)]
        print('test')

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(1, d+1)):
                train, teach = self.create_randrange()

                pred = self.model(train)
                self.test_loss += self.loss_fn(pred, teach).item()

                for m, t in zip(pred.argmax(dim=1), teach):
                    self.co, psum[m], ans = self.co+t[m], psum[m]+1, ans+t

                if (i % (1000/batch) == 0 or i == 1) and self.p.execute:
                    self.p.saveimg(self.model, teach, i)

            self.test_loss, self.co = self.test_loss/d, self.co/d/batch
            print(f'Accuracy: {(100*self.co):>0.1f}%, Avg loss: {self.test_loss:>8f}')
            print(f'Sum: {list(map(int, psum))}, Ans: {list(map(int, ans))}')

    def create_randrange(self):
        r = np.random.randint(0, len(self.data), batch)
        idx = np.array(list(map(lambda e: np.argmin(np.abs(self.div-e)), r)))
        tE = np.array(list(map(lambda e,i:e-arr_size if 0<e-self.div[i]<arr_size else e,r,idx)))
        trainN = np.array(list(map(lambda e: np.arange(e, e+arr_size), tE)))
        teachN = np.array(list(map(lambda e,i:e-(i if e<self.div[i] else i+1)*arr_size,tE,idx)))
        return torch.Tensor(self.data[trainN]), torch.Tensor(self.teach[teachN])