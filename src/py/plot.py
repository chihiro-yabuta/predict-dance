import matplotlib.pyplot as plt

class plot:
    def __init__(self, execute):
        self.epoch = 0
        self.test = False
        self.execute = execute

    def saveloss(self, loss):
        fig = plt.figure(figsize=(20, 4), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(loss)
        fig.savefig(f'out/img/loss/epoch_{str(self.epoch)}.pdf')

    def saveimg(self, m, ans, idx):
        conv, em, enc = map(lambda l: l[0].detach().clone(), (m.c,m.em,m.e))
        self.fig = plt.figure(figsize=(16, 4), tight_layout=True)
        self.fig.suptitle(f'{ans[0]}')
        ax1 = self.fig.add_subplot(1, 3, 1)
        ax2 = self.fig.add_subplot(1, 3, 2)
        ax3 = self.fig.add_subplot(1, 3, 3)
        ax1.set_title('conv')
        ax2.set_title('pixel_embedding')
        ax3.set_title('encode')
        self.eachplot((conv, em, enc))
        s = 'test' if self.test else 'epoch_'+str(self.epoch)
        plt.close(self.fig)
        self.fig.savefig(f'out/img/{s}/estimate_{idx}.pdf')

    def eachplot(self, nns):
        for ax, nn in zip(self.fig.axes, nns):
            for e in nn.transpose(0,1):
                ax.plot(list(map(float, e)))