import torch

#Set seed for reproducibility
torch.manual_seed(2022)

class Map(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.W = torch.nn.Linear(1, 1)
        self.act = torch.nn.ReLU()

    def forward(self, x):

        return self.act(self.W(x))

    def ppWeights(self):

        print(f"W: \n\t{self.W.weight}")
        print(f"b: \n\t{self.W.bias}")

    def ppGrads(self):

        print(f"W grad: \n\t{self.W.weight.grad}")
        print(f"b grad: \n\t{self.W.bias.grad}")


if __name__ == "__main__":

    x = torch.tensor([2.])
    y = torch.tensor([5.])

    criterion = torch.nn.MSELoss()

    model = Map()

    print('Init stuff')
    model.ppWeights()
    model.ppGrads()

    print('---------------------------')
    model.zero_grad()

    out = model(x)
    print(f"\tPrediction {out}")

    loss = criterion(out, y)

    print(f"\tNew loss: {loss.item()}")

    loss.backward()
    model.ppGrads()

    lr = 0.1
    for f in model.parameters():
        f.data.sub_(f.grad.data * lr)

    model.ppWeights()
    print('---------------------------')

    print('Final model')
    model.ppWeights()
    print()
    print('Final prediction {model(x)}')
