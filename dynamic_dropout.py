import torch.nn as nn

class DynamicDropout(nn.Module):
    def __init__(self, initial_p=0, adjustment_rate=0.02, max_p=0.1):
        super(DynamicDropout, self).__init__()
        self.dropout = nn.Dropout(p=initial_p)
        self.adjustment_rate = adjustment_rate
        self.max_p = max_p

    def forward(self, x):
        return self.dropout(x)

    def increase_dropout(self):
        check = self.dropout.p + self.adjustment_rate
        if check > self.max_p:
            self.dropout.p = self.max_p
        else:
            self.dropout.p = check

        print(f"Dropout increased to {self.dropout.p}")

    def decrease_dropout(self):
        check = self.dropout.p - (self.adjustment_rate*2)
        if check < 0:
            self.dropout.p = 0
        else:
            self.dropout.p = check

        print(f"Dropout decreased to {self.dropout.p}")

    def reset_dropout(self):
        self.dropout.p = 0
        # print(f"Dropout reset to {self.dropout.p}")
