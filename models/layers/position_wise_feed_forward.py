from torch import nn

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
        class for Position-wise Feedforward Neural Network
        Args:
            d_model: Dimension of model
            hidden: Hidden layer size
            drop_prob: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(hidden, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x