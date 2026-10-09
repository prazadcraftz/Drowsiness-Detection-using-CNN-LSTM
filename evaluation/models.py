"""Fixed Phase 13 registry; importing defines models and never trains them."""
import torch
from torch import nn
from torch.nn import functional as F

from model import CNN_LSTM
from .train import MODEL_CONFIG

NAMES = ('cnn_only', 'lstm_only', 'cnn_lstm', 'cnn_gru', 'cnn_bilstm')


def initialize(module):
    """Extend the existing model's explicit initialization policy to GRU."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.LSTM, nn.GRU)):
            for name, parameter in layer.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_uniform_(parameter, nonlinearity='linear')
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(parameter)
                elif 'bias' in name:
                    nn.init.zeros_(parameter)


class Alternative(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name != 'lstm_only':
            self.conv1 = nn.Conv1d(5, 32, 3)
            self.conv2 = nn.Conv1d(32, 64, 3)
            self.maxpool = nn.MaxPool1d(2)
            self.dropout_cnn = nn.Dropout(.25)
        if name == 'lstm_only':
            self.lstm = nn.LSTM(5, 128, 2, batch_first=True)
        elif name == 'cnn_gru':
            self.gru = nn.GRU(64, 128, 2, batch_first=True)
        elif name == 'cnn_bilstm':
            self.lstm = nn.LSTM(64, 64, 2, batch_first=True, bidirectional=True)
        elif name != 'cnn_only':
            raise ValueError('Unknown alternative')
        self.dropout_lstm = nn.Dropout(.25)
        self.fc1 = nn.Linear(64 if name == 'cnn_only' else 128, 64)
        self.fc2 = nn.Linear(64, 3)
        initialize(self)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (5, 16):
            raise ValueError('Phase 13 requires (B,5,16)')
        if self.name != 'lstm_only':
            x = self.dropout_cnn(self.maxpool(F.relu(self.conv2(F.relu(self.conv1(x))))))
        if self.name == 'cnn_only':
            x = self.dropout_lstm(x.mean(dim=2))
        else:
            x = x.permute(0, 2, 1)
            if self.name == 'cnn_gru':
                x, _ = self.gru(x)
                x = self.dropout_lstm(x)[:, -1, :]
            elif self.name == 'cnn_bilstm':
                _, (hidden, _) = self.lstm(x)
                # Top layer: complete forward and reverse summaries, not the
                # reverse output at the last temporal position (a partial view).
                x = self.dropout_lstm(torch.cat((hidden[-2], hidden[-1]), dim=1))
            else:
                x, _ = self.lstm(x)
                x = self.dropout_lstm(x)[:, -1, :]
        return self.fc2(F.relu(self.fc1(x)))


def build_model(name):
    if name not in NAMES:
        raise ValueError('Unknown architecture')
    # Direct reuse ensures identical module construction, RNG draws, state keys
    # and forward implementation for the established reference.
    return CNN_LSTM(dict(MODEL_CONFIG)) if name == 'cnn_lstm' else Alternative(name)


def architecture_spec():
    hypotheses = {'cnn_only': 'Does recurrence add value beyond the fixed convolutional stack?',
                  'lstm_only': 'Does the convolutional front-end add useful representation learning?',
                  'cnn_lstm': 'Exact established control',
                  'cnn_gru': 'Can GRU retain quality with smaller recurrent state/parameter cost?',
                  'cnn_bilstm': 'Does both-direction context inside a completed window add value?'}
    result = {}
    # Preserve caller RNG: specification introspection is not initialization of
    # the subsequent training run.
    with torch.random.fork_rng(devices=[]):
        for name in NAMES:
            model = build_model(name)
            result[name] = {'module_definition': str(model), 'input_shape': ['B', 5, 16], 'output_shape': ['B', 3],
                            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                            'state_shapes': {k: list(v.shape) for k, v in model.state_dict().items()},
                            'hypothesis': hypotheses[name], 'conv_shapes': None if name == 'lstm_only' else ['B,5,16 -> B,32,14', 'B,32,14 -> B,64,12', 'maxpool -> B,64,6'],
                            'readout': 'global temporal mean (B,64)' if name == 'cnn_only' else 'top-layer forward/reverse final hidden concatenation (B,128)' if name == 'cnn_bilstm' else 'last recurrent output (B,128)',
                            'hidden_size': None if name == 'cnn_only' else 64 if name == 'cnn_bilstm' else 128,
                            'recurrent_layers': 0 if name == 'cnn_only' else 2,
                            'bidirectional': name == 'cnn_bilstm', 'recurrent_internal_dropout': 0.,
                            'external_readout_dropout': .25, 'conv_dropout': None if name == 'lstm_only' else .25,
                            'initialization': 'PyTorch module construction then Conv He-ReLU, dense Xavier, recurrent input He-linear, recurrent hidden orthogonal, all biases zero; same seed policy'}
    return result
