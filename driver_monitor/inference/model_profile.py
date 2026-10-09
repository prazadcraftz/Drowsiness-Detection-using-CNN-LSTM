"""The two immutable local model contracts; importing this module does no inference."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CLASS_MAPPING = ((0, 'Normal'), (1, 'Yawning'), (2, 'Eye Close'))


@dataclass(frozen=True)
class ModelProfile:
    name: str
    architecture: str
    checkpoint: str
    normalizer: str
    sequence_length: int
    parameters: int
    checkpoint_sha256: str
    normalizer_sha256: str
    compatibility_mode: str

    @property
    def class_mapping(self):
        return dict(CLASS_MAPPING)

    def paths(self):
        paths = (ROOT / self.checkpoint, ROOT / self.normalizer)
        if any(not p.resolve().is_relative_to(ROOT) for p in paths):
            raise ValueError('Profile asset outside repository root')
        return paths

    def validate_assets(self):
        paths = self.paths()
        for path, expected in zip(paths, (self.checkpoint_sha256, self.normalizer_sha256)):
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError(f'{self.name} asset SHA256 mismatch: {path}')
        return paths


PROFILES = {
    'legacy': ModelProfile('legacy', 'cnn_lstm', 'saved_model.pt', 'norm.npy', 16, 246595,
        '162426d6292a87162188b70d7bac6704e5bf217f8ff1fe5eaafc41085aa0c551',
        '54b2a1fc66cc8eab3b68ba2b5fe081558aabc6e48c79f752d1d49026494e1900',
        'historical CNN-LSTM16, unchanged feature extraction and zero-anchored normalization'),
    'lstm32': ModelProfile('lstm32', 'lstm_only',
        'models/phase15/lstm_only_w32_full_development.pt', 'models/phase15/lstm32_dev_norm.npy',
        32, 209667, '1dcdcdbd17378190a6b7c429fa5ccf5b621dcf0adae6541d89d789a1d5b764ed',
        '9b397ae1327b515c5505610fffa37e198eb5ccd24ec7bf1d112eacec0f950edc',
        'Phase15 full-development normalization, unchanged historical raw features'),
}


def get_profile(name):
    try:
        return PROFILES[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f'Unknown model profile {name!r}; choose legacy or lstm32') from error


def build_model(profile, legacy_settings=None, *, legacy_constructor=None):
    if profile.name == 'legacy':
        if legacy_constructor is None:
            from model import CNN_LSTM
            legacy_constructor = CNN_LSTM
        model = legacy_constructor(legacy_settings)
    elif profile.name == 'lstm32':
        # Reuse the verified Phase14 variable-window adapter over the frozen
        # Phase13 Alternative constructor/state/operations. No second architecture.
        from evaluation.windows import build_model as frozen_variable_model
        model = frozen_variable_model('lstm_only', 32)
    else:
        raise ValueError('Unknown model profile')
    if sum(p.numel() for p in model.parameters()) != profile.parameters:
        raise ValueError('Model parameter contract changed')
    return model.to('cpu')
