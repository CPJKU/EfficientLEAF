# AudioClassifier Class
## Imports
#basics
from typing import Any, Optional, Sequence, Tuple

#torch
import torch
import torch.nn as nn

class AudioClassifier(torch.nn.Module):
    """Neural network architecture to train an audio classifier from waveforms."""
    def __init__(self,
                 num_outputs: int,
                 frontend: Optional[torch.nn.Module] = None,
                 encoder: Optional[torch.nn.Module] = None):
        super(AudioClassifier, self).__init__()
        self._frontend = frontend
        self._encoder = encoder
        self._pool = torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool2d(1),
            torch.nn.Flatten()
        )
        self._head = torch.nn.Linear(in_features=1280, out_features=num_outputs)
        #self._head = torch.nn.LazyLinear(out_features=num_outputs)

    def forward(self, inputs: torch.Tensor):
        output = inputs
        if self._frontend is not None:
            output = self._frontend(output)  # pylint: disable=not-callable
            if output.ndim == 3:
                output = output[:,None,:,:] #add 1 "color" channel
        if self._encoder:
            output = self._encoder(output)
        output = self._pool(output)
        return self._head(output)
