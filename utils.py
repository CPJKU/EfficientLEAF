# Util Functions
## Imports
#torch
import torch

#from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
def optimizer_to(optim, device):
    '''
    Used to move parameters of optimizer onto a device (like .to(device))
    When using --resume this saves space on the GPU and avoids OOM error
    '''
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

#from https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385
def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)