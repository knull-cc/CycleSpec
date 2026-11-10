import copy
import torch


class ModelEMA:
    """Exponential moving average of model weights (a 'shadow' copy used for validation / testing).

    decay   : target decay per optimizer step. With warmup=True the effective decay is
              min(decay, (1+t)/(10+t)), which avoids the shadow being dominated by the random init
              when the run is short (ETTh1 has only ~30 steps per epoch at batch size 256).
    """

    def __init__(self, model, decay=0.996, warmup=True):
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.updates = 0

    def current_decay(self):
        if self.warmup:
            return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        return self.decay

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.current_decay()
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(src[k])
