"""PhaseFormer with a re-parameterised (over-parameterised linear) prediction head.

Only PhaseFormer's final nn.Linear(latent_dim, P_out) is replaced by layers.RepLinear; the
function class is unchanged (still affine in the latent), only the training parameterisation.
`merge_head()` collapses it back, after which the model is a plain PhaseFormer with exactly the
original parameter count.

args: --rep_mode linear|chain|cascade  --rep_depth  --rep_width (0 = 4*max(D, P_out))  --rep_stages
"""
from layers.RepHead import RepLinear
from models.PhaseFormer import Model as PhaseFormerModel


class Model(PhaseFormerModel):
    def __init__(self, configs):
        super().__init__(configs)
        if getattr(configs, 'predictor_use_mlp', False):
            raise NotImplementedError('PhaseFormerRep re-parameterises the default linear predictor only')
        self.predictor.decoder = RepLinear(
            self.latent_dim, self.num_periods_output,
            mode=getattr(configs, 'rep_mode', 'chain'),
            depth=getattr(configs, 'rep_depth', 2),
            width=getattr(configs, 'rep_width', 0),
            stages=getattr(configs, 'rep_stages', 3),
        )

    def merge_head(self):
        """In place: replace the RepLinear by its equivalent nn.Linear (inference form)."""
        if isinstance(self.predictor.decoder, RepLinear):
            self.predictor.decoder = self.predictor.decoder.merged()
        return self
