"""Same fixed 2560-dimensional NAV2 head and bridge rule for 4B/9B."""
import torch
from torch import nn
import nav2_factorized_20261030 as X
import nav2_lora_model_20261028 as L

class Model(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone;self.backbone.requires_grad_(False)
        config=backbone.config.text_config
        self.hidden_size=config.hidden_size
        # Independent streams make the task-head initialization identical in both arms.
        with torch.random.fork_rng():
            torch.manual_seed(20261027);self.head=X.FactHead()
        with torch.random.fork_rng():
            torch.manual_seed(20261112)
            self.input_projection=nn.Linear(self.hidden_size,2560,bias=False)
        with torch.random.fork_rng():
            torch.manual_seed(20261027);self.targets=L.install(self.backbone)
        expected=sum(2 if x=='full_attention' else 1 for x in config.layer_types)
        if len(self.targets)!=expected:raise RuntimeError('Unexpected LoRA coverage')
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

    def train(self,mode=True):
        super().train(mode)
        # Match historical backbone eval semantics; checkpointing only activates
        # in modules marked training, so explicitly enable it for text blocks.
        self.backbone.eval()
        if mode:
            for module in self.backbone.modules():
                if getattr(module,'gradient_checkpointing',False):module.training=True
        return self

    def forward(self,inputs):
        out=self.backbone(**inputs,use_cache=False)
        return self.head(self.input_projection(out.last_hidden_state.float()),inputs.get('attention_mask'))

    def checkpoint(self,update,metadata):
        state=dict(head=self.head.state_dict(),input_projection=self.input_projection.state_dict(),
                    adapter=L.adapter_state(self),update=update,lora_rank=8,lora_alpha=16,
                    targets=self.targets,action_head='3x4',hidden_size=self.hidden_size)
        state.update(metadata)
        return state
