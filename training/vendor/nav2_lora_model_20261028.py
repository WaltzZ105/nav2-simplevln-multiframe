"""Standalone LoRA for Qwen3.5 NAV2, shared base and identical head initialization."""
import math,re,torch
import torch.nn as nn
import nav2_model_packed_20261028 as M
class LoRALinear(nn.Module):
 def __init__(self, base, rank=8, alpha=16):
  super().__init__();self.base=base;self.rank=rank;self.scale=alpha/rank
  self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32));self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32));nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
 def forward(self,x):
  y=self.base(x);d=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.lora_A),self.lora_B)*self.scale
  return y+d.to(y.dtype)
PATTERN=re.compile(r'language_model\.layers\.\d+\.(?:linear_attn\.in_proj_qkv|self_attn\.(?:q_proj|v_proj))$')
def install(backbone):
 targets=[name for name,mod in backbone.named_modules() if PATTERN.search(name) and isinstance(mod,nn.Linear)]
 if len(targets)!=40:raise RuntimeError(f'expected 40 Qwen attention projections; got {len(targets)}: {targets}')
 for name in targets:
  parent_name,key=name.rsplit('.',1);parent=backbone.get_submodule(parent_name);setattr(parent,key,LoRALinear(getattr(parent,key)))
 return targets
class Model(M.Nav2Model):
 def __init__(self,backbone):
  super().__init__(backbone);self.targets=install(self.backbone)
 def forward(self,inputs):
  out=self.backbone(**inputs,use_cache=False)
  return self.head(out.last_hidden_state,inputs.get('attention_mask'))
def adapter_state(model):
 return {k:v.detach().cpu() for k,v in model.backbone.state_dict().items() if k.endswith('.lora_A') or k.endswith('.lora_B')}
def load_adapter(model,state):
 expected={k for k in model.backbone.state_dict() if k.endswith('.lora_A') or k.endswith('.lora_B')}
 if expected!=set(state):raise ValueError('adapter keys mismatch')
 result=model.backbone.load_state_dict(state,strict=False)
 if result.unexpected_keys:raise ValueError('unexpected adapter keys')
