"""Three 4-class action distributions preserving original 40-way soft labels."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import nav2_model_packed_20261028 as M
import nav2_lora_model_20261028 as L
CODES=[]
for c in range(64):
 d=((c>>4)&3,(c>>2)&3,c&3)
 if all(d[i]==3 for i in range(3) if 3 in d[:i]):CODES.append(c)
assert len(CODES)==40 and CODES[0]==0 and CODES[-1]==63
DIGITS=torch.tensor([[c//16,(c//4)%4,c%4] for c in CODES],dtype=torch.long)
class FactHead(M.Nav2Head):
 def __init__(self):
  super().__init__();self.ac_head=nn.Linear(1024,12)
 def forward(self,memory,attention_mask=None):
  B=memory.shape[0];padding_mask=None
  if attention_mask is not None:
   if tuple(attention_mask.shape)!=tuple(memory.shape[:2]):raise ValueError('attention mask shape')
   padding_mask=~attention_mask.to(dtype=torch.bool,device=memory.device)
   if padding_mask.all(dim=1).any():raise ValueError('empty attention mask')
  q=self.query_in(torch.cat([self.geom.unsqueeze(0).expand(B,-1,-1),self.cell_embed.weight.unsqueeze(0).expand(B,-1,-1)],-1));x=q
  for ln1,selfat,ln2,xat,ln3,ffn,ln4 in self.dec_layers:
   h=ln1(x);x=x+selfat(h,h,h,need_weights=False)[0];h=ln2(x);x=x+xat(h,memory.float(),memory.float(),key_padding_mask=padding_mask,need_weights=False)[0];h=ln3(x);x=x+ffn(h);x=ln4(x)
  z=self.trunk(x);return self.hm_head(z).squeeze(-1).view(B,31,37),self.ac_head(z).view(B,31,37,3,4)
class Model(nn.Module):
 def __init__(self,backbone):
  super().__init__();self.backbone=backbone;self.backbone.requires_grad_(False);self.head=FactHead();self.targets=L.install(self.backbone)
 def train(self,mode=True):
  super().train(mode);self.backbone.eval();return self
 def forward(self,inputs):
  out=self.backbone(**inputs,use_cache=False);return self.head(out.last_hidden_state,inputs.get('attention_mask'))
def marginals(dist):
 if dist.shape[-1]!=40:raise ValueError('expected 40 soft sequence labels')
 onehot=F.one_hot(DIGITS.to(dist.device),num_classes=4).to(dist.dtype)
 return torch.einsum('...c,cta->...ta',dist,onehot)
def loss(hm,ac,hm_t,dist_t,mask):
 lhm=F.binary_cross_entropy_with_logits(hm,hm_t,pos_weight=torch.tensor(10.,device=hm.device));valid=mask.bool()
 if not valid.any(): raise ValueError('no supervised action cells')
 target=marginals(dist_t[valid]); logp=F.log_softmax(ac[valid].float(),dim=-1)
 assert torch.isfinite(target).all() and torch.allclose(target.sum(-1),torch.ones_like(target.sum(-1)),atol=1e-4)
 lac=-(target*logp).sum(-1).mean();return lhm,lac,lhm+lac
def joint_logits(ac):
 lp=F.log_softmax(ac.float(),-1);d=DIGITS.to(ac.device)
 return lp[...,0,:][...,d[:,0]]+lp[...,1,:][...,d[:,1]]+lp[...,2,:][...,d[:,2]]
