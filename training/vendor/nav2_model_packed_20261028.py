"""NAV2 head on top of a frozen Qwen3.5-4B backbone.

Data flow: RGB+instruction -> Qwen3_5Model (frozen, bf16) -> last_hidden [B,T,2560]
        -> polar query decoder (2 layers, self+cross attention) -> shared MLP
        -> heatmap head [B,31,37] + action head [B,31,37,40]

The head runs in fp32 on top of the bf16 backbone output (cast).
Polar grid: rho 0..1.5m step 0.05 (31), theta -90..+90 step 5deg (37).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NR, NA = 31, 37
NQ = NR * NA
D = 2560


def polar_geom_features():
    rhos = torch.arange(NR, dtype=torch.float32) * 0.05
    ths = torch.deg2rad(torch.arange(NA, dtype=torch.float32) * 5.0 - 90.0)
    rr, tt = torch.meshgrid(rhos, ths, indexing="ij")
    r = rr.reshape(-1)
    t = tt.reshape(-1)
    return torch.stack([r, r * torch.sin(t), r * torch.cos(t),
                        torch.sin(t), torch.cos(t), (r / 1.5) ** 2,
                        torch.sin(2 * t), torch.cos(2 * t)], -1)  # [NQ, 8]


class Nav2Head(nn.Module):
    def __init__(self, d_model=D, n_heads=8, ffn=5120, layers=2, n_actions=40,
                 polar_dim=8, cell_embed=256):
        super().__init__()
        self.cell_embed = nn.Embedding(NQ, cell_embed)
        self.query_in = nn.Linear(polar_dim + cell_embed, d_model)
        self.dec_layers = nn.ModuleList()
        for _ in range(layers):
            self.dec_layers.append(nn.ModuleList([
                nn.LayerNorm(d_model),
                nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                nn.LayerNorm(d_model),
                nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                nn.LayerNorm(d_model),
                nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(),
                              nn.Linear(ffn, d_model)),
                nn.LayerNorm(d_model),
            ]))
        self.trunk = nn.Sequential(nn.Linear(d_model, 1024), nn.GELU())
        self.hm_head = nn.Linear(1024, 1)
        self.ac_head = nn.Linear(1024, n_actions)
        self.register_buffer("geom", polar_geom_features(), persistent=False)

    def forward(self, memory, attention_mask=None):
        # memory: [B, T, D] backbone hidden states
        B = memory.shape[0]
        padding_mask = None
        if attention_mask is not None:
            if tuple(attention_mask.shape) != tuple(memory.shape[:2]):
                raise ValueError('attention_mask must match [batch, token] dimensions')
            padding_mask = ~attention_mask.to(dtype=torch.bool, device=memory.device)
            if padding_mask.all(dim=1).any():
                raise ValueError('each sample must contain at least one unmasked token')
        q = self.query_in(torch.cat(
            [self.geom.unsqueeze(0).expand(B, -1, -1),
             self.cell_embed.weight.unsqueeze(0).expand(B, -1, -1)], -1))
        x = q
        for ln1, selfat, ln2, xat, ln3, ffn, ln4 in self.dec_layers:
            h = ln1(x)
            x = x + selfat(h, h, h, need_weights=False)[0]
            h = ln2(x)
            x = x + xat(h, memory.float(), memory.float(),
                        key_padding_mask=padding_mask, need_weights=False)[0]
            h = ln3(x)
            x = x + ffn(h)
            x = ln4(x)
        z = self.trunk(x)
        hm = self.hm_head(z).squeeze(-1).view(B, NR, NA)
        ac = self.ac_head(z).view(B, NR, NA, -1)
        return hm, ac


class Nav2Model(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.head = Nav2Head()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs):
        with torch.no_grad():
            out = self.backbone(**inputs)
        memory = out.last_hidden_state
        return self.head(memory, inputs.get('attention_mask'))


def nav2_loss(hm, ac, hm_t, dist_t, msk, pos_weight=10.0):
    """Multi-peak loss: full-image heatmap BCE + soft-label CE over ALL mask
    cells using action_class_distribution [B,31,37,40]."""
    loss_hm = F.binary_cross_entropy_with_logits(
        hm, hm_t, pos_weight=torch.tensor(pos_weight, device=hm.device))
    B, NR_, NA_, C = ac.shape
    flat_ac = ac.reshape(-1, C)
    flat_d = dist_t.reshape(-1, C)
    flat_m = msk.reshape(-1) > 0
    if flat_m.any():
        logp = F.log_softmax(flat_ac[flat_m].float(), dim=-1)
        loss_ac = -(flat_d[flat_m] * logp).sum(-1).mean()
    else:
        loss_ac = torch.zeros((), device=hm.device)
    return loss_hm, loss_ac, loss_hm + loss_ac


@torch.no_grad()
def nav2_metrics(hm, ac, hm_t, dist_t, msk, radius=2):
    """Diagnostics only — NOT navigation SR.

    peak_hit: judged at the model's ACTUAL predicted peak (global argmax of
    sigmoid(heatmap)); hit if within Chebyshev radius of any target peak cell.
    action_at_peak: checked AT the predicted peak cell — valid only if that
    cell carries action supervision (mask); argmax(logits) vs argmax(dist).
    """
    prob = torch.sigmoid(hm)
    hit, act_ok, act_masked, act_na, n = 0, 0, 0, 0, 0
    for b in range(hm.shape[0]):
        tgt = (hm_t[b] > 0.5).nonzero(as_tuple=False)
        if tgt.numel() == 0:
            continue
        n += 1
        k = int(prob[b].reshape(-1).argmax())
        pi, pj = k // NA, k % NA
        cheb = torch.max((tgt[:, 0] - pi).abs(), (tgt[:, 1] - pj).abs())
        if int(cheb.min()) <= radius:
            hit += 1
        if bool(msk[b, pi, pj] > 0):
            act_masked += 1
            if int(ac[b, pi, pj].argmax()) == int(dist_t[b, pi, pj].argmax()):
                act_ok += 1
        else:
            act_na += 1
    return {"peak_hit_rate": hit / max(n, 1),
            "action_match_at_pred_peak": act_ok / max(act_masked, 1),
            "pred_peak_unmasked": act_na, "n": n}
