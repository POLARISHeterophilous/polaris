"""
Polarity- and Selectivity-controlled Graph Attention (POLARIS): entropy-controlled message passing.

A POLARIS layer, for each node i over the augmented neighborhood
N~(i) = N(i) U {i}:

    h_i   = W^T x_i                                   # linear projection
    z_ij  = LeakyReLU(a^T [h_i || h_j])              # additive attention logit
    z_ij += lambda_disc * nu_ij * delta_ij * 1[train] # training-only label bias
    alpha = softmax_{j in N~(i)}(z_ij / tau)         # temperature-scaled weights
    m_i   = AGG_j( alpha_ij * [psi_ij] * h_j )       # contextual prototype
    u_i   = lambda_self h_i + (1-lambda_self) m_i + b # convex self-neighbor mix
    h'_i  = g_i u_i + (1-g_i) h_i                     # gated residual

Bounded scalars: tau = tau_min+(tau_max-tau_min) sigmoid(theta_tau);
lambda_self, lambda_disc, rho = sigmoid(theta).

Signed aggregation (signed=True): a per-edge coefficient
psi_ij = tanh(c^T [h_i || h_j]) in (-1,1) multiplies the message, letting the
operator *subtract* (repel) dissimilar neighbours rather than only down-weight
them. Because |psi_ij| <= 1 and sum_j alpha_ij = 1, the neighbour operator stays
non-expansive (sum_j |alpha_ij psi_ij| <= 1), so Theorem 1 still holds. The
aggregation entropy H_i is still defined on the softmax weights alpha, so the
entropy-control mechanism is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, softmax as scatter_softmax
from torch_geometric.utils import scatter


def _segment_sparsemax(z, index, num_nodes):
    """Sparsemax normalisation within each node's neighbourhood (a sparse
    alternative to the softmax: some weights become exactly zero). Used only for
    the attention-normalisation ablation; the default operator uses softmax."""
    device = z.device
    deg = scatter(torch.ones_like(z), index, dim=0, dim_size=num_nodes, reduce="sum")
    Dmax = int(deg.max().item())
    perm = torch.argsort(index, stable=True)
    idx_s = index[perm]
    posE = torch.arange(idx_s.numel(), device=device)
    first = scatter(posE.to(z.dtype), idx_s, dim=0, dim_size=num_nodes, reduce="min")
    within = (posE - first[idx_s].long())
    M = z.new_full((num_nodes, Dmax), float("-inf"))
    M[idx_s, within] = z[perm]
    zsorted, _ = torch.sort(M, descending=True, dim=1)
    rng = torch.arange(1, Dmax + 1, device=device, dtype=z.dtype)
    cssv = torch.cumsum(zsorted, dim=1) - 1.0
    k = ((zsorted - cssv / rng) > 0).sum(dim=1, keepdim=True).clamp(min=1)
    tau = torch.gather(cssv, 1, k - 1).squeeze(1) / k.squeeze(1).to(z.dtype)
    P = torch.clamp(M - tau.unsqueeze(1), min=0.0)
    out = z.new_zeros(z.numel())
    out[perm] = P[idx_s, within]
    return out


class POLARISLayer(nn.Module):
    """A single Polarity- and Selectivity-controlled Graph Attention layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        agg: str = "sum",            # 'sum' | 'mean' | 'max' | 'logsumexp' | 'multi'
        tau_min: float = 0.25,
        tau_max: float = 4.0,
        negative_slope: float = 0.2,
        multi_aggs: tuple[str, ...] = ("sum", "mean", "max"),
        fixed_rho: float | None = None,   # target NORMALISED aggregation entropy in (0,1)
        signed: bool = False,             # enable signed (attract/repel) aggregation
        use_disc_bias: bool = True,       # training-only label-aware attention bias
        fixed_lam: float | None = None,   # ablation: fix lambda_self (None = learned)
        use_gate: bool = True,            # ablation: disable the gated residual
        n_heads: int = 1,                 # multi-head attention (1 = original single head)
        tau_mode: str = "global",         # 'global' (scalar tau) | 'node' (per-node tau)
        sign_gate: bool = False,          # learn a signed<->unsigned gate
        sign_gate_scope: str = "edge",    # 'edge' | 'node' | 'layer' granularity
        rho_mode: str = "global",         # entropy target: 'global' scalar | 'node' per-node
        attn_norm: str = "softmax",       # 'softmax' | 'sparsemax' (ablation)
    ):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        assert tau_mode in ("global", "node")
        assert rho_mode in ("global", "node")
        assert attn_norm in ("softmax", "sparsemax")
        if attn_norm != "softmax":
            assert n_heads == 1, "sparsemax normalisation is single-head only"
        assert sign_gate_scope in ("edge", "node", "layer")
        if tau_mode == "node" or sign_gate:
            assert n_heads == 1, "per-node tau / sign_gate are single-head options"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.agg = agg
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.negative_slope = negative_slope
        self.multi_aggs = multi_aggs
        self.fixed_rho = fixed_rho
        self.signed = signed
        self.use_disc_bias = use_disc_bias
        self.fixed_lam = fixed_lam
        self.use_gate = use_gate
        self.tau_mode = tau_mode
        self.sign_gate = sign_gate
        self.sign_gate_scope = sign_gate_scope
        self.rho_mode = rho_mode
        self.attn_norm = attn_norm

        # Linear embedding  H = X W
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Additive attention vector a in R^{2*out_dim}, split into src/dst halves.
        self.att_src = nn.Parameter(torch.empty(out_dim))
        self.att_dst = nn.Parameter(torch.empty(out_dim))
        # Convex-mixing learnable bias b
        self.b = nn.Parameter(torch.zeros(out_dim))
        # Gated residual: gate([h_i || u_i]) -> per-feature gate
        self.gate = nn.Linear(2 * out_dim, out_dim)
        # Signed aggregation coefficient: psi_ij = tanh(c^T [h_i || h_j])
        # (per head; for n_heads=1 this is 2*out_dim, identical to before)
        if signed:
            self.sign = nn.Linear(2 * self.head_dim, 1)
        # Signedness gate g in (0,1): psi_eff = g*psi + (1-g)*1.
        # g->0 recovers the unsigned operator (psi_eff=1), g->1 the signed one.
        # Granularity set by sign_gate_scope: per-edge, per-(destination-)node,
        # or a single per-layer scalar switch.
        if sign_gate:
            if sign_gate_scope == "edge":
                self.sign_gate_lin = nn.Linear(2 * self.head_dim, 1)
            elif sign_gate_scope == "node":
                self.sign_gate_lin = nn.Linear(self.head_dim, 1)
            else:  # 'layer': one learned scalar
                self.theta_signgate = nn.Parameter(torch.zeros(()))
        # Per-node temperature head: tau_i from a linear map of h_i (Q1).
        if tau_mode == "node":
            self.tau_lin = nn.Linear(out_dim, 1)
        # Per-node entropy-target head: rho_i from a linear map of h_i.
        if rho_mode == "node":
            self.rho_lin = nn.Linear(out_dim, 1)

        # Bounded-interval scalar parameters (stored as logits, passed through sigmoid)
        self.theta_tau = nn.Parameter(torch.zeros(()))    # -> tau in [tau_min, tau_max]
        self.theta_self = nn.Parameter(torch.zeros(()))   # -> lambda_self in (0,1)
        self.theta_disc = nn.Parameter(torch.zeros(()))   # -> lambda_disc in (0,1)
        self.theta_rho = nn.Parameter(torch.zeros(()))    # -> rho in (0,1)

        # Multi-aggregation convex mixer pi (softmax over aggregators)
        if agg == "multi":
            self.pi_logits = nn.Parameter(torch.zeros(len(multi_aggs)))

        self.reset_parameters()

    def reset_parameters(self):
        self.W.reset_parameters()
        nn.init.xavier_uniform_(self.att_src.view(1, -1))
        nn.init.xavier_uniform_(self.att_dst.view(1, -1))
        nn.init.zeros_(self.b)
        self.gate.reset_parameters()
        if self.signed:
            self.sign.reset_parameters()
        if self.sign_gate and self.sign_gate_scope != "layer":
            self.sign_gate_lin.reset_parameters()
        if self.tau_mode == "node":
            self.tau_lin.reset_parameters()
        if self.rho_mode == "node":
            self.rho_lin.reset_parameters()
        nn.init.zeros_(self.theta_tau)
        nn.init.zeros_(self.theta_self)
        nn.init.zeros_(self.theta_disc)
        nn.init.zeros_(self.theta_rho)

    # -- bounded scalar accessors ----------------------------------------
    @property
    def tau(self) -> torch.Tensor:
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.theta_tau)

    @property
    def lambda_self(self) -> torch.Tensor:
        return torch.sigmoid(self.theta_self)

    @property
    def lambda_disc(self) -> torch.Tensor:
        return torch.sigmoid(self.theta_disc)

    @property
    def rho(self) -> torch.Tensor:
        if self.fixed_rho is not None:
            return torch.as_tensor(self.fixed_rho, device=self.theta_rho.device,
                                   dtype=self.theta_rho.dtype)
        return torch.sigmoid(self.theta_rho)

    # -- aggregation helpers ---------------------------------------------
    def _aggregate(self, weighted_h, index, num_nodes, kind):
        """Aggregate already-weighted messages into node i over N~(i)."""
        if kind in ("sum", "logsumexp"):
            return scatter(weighted_h, index, dim=0, dim_size=num_nodes, reduce="sum")
        if kind == "mean":
            return scatter(weighted_h, index, dim=0, dim_size=num_nodes, reduce="mean")
        if kind == "max":
            return scatter(weighted_h, index, dim=0, dim_size=num_nodes, reduce="max")
        raise ValueError(f"unknown aggregator {kind}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor | None = None,
        train_mask: torch.Tensor | None = None,
        return_entropy: bool = False,
        return_aux: bool = False,
    ):
        if self.n_heads > 1:
            return self._forward_mh(x, edge_index, y, train_mask,
                                    return_entropy, return_aux)
        n = x.size(0)
        h = self.W(x)                                            # (n, out)

        # Augmented neighborhood N~(i) = N(i) U {i}
        edge_index, _ = add_self_loops(edge_index, num_nodes=n)
        src, dst = edge_index[0], edge_index[1]                  # message j(src) -> i(dst)

        # Additive attention logits  z_ij = LeakyReLU(a_dst . h_i + a_src . h_j)
        z = (h * self.att_dst).sum(-1)[dst] + (h * self.att_src).sum(-1)[src]
        z = F.leaky_relu(z, self.negative_slope)

        # Training-only discriminative bias (label-safe: skipped at inference)
        if self.use_disc_bias and self.training and y is not None and train_mask is not None:
            both_train = train_mask[src] & train_mask[dst]
            not_self = src != dst
            nu = (both_train & not_self)
            delta = torch.where(y[src] == y[dst],
                                torch.ones_like(z), -torch.ones_like(z))
            z = z + self.lambda_disc * nu.float() * delta

        # Bounded temperature scaling + neighborhood softmax.
        # 'global': one scalar tau; 'node': a per-node tau_i mapped from h_i (Q1),
        # divided by the destination node's temperature.
        if self.tau_mode == "node":
            tau_node = self.tau_min + (self.tau_max - self.tau_min) * \
                torch.sigmoid(self.tau_lin(h).squeeze(-1))        # (n,)
            zt = z / tau_node[dst]
        else:
            zt = z / self.tau
        if self.attn_norm == "sparsemax":
            alpha = _segment_sparsemax(zt, dst, n)
        else:
            alpha = scatter_softmax(zt, dst, num_nodes=n)        # alpha_ij over N~(i)

        # Per-edge message weight: alpha (and optional signed coefficient psi)
        w = alpha
        if self.signed:
            psi = torch.tanh(self.sign(torch.cat([h[dst], h[src]], dim=-1)).squeeze(-1))
            if self.sign_gate:
                # Learned blend between signed (psi) and unsigned (1) at the
                # configured granularity; |psi_eff|<=1 so Theorem 1 still holds.
                if self.sign_gate_scope == "edge":
                    g = torch.sigmoid(self.sign_gate_lin(
                        torch.cat([h[dst], h[src]], dim=-1)).squeeze(-1))
                elif self.sign_gate_scope == "node":
                    g = torch.sigmoid(self.sign_gate_lin(h).squeeze(-1))[dst]
                else:  # 'layer': one scalar switch shared by all edges
                    g = torch.sigmoid(self.theta_signgate)
                psi = g * psi + (1.0 - g)                         # in (-1,1], |psi_eff|<=1
            w = alpha * psi                                      # in [-alpha, alpha]
        msg = w.unsqueeze(-1) * h[src]

        # Contextual prototype m_i
        if self.agg == "multi":
            pi = torch.softmax(self.pi_logits, dim=0)
            m = sum(pi[k] * self._aggregate(msg, dst, n, kind)
                    for k, kind in enumerate(self.multi_aggs))
        else:
            m = self._aggregate(msg, dst, n, self.agg)

        # Convex self-neighbor mixing  u_i = lam*h_i + (1-lam)*m_i + b
        if self.fixed_lam is None:
            lam = self.lambda_self
        else:
            lam = torch.as_tensor(self.fixed_lam, dtype=h.dtype, device=h.device)
        u = lam * h + (1.0 - lam) * m + self.b

        # Gated residual (ablatable: use_gate=False -> plain mixed state)
        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([h, u], dim=-1)))
            h_out = g * u + (1.0 - g) * h
        else:
            h_out = u

        return self._finish(h_out, alpha, dst, n, h, return_entropy, return_aux)

    def _forward_mh(self, x, edge_index, y, train_mask, return_entropy, return_aux):
        """Multi-head variant: K heads of width head_dim, attention/sign/entropy
        computed per head, head outputs concatenated back to out_dim. For
        n_heads=1 this is numerically equivalent to forward()."""
        n = x.size(0)
        K, Dh = self.n_heads, self.head_dim
        h = self.W(x)                                            # (n, D)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n)
        src, dst = edge_index[0], edge_index[1]
        hh = h.view(n, K, Dh)                                    # (n, K, Dh)
        a_dst = self.att_dst.view(K, Dh); a_src = self.att_src.view(K, Dh)
        z = (hh * a_dst).sum(-1)[dst] + (hh * a_src).sum(-1)[src]   # (E, K)
        z = F.leaky_relu(z, self.negative_slope)
        if self.use_disc_bias and self.training and y is not None and train_mask is not None:
            both = (train_mask[src] & train_mask[dst] & (src != dst)).float()
            delta = torch.where(y[src] == y[dst],
                                torch.ones_like(both), -torch.ones_like(both))
            z = z + (self.lambda_disc * both * delta).unsqueeze(-1)   # (E,1) broadcast
        alpha = scatter_softmax(z / self.tau, dst, num_nodes=n)   # (E, K)
        w = alpha
        if self.signed:
            psi = torch.tanh(self.sign(
                torch.cat([hh[dst], hh[src]], dim=-1)).squeeze(-1))  # (E, K)
            w = alpha * psi
        msg = (w.unsqueeze(-1) * hh[src]).reshape(-1, self.out_dim)  # (E, D)
        if self.agg == "multi":
            pi = torch.softmax(self.pi_logits, dim=0)
            m = sum(pi[k] * self._aggregate(msg, dst, n, kind)
                    for k, kind in enumerate(self.multi_aggs))
        else:
            m = self._aggregate(msg, dst, n, self.agg)           # (n, D)
        if self.fixed_lam is None:
            lam = self.lambda_self
        else:
            lam = torch.as_tensor(self.fixed_lam, dtype=h.dtype, device=h.device)
        u = lam * h + (1.0 - lam) * m + self.b
        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([h, u], dim=-1)))
            h_out = g * u + (1.0 - g) * h
        else:
            h_out = u
        return self._finish(h_out, alpha.mean(-1), dst, n, h,
                            return_entropy, return_aux)

    def _finish(self, h_out, alpha, dst, n, h, return_entropy, return_aux):
        """Shared tail: optionally compute aggregation entropy / aux targets."""
        if not (return_entropy or return_aux):
            return h_out
        # Per-node aggregation entropy  H_i = -sum_j alpha_ij log alpha_ij
        ent_contrib = -alpha * (alpha + 1e-12).log()
        node_entropy = scatter(ent_contrib, dst, dim=0, dim_size=n, reduce="sum")
        if return_aux:
            ones = torch.ones_like(dst, dtype=h.dtype)
            deg = scatter(ones, dst, dim=0, dim_size=n, reduce="sum").clamp(min=1.0)
            log_deg = deg.log()
            if self.rho_mode == "node":
                rho = torch.sigmoid(self.rho_lin(h)).squeeze(-1)   # (n,) per-node
                target = rho * log_deg
                rho_report = rho.mean()
            else:
                rho = self.rho
                target = rho * log_deg
                rho_report = rho
            return h_out, {"entropy": node_entropy, "log_deg": log_deg,
                           "target": target, "rho": rho_report,
                           "tau": self.tau}
        return h_out, node_entropy.mean().detach()


class POLARIS(nn.Module):
    """Multi-layer POLARIS model with a linear classifier head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        agg: str = "sum",
        dropout: float = 0.5,
        **layer_kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for l in range(num_layers):
            self.layers.append(POLARISLayer(dims[l], dims[l + 1], agg=agg, **layer_kwargs))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, y=None, train_mask=None,
                return_entropy=False, return_entropy_loss=False):
        entropies, ent_losses = [], []
        for i, layer in enumerate(self.layers):
            if return_entropy or return_entropy_loss:
                x, aux = layer(x, edge_index, y, train_mask, return_aux=True)
                entropies.append(aux["entropy"].mean().detach())
                ent_losses.append(((aux["entropy"] - aux["target"]) ** 2).mean())
            else:
                x = layer(x, edge_index, y, train_mask)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x)
        if return_entropy_loss:
            ent_loss = torch.stack(ent_losses).mean()
            if return_entropy:
                return logits, torch.stack(entropies), ent_loss
            return logits, ent_loss
        if return_entropy:
            return logits, torch.stack(entropies)
        return logits
