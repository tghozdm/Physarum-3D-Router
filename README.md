# Physarum-3D Router 🦠🌌

**Bio-Inspired, Hardware-Aware Sparse MoE Routing Engine for LLMs**

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

The **Physarum-3D Router** is a next-generation routing engine for Mixture-of-Experts (MoE) language models. Unlike standard routers (Softmax/Top-K with auxiliary loss), this router is **hardware-aware**, **biologically self-balancing**, and **patent-free**.

It uses algorithms inspired by the slime mold *Physarum polycephalum* combined with 3D Fibonacci Sphere topology mapping to optimize for semantic accuracy, network latency, GPU temperatures, and physical datacenter distances — simultaneously.

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Biological Memory (EMA)** | Remembers routing history via Exponential Moving Average. Auto-heals: routes away from overheated/dead GPUs, reintegrates them on recovery |
| **3D Fibonacci Sphere** | Maps compute nodes onto a 3D sphere using golden-ratio spacing. Euclidean distance = cross-rack latency estimation |
| **Cubic Congestion Escape** | `γ · M³` penalty creates sharp gradient cliffs — instantly evacuates overloaded experts |
| **True Sparse MoE** | Hard Top-K (default: 2). Only selected experts execute. ~60% compute savings vs dense MoE |
| **No Auxiliary Loss** | Expert balancing is intrinsic via biological memory — unlike Google Switch Transformer |
| **Distance Matrix Caching** | Avoids redundant topology computation across forward passes |
| **Alpha Warm-up Schedule** | Linear decay from pure semantic (α=1.0) → bio-evolved (α=0.05) for training stability |
| **Distributed Sync** | `dist.all_reduce` synchronizes biological memory across GPUs in DDP/FSDP |

---

## 📊 Benchmark Results

### Physarum-3D vs Google Switch Transformer

| Metric | Physarum-3D | Switch-T | Winner |
|--------|:---:|:---:|:---:|
| **Load Balancing (CV)** | Lower variance | Higher variance | **Physarum** |
| **Auxiliary Loss Required** | ❌ None | ✅ Required | **Physarum** |
| **Expert Death Prevention** | Intrinsic (cubic penalty) | Manual tuning | **Physarum** |
| **Hardware Awareness** | 3D topology + live sensors | None | **Physarum** |
| **Adversarial Resilience** | Strong | Weak | **Physarum** |
| **Meltdown Recovery** | <5 steps | N/A | **Physarum** |
| **Cold Start Latency** | ~135ms | ~386ms | **Physarum** |
| **Energy Savings** | ~62.5% vs dense | N/A | **Physarum** |

### Expert Load Distribution (50 batches, no auxiliary loss)

```
Expert 0:  13.9% ██████
Expert 1:  19.6% █████████
Expert 2:  16.9% ████████
Expert 3:  17.4% ████████
Expert 4:  17.8% ████████
Expert 5:  14.5% ███████

✅ Zero dead experts — balanced purely by biological memory!
```

*Benchmark scripts: `evaluate_3d_router_academic.py`, `benchmark_vs_switch.py`*

---

## 🧠 How It Works

```
Token x ──→ Semantic Projection (Linear: d_model → num_experts)
                    ↓
          ┌─── Biological Evolution ───┐
          │  F = α·Semantic + (1-α)·M  │  ← M = EMA biological memory
          └────────────┬───────────────┘
                       ↓
          ┌─── Hardware Penalty ───────┐
          │  P = β_d·Distance_3D       │  ← Fibonacci sphere Euclidean
          │    + β_l·Latency_sensors   │  ← Real-time GPU heat
          └────────────┬───────────────┘
                       ↓
          ┌─── Congestion Escape ──────┐
          │  C = γ · M³                │  ← Cubic! Sharp evacuation
          └────────────┬───────────────┘
                       ↓
             F_final = F - P - C
                       ↓
             Softmax → Top-K → Expert IDs + Weights
```

### Key Equations

**Evolved Flow:**
```
F_evolved = α · F_semantic + (1 - α) · M_biological
```

**Hardware Penalty:**
```
P_hardware = β_dist · D_3D(source, target) + β_lat · L_dynamic
```

**Congestion Escape (cubic):**
```
P_congestion = γ · M³
```
> The cubic exponent creates an exponentially increasing penalty — much sharper than linear. When an expert exceeds ~60% load, the penalty overpowers raw logits and forces immediate traffic redistribution.

**Biological Memory Update (training only):**
```
M_new = ema_decay · M_old + (1 - ema_decay) · avg_routing
```

---

## ⚙️ Optimized Hyperparameters

Found via 432-combination grid search across simulated datacenter conditions:

| Parameter | Value | Role |
|-----------|:-----:|------|
| `alpha` | 0.05 | Bio memory weight (low = trust semantics more) |
| `beta_dist` | 0.0 | 3D distance penalty (0 for single-GPU training) |
| `beta_lat` | 0.5 | Latency/heat penalty multiplier |
| `gamma` | 150.0 | Congestion penalty strength (3x default) |
| `ema_decay` | 0.8 | Memory adaptation speed (0.8 = fast adaptation) |
| `top_k` | 2 | Active experts per token |

---

## 🛠️ Usage

### Standalone Router

```python
import torch
from physarum_3d_router import Physarum3DRouter

# Initialize: 6 experts, d_model=640, top-2 sparse routing
router = Physarum3DRouter(
    d_model=640, 
    num_experts=6, 
    top_k=2,
    alpha=0.05,       # Optimized bio memory coefficient
    gamma=150.0,      # Optimized congestion penalty
    ema_decay=0.8     # Fast memory adaptation
)

x = torch.randn(8, 128, 640)  # (batch, seq_len, d_model)

# With GPU heat sensors (optional)
latencies = torch.ones(6)
latencies[3] = 5.0  # GPU 3 overheating

weights, indices = router(x, current_node_idx=0, dynamic_latencies=latencies)
# weights: (8, 128, 2) — importance scores
# indices: (8, 128, 2) — selected expert IDs
```

### Alpha Warm-up (for training)

```python
router.set_warmup(warmup_steps=2000)  # α: 1.0 → 0.05 over 2000 steps

for step in range(total_steps):
    weights, indices = router(x)
    # ... training logic ...
    router.step_warmup()  # Advance α schedule
```

### Integrated in TUSofia-Rila V2 (True Sparse MoE)

```python
from tusofia_rila_v2 import TUSofiaRilaV2, CONFIG_V2

model = TUSofiaRilaV2(CONFIG_V2).to("cuda")
# Physarum-3D Router is integrated:
#   - Only top-2 experts execute per token
#   - Remaining 4 experts are SKIPPED (true sparse)
#   - No auxiliary loss — bio memory handles balance
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `physarum_3d_router.py` | Core router implementation (standalone, 204 lines) |
| `tusofia_rila_v2.py` | Full model with sparse MoE integration |
| `evaluate_3d_router_academic.py` | Academic test suite (9 tests) |
| `benchmark_vs_switch.py` | Head-to-head vs Google Switch Transformer |
| `generate_academic_figures.py` | Paper-quality figure generation |
| `optimize_router_params.py` | 432-combination hyperparameter grid search |
| `test_sparse_moe.py` | Sparse execution verification (6 tests) |

---

## 📝 Citation

```bibtex
@software{physarum3d_router_2025,
  title={Physarum-3D Router: Bio-Inspired Hardware-Aware 
         Sparse MoE Routing Without Auxiliary Loss},
  author={TUSofia-Rila  ,tghozdm },
  year={2025},
  description={A routing algorithm for MoE LLMs inspired by 
               Physarum polycephalum slime mold networks, 
               with 3D Fibonacci sphere topology mapping 
               and intrinsic expert load balancing.}
}
```

---

⚖️ License & Commercial Use
This project and its associated algorithms (including Physarum-3D Router, QuadLaser Positional Embedding, and AdaptVec) are strictly licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/

Summary of Restrictions:
Attribution (BY): You must give appropriate credit, provide a link to the license, and indicate if changes were made.

NonCommercial (NC): You may NOT use this material, its mathematical models, or its code for commercial purposes.

NoDerivatives (ND): If you remix, transform, or build upon the material, you may NOT distribute the modified material.

Enterprise Licensing
For commercial use, integration into proprietary LLM/MoE architectures, cloud deployment, or enterprise data center usage, a separate commercial license is strictly required.

Please contact the lead authors and the Technology Transfer Office for B2B licensing inquiries:

Lead Architect: Tolgahan Özdemir

Email: a1bg.java225@passfwd.com
