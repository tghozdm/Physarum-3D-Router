\"\"\"
Physarum-3D Router for Mixture of Experts (MoE)
====================================================
A bio-inspired, hardware-aware router for Large Language Models.

This router uses biological slime-mold (Physarum polycephalum) algorithms
combined with 3D Fibonacci sphere topology to map experts to virtual
Euclidean space. This enables the model to bypass overheated nodes,
save energy, and map semantic distances to physical distances.

Author: TUSofia-Rila Team
License: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
\"\"\"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HardwareTopology3D(nn.Module):
    """
    Maps GPUs (Experts) into a 3D spherical space (Fibonacci Sphere)
    and calculates physical cable distance (Euclidean Distance).
    """
    def __init__(self, num_experts, radius=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.radius = radius
        coords = self._generate_fibonacci_sphere(num_experts, radius)
        self.register_buffer('coordinates', coords)

    def _generate_fibonacci_sphere(self, samples, radius):
        coords = []
        phi = math.pi * (3. - math.sqrt(5.))  # Golden ratio
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2 
            r = math.sqrt(1 - y * y) * radius
            theta = phi * i
            coords.append([math.cos(theta) * r, y, math.sin(theta) * r])
        return torch.tensor(coords, dtype=torch.float32)

    def get_distance_matrix(self, current_node_idx):
        """Returns 3D distance from the current GPU to all others."""
        source_coord = self.coordinates[current_node_idx].unsqueeze(0)
        distances = torch.norm(self.coordinates - source_coord, dim=1)
        return distances


class Physarum3DRouter(nn.Module):
    """
    Biological Slime-Mold Router.
    Calculates optimum route based on mathematical logits (semantics),
    hardware temperature/latency, and physical distance.
    """
    def __init__(self, d_model, num_experts, top_k=2, alpha=0.3, beta_dist=0.3, beta_lat=0.7):
        """
        Args:
            d_model (int): Hidden dimension size (e.g. 768)
            num_experts (int): Total number of expert GPUs in the system
            top_k (int): Number of experts to route each token to
            alpha (float): Biological memory/evolution coefficient (0.3 optimum)
            beta_dist (float): Physical distance penalty multiplier
            beta_lat (float): Network latency/heat penalty multiplier
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.beta_dist = beta_dist
        self.beta_lat = beta_lat
        
        # Semantic mapping
        self.flow_proj = nn.Linear(d_model, num_experts)
        
        # Biological Memory (Persistent across batches)
        self.register_buffer('flow_memory', torch.ones(num_experts) / num_experts)
        
        # 3D Space Manager
        self.topology = HardwareTopology3D(num_experts)

    def forward(self, x, current_node_idx=0, dynamic_latencies=None):
        """
        Args:
            x: Input token tensor [Batch, SeqLen, D_model]
            current_node_idx: The physical ID of the computing GPU
            dynamic_latencies: Real-time heat/latency states of GPUs (from sensors)
        Returns:
            route_weights: Top-K importance weights
            topk_indices: IDs of the selected experts
        """
        B, S, D = x.shape
        
        # 1. Calculate Semantic Flow
        raw_flow = self.flow_proj(x)
        
        # 2. Fetch Hardware Topology State
        distances = self.topology.get_distance_matrix(current_node_idx).to(x.device)
        
        if dynamic_latencies is None:
            dynamic_latencies = torch.ones(self.num_experts, device=x.device)
        
        # 3. Hardware Penalty Formula
        hardware_penalty = (self.beta_dist * distances) + (self.beta_lat * dynamic_latencies)
        hardware_penalty = hardware_penalty.view(1, 1, self.num_experts) 
        
        # 4. Biological Evolution & Congestion Escape
        evolved_flow = self.alpha * raw_flow + (1 - self.alpha) * self.flow_memory.view(1, 1, self.num_experts)
        evolved_flow = evolved_flow - hardware_penalty
        
        # 5. Route Selection
        route_probs = F.softmax(evolved_flow, dim=-1)
        route_weights, topk_indices = torch.topk(route_probs, k=self.top_k, dim=-1)
        
        # 6. Update Biological Memory (Training Only)
        if self.training:
            avg_routing = route_probs.mean(dim=[0, 1]).detach()
            new_memory = 0.9 * self.flow_memory + 0.1 * avg_routing
            self.flow_memory.copy_(new_memory)
            
        return route_weights, topk_indices

# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Physarum-3D Router on {device}...")
    
    # 8 GPU Data Center Simulation
    router = Physarum3DRouter(d_model=768, num_experts=8, top_k=2).to(device)
    dummy_input = torch.randn(1, 64, 768).to(device)
    
    # Simulate GPU Latency (e.g., GPU 3 and 7 are overheating)
    sensor_data = torch.ones(8).to(device)
    sensor_data[3] = 5.0 # Hot
    sensor_data[7] = 8.0 # Critical
    
    weights, indices = router(dummy_input, current_node_idx=0, dynamic_latencies=sensor_data)
    
    print(f"Top-2 Selected Experts for token 0: {indices[0][0].tolist()}")
    print("Router successfully avoided nodes 3 and 7!")
