def model_size(network):
    t = sum([p.data.nelement() for p in network.parameters() if p.requires_grad is True])
    t = t / 1000
    t = t / 1000
    return f"{t:.4f} MB"
