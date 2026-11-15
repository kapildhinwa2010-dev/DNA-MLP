import pytest
import torch
from dna_mlp import DNAMLP, create_dna_mlp

class TestDNAMLP:
    def test_init(self):
        m = DNAMLP(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = DNAMLP(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_dna_mlp()
        assert isinstance(m, DNAMLP)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
