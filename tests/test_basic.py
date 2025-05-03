import numpy as np
from pyrga.RGA import generate_synthetic_field, prepare_physical_domain

def test_generate_synthetic_field():
    nx, ny = 64, 64
    K, alpha, V, beta = generate_synthetic_field(
        nx, ny,
        k=50,
        mu=-4,
        NR=400,
        cov_type='gaussian',
        variance=1.0,
        lx=0.15,
        ly=0.2
    )
    assert K.shape == (nx * ny,)
    assert alpha.shape == (50,)
    assert V.shape == (nx * ny, 50)
    assert isinstance(beta, float)

def test_prepare_physical_domain():
    nx, ny = 64, 64
    well_nodes, Q, dx, dy = prepare_physical_domain(
        nx, ny,
        Lox=320,
        Loy=320,
        q_original=-0.02
    )
    assert len(well_nodes) == 25  # Default 5x5 grid
    assert len(Q) == 25
    assert isinstance(dx, float)
    assert isinstance(dy, float) 