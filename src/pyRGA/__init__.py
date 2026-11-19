"""
pyRGA - A Python package for groundwater flow simulation and hydraulic tomography
using the Reformulated Geostatistical Approach (RGA).
"""

from .gwsolver import groundwater_solver, transient_groundwater_solver
from .hydraulic_tomography import ht_transient, hydraulic_tomography
from .RGA import (
    add_noise,
    forward_model,
    gauss_newton_dynamic_lambda,
    generate_synthetic_field,
    observation_operator,
    prepare_physical_domain,
)
from .models import DarcyDataset, ForwardResult, GridSpec, HeadObservation, Provenance, Well
from .workflows import run_forward
from .experiments import generate_field, run_pumping_test
from .inverse import run_inverse
from .experiment_models import (
    ExperimentExport, GenerateFieldRequest, GeneratedField, InstalledWell,
    PumpingTestRequest, PumpingTestResult, InverseRequest, InverseResult,
)

__all__ = [
    "add_noise", "forward_model", "gauss_newton_dynamic_lambda",
    "generate_synthetic_field", "groundwater_solver", "ht_transient",
    "hydraulic_tomography", "observation_operator",
    "prepare_physical_domain", "transient_groundwater_solver",
    "DarcyDataset", "ForwardResult", "GridSpec", "HeadObservation",
    "Provenance", "Well", "run_forward",
    "generate_field", "run_pumping_test", "run_inverse", "ExperimentExport",
    "GenerateFieldRequest", "GeneratedField", "InstalledWell",
    "PumpingTestRequest", "PumpingTestResult", "InverseRequest", "InverseResult",
]

__version__ = "0.2.0.dev0"
