# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[mace, io]"
# ]
# ///


# %% [markdown]
"""
# Introduction to TorchSim

This tutorial introduces TorchSim's high-level API for molecular dynamics simulations
and geometry optimizations. The high-level API provides simple, powerful interfaces
that abstract away the complexities of setting up atomistic simulations while still
allowing for customization.

## Introduction

TorchSim's high-level API consists of three primary functions:

1. `integrate` - For running molecular dynamics simulations
2. `optimize` - For geometry optimization
3. `static` - For one-time energy/force calculations on a diversity set of systems

These functions handle:
* Automatic state initialization from various input formats
* Memory-efficient GPU operations via autobatching
* Trajectory reporting and property calculation
* Custom convergence criteria

Over the course of the tutorial, we will fully explain the example in the README
by steadily adding functionality.
"""

# %% [markdown]
"""
## Basic Molecular Dynamics

We'll start with a simple example: simulating a silicon system using a Lennard-Jones
potential. First, let's set up our model and create an atomic structure:
"""

# %%
import torch_sim as ts
import torch
from ase.build import bulk
from ase.io import read, write
from torch_sim.models.lennard_jones import LennardJonesModel
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Force CPU for this tutorial

n_steps = 50
# Create a Lennard-Jones model with parameters suitable for Si
lj_model = LennardJonesModel(
    sigma=2.0,  # Å, typical for Si-Si interaction
    epsilon=0.1,  # eV, typical for Si-Si interaction
    device=device,
    dtype=torch.float64,
)

lj_model = LennardJonesModel(
    use_neighbor_list=True,
    sigma=2.0,  # Å, typical for Si-Si interaction
    epsilon=0.1,  # eV, typical for Si-Si interaction
    cutoff=10.0,  # Å, a common cutoff for LJ potentials
    device=device,
    dtype=torch.float64,
    compute_forces=True,
    compute_stress=True,
)

# Load the MACE "small" foundation model
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    dtype=torch.float64,
    compute_forces=True,
)



# %% Create multiple systems to simulate
systems = []
for _ in range(10):
    atoms = read("POSCAR")
    systems.append(atoms)


# %% [markdown]
"""
## Autobatching

The `integrate` function also supports autobatching, which automatically determines
the maximum number of systems that can fit in memory and splits up the systems to make
optimal use of the GPU. This abstracts away the complexity of managing memory when
running more systems than can fit on the GPU.

Ignore the following cell, it just exists so that the example runs on CPU.
"""


# %%
ts.autobatching.determine_max_batch_size = lambda *args, **kwargs: 10  # type: ignore[invalid-assignment]


# %% [markdown]
"""
We enable autobatching by simply setting the `autobatcher` argument to `True`.
"""

filenames = [f"tmp/batch_traj_{i}.h5" for i in range(len(systems))]
# filenames = "batch_traj.h5"
prop_calculators = { 
    10: {"potential_energy": lambda state: state.energy},
    20: {
        "kinetic_energy": lambda state: ts.calc_kinetic_energy(
            momenta=state.momenta, masses=state.masses
        )
    },  
}

# Create a reporter that handles multiple trajectories
batch_reporter = ts.TrajectoryReporter(
    filenames,
    state_frequency=10,
    prop_calculators=prop_calculators,
)


# %% Run the simulation with batch reporting
final_state = ts.integrate(
    system=systems,
    model=lj_model,
    integrator=ts.Integrator.nvt_nose_hoover,
    n_steps=n_steps,
    temperature=2000,
    timestep=0.002,
    autobatcher=True,
    trajectory_reporter=batch_reporter,
)

# %% [markdown]
"""
We can analyze each trajectory individually:
"""

# %% Calculate final energy per atom for each system
final_energies_per_atom = []
for sys_idx, filename in enumerate(filenames):
    with ts.TorchSimTrajectory(filename) as traj:
        final_energy = traj.get_array("potential_energy")[-1].item()
        n_atoms = len(traj.get_atoms(-1))
        final_energies_per_atom.append(final_energy / n_atoms)
        print(
            f"System {sys_idx}: {final_energy:.6f} eV, {final_energy / n_atoms:.6f} eV/atom"
        )

