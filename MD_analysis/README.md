# GROMACS Molecular Dynamics Pipeline

## This repository contains a GROMACS-based workflow for peptide–protein molecular dynamics simulations.

**Requirements**
GROMACS 2023.3 or later
CUDA-enabled GPU (optional but recommended)
Linux environment
Repository Structure
scripts/
└── run_md_pipeline.sh

The pipeline automatically generates all required GROMACS parameter files during execution.

**Generated files include:**

ions.mdp
minim.mdp
nvt.mdp
npt.mdp
md_50ns.mdp
md_150ns.mdp
Usage
150 ns Simulation
bash scripts/run_md_pipeline.sh complex.pdb md 150
50 ns Simulation
bash scripts/run_md_pipeline.sh pep6_local20A.pdb md 50
Workflow

**The script performs the following steps:**

Topology generation (pdb2gmx)
Simulation box construction (editconf)
Solvation (solvate)
Ion addition (genion)
Energy minimization
NVT equilibration
NPT equilibration
Production molecular dynamics simulation
Simulation Settings
Force field: AMBER99SB-ILDN
Water model: TIP3P
Electrostatics: Particle Mesh Ewald (PME)
Temperature coupling: V-rescale
Pressure coupling: Parrinello–Rahman
Periodic boundary conditions: XYZ
Integration timestep: 2 fs
Notes

The same workflow can be used for different production simulation lengths by specifying the desired duration (in ns) as the third command-line argument.

## Examples:
```
bash scripts/run_md_pipeline.sh complex.pdb md 50
bash scripts/run_md_pipeline.sh complex.pdb md 100
bash scripts/run_md_pipeline.sh complex.pdb md 150
```
