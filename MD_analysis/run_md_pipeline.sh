#!/bin/bash
set -e

GMX=gmx
GPU_ID=${GPU_ID:-1}

PDB=${1}
PREFIX=${2:-md}
NS=${3:-150}

NSTEPS=$((NS * 500000))

echo "Input: $PDB"
echo "Prefix: $PREFIX"
echo "Production length: ${NS} ns"
echo "Steps: $NSTEPS"

$GMX pdb2gmx -f "$PDB" -o processed.gro -p topol.top -ignh

$GMX editconf -f processed.gro -o boxed.gro -c -d 1.0 -bt dodecahedron
$GMX solvate -cp boxed.gro -cs spc216.gro -o solv.gro -p topol.top

cat > ions.mdp << EOF
integrator  = steep
emtol       = 1000
emstep      = 0.001
nsteps      = 500
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.2
rvdw        = 1.2
pbc         = xyz
EOF

$GMX grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2
echo "Select SOL for ion replacement"
$GMX genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral

cat > minim.mdp << EOF
integrator      = steep
emtol           = 1000
emstep          = 0.001
nsteps          = 50000
cutoff-scheme   = Verlet
nstlist         = 20
rlist           = 1.2
vdwtype         = Cut-off
rvdw            = 1.2
coulombtype     = PME
rcoulomb        = 1.2
pbc             = xyz
constraints     = none
EOF

$GMX grompp -f minim.mdp -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 2
$GMX mdrun -v -deffnm em -gpu_id "$GPU_ID"

cat > nvt.mdp << EOF
define                  = -DPOSRES
integrator              = md
dt                      = 0.001
nsteps                  = 100000
nstxout-compressed      = 1000
nstenergy               = 500
nstlog                  = 500
continuation            = no
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.2
vdwtype                 = Cut-off
rvdw                    = 1.2
coulombtype             = PME
rcoulomb                = 1.2
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1 0.1
ref_t                   = 300 300
pcoupl                  = no
pbc                     = xyz
gen_vel                 = yes
gen_temp                = 300
gen_seed                = -1
EOF

$GMX grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2
$GMX mdrun -v -deffnm nvt -gpu_id "$GPU_ID"

cat > npt.mdp << EOF
define                  = -DPOSRES
integrator              = md
dt                      = 0.001
nsteps                  = 500000
nstxout-compressed      = 5000
nstenergy               = 1000
nstlog                  = 1000
continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.2
vdwtype                 = Cut-off
rvdw                    = 1.2
coulombtype             = PME
rcoulomb                = 1.2
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1 0.1
ref_t                   = 300 300
pcoupl                  = Berendsen
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5
refcoord_scaling        = com
pbc                     = xyz
gen_vel                 = no
EOF

$GMX grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 2
$GMX mdrun -v -deffnm npt -gpu_id "$GPU_ID"

cat > "${PREFIX}_${NS}ns.mdp" << EOF
integrator              = md
dt                      = 0.002
nsteps                  = $NSTEPS
nstxout-compressed      = 5000
nstenergy               = 5000
nstlog                  = 5000
continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
cutoff-scheme           = Verlet
nstlist                 = 100
rlist                   = 1.2
vdwtype                 = Cut-off
rvdw                    = 1.2
coulombtype             = PME
rcoulomb                = 1.2
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1 0.1
ref_t                   = 300 300
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 5.0
ref_p                   = 1.0
compressibility         = 4.5e-5
refcoord_scaling        = com
pbc                     = xyz
gen_vel                 = no
EOF

$GMX grompp -f "${PREFIX}_${NS}ns.mdp" -c npt.gro -t npt.cpt -p topol.top -o "${PREFIX}_${NS}ns.tpr" -maxwarn 2

nohup $GMX mdrun -v -deffnm "${PREFIX}_${NS}ns" -gpu_id "$GPU_ID" > "${PREFIX}_${NS}ns.nohup.log" 2>&1 &

echo "Submitted: ${PREFIX}_${NS}ns"
echo "Log: ${PREFIX}_${NS}ns.nohup.log"
