# Pose
A bare-metal Python library for building and manipulating protein and nucleic acid molecular structures

![Python >= 3](https://img.shields.io/badge/python-%3E%3D3-blue)
![NumPy](https://img.shields.io/badge/dependency-NumPy-orange)
![License: GPL v2](https://img.shields.io/badge/license-GPL%20v2-green)

<img src="pose/Video1.gif" width="25%"/><img src="pose/Video2.gif" width="25%"/><img src="pose/Video3.gif" width="25%"/><img src="pose/Video4.gif" width="25%"/>
<img src="pose/Video5.gif" width="25%"/><img src="pose/Video6.gif" width="25%"/><img src="pose/Video8.gif" width="25%"/><img src="pose/Video7.gif" width="25%"/>

---

## Video Tutorial

**Watch the full walkthrough:** [Video Tutorial on YouTube](https://www.youtube.com/@SariSabban)

---

## What is Pose?

Pose constructs a data structure for protein or nucleic acid molecules that contains all relevant information defining a polymer. Primary information includes the XYZ cartesian coordinates of each atom, the identity and charge of each atom, and the bond graph of the entire molecule. Secondary information includes the FASTA sequence, radius of gyration, potential energy, and the secondary structure assignment for each protein residue.

Using this data structure, Pose can build and manipulate polypeptides and nucleic acids: construct any polypeptide or nucleic acid from sequence, move dihedral and rotamer angles, mutate residues and base pairs, and measure bond lengths and angles. It is designed as a substrate for higher-level protocols such as simulated annealing, molecular dynamics, and machine learning-based molecular design.

**Key features:**
- Designed to be extremely stable bare-metal python: NumPy is the only dependency for the core `Pose` and `Molecule` classes
- 26 amino acids supported by default (20 canonical + 6 non-canonical: ALY, MSE, TPO, SEC, TRF, PTR), can be extended to 100+
- Support for both L-amino acids and D-amino acids (mixed sequences fully supported)
- 5 DNA and RNA canonical nucleotides
- Full bond graph with atom partial charges
- Measure and rotate protein dihedral and rotamer angles (φ/ψ/ω/χ)
- Measure and rotate nucleic acids dihedral angles (α/β/γ/δ/ε/ζ/χ)
- Measure and adjust the distance and angle between any atoms
- PDB and mmCIF file import and export
- Pythonic zero-based indexing throughout (unlike PDB's one-based convention)
- Bundled bare-metal force field with analytic gradients, periodic-boundary support, and a hash-cached topology
- Hybrid physics + statistical `Score()` for protein design — chirality-aware end to end across L-AA, D-AA, mixed L/D, and non-canonical residues
- Three production protocols: minimisation, simulated annealing, and molecular dynamics simulation. (Rotamer sidechain packing is pending re-implementation alongside `Score()`.)

---

## Installation

**Dependencies:** Python >= 3, NumPy

For virtualenv:
```bash
pip install git+https://github.com/sarisabban/Pose
```

For anaconda:
```bash
conda create -n ENVIRONMENT python=3
conda activate ENVIRONMENT
pip3 install git+https://github.com/sarisabban/Pose
```

---

## Quick Start

```python
from pose import *

# === Build a peptide (no external files needed) ===
p = Pose()
p.Build('MSLESNRGI', chain='A', fmt='Protein')   # Uppercase=L, lowercase=D
p.GetInfo()                                      # Print structured summary

# === Inspect ===
print('Sequence:', p.data['FASTA'])
print('Mass:',     p.data['Mass'], 'Da')
print('Rg:',       p.data['Rg'], 'Å')

# === Manipulate backbone (zero-based indexing) ===
p.RotateDihedral(1, -60, 'PHI')
p.RotateDihedral(1, -45, 'PSI')
p.Mutate(2, 'V')                                 # residue 2 (Leu) → Val
p.Export('peptide.pdb')

# === Same APIs work for D-amino acids and mixed L/D sequences ===
p_d = Pose()
p_d.Build('MsLeSnRgI', chain='A', fmt='Protein') # mixed L/D — chirality-aware

# === Energy and protocols ===
ff = ForceField()

E           = ff(p)                              # potential energy (kJ/mol)
E, F        = ff(p, grad=True)                   # also return per-atom forces

E_min, log  = Minimise(p, ff)                    # FIRE2 relaxation
E_md,  log  = MolecularDynamics(p, ff, n_steps=1000,
              dt_fs=2.0, T=300.0, thermostat='langevin')
```

> ⚠️ **`Score()` and `Pack()` are temporarily non-functional.** `Score.__init__` raises `NotImplementedError` because the database keys it consumed (`weights`, `ref_state`, `lk_solvation`, `hbond`, `kbp`, `rotamer`) were removed during the recent force-field refactor. `Pack()` depends on `Score()` and is therefore also broken. Both are pending migration to SMIRKS-driven score terms; the math and API surfaces below are preserved as developer reference.

> Uppercase sequence letters build L-amino acids (natural form), lowercase builds D-amino acids (mirror images), and mixed sequences (e.g. `'MsLeSnRgI'` above) are fully supported.

**Importing a PDB file:**
```python
p = Pose()
p.Import('1TQG.pdb', chain='A')   # Or '1BNA.pdb' for a DNA/RNA structure
p.ReBuild()                       # Adds missing hydrogens
```

You can run `p.ReBuild()` after `Import()` to add hydrogens to the structure. But understand that a new synthetic structure will be built, therefore you will lose the original occupancy and temperature-factor for each atom (replaces with 1.0 and 0.0).

**Building DNA/RNA:**
```python
p = Pose()
p.Build('ATGCGTACGTTCCGGCAGACGT', chain='A', fmt='DNA')
p.GetInfo()
```

**Importing a molecule:**
```python
m = Molecule()
m.Import('caffiene.sdf')
m.GetInfo()
```

**OpenMM plugin:**
```python
import io
import numpy as np
from sys import stdout
from pose import Pose
from openmm import *
from openmm.app import *
from openmm.unit import *

p = Pose()
p.Import('1YN3.pdb', chain='A')

pdb = PDBFile(io.StringIO(p.Export(fmt='PDB')))        # <--- The plugin
forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(forcefield)
modeller.addSolvent(forcefield, padding=1*nanometer)
pos = np.array(modeller.positions.value_in_unit(nanometer))
box = np.array(modeller.topology.getPeriodicBoxVectors().value_in_unit(nanometer))
shift = 0.5*(box[0]+box[1]+box[2]) - 0.5*(pos.min(axis=0)+pos.max(axis=0))
modeller.positions = [Vec3(*(p+shift)) for p in pos]*nanometer
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter('output.pdb', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
simulation.step(10000)
```

> Note: `from openmm.app import *` rebinds `ForceField` to OpenMM's class. If you mix this snippet with the other Quick Start examples (which start with `from pose import *`), make sure the OpenMM star imports come *after* `from pose import *` — the last star import wins, and this example needs OpenMM's `ForceField('amber99sb.xml', ...)`. If you want both classes accessible in the same session, use explicit imports instead of `*` (e.g. `from openmm.app import ForceField as OpenMMForceField`).

**RDKit plugin:**
```python
from rdkit import Chem

CFF = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
m = Chem.MolFromSmiles(CFF)
# Manipulate a molecule using RDKit here
molstr = Chem.MolToMolBlock(m)

m = Molecule()
m.Import(molstr)
m.GetInfo()
```

This RDKit plugin gives you the power and flexibility to manipulate molecules using RDKit and then import them to the Molecule() class when they are ready.

---

## API Reference

### Call Class

| Class            | Description |
|------------------|-------------|
| `p = Pose()`     | Calls the `Pose()` class for proteins, DNA, and RNA |
| `m = Molecule()` | Calls the `Molecule()` class for small organic molecules |

Each class have similar methods and data structure, but with slight differences in the way they are used.

### Building & I/O

| Method                                                     | Description |
|------------------------------------------------------------|-------------|
| `p.Import(filename='1YN3.pdb', chain=['A', 'B'], model=1)` | Imports a structure from a PDB or mmCIF file and constructs the `p.data` object. Can import a protein, DNA, or RNA structure. `chain` accepts a single chain ID (`'A'`), a list of chains (`['A', 'B']`), or `None` to import all chains. `model` selects which model to import from multi-model files (e.g. NMR ensembles); defaults to `1`. For atoms with multiple conformers, the highest-occupancy conformer is kept. Cannot import a structure that is a mixture of proteins and nucleic acids in separate chains, import each macromolecule type as a separate pose |
| `m.Import(filename='caffiene.sdf')`                        | Imports a structure from a PDB, SDF, mmCIF, MOL, or MOL2 files, or an RDKit block string and constructs the `m.data` object |
| `p.Export('out.pdb', fmt=None)`                            | Write the full structure, and all chains, to a PDB or mmCIF file. `fmt='PDB'` or `fmt='CIF'` will export the structure as a string and not a file (ideal to plug the structure to other libraries such as OpenMM) |
| `m.Export('out.sdf', fmt=None)`                            | Write the full structure to a PDB, SDF, mmCIF, MOL, or MOL2 file. `fmt='PDB'` or `fmt='CIF'` or `'SDF'`/`'MOL'`/`'MOL2'` will export the structure as a string and not a file (ideal to plug the structure to other libraries such as OpenMM) |
| `p.Build('MSLESNRGI', chain='A', fmt='protein')`           | Build a macromolecule from a one-letter sequence. For a polypeptide add the sequence and choose the format `fmt='Protein'`, uppercase = L-amino acids, lowercase = D-amino acids. For a nucleic acid add the sequence and choose the format `fmt='DNA'` or `fmt='RNA'`. You can add more chains by repeating the command with different chain `chain='A'` values. A structure can either be a protein, or a nucleic acid (DNA/RNA), it cannot be a mixture of the two |
| `p.ReBuild(sequence=None, mirror=False, _mutated=None)`    | Rebuild the polypeptide or nucleic acid. Use `sequence='AGLMTSWVLVA'` to rebuild the structure with multiple bulk mutations on chain A. Use `sequence={'A':'MSLKLSTVVA', 'B':'ASLKSWFWVA'}` to perform mutations at multiple chains at the same time. Use `mirror=True` to rebuild a protein and convert L-amino acids → D-amino acids and D-amino acids → L-amino acids. Will add missing hydrogens. For DNA and RNA, the `sequence=''` length must match exactly the original sequence length, otherwise an error will be raised |
| `p.Mutate(1, 'V', fast=True)`                              | Mutate a single monomer. For proteins: `p.Mutate(1, 'V')` = residue 1 → L-Valine, `p.Mutate(1, 'v')` = residue 1 → D-Valine. For DNA: `p.Mutate(0, 'T')` = nucleotide 0 → Thymine. For RNA: `p.Mutate(0, 'U')` = nucleotide 0 → Uracil. For double-stranded nucleic acids, the complementary base is also updated automatically. The `fast=True` argument means the mutation is performed by vector addition without ensuring the stability of the backbone (also the `CalcDSSP(), CalcSASA, and CalcRg()` etc.. are not re-computed) so these needs to be called after the mutation, in return the mutation is very fast, ideal for large mutation simulations. If `fast=False` the mutated residue is added to the structure and the entire structure rebuilt using ReBuild(), this is more accurate but very slow for large simulations |

### Measurements

| Method                                       | Description |
|----------------------------------------------|-------------|
| `p.GetDistance(0, 'N', 5, 'CA')`             | Get the distance in Å between any two atoms. Example: residue 0 nitrogen atom to residue 5 CA atom |
| `m.GetDistance(0, 5)`                        | Get the distance in Å between any two atoms. Example: atom 0 to atom 5 |
| `p.GetDihedral(2, 'PHI')`                    | Calculate the amino acid φ/ψ/ω/χ and nucleotide α/β/γ/δ/ε/ζ/χ dihedral angles. In this example we are measuring the PHI angle of the 3rd protein residue (index 2). For protein χ dihedral use `p.GetDihedral(4, 'chi', 1)` 5th residue (index 4), CHI 1 angle |
| `m.GetDihedral(0, 1, 2, 3)`                  | Calculate a dihedral angle between 4 atoms. In this example the dihedral angle is made up of the atoms at indeces 0, 1, 2, and 3 |
| `p.GetAngle(0, 'N', 5, 'CA', 17, 'C')`       | Get the angle between any three atoms in the whole structure. Example: N of residue 1, CA of residue 5, and C angle of residue 17, with the CA atom in the middle being the pivot |
| `m.GetAngle(0, 5, 17)`                       | Get the angle between any three atoms in the whole structure. Example: atom at index 1, atom at index 5, and atom at index 17, with atom at index 5 being the pivot |
| `p.GetAtomBonds(0, 1)`                       | Confirm and get the PDB name and element name `[atom 1 element name, atom 1 PDB name, atom 2 PDB name, atom 2 element name]` for two atoms (if they are bonded together). Use the atom indeces. If the two atoms are not bonded an error will be raised |
| `m.GetAtomBonds(1)`                          | Get all atom names bonded to this atom index ['atom name 1', 'atom name 2', 'atom name 3'] |
| `p.GetAtomCoord(3, 'N')`                     | Get the XYZ coordinates of an atom of a residue or a nucleotide (monomers). Example: `N` nitrogen of monomer index `3` |
| `m.GetAtomCoord(3)`                          | Get the XYZ coordinates of an atom given its index. Example: atom at index `3` |
| `p.GetAtomList(PDB=False)`                   | Get a list of all atom element names for the entire structure. Use `PDB=True` for PDB-formatted names |
| `m.GetAtomList()`                            | Get a list of all atom element names for the entire structure |
| `p.GetAtomHybridisation()`                   | Get a list of all atom hybridisations for the entire structure |
| `m.GetAtomHybridisation()`                   | Get a list of all atom hybridisations for the entire structure |
| `p.GetAtomIdx(3, 'N')`                       | Get the atom index in `p.data['Coordinates']` from its name within a monomer. This is the opposite of `p.GetAtomCoord(3, 'N')` |
| `p.GetIdentity(0, 'Atom')`                   | Identify the PDB name of an atom, or an amino acid, or a nucleotide by its index. Example `p.GetIdentity(5, 'Atom')` or `p.GetIdentity(5, 'amino acid')` or `p.GetIdentity(5, 'nucleotide')`. Also, specifically just for atoms, you are return its partial charge using `p.GetIdentity(3, 'Atom', charge=True)` |
| `p.GetInfo()`                                | Print a formatted summary of the structure's information |
| `m.GetInfo()`                                | Print a formatted summary of the structure's information and a graphical representation of the molecule |
| `m.CalcSMILES()`                             | Calculate the SMILES representation of a molecule and add it to `m.data['SMILES']` |
| `m.CalcSMARTS()`                             | Calculate the SMARTS representation of a molecule and add it to `m.data['SMARTS']` |
| `m.CalcSMIRKS()`                             | Calculate an atom-mapped SMIRKS-style string with hybridisation, connectivity, H-count, and formal-charge tags on each heavy atom. Atoms carry a 1-based atom-map index ':N' where N is the atom's index+1. Hybridization symbol follows SMARTS convention (^1, ^2, ^3 for sp/sp2/sp3; omitted for 's') |
| `p.CalcMass()`                               | Calculates the entire molecular mass of a molecule (all chains) in Da (Daltons), updates the value of p.data['Mass'] |
| `m.CalcMass()`                               | Calculates the entire molecular mass of a molecule |
| `p.CalcSize()`                               | Calculates the length of each chain in a structure, updates the value of p.data['Size']. You can get the length of each chain using `p.data['Size'][CHAIN]` |
| `p.CalcFASTA()`                              | Compiles the FASTA sequence of each chain, updates the value of p.data['FASTA']. You can get the FASTA sequence of each chain using `p.data['FASTA'][CHAIN]` |
| `p.CalcRg()`                                 | Calculates the entire Radius of Gyration of a molecule (all chains) in Å (angstrom), updates the value of p.data['Rg'] |
| `m.CalcRg()`                                 | Calculates the entire Radius of Gyration of a molecule |
| `p.CalcCharge(iterations=6)`                 | Calculate the Gasteiger-Marsili partial charges to all atoms using iterative equalization (default 6 iterations), updates the value of `p.data['Atoms'][index][2]` |
| `m.CalcCharge(iterations=6)`                 | Calculate the Gasteiger-Marsili partial charges to all atoms using iterative equalization (default 6 iterations), updates the value of `m.data['Atoms'][index][2]` |
| `p.CalcDSSP()`                               | Calculates each amino acid's secondary structure assignments, only for proteins, and stores them in `p.data['Amino Acids'][i][4]` and updates `p.data['SS'][CHAIN]`, therefore this is where you can get the SS sequence of each chain. Codes: H=α-helix, G=3₁₀-helix, I=π-helix, E=β-sheet, B=β-bridge, T=turn, S=bend, L=loop, P=PPII-helix |
| `p.CalcSASA(n_points=960, probe_radius=1.4)` | Calculates the Solvent Accessible Surface Area (SASA) for each amino acid, only for proteins, using golden sphere sampling. `n_points` controls sampling density, `probe_radius` is the solvent probe radius in Å (default 1.4 for water). Adds the value to `p.data['Amino Acids'][i][6]` |

### Manipulation

| Method                                                   | Description |
|----------------------------------------------------------|-------------|
| `p.AdjustDistance(0, 'N', 4, 'C', 17)`                   | Set the distance between any two atoms in (Å). Example: set the distance between N in residue 0 and C in residue 4 to 17 Å. Order matters: the second atom (and all atoms downstream of it on the same chain) moves, while the first atom stays fixed. `(0, 'N', 0, 'CA', d)` ≠ `(0, 'CA', 0, 'N', d)` |
| `m.AdjustDistance(0, 4, 17)`                             | Set the distance between any two atoms in (Å). Example: set the distance between atom at index 0 and atom at index 4 to 17 Å. Order matters: the second atom (and all atoms downstream of it) moves, while the first atom stays fixed. `(0, 1, d)` ≠ `(1, 0, d)` |
| `p.AdjustAngle(1, 'N', 1, 'CA', 1, 'C', -2)`             | Add/subtract degrees from a three-atom angle, with the middle atom being the pivot point. Example: subtract 2° from N–CA–C angle of residue 1, with the CA atom being the pivot |
| `m.AdjustAngle(0, 1, 2, -2)`                             | Add/subtract degrees from a three-atom angle, with the middle atom being the pivot point. Example: subtract 2° from the angle represented by atom 0, atom 1, and atom 2, with atom 1 being the pivot |
| `p.RotateDihedral(1, -60, 'PHI')`                        | Rotate the amino acid φ/ψ/ω/χ and nucleotide α/β/γ/δ/ε/ζ/χ dihedral angles. Example: residue 1 PHI dihedral to -60° |
| `m.RotateDihedral(0, 1, 2, 3, -60)`                      | Rotate any dihedral angle represented by four atoms. Example: rotate a dihedral angle represented by atom index 0, atom index 1, atom index 2, and atoms index 3 to become -60° |
| `p.MovePose(theta=5, u=[18, 10, 5], l=6, ori=[0, 0, 0])` | Rotate and/or translate the whole structure. `theta` = rotation angle in degrees, `u` = rotation axis vector (will be normalised), `l` = translation distance in Å, `ori` = target point to translate towards. All parameters are optional (default `None`); you can rotate only, translate only, or both |
| `m.MovePose(theta=5, u=[18, 10, 5], l=6, ori=[0, 0, 0])` | Rotate and/or translate the whole structure. `theta` = rotation angle in degrees, `u` = rotation axis vector (will be normalised), `l` = translation distance in Å, `ori` = target point to translate towards. All parameters are optional (default `None`); you can rotate only, translate only, or both |

### Force Field &  Energy Score

The `ForceField()` class evaluates the total potential energy and analytical per-atom forces of a `Pose`, summing six terms (Molecule poses, via `MOL_TERMS`) or nine terms (`Pose` proteins / DNA / RNA, via `DEFAULT_TERMS`) — dispatched automatically by `pose.data['Type']` — with a hash-cached topology so that repeated calls during minimisation, annealing, or MD only recompute the coordinate-dependent quantities.

| Method                                                                          | Description |
|---------------------------------------------------------------------------------|-------------|
| `ff = ForceField(terms=None)`                                                   | Build a force field. `terms` is an optional list of `(method_name, kwargs)` tuples that selects which energy terms to sum and which algorithm variants to use. With `terms=None` for the default setup. Force-field parameters are loaded once from `database.json['Energy Parameters']` via the cached `DBLoad()` module-level function |
| `E = ff(pose, grad=False, box=None)`                                            | Evaluate total potential energy in kJ/mol. `box=None` disables PBC; pass a `(3,)` array for an orthorhombic box or a `(3, 3)` array for a triclinic box, in Å |
| `E, F = ff(pose, grad=True, box=None)`                                          | Evaluate total potential energy plus per-atom forces. Returns a tuple `(float, ndarray)` where forces are shape `(N, 3)` in kJ/mol/Å |
| `ff.BondPotential(pose, cache, alg='harmonic', grad=True, box=None)`            | Bond-stretching term. `alg='harmonic'` uses `Σ K_b·(r − r₀)²`, `alg='morse'` uses `Σ D_e·(1 − e^(−a(r − r₀)))²` |
| `ff.AnglePotential(pose, cache, grad=True, box=None)`                           | Harmonic three-atom angle term, `Σ K_θ·(θ − θ₀)²` over every bonded triplet |
| `ff.UBPotential(pose, cache, grad=True, box=None)`                              | Urey-Bradley 1-3 stretching term, `Σ K_UB·(s − s₀)²` between the outer atoms of every bonded triplet |
| `ff.ProperTorsionPotential(pose, cache, grad=True, box=None)`                   | Proper dihedral (torsion) term, multi-component Fourier `Σ k_φ·(1 + cos(n·φ − φ₀))` over every i-j-k-l quartet |
| `ff.ImproperTorsionPotential(pose, cache, alg='harmonic', grad=True, box=None)` | Improper dihedral term over degree-3 atoms. `alg='harmonic'` uses `Σ k·(ψ − ψ₀)²`, `alg='fourier'` uses `Σ k·(1 + cos(n·ψ − ψ₀))` |
| `ff.VDWPotential(pose, cache, alg='12-6', grad=True, box=None)`                 | Van der Waals (Lennard-Jones) non-bonded term, with 1-4 scaling masks. `alg='12-6'` is the standard form, `alg='9-6'` is a softer variant |
| `ff.ElectrostaticPotential(pose, cache, alg='constant', grad=True, box=None)`   | Electrostatic non-bonded term. `alg='constant'` uses uniform εᵣ; `alg='ddd'` uses a distance-dependent dielectric `ε(r) = εᵣ·r` |
| `ff.PolarisationPotential(pose, cache, alg='constant', grad=True, box=None)`    | Induced-dipole polarisation term, `−½·Σ α_i·\|E_i\|²` with the per-atom field built from neighbour charges |
| `ff.CMAPPotential(pose, cache, grad=True, box=None)`                            | CMAP backbone (φ, ψ) cross-term correction over every interior protein residue, evaluated by bicubic Catmull-Rom interpolation on the per-residue 24×24 energy grids in `database.json['Energy Parameters']['cmap']` |

> **Charge model & parameter assignment**: For `Molecule` poses, partial charges are computed by `ForceField.NAGLCharges(pose)` — a NumPy reimplementation of the [`openff-gnn-am1bcc-1.0.0`](https://github.com/openforcefield/openff-nagl-models) graph neural network released by the [Open Force Field Initiative](https://github.com/openforcefield). Output charges are bit-equivalent to upstream NAGL float32 inference, with the total constrained to the molecule's formal charge via electronegativity equalisation. Bonded and vdW parameters are assigned by `pose.energy.SMIRKSMatch(pose, params)`, a pure-NumPy SMIRKS pattern matcher that applies [Sage 2.3.0](https://github.com/openforcefield/openff-forcefields) patterns to the pose's bond graph. All energy parameters in `database.json` are stored in **kJ/mol** (lengths in Å, angles in degrees).

> **Protein / DNA / RNA support**: `ForceField()` runs on Pose protein, DNA, and RNA poses with **100% SMIRKS coverage** across all 26 supported amino acids and the 5 canonical nucleobases. Coverage is achieved by (1) a real Kekulisation algorithm inside `SMIRKSMatch` (constraint-propagation + DFS backtracking for resonance bonds), (2) Sage 2.3.0's existing patterns for the chemistry it covers, and (3) **21 biomolecule-specific SMIRKS additions, all 21 now QM-fit**: 20 via direct B3LYP/cc-pVTZ scans (cc-pVTZ-PP + ECP for selenium) using [PySCF 2.13](https://github.com/pyscf/pyscf) (Apache 2.0) — bonds and angles via subgroup-rigid scan + parabolic fit on 13 model molecules, proper torsions via 24-step relaxed dihedral scan + Fourier fit, the phosphate improper via out-of-plane scan + harmonic fit; and 1 (the `[#34:1]` selenium vdW) via SAPT0/def2-TZVP dimer scan on H2Se using [Psi4 1.10](https://github.com/psi4/psi4) (LGPL-3.0, used as a tool — its dispersion + exchange components are extracted to fit the LJ form). Energies for proteins / DNA / RNA are *not* yet validated against an external reference and should be treated as experimental. `ForceField(strict=True)` raises `RuntimeError` on any uncovered fragment for users who want explicit failure on coverage gaps.

> ⚠️ **`Score()` currently raises `NotImplementedError`** — the database keys it consumed (`weights`, `ref_state`, `lk_solvation`, `hbond`, `kbp`, `rotamer`) were removed during the recent force-field refactor. Score is pending migration to SMIRKS-driven score terms. The math, vectorisation, and chirality logic in this section are correct and preserved as developer reference for the re-implementation.

The `Score()` class is a hybrid physics + statistical energy function for protein design. It sums eight terms (three reused from `ForceField`, five added) and returns a single design score in kJ/mol-equivalent units. **Score is chirality-aware from line zero** — L-amino acids, D-amino acids, mixed L/D sequences, and non-canonical residues all score correctly with no extra arguments or special-cased call sites.

> The Score implementation below describes the **target design** for the score function. Once the missing `weights`, `ref_state`, `lk_solvation`, `hbond`, `kbp`, and `rotamer` blocks are re-introduced into `database.json` (in the SMIRKS-driven schema), `Score()` will be re-enabled. Until then, instantiating it raises `NotImplementedError`. The methods (`_lk_solvation`, `_hbond_geom`, `_kbp_score`, `_rotamer_prior`, `_reference_state`) still exist in the source for reference.

| Method                                    | Description |
|-------------------------------------------|-------------|
| `sc = Score(ff=None, box=None)`           | Build a design scorer. `ff` is a reusable `ForceField` instance used for the three physics terms (LJ, Electrostatic, CMAP) and the topology cache; created internally if `None`. `box` is the optional PBC box: `None` disables PBC, `(3,)` orthorhombic, `(3, 3)` triclinic, in Å. The 8 term weights, LK Lazaridis-Karplus parameters, H-bond geometry parameters, KBP table, and per-aa reference-state values are all read once from `database.json['Energy Parameters']`; the per-(residue, rotamer, χ) means and σ values consumed by `_rotamer_prior` are read from `database.json['Rotamer Library']` (no global rotamer σ — every rotamer carries its own σ) |
| `total = sc(pose, decompose=False)`       | Evaluate the weighted total score in kJ/mol. Topology and atom-type caches are built on the first call and reused on subsequent calls until the pose's `Atoms` / `Bonds` / `Amino Acids` records change |
| `total, terms = sc(pose, decompose=True)` | Same as above but also return a per-term dict with keys `'LJ'`, `'Electrostatic'`, `'LK'`, `'Hbond'`, `'CMAP'`, `'Rotamer'`, `'Reference'`, `'KBP'`. Useful for term-weight fitting and for diagnosing why a design scores poorly |

The eight terms (three reused from `ForceField`, five added). LJ, Electrostatic, and CMAP are evaluated by reusing the corresponding `ForceField` methods; the other five live on `Score()`:

| Term            | Source                       | Form |
|-----------------|------------------------------|------|
| `LJ`            | `ff.VDWPotential`            | Van der Waals (Lennard-Jones) with 1-4 scaling |
| `Electrostatic` | `ff.ElectrostaticPotential`  | Coulomb with `ε_r` or distance-dependent dielectric |
| `CMAP`          | `ff.CMAPPotential`           | Per-residue (φ, ψ) backbone correction grid; D-AA grids are mirrored from L-AA grids in `_compile` automatically |
| `LK`            | Added                        | Lazaridis-Karplus EEF1 implicit solvation. Per-atom (ΔG_free, λ, V) parameters from `database.json['Energy Parameters']['lk_solvation']` |
| `Hbond`         | Added                        | Kortemme-Baker geometric H-bond term, `E_r(r_HA) · F(θ_DHA) · F(θ_HAB)` over every donor-H/acceptor-base quartet detected from the bond graph (donor = N or O bonded to ≥1 H; acceptor = N or O with ≥1 heavy neighbour) |
| `Rotamer`       | Added                        | Multimodal rotamer prior. For each protein residue with χ angles, the residue's backbone (φ, ψ) is snapped to the nearest 10° cell of the Rotamer Library. The cell's full set of rotamer mixture components `{P_k, μ_k_χ_c, σ_k_χ_c}` is read, and the per-residue energy is the mixture-of-Gaussians log-likelihood `E_r = −kT · log Σ_k P_k(φ,ψ) · ∏_c (1/(√(2π)·σ_kc)) · exp(−½·(Δχ_c/σ_kc)²)`, evaluated stably via logsumexp. Per-rotamer σ values come from the library (no global σ); `kT = 2.494` kJ/mol (RT at 300 K). For D-AAs the cell is fetched at (−φ, −ψ) and library μ values are negated, recovering the mirror-symmetric mixture |
| `Reference`     | Added                        | Per-aa unfolded-state baseline `E_ref[aa]` summed over the sequence. Required for fair sequence-to-sequence comparison |
| `KBP`           | Added                        | Knowledge-based pair potential, DFIRE-style. Sums `KBP[type_i, type_j, distance_bin]` over every long-range atom pair (`mask_far` in the cache) |

> `Score()` is the single API surface in this library that is chirality-aware end to end — it produces correct, comparable design scores for D-amino acids, mixed L/D sequences, mirror-image proteins, and non-canonical residues without any extra arguments. ML-based ranking proxies (AlphaFold pLDDT, ProteinMPNN log-likelihoods) do not work in this regime because they were trained on canonical-L PDB data only.

> Numerical parameters for the four added terms (`lk_solvation`, `hbond`, `kbp`, `ref_state`) and the term `weights` are **not currently shipped** in `database.json` — they were removed during the recent force-field refactor and are pending re-fitting against a curated PDB-derived dataset. The math, vectorisation, and chirality logic are production-quality and preserved in source. The Rotamer Library values themselves are derived from Dunbrack BBDEP2010 (CC-BY-4.0) and are real and current.

### Tools

These are standalone tools (not Pose() class methods) and thus are called on their own:

| Function                                                           | Description |
|--------------------------------------------------------------------|-------------|
| `Parameterise('PTR.cif', 'ptr_rot.json', 'PTR', 'B', backup=True)` | Add a non-canonical amino acid to the unified `database.json`. Takes the RCSB CCD `.cif` file, a Dunbrack BBDEP2010-format rotamer-library JSON (e.g. produced by `nnca_pipeline/scripts/build_*_rotamer.py`), the three-letter tricode, and a single-letter unicode for the `Amino Acids` slot. Inserts the residue into both `Amino Acids[unicode]` (atoms / bonds / hybridisation inferred from the CIF) and `Rotamer Library["residues"][tricode]` (chi means, sigmas, populations from the JSON) in one atomic write. `backup=True` (default) timestamps a `database.json.bak.<YYYYMMDD-HHMMSS>` before modifying. Chi axes are taken from `rot_entry["method"]["chi_axes"]` (the user-confirmed source of truth); the CIF walker's chi tracing is no longer used. Calls `DBLoad.cache_clear()` on success so subsequently constructed Pose / ForceField / Score / Rotamers instances see the new residue immediately. The rotamer JSON file can be generated using [this repo](https://github.com/sarisabban/ncaarotamers) |
| `RMSD(pose1, pose2, alg='align', export='aligned.pdb')`            | Computes the Root Mean Squared Deviation between two protein or nucleic acids `Pose` structures using Cα (alpha-carbon) atoms for proteins, or C1 atoms for nulceic acids. Returns the RMSD in (Å). Supported algorithms: `'align'` (sequence alignment + iterative Kabsch), `'kabsch'` (SVD-based optimal rotation), `'quaternion'` (eigenvalue-based optimal rotation), or `'simple'` (translation only, no rotation). Can export the aligned structures to `aligned_1.pdb, aligned_2.pdb` |
| `BLAST(sequence1, sequence2)`                                      | Perform pairwise protein or nucleic acid sequence alignment using the Smith-Waterman local alignment algorithm with BLOSUM62 substitution scores, matching the statistical model used by NCBI BLASTP. Returns: `(alignment_string, percent_identity, e_value)` |
| `MSA([sequence1, sequence2, sequence3....])`                       | Aligns three or more protein or nucleic acid sequences using a ClustalW-like progressive alignment strategy, pairwise distances are computed with `BLAST()`. Returns: `(alignment_string, aligned_list, conservation_list, entropy_list, pssm_array, dca_array)` where `conservation_list` is a per-column score in [0, 1] (1 = fully conserved), `entropy_list` is per-column Shannon entropy in bits, `pssm_array` is a `(L, 20)` log-odds matrix in BLOSUM62 column order (`ARNDCQEGHILKMFPSTWYV`), and `dca_array` is an `(L, L)` APC-corrected mean-field DCA direct-information matrix |
| `Isoelectric(sequence)`                                            | Calculates the protein's isoelectric point (pI) using the EMBOSS pKa scale and bisection on `[0, 14]`. Takes a protein sequence and returns a float, the pH at which the protein has zero net charge |
| `Hydrophobicity(sequence, window=9, scale='eisenberg')`            | Calculates the hydrophobicity profile from a protein sequence using a sliding window. Supported scales: `'eisenberg'` (default, normalized consensus), `'kyte-doolittle'`, `'hopp-woods'`, `'engelman'`. Returns a tuple of two lists `(positions, scores)` where `positions` are zero-based indices of the window centers — these lists are used to plot the graph |
| `Aliphatic(sequence)`                                              | Calculates the Aliphatic index of a protein from its sequence (Ikai 1980: `AI = X(A) + 2.9·X(V) + 3.9·(X(I) + X(L))`), returns a float value |
| `ExtinctCoeff(sequence, reduced=True)`                             | Calculates the molar extinction coefficient at 280 nm in water (Pace 1995: `ε = nW·5500 + nY·1490 + (nC/2)·125`). With `reduced=True` (default) cysteines are treated as reduced and contribute 0; with `reduced=False` cysteines are treated as cystines and contribute `(nC // 2) · 125`. Returns an int value in M⁻¹ cm⁻¹ |
| `Instability(sequence)`                                            | Calculates the Instability index of a protein (Guruprasad et al. 1990) using the DIWV dipeptide weight table. Returns a float; values below 40 generally indicate a stable protein |
| `GRAVY(sequence)`                                                  | Calculates the Grand Average of Hydropathy using the Kyte-Doolittle hydropathy scale, returns a float value |
| `Split(pose, chain=None, start=None, end=None)`                    | Slice a Pose into a new Pose object. Takes the original `pose`, the `chain` if you want to split out an entire chain, or `start, end` if you want to split out a range of monomer residues (zero-based, inclusive). Works for proteins, DNA, and RNA. Atom and residue indices, the bond graph, and coordinates are all renumbered densely from zero in the returned pose |
| `Concatenate(pose1, pose2, fuse=False)`                            | Combine two poses of the same Type. With `fuse=False` (default) `pose2` is appended to `pose1` as additional chains, preserving the original coordinates of both poses; chain IDs in `pose2` that collide with `pose1` are renamed to the next free letter. With `fuse=True` the concatenated FASTA is rebuilt as a single continuous polymer with idealised geometry, the original input coordinates are discarded |
| `PCR(sequence)`                                                    | Generates forward and reverse PCR primers for a DNA template (DNA only, accepts only A/C/G/T, template must be ≥ 36 bp). Uses a 5-tier relaxation strategy so that any chemically valid template always returns a primer pair. **Ideal** tier requires length 18–25, GC 40–60%, nearest-neighbor SantaLucia 1998 Tm in `[55, 65]` °C, a 3' GC clamp, no run of 4 identical bases, no internal palindrome (hairpin), no 3' self-dimer, and &#124;ΔTm&#124; ≤ 2 °C. If no pair satisfies it the search falls through progressively relaxed **Good** / **Fair** / **Poor** / **Last resort** tiers, each widening the length / GC / Tm / ΔTm bounds and dropping the GC clamp / hairpin / dimer gates. When the result comes from any tier below Ideal, a warning is printed to stdout naming the tier and which gates were relaxed (e.g. `Warning: PCR primers are suboptimal (Poor tier) — GC% outside 40-60; Tm outside 55-65 °C; GC clamp missing`). Returns a tuple `(forward_string, reverse_string, warning_message_for_suboptimal_primers)` |
| `Translate(sequence, fmt='protein', organism='ecoli')`             | Translates between protein, DNA, and RNA. The input alphabet is auto-detected. Takes a sequence and translates it to the requested `fmt` format. Nucleotide → protein translation uses the standard genetic code and returns `*` for stop codons. Protein → DNA/RNA back-translation is codon-optimised by selecting the highest-frequency codon (deterministic) for the chosen `organism`, which takes `'ecoli'` (default) or `'human'`. Returns the translated sequence as an uppercase string |
| `PROSITE(sequence, pattern)`                                       | Search a protein sequence for a PROSITE-style pattern. Pattern grammar: `[ABC]` = any of A/B/C, `{ABC}` = any except A/B/C, `x` = any residue, `x(n)` / `x(n,m)` = quantifiers, `A(n)` / `A(n,m)` = repeat literal residues, `<` / `>` = anchor at sequence start/end, `-` = token separator (stripped). Returns a list of tuples `[(start, end, match), ...]` with 1-based, inclusive positions |
| `HydrogenBondMap(pose)`                                            | Generates a backbone hydrogen-bond donor/acceptor map for a protein pose (proteins only). Uses the same DSSP electrostatic criterion as `p.CalcDSSP()` (Kabsch & Sander 1983: `E < -2.092` kJ/mol). Returns an array of shape `(N_atoms, N_atoms)` where 0 = no bond, 1 = this atom is a donor (backbone N), 2 = this atom is an acceptor (backbone O) |
| `ContactMap(pose)`                                                 | Generates a monomer-monomer distance map in angstroms. The molecule type is auto-detected from `pose.data['Type']`: distances between protein residues are calculated from the Cα atoms, while distances between DNA and RNA bases are calculated from their C1' atoms. Returns an array of shape `(N_residues, N_residues)` with zero on the diagonal |
| `Rotamers(10, pose)`                                               | Single-amino-acid rotamer packer: snap the residue's backbone (φ, ψ) to the nearest 10° cell of `database.json['Rotamer Library']`, pick the rotamer k\* with the largest `P_k` in that cell, and apply its mean χ values to every χ of the residue via `pose.RotateDihedral`. No-op (silent) for residues with no χ atoms (Gly, Ala), residues at chain ends with undefined backbone, and non-canonical residues missing from the library. Handles D-amino acids automatically via lookup at (−φ, −ψ) and μ negation. Derived from the Dunbrack BBDEP2010 rotamer library (CC-BY-4.0) |
| `Minimise(pose, ff=None, max_steps=500, ftol=1.0, dt_fs=0.1, dt_max_fs=2.0, step_max=0.2, etol=1e-6, stall_k=10, box=None)`                                                                           | Relax pose coordinates using the FIRE2 algorithm (Guénolé et al. 2020) with a trust-region step limiter that bounds per-atom displacement to `step_max` Å. Mutates `pose.data['Coordinates']` in place. `ftol` is the convergence threshold on max\|force\| in kJ/mol/Å; `dt_fs` is the initial integration step in fs and `dt_max_fs` the adaptive ceiling; `etol` and `stall_k` trigger early stop after K consecutive stalled energy steps. Returns `(final_E, log)` where `log` carries `'energies'`, `'fmax'`, `'max_step'`, `'converged'`, `'n_steps'` |
| `Anneal(pose, ff=None, n_steps=10000, T_start=2000.0, T_end=10.0, sigma_small=5.0, sigma_large=30.0, p_large=0.2, p_shear=0.5, target_acc=0.30, adapt_window=100, seed=None, box=None)`               | Simulated annealing over backbone φ/ψ with two Metropolis move types — single-angle (random φ or ψ) and shear (compensating ψᵢ +Δ / φᵢ₊₁ −Δ that leaves residues 0..i−1 unmoved). Each step picks a small (adaptive `sigma_small`) or large (fixed `sigma_large`) Gaussian perturbation; `sigma_small` is updated by Robbins-Monro every `adapt_window` small moves to track `target_acc` ~ 0.30. Geometric cooling from `T_start` to `T_end`. Returns `(E_best, log)` with `'energies'`, `'temperatures'`, `'accepted'`, `'move_types'` (0=single, 1=shear, 2=invalid), `'sigma_history'`, `'best_step'`. The pose is left at the lowest-energy frame |
| `Pack(pose, score=None, ff=None, n_steps=2000, T_start=10.0, T_end=0.1, patience=400, seed=None, box=None)` | Sidechain repacking via simulated annealing over the **full Rotamer Library ensemble** at each residue's current backbone (φ, ψ). At construction the candidate set per repackable residue is built once from `database.json['Rotamer Library']` (the full list of (μ_χ tuple, P_k) entries at that residue's grid cell — this can be 3 rotamers for Val, up to ~80 for Lys/Arg). The SA loop picks a random repackable residue, samples one of its rotamers k weighted by `P_k` (so dominant rotamers are explored more often but rare ones remain reachable), applies the trial χ tuple, rescores, and accepts via Metropolis: `dE ≤ 0` or `random() < exp(−dE/T)`. Geometric cooling from `T_start` to `T_end`. Tracks the best-scoring configuration seen and restores it before returning. Early-exit if no acceptance occurs in `patience` consecutive steps. `score` is a reusable `Score` instance; if `None`, one is built from `ff` (or a fresh `ForceField` if `ff` is also `None`). Using `Score` rather than the bare force field matters because the statistical terms (rotamer prior, KBP, reference state) discriminate native-like rotamer choices in a way pure-physics forces cannot. D-amino acids handled automatically. Returns `(E_final, log)` where `log` carries `'energies'`, `'temperatures'`, `'accepts'` (bool array of accept/reject per step), `'best_E'`, `'steps_run'`, `'converged'` (True if early-exited via stagnation), `'n_residues'` (count of repackable residues) |
| `MolecularDynamics(pose, ff=None, n_steps=1000, dt_fs=2.0, T=300.0, thermostat='nve', friction_ps=1.0, constraints='hbonds', shake_tol=1e-8, shake_max=100, seed=None, trajectory_every=0, box=None)` | Velocity-Verlet NVE or BAOAB Langevin NVT integration. Initial velocities are sampled from Maxwell-Boltzmann at `T` with the centre-of-mass momentum zeroed and projected onto the constraint manifold. `thermostat='nve'` runs energy-conserving dynamics; `thermostat='langevin'` runs the BAOAB stochastic splitting at temperature `T` with friction `friction_ps` ps⁻¹. `constraints='hbonds'` enables vectorised SHAKE/RATTLE on every X–H bond (target lengths read from `database.json['Energy Parameters']`), making `dt_fs=2.0` stable; `constraints='none'` disables them. `trajectory_every=k` saves a coordinate snapshot every k steps. Returns `(final_E, log)` with `'energies'`, `'kinetic'`, `'temperatures'`, `'frames'`, `'n_constraints'`, `'dof'` |

> BLAST handles sequences beyond the 20 canonical L-amino acids automatically: **D-amino acids**: stored as lowercase letters in `pose.data['FASTA']`. BLAST uppercases both sequences before alignment, treating each D-amino acid as its L-counterpart for scoring purposes. This correctly reflects the chemical reality that D- and L-forms of the same residue have identical side-chain chemistry. **Non-canonical amino acids**: any letter not in the 20-letter BLOSUM62 alphabet falls back to: `+4` for a self-match (equal to the minimum BLOSUM62 diagonal), `−1` for a mismatch. This keeps non-canonical residues visible to the aligner without inflating scores.

> MSA handles sequences beyond the 20 canonical L-amino acids, identical to `BLAST()`

For Parameterise() this is the workflow:

1. Download the CIF file for the amino acid from the [RCSB Chemical Component Dictionary](https://www.rcsb.org/ligand/) (e.g. `https://files.rcsb.org/ligands/download/PTR.cif`).
2. Produce a backbone-dependent rotamer library JSON in Dunbrack BBDEP2010 schema. The `nnca_pipeline/scripts/build_*_rotamer.py` scripts at the project root generate one from PDB-mining + adaptive KDE + BGMM (THGLab `ptm_sc` methodology, MIT-licensed). The output JSON must carry `method.chi_axes` (atom-name 4-tuples) — Parameterise reads this as the source of truth for the `Chi Angle Atoms` field.
3. Call `Parameterise(cif_file, rotamer_json_file, tricode, unicode)`. A timestamped backup of `database.json` is created automatically; pass `backup=False` to opt out.

---

## Key Concepts

### Zero-based indexing

All residue and atom indices start at 0, not 1. Residue 0 is the N-terminal amino acid. This is the **opposite** of PDB convention.

```python
p.Build('MSLESNRGI', chain='A', fmt='protein') # Construct a polypeptide
p.GetDihedral(0, 'PHI')                        # PHI of first residue (index 0)
p.GetDihedral(2, 'chi', 1)                     # CHI 1 of third residue (index 2)
p.GetDistance(0, 'N', 1, 'CA')                 # N of residue 0 to CA of residue 1
p.Build('MSLESNRGI', chain='B', fmt='protein') # Add a second chain
```

### Accessing the data structure directly

```python
p.data['FASTA']              # Sequence string
p.data['Size']               # Number of residues (int)
p.data['Amino Acids'][0]     # [letter, chain, bb_indices, sc_indices, secondary structure, tricode, SASA]
p.data['Atoms'][0]           # [pdb_name, element, charge, occupancy, temp_factor, hybridisation]
p.data['Coordinates']        # Numpy array, shape (N, 3)
p.data['Bonds']              # Adjacency list: {atom_index: [bonded_atom_indices]}
```

Iterating over residues and atoms:
```python
for idx, aa in p.data['Amino Acids'].items():
    symbol, chain, bb, sc, ss, tricode, sasa = aa
    print(f'Residue {idx}: {tricode} ({symbol}), SS={ss}')

for idx, atom in p.data['Atoms'].items():
    name, element, charge, occupancy, temp, hybrid = atom
    xyz = p.data['Coordinates'][idx]
    print(f'Atom {idx}: {name} ({element}) at {xyz}')
```

---

## Supported Amino Acids

> Uppercase = L-form, lowercase = D-form. All 26 are supported in mixed L/D sequences. Additional amino acids can be added to the **database.json** file.

> The N-terminus is protonated, as expected at physiological pH (~7.4), and therefore exists as a positively charged ammonium group (–NH<sub>3</sub><sup>+</sup>)

|       |       |       |       |       |
|-------|-------|-------|-------|-------|
|A - ALA|B - ALY|C - CYS|D - ASP|E - GLU|
|F - PHE|G - GLY|H - HIS|I - ILE|J - MSE|
|K - LYS|L - LEU|M - MET|N - ASN|O - TPO|
|P - PRO|Q - GLN|R - ARG|S - SER|T - THR|
|U - SEC|V - VAL|W - TRP|X - TRF|Y - TYR|
|Z - PTR|

## Supported Nucleotides

### DNA

|       |       |       |       |
|-------|-------|-------|-------|
|A - DA |T - DT |C - DC |G - DG |

### RNA

|      |      |      |      |
|------|------|------|------|
|A - A |U - U |C - C |G - G |

---

## Data Structure Reference

Get the content of the structure's JSON object using `print(p.data[KEY])`

This `p.data` structure from the `Pose()` class represents proteins, DNA, and RNA:

| Key           | Value Type  | Description |
|---------------|-------------|-------------|
| `Type`        | String      | Identifies the structure as a protein, DNA, or RNA |
| `Energy`      | Float       | Potential energy of the molecule |
| `Rg`          | Float       | Radius of gyration |
| `Mass`        | Float       | Mass in Daltons |
| `Size`        | Dict        | Length of each chain, ie: the number of monomers for each chain |
| `FASTA`       | Dict        | One-letter sequence for each chain |
| `SS`          | Dict        | One-letter amino acid secondary structure asignments for each chain |
| `Nucleotides` | Dict        | `{index: [symbol, chain, bb_atom_indices, sc_atom_indices, tricode]}`, **zero-based indexing** |
| `Amino Acids` | Dict        | `{index: [symbol, chain, bb_atom_indices, sc_atom_indices, secondary_struct, tricode, SASA]}`, **zero-based indexing** |
| `Atoms`       | Dict        | `{atom_index: [pdb_name, element, partial charge, occupancy, temp_factor, hybridisation]}`, **zero-based indexing** |
| `Bonds`       | Dict        | Bond graph as adjacency list: `{atom_index: [bonded_atom_indices]}` |
| `BondOrders`  | Dict        | Bond order as an adjacency list, 1 = single bonds, 1.5 = aromatic resonance partial-double bond, 2 = double bonds, 3 = triple bonds |
| `Coordinates` | NumPy array | Shape `(N, 3)`, Cartesian XYZ for each atom |

This `m.data` structure from the `Molecule()` class represents small organic molecules:

| Key           | Value Type  | Description |
|---------------|-------------|-------------|
| `Type`        | String      | Identifies the structure as a molecule |
| `Energy`      | Float       | Potential energy of the molecule |
| `Rg`          | Float       | Radius of gyration |
| `Mass`        | Float       | Molecule's molecular mass |
| `SMILES`      | Str         | The SMILES representation of the molecule as a string |
| `SMARTS`      | Str         | The SMARTS representation of the molecile as a string |
| `SMIRKS`      | Str         | An atom-mapped SMIRKS-style string that contain hybridisation, formal-charge, and other info about the molecule |
| 'Formula'     | Str         | The molecular formula of the molecule |
| `Atoms`       | Dict        | `{atom_index: [pdb_name, element, partial charge, hybridisation]}`, **zero-based indexing** |
| `Bonds`       | Dict        | Bond graph as adjacency list: `{atom_index: [bonded_atom_indices]}` |
| `BondOrders`  | Dict        | Bond order as an adjacency list, 1 = single bonds, 1.5 = aromatic resonance partial-double bond, 2 = double bonds, 3 = triple bonds |
| `Coordinates` | NumPy array | Shape `(N, 3)`, Cartesian XYZ for each atom |

> The `hybridisation` field is one of `'s'` (hydrogens), `'sp'`, `'sp2'`, or `'sp3'`.

> Each atom's hybridisation is stored as the last element of its atom record: `p.data['Atoms'][i][5]` for `Pose()`, `m.data['Atoms'][i][3]` for `Molecule()`.

---

## database.json overview

Pose ships a single `database.json` file (~63 MB) under `pose/` with **four** top-level keys:

| Top-level key      | Purpose |
|--------------------|---------|
| `Amino Acids`      | Per-residue topology templates: backbone & sidechain atoms, vectors, bonds, χ angle atoms. Used by `Pose.Build`, `Pose.Mutate`, all dihedral routines |
| `Nucleotides`      | Per-nucleotide topology templates for DNA and RNA |
| `Rotamer Library`  | Backbone-dependent rotamer mixture data (Dunbrack BBDEP2010 derived) — multimodal `{P_k, μ_k_χ_c, σ_k_χ_c}` per residue type per 10° (φ, ψ) cell. Consumed by `Score._rotamer_prior`, `tools.Rotamers`, and `tools.Pack` |
| `Energy Parameters`| All numerical force-field parameters. SMIRKS-keyed bonded + nonbonded parameters from OpenFF Sage 2.3.0 (CC-BY-4.0) plus 21 biomolecule extensions (20 QM-fit via PySCF B3LYP/cc-pVTZ scans; 1 selenium vdW QM-fit via Psi4 SAPT0/def2-TZVP dimer scan on H2Se). Lives under `Bonds`, `Angles`, `UB` (Urey-Bradley, CHARMM-style narrow SMIRKS, separate from `Angles`), `ProperTorsions`, `ImproperTorsions`, `vdW`, `Electrostatic`, `Constraints`, `Constants`. NAGL AM1-BCC neural-network weights live under `AM1BCC` as base64-encoded float32 tensors. CMAP placeholder grids under `cmap`. All energy values stored in kJ/mol. See "Description of energy parameters in database.json" below for the full schema and provenance |

The whole file is loaded once per Python process via the cached module-level loader `pose.DBLoad()`:

```python
from pose import DBLoad
db = DBLoad()                       # parses 54 MB once; subsequent calls are free
DBLoad.cache_clear()                # force a re-read after Parameterise() writes the file
```

`Pose()`, `ForceField()`, `Score()`, `Rotamers()`, and `Pack()` all share the same cached parse — no duplicate I/O.

## Description of amino acids in database.json:

This information resides in `database['Amino Acids'][AMINO_ACID_UNICODE or BACKBONE]`

| Dictionary Key                        | Value Type     | Description of Values |
|---------------------------------------|----------------|-----------------------|
| `Vectors`                             | List of lists  | The position of each atom relative to the N of the backbone. If the N coorinate is X, Y, Z = 0, 0, 0 you will get these vectors. To find the correct vectors position the N at coordinate X, Y, Z = 0, 0, 0, and use the corresponding coordinates of each atom |
| `Tricode`                             | String         | The three letter code for each amino acid |
| `Fused`                               | Boolian        | True = the sidechain is fused to the backbone |
| `Backbone Atoms` or `Sidechain Atoms` | List of lists  | The atom identity of each coordinate point, for example: first coordinate point is the nitrogen with symbol N and PDB entry N, next atom is the hydrogen that is bonded to the nitrogen with symbol H and PDB entry 1H etc... Unlike the PDB where all hydrogens are collected after the amino acid, here each atom's hydrogens come right after it. This makes for easier matrix operations. Order is index [0] = PDB atom's name, index [1] = element, index [2] = partial charge, index [3] = occupancy, index [4] = temperature factor, index [5] = hybridisation |
| `Chi Angle Atoms`                     | List of lists  | The atoms in the sidechain that are contributing to a chi angle |
| `Bonds`                               | Dictionary     | The bond graph as an adjacency list |
| `BondOrders`                          | Dictionary     | The bond order graph as an adjacency list, 1 = single bonds, 1.5 = aromatic resonance partial-double bond, 2 = double bonds, 3 = triple bonds |

> Backbone-dependent rotamer data (formerly carried as a per-amino-acid `BBDEP` field on this object) now lives in the separate `Rotamer Library` top-level block — see "Description of the Rotamer Library in database.json" below.

## Description of nucleotides in database.json:

This information resides in `database['Nucleotides'][NUCEOTIDE_TRICODE]`

| Dictionary Key    | Value Type     | Description of Values |
|-------------------|----------------|-----------------------|
| `Vectors`         | List of lists  | The position of each atom relative to the N of the backbone. If the N coorinate is X, Y, Z = 0, 0, 0 you will get these vectors. To find the correct vectors position the N at coordinate X, Y, Z = 0, 0, 0, and use the corresponding coordinates of each atom |
| `Tricode`         | String         | The three letter code for each nucleotide |
| `Type`            | String         | Identify as `DNA` or `RNA` |
| `Backbone Atoms`  | List of lists  | The atom identity of each backbone coordinate point, first coordinate point is the phosphorus with symbol P and PDB entry P, next atom is the oxygen atom that is bonded to the phosphorus with symbol O and PDB entry OP1 etc... Order is index [0] = PDB atom's name, index [1] = element, index [2] = partial charge, index [3] = occupancy, index [4] = temperature factor, index [5] = hybridisation |
| `Base Atoms`      | List of lists  | The atom identity of each nitrogen base coordinate point. Order is index [0] = PDB atom's name, index [1] = element, index [2] = partial charge, index [3] = occupancy, index [4] = temperature factor, index [5] = hybridisation |
| `Chi Angle Atoms` | List of lists  | The atoms in the sidechain that are contributing to a chi angle |
| `Bonds`           | Dictionary     | The bond graph as an adjacency list |
| `BondOrders`      | Dictionary     | The bond order graph as an adjacency list, 1 = single bonds, 1.5 = aromatic resonance partial-double bond, 2 = double bonds, 3 = triple bonds |

## Description of the Rotamer Library in database.json:

This information resides in `database['Rotamer Library']` and is consumed by `Score()`, `tools.Rotamers()`, and `tools.Pack()`. Derived from the Dunbrack BBDEP2010 rotamer library (Shapovalov & Dunbrack 2011, CC-BY-4.0). Currently covers 24 chi-bearing residue types: ALY, ARG, ASN, ASP, CYS, GLN, GLU, HIS, ILE, LEU, LYS, MET, MSE, PHE, PRO, PTR, SEC, SER, THR, TPO, TRF, TRP, TYR, VAL. Glycine and Alanine carry no χ angles and need no entry.

**Top-level shape:**

| Key             | Value Type     | Description |
|-----------------|----------------|-------------|
| `format`        | String         | Format identifier (currently `"rot_v1"`) |
| `version`       | Int            | Schema version |
| `phi_start`     | Float          | First (φ, ψ) bin lower edge in degrees, default `-180` |
| `phi_step`      | Float          | (φ, ψ) bin width in degrees, default `10` |
| `phi_n`         | Int            | Number of φ bins, default `36` |
| `psi_start`     | Float          | (analogous) |
| `psi_step`      | Float          | (analogous) |
| `psi_n`         | Int            | Number of ψ bins, default `36` |
| `density_grids` | List           | Per-residue total-density grids (auxiliary, not consumed by current code paths) |
| `residues`      | Dict           | Per-residue rotamer mixture data, keyed by 3-letter code |

**Per-residue entry — `database['Rotamer Library']['residues'][TRICODE]`:**

| Key        | Value Type | Description |
|------------|------------|-------------|
| `n_chi`    | Int        | Number of χ angles for this residue type (1 for VAL/SER/THR/CYS/SEC, 2 for LEU/ILE/PHE/TYR/TRP/HIS/ASN/ASP/PRO/MSE/TRF, 3 for MET/GLN/GLU, 4 for ARG/LYS) |
| `rotamers` | Dict       | The CSR-packed rotamer mixture (see below) |
| `densities`| List/None  | Optional per-cell density auxiliary; usually `null` |

**Per-residue `rotamers` block — CSR-packed for compactness:**

| Key            | Value Type     | Description |
|----------------|----------------|-------------|
| `columns`      | List of strings| Column schema for each row of `table`, e.g. for VAL: `['count', 'prob', 'chi1', 'sig1']`; for LEU (n_chi=2): `['count', 'prob', 'chi1', 'chi2', 'sig1', 'sig2']`; in general `[count, prob, chi1..chiN, sig1..sigN]` |
| `table`        | List of rows   | Flat list of all rotamer rows across every (φ, ψ) cell. Each row matches `columns`: the empirical observation `count`, the per-cell normalised probability `prob` (sums to 1 within a cell), the per-rotamer mean χ values in degrees, and the per-rotamer χ standard deviations in degrees |
| `bin_offsets`  | List of ints   | CSR indexing — length `phi_n × psi_n + 1 = 1297` for the default 36×36 grid. Cell `(i_phi, i_psi)` is at `bin_idx = i_phi × psi_n + i_psi`, and its rotamer rows are `table[bin_offsets[bin_idx] : bin_offsets[bin_idx+1]]` |
| `top_chi`      | List           | Optional precomputed (most-probable rotamer mean χ values per cell) lookup, currently unused by `Score._rotamer_prior` / `Rotamers` / `Pack` (which slice `table` directly) |

**Lookup pattern (used internally by `Rotamers`, `Pack`, and `Score._rotamer_prior`):**

```python
i_phi = int(math.floor((phi - phi_start) / phi_step)) % phi_n
i_psi = int(math.floor((psi - psi_start) / psi_step)) % psi_n
bidx  = i_phi * psi_n + i_psi
rows  = table[bin_offsets[bidx] : bin_offsets[bidx + 1]]
# each row is [count, prob, chi1..chiN, sig1..sigN]
```

D-amino acid handling: the library is keyed on the L-form 3-letter code only. Consumers fetch the L cell at `(−φ, −ψ)` and negate the recovered μ values when applying them, exploiting the chi/Ramachandran mirror symmetry between enantiomers.

## Description of energy parameters in database.json:

This information resides in `database['Energy Parameters']` and holds every numerical parameter consumed by `ForceField()`. The bulk of the bonded and non-bonded parameters are derived from **[OpenFF Sage 2.3.0](https://github.com/openforcefield/openff-forcefields)**, released by the [Open Force Field Initiative](https://github.com/openforcefield) under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). An additional 21 SMIRKS for biomolecule chemistry not covered by Sage (aromatic neutral X3 N in nucleobases / HIS / TRP / TRF; selenium in MSE / SEC; phosphate improper) were added during the protein/DNA/RNA SMIRKS-coverage work; **all 21 are QM-fit**: 20 via direct B3LYP/cc-pVTZ scans (cc-pVTZ-PP + ECP for selenium) using [PySCF 2.13](https://github.com/pyscf/pyscf) (Apache 2.0) — bonds and angles via subgroup-rigid scan + parabolic fit on 13 model molecules, proper torsions via 24-step relaxed dihedral scan + Fourier fit, the phosphate improper via out-of-plane scan + harmonic fit; and 1 (the `[#34:1]` selenium vdW) via Psi4 SAPT0/def2-TZVP dimer scan on H2Se from r=3.0 to 7.0 Å in 0.2 Å steps — the LJ form is fit to the (dispersion + exchange-repulsion) sum extracted from SAPT0. Resulting Se vdW: ε = 1.12 kJ/mol, rmin/2 = 2.50 Å (in line with UFF Se ε = 1.18 kJ/mol). The **`UB` (Urey-Bradley) section** is a separate top-level key with **281 CHARMM-style narrow SMIRKS** — one per unique angle chemical environment in Pose's chemistry (proteins, DNA, RNA, plus the small-organic model pool). Each narrow SMIRKS specifies all three atoms by element, connectivity, H-count, and ring membership, giving each a well-defined equilibrium 1-3 distance s0 (unlike OpenFF's broad Angles SMIRKS, which would conflate many distinct environments under one parameter). The UB values are derived from **Seminario-method Hessian projection with analytical bond/angle subtraction** (Seminario, J. M. *Int. J. Quantum Chem.* **60**, 1271 (1996); CHARMM/AMBER-standard variant by MacKerell et al., *J. Phys. Chem. B* **102**, 3586 (1998)): k_ub = (1/4)·u_ik^T·(H_ii + H_kk − 2H_ik)·u_ik − Σ 2K_b·(∂L/∂s)² − 2K_θ·(∂θ/∂s)². The Seminario projection avoids the pseudoinverse instability that plagues full Wilson G/F decomposition on aromatic redundant-internal systems. Currently **114 of 281 UB SMIRKS have QM-fit values** from a 31-mol small-organic reference pool; the remaining 167 are placeholders for environments that only appear in biomolecule poses (full GXG tripeptide Hessians at cc-pVTZ exceed disk-budget on PySCF's HDF5 intermediates — a numerical-Hessian + smaller-fragment redo is planned). The biomolecule UB contribution to total `ForceField()` energy is small and well-behaved (~0.3 % on DNA/RNA, ~2 % on tripeptides). Bonded sections (`Bonds`, `Angles`, `UB`, `ProperTorsions`, `ImproperTorsions`) and `vdW` use SMIRKS strings as keys (e.g. `[#6X4:1]-[#6X4:2]`, `[#6X3:1]:[#6X3:2]`); the SMIRKS pattern matcher in `pose.energy.SMIRKSMatch()` assigns the right parameter to each topology atom on `_compile`. Energy units throughout are **kJ/mol**; lengths are in **Å**, angles in **degrees**. The legacy Score-side blocks (`weights`, `ref_state`, `lk_solvation`, `hbond`, `kbp`) were removed during the recent force-field refactor and are pending re-implementation in the SMIRKS-driven schema.

| Top-level Key      | Value Type | Description of Values |
|--------------------|------------|-----------------------|
| `Constants`        | Dict       | Global constants. `epsilon_r` is the relative dielectric (default 1.0); `f_lj` and `f_elec` are the 1-4 non-bonded scaling factors (LJ = 0.5, electrostatics = 5/6) |
| `Constraints`      | Dict       | SMIRKS-keyed bond constraints. Sage 2.3.0 declares every X–H bond as constrained (`[#1:1]-[*:2]`), making them rigid under SHAKE/RATTLE. `_compile` zeros their bond-stretch energy; equilibrium length is preserved for MD constraint solving |
| `Bonds`            | Dict       | SMIRKS-keyed harmonic bond parameters. Each value is `{id, length, k}` — equilibrium length in Å, force constant in kJ/mol/Å². Stored using the upstream `E = ½k(x − x₀)²` convention (the ½ factor is absorbed into `K` at compile time) |
| `Angles`           | Dict       | SMIRKS-keyed harmonic angle parameters. Each value is `{id, angle, k}` — equilibrium angle in degrees, force constant in kJ/mol/rad². Same ½-factor convention as bonds |
| `ProperTorsions`   | Dict       | SMIRKS-keyed proper-torsion Fourier components. Each value is `{id, components: [{periodicity, phase, k, idivf}, ...]}` — periodicity (int ≥ 1), phase in degrees, barrier height in kJ/mol, divisor factor (typically 1.0) |
| `ImproperTorsions` | Dict       | SMIRKS-keyed improper torsions, trefoil-expanded into the three cyclic permutations of the outer atoms (each contributing `k/3`). Same component shape as `ProperTorsions` |
| `vdW`              | Dict       | SMIRKS-keyed Lennard-Jones parameters. Each value is `{id, epsilon, rmin_half}` (or `sigma` for some entries) — well depth in kJ/mol, half-min-distance or σ in Å |
| `Electrostatic`    | Dict       | Library charges (water, ions, Xe). SMIRKS-keyed; each value is `{id, charges: [...]}` — literal partial charges in elementary charge units, applied in preference to NAGL for the matched atoms |
| `cmap`             | Dict       | Per-amino-acid backbone (φ, ψ) correction grids. Key is a one-letter aa code; value is a 24×24 list-of-lists. Currently populated with a placeholder sin/cos pattern (max ±0.1) reserved for the eventual biomolecule-SMIRKS port; not consumed for `Molecule` poses (set to empty arrays at compile) |
| `AM1BCC`           | Dict       | NAGL graph neural network weights for AM1-BCC partial-charge prediction. `gcn_layers[0..5]` (each with `fc_neigh_w`, `fc_self_w`, `fc_self_b`) and `readout` (`linear_0_w/b`, `linear_1_w/b`). Each weight tensor is `{shape, data}` where `data` is base64-encoded float32 bytes — bit-exact NAGL inference, ~13 MB total |

---

## Community & Contributions

Contributions are welcome! Open an issue or pull request on GitHub, or just email me.

Chat with users and contributors in real time: **IRC:** `#pose` channel on the `irc.libera.chat` network, Or use the [Libera web chat](https://web.libera.chat/#pose), no install needed.

Come ask questions, share what you've built with Pose, or discuss contributions.

---

## How to Cite

If Pose is useful in your research, please cite it. The repository ships a `CITATION.cff` file at the project root with the canonical citation metadata; GitHub's "Cite this repository" button and most reference managers (Zotero, Mendeley) can import it directly. The current entry is:

> Sabban, S. *Pose: A bare metal Python library for building and manipulating protein molecular structures.* 2023. https://github.com/sarisabban/Pose (ORCID: [0000-0002-9621-2395](https://orcid.org/0000-0002-9621-2395))

---

## License

Pose is released under the **GNU General Public License v2.0 (GPL-2.0)**. The full licence text lives in the [`LICENSE`](LICENSE) file at the project root.
