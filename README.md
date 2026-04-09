<div align="center">
<pre>
██████╗  ██████╗ ███████╗███████╗
██╔══██╗██╔═══██╗██╔════╝██╔════╝
██████╔╝██║   ██║███████╗█████╗  
██╔═══╝ ██║   ██║╚════██║██╔══╝  
██║     ╚██████╔╝███████║███████╗
╚═╝      ╚═════╝ ╚══════╝╚══════╝
</pre></div>

# Pose
A bare-metal Python library for building and manipulating protein and nucleic acid molecular structures

![Python >= 3](https://img.shields.io/badge/python-%3E%3D3-blue)
![NumPy](https://img.shields.io/badge/dependency-NumPy-orange)
![License: GPL v2](https://img.shields.io/badge/license-GPL%20v2-green)

<img src="pose/Video1.gif" width="25%"/><img src="pose/Video2.gif" width="25%"/><img src="pose/Video3.gif" width="25%"/><img src="pose/Video4.gif" width="25%"/>
<img src="pose/Video5.gif" width="25%"/><img src="pose/Video6.gif" width="25%"/><img src="pose/Video7.gif" width="25%"/><img src="pose/Video8.gif" width="25%"/>

---

## Video Tutorial

**Watch the full walkthrough:** [Video Tutorial on YouTube](https://youtu.be/r0exhjDjUhs)

---

## What is Pose?

Pose constructs a data structure for protein or nucleic acid molecules that contains all relevant information defining a polymer. Primary information includes the XYZ cartesian coordinates of each atom, the identity and charge of each atom, and the bond graph of the entire molecule. Secondary information includes the FASTA sequence, radius of gyration, potential energy, and the secondary structure assignment for each protein residue.

Using this data structure, Pose can build and manipulate polypeptides and nucleic acids: construct any polypeptide or nucleic acid from sequence, move dihedral and rotamer angles, mutate residues and base pairs, and measure bond lengths and angles. It is designed as a substrate for higher-level protocols such as simulated annealing, molecular dynamics, and machine learning-based molecular design.

**Key features:**
- Designed to be extremely stable bare-metal python, with zero external dependencies beyond NumPy
- 26 amino acids supported by default (20 canonical + 6 non-canonical: LYX, MSE, PYL, SEC, TRF, TSO), can be extended to 100+
- Support for both L-amino acids and D-amino acids (mixed sequences fully supported)
- 5 DNA and RNA canonical nucleotides
- Full bond graph with atom partial charges
- Measure and rotate protein dihedral and rotamer angles (φ/ψ/ω/χ)
- Measure and rotate nucleic acids dihedral angles (α/β/γ/δ/ε/ζ/χ)
- Measure and adjust the distance and angle between any atoms
- PDB and mmCIF file import and export
- Pythonic zero-based indexing throughout (unlike PDB's one-based convention)

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

# Build a peptide
p = Pose()
p.Build('MSLESNRGI', chain='A', fmt='protein') # Uppercase = L-amino acids, lowercase = D-amino acids
p.Build('MSLESNRGI', chain='B', fmt='protein') # Add a second chain
p.GetInfo()                                    # Print structured summary

# Inspect properties
print('Sequence:', p.data['FASTA'])
print('Mass:', p.data['Mass'], 'Da')
print('Rg:', p.data['Rg'], 'Å')

# Rotate backbone angles (indices are zero-based)
p.RotateDihedral(1, -60, 'PHI')
p.RotateDihedral(1, -45, 'PSI')

# Mutate and export
p.Mutate(2, 'V')        # Change residue at index 2 (Leu) → Val
p.Export('peptide.pdb')

# Import a protein
p = Pose()
p.Import('1YN3.pdb')
p.GetInfo()

# Import a nucleic acid
p = Pose()
p.Import('1BNA.pdb')
p.GetInfo()


# Build a nucleic acid
p = Pose()
p.Build('ATGCGTACGTTCCGGCAGACGT', chain='A', fmt='DNA')
p.GetInfo()
```

**D-amino acids** — use lowercase letters:
Uppercase sequence letters build L-amino acids (natural form). Lowercase builds D-amino acids (mirror images). Mixed sequences are fully supported.
```python
p.Build('ACEG')   # All L-amino acids
p.Build('aceg')   # All D-amino acids
p.Build('GAg')    # G=L-Gly, A=L-Ala, g=D-Gly
p.Build('AcEg')   # Mixed L/D sequence
```

**Importing a PDB file:**
```python
p = Pose()
p.Import('1TQG.pdb', chain='A')
p.ReBuild()     # Adds missing hydrogens
```

You can run p.ReBuild() after Import() to add hydrogens to the structure. But understand that a new synthetic structure will be built, therefore you will lose the original occupancy and temperature-factor for each atom (replaces with 1.0 and 0.0).

---

## API Reference

### Building & I/O

| Method                                                     | Description |
|------------------------------------------------------------|-------------|
| `p.Import(filename='1YN3.pdb', chain=['A', 'B'], model=1)` | Imports a structure from a PDB or mmCIF file and constructs the `p.data` object. Can import a protein, DNA, or RNA structure. `chain` accepts a single chain ID (`'A'`), a list of chains (`['A', 'B']`), or `None` to import all chains. `model` selects which model to import from multi-model files (e.g. NMR ensembles); defaults to `1`. For atoms with multiple conformers, the highest-occupancy conformer is kept. Cannot import a structure that is a mixture of proteins and nucleic acids in separate chains, import each macromolecule type as a separate pose |
| `p.Export('out.pdb')`                                      | Write the full structure, and all chains, to a PDB or mmCIF file |
| `p.Build('MSLESNRGI', chain='A', fmt='protein')`           | Build a macromolecule from a one-letter sequence. For a polypeptide add the sequence and choose the format `fmt='Protein'`, uppercase = L-amino acids, lowercase = D-amino acids. For a nucleic acid add the sequence and choose the format `fmt='DNA'` or `fmt='RNA'`. You can add more chains by repeating the command with different chain `chain='A'` values. A structure can either be a protein, or a nucleic acid (DNA/RNA), it cannot be a mixture of the two |
| `p.ReBuild(sequence=None, D_AA=False, _mutate=None)`       | Rebuild the polypeptide or nucleic acid. Use `sequence='AGLMTSWVLVA'` to rebuild the structure with multiple bulk mutations on chain A. Use `sequence={'A':'MSLKLSTVVA', 'B':'ASLKSWFWVA'}` to perform mutations at multiple chains at the same time. Use `D_AA=True` to rebuild a protein entirely in D-amino acids. Will add missing hydrogens. For DNA and RNA, the `sequence=''` length must match exactly the original sequence length, otherwise an error will be raised |
| `p.Mutate(1, 'V')`                                         | Mutate a single monomer. For proteins: `p.Mutate(1, 'V')` = residue 1 → L-Valine, `p.Mutate(1, 'v')` = residue 1 → D-Valine. For DNA: `p.Mutate(0, 'T')` = nucleotide 0 → Thymine. For RNA: `p.Mutate(0, 'U')` = nucleotide 0 → Uracil. For double-stranded nucleic acids, the complementary base is also updated automatically |

### Measurements
| Method                                       | Description |
|----------------------------------------------|-------------|
| `p.GetDistance(0, 'N', 5, 'CA')`             | Get the distance in Å between any two atoms. Example: residue 0 nitrogen atom to residue 5 CA atom |
| `p.GetDihedral(2, 'PHI')`                    | Calculate the amino acid φ/ψ/ω/χ and nucleotide α/β/γ/δ/ε/ζ/χ dihedral angles. In this example we are measuring the PHI angle of the 3rd protein residue (index 2). For protein χ dihedral use `p.GetDihedral(4, 'chi', 1)` 5th residue (index 4), CHI 1 angle |
| `p.GetAngle(0, 'N', 5, 'CA', 17, 'C')`       | Get the angle between any three atoms in the whole structure. Example: N of residue 1, CA of residue 5, and C angle of residue 17 |
| `p.GetAtomBonds(0, 1)`                       | Confirm and get the PDB name and element name `[atom 1 element name, atom 1 PDB name, atom 2 PDB name, atom 2 element name]` for two atoms (if they are bonded together). Use the atom indeces. If the two atoms are not bonded an error will be raised |
| `p.GetIdentity(0, 'Atom')`                   | Identify the PDB name of an atom, or an amino acid, or a nucleotide by its index. Example `p.GetIdentity(5, 'Atom')` or `p.GetIdentity(5, 'amino acid')` or `p.GetIdentity(5, 'nucleotide')`. Also, specifically just for atoms, you are return its partial charge using `p.GetIdentity(3, 'Atom', charge=True)` |
| `p.GetInfo()`                                | Print a formatted summary of the structure's information |
| `p.GetAtomCoord(3, 'N')`                     | Get the XYZ coordinates of an atom of a residue or a nucleotide (monomers). Example: `N` nitrogen of monomer index `3` |
| `p.GetAtomIdx(3, 'N')`                       | Get the atom index in `p.data['Coordinates']` from it's name within a monomer. This is the opposite of `p.GetAtomCoord(3, 'N')` |
| `p.GetAtomList(PDB=True)`                    | Get a list of all atom element names for the entire structure. Use `PDB=True` for PDB-formatted names |
| `p.CalcMass()`                               | Calculates the entire molecular mass of a molecule (all chains) in Da (Daltons), updates the value of p.data['Mass'] |
| `p.CalcSize()`                               | Calculates the length of each chain in a structure, updates the value of p.data['Size']. You can get the length of each chain using `p.data['Size'][CHAIN]` |
| `p.CalcFASTA()`                              | Compiles the FASTA sequence of each chain, updates the value of p.data['FASTA']. You can get the FASTA sequence of each chain using `p.data['FASTA'][CHAIN]` |
| `p.CalcRg()`                                 | Calculates the entire Radius of Gyration of a molecule (all chains) in Å (angstrom), updates the value of p.data['Rg'] |
| `p.CalcCharge(iterations=6)`                 | Calculate the Gasteiger-Marsili partial charges to all atoms using iterative equalization (default 6 iterations), updates the value of `p.data['Atoms'][index][2]` |
| `p.CalcDSSP()`                               | Calculates each amino acid's secondary structure assignments, only for proteins, and stores them in `p.data['Amino Acids'][i][4]` and updates `p.data['SS'][CHAIN]`, therefore this is where you can get the SS sequence of each chain. Codes: H=α-helix, G=3₁₀-helix, I=π-helix, E=β-sheet, B=β-bridge, T=turn, S=bend, L=loop, P=PPII-helix |
| `p.CalcSASA(n_points=100, probe_radius=1.4)` | Calculates the Solvent Accessible Surface Area (SASA) for each amino acid, only for proteins, using golden sphere sampling. `n_points` controls sampling density, `probe_radius` is the solvent probe radius in Å (default 1.4 for water). Adds the value to `p.data['Amino Acids'][i][6]` |

### Manipulation
| Method                                                   | Description |
|----------------------------------------------------------|-------------|
| `p.MovePose(theta=5, u=[18, 10, 5], l=6, ori=[0, 0, 0])` | Rotate and/or translate the whole structure. `theta` = rotation angle in degrees, `u` = rotation axis vector (will be normalised), `l` = translation distance in Å, `ori` = target point to translate towards. All parameters are optional (default `None`); you can rotate only, translate only, or both |
| `p.AdjustDistance(0, 'N', 4, 'C', 17)`                   | Set the distance between any two atoms in (Å). Example: set the distance between N in residue 0 and C in residue 4 to 17 Å. Order matters: the second atom (and all atoms downstream of it on the same chain) moves, while the first atom stays fixed. `(0, 'N', 0, 'CA', d)` ≠ `(0, 'CA', 0, 'N', d)` |
| `p.AdjustAngle(1, 'N', 1, 'CA', 1, 'C', -2)`             | Add/subtract degrees from a three-atom angle, with atom 2 being the pivot point. Example: subtract 2° from N–CA–C angle of residue 1, with the CA atom being the pivot |
| `p.RotateDihedral(1, -60, 'PHI')`                        | Rotate the amino acid φ/ψ/ω/χ and nucleotide α/β/γ/δ/ε/ζ/χ dihedral angles. Example: residue 1 PHI dihedral to -60° |

### Tools

These are standalone tools (not Pose() class methods) and thus are called on their own

| Function                                        | Description |
|-------------------------------------------------|-------------|
| `RMSD(pose1, pose2, alg='align')`               | Computes the Root Mean Squared Deviation between two protein `Pose` structures using Cα (alpha-carbon) atoms only. Returns the RMSD in (Å). Supported algorithms: `'align'` (sequence alignment + iterative Kabsch), `'kabsch'` (SVD-based optimal rotation), `'quaternion'` (eigenvalue-based optimal rotation), or `'simple'` (translation only, no rotation) |
| `RMSD_N(pose1, pose2, alg='kabsch')`            | Computes the Root Mean Squared Deviation between two nucleic acid `Pose` structures using C1' atoms only. Returns the RMSD in (Å). Supported algorithms: `'kabsch'`, `'quaternion'`, or `'simple'` |
| `BLAST(FASTA1, FASTA2)`                         | Perform pairwise protein sequence alignment using the Smith-Waterman local alignment algorithm with BLOSUM62 substitution scores, matching the statistical model used by NCBI BLASTP. Returns: `(alignment_string, percent_identity, e_value)` |
| `MSA([FASTA1, FASTA2, FASTA3....])`             | Aligns three or more protein sequences using a ClustalW-like progressive alignment strategy, pairwise distances are computed with `BLAST()`. Returns: `(alignment_string, aligned_list)` |
| `Parameterise('MSE.cif', 'J', 'MSE')`           | To add a new amino acid to the `database.json` library. Takes `filename`, single letter unicode, three letter tricode |

> BLAST handles sequences beyond the 20 canonical L-amino acids automatically: **D-amino acids**: stored as lowercase letters in `pose.data['FASTA']`. BLAST uppercases both sequences before alignment, treating each D-amino acid as its L-counterpart for scoring purposes. This correctly reflects the chemical reality that D- and L-forms of the same residue have identical side-chain chemistry. **Non-canonical amino acids**: any letter not in the 20-letter BLOSUM62 alphabet falls back to: `+4` for a self-match (equal to the minimum BLOSUM62 diagonal), `−1` for a mismatch. This keeps non-canonical residues visible to the aligner without inflating scores.

> MSA handles sequences beyond the 20 canonical L-amino acids, identical to `BLAST()`

For Parameterise() this is the workflow:

1. Download the CIF file for the amino acid from [RCSB Chemical Sketch](https://www.rcsb.org/chemical-sketch)
2. Call `Parameterise()` with the CIF file path, a single-letter key, and the three-letter residue code.

### Small Molecules (PoseM)

The `PoseM` class (in `mol.py`) handles small molecule structures via SMILES parsing. It is separate from the macromolecule `Pose` class.

```python
from pose import *

m = PoseM()
m.FromSMILES('c1ccccc1')  # Benzene
print(m.data['Coordinates'])
print(m.data['Atoms'])
print(m.data['Bonds'])
```

`PoseM.data` structure:

| Key           | Value Type  | Description |
|---------------|-------------|-------------|
| `Energy`      | Float       | Potential energy (default 0) |
| `Rg`          | Float       | Radius of gyration (default 0) |
| `Mass`        | Float       | Molecular mass (default 0) |
| `SMILES`      | String      | The input SMILES string |
| `Atoms`       | Dict        | `{atom_index: [name, element]}` |
| `Bonds`       | Dict        | Bond graph as adjacency list: `{atom_index: [bonded_atom_indices]}` |
| `Coordinates` | NumPy array | Shape `(N, 3)`, Cartesian XYZ for each atom |

`FromSMILES()` parses the SMILES string, adds implicit hydrogens, assigns hybridisation (sp/sp2/sp3), detects rings, and generates 3D coordinates using idealised bond lengths and angles.

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
p.data['Atoms'][0]           # [pdb_name, element, charge, occupancy, temp_factor]
p.data['Coordinates']        # Numpy array, shape (N, 3)
p.data['Bonds']              # Adjacency list: {atom_index: [bonded_atom_indices]}
```

Iterating over residues and atoms:
```python
for idx, aa in p.data['Amino Acids'].items():
    symbol, chain, bb, sc, ss, tricode, sasa = aa
    print(f'Residue {idx}: {tricode} ({symbol}), SS={ss}')

for idx, atom in p.data['Atoms'].items():
    name, element, charge, occupancy, temp = atom
    xyz = p.data['Coordinates'][idx]
    print(f'Atom {idx}: {name} ({element}) at {xyz}')
```

---

## Supported Amino Acids

> Uppercase = L-form, lowercase = D-form. All 26 are supported in mixed L/D sequences. Additional amino acids can be added to the **database.json** file.

> The N-terminus is protonated, as expected at physiological pH (~7.4), and therefore exists as a positively charged ammonium group (–NH<sub>3</sub><sup>+</sup>)

|       |       |       |       |       |
|-------|-------|-------|-------|-------|
|A - ALA|B - LYX|C - CYS|D - ASP|E - GLU|
|F - PHE|G - GLY|H - HIS|I - ILE|J - MSE|
|K - LYS|L - LEU|M - MET|N - ASN|O - PYL|
|P - PRO|Q - GLN|R - ARG|S - SER|T - THR|
|U - SEC|V - VAL|W - TRP|X - TRF|Y - TYR|
|Z - TSO|

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
| `Atoms`       | Dict        | `{atom_index: [pdb_name, element, partial charge, occupancy, temp_factor]}`, **zero-based indexing** |
| `Bonds`       | Dict        | Bond graph as adjacency list: `{atom_index: [bonded_atom_indices]}` |
| `Coordinates` | NumPy array | Shape `(N, 3)`, Cartesian XYZ for each atom |

---

## Description of amino acids in database.json:

This information resides in `database['Amino Acids'][AMINO_ACID_UNICODE or BACKBONE]`

| Dictionary Key                        | Value Type     | Description of Values |
|---------------------------------------|----------------|-----------------------|
| `Vectors`                             | List of lists  | The position of each atom relative to the N of the backbone. If the N coorinate is X, Y, Z = 0, 0, 0 you will get these vectors. To find the correct vectors position the N at coordinate X, Y, Z = 0, 0, 0, and use the corresponding coordinates of each atom|
| `Tricode`                             | String         | The three letter code for each amino acid|
| `Fused`                               | Boolian        | True = the sidechain is fused to the backbone|
| `Backbone Atoms` or `Sidechain Atoms` | List of lists  | The atom identity of each coordinate point, for example: first coordinate point is the nitrogen with symbol N and PDB entry N, next atom is the hydrogen that is bonded to the nitrogen with symbol H and PDB entry 1H etc... Unlike the PDB where all hydrogens are collected after the amino acid, here each atom's hydrogens come right after it. This makes for easier matrix operations. Order is index [0] = PDB atom's name, index [1] = element, index [2] = partial charge, index [3] = occupancy, index [4] = temperature factor |
| `Chi Angle Atoms`                     | List of lists  | The atoms in the sidechain that are contributing to a chi angle|
| `Bonds`                               | Dictionary     | The bond graph as an adjacency list|

## Description of nucleotides in database.json:

This information resides in `database['Nucleotides'][NUCEOTIDE_TRICODE]`

| Dictionary Key    | Value Type     | Description of Values |
|-------------------|----------------|-----------------------|
| `Vectors`         | List of lists  | The position of each atom relative to the N of the backbone. If the N coorinate is X, Y, Z = 0, 0, 0 you will get these vectors. To find the correct vectors position the N at coordinate X, Y, Z = 0, 0, 0, and use the corresponding coordinates of each atom |
| `Tricode`         | String         | The three letter code for each nucleotide |
| `Type`            | String         | Identify as `DNA` or `RNA` |
| `Backbone Atoms`  | List of lists  | The atom identity of each backbone coordinate point, first coordinate point is the phosphorus with symbol P and PDB entry P, next atom is the oxygen atom that is bonded to the phosphorus with symbol O and PDB entry OP1 etc... |
| `Base Atoms`      | List of lists  | The atom identity of each nistrogen base coordinate point |
| `Chi Angle Atoms` | List of lists  | The atoms in the sidechain that are contributing to a chi angle |
| `Bonds`           | Dictionary     | The bond graph as an adjacency list |

---

## Community & Contributions

Contributions are welcome! Open an issue or pull request on GitHub, or just email me.

Chat with users and contributors in real time: **IRC:** `#pose` channel on the `irc.libera.chat` network, Or use the [Libera web chat](https://web.libera.chat/#pose), no install needed.

Come ask questions, share what you've built with Pose, or discuss contributions.
