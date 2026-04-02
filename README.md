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
A bare-metal Python library for building and manipulating protein molecular structures

![Python >= 3](https://img.shields.io/badge/python-%3E%3D3-blue)
![NumPy](https://img.shields.io/badge/dependency-NumPy-orange)
![License: GPL v2](https://img.shields.io/badge/license-GPL%20v2-green)

<img src="pose/Video1.gif" width="25%"/><img src="pose/Video2.gif" width="25%"/><img src="pose/Video3.gif" width="25%"/><img src="pose/Video4.gif" width="25%"/>
---

## Video Tutorial

**Watch the full walkthrough:** [Video Tutorial on YouTube](https://youtu.be/r0exhjDjUhs)

---

## What is Pose?

Pose constructs a data structure for a protein molecule that contains all relevant information defining a polypeptide. Primary information includes the XYZ cartesian coordinates of each atom, the identity and charge of each atom, and the bond graph of the entire molecule. Secondary information includes the FASTA sequence, radius of gyration, potential energy, and the secondary structure assignment for each residue.

Using this data structure, Pose can build and manipulate polypeptides: construct any polypeptide from sequence, move torsion and rotamer angles, mutate residues, and measure bond lengths and angles. It is designed as a substrate for higher-level protocols such as simulated annealing, molecular dynamics, and machine learning-based protein design.

**Key features:**
- Zero external dependencies beyond NumPy
- 26 amino acids supported by default (20 canonical + 6 non-canonical: LYX, MSE, PYL, SEC, TRF, TSO), can be extended to 100+
- Support for both L-amino acids and D-amino acids (mixed sequences fully supported)
- Full bond graph with partial charges, torsion angles (PHI, PSI, OMEGA, CHI)
- PDB/mmCIF file import and export
- Zero-based indexing throughout (unlike PDB's one-based convention)

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
p.Build('GAL')       # Gly-Ala-Leu (uppercase = L-amino acids)
p.GetInfo()          # Print structured summary

# Inspect properties
print('Sequence:', p.GetFASTA())
print('Mass:', p.GetMass(), 'Da')
print('Rg:', p.GetRg(), 'Å')

# Rotate backbone angles (indices are zero-based)
p.RotateDihedral(1, -60, 'PHI')
p.RotateDihedral(1, -45, 'PSI')

# Mutate and export
p.Mutate(2, 'V')        # Change residue at index 2 (Leu) → Val
p.Export('peptide.pdb')
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
p.ReBuild()     # Adds missing hydrogens, and calculates SASA, atomic partial charge, and amino acid secondary structures
```

It is advised that you always run p.ReBuild() after Import(). But understand that a new synthetic structure will be built, therefore you will lose the original occupancy and temperature-factor for each atom (replaces with 1.0 and 0.0).

If you want to use a protein complex with multiple chains, then add each chain as a seperate pose:

```python
pA = Pose()
pA.Import('9ATK.pdb', chain='A')
pA.ReBuild()

pB = Pose()
pB.Import('9ATK.pdb', chain='B')
pB.ReBuild()

pC = Pose()
pC.Import('9ATK.pdb', chain='C')
pC.ReBuild()
```

---

## Key Concepts

### Zero-based indexing

All residue and atom indices start at 0, not 1. Residue 0 is the N-terminal amino acid. This is the **opposite** of PDB convention.

```python
p.Build('GAL')
p.GetDihedral(0, 'PHI')            # PHI of first residue (index 0)
p.GetDihedral(2, 'chi', 1)         # CHI 1 of third residue (index 2)
p.GetDistance(0, 'N', 1, 'CA')     # N of residue 0 to CA of residue 1
```

### Accessing the data structure directly

```python
p.data['FASTA']              # sequence string
p.data['Size']               # number of residues (int)
p.data['Amino Acids'][0]     # [letter, chain, bb_indices, sc_indices, secondary structure, tricode, SASA]
p.data['Atoms'][0]           # [pdb_name, element, charge, occupancy, temp_factor]
p.data['Coordinates']        # numpy array, shape (N, 3)
p.data['Bonds']              # adjacency list: {atom_index: [bonded_atom_indices]}
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

> Uppercase = L-form, lowercase = D-form. All 26 are supported in mixed L/D sequences. Additional amino acids can be added to the **AminoAcid.json** file.

> The N-terminus is protonated, as expected at physiological pH (~7.4), and therefore exists as a positively charged ammonium group (–NH<sub>3</sub><sup>+</sup>)

|       |       |       |       |       |
|-------|-------|-------|-------|-------|
|A - ALA|B - LYX|C - CYS|D - ASP|E - GLU|
|F - PHE|G - GLY|H - HIS|I - ILE|J - MSE|
|K - LYS|L - LEU|M - MET|N - ASN|O - PYL|
|P - PRO|Q - GLN|R - ARG|S - SER|T - THR|
|U - SEC|V - VAL|W - TRP|X - TRF|Y - TYR|
|Z - TSO|

---

## API Reference

### Building & I/O

| Method                                     | Description |
|--------------------------------------------|-------------|
| `Pose()`                                   | Construct a new Pose object |
| `p.Build('SYKDLEGKVKSVLESNRGI')`           | Build a polypeptide from a one-letter sequence. Uppercase = L-amino acids, lowercase = D-amino acids |
| `p.Import('1YN3.cif', chain='A', model=1)` | Load a structure from a PDB or mmCIF file (specific chain). It is best to use ReBuild() after importing a structure to optimise it. Cannot load structures with broken/non-continuous chains or missing backbone atoms. Choose a model `model=1` or `model=2` if an ensemble of models is found. If the same atom is found with multiple occupancies, Import() will only import that atom with the highest occupancy, if they are exactly 0.5, then the function will import that first record only |
| `p.ReBuild()`                              | Rebuild the polypeptide as a primary structure then refold it using current angles and bond lengths. Best to use right after Import(). Use `D_AA=True` to rebuild entirely in D-amino acids. Will add missing hydrogens, calculate each atom's partial charge, as well as each amino acid's secondary structure |
| `p.Export('out.pdb')`                      | Write the polypeptide to a PDB or mmCIF file |

### Measurements

| Method                                  | Description |
|-----------------------------------------|-------------|
| `p.GetDistance(0, 'N', 1, 'CA')`        | Get the distance (Å) between any two atoms. Example: N of residue 0 to CA of residue 1 |
| `p.GetDihedral(2, 'PHI')`               | Get the PHI, PSI, or OMEGA angle of a residue. Example: PHI of residue 2 |
| `p.GetDihedral(2, 'chi', 1)`            | Get the CHI 1–4 angle. Example: CHI 1 of residue 2 (Note: dihedrals are not case sensitive) |
| `p.GetAngle(0, 'N', 0, 'CA', 0, 'C')`   | Get the angle between any three atoms. Example: N–CA–C angle of residue 0 |
| `p.GetMass()`                           | Get the molecular mass of a peptide (Daltons) |
| `p.GetSize()`                           | Get the number of residues in a peptide (length of peptide)|
| `p.GetFASTA()`                          | Get the FASTA sequence of a peptide as a list |
| `p.GetSS()`                             | Get the secondary structure assignments for each amino acid as a list: H = α-helix, G = 3₁₀-helix, I = π-helix, E = β-strand, B = β-bridge, T = Turn, S = Bend, L = Loop. `GetDSSP()` needs to be called first to compute the secondary structures otherwise the default assignment is L for loops |
| `p.GetRg()`                             | Get the radius of gyration (Å) for the whole structure |
| `p.GetCharge()`                         | Get the Gasteiger-Marsili partial charges for every atom and store them in `p.data['Atoms'][i][2]`. Use `iterations=` to control convergence (default 6) |
| `p.GetDSSP()`                           | Get the secondary structure for every amino acid and store them in `p.data['Amino Acids'][i][4]` |
| `p.GetSASA()`                           | Get the Solvent Accessible Surface Area (SASA) for each amino acid, and add the values to `p.data['Amino Acids'][i][6]` |
| `p.GetInfo()`                           | Print a formatted summary of the polypeptide information |
| `p.GetAtomCoord(3, 'N')`                | Get the XYZ coordinates of an atom of a residue. Example: N of residue 3 |
| `p.GetAtomList(PDB=True)`               | Get a list of all atom element names for the entire structure. Use `PDB=True` for PDB-formatted names |
| `p.GetAtomBonds(0, 1)`                  | Get the PDB name and element name `[atom 1 element name, atom 1 PDB name, atom 2 PDB name, atom 2 element name]` for two atoms (if they are bonded together). Use the atom indeces |
| `p.GetIdentity(3, 'atom')`              | Identify an index. what type atom/residue/amino acid an index belongs to. Use `q=True` for charge |
| `print(p.data)`                         | Print the full data JSON object |

You can also inspect the p.data JSON object and extract relevent info using `print(p.data['FASTA'])` or `p.data['Atoms']`.

### Manipulation

| Method                                          | Description |
|-------------------------------------------------|-------------|
| `p.RotateDihedral(1, -60, 'PHI')`               | Rotate a backbone dihedral angle. Example: PHI of residue 1 → -60° |
| `p.RotateDihedral(2, 20, 'chi', 1)`             | Rotate a sidechain dihedral angle. Example: CHI 1 of residue 2 → 20° |
| `p.MovePose(5, [18, 10, 5], 6, [0, 0, 0])`      | Rotate and/or translate the whole structure. Example: rotate `5`° degrees around axis `[18, 10, 5]` and move `6`Å towards point `[0, 0, 0]` |
| `p.MovePose(5, [18, 10, 5], None, None)`        | Rotate without translating |
| `p.AdjustAngle(1, 'N', 1, 'CA', 1, 'C', -2)`    | Add/subtract degrees from a three-atom angle. Example: subtract 2° from N–CA–C angle of residue 1 |
| `p.Mutate(1, 'V')`                              | Mutate a residue. Example: residue 1 → L-Valine. `v` = 1 → D-Valine |
| `p.AdjustDistance(0, 'N', 0, 'CA', 1.46)`       | Set the distance between two atoms (Å). Example: N–CA bond of residue 0 → 1.46 Å. Order matters: `(0,'N',0,'CA',d)` ≠ `(0,'CA',0,'N',d)` |

### Tools

These are standalone tools (not Pose() class methods) and thus are called on their own

| Function                                        | Description |
|-------------------------------------------------|-------------|
| `RMSD(pose1, pose2, alg='align')`               | Computes the Root Mean Squared Deviation between two `Pose` structures using Cα (alpha-carbon) atoms only. Returns the  RMSD in (Å). Supported algorithms: `'align'`, `'kabsch'`, `'quaternion'`, or `'simple'` |
| `BLAST(FASTA1, FASTA2)`                         | Perform pairwise protein sequence alignment using the Smith-Waterman local alignment algorithm with BLOSUM62 substitution scores, matching the statistical model used by NCBI BLASTP. Returns: `(alignment_string, percent_identity, e_value)` |
| `MSA([FASTA1, FASTA2, FASTA3....])`             | Aligns three or more protein sequences using a ClustalW-like progressive alignment strategy, pairwise distances are computed with `BLAST()`. Returns: `(alignment_string, aligned_list)` |
| `Parameterise('MSE.cif', 'J', 'MSE')`           | To add a new amino acid to the `AminoAcids.json` library. Takes `filename`, single letter unicode, three letter tricode |

> BLAST handles sequences beyond the 20 canonical L-amino acids automatically: **D-amino acids**: stored as lowercase letters in `pose.data['FASTA']`. BLAST uppercases both sequences before alignment, treating each D-amino acid as its L-counterpart for scoring purposes. This correctly reflects the chemical reality that D- and L-forms of the same residue have identical side-chain chemistry. **Non-canonical amino acids**: any letter not in the 20-letter BLOSUM62 alphabet falls back to: `+4` for a self-match (equal to the minimum BLOSUM62 diagonal), `−1` for a mismatch. This keeps non-canonical residues visible to the aligner without inflating scores.

> MSA handles sequences beyond the 20 canonical L-amino acids, identical to `BLAST()`

For Parameterise() this is the workflow:

1. Download the CIF file for the amino acid from [RCSB Chemical Sketch](https://www.rcsb.org/chemical-sketch)
2. Call `Parameterise()` with the CIF file path, a single-letter key, and the three-letter residue code.

---

## Data Structure Reference

| Key           | Value Type  | Description |
|---------------|-------------|-------------|
| `Energy`      | Float       | Potential energy of the molecule |
| `Rg`          | Float       | Radius of gyration |
| `Mass`        | Float       | Mass in Daltons |
| `Size`        | Integer     | Number of residues |
| `FASTA`       | String      | One-letter sequence |
| `SS`          | String      | One-letter amino acid secondary structure asignments |
| `Amino Acids` | Dict        | `{index: [symbol, chain, bb_atom_indices, sc_atom_indices, secondary_struct, tricode, SASA]}`, **zero-based indexing** |
| `Atoms`       | Dict        | `{atom_index: [pdb_name, element, partial charge, occupancy, temp_factor]}`, **zero-based indexing** |
| `Bonds`       | Dict        | Bond graph as adjacency list: `{atom_index: [bonded_atom_indices]}` |
| `Coordinates` | NumPy array | Shape `(N, 3)`, Cartesian XYZ for each atom |

---

## Description of the AminoAcid.json:
| Dictionary Key    | Value Type     | Description of Values |
|-------------------|----------------|-----------------------|
| `Vectors`         | List of lists  | The position of each atom relative to the N of the backbone. If the N coorinate is X, Y, Z = 0, 0, 0 you will get these vectors. To find the correct vectors position the N at coordinate X, Y, Z = 0, 0, 0, and use the corresponding coordinates of each atom|
| `Tricode`         | String         | The three letter code for each amino acid|
| `Fused`           | Boolian        | True = the sidechain is fused to the backbone|
| `Sidechain Atoms` | List of lists  | The atom identity of each coordinate point, first coordinate point is the nitrogen with symbol N and PDB entry N, next atom is the hydrogen that is bonded to the nitrogen with symbol H and PDB entry 1H etc... Unlike the PDB where all hydrogens are collected after the amino acid, here each atom's hydrogens come right after it. This makes for easier matrix operations. Order is index [0] = PDB atom's name, index [1] = element, index [2] = partial charge, index [3] = occupancy, index [4] = temperature factor |
| `Chi Angle Atoms` | List of lists  | The atoms in the sidechain that are contributing to a chi angle|
| `Bonds`           | Dictionary     | The bond graph as an adjacency list|

---

## Contributing

Contributions are welcome! Open an issue or pull request on GitHub, or just email me.

---

## Community

Chat with users and contributors in real time:

**IRC:** `#pose` channel on the `irc.libera.chat` network, Or use the [Libera web chat](https://web.libera.chat/#pose), no install needed

Come ask questions, share what you've built with Pose, or discuss contributions.
