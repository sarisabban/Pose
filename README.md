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

<div align="center">
![Python >= 3](https://img.shields.io/badge/python-%3E%3D3-blue)
![NumPy](https://img.shields.io/badge/dependency-NumPy-orange)
![License: GPL v2](https://img.shields.io/badge/license-GPL%20v2-green)
</div>

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
- Both L-amino acids and D-amino acids (mixed sequences fully supported)
- Full bond graph with partial charges, torsion angles (PHI, PSI, OMEGA, CHI 1–4)
- PDB import and export
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
p.Info()             # Print structured summary

# Inspect properties
print('Sequence:', p.FASTA())
print('Mass:', p.Mass(), 'Da')
print('Rg:', p.Rg(), 'Å')

# Rotate backbone angles (indices are zero-based)
p.Rotate(1, -60, 'PHI')
p.Rotate(1, -45, 'PSI')

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

It is advised that you always run p.ReBuild() after Import().

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
p.Angle(0, 'PHI')            # PHI of first residue (index 0)
p.Angle(2, 'chi', 1)         # CHI 1 of third residue (index 2)
p.Distance(0, 'N', 1, 'CA')  # N of residue 0 to CA of residue 1
```

### Accessing the data structure directly

```python
p.data['FASTA']              # sequence string
p.data['Size']               # number of residues (int)
p.data['Amino Acids'][0]     # [letter, chain, bb_indices, sc_indices, secondary structure, tricode, SASA]
p.data['Atoms'][0]           # [pdb_name, element, charge, temp_factor]
p.data['Coordinates']        # numpy array, shape (N, 3)
p.data['Bonds']              # adjacency list: {atom_index: [bonded_atom_indices]}
```

Iterating over residues and atoms:
```python
for idx, aa in p.data['Amino Acids'].items():
    symbol, chain, bb, sc, ss, tricode, sasa = aa
    print(f'Residue {idx}: {tricode} ({symbol}), SS={ss}')

for idx, atom in p.data['Atoms'].items():
    name, element, charge, temp = atom
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

| Method                           | Description |
|----------------------------------|-------------|
| `Pose()`                         | Construct a new Pose object |
| `p.Build('SYKDLEGKVKSVLESNRGI')` | Build a polypeptide from a one-letter sequence. Uppercase = L-amino acids, lowercase = D-amino acids |
| `p.Import('1YN3.cif', chain='A')`| Load a structure from a PDB or mmCIF file (specific chain). It is best to use ReBuild() after importing a structure to optimise it. Cannot load structures with broken/non-continuous chains |
| `p.ReBuild()`                    | Rebuild the polypeptide as a primary structure then refold it using current angles and bond lengths. Best to use right after Import(). Use `D_AA=True` to rebuild entirely in D-amino acids. Will add missing hydrogens, calculate each atom's partial charge, as well as each amino acid's secondary structure |
| `p.Export('out.pdb')`            | Write the polypeptide to a PDB or mmCIF file |

### Measurements

| Method                                  | Description |
|-----------------------------------------|-------------|
| `p.Distance(0, 'N', 1, 'CA')`           | Distance (Å) between any two atoms. Example: N of residue 0 to CA of residue 1 |
| `p.Angle(2, 'PHI')`                     | Get PHI, PSI, or OMEGA angle of a residue. Example: PHI of residue 2 |
| `p.Angle(2, 'chi', 1)`                  | Get CHI 1–4 angle. Example: CHI 1 of residue 2 |
| `p.Atom3Angle(0, 'N', 0, 'CA', 0, 'C')` | Angle between any three atoms. Example: N–CA–C angle of residue 0 |
| `p.Mass()`                              | Calculates the molecular mass of a peptide (Daltons) |
| `p.Size()`                              | Calculates the number of residues in a peptide (length of peptide)|
| `p.FASTA()`                             | Compile the FASTA sequence of a peptide as a list |
| `p.SecondaryStructures()`               | Compile the secondary structure assignments as a list: H = α-helix, G = 3₁₀-helix, I = π-helix, E = β-strand, B = β-bridge, T = Turn, S = Bend, L = Loop |
| `p.Rg()`                                | Compute Radius of gyration (Å) |
| `p.Gasteiger()`                         | Compute Gasteiger-Marsili partial charges for every atom and store them in `p.data['Atoms'][i][2]`. Use `iterations=` to control convergence (default 6) |
| `p.DSSP()`                              | Compute the secondary structure for every amino acid and store them in `p.data['Amino Acids'][i][4]` |
| `p.SASA()`                              | Compute the Solvent Accessible Surface Area (SASA) for each amino acid, and add the value to `p.data['Amino Acids'][i][6]` |

### Manipulation

| Method                                          | Description |
|-------------------------------------------------|-------------|
| `p.Rotate(2, 20, 'chi', 1)`                     | Rotate an angle to a target value (degrees). Example: CHI 1 of residue 2 → 20° |
| `p.Rotate(1, -60, 'PHI')`                       | Rotate a backbone angle. Example: PHI of residue 1 → -60° |
| `p.Mutate(1, 'V')`                              | Mutate a residue. Example: residue 1 → L-Valine |
| `p.Adjust(0, 'N', 0, 'CA', 1.46)`               | Set the distance between two atoms (Å). Example: N–CA bond of residue 0 → 1.46 Å. Order matters: `(0,'N',0,'CA',d)` ≠ `(0,'CA',0,'N',d)` |
| `p.Rotation3Angle(1, 'N', 1, 'CA', 1, 'C', -2)` | Add/subtract degrees from a three-atom angle. Example: subtract 2° from N–CA–C angle of residue 1 |

### Inspection & Utilities

| Method                    | Description |
|---------------------------|-------------|
| `p.Info()`                | Print a formatted summary of all polypeptide information |
| `p.GetAtom(3, 'N')`       | XYZ coordinates of a named atom in a residue. Example: N of residue 3 |
| `p.AtomList(PDB=True)`    | List of all atom names. Use `PDB=True` for PDB-formatted names |
| `p.Identify(3, 'atom')`   | Identify what type an atom index belongs to. Use `q=True` for charge. Use `'residue'` or `'amino acid'` to look up by residue index |
| `p.GetBondAtoms(0, 1)`    | PDB name and element for both atoms of a bond by atom indices |
| `print(p.data)`           | Print the full data JSON |

You can also inspect the p.data JSON object and extract relevent info using `print(p.data['FASTA'])` or `p.data['Atoms']`.

### Structure Alignment and Comparison

The `RMSD()` function is a standalone utility (not a pose class method) that computes the Root Mean Squared Deviation between two `Pose` structures using Cα (alpha-carbon) atoms only.

```python
from pose import *

p1 = Pose()
p1.Import('1YN3.pdb', chain='A')

p2 = Pose()
p2.Import('1YN5.pdb', chain='A')

print(RMSD(p1, p1))        # 0.0
print(RMSD(p1, p2))        # 0.86355
```

| Parameter | Type   | Default   | Description                     |
|-----------|--------|-----------|---------------------------------|
| `pose1`   | `Pose` | —         | Reference structure             |
| `pose2`   | `Pose` | —         | Structure to compare            |
| `alg`     | `str`  | `'align'` | Alignment algorithm (see below) |

**Returns:** `float`, RMSD in Ångströms, rounded to 5 decimal places.

The following are the alignment Algorithms:

| `alg`          | Method | Notes |
|----------------|--------|-------|
| `'align'`      | Needleman–Wunsch sequence alignment, iterative Kabsch with 2 Å outlier cutoff | Default, handles structures of different lengths |
| `'kabsch'`     | SVD-based optimal rotation over first N Cα atoms                              | N = min(len1, len2) |
| `'quaternion'` | Eigenvalue-based optimal rotation over first N Cα atoms                       | Equivalent to `'kabsch'` |
| `'simple'`     | Centroid subtraction only, no rotation                                        | Upper bound on RMSD |

```python
RMSD(p1, p2, alg='align')       # 0.86355  (sequence-aligned core)
RMSD(p1, p2, alg='kabsch')      # 8.26798  (all first-N residues)
RMSD(p1, p2, alg='quaternion')  # 8.26798
RMSD(p1, p2, alg='simple')      # 21.23132
```

### Pairwise Sequence Alignment (BLAST)

`BLAST()` performs pairwise protein sequence alignment using the Smith-Waterman local alignment algorithm with BLOSUM62 substitution scores, affine gap penalties (open=11, extend=1), and Karlin-Altschul E-value statistics, matching the statistical model used by NCBI BLASTP.

```python
from pose import *

p1 = Pose(); p1.Import('1YN3.pdb', chain='A')
p2 = Pose(); p2.Import('8D4Q.pdb', chain='D')

alignment, percent_id, e_val = BLAST(p1.data['FASTA'], p2.data['FASTA'])
print(alignment)
# Query length=98  Subject length=99
# Score: 154.2 bits (393), E-value: 3.401e-45
# Identities: 80/99 (80.81%), Positives: 88/99 (88.9%), Gaps: 1/99 (1.0%)
# ...

print(pct_id)   # 80.81
print(e_val)    # 3.401e-45
```

| Parameter | Type  | Description |
|-----------|-------|-------------|
| `seq1`    | `str` | FASTA sequence of the first protein  |
| `seq2`    | `str` | FASTA sequence of the second protein |

**Returns:** `(alignment_string, percent_identity, e_value)`

| Return value       | Type    | Description |
|--------------------|---------|-------------|
| `alignment_string` | `str`   | BLAST-style formatted alignment with match symbols (`\|` identical, `+` positive, ` ` mismatch/gap) |
| `percent_identity` | `float` | Percentage of identical residues in the aligned region |
| `e_value`          | `float` | Karlin-Altschul expect value (lower = more significant) |

**Non-canonical amino acid handling:**

BLAST handles sequences beyond the 20 canonical L-amino acids automatically:

- **D-amino acids**: stored as lowercase letters in `pose.data['FASTA']` (e.g. `'a'` for D-Ala). BLAST uppercases both sequences before alignment, treating each D-amino acid as its L-counterpart for scoring purposes. This correctly reflects the chemical reality that D- and L-forms of the same residue have identical side-chain chemistry.

- **Non-canonical amino acids**: any letter not in the 20-letter BLOSUM62 alphabet falls back to: `+4` for a self-match (equal to the minimum BLOSUM62 diagonal), `−1` for a mismatch. This keeps non-canonical residues visible to the aligner without inflating scores.

---

### Multiple Sequence Alignment (MSA)

`MSA()` aligns three or more protein sequences using a ClustalW-like progressive alignment strategy: pairwise distances are computed with `BLAST()`, a UPGMA guide tree is built from those distances, and sequences are merged in guide-tree order using Needleman-Wunsch profile-profile alignment (BLOSUM62, affine gap penalties).

```python
from pose import *

s1 = Pose(); p1.Import('1YN3.pdb', chain='A').data['FASTA']
s2 = Pose(); p2.Import('8D4Q.pdb', chain='D').data['FASTA']
s3 = Pose(); p2.Import('1YN4.pdb', chain='A').data['FASTA']
s4 = Pose(); p2.Import('4NZL.pdb', chain='B').data['FASTA']
s5 = Pose(); p2.Import('9ASS.pdb', chain='B').data['FASTA']

alignment, sequences = MSA([s1, s2, s3, s4, s5])
print(alignment)

# Multiple Sequence Alignment (5 sequences, 99 columns)
#
# Seq1  GS-TVPYTITVNGTSQNILSNLTFNKNQNISYKDLEGKVKSVLESNRGITDVDLRLSKQA  59
# Seq2  STIQIPYTITVNGTSQNILSSLTFNKNQNISYKDIENKVKSVLYFNRGISDIDLRLSKQA  60
# ...
#       ....:****:*:*.:. ..:...*.:*:.:.*::::.***:.*...**::...:..::.*
```

| Parameter   | Type        | Description |
|-------------|-------------|-------------|
| `sequences` | `list[str]` | FASTA sequences to align (minimum 2) |

**Returns:** `(alignment_string, aligned_list)`

| Return value       | Type        | Description |
|--------------------|-------------|-------------|
| `alignment_string` | `str`       | ClustalW-style formatted text with conservation symbols |
| `aligned_list`     | `list[str]` | Gap-padded sequences in input order, all strings are the same length |

**Conservation symbols** (bottom row of each alignment block):

| Symbol | Meaning                                            |
|--------|----------------------------------------------------|
| `*`    | All sequences have the same residue in this column |
| `:`    | All pairwise residues score positively on BLOSUM62 |
| `.`    | Average pairwise BLOSUM62 score is positive        |
| ` `    | Low or no conservation                             |

**Non-canonical amino acid handling:** identical to `BLAST()`, D-amino acids (lowercase) are uppercased before scoring, and non-canonical letters fall back to `+4` self-match / `−1` mismatch in BLOSUM62.

**Algorithm summary:**
1. Compute all n(n−1)/2 pairwise distances via `BLAST()` (distance = 1 − identity)
2. Build a UPGMA guide tree from those distances
3. Align sequence groups in merge order using Needleman-Wunsch on column profiles
4. Report conservation per column using BLOSUM62 scores

---

### Adding New Amino Acids

To add a new amino acid to the library use the Parameterise() function, which is a standalone utility (not a pose class method):

1. Download the CIF file for the amino acid from
   [RCSB Chemical Sketch](https://www.rcsb.org/chemical-sketch)
2. Call `Parameterise()` with the CIF file path, a single-letter key, and the
   three-letter residue code:
   ```python
   from pose import *

   Parameterise('MSE.cif', 'J', 'MSE')
   ```

The function reads the CIF geometry, superimposes the amino acid onto the ALA
backbone reference frame, detects chi angles and bond connectivity
automatically, and writes the new entry directly into `AminoAcids.json`.
All 26 canonical/non-canonical entries were generated this way.

| Argument   | Description                                              | Example     |
|------------|----------------------------------------------------------|-------------|
| `filename` | Path to the downloaded CIF file                          | `'MSE.cif'` |
| `unicode`  | Single-letter key for the amino acid (case-insensitive)  | `'J'`       |
| `tricode`  | Three-letter residue code from RCSB (case-insensitive)   | `'MSE'`     |

> **Note:** GLY is not supported (no CB atom). Use any unused character as
> the key; all uppercase letters A–Z are already assigned (see the Supported
> Amino Acids table above).

---

## Data Structure Reference

| Key           | Value Type  | Description |
|---------------|-------------|-------------|
| `Energy`      | Float       | Potential energy of the molecule |
| `Rg`          | Float       | Radius of gyration |
| `Mass`        | Float       | Mass in Daltons |
| `Size`        | Integer     | Number of residues |
| `FASTA`       | String      | One-letter sequence |
| `Amino Acids` | Dict        | `{index: [symbol, chain, bb_atom_indices, sc_atom_indices, secondary_struct, tricode, SASA]}`, **zero-based indexing** |
| `Atoms`       | Dict        | `{atom_index: [pdb_name, element, partial charge, temp_factor]}`, **zero-based indexing** |
| `Bonds`       | Dict        | Bond graph as adjacency list: `{atom_index: [bonded_atom_indices]}` |
| `Coordinates` | NumPy array | Shape `(N, 3)`, Cartesian XYZ for each atom |

---

## Description of the AminoAcid.json:
| Dictionary Key    | Value Type     | Description of Values |
|-------------------|----------------|-----------------------|
| `Vectors`         | List of lists  | The position of each atom relative to the N of the backbone. If the N coorinate is X, Y, Z = 0, 0, 0 you will get these vectors. To find the correct vectors position the N at coordinate X, Y, Z = 0, 0, 0, and use the corresponding coordinates of each atom|
| `Tricode`         | String         | The three letter code for each amino acid|
| `Fused`           | Boolian        | True = the sidechain is fused to the backbone|
| `Atoms`           | List of lists  | The atom identity of each coordinate point, first coordinate point is the nitrogen with symbol N and PDB entry N, next atom is the hydrogen that is bonded to the nitrogen with symbol H and PDB entry 1H etc... Unlike the PDB where all hydrogens are collected after the amino acid, here each atom's hydrogens come right after it. This makes for easier matrix operations. Order is index [0] == PDB atom's name, index [1] == element, index [2] == partial charge, index [3] == temperature factor|
| `Chi Angle Atoms` | List of lists  | The atoms in the sidechain that are contributing to a chi angle|
| `Bonds`           | Dictionary     | The bond graph as an adjacency list|

---

## Contributing

Contributions are welcome! Open an issue or pull request on GitHub.

These are functions that would make valuable additions to the library:

1. **pose.py**: Add organic molecule suppport
2. **tools.py**: Pocket and void calculation
3. **tools.py**: AMBER energy function or general input and structure minimisation

Please follow the existing code style: tabs for indentation, 80 characters max line length.

---

## Community

Chat with users and contributors in real time:

**IRC:** `#pose` channel on the `irc.libera.chat` network, Or use the [Libera web chat](https://web.libera.chat/#pose), no install needed

Come ask questions, share what you've built with Pose, or discuss contributions.
