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
p2 = Pose()
p2.Import('1tqg.pdb', chain='A')
p2.ReBuild()     # Re-adds hydrogens if the PDB lacked them
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
p.data['Amino Acids'][0]     # [letter, chain, bb_indices, sc_indices, secondary structure, tricode]
p.data['Atoms'][0]           # [pdb_name, element, charge, temp_factor]
p.data['Coordinates']        # numpy array, shape (N, 3)
p.data['Bonds']              # adjacency list: {atom_index: [bonded_atom_indices]}
```

Iterating over residues and atoms:
```python
for idx, aa in p.data['Amino Acids'].items():
    symbol, chain, bb, sc, ss, tricode = aa
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

| Method                            | Description |
|-----------------------------------|-------------|
| `Pose()`                          | Construct a new Pose object |
| `p.Build('SARI')`                 | Build a polypeptide from a one-letter sequence. Uppercase = L-amino acids, lowercase = D-amino acids |
| `p.Import('1tqg.pdb', chain='A')` | Load a structure from a PDB file (specific chain). If no hydrogens are present they will not be added, use `ReBuild()` afterwards to add them. Cannot load structures with broken/non-continuous chains |
| `p.Export('out.pdb')`             | Write the polypeptide to a PDB file |
| `p.ReBuild()`                     | Rebuild the polypeptide as a primary structure then refold it using current angles and bond lengths. Use `D_AA=True` to rebuild entirely in D-amino acids |

### Measurements

| Method                                  | Description |
|-----------------------------------------|-------------|
| `p.Distance(0, 'N', 1, 'CA')`           | Distance (Å) between any two atoms. Example: N of residue 0 to CA of residue 1 |
| `p.Angle(2, 'PHI')`                     | Get PHI, PSI, or OMEGA angle of a residue. Example: PHI of residue 2 |
| `p.Angle(2, 'chi', 1)`                  | Get CHI 1–4 angle. Example: CHI 1 of residue 2 |
| `p.Atom3Angle(0, 'N', 0, 'CA', 0, 'C')` | Angle between any three atoms. Example: N–CA–C angle of residue 0 |
| `p.Rg()`                                | Radius of gyration (Å) |
| `p.Mass()`                              | Molecular mass (Daltons) |
| `p.Size()`                              | Number of residues |
| `p.FASTA()`                             | One-letter sequence string |

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
| `p.SecondaryStructures()` | List of secondary structure assignments: H=Helix, S=Sheet, L=Loop |
| `print(p.data)`           | Print the full data dictionary |

---

## Data Structure Reference

| Key           | Value Type  | Description |
|---------------|-------------|-------------|
| `Energy`      | Float       | Potential energy of the molecule |
| `Rg`          | Float       | Radius of gyration |
| `Mass`        | Float       | Mass in Daltons |
| `Size`        | Integer     | Number of residues |
| `FASTA`       | String      | One-letter sequence |
| `Amino Acids` | Dict        | `{index: [symbol, chain, bb_atom_indices, sc_atom_indices, secondary_struct, tricode]}`, **zero-based** |
| `Atoms`       | Dict        | `{atom_index: [pdb_name, element, charge, temp_factor]}`, **zero-based** |
| `Bonds`       | Dict        | Bond graph as adjacency list: `{atom_index: [bonded_atom_indices]}` |
| `Coordinates` | NumPy array | Shape `(N, 3)`, Cartesian XYZ for each atom |

---

## Description of the AminoAcid.json:
| Dictionary Key    | Value Type     | Description of Values |
|-------------------|----------------|-----------------------|
| `Vectors`         | List of lists  | The position of each atom relative to the N of the backbone. If the N coorinate is X, Y, Z = 0, 0, 0 you will get these vectors. To find the correct vectors position the N at coordinate X, Y, Z = 0, 0, 0, and use the corresponding coordinates of each atom|
| `Tricode`         | String         | The three letter code for each amino acid|
| `Fused`           | Boolian        | True = the sidechain is fused to the backbone|
| `Atoms`           | List of lists  | The atom identity of each coordinate point, first coordinate point is the nitrogen with symbol N and PDB entry N, next atom is the hydrogen that is bonded to the nitrogen with symbol H and PDB entry 1H etc... Unlike the PDB where all hydrogens are collected after the amino acid, here each atom's hydrogens come right after it. This makes for easier matrix operations. Order is index [0] == PDB atom's name, index [1] == element, index [2] == charge, index [3] == temperature factor|
| `Chi Angle Atoms` | List of lists  | The atoms in the sidechain that are contributing to a chi angle|
| `Bonds`           | Dictionary     | The bond graph as an adjacency list|

---

## Adding New Amino Acids

To add a new amino acid to the library:

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

## Contributing

Contributions are welcome! Open an issue or pull request on GitHub.

These are functions that would make valuable additions to the library:

0. **Easy**: Support CIF file in addition to PDB file formats
1. **Easy**: Structure alignment (RMSD between two poses)
2. **Easy**: Sequence alignment (BLAST & MSA)
3. **Easy**: Remove Proline exception and generalise to any amino acid with a restricted sidechain
4. **Moderate**: Calculating Gasteiger partial charges for each atom
5. **Moderate**: Find all H-bonds
6. **Moderate**: Calculate DSSP for each amino acid
7. **Hard**: SASA calculation for each amino acid
8. **Hard**: Pocket and void calculation
9. **Hard**: AMBER energy function or general input and structure minimisation

Please follow the existing code style: tabs for indentation, 80 characters max line length.

---

## Community

Chat with users and contributors in real time:

**IRC:** `#pose` channel on the `irc.libera.chat` network.
- Or use the [Libera web chat](https://web.libera.chat/#pose), no install needed

Come ask questions, share what you've built with Pose, or discuss contributions.
