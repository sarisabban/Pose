# Pose
A bare metal Python library for building and manipulating protein molecular structures

## Description:
This library constructs a pose for a protein molecule, which is a data structure for that contains relevant information that defines the polypeptide molecule. Primary information includes the XYZ cartesian coordinates of each atom, the identify and charge of each atom, and the bond graph of the entire molecule, as well as other secondary information such as the FASTA sequence of the molecule, the molecule's radius of gyration, potential energy, and the secondary structure that each amino acid belongs to.

Using this information, the pose can build and manipulate polypeptides, such as building any polypeptide from sequence, move the torsion and rotamer angles, mutate residues, as well as measure the bond lengths and angles. This data structure can be used to build higher level protocols such as simulated annealing, and machine learning-based protein design.

> __Note__
It is important to note that this library uses **zero-based array indexing**, not one-based as is in the PDB. It is thus important to note that the first amino acid and/or the first atom is indexed as 0 and not 1.

### Description of the AminoAcid.json:
| Dictionary Key | Value Type    | Description of Values |
|----------------|---------------|-----------------------|
| Vectors        | List of lists | The position of each atom relative to the N of the backbone. If the N coorinate is X, Y, Z = 0, 0, 0 you will get these vectors. To find the correct vectors position the N at coordinate X, Y, Z = 0, 0, 0, and use the corresponding coordinates of each atom.
| Tricode        | String        | The three letter code for each structure.
| Atoms          | List of lists | The atom identity of each coordinate point, first coordinate point is the nitrogen with symbol N and PDB entry N, next atom is the hydrogen that is bonded to the nitrogen with symbol H and PDB entry 1H etc... Unlike the PDB where all hydrogens are collected after the amino acid, here each atom's hydrogens come right after it. This makes for easier matrix operations.
| Chi Angle Atoms| List of lists | The atoms in the sidechain that are contributing to a chi angle.
| Bonds          | Dictionary    | The bond graph as an adjacency list.

### Description of the polypeptide's data structure:
| Dictionary Key | Value Type  | Description of Values |
|----------------|-------------|-----------------------|
| Energy         | Float       | The potential energy of the molecule.
| Rg             | Float       | The radius of gyration of the molecule.
| Mass           | Float       | The mass of the molecule in Daltons.
| Size           | Int         | The sequence length of the molecule.
| FASTA          | String      | The FASTA sequence of the molecule.
| Amino Acids    | Dictionary  | The key is the index in sequence, the value is the amino acid symbol, chain, backbone atom indices, sidechain atom indices, and the secondary structure the amino acid belongs to.
| Atoms          | Dictionary  | The key is the index in the coordinates matrix, the value is the atom's PDB symbol, the element symbol, and the charge.
| Bonds          | Dictionary  | The bond graph of the molecule as an adjecency list.
| Coordinates    | Numpy array | The XYZ cartesian coordinates of each atom.

## Table of methods:
| Method                                   | Description with example |
|------------------------------------------|--------------------------|
| pose = Pose()                            | Construct the Pose class |
| pose.Build('SARI')                       | Build a polypeptide using a sequence, the polypeptide will be in primary structure. Example: the sequence 'SARI' |
| pose.Export('out.pdb')                   | Export the polypeptide to a .pdb file. Example: the output file's name is out.pdb |
| pose.GetAtom(3, 'N')                     | Get XYZ cartesian coordinates of an atom. Example: fourth amino acid's Nitrogen atom |
| pose.SecondaryStructures()               | Get a list of each amino acid's secondary structure H:Helix, S:Sheet, L:Loop |
| pose.Distance(0, 'N', 1, 'CA')           | Get the distance (in Å) between any two atoms in any amno acid. Example: distance between first amino acid's Nitrogen atom and second amino acid's Carbon alpha atom |
| pose.AtomList(PDB=True)                  | Get a list of all the atoms in the polypeptide, use PDB=True to get their PDB formatted names |
| pose.Identify(3, 'atom', q=True)         | Identify what 'atom' type belongs to a particular index in the coordinates matrix, use q=True to identify the atom's charge, use 'redisue' or 'amino acid' to instead identify the amino acid by index in the polypeptide sequence |
| pose.Atom3Angle(0, 'N', 0, 'CA', 0, 'C') | Get the angle between any three atoms in any amino acid. Example: first amino acid's Nitrogen, first amino acid's Carbon alpha, and first amino acid's Carbon |
| print(pose.data)                         | Print the dictionary data structure where all the polypeptide's information reside |
| pose.Info()                              | Print all the information about the polypeptide in an organised printout |
| pose.Angle(2, 'chi', 1)                  | Get the PHI, PSI, OMEGA, or CHI 1-4 angles of an amino acid. Example: second amino acid's CHI 1 angle. For the PHI, PSI, and OMEGA angles no need to include the second argument (the 1 in this example) | 
| pose.Rotate(2, 20, 'chi', 1)             | Change an angle to reach a degrees. Example: third amino acid, change angle to become 20 degrees, the angle type is CHI 1 |
| pose.Adjust(0, 'N', 0, 'CA', 10)         | Adjust the distance between any two atoms in any amno acid. Example: distance between first amino acid's Nitrogen and first amino acid's Carbon alpha to become 10 Å |
| pose.Mutate(1, 'V')                      | Mutate an amno acid. Example: Mutate second amino acid to become Valine |
| pose.Rotation_NCaC(1, -2)                 | Add/Subtract the N-Ca-C angle from current degrees. Example: second amino acid, subtract 2 degrees | 

## Example code:
```
sequence = 'GSHMEYLGVFVDETKEYLQNLNDTLLELEKNPEDMELINEAFRALHTLKGMAGTMGFSSMAKLCHTLENILDKARNSEIKITSDLLDKIFAGVDMITRMVDKIVS'
phi = [360.0, 207.4, 298.1, 295.0, 301.9, 292.7, 297.9, 298.6, 297.7, 300.0, 296.3, 296.0, 296.9, 298.2, 299.3, 302.4, 291.0, 296.4, 293.3, 293.0, 293.9, 314.3, 301.2, 294.7, 293.3, 291.9, 291.6, 290.0, 291.8, 281.4, 216.3, 280.4, 273.3, 282.2, 283.0, 286.2, 298.4, 301.0, 295.3, 295.9, 295.5, 302.5, 301.1, 297.9, 290.0, 300.5, 296.2, 295.5, 301.4, 300.4, 297.7, 297.8, 295.3, 282.4, 276.3, 101.9, 279.4, 306.1, 291.7, 287.5, 296.8, 293.4, 297.2, 289.9, 296.6, 305.2, 288.8, 299.8, 297.9, 298.4, 285.1, 294.4, 292.1, 294.2, 289.1, 264.6, 54.1, 265.9, 213.0, 267.7, 282.3, 231.9, 297.3, 291.4, 297.3, 293.3, 296.8, 296.3, 293.8, 298.0, 302.1, 291.6, 297.5, 297.7, 296.9, 296.5, 300.4, 299.9, 297.7, 297.3, 298.7, 295.5, 300.9, 249.1, 272.8]
psi = [98.8, 163.8, 313.6, 324.1, 314.1, 329.3, 319.7, 316.2, 312.8, 311.4, 313.9, 313.6, 314.9, 315.5, 323.9, 312.0, 326.8, 322.0, 323.4, 319.9, 314.5, 290.8, 320.0, 315.4, 327.5, 325.5, 325.0, 332.8, 319.9, 326.9, 91.3, 355.6, 351.0, 106.6, 347.4, 323.9, 321.6, 313.9, 324.7, 310.3, 322.1, 311.6, 316.9, 312.7, 324.7, 311.9, 316.0, 322.8, 308.1, 315.7, 313.6, 320.8, 321.2, 334.9, 342.3, 0.6, 72.3, 312.6, 329.7, 321.5, 324.6, 320.6, 316.4, 317.8, 313.1, 310.6, 325.8, 314.4, 323.6, 313.5, 337.9, 320.4, 320.7, 316.4, 341.3, 11.2, 36.6, 344.8, 141.3, 151.1, 123.7, 169.2, 320.2, 322.9, 317.3, 326.9, 318.2, 316.8, 311.9, 323.6, 319.8, 324.5, 318.8, 320.0, 322.8, 316.6, 318.7, 317.9, 321.3, 317.1, 324.0, 323.3, 320.1, 24.8, 360.0]

pose = Pose()
pose.Build(sequence)

for i in range(len(sequence)):
	pose.Rotate(i, phi[i], 'phi')
	pose.Rotate(i, psi[i], 'psi')

pose.PDB('out.pdb')
```

## For collaboration:
If anyone is interested in collaborating and contributing to this library, these are the functions that needs to be developed and added:
1. **Easy**: Import from PDB, remove all Hs, then add missing Hs, separate each chain to a different pose, import only peptide atoms, deal with MSE Selenomethionine
2. **Easy**: 2 poses BLAST & MSA
3. **Moderate**: 2 poses RMSD between them
4. **Moderate**: Calculating Gasteiger Partial Charges
5. **Moderate**: Find H-bonds in pose
6. **Moderate**: calculate DSSP
7. **Hard**: AMBER energy function
8. **Easy** - if energy function is available: Simulated Annealing (Minimisatin/Relax protocol)
