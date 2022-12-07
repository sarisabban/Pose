# Pose
A bare metal Python library to build and manipulate protein molecular structures

## Description:
undertstand that indexing starts at 0 for atoms and residues, unlike the PDB where indexing starts at 1












## How to use:


## Description of AminoAcid.json:
* Vectors: Are the position of each atom relative to the N of the backbone
	if the N coorinate is X, Y, Z = 0, 0, 0 you will get the vectors in the
	JSON file. To find the correct vectors position the N at coordinate
	X, Y, Z = 0, 0, 0, and use the corresponding coordinates of each atom.
* Tricode: Is the three letter code for each structure
* Atoms: The atom identity of each coordinate point, first coordinate point
	is the nitrogen with symbol N and PDB entry N, next atom is the hydrogen
	that is bonded to the nitrogen with symbol H and PDB entry 1H etc...
	Unlike the PDB where all hydrogens of an amino acids are collected after
	the amino acid, here each atom's hydrogens come right after it. It is so
	to make the matrix operation easier.
* Chi Angle Atoms: The atoms in the sidechain that are contributing to a
	chi angle.
* Bonds: The bond graph as an adjacency list.



## Description of the polypeptide's data structure:
describe the pose dictionary:
'Energy':0,								float, default 0, potential energy of molecule
'Rg':0,									float, default 0, radius of gyration of the molecule
'Mass':0,								float, default 0, mass in Da of the molecule
'Size':0,								int, default, sequence length of the molecule
'FASTA':None,							string, default None, FASTA sequence of the molecule
'Amino Acids':{},						dictionary, key is amino acid index in sequence, values are tuples of amino acid symbol, chain, backbone atom indices, sidechain atom indices, and secondary structure of the amino acid
'Atoms':{},								dictionary, key is index in coordinates, value is atom name, symbol, and charge of each index in coordinates
'Bond Tree':{},							dictionary, bond graph of the molecule in adjecency list
'Coordinates':np.array([[0, 0, 0]])}	numpy array, XYZ cartesian coordinates of each atom


## Table of methods:
pose = Pose()                             Construct class
pose.Build('GGG')                         Construct a polypeptide with this sequence
pose.PDB('temp.pdb')                      Export pose as .pdb file
pose.GetAtom(0, 'N')                      Get coordinates of first (0) amino acid's Nitrogen (N)
pose.SecondaryStructures()                Get a list of each amino acids secondary structure
pose.Distance(0, 'N', 'CA')               Get distance between any two atoms within the same amino acid
pose.AtomList(PDB=True)                   Get a list of all the atoms in the polypeptide, PDB=True to get their PDB formatted names
pose.Identify(0, 'atom', q=True)          Identify what is atom belongs to a particular index in the coordinates, q=True to identify the atom's charge, use 'redisue' or 'amino acid' to identify the amino acid by index in the polypeptide sequence
pose.Atom3Angle(0, 'N', 0, 'CA', 0, 'C')  Get angle between any three atoms in any amino acid
print(pose.data)                          Print the dictionary data where all the molecule's information reside
pose.Info()                               Print all information about the molecule in an organised printout
pose.Angle(2, 'chi', 1)                   Get phi, psi, omega, chi angles
pose.Rotate(2, 20, 'chi', 1)              Rotate an angle to x degrees given amino acid position and angle type
pose.Adjust(0, 'N', 0, 'CA', 10)          Adjust the distance between any two atoms in any residue
pose.Mutate(1, 'V')                       Mutate a residue at sequence index by new residue










## Example code:
```
sequence = 'GSHMEYLGVFVDETKEYLQNLNDTLLELEKNPEDMELINEAFRALHTLKGMAGTMGFSSMAKLCHTLENILDKARNSEIKITSDLLDKIFAGVDMITRMVDKIVS'
pose = Pose()
pose.Build(sequence)
phi = [360.0, 207.4, 298.1, 295.0, 301.9, 292.7, 297.9, 298.6, 297.7, 300.0, 296.3, 296.0, 296.9, 298.2, 299.3, 302.4, 291.0, 296.4, 293.3, 293.0, 293.9, 314.3, 301.2, 294.7, 293.3, 291.9, 291.6, 290.0, 291.8, 281.4, 216.3, 280.4, 273.3, 282.2, 283.0, 286.2, 298.4, 301.0, 295.3, 295.9, 295.5, 302.5, 301.1, 297.9, 290.0, 300.5, 296.2, 295.5, 301.4, 300.4, 297.7, 297.8, 295.3, 282.4, 276.3, 101.9, 279.4, 306.1, 291.7, 287.5, 296.8, 293.4, 297.2, 289.9, 296.6, 305.2, 288.8, 299.8, 297.9, 298.4, 285.1, 294.4, 292.1, 294.2, 289.1, 264.6, 54.1, 265.9, 213.0, 267.7, 282.3, 231.9, 297.3, 291.4, 297.3, 293.3, 296.8, 296.3, 293.8, 298.0, 302.1, 291.6, 297.5, 297.7, 296.9, 296.5, 300.4, 299.9, 297.7, 297.3, 298.7, 295.5, 300.9, 249.1, 272.8]
psi = [98.8, 163.8, 313.6, 324.1, 314.1, 329.3, 319.7, 316.2, 312.8, 311.4, 313.9, 313.6, 314.9, 315.5, 323.9, 312.0, 326.8, 322.0, 323.4, 319.9, 314.5, 290.8, 320.0, 315.4, 327.5, 325.5, 325.0, 332.8, 319.9, 326.9, 91.3, 355.6, 351.0, 106.6, 347.4, 323.9, 321.6, 313.9, 324.7, 310.3, 322.1, 311.6, 316.9, 312.7, 324.7, 311.9, 316.0, 322.8, 308.1, 315.7, 313.6, 320.8, 321.2, 334.9, 342.3, 0.6, 72.3, 312.6, 329.7, 321.5, 324.6, 320.6, 316.4, 317.8, 313.1, 310.6, 325.8, 314.4, 323.6, 313.5, 337.9, 320.4, 320.7, 316.4, 341.3, 11.2, 36.6, 344.8, 141.3, 151.1, 123.7, 169.2, 320.2, 322.9, 317.3, 326.9, 318.2, 316.8, 311.9, 323.6, 319.8, 324.5, 318.8, 320.0, 322.8, 316.6, 318.7, 317.9, 321.3, 317.1, 324.0, 323.3, 320.1, 24.8, 360.0]
for i in range(len(sequence)):
	P = phi[i]
	S = psi[i]
	if P > 180: P = P - 360
	if S > 180: S = S - 360
	pose.Rotate(i, P, 'phi')
	pose.Rotate(i, S, 'psi')
pose.PDB('out.pdb')
```

## For collaboration:
If anyone is interested in collaborating and contributing to this library, these are the functions that needs to be developed and added:
* Mutate residues (complete def Mutate())
* Replace atom and adjust all bonds attached to it
* Import/Export from/to PDB, cif, omol, topol, mmft (re-organise Hs, add missing Hs after importing, separate each chain to a different pose, import only peptide atoms, deal with MSE Selenomethionine)
* 2 poses MSA
* 2 poses RMSD between them
* DSSP
* OpenBabel
* Find H-bonds in pose
* AMBER energy function
* Simulated Annealing
* Relax protocol
