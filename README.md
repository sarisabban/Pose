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
| Tricode        | String        | The three letter code for each amino acid.
| Atoms          | List of lists | The atom identity of each coordinate point, first coordinate point is the nitrogen with symbol N and PDB entry N, next atom is the hydrogen that is bonded to the nitrogen with symbol H and PDB entry 1H etc... Unlike the PDB where all hydrogens are collected after the amino acid, here each atom's hydrogens come right after it. This makes for easier matrix operations. Order is index [0] == PDB atom's name, index [1] == element, index [2] == charge, index [3] == temperature factor.
| Chi Angle Atoms| List of lists | The atoms in the sidechain that are contributing to a chi angle.
| Bonds          | Dictionary    | The bond graph as an adjacency list.

### Description of the polypeptide's data structure:
| Dictionary Key | Value Type  | Description of Values |
|----------------|-------------|-----------------------|
| Energy         | Float       | The potential energy of the molecule.
| Rg             | Float       | The radius of gyration of the molecule.
| Mass           | Float       | The mass of the molecule in Daltons.
| Size           | Integer     | The sequence length of the molecule.
| FASTA          | String      | The FASTA sequence of the molecule.
| Amino Acids    | Dictionary  | The key is the index in sequence, the value is the amino acid symbol, chain, backbone atom indices, sidechain atom indices, and the secondary structure the amino acid belongs to.
| Atoms          | Dictionary  | The key is the index in the coordinates matrix, the value is the atom's PDB symbol, the element symbol, the charge, and the temperature factor.
| Bonds          | Dictionary  | The bond graph of the molecule as an adjacency list.
| Coordinates    | Numpy array | The XYZ cartesian coordinates of each atom.

## Table of methods:
| Method                                          | Description with example |
|-------------------------------------------------|--------------------------|
| pose = Pose()                                   | Construct the Pose class |
| pose.Build('SARI')                              | Build a polypeptide using a sequence, the polypeptide will be in primary structure. Example: the sequence 'SARI' |
| pose.Export('out.pdb')                          | Export the polypeptide to a .pdb file. Example: the output file's name is out.pdb |
| pose.GetAtom(3, 'N')                            | Get XYZ cartesian coordinates of an atom. Example: fourth amino acid's Nitrogen atom |
| pose.SecondaryStructures()                      | Get a list of each amino acid's secondary structure H:Helix, S:Sheet, L:Loop |
| pose.Distance(0, 'N', 1, 'CA')                  | Get the distance (in Å) between any two atoms in any amino acid. Example: distance between first amino acid's Nitrogen atom and second amino acid's Carbon alpha atom |
| pose.AtomList(PDB=True)                         | Get a list of all the atoms in the polypeptide, use PDB=True to get their PDB formatted names |
| pose.Identify(3, 'atom', q=True)                | Identify what 'atom' type belongs to a particular index in the coordinates matrix, use q=True to identify the atom's charge, use 'redisue' or 'amino acid' to instead identify the amino acid by index in the polypeptide sequence |
| pose.Atom3Angle(0, 'N', 0, 'CA', 0, 'C')        | Get the angle between any three atoms in any amino acid. Example: first amino acid's Nitrogen, first amino acid's Carbon alpha, and first amino acid's Carbon |
| pose.GetBondAtoms(0, 1)                         | Get the atom pair that participate in a bond from their index. Example: atom index 0 and atom index 1 return ['N', 'N', 'HA', 'H'], this returns both atom's PDB name and the element's name |
| print(pose.data)                                | Print the dictionary data structure where all the polypeptide's information reside |
| pose.Info()                                     | Print all the information about the polypeptide in an organised printout |
| pose.Angle(2, 'chi', 1)                         | Get the PHI, PSI, OMEGA, or CHI 1-4 angles of an amino acid. Example: second amino acid's CHI 1 angle. For the PHI, PSI, and OMEGA angles no need to include the second argument (the 1 in this example) | 
| pose.Rotate(2, 20, 'chi', 1)                    | Change an angle to reach a degrees. Example: third amino acid, change angle to become 20 degrees, the angle type is CHI 1 |
| pose.Adjust(0, 'N', 0, 'CA', 10)                | Adjust the distance between any two atoms in any amno acid. Example: distance between first amino acid's Nitrogen and first amino acid's Carbon alpha to become 10 Å. The order of the atoms makes a difference, (0, 'N', 0, 'CA', 10) ≠ (0, 'CA', 0, 'N', 10), useful to seperate the chain behind the N |
| pose.Mutate(1, 'V')                             | Mutate an amno acid. Example: Mutate second amino acid to become Valine |
| pose.Rotation3Angle(1, 'N', 1, 'CA', 1, 'C', -2)| Add/Subtract any three atom backbone angle from current degrees. Example: second amino acid, subtract 2 degrees from the N-Ca-C angle| 
| pose.Import('1tqg.pdb', Build=True)             | Import a .pdb file (no hydrogens nor a bond graph). If the argument Build=True is used the molecule will be re-built to include hydrogens and a bond graph, but it will very slightly deviate from the original molecule. currently i would advise to add hydrogens to a .pdb file (using pymol, export it from pymol into a new hydrated .pdb file) THEN import it here to the pose and NOT include the Build=True argument, this would be the most accurate representation of an imported hydrated polypeptide| 


## Example code:
```
from Pose.pose import *

# PDB ID: 1TQG
sequence = 'GSHMEYLGVFVDETKEYLQNLNDTLLELEKNPEDMELINEAFRALHTLKGMAGTMGFSSMAKLCHTLENILDKARNSEIKITSDLLDKIFAGVDMITRMVDKIVS'
PHIs = [0.0, -129.93856695226455, -59.712848046330365, -63.38881734409807, -57.39984063054933, -71.95788713408446, -61.040413492264, -62.13007515686055, -64.71703088355108, -53.2076118190218, -64.85456307551289, -62.485521693172686, -65.32168586850385, -59.12264698468749, -60.97567370218626, -59.37380663933866, -70.63021390579073, -65.46672647908971, -65.34154755098625, -59.17511317498296, -63.43042449399478, -43.27247187347376, -57.189626909419225, -64.45550445188805, -64.6867011809046, -61.510967673771006, -72.47901270262388, -67.2919293357302, -59.65182775317114, -93.27286092329346, -137.5245484979932, -80.74689309939438, -93.76540838773087, -75.69486933677189, -65.63555537980434, -68.51000337214035, -64.2531099242099, -57.36693749876176, -60.54971388911287, -57.08191330872308, -65.32487862553631, -51.55487656491448, -62.992493725877715, -66.30040151232146, -62.57648204363885, -65.35259977519011, -62.25326690697761, -59.79923930939864, -59.992758039877906, -65.94795126074656, -60.09166855710383, -59.01731553934177, -62.96475890362522, -74.63033193946048, -94.85402835787156, 72.62462914751097, -87.68075000419785, -57.655074107014, -67.99569393126318, -66.06789333991095, -60.639849111022194, -68.2661572176063, -63.28700023085097, -59.90275305070758, -67.85036883484271, -55.93782036733899, -71.90582109522832, -60.01073623284808, -61.14991713136732, -59.69985728002312, -67.427400658517, -69.64527845314686, -58.381855680819434, -69.22355435547885, -70.3545497322312, -96.0809015175212, 58.0921856851976, -81.49906557941276, -142.28839783654982, -103.00038322611269, -74.7102812947419, -111.68797933876805, -51.17796499144992, -60.57266222981429, -64.04919820514638, -62.33197257209529, -60.88369142673108, -66.32706141989027, -66.51760672633796, -62.19252207700386, -58.84263895944511, -72.07076716619108, -55.57643777995639, -64.50927074339744, -69.59746833125563, -63.69023484140047, -55.35044587376017, -64.3713195949764, -58.03050238004633, -65.27050862786186, -64.42939642704809, -66.02556114096168, -58.28052973435344, -107.02390287251387, -72.32985504817596]
PSIs = [129.62526889057597, 162.96133486568056, -42.19324591544922, -37.45070244399084, -47.74878918473243, -26.182306560329877, -39.62008540007867, -40.739688575335265, -46.90490760070101, -48.587156822703285, -41.66637157391476, -44.94241305215979, -45.95837717584182, -40.32754341720533, -35.41504714985775, -44.65047574404526, -36.573424849699904, -35.85108091192615, -41.48292255744211, -45.31525857473604, -65.25254402553432, -46.73727453414999, -42.30362583546054, -45.696765819049226, -34.437273207541374, -35.95562381534413, -34.76860072949235, -23.529650520071886, -31.287649238658428, -34.40019237696436, 89.83224108792875, -12.739291431332541, 1.8333488590629452, 105.46811590682674, -27.027559550702573, -43.63221392950984, -39.39291278904837, -47.13512832956771, -40.40638912795955, -46.86549166675752, -44.917214736801455, -47.971327766029695, -40.33033677801302, -44.19688437075066, -39.912044175623926, -40.91571114774511, -45.464958648089784, -40.55361661260633, -46.10486373382228, -39.72090149603454, -47.21861725587194, -40.320083014166215, -36.02427955768149, -16.765976104631836, -9.542689612805447, 27.82008569457047, 83.16148351613737, -49.189524695237935, -39.35735801157316, -42.83245480430646, -45.27033239496801, -36.67486353545024, -48.28558491786061, -33.76793249965521, -46.166533850964285, -43.09924433123552, -41.79572919150106, -38.556510945691656, -37.078663402139476, -51.90251810690203, -29.143226822765733, -41.687756198310495, -43.49239532835969, -37.317752064688875, -23.318960552387153, 14.143053701403748, 32.223545408161065, -29.233107350896855, 158.723711103232, 149.96230383067444, 108.18194083675105, 172.98664363533234, -46.797489612568924, -42.97105264268241, -40.357327373023686, -35.019639261972465, -40.88818313272655, -42.954347115901555, -44.84888941624711, -38.6822970872757, -37.59021045672718, -42.463007716753026, -46.99398544458111, -35.60731624835766, -40.29163158622568, -38.21135643562432, -45.756180177185726, -42.49351014161504, -48.21002940281777, -41.860103849633305, -28.224897456766218, -34.66732555511994, -37.56797477926108, 2.8698793559436675, 0.0]
OMGs = [-178.7949392511682, 176.18635487931576, -178.93327068730375, 174.36192828953497, -177.4586584503098, 168.87695665354065, 176.34264149123476, 177.69695720795767, 175.548243589846, -178.7469586027285, -179.46297578484766, -179.38257366543814, 179.72382043891395, 176.23224850868445, 175.61298210101543, -178.3036565461278, 175.03192767948758, 174.51771797873744, 176.49598416619492, -179.07744893995968, -170.9759228904379, 169.75232426411932, 176.15255097466425, 179.5988011487601, 176.31457447514697, 178.6164194747927, 177.77125101694793, 173.2682174881364, -174.7640002483977, -174.1685707452991, -176.50898476086937, 179.25295078219114, -176.93240492632992, -176.0201497472126, 178.0692070570761, 177.84025635950943, 173.53691023032658, 178.39047533916735, 176.31698022183204, 178.55079714129667, 174.41720520719994, -174.4284286537277, 178.76081344856823, 178.06622200527568, -179.49694798396928, 177.55213208864706, 178.4570762618047, 179.1517083808171, -178.6422259012037, 174.25436045846007, 178.94410826076256, -178.92195119996086, -177.46426775980234, 175.68232023239668, -175.28931205876978, 179.9700988669865, -169.33240400371494, -174.95803507044087, 179.19756883587465, -178.1747432178388, -176.9476697016498, 175.50314017460886, 176.48074519608375, 176.60486529106763, 176.39536386938332, -177.2935908709208, 177.7785591771783, 179.72811769713903, 175.31172316708194, -177.57554921513534, 177.27723735371296, 172.3333696605687, -174.72510038928522, -177.0205272147921, 176.28833584987044, 177.32886653497306, 176.45028322528367, 178.3540805192419, -178.57704184192082, 172.4128436783304, -176.91203814084662, 168.99304061530424, -178.63346559132694, -177.360861265559, 179.52203744150108, 177.3677351326223, -178.7243729514082, 178.42062666375355, 176.2709052186727, 175.78282598239028, 178.03378271143924, 177.58655502565216, -179.90788284912776, 178.35947430580038, 178.56177465644652, 174.52588498542755, -178.90195168670397, 175.82525327729735, -177.30953009378112, 179.66977994201244, 179.40723335301638, -178.14235170018583, -167.67127706759547, 177.81093843899467, 0.0]
NCaC = [117.74124075373831, 108.70952569761059, 111.58951905456576, 111.86752618214703, 111.28514858910852, 112.15936245895683, 111.18780471367374, 112.22026855339129, 110.089529510099, 111.9815313059972, 109.87105360847002, 110.4385593383941, 110.97713462028726, 110.85505267950639, 112.90632467259408, 111.86943172761453, 111.5581131788777, 110.626446110049, 111.49092414279653, 110.52302798891974, 112.70541571330858, 107.2493393890885, 116.40823825353276, 110.35146366216216, 111.8948559782579, 108.44375673528111, 110.20269947779917, 110.87563184287916, 112.12556878396681, 110.0221331953935, 115.23592433361516, 125.86855330992327, 110.46963009182326, 110.88333591431348, 112.96891120648614, 109.89008290579883, 111.48184983121912, 109.45470718236894, 112.31592155222945, 112.4139279470861, 111.94147679488597, 112.45216066553279, 112.38018655057463, 110.75974549224345, 111.43372905044664, 109.13623118465189, 109.55681288464403, 111.62118197089212, 109.74669398801419, 113.4295966960033, 111.39743546752611, 109.8830783950786, 113.41575211398764, 109.71009965130408, 110.99850997396598, 112.94051259508059, 109.48138965275444, 113.04680549870616, 109.45846528011263, 108.66355218111258, 109.28618078994457, 110.35240180410437, 110.27876491897118, 112.77991987939177, 111.27289906243568, 110.04640246138527, 111.62867277387357, 107.24934593048359, 111.7066580169678, 108.45662949764076, 112.84721998814402, 111.52353937028523, 109.37858814116312, 110.12352041297774, 110.23425096073318, 110.27611860068426, 112.88683797638107, 112.26235272749999, 109.2960472470628, 107.3482486692089, 110.92421106179891, 105.27062129050132, 114.2097574142521, 108.04172378819578, 111.33338262823614, 111.87848437972174, 108.80500518253972, 111.583852647808, 112.70797680149333, 111.8390289197232, 113.55588125373455, 108.79432302452648, 108.48943206093595, 112.38088422082468, 108.85079590984142, 110.40896033159635, 110.9197655060676, 110.67961661642732, 111.25988234314498, 109.44474850364305, 114.15529935071291, 108.87704085451266, 110.74646106098588, 107.80483186307939, 112.22000375018506]

pose = Pose()
pose.Build(sequence)

for i in range(len(sequence)):
	pose.Rotate(i, PHIs[i], 'phi')
	pose.Rotate(i, PSIs[i], 'psi')
	N = pose.Atom3Angle(i, 'N', i, 'CA', i, 'C')
	pose.Rotation3Angle(i, 'N', i, 'CA', i, 'C', NCaC[i]-N)
	if i != len(sequence) -1:
		pose.Rotate(i, OMGs[i], 'omega')

pose.Export('out.pdb')
```

## For collaboration:
If anyone is interested in collaborating and contributing to this library, these are the functions that needs to be developed and added:
1. **Hard**: Add hydrogens to polypeptide algorithm
2. **Easy**: Sequences alignment (BLAST & MSA)
3. **Easy**: Structure alignment (RMSD between two poses)
4. **Moderate**: Calculating Gasteiger Partial Charges for each atom
5. **Moderate**: Find H-bonds in pose
6. **Moderate**: Calculate DSSP
7. **Hard**: AMBER energy function
8. **Easy** - if energy function is available: Simulated Annealing (Minimisatin/Relax protocol)
9. **Easy**: Script to automate amino acid parametrisation
10. **Easy**: Update Build to include Non-canonical amino acids including all D-amino acid backbones
