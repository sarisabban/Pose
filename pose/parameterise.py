import numpy as np
from collections import defaultdict

'''
1. Download CIF structure from RCSB 
'''
filename, UNICODE = 'MSE.cif', 'J'

def RigidMotion(A, B, Ni, CAi, CBi, Ci):
	''' Superimpose amino acid B into A '''
	A1s = np.ones(len(A))
	B1s = np.ones(len(B))
	A = np.c_[A, A1s]
	B = np.c_[B, B1s]
	Aa, Ao, Ab, Ac = A[0], A[4], A[6], A[-3]
	Ba, Bo, Bb, Bc = B[Ni], B[CAi], B[CBi], B[Ci]
	AL = np.array([Aa - Ao, Ab - Ao, Ac - Ao, Ao])
	BL = np.array([Ba - Bo, Bb - Bo, Bc - Bo, Bo])
	BL_= np.linalg.inv(BL)
	M = np.matmul(BL_, AL)
	B = [np.matmul(i, M)[:3] for i in B]
	B = np.array(B)
	Aoxy = A[3][:3]
	B[-1] = Aoxy
	return(B)

ALA = np.array([
[ 0.000, 0.000, 0.000],  # N
[-0.334,-0.943, 0.000],  # H1
[-0.334, 0.471, 0.816],  # H2
[-0.334, 0.471,-0.816],  # H3
[ 1.458, 0.000, 0.000],  # CA
[ 1.822,-0.535, 0.877],  # HA
[ 1.988,-0.773,-1.199],  # CB
[ 3.078,-0.764,-1.185],  # 1HB
[ 1.633,-1.802,-1.154],  # 2HB
[ 1.633,-0.307,-2.117],  # 3HB
[ 2.009, 1.420, 0.000],  # C
[ 2.058, 2.045, 1.023],  # O
[ 2.394, 1.914,-1.023]]) # OXT

COORD = []
ATOMS = []
BONDS = []
CIF = []
TRICODE = None
with open(filename) as f:
	for i, line in enumerate(f):
		line = line.strip().split()
		if i == 3: TRICODE = line[-1]
		if i >  3:
			if (line[0] == TRICODE and len(line) == 18):
				x = float(line[12])
				y = float(line[13])
				z = float(line[14])
				q = 0
				t = 0
				atom = line[2]
				element = line[3]
				COORD.append([x, y, z])
				ATOMS.append([atom, element, q, t])
				CIF.append(line[1])
			if (line[0] == TRICODE and len(line) == 7):
				atom1 = line[1]
				atom2 = line[2]
				BONDS.append([atom1, atom2])
COORD = np.array(COORD)
Ni  = CIF.index('N')
CAi = CIF.index('CA')
CBi = CIF.index('CB')
Ci  = CIF.index('C')
COORD = RigidMotion(ALA, COORD, Ni, CAi, CBi, Ci)
print(COORD)
atoms = [x[0] for x in CIF]

'''
2. Manually add indeces to order
'''
order = [0, 9, 10, 1, 11, 5, 13, 14, 6, 15, 16, 7, 8, 17, 18, 19, 2, 3, 4, 12]

tempCOORD = []
tempATOMS = []
tempCIF = []
for i in order:
	tempCOORD.append(COORD[i])
	tempATOMS.append(ATOMS[i])
	tempCIF.append(CIF[i])
COORD, ATOMS, CIF = tempCOORD, tempATOMS, tempCIF
tempCOORD, tempATOMS, tempCIF = [], [], []
CBi = [i for i, x in enumerate(ATOMS) if x[0] == 'CB'][0]
Ci  = [i for i, x in enumerate(ATOMS) if x[0] == 'C'][0]
COORD = COORD[CBi:Ci]
ATOMS = ATOMS[CBi:Ci]
CIF = CIF[CBi:Ci]
COORD = np.array(COORD)
tempBONDS = defaultdict(list)
for b in BONDS:
	atom1 = [i for i, x in enumerate(CIF) if x == b[0]]
	atom2 = [i for i, x in enumerate(CIF) if x == b[1]]
	if (atom1 != [] and atom2 != []):
		tempBONDS[atom1[0]].append(atom2[0])
		tempBONDS[atom2[0]].append(atom1[0])
BONDS = {}
for k, v in zip(tempBONDS.keys(), tempBONDS.values()):
	v.sort()
	BONDS[k] = v
tempBONDS = []
Keys = list(BONDS.keys())
Keys.sort()
BONDS = {i: BONDS[i] for i in Keys}

A = f'"{UNICODE}"'
B = ': {\n'
C = '\t"Vectors": [\n'
D = ''
for c in COORD: D += '\t\t' + repr(c)[6:-1] + ',\n'
D = D[:-2]
D += '],\n'
E = f'\t"Tricode": "{TRICODE}",\n\t"Sidechain Atoms": [\n'
F = ''
for a in ATOMS: F += '\t\t' + repr(a) + ',\n'
F = F[:-2]
F += '],\n'
F = F.replace("'", '"')
G = '\t"Chi Angle Atoms": [\n\t\t],\n'
H = '\t"Bonds": {\n'
I = ''
for k, v in zip(BONDS.keys(), BONDS.values()):
	x = f'\t\t"{k}": '
	y = repr(v)
	z = x + y + ',\n'
	I += z
I = I[:-2]
I += '}},'
print(A + B + C + D + E + F + G + H + I)

'''
3. Manually get atoms of CHI angels
'''
