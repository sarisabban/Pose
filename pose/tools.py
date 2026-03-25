import re, json, os
import numpy as np
from collections import defaultdict, deque

'''
Add a new amino acid to AminoAcids.json.

Usage:
    from pose import *
    Parameterise('MSE.cif', 'J', 'MSE')

Arguments:
    filename : path to the CIF file (download from RCSB)
    unicode  : single-letter key for AminoAcids.json (e.g. 'J')
    tricode  : three-letter residue code (e.g. 'MSE')

unicode and tricode are uppercased automatically.
The entry is written directly into pose/AminoAcids.json.
'''

# ALA reference frame
ALA = np.array([
	[ 0.000,  0.000,  0.000],  # N
	[-0.334, -0.943,  0.000],  # H1
	[-0.334,  0.471,  0.816],  # H2
	[-0.334,  0.471, -0.816],  # H3
	[ 1.458,  0.000,  0.000],  # CA
	[ 1.822, -0.535,  0.877],  # HA
	[ 1.988, -0.773, -1.199],  # CB
	[ 3.078, -0.764, -1.185],  # 1HB
	[ 1.633, -1.802, -1.154],  # 2HB
	[ 1.633, -0.307, -2.117],  # 3HB
	[ 2.009,  1.420,  0.000],  # C
	[ 2.058,  2.045,  1.023],  # O
	[ 2.394,  1.914, -1.023]]) # OXT

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
	BL_ = np.linalg.inv(BL)
	M = np.matmul(BL_, AL)
	B = [np.matmul(i, M)[:3] for i in B]
	B = np.array(B)
	return B

def is_h(element):
	return element.upper() in ('H', 'D')

def bfs_sidechain(start, adj, skip, elem, cif_ord):
	result, seen = [], set(skip)
	seen.add(start)
	q = deque([start])
	while q:
		atom = q.popleft()
		if is_h(elem.get(atom, '')):
			continue
		result.append(atom)
		nbrs = sorted(
			adj[atom],
			key=lambda n: cif_ord.get(n, 9999)
		)
		for n in nbrs:
			if n in seen:
				continue
			seen.add(n)
			if is_h(elem.get(n, '')):
				result.append(n)
			else:
				q.append(n)
	return result

def rename_atoms(names, elem):
	counter = defaultdict(int)
	out = {}
	for name in names:
		m = re.match(r'^([A-Z]+)(\d+)$', name)
		if is_h(elem.get(name, '')) and m:
			base = m.group(1)
			counter[base] += 1
			out[name] = f'{counter[base]}{base}'
		else:
			out[name] = name
	return out

def trace_main_chain(start, adj, skip, elem, cif_ord):
	chain, visited = [], set(skip) | {'CA'}
	cur = start
	while cur is not None:
		chain.append(cur)
		visited.add(cur)
		hvs = [
			n for n in adj[cur]
			if n not in visited and not is_h(elem.get(n, ''))
		]
		if not hvs:
			break
		cur = min(hvs, key=lambda n: cif_ord.get(n, 9999))
	return chain

def fmt_vec(v):
	x, y, z = round(v[0], 3), round(v[1], 3), round(v[2], 3)
	return f'[{x:6.3f},{y:7.3f},{z:7.3f}]'

def fmt_atom(a):
	name, el, q, t = a
	return f'["{name}", "{el}", {float(q):.1f}, {float(t):.1f}]'

def fmt_chi(c):
	return '[' + ', '.join(f'"{x}"' for x in c) + ']'

def format_entry(key, e, is_last):
	sep = '' if is_last else ','
	L = [f'"{key}": {{']
	for field, val in e.items():
		if field == 'Vectors':
			L.append('    "Vectors": [')
			for vi, v in enumerate(val):
				row = f'        {fmt_vec(v)}'
				L.append(row + (',' if vi < len(val)-1 else '],'))
		elif field == 'Tricode':
			L.append(f'    "Tricode": "{val}",')
		elif field == 'Fused':
			L.append(f'    "Fused": {"true" if val else "false"},')
		elif field in ('Sidechain Atoms', 'Backbone Atoms'):
			L.append(f'    "{field}": [')
			for ai, a in enumerate(val):
				row = f'        {fmt_atom(a)}'
				L.append(row + (',' if ai < len(val)-1 else '],'))
		elif field == 'Chi Angle Atoms':
			if not val:
				L.append('    "Chi Angle Atoms": [')
				L.append('        ],')
			else:
				L.append('    "Chi Angle Atoms": [')
				for ci, c in enumerate(val):
					row = f'        {fmt_chi(c)}'
					L.append(row + (',' if ci < len(val)-1 else '],'))
		elif field == 'Bonds':
			L.append('    "Bonds": {')
			items = list(val.items())
			for bi, (bk, bv) in enumerate(items):
				vals = ', '.join(str(x) for x in bv)
				row  = f'        "{bk}":[{vals}]'
				if bi < len(items) - 1:
					L.append(row + ',')
				else:
					L.append(row + '}}' + sep)
	return '\n'.join(L)

def format_db(db):
	L     = ['{']
	items = list(db.items())
	for i, (k, v) in enumerate(items):
		L.append(format_entry(k, v, is_last=(i == len(items)-1)))
		if i < len(items) - 1:
			L.append('')
	L.append('')
	L.append('}')
	return '\n'.join(L)

def Parameterise(filename, unicode, tricode):
	'''
	Add a new amino acid entry to AminoAcids.json.

	Parameters
	----------
	filename : str
	    Path to the CIF file (download from RCSB Chemical Sketch).
	unicode  : str
	    Single-letter key to use in AminoAcids.json (e.g. 'J').
	tricode  : str
	    Three-letter residue code matching the CIF file (e.g. 'MSE').
	'''
	unicode = unicode.upper()
	tricode = tricode.upper()

	# 1. Parse CIF
	# Atom lines: len >= 18, ideal coords at [15][16][17], bb flag at [9]
	# Bond lines: len == 7
	COORD_RAW, ATOMS_RAW, BONDS = [], [], []
	with open(filename) as f:
		for line in f:
			line = line.strip().split()
			if not line:
				continue
			if line[0] == tricode:
				if len(line) == 7:
					BONDS.append((line[1], line[2]))
				elif len(line) >= 18:
					try:
						try:
							x = float(line[15])
							y = float(line[16])
							z = float(line[17])
						except (ValueError, IndexError):
							x = float(line[12])
							y = float(line[13])
							z = float(line[14])
						bb = (line[9] == 'Y')
						COORD_RAW.append([x, y, z])
						ATOMS_RAW.append({
							'id':   line[1],
							'elem': line[3].capitalize(),
							'bb':   bb,
						})
					except (ValueError, IndexError):
						pass

	COORD   = np.array(COORD_RAW)
	CIF_IDS = [a['id'] for a in ATOMS_RAW]

	if 'CB' not in CIF_IDS:
		raise ValueError(
			f'No CB atom found in {filename}. '
			'Only standard amino acids (not GLY) are supported.'
		)

	# Backbone atom set (from CIF flag; fallback to known names)
	bb_set = {a['id'] for a in ATOMS_RAW if a['bb']}
	if not bb_set:
		bb_set = {
			'N', 'CA', 'C', 'O', 'OXT',
			'H', 'H1', 'H2', 'H3',
			'HA', 'HA2', 'HA3', 'HXT',
		}

	elem    = {a['id']: a['elem'] for a in ATOMS_RAW}
	cif_ord = {a['id']: i for i, a in enumerate(ATOMS_RAW)}

	# 2. Superimpose onto ALA backbone frame
	try:
		Ni  = CIF_IDS.index('N')
		CAi = CIF_IDS.index('CA')
		CBi = CIF_IDS.index('CB')
		Ci  = CIF_IDS.index('C')
	except ValueError as e:
		raise ValueError(
			f'Missing backbone atom in {filename}: {e}'
		)

	COORD = RigidMotion(ALA, COORD, Ni, CAi, CBi, Ci)

	# 3. Bond graph by atom_id
	adj = defaultdict(set)
	for a1, a2 in BONDS:
		adj[a1].add(a2)
		adj[a2].add(a1)

	# 4. BFS from CB in CIF ordinal order
	ordered = bfs_sidechain('CB', adj, bb_set, elem, cif_ord)

	# 5. Rename atoms: CIF suffix → Pose prefix (HB2→1HB, HB3→2HB)
	name_map = rename_atoms(ordered, elem)

	# 6. Detect fused sidechain (e.g. PRO: CD bonds back to N)
	sc_set     = set(ordered)
	fused_atom = None
	for sc in ordered:
		if 'N' in adj[sc]:
			fused_atom = sc
			break
	fused = fused_atom is not None

	# 7. Sidechain bond adjacency (fused ring uses -5 sentinel)
	new_idx  = {n: i for i, n in enumerate(ordered)}
	sc_bonds = defaultdict(list)
	for a1, a2 in BONDS:
		if a1 in sc_set and a2 in sc_set:
			i1, i2 = new_idx[a1], new_idx[a2]
			sc_bonds[i1].append(i2)
			sc_bonds[i2].append(i1)
	if fused:
		i_f = new_idx[fused_atom]
		sc_bonds[i_f].append(-5)
		sc_bonds[-5].append(i_f)

	pos_keys    = sorted(k for k in sc_bonds if k >= 0)
	final_bonds = {k: sorted(sc_bonds[k]) for k in pos_keys}
	if fused:
		final_bonds[-5] = sorted(sc_bonds[-5])

	# 8. Chi angles: trace main chain by CIF ordinal preference
	mc         = trace_main_chain('CB', adj, bb_set, elem, cif_ord)
	full_chain = ['N', 'CA'] + mc

	chis = []
	for i in range(len(full_chain) - 3):
		chis.append([
			full_chain[i],   full_chain[i+1],
			full_chain[i+2], full_chain[i+3],
		])
	if fused and len(full_chain) >= 5:
		chis.append(full_chain[-3:] + ['N'])
		chis.append(full_chain[-2:] + ['N', 'CA'])

	# 9. Assemble output arrays
	id_to_i   = {cid: i for i, cid in enumerate(CIF_IDS)}
	coord_out = [COORD[id_to_i[n]].tolist() for n in ordered]
	atoms_out = [[name_map[n], elem[n], 0, 0] for n in ordered]

	# 10. Build entry
	entry = {
		'Vectors':         coord_out,
		'Tricode':         tricode,
		'Fused':           fused,
		'Sidechain Atoms': atoms_out,
		'Chi Angle Atoms': chis,
		'Bonds':           {str(k): v for k, v in final_bonds.items()},
	}

	# 11. Write to AminoAcids.json in the pose package directory
	db_path = os.path.join(
		os.path.dirname(os.path.abspath(__file__)),
		'AminoAcids.json')

	with open(db_path) as f:
		db = json.load(f)

	if unicode in db:
		print(f'Warning: "{unicode}" already exists... overwriting.')

	db[unicode] = entry

	with open(db_path, 'w') as f:
		f.write(format_db(db))

	print(f'Added {tricode} as "{unicode}" to AminoAcids.json')

def RMSD(pose1, pose2, alg='align'):
	'''
	Calculate RMSD between two poses using CA atoms only.
	Alignment algorithms:
		align      - sequence alignment + iterative Kabsch (default)
		kabsch     - SVD-based optimal rotation, all residues
		quaternion - eigenvalue-based optimal rotation, all residues
		simple     - translation only, no rotation
	'''
	if alg not in ('align', 'kabsch', 'quaternion', 'simple'):
		raise Exception('Unknown algorithm: ' + str(alg))
	def get_CA(pose):
		coords = []
		AAs = pose.data['Amino Acids']
		atoms = pose.data['Atoms']
		crds = pose.data['Coordinates']
		for res_idx in sorted(AAs.keys()):
			for atom_idx in AAs[res_idx][2]:
				if atoms[atom_idx][0] == 'CA':
					coords.append(crds[atom_idx].copy().astype(float))
					break
		return(np.array(coords))
	def kabsch_R(Pc, Qc):
		H = Pc.T @ Qc
		U, S, Vt = np.linalg.svd(H)
		d = np.sign(np.linalg.det(Vt.T @ U.T))
		return(Vt.T @ np.diag(np.array([1.0, 1.0, d])) @ U.T)
	if alg == 'align':
		rk1 = sorted(pose1.data['Amino Acids'].keys())
		rk2 = sorted(pose2.data['Amino Acids'].keys())
		seq1 = ''.join(
			pose1.data['Amino Acids'][k][0].upper()
			for k in rk1)
		seq2 = ''.join(
			pose2.data['Amino Acids'][k][0].upper()
			for k in rk2)
		m, n = len(seq1), len(seq2)
		match, mis, gap = 1.0, -0.5, -1.0
		dp = np.zeros((m+1, n+1))
		for i in range(1, m+1):
			dp[i, 0] = i * gap
		for j in range(1, n+1):
			dp[0, j] = j * gap
		for i in range(1, m+1):
			for j in range(1, n+1):
				s = match if seq1[i-1]==seq2[j-1] else mis
				dp[i, j] = max(
					dp[i-1, j-1] + s,
					dp[i-1, j] + gap,
					dp[i, j-1] + gap)
		pairs, i, j = [], m, n
		while i > 0 and j > 0:
			s = match if seq1[i-1]==seq2[j-1] else mis
			if abs(dp[i,j] - (dp[i-1,j-1]+s)) < 1e-9:
				pairs.append((i-1, j-1))
				i -= 1; j -= 1
			elif abs(dp[i,j] - (dp[i-1,j]+gap)) < 1e-9:
				i -= 1
			else:
				j -= 1
		pairs = list(reversed(pairs))
		if len(pairs) < 3:
			raise Exception('Too few aligned residue pairs')
		def get_CA_res(pose, res_key):
			for idx in pose.data['Amino Acids'][res_key][2]:
				if pose.data['Atoms'][idx][0] == 'CA':
					return(pose.data['Coordinates'][idx].copy().astype(float))
		P_aln = np.array(
			[get_CA_res(pose1, rk1[i]) for i,j in pairs])
		Q_aln = np.array(
			[get_CA_res(pose2, rk2[j]) for i,j in pairs])
		mask = np.ones(len(pairs), dtype=bool)
		for _ in range(5):
			Pm = P_aln[mask]
			Qm = Q_aln[mask]
			t_P = Pm.mean(axis=0)
			t_Q = Qm.mean(axis=0)
			R = kabsch_R(Pm - t_P, Qm - t_Q)
			dists = np.sqrt((
				((P_aln - t_P) - (Q_aln - t_Q) @ R)**2
				).sum(axis=1))
			new_mask = dists < 2.0
			if (np.array_equal(new_mask, mask) or new_mask.sum() < 3): break
			mask = new_mask
		Pm = P_aln[mask]
		Qm = Q_aln[mask]
		Pc = Pm - Pm.mean(axis=0)
		Qc = Qm - Qm.mean(axis=0)
		R = kabsch_R(Pc, Qc)
		diff = Pc - Qc @ R
		rmsd = np.sqrt(np.mean((diff**2).sum(axis=1)))
	else:
		P_full = get_CA(pose1)
		Q_full = get_CA(pose2)
		if len(P_full) == 0 or len(Q_full) == 0:
			raise Exception('No CA atoms found in one or both poses')
		n = min(len(P_full), len(Q_full))
		P = P_full[:n]
		Q = Q_full[:n]
		P = P - P.mean(axis=0)
		Q = Q - Q.mean(axis=0)
		if alg == 'simple':
			diff = P - Q
			rmsd = np.sqrt(np.mean((diff**2).sum(axis=1)))
		elif alg == 'kabsch':
			R = kabsch_R(P, Q)
			diff = P - (Q @ R)
			rmsd = np.sqrt(np.mean((diff**2).sum(axis=1)))
		elif alg == 'quaternion':
			H = P.T @ Q
			R11,R12,R13 = H[0,0], H[0,1], H[0,2]
			R21,R22,R23 = H[1,0], H[1,1], H[1,2]
			R31,R32,R33 = H[2,0], H[2,1], H[2,2]
			F = np.array([
				[R11+R22+R33, R23-R32,     R31-R13,     R12-R21],
				[R23-R32,     R11-R22-R33, R12+R21,     R13+R31],
				[R31-R13,     R12+R21,    -R11+R22-R33, R23+R32],
				[R12-R21,     R13+R31,     R23+R32,    -R11-R22+R33]])
			_, vecs = np.linalg.eigh(F)
			q0,q1,q2,q3 = vecs[:,-1]
			R = np.array([
				[q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
				[2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],
				[2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2-q1**2-q2**2+q3**2]])
			diff = P - (Q @ R)
			rmsd = np.sqrt(np.mean((diff**2).sum(axis=1)))
	return(round(float(rmsd), 5))