import re
import os
import json
import math
import numpy as np
from collections import defaultdict, deque

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
	# ALA reference frame (N at origin)
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
				key=lambda n: cif_ord.get(n, 9999))
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
				if n not in visited and not is_h(elem.get(n, ''))]
			if not hvs:
				break
			cur = min(hvs, key=lambda n: cif_ord.get(n, 9999))
		return chain
	def fmt_vec(v):
		x, y, z = round(v[0], 3), round(v[1], 3), round(v[2], 3)
		return f'[{x:6.3f},{y:7.3f},{z:7.3f}]'
	def fmt_atom(a):
		name, el, q, o, t = a
		return f'["{name}", "{el}", {float(q):.1f}, {float(o):.1f}, {float(t):.1f}]'
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
			'Only standard amino acids (not GLY) are supported.')
	# Backbone atom set (from CIF flag; fallback to known names)
	bb_set = {a['id'] for a in ATOMS_RAW if a['bb']}
	if not bb_set:
		bb_set = {
			'N', 'CA', 'C', 'O', 'OXT',
			'H', 'H1', 'H2', 'H3',
			'HA', 'HA2', 'HA3', 'HXT',}
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
			f'Missing backbone atom in {filename}: {e}')
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
	atoms_out = [[name_map[n], elem[n], 0, 1.0, 0] for n in ordered]
	# 10. Build entry
	entry = {
		'Vectors':         coord_out,
		'Tricode':         tricode,
		'Fused':           fused,
		'Sidechain Atoms': atoms_out,
		'Chi Angle Atoms': chis,
		'Bonds':           {str(k): v for k, v in final_bonds.items()},}
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

# BLOSUM62 scoring matrix — shared by BLAST() and MSA()
_aa  = 'ARNDCQEGHILKMFPSTWYV'
_bm  = [
	[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
	[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
	[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
	[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
	[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
	[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
	[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
	[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
	[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
	[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
	[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
	[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
	[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
	[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
	[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
	[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
	[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-3,-2, 0],
	[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-3,11, 2,-3],
	[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
	[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
]
_idx = {c: i for i, c in enumerate(_aa)}

def _blosum(a, b):
	''' BLOSUM62 score; D-AAs are pre-uppercased by callers. '''
	ia, ib = _idx.get(a, -1), _idx.get(b, -1)
	if ia < 0 or ib < 0:
		return(4 if a == b else -1)
	return(_bm[ia][ib])

def BLAST(seq1, seq2):
	'''
	Pairwise protein sequence alignment using Smith-Waterman
	(BLOSUM62, gap open=11, gap extend=1) with Karlin-Altschul
	E-value statistics.

	Parameters
	----------
	seq1 : str  FASTA sequence of the first protein.
	seq2 : str  FASTA sequence of the second protein.

	Returns
	-------
	tuple : (alignment_string, percent_identity, e_value)
	    alignment_string : str   alignment in BLAST-like format
	    percent_identity : float percentage of identical residues
	    e_value          : float Karlin-Altschul expect value
	'''
	seq1 = seq1.upper()
	seq2 = seq2.upper()
	m, n = len(seq1), len(seq2)
	# Affine gap penalties (NCBI BLASTP defaults for BLOSUM62)
	go, ge = 11, 1
	INF    = float('-inf')
	# Smith-Waterman DP with affine gaps
	H  = np.zeros((m+1, n+1))
	E  = np.full((m+1, n+1), INF)
	F  = np.full((m+1, n+1), INF)
	tb = np.zeros((m+1, n+1), dtype=np.int8)
	best, bi, bj = 0.0, 0, 0
	for i in range(1, m+1):
		for j in range(1, n+1):
			s       = _blosum(seq1[i-1], seq2[j-1])
			diag    = H[i-1, j-1] + s
			E[i, j] = max(
				H[i, j-1] - go - ge, E[i, j-1] - ge)
			F[i, j] = max(
				H[i-1, j] - go - ge, F[i-1, j] - ge)
			h       = max(0.0, diag, E[i, j], F[i, j])
			H[i, j] = h
			if h > best:
				best, bi, bj = h, i, j
			if   h == 0:       tb[i, j] = 0
			elif h == diag:    tb[i, j] = 1
			elif h == F[i, j]: tb[i, j] = 2
			else:              tb[i, j] = 3
	if best == 0:
		raise Exception('No alignment found between the sequences')
	# Traceback from the highest-scoring cell
	aq, as_ = [], []
	i, j    = bi, bj
	while i > 0 and j > 0 and H[i, j] > 0:
		t = int(tb[i, j])
		if t == 1:
			aq.append(seq1[i-1]); as_.append(seq2[j-1])
			i -= 1; j -= 1
		elif t == 2:
			aq.append(seq1[i-1]); as_.append('-')
			i -= 1
		else:
			aq.append('-'); as_.append(seq2[j-1])
			j -= 1
	aq  = ''.join(reversed(aq))
	as_ = ''.join(reversed(as_))
	qs, ss  = i + 1, j + 1
	aln_len = len(aq)
	n_id = sum(
		1 for a, b in zip(aq, as_) if a == b and a != '-')
	n_pos = sum(
		1 for a, b in zip(aq, as_)
		if a != '-' and b != '-' and _blosum(a, b) > 0)
	n_gap   = aq.count('-') + as_.count('-')
	pct     = round(n_id / aln_len * 100, 2)
	# Karlin-Altschul E-value (BLOSUM62, gap_open=11, gap_extend=1)
	lam, K  = 0.270, 0.041
	e_value = K * m * n * math.exp(-lam * best)
	bits    = (lam * best - math.log(K)) / math.log(2)
	# Match symbol line
	mid = ''
	for a, b in zip(aq, as_):
		if   a == '-' or b == '-': mid += ' '
		elif a == b:               mid += '|'
		elif _blosum(a, b) > 0:    mid += '+'
		else:                      mid += ' '
	pct_pos = round(n_pos / aln_len * 100, 1)
	pct_gap = round(n_gap / aln_len * 100, 1)
	out = [
		f'Query length={m}  Subject length={n}',
		'',
		(f'Score: {bits:.1f} bits ({int(best)}), '
			f'E-value: {e_value:.3e}'),
		(f'Identities: {n_id}/{aln_len} ({pct}%), '
			f'Positives: {n_pos}/{aln_len} ({pct_pos}%), '
			f'Gaps: {n_gap}/{aln_len} ({pct_gap}%)'),
		'',]
	w = 60
	qp, sp = qs, ss
	for st in range(0, aln_len, w):
		bq = aq[st:st+w]
		bm = mid[st:st+w]
		bs = as_[st:st+w]
		qr = len(bq) - bq.count('-')
		sr = len(bs) - bs.count('-')
		out += [
			f'Query  {qp:>6}  {bq}  {qp+qr-1}',
			f'       {"":>6}  {bm}',
			f'Sbjct  {sp:>6}  {bs}  {sp+sr-1}',
			'',]
		qp += qr
		sp += sr
	return('\n'.join(out), pct, e_value)

def MSA(sequences):
	'''
	Progressive multiple sequence alignment (ClustalW-like).

	Uses UPGMA guide tree built from pairwise BLAST distances,
	then aligns profiles progressively with Needleman-Wunsch
	(BLOSUM62, gap open=11, gap extend=1).  Handles L-AAs,
	D-AAs (uppercased to L-counterpart), and non-canonical AAs.

	Parameters
	----------
	sequences : list of str
	    FASTA sequences to align (at least 2).

	Returns
	-------
	tuple : (alignment_string, aligned_list)
	    alignment_string : str        ClustalW-style formatted text
	    aligned_list     : list[str]  gap-padded sequences, same order
	'''
	n = len(sequences)
	if n < 2:
		raise Exception('MSA requires at least 2 sequences')
	seqs   = [s.upper() for s in sequences]
	labels = [f'Seq{i+1}' for i in range(n)]
	go, ge = 11, 1
	INF    = float('-inf')
	def col_score(p1, p2, ci, cj):
		col_a = [s[ci] for s in p1 if s[ci] != '-']
		col_b = [s[cj] for s in p2 if s[cj] != '-']
		if not col_a or not col_b:
			return(0.0)
		total = sum(
			_blosum(a, b) for a in col_a for b in col_b)
		return(total / (len(col_a) * len(col_b)))
	def align_profiles(p1, p2):
		L1 = len(p1[0])
		L2 = len(p2[0])
		H  = np.zeros((L1+1, L2+1))
		E  = np.full((L1+1, L2+1), INF)
		F  = np.full((L1+1, L2+1), INF)
		tb = np.zeros((L1+1, L2+1), dtype=np.int8)
		for i in range(1, L1+1):
			H[i, 0] = -(go + ge * i)
			tb[i, 0] = 2
		for j in range(1, L2+1):
			H[0, j] = -(go + ge * j)
			tb[0, j] = 3
		for i in range(1, L1+1):
			for j in range(1, L2+1):
				s       = col_score(p1, p2, i-1, j-1)
				diag    = H[i-1, j-1] + s
				E[i, j] = max(
					H[i, j-1] - go - ge, E[i, j-1] - ge)
				F[i, j] = max(
					H[i-1, j] - go - ge, F[i-1, j] - ge)
				h       = max(diag, E[i, j], F[i, j])
				H[i, j] = h
				if   h == diag:    tb[i, j] = 1
				elif h == F[i, j]: tb[i, j] = 2
				else:              tb[i, j] = 3
		np1 = [[] for _ in p1]
		np2 = [[] for _ in p2]
		i, j = L1, L2
		while i > 0 or j > 0:
			if i == 0:
				for k in range(len(p1)): np1[k].append('-')
				for k, s in enumerate(p2): np2[k].append(s[j-1])
				j -= 1
			elif j == 0:
				for k, s in enumerate(p1): np1[k].append(s[i-1])
				for k in range(len(p2)): np2[k].append('-')
				i -= 1
			else:
				t = int(tb[i, j])
				if t == 1:
					for k, s in enumerate(p1):
						np1[k].append(s[i-1])
					for k, s in enumerate(p2):
						np2[k].append(s[j-1])
					i -= 1; j -= 1
				elif t == 2:
					for k, s in enumerate(p1):
						np1[k].append(s[i-1])
					for k in range(len(p2)):
						np2[k].append('-')
					i -= 1
				else:
					for k in range(len(p1)):
						np1[k].append('-')
					for k, s in enumerate(p2):
						np2[k].append(s[j-1])
					j -= 1
		r1 = [''.join(reversed(row)) for row in np1]
		r2 = [''.join(reversed(row)) for row in np2]
		return(r1, r2)
	def upgma(dist):
		sizes  = {k: 1 for k in range(n)}
		active = list(range(n))
		d      = dist.copy()
		order  = []
		for _ in range(n - 1):
			bi, bj, best = -1, -1, float('inf')
			for x in range(len(active)):
				for y in range(x+1, len(active)):
					ii, jj = active[x], active[y]
					if d[ii, jj] < best:
						best, bi, bj = d[ii, jj], ii, jj
			order.append((bi, bj))
			ni, nj = sizes[bi], sizes[bj]
			for k in active:
				if k == bi or k == bj:
					continue
				d[bi, k] = d[k, bi] = (
					ni * d[bi, k] + nj * d[bj, k]
				) / (ni + nj)
			sizes[bi] += sizes[bj]
			active.remove(bj)
		return(order)
	def cons_sym(col):
		non_gap = [c for c in col if c != '-']
		if not non_gap:
			return(' ')
		if (len(non_gap) == n
				and all(c == non_gap[0] for c in non_gap)):
			return('*')
		pairs = [
			_blosum(a, b)
			for x, a in enumerate(non_gap)
			for b in non_gap[x+1:]]
		if not pairs:
			return('*' if len(non_gap) == 1 else ' ')
		if all(s > 0 for s in pairs):
			return(':')
		if sum(pairs) / len(pairs) > 0:
			return('.')
		return(' ')
	dist = np.zeros((n, n))
	for i in range(n):
		for j in range(i+1, n):
			try:
				_, pct, _ = BLAST(seqs[i], seqs[j])
				d = 1.0 - pct / 100.0
			except Exception:
				d = 1.0
			dist[i, j] = dist[j, i] = d
	merge_order = upgma(dist)
	profiles = {k: [seqs[k]] for k in range(n)}
	for (ci, cj) in merge_order:
		a1, a2 = align_profiles(profiles[ci], profiles[cj])
		profiles[ci] = a1 + a2
		del profiles[cj]
	final = list(profiles.values())[0]
	L   = len(final[0])
	lw  = max(max(len(lb) for lb in labels), 4)
	con = ''.join(
		cons_sym([final[k][ci] for k in range(n)])
		for ci in range(L))
	hdr = (
		f'Multiple Sequence Alignment '
		f'({n} sequences, {L} columns)')
	out = [hdr, '']
	pos = [0] * n
	w   = 60
	for st in range(0, L, w):
		for k, lb in enumerate(labels):
			blk  = final[k][st:st+w]
			nres = len(blk) - blk.count('-')
			pos[k] += nres
			out.append(
				f'{lb:<{lw}}  {blk}  {pos[k]}')
		out.append(f'{"":>{lw}}  {con[st:st+w]}')
		out.append('')
	return('\n'.join(out), final)
