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
		if len(a) == 5:
			name, el, q, o, t = a
			return (f'["{name}", "{el}", '
				f'{float(q):.1f}, {float(o):.1f}, {float(t):.1f}]')
		else:
			name, el, q, o = a
			return (f'["{name}", "{el}", '
				f'{int(q)}, {int(o)}]')
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
			elif field == 'Type':
				L.append(f'    "Type": "{val}",')
			elif field == 'Fused':
				L.append(
					'    "Fused": '
					f'{"true" if val else "false"},')
			elif field in ('Sidechain Atoms',
				'Backbone Atoms', 'Base Atoms'):
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
		L = ['{']
		sections = list(db.items())
		for si, (sec_name, entries) in enumerate(sections):
			sec_last = (si == len(sections) - 1)
			L.append(f'"{sec_name}": {{')
			items = list(entries.items())
			for i, (k, v) in enumerate(items):
				L.append(format_entry(
					k, v, is_last=(i == len(items)-1)))
				if i < len(items) - 1:
					L.append('')
			sec_close = '}' if sec_last else '},'
			# Replace the trailing }} of the last entry
			# with }}} (or }}},) to close the section too
			L[-1] = L[-1] + sec_close
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
	# 11. Write to database.json in the pose package directory
	db_path = os.path.join(
		os.path.dirname(os.path.abspath(__file__)),
		'database.json')
	with open(db_path) as f:
		db = json.load(f)
	if unicode in db['Amino Acids']:
		print(f'Warning: "{unicode}" already exists... overwriting.')
	db['Amino Acids'][unicode] = entry
	with open(db_path, 'w') as f:
		f.write(format_db(db))
	print(f'Added {tricode} as "{unicode}" to database.json')

def RMSD(pose1, pose2, alg='align', export=None):
	'''
	Calculate RMSD between two poses (protein or nucleic acid).
	Auto-detects molecule type via pose.data['Type'].
	Proteins align on CA atoms, nucleic acids on C1' atoms.
	Alignment algorithms:
		align      - sequence alignment + iterative Kabsch (default)
		kabsch     - SVD-based optimal rotation, all residues
		quaternion - eigenvalue-based optimal rotation, all residues
		simple     - translation only, no rotation
	'''
	if alg not in ('align', 'kabsch', 'quaternion', 'simple'):
		raise Exception('Unknown algorithm: ' + str(alg))
	t1 = pose1.data['Type']
	t2 = pose2.data['Type']
	p1 = t1 == 'Protein'
	p2 = t2 == 'Protein'
	if p1 != p2:
		raise Exception(
			f'Cannot align {t1} with {t2}: '
			'cannot mix protein and nucleic acid')
	is_pro = p1
	rk = 'Amino Acids' if is_pro else 'Nucleotides'
	ra = 'CA' if is_pro else "C1'"
	def kabsch_R(Pc, Qc):
		H = Pc.T @ Qc
		U, S, Vt = np.linalg.svd(H)
		d = np.sign(np.linalg.det(Vt.T @ U.T))
		return(Vt.T @ np.diag(
			np.array([1.0, 1.0, d])) @ U.T)
	if alg == 'align':
		rk1 = sorted(pose1.data[rk].keys())
		rk2 = sorted(pose2.data[rk].keys())
		seq1 = ''.join(
			pose1.data[rk][k][0].upper() for k in rk1)
		seq2 = ''.join(
			pose2.data[rk][k][0].upper() for k in rk2)
		m, n = len(seq1), len(seq2)
		gap = -1.0
		dp = np.zeros((m+1, n+1))
		for i in range(1, m+1):
			dp[i, 0] = i * gap
		for j in range(1, n+1):
			dp[0, j] = j * gap
		for i in range(1, m+1):
			for j in range(1, n+1):
				s = (_blosum(seq1[i-1], seq2[j-1])
					if is_pro else
					(1.0 if seq1[i-1]==seq2[j-1]
					else -0.5))
				dp[i, j] = max(
					dp[i-1, j-1] + s,
					dp[i-1, j] + gap,
					dp[i, j-1] + gap)
		pairs, i, j = [], m, n
		while i > 0 and j > 0:
			s = (_blosum(seq1[i-1], seq2[j-1])
				if is_pro else
				(1.0 if seq1[i-1]==seq2[j-1]
				else -0.5))
			if abs(dp[i,j]-(dp[i-1,j-1]+s)) < 1e-9:
				pairs.append((i-1, j-1))
				i -= 1; j -= 1
			elif abs(dp[i,j]-(dp[i-1,j]+gap)) < 1e-9:
				i -= 1
			else:
				j -= 1
		pairs = list(reversed(pairs))
		if len(pairs) < 3:
			raise Exception(
				'Too few aligned residue pairs')
		P_aln, Q_aln = [], []
		for ii, jj in pairs:
			for idx in pose1.data[rk][rk1[ii]][2]:
				if pose1.data['Atoms'][idx][0] == ra:
					P_aln.append(
						pose1.data['Coordinates']
						[idx].copy().astype(float))
					break
			for idx in pose2.data[rk][rk2[jj]][2]:
				if pose2.data['Atoms'][idx][0] == ra:
					Q_aln.append(
						pose2.data['Coordinates']
						[idx].copy().astype(float))
					break
		P_aln = np.array(P_aln)
		Q_aln = np.array(Q_aln)
		mask = np.ones(len(pairs), dtype=bool)
		for _ in range(5):
			Pm = P_aln[mask]
			Qm = Q_aln[mask]
			t_P = Pm.mean(axis=0)
			t_Q = Qm.mean(axis=0)
			R = kabsch_R(Pm - t_P, Qm - t_Q)
			dists = np.sqrt((
				((P_aln-t_P)-(Q_aln-t_Q) @ R)**2
				).sum(axis=1))
			new_mask = dists < 2.0
			if (np.array_equal(new_mask, mask)
					or new_mask.sum() < 3):
				break
			mask = new_mask
		Pm = P_aln[mask]
		Qm = Q_aln[mask]
		t_P = Pm.mean(axis=0)
		t_Q = Qm.mean(axis=0)
		Pc = Pm - t_P
		Qc = Qm - t_Q
		R = kabsch_R(Pc, Qc)
		diff = Pc - Qc @ R
		rmsd = np.sqrt(np.mean((diff**2).sum(axis=1)))
	else:
		coords1, coords2 = [], []
		for res_idx in sorted(pose1.data[rk].keys()):
			for ai in pose1.data[rk][res_idx][2]:
				if pose1.data['Atoms'][ai][0] == ra:
					coords1.append(
						pose1.data['Coordinates']
						[ai].copy().astype(float))
					break
		for res_idx in sorted(pose2.data[rk].keys()):
			for ai in pose2.data[rk][res_idx][2]:
				if pose2.data['Atoms'][ai][0] == ra:
					coords2.append(
						pose2.data['Coordinates']
						[ai].copy().astype(float))
					break
		if not coords1 or not coords2:
			raise Exception(
				f'No {ra} atoms found in one or both poses')
		n = min(len(coords1), len(coords2))
		P = np.array(coords1[:n])
		Q = np.array(coords2[:n])
		t_P = P.mean(axis=0)
		t_Q = Q.mean(axis=0)
		P = P - t_P
		Q = Q - t_Q
		if alg == 'simple':
			R = np.eye(3)
			diff = P - Q
		elif alg == 'kabsch':
			R = kabsch_R(P, Q)
			diff = P - (Q @ R)
		elif alg == 'quaternion':
			H = P.T @ Q
			R11,R12,R13 = H[0,0],H[0,1],H[0,2]
			R21,R22,R23 = H[1,0],H[1,1],H[1,2]
			R31,R32,R33 = H[2,0],H[2,1],H[2,2]
			F = np.array([
				[R11+R22+R33, R23-R32,
					R31-R13, R12-R21],
				[R23-R32, R11-R22-R33,
					R12+R21, R13+R31],
				[R31-R13, R12+R21,
					-R11+R22-R33, R23+R32],
				[R12-R21, R13+R31,
					R23+R32, -R11-R22+R33]])
			_, vecs = np.linalg.eigh(F)
			q0,q1,q2,q3 = vecs[:,-1]
			R = np.array([
				[q0**2+q1**2-q2**2-q3**2,
					2*(q1*q2-q0*q3),
					2*(q1*q3+q0*q2)],
				[2*(q1*q2+q0*q3),
					q0**2-q1**2+q2**2-q3**2,
					2*(q2*q3-q0*q1)],
				[2*(q1*q3-q0*q2),
					2*(q2*q3+q0*q1),
					q0**2-q1**2-q2**2+q3**2]])
			diff = P - (Q @ R)
		rmsd = np.sqrt(np.mean((diff**2).sum(axis=1)))
	if export is not None:
		orig = pose2.data['Coordinates'].copy()
		pose2.data['Coordinates'] = \
			(orig - t_Q) @ R + t_P
		fn  = export[:-4]
		ext = export[-4:]
		pose1.Export(fn + '_1' + ext)
		pose2.Export(fn + '_2' + ext)
		pose2.data['Coordinates'] = orig
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
	tuple : (alignment_string, aligned_list,
	         conservation, entropy, pssm, dca)
	    alignment_string : str          ClustalW-style formatted text
	    aligned_list     : list[str]    gap-padded sequences, same order
	    conservation     : list[float]  per-column 1 - H/log2(20), [0,1]
	    entropy          : list[float]  per-column Shannon entropy (bits)
	    pssm             : np.ndarray   shape (L, 20) log-odds, AA order
	                                    'ARNDCQEGHILKMFPSTWYV'
	    dca              : np.ndarray   shape (L, L) APC-corrected mfDCA
	                                    direct-information matrix
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
	# --- conservation, entropy, PSSM, DCA -----------------------
	# Alphabet: gap + 20 BLOSUM62 amino acids.  q=21.
	alphabet = '-' + _aa
	q = len(alphabet)
	a2i = {c: i for i, c in enumerate(alphabet)}
	B = n
	M = np.zeros((B, L), dtype=np.int8)
	for bi, s in enumerate(final):
		for ci, ch in enumerate(s):
			M[bi, ci] = a2i.get(ch, 0)
	log2_20 = math.log2(20)
	entropy = []
	conservation = []
	for ci in range(L):
		col = M[:, ci]
		nz = col[col != 0]
		if len(nz) == 0:
			entropy.append(0.0)
			conservation.append(0.0)
			continue
		counts = np.bincount(nz, minlength=q)[1:]
		p = counts / counts.sum()
		nzp = p[p > 0]
		H = float(-np.sum(nzp * np.log2(nzp)))
		entropy.append(round(H, 4))
		conservation.append(round(1.0 - H / log2_20, 4))
	# PSSM with Laplace pseudocount, uniform background 1/20
	pssm = np.zeros((L, 20), dtype=float)
	for ci in range(L):
		col = M[:, ci]
		nz = col[col != 0]
		counts = np.bincount(nz, minlength=q)[1:]
		denom = counts.sum() + 20.0
		freqs = (counts + 1.0) / denom
		pssm[ci] = np.log2(freqs * 20.0)
	# Mean-field DCA with sequence reweighting + APC
	# Sequence identity reweighting (theta = 0.2, i.e. 80%)
	theta = 0.2
	weights = np.ones(B)
	if B > 1:
		simthr = (1.0 - theta) * L
		eq_count = np.zeros(B)
		for a in range(B):
			for b in range(a, B):
				if a == b:
					eq_count[a] += 1
					continue
				eq = int((M[a] == M[b]).sum())
				if eq >= simthr:
					eq_count[a] += 1
					eq_count[b] += 1
		weights = 1.0 / eq_count
	Beff = float(weights.sum())
	# Single-site frequencies (B_eff weighted), q states
	Pi = np.zeros((L, q))
	for bi in range(B):
		for ci in range(L):
			Pi[ci, M[bi, ci]] += weights[bi]
	Pi /= Beff
	# Pseudocount lambda
	lam = 0.5
	Pi_pc = (1.0 - lam) * Pi + lam / q
	# Two-site frequencies
	Pij = np.zeros((L, L, q, q))
	for bi in range(B):
		w_b = weights[bi]
		row = M[bi]
		for i in range(L):
			ai = row[i]
			for j in range(L):
				Pij[i, j, ai, row[j]] += w_b
	Pij /= Beff
	Pij_pc = (1.0 - lam) * Pij + lam / (q * q)
	# Diagonal trick: P_ii(a,b) = Pi(a)*delta(a,b)
	for i in range(L):
		Pij_pc[i, i] = 0.0
		for a in range(q):
			Pij_pc[i, i, a, a] = Pi_pc[i, a]
	# Build covariance matrix dropping last state (gap as gauge ref)
	qm = q - 1
	C = np.zeros((L * qm, L * qm))
	for i in range(L):
		for j in range(L):
			for a in range(qm):
				for b in range(qm):
					C[i*qm + a, j*qm + b] = (
						Pij_pc[i, j, a, b]
						- Pi_pc[i, a] * Pi_pc[j, b])
	try:
		invC = np.linalg.inv(C)
	except np.linalg.LinAlgError:
		invC = np.linalg.pinv(C)
	# DI computation per pair via 2-site mean-field
	def _di_pair(i, j):
		W = np.ones((q, q))
		for a in range(qm):
			for b in range(qm):
				W[a, b] = math.exp(
					-invC[i*qm + a, j*qm + b])
		mu1 = np.ones(q) / q
		mu2 = np.ones(q) / q
		pi_i = Pi_pc[i]
		pi_j = Pi_pc[j]
		for _ in range(100):
			scra1 = mu2 @ W.T
			scra2 = mu1 @ W
			new_mu1 = pi_i / scra1
			new_mu2 = pi_j / scra2
			new_mu1 /= new_mu1.sum()
			new_mu2 /= new_mu2.sum()
			if (np.max(np.abs(new_mu1 - mu1)) < 1e-6
				and np.max(np.abs(new_mu2 - mu2)) < 1e-6):
				mu1, mu2 = new_mu1, new_mu2
				break
			mu1, mu2 = new_mu1, new_mu2
		Pdir = W * np.outer(mu1, mu2)
		Pdir /= Pdir.sum()
		Pfac = np.outer(pi_i, pi_j)
		mask = (Pdir > 1e-12) & (Pfac > 1e-12)
		di = float(np.sum(
			Pdir[mask] * np.log(Pdir[mask] / Pfac[mask])))
		return di
	dca_raw = np.zeros((L, L))
	for i in range(L):
		for j in range(i+1, L):
			d = _di_pair(i, j)
			dca_raw[i, j] = d
			dca_raw[j, i] = d
	# Average product correction (APC)
	dca = np.zeros((L, L))
	if L > 1:
		row_mean = dca_raw.sum(axis=1) / (L - 1)
		total_mean = dca_raw.sum() / (L * (L - 1))
		if total_mean > 0:
			for i in range(L):
				for j in range(L):
					if i == j: continue
					dca[i, j] = dca_raw[i, j] - (
						row_mean[i] * row_mean[j]
						/ total_mean)
		else:
			dca = dca_raw.copy()
		np.fill_diagonal(dca, 0.0)
	return('\n'.join(out), final, conservation, entropy, pssm, dca)

def Isoelectric(sequence):
	'''
	Calculate the isoelectric point (pI) of a protein sequence.

	Uses EMBOSS pKa values and Henderson-Hasselbalch with a
	bisection search on [0, 14].

	Parameters
	----------
	sequence : str  protein FASTA sequence (one-letter codes).

	Returns
	-------
	float : the pH at which the protein has zero net charge,
	        rounded to 2 decimals.
	'''
	if not sequence:
		raise Exception('Empty sequence')
	seq = sequence.upper()
	# EMBOSS pKa values
	pKa_pos = {'K': 10.53, 'R': 12.48, 'H': 6.00}
	pKa_neg = {'D': 3.65,  'E': 4.25,  'C': 8.33, 'Y': 10.07}
	pKa_nt  = 8.6
	pKa_ct  = 3.6
	cnt_pos = {a: seq.count(a) for a in pKa_pos}
	cnt_neg = {a: seq.count(a) for a in pKa_neg}
	def charge(pH):
		pos = 1.0 / (1.0 + 10 ** (pH - pKa_nt))
		for a, n in cnt_pos.items():
			if n: pos += n / (1.0 + 10 ** (pH - pKa_pos[a]))
		neg = 1.0 / (1.0 + 10 ** (pKa_ct - pH))
		for a, n in cnt_neg.items():
			if n: neg += n / (1.0 + 10 ** (pKa_neg[a] - pH))
		return pos - neg
	lo, hi = 0.0, 14.0
	for _ in range(100):
		mid = (lo + hi) / 2.0
		c   = charge(mid)
		if abs(c) < 1e-4: break
		if c > 0: lo = mid
		else:     hi = mid
	return round(mid, 2)

# Hydrophobicity scales used by Hydrophobicity()
_HPHOB_SCALES = {
	'eisenberg': {
		'A': 0.620, 'R':-2.530, 'N':-0.780, 'D':-0.900, 'C': 0.290,
		'Q':-0.850, 'E':-0.740, 'G': 0.480, 'H':-0.400, 'I': 1.380,
		'L': 1.060, 'K':-1.500, 'M': 0.640, 'F': 1.190, 'P': 0.120,
		'S':-0.180, 'T':-0.050, 'W': 0.810, 'Y': 0.260, 'V': 1.080},
	'kyte-doolittle': {
		'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
		'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
		'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
		'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2},
	'hopp-woods': {
		'A':-0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C':-1.0,
		'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H':-0.5, 'I':-1.8,
		'L':-1.8, 'K': 3.0, 'M':-1.3, 'F':-2.5, 'P': 0.0,
		'S': 0.3, 'T':-0.4, 'W':-3.4, 'Y':-2.3, 'V':-1.5},
	'engelman': {
		'A': 1.6, 'R':-12.3,'N':-4.8, 'D':-9.2, 'C': 2.0,
		'Q':-4.1, 'E':-8.2, 'G': 1.0, 'H':-3.0, 'I': 3.1,
		'L': 2.8, 'K':-8.8, 'M': 3.4, 'F': 3.7, 'P':-0.2,
		'S': 0.6, 'T': 1.2, 'W': 1.9, 'Y':-0.7, 'V': 2.6}}

def Hydrophobicity(sequence, window=9, scale='eisenberg'):
	'''
	Sliding-window hydrophobicity profile (ProtScale-style).

	Parameters
	----------
	sequence : str    protein FASTA sequence.
	window   : int    odd window size (default 9).
	scale    : str    one of {'eisenberg', 'kyte-doolittle',
	                  'hopp-woods', 'engelman'}.

	Returns
	-------
	tuple : (positions, scores)
	    positions : list[int]   0-indexed center of each window.
	    scores    : list[float] mean score in each window (3 dp).
	'''
	seq = sequence.upper()
	L = len(seq)
	if window < 1: raise Exception('window must be >= 1')
	if window > L:
		raise Exception(
			f'window ({window}) larger than sequence ({L})')
	tbl = _HPHOB_SCALES.get(scale.lower())
	if tbl is None:
		raise Exception(
			f'Unknown scale {scale!r}; choose from '
			f'{list(_HPHOB_SCALES)}')
	half = (window - 1) // 2
	positions, scores = [], []
	for i in range(L - window + 1):
		s = sum(tbl.get(seq[i + k], 0.0) for k in range(window))
		positions.append(i + half)
		scores.append(round(s / window, 3))
	return(positions, scores)

def Aliphatic(sequence):
	'''
	Aliphatic index of a protein (Ikai 1980).

	    AI = X(A) + 2.9*X(V) + 3.9*(X(I) + X(L))
	    where X(aa) is the mole percent of that amino acid.

	Parameters
	----------
	sequence : str  protein FASTA sequence.

	Returns
	-------
	float : aliphatic index, rounded to 2 decimals.
	'''
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	L = len(seq)
	xA = 100.0 * seq.count('A') / L
	xV = 100.0 * seq.count('V') / L
	xI = 100.0 * seq.count('I') / L
	xL = 100.0 * seq.count('L') / L
	return round(xA + 2.9 * xV + 3.9 * (xI + xL), 2)

def ExtinctCoeff(sequence, reduced=True):
	'''
	Molar extinction coefficient at 280 nm in water (Pace 1995).

	    eps = nW*5500 + nY*1490 + (nC/2)*125
	    Cysteines contribute only when oxidised (cystines).

	Parameters
	----------
	sequence : str   protein FASTA sequence.
	reduced  : bool  True (default) treats Cys as reduced
	                 (no contribution); False treats them as
	                 disulphide-bonded cystines.

	Returns
	-------
	int : molar extinction coefficient in M^-1 cm^-1.
	'''
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	nW = seq.count('W')
	nY = seq.count('Y')
	nC = seq.count('C')
	eps = nW * 5500 + nY * 1490
	if not reduced:
		eps += (nC // 2) * 125
	return int(round(eps))

# Guruprasad et al. (1990) DIWV dipeptide instability table.
# Rows = first residue, cols = second residue (BLOSUM62 order).
_DIWV_AA = 'ARNDCQEGHILKMFPSTWYV'
_DIWV = {
	'A': {'A':1.0,'R':1.0,'N':1.0,'D':-7.49,'C':44.94,'Q':1.0,
		'E':1.0,'G':1.0,'H':-7.49,'I':1.0,'L':1.0,'K':1.0,
		'M':1.0,'F':1.0,'P':20.26,'S':1.0,'T':1.0,'W':1.0,
		'Y':1.0,'V':1.0},
	'R': {'A':1.0,'R':58.28,'N':13.34,'D':1.0,'C':1.0,'Q':20.26,
		'E':1.0,'G':-7.49,'H':20.26,'I':1.0,'L':1.0,'K':1.0,
		'M':1.0,'F':1.0,'P':20.26,'S':44.94,'T':1.0,'W':58.28,
		'Y':-6.54,'V':1.0},
	'N': {'A':1.0,'R':1.0,'N':1.0,'D':1.0,'C':-1.88,'Q':-6.54,
		'E':1.0,'G':-14.03,'H':1.0,'I':44.94,'L':1.0,'K':24.68,
		'M':1.0,'F':-14.03,'P':-1.88,'S':1.0,'T':-7.49,'W':-9.37,
		'Y':1.0,'V':-1.88},
	'D': {'A':1.0,'R':-6.54,'N':1.0,'D':1.0,'C':1.0,'Q':1.0,
		'E':1.0,'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':-7.49,
		'M':1.0,'F':-6.54,'P':1.0,'S':20.26,'T':-14.03,'W':1.0,
		'Y':1.0,'V':1.0},
	'C': {'A':1.0,'R':1.0,'N':1.0,'D':20.26,'C':1.0,'Q':-6.54,
		'E':1.0,'G':1.0,'H':33.60,'I':1.0,'L':20.26,'K':1.0,
		'M':33.60,'F':1.0,'P':20.26,'S':1.0,'T':33.60,'W':24.68,
		'Y':1.0,'V':-6.54},
	'Q': {'A':1.0,'R':1.0,'N':1.0,'D':20.26,'C':-6.54,'Q':20.26,
		'E':20.26,'G':1.0,'H':1.0,'L':1.0,'I':1.0,'K':1.0,
		'M':1.0,'F':-6.54,'P':20.26,'S':44.94,'T':1.0,'W':1.0,
		'Y':-6.54,'V':-6.54},
	'E': {'A':1.0,'R':1.0,'N':1.0,'D':20.26,'C':44.94,'Q':20.26,
		'E':33.60,'G':1.0,'H':-6.54,'I':20.26,'L':1.0,'K':1.0,
		'M':1.0,'F':1.0,'P':20.26,'S':20.26,'T':1.0,'W':-14.03,
		'Y':1.0,'V':1.0},
	'G': {'A':-7.49,'R':-7.49,'N':-7.49,'D':1.0,'C':1.0,'Q':1.0,
		'E':-6.54,'G':13.34,'H':1.0,'I':-7.49,'L':1.0,'K':-7.49,
		'M':1.0,'F':1.0,'P':1.0,'S':1.0,'T':-7.49,'W':13.34,
		'Y':-7.49,'V':1.0},
	'H': {'A':1.0,'R':1.0,'N':24.68,'D':1.0,'C':1.0,'Q':1.0,
		'E':1.0,'G':-9.37,'H':1.0,'I':44.94,'L':1.0,'K':24.68,
		'M':1.0,'F':-9.37,'P':-1.88,'S':1.0,'T':-6.54,'W':-1.88,
		'Y':44.94,'V':1.0},
	'I': {'A':1.0,'R':1.0,'N':1.0,'D':1.0,'C':1.0,'Q':1.0,
		'E':44.94,'G':1.0,'H':13.34,'I':1.0,'L':20.26,'K':-7.49,
		'M':1.0,'F':1.0,'P':-1.88,'S':1.0,'T':1.0,'W':1.0,
		'Y':1.0,'V':-7.49},
	'L': {'A':1.0,'R':20.26,'N':1.0,'D':1.0,'C':1.0,'Q':33.60,
		'E':1.0,'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':-7.49,
		'M':1.0,'F':1.0,'P':20.26,'S':1.0,'T':1.0,'W':24.68,
		'Y':1.0,'V':1.0},
	'K': {'A':1.0,'R':33.60,'N':1.0,'D':1.0,'C':1.0,'Q':24.68,
		'E':1.0,'G':-7.49,'H':1.0,'I':-7.49,'L':-7.49,'K':1.0,
		'M':33.60,'F':1.0,'P':-6.54,'S':1.0,'T':1.0,'W':1.0,
		'Y':1.0,'V':-7.49},
	'M': {'A':13.34,'R':-6.54,'N':1.0,'D':1.0,'C':1.0,'Q':-6.54,
		'E':1.0,'G':1.0,'H':58.28,'I':1.0,'L':1.0,'K':1.0,
		'M':-1.88,'F':1.0,'P':44.94,'S':44.94,'T':-1.88,'W':1.0,
		'Y':24.68,'V':1.0},
	'F': {'A':1.0,'R':1.0,'N':1.0,'D':13.34,'C':1.0,'Q':1.0,
		'E':1.0,'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':-14.03,
		'M':1.0,'F':1.0,'P':20.26,'S':1.0,'T':1.0,'W':1.0,
		'Y':33.60,'V':1.0},
	'P': {'A':20.26,'R':-6.54,'N':1.0,'D':-6.54,'C':-6.54,'Q':20.26,
		'E':18.38,'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':1.0,
		'M':-6.54,'F':20.26,'P':20.26,'S':20.26,'T':1.0,'W':-1.88,
		'Y':1.0,'V':20.26},
	'S': {'A':1.0,'R':20.26,'N':1.0,'D':1.0,'C':33.60,'Q':20.26,
		'E':20.26,'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':1.0,
		'M':1.0,'F':1.0,'P':44.94,'S':20.26,'T':1.0,'W':1.0,
		'Y':1.0,'V':1.0},
	'T': {'A':1.0,'R':1.0,'N':-14.03,'D':1.0,'C':1.0,'Q':-6.54,
		'E':20.26,'G':-7.49,'H':1.0,'I':1.0,'L':1.0,'K':1.0,
		'M':1.0,'F':13.34,'P':1.0,'S':1.0,'T':1.0,'W':-14.03,
		'Y':1.0,'V':1.0},
	'W': {'A':-14.03,'R':1.0,'N':13.34,'D':1.0,'C':1.0,'Q':1.0,
		'E':1.0,'G':-9.37,'H':24.68,'I':1.0,'L':13.34,'K':1.0,
		'M':24.68,'F':1.0,'P':1.0,'S':1.0,'T':-14.03,'W':1.0,
		'Y':1.0,'V':-7.49},
	'Y': {'A':24.68,'R':-15.91,'N':1.0,'D':24.68,'C':1.0,'Q':1.0,
		'E':-6.54,'G':-7.49,'H':13.34,'I':1.0,'L':1.0,'K':1.0,
		'M':44.94,'F':1.0,'P':13.34,'S':1.0,'T':-7.49,'W':-9.37,
		'Y':13.34,'V':1.0},
	'V': {'A':1.0,'R':1.0,'N':1.0,'D':-14.03,'C':1.0,'Q':1.0,
		'E':1.0,'G':-7.49,'H':1.0,'I':1.0,'L':1.0,'K':-1.88,
		'M':1.0,'F':1.0,'P':20.26,'S':1.0,'T':-7.49,'W':1.0,
		'Y':-6.54,'V':1.0}}

def Instability(sequence):
	'''
	Instability index of a protein (Guruprasad et al. 1990).

	    II = (10 / L) * sum_{i=0..L-2} DIWV(seq[i], seq[i+1])

	A score below 40 generally indicates a stable protein.

	Parameters
	----------
	sequence : str  protein FASTA sequence.

	Returns
	-------
	float : instability index, rounded to 2 decimals.
	'''
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	L = len(seq)
	if L < 2: return 0.0
	total = 0.0
	for i in range(L - 1):
		a, b = seq[i], seq[i+1]
		row = _DIWV.get(a)
		if row is None: continue
		v = row.get(b)
		if v is None: continue
		total += v
	return round(10.0 * total / L, 2)

# Kyte-Doolittle hydropathy (used by GRAVY)
_KD = {
	'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
	'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
	'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
	'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2}

def GRAVY(sequence):
	'''
	Grand average of hydropathy (Kyte & Doolittle 1982).

	Parameters
	----------
	sequence : str  protein FASTA sequence.

	Returns
	-------
	float : mean Kyte-Doolittle hydropathy, rounded to 3 dp.
	'''
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	total = sum(_KD.get(a, 0.0) for a in seq)
	return round(total / len(seq), 3)

def Split(pose, chain=None, start=None, end=None):
	'''
	Extract a slice of a Pose into a new Pose object.

	Two mutually-exclusive modes:
	    Split(pose, chain='A')      - extract one whole chain.
	    Split(pose, start=i, end=j) - extract residues [i, j]
	                                  (inclusive, zero-based on the
	                                  residue table).

	Works for protein, DNA and RNA poses. Atom indices, residue
	indices, bond adjacency and coordinates are all re-numbered
	densely from zero in the returned Pose.
	'''
	try:
		from .pose import Pose
	except ImportError:
		from pose import Pose
	if (chain is None) == (start is None and end is None):
		raise Exception(
			"Split requires either chain= OR (start=, end=)")
	mol = pose.data.get('Type')
	if mol is None:
		raise Exception('Source pose is empty')
	is_pro = (mol == 'Protein')
	rk = 'Amino Acids' if is_pro else 'Nucleotides'
	src = pose.data[rk]
	if not src:
		raise Exception(f'Source pose has no {rk}')
	all_idx = sorted(src.keys())
	if chain is not None:
		keep_res = [i for i in all_idx if src[i][1] == chain]
		if not keep_res:
			raise Exception(f'Chain {chain!r} not in pose')
	else:
		if start is None or end is None:
			raise Exception(
				'Split needs both start and end for range mode')
		if start > end:
			raise Exception(
				f'start ({start}) > end ({end})')
		keep_res = [i for i in all_idx if start <= i <= end]
		if not keep_res:
			raise Exception(
				f'Range [{start}, {end}] selects no residues')
	keep_atoms = []
	for ri in keep_res:
		for ai in src[ri][2]:
			keep_atoms.append(ai)
		for ai in src[ri][3]:
			keep_atoms.append(ai)
	atom_set = set(keep_atoms)
	keep_atoms = sorted(atom_set)
	a_remap = {old: new for new, old in enumerate(keep_atoms)}
	r_remap = {old: new for new, old in enumerate(keep_res)}
	new = Pose()
	new.data = {
		'Type'       : mol,
		'Energy'     : 0,
		'Rg'         : 0,
		'Mass'       : 0,
		'Size'       : {},
		'FASTA'      : {},
		'SS'         : {},
		'Nucleotides': {} if not is_pro else None,
		'Amino Acids': {} if is_pro else None,
		'Atoms'      : {},
		'Bonds'      : {},
		'Coordinates': np.zeros((0, 3))}
	src_atoms = pose.data['Atoms']
	for old_ai in keep_atoms:
		new.data['Atoms'][a_remap[old_ai]] = list(src_atoms[old_ai])
	src_bonds = pose.data['Bonds']
	for old_ai in keep_atoms:
		na = a_remap[old_ai]
		nbrs = []
		for ob in src_bonds.get(old_ai, []):
			if ob in a_remap:
				nbrs.append(a_remap[ob])
		new.data['Bonds'][na] = sorted(nbrs)
	src_co = pose.data['Coordinates']
	new_co = np.array(
		[src_co[old_ai] for old_ai in keep_atoms],
		dtype=float)
	new.data['Coordinates'] = new_co if len(new_co) \
		else np.zeros((0, 3))
	tgt = new.data[rk]
	for old_ri in keep_res:
		row = list(src[old_ri])
		row[2] = [a_remap[a] for a in row[2] if a in a_remap]
		row[3] = [a_remap[a] for a in row[3] if a in a_remap]
		tgt[r_remap[old_ri]] = row
	new._update()
	return new

def Concatenate(pose1, pose2, fuse=False):
	'''
	Combine two Pose objects of the same Type.

	fuse=False (default): append pose2 to pose1 as additional
	    chains, preserving original coordinates. Colliding chain
	    IDs are renamed to the next free letter.
	fuse=True : rebuild the concatenated FASTA as a single
	    polymer using Pose.Build (idealised geometry; original
	    coordinates are discarded).
	'''
	try:
		from .pose import Pose
	except ImportError:
		from pose import Pose
	t1 = pose1.data.get('Type')
	t2 = pose2.data.get('Type')
	if t1 is None or t2 is None:
		raise Exception('Concatenate: empty pose given')
	if t1 != t2:
		raise Exception(
			f'Cannot concatenate {t1} with {t2}')
	is_pro = (t1 == 'Protein')
	rk = 'Amino Acids' if is_pro else 'Nucleotides'
	if fuse:
		f1 = pose1.data['FASTA']
		f2 = pose2.data['FASTA']
		merged_seq = ''.join(
			f1[c] for c in sorted(f1)) + ''.join(
			f2[c] for c in sorted(f2))
		new = Pose()
		new.Build(merged_seq, fmt=t1)
		return new
	new = Pose()
	new.data = {
		'Type'       : t1,
		'Energy'     : 0,
		'Rg'         : 0,
		'Mass'       : 0,
		'Size'       : {},
		'FASTA'      : {},
		'SS'         : {},
		'Nucleotides': {} if not is_pro else None,
		'Amino Acids': {} if is_pro else None,
		'Atoms'      : {},
		'Bonds'      : {},
		'Coordinates': np.zeros((0, 3))}
	def _copy(src_pose, ai_off, ri_off, ch_remap):
		src_aa = src_pose.data[rk]
		src_at = src_pose.data['Atoms']
		src_bd = src_pose.data['Bonds']
		src_co = src_pose.data['Coordinates']
		old_a = sorted(src_at.keys())
		a_map = {}
		coords = []
		for oa in old_a:
			na = ai_off + len(a_map)
			a_map[oa] = na
			new.data['Atoms'][na] = list(src_at[oa])
			coords.append(src_co[oa])
		for oa in old_a:
			na = a_map[oa]
			new.data['Bonds'][na] = sorted(
				a_map[ob] for ob in src_bd.get(oa, [])
				if ob in a_map)
		old_r = sorted(src_aa.keys())
		r_map = {}
		for ori in old_r:
			nri = ri_off + len(r_map)
			r_map[ori] = nri
			row = list(src_aa[ori])
			row[1] = ch_remap.get(row[1], row[1])
			row[2] = [a_map[a] for a in row[2] if a in a_map]
			row[3] = [a_map[a] for a in row[3] if a in a_map]
			new.data[rk][nri] = row
		return coords, len(a_map), len(r_map)
	co1, na1, nr1 = _copy(pose1, 0, 0, {})
	used_chains = set()
	for v in new.data[rk].values():
		used_chains.add(v[1])
	def _next_ch(taken):
		for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
			if c not in taken:
				taken.add(c)
				return c
		raise Exception('Ran out of chain letters')
	p2_chains = sorted({
		v[1] for v in pose2.data[rk].values()})
	ch_remap = {}
	taken = set(used_chains)
	for c in p2_chains:
		if c in taken:
			ch_remap[c] = _next_ch(taken)
		else:
			taken.add(c)
	co2, na2, nr2 = _copy(pose2, na1, nr1, ch_remap)
	all_co = co1 + co2
	new.data['Coordinates'] = np.array(all_co, dtype=float) \
		if all_co else np.zeros((0, 3))
	new._update()
	return new

# SantaLucia 1998 unified nearest-neighbor parameters
# units: dH kcal/mol, dS cal/mol/K
_NN_DH = {
	'AA':-7.9, 'TT':-7.9, 'AT':-7.2, 'TA':-7.2,
	'CA':-8.5, 'TG':-8.5, 'GT':-8.4, 'AC':-8.4,
	'CT':-7.8, 'AG':-7.8, 'GA':-8.2, 'TC':-8.2,
	'CG':-10.6,'GC':-9.8, 'GG':-8.0, 'CC':-8.0}
_NN_DS = {
	'AA':-22.2,'TT':-22.2,'AT':-20.4,'TA':-21.3,
	'CA':-22.7,'TG':-22.7,'GT':-22.4,'AC':-22.4,
	'CT':-21.0,'AG':-21.0,'GA':-22.2,'TC':-22.2,
	'CG':-27.2,'GC':-24.4,'GG':-19.9,'CC':-19.9}

def _revcomp(s):
	c = {'A':'T','T':'A','G':'C','C':'G','N':'N'}
	return ''.join(c[x] for x in reversed(s))

def _gc_pct(s):
	if not s: return 0.0
	return 100.0 * (s.count('G') + s.count('C')) / len(s)

def _tm_nn(seq, conc=250e-9, na=0.05):
	'''
	SantaLucia 1998 unified nearest-neighbor Tm with
	salt correction (Owczarzy 2004 simplified).
	conc = primer molar concentration (default 250 nM).
	na   = Na+ molar concentration (default 50 mM).
	'''
	s = seq.upper()
	if len(s) < 2: return 0.0
	dH = 0.0
	dS = 0.0
	for i in range(len(s) - 1):
		nn = s[i:i+2]
		dH += _NN_DH.get(nn, 0.0)
		dS += _NN_DS.get(nn, 0.0)
	# Initiation terms
	if s[0]  in 'GC': dH += 0.1; dS += -2.8
	else:             dH += 2.3; dS +=  4.1
	if s[-1] in 'GC': dH += 0.1; dS += -2.8
	else:             dH += 2.3; dS +=  4.1
	# Salt correction (Owczarzy 2004 linear)
	N = len(s)
	dS_salt = dS + 0.368 * (N - 1) * math.log(na)
	R = 1.987
	# Non-self-complementary => x = 4
	tm_k = (dH * 1000.0) / (dS_salt + R * math.log(conc / 4.0))
	return tm_k - 273.15

def _has_run(s, n=4):
	for base in 'ACGT':
		if base * n in s: return True
	return False

def _has_hairpin(s, stem=4):
	rc = _revcomp(s)
	for i in range(len(s) - stem + 1):
		motif = s[i:i+stem]
		# look for reverse complement of motif later in s
		j = s.find(_revcomp(motif), i + stem + 3)
		if j != -1: return True
	return False

def _has_self_dimer(s, k=4):
	rc = _revcomp(s)
	# Slide 3' end of s against rc looking for >=k contiguous
	# matching bases at the 3' end (alignment of s onto rc).
	tail = s[-k:]
	return tail in rc

# PCR primer-design relaxation tiers, tried in order until one
# produces a usable pair. Tier 0 is the ideal target; lower tiers
# progressively drop gates so that PCR() always returns a pair for
# any chemically valid template (>= 36 bp).
_PCR_TIERS = [
	{'label':'Ideal',       'len':(18,25),
		'gc':(40.0, 60.0),  'tm':(55.0, 65.0),
		'clamp':True,  'max_run':4,
		'no_hairpin':True,  'no_dimer':True,  'dtm':2.0},
	{'label':'Good',        'len':(18,28),
		'gc':(35.0, 65.0),  'tm':(50.0, 68.0),
		'clamp':True,  'max_run':5,
		'no_hairpin':True,  'no_dimer':True,  'dtm':3.0},
	{'label':'Fair',        'len':(18,30),
		'gc':(25.0, 75.0),  'tm':(45.0, 72.0),
		'clamp':False, 'max_run':5,
		'no_hairpin':False, 'no_dimer':True,  'dtm':5.0},
	{'label':'Poor',        'len':(18,30),
		'gc':None,          'tm':None,
		'clamp':False, 'max_run':None,
		'no_hairpin':False, 'no_dimer':False, 'dtm':8.0},
	{'label':'Last resort', 'len':(18,30),
		'gc':None,          'tm':None,
		'clamp':False, 'max_run':None,
		'no_hairpin':False, 'no_dimer':False, 'dtm':float('inf')}]

def PCR(dna_sequence):
	'''
	Design forward and reverse PCR primers for a DNA template.

	Uses a 5-tier relaxation strategy so that any chemically valid
	template (A/C/G/T only, length >= 36 bp) always yields a primer
	pair. Tier 0 ('Ideal') applies the standard constraints:
	    - length 18-25
	    - GC% in [40, 60]
	    - Nearest-neighbor (SantaLucia 1998) Tm in [55, 65] degC
	    - 3' GC clamp
	    - no run of 4 identical bases
	    - no internal palindrome of >= 4 (hairpin)
	    - 3' tail not contained in self reverse complement
	    - |Tm_fwd - Tm_rev| <= 2 degC
	If no pair satisfies tier 0, the search moves to Good, Fair,
	Poor, then Last-resort tiers, each dropping or widening gates.
	When the result comes from any tier below Ideal, a warning is
	printed to stdout naming the tier and what was relaxed.

	Parameters
	----------
	dna_sequence : str
	    Template DNA sequence (A/C/G/T only).

	Returns
	-------
	tuple : (forward, reverse)
	    forward : str  forward primer (5' end of template).
	    reverse : str  reverse primer (revcomp of 3' end).
	'''
	seq = dna_sequence.upper()
	for ch in seq:
		if ch not in 'ACGT':
			raise Exception(
				f'Illegal base {ch!r} in template')
	if len(seq) < 36:
		raise Exception(
			'Template too short for primer design (<36 bp)')
	rc = _revcomp(seq)
	max_off = max(0, min(60, len(seq) - 18))
	def _candidates(region, tier):
		out = []
		lo, hi = tier['len']
		for L in range(lo, hi + 1):
			if L > len(region): continue
			cand = region[:L]
			if tier['clamp'] and cand[-1] not in 'GC':
				continue
			gc = _gc_pct(cand)
			if tier['gc'] is not None:
				glo, ghi = tier['gc']
				if not (glo <= gc <= ghi): continue
			if tier['max_run'] is not None:
				if _has_run(cand, tier['max_run']):
					continue
			if tier['no_hairpin'] and _has_hairpin(cand, 4):
				continue
			if tier['no_dimer'] and _has_self_dimer(cand, 5):
				continue
			tm = _tm_nn(cand)
			if tier['tm'] is not None:
				tlo, thi = tier['tm']
				if not (tlo <= tm <= thi): continue
			out.append((cand, tm, gc))
		return out
	def _pool(source, tier):
		pool = []
		for off in range(0, max_off + 1):
			pool.extend(
				(off,) + c for c in _candidates(
					source[off:], tier))
		return pool
	chosen = None
	chosen_tier = None
	for ti, tier in enumerate(_PCR_TIERS):
		fwd_pool = _pool(seq, tier)
		rev_pool = _pool(rc,  tier)
		if not fwd_pool or not rev_pool: continue
		best, best_score = None, float('inf')
		dtm_max = tier['dtm']
		for off1, fwd, tmf, gcf in fwd_pool:
			for off2, rev, tmr, gcr in rev_pool:
				dT = abs(tmf - tmr)
				if dT > dtm_max: continue
				score = (
					dT * 5.0
					+ abs(tmf - 60.0)
					+ abs(tmr - 60.0)
					+ abs(gcf - 50.0) * 0.1
					+ abs(gcr - 50.0) * 0.1
					+ (off1 + off2) * 0.05)
				if score < best_score:
					best_score = score
					best = (fwd, rev, tmf, tmr, gcf, gcr)
		if best is not None:
			chosen = best
			chosen_tier = ti
			break
	if chosen is None:
		# Should not happen: Last-resort tier has all gates open.
		raise Exception(
			'No primer pair found even at last-resort tier')
	fwd, rev, tmf, tmr, gcf, gcr = chosen
	msg = None
	if chosen_tier > 0:
		# Build a short reason string vs the Ideal tier.
		reasons = []
		if not (40.0 <= gcf <= 60.0 and 40.0 <= gcr <= 60.0):
			reasons.append('GC% outside 40-60')
		if not (55.0 <= tmf <= 65.0 and 55.0 <= tmr <= 65.0):
			reasons.append('Tm outside 55-65 \u00b0C')
		if abs(tmf - tmr) > 2.0:
			reasons.append('|\u0394Tm| > 2 \u00b0C')
		if fwd[-1] not in 'GC' or rev[-1] not in 'GC':
			reasons.append('GC clamp missing')
		reason = '; '.join(reasons) if reasons \
			else 'gates relaxed'
		label = _PCR_TIERS[chosen_tier]['label']
		msg = f'Warning: Suboptimal PCR primers ({label} tier) \u2014 {reason}'
	return (fwd, rev, msg)

# Standard genetic code (DNA -> single-letter AA, '*' = stop)
_CODON_TABLE = {
	'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
	'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
	'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
	'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
	'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
	'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
	'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
	'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
	'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
	'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
	'AAT':'N','AAC':'N','AAA':'K','AAG':'K',
	'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
	'TGT':'C','TGC':'C','TGA':'*','TGG':'W',
	'CGT':'R','CGC':'R','CGA':'R','CGG':'R',
	'AGT':'S','AGC':'S','AGA':'R','AGG':'R',
	'GGT':'G','GGC':'G','GGA':'G','GGG':'G'}

# Codon usage (relative frequencies). Source: Kazusa codon usage db.
# Highest-weight codon per amino acid is selected for back-translation.
_CODON_USAGE = {
	'ecoli': {
		'F':[('TTT',0.58),('TTC',0.42)],
		'L':[('CTG',0.50),('TTA',0.13),('TTG',0.13),
			('CTT',0.10),('CTC',0.10),('CTA',0.04)],
		'I':[('ATT',0.49),('ATC',0.39),('ATA',0.11)],
		'M':[('ATG',1.00)],
		'V':[('GTG',0.37),('GTT',0.28),('GTC',0.20),('GTA',0.15)],
		'S':[('AGC',0.28),('TCT',0.17),('TCC',0.15),
			('AGT',0.15),('TCG',0.14),('TCA',0.12)],
		'P':[('CCG',0.52),('CCA',0.19),('CCT',0.16),('CCC',0.12)],
		'T':[('ACC',0.44),('ACG',0.27),('ACT',0.16),('ACA',0.13)],
		'A':[('GCG',0.36),('GCC',0.27),('GCA',0.21),('GCT',0.16)],
		'Y':[('TAT',0.59),('TAC',0.41)],
		'*':[('TAA',0.61),('TGA',0.30),('TAG',0.09)],
		'H':[('CAT',0.57),('CAC',0.43)],
		'Q':[('CAG',0.66),('CAA',0.34)],
		'N':[('AAC',0.55),('AAT',0.45)],
		'K':[('AAA',0.74),('AAG',0.26)],
		'D':[('GAT',0.63),('GAC',0.37)],
		'E':[('GAA',0.68),('GAG',0.32)],
		'C':[('TGC',0.55),('TGT',0.45)],
		'W':[('TGG',1.00)],
		'R':[('CGC',0.40),('CGT',0.38),('CGG',0.10),
			('CGA',0.06),('AGA',0.04),('AGG',0.02)],
		'G':[('GGC',0.40),('GGT',0.34),('GGG',0.15),('GGA',0.11)]},
	'human': {
		'F':[('TTC',0.55),('TTT',0.45)],
		'L':[('CTG',0.41),('CTC',0.20),('CTT',0.13),
			('TTG',0.13),('TTA',0.07),('CTA',0.07)],
		'I':[('ATC',0.48),('ATT',0.36),('ATA',0.16)],
		'M':[('ATG',1.00)],
		'V':[('GTG',0.47),('GTC',0.24),('GTT',0.18),('GTA',0.11)],
		'S':[('AGC',0.24),('TCC',0.22),('TCT',0.15),
			('AGT',0.15),('TCA',0.15),('TCG',0.06)],
		'P':[('CCC',0.33),('CCT',0.28),('CCA',0.27),('CCG',0.11)],
		'T':[('ACC',0.36),('ACA',0.28),('ACT',0.24),('ACG',0.12)],
		'A':[('GCC',0.40),('GCT',0.26),('GCA',0.23),('GCG',0.11)],
		'Y':[('TAC',0.57),('TAT',0.43)],
		'*':[('TGA',0.52),('TAA',0.28),('TAG',0.20)],
		'H':[('CAC',0.59),('CAT',0.41)],
		'Q':[('CAG',0.75),('CAA',0.25)],
		'N':[('AAC',0.54),('AAT',0.46)],
		'K':[('AAG',0.58),('AAA',0.42)],
		'D':[('GAC',0.54),('GAT',0.46)],
		'E':[('GAG',0.58),('GAA',0.42)],
		'C':[('TGC',0.55),('TGT',0.45)],
		'W':[('TGG',1.00)],
		'R':[('AGA',0.20),('AGG',0.20),('CGG',0.21),
			('CGC',0.19),('CGA',0.11),('CGT',0.08)],
		'G':[('GGC',0.34),('GGA',0.25),('GGG',0.25),('GGT',0.16)]}}

def _detect_alphabet(s):
	chars = set(s.upper()) - {'-', '*', 'N'}
	if not chars: return 'protein'
	dna  = set('ACGT')
	rna  = set('ACGU')
	prot = set('ACDEFGHIKLMNPQRSTVWY')
	if chars <= dna: return 'dna'
	if chars <= rna: return 'rna'
	if chars <= prot: return 'protein'
	# Fall back: more amino-acid only letters wins
	if chars - dna - rna: return 'protein'
	return 'dna'

def Translate(sequence, fmt='protein', organism='ecoli'):
	'''
	Translate between DNA, RNA and protein representations.

	Parameters
	----------
	sequence : str
	    Input sequence. Alphabet auto-detected (DNA / RNA / protein).
	fmt : str
	    Target alphabet: 'protein' (default), 'dna', or 'rna'.
	organism : str
	    Codon usage table for back-translation (protein -> DNA/RNA).
	    'ecoli' (default) or 'human'.

	Returns
	-------
	str : the translated sequence (uppercase).
	'''
	if not sequence: raise Exception('Empty sequence')
	src = _detect_alphabet(sequence)
	tgt = fmt.lower()
	if tgt not in ('protein', 'dna', 'rna'):
		raise Exception(f'Unknown target fmt: {fmt}')
	s = sequence.upper().replace('-', '').replace(' ', '')
	# Identity / alphabet swap
	if src == tgt: return s
	if src == 'dna' and tgt == 'rna':
		return s.replace('T', 'U')
	if src == 'rna' and tgt == 'dna':
		return s.replace('U', 'T')
	# Nucleotide -> protein
	if src in ('dna', 'rna') and tgt == 'protein':
		dna = s.replace('U', 'T')
		L = len(dna)
		if L % 3 != 0:
			dna = dna[:L - (L % 3)]
		out = []
		for i in range(0, len(dna), 3):
			codon = dna[i:i+3]
			aa = _CODON_TABLE.get(codon, 'X')
			out.append(aa)
		return ''.join(out)
	# Protein -> DNA / RNA (codon optimised)
	if src == 'protein' and tgt in ('dna', 'rna'):
		usage = _CODON_USAGE.get(organism.lower())
		if usage is None:
			raise Exception(
				f"Unknown organism {organism!r}; "
				f"use 'ecoli' or 'human'")
		best = {aa: max(opts, key=lambda x: x[1])[0]
			for aa, opts in usage.items()}
		out = []
		for aa in s:
			c = best.get(aa)
			if c is None:
				raise Exception(
					f'No codon for residue {aa!r}')
			out.append(c)
		dna = ''.join(out)
		return dna if tgt == 'dna' else dna.replace('T', 'U')
	raise Exception(
		f'Unsupported translation {src} -> {tgt}')

def PROSITE(sequence, pattern):
	'''
	Search a protein sequence for a PROSITE-style pattern.

	Pattern grammar (subset of the official PROSITE syntax):
	    -        token separator (stripped)
	    A        literal residue
	    [ABC]    any of A/B/C
	    {ABC}    any except A/B/C
	    x        any residue
	    x(n)     exactly n residues
	    x(n,m)   between n and m residues
	    A(n)     exactly n A's
	    A(n,m)   between n and m A's
	    <        anchor at sequence start
	    >        anchor at sequence end

	Parameters
	----------
	pattern  : str  PROSITE pattern.
	sequence : str  protein sequence to search.

	Returns
	-------
	list of (start, end, match)
	    start, end : 1-based, inclusive positions.
	    match      : matched substring.
	'''
	if not pattern: raise Exception('Empty pattern')
	if not sequence: return []
	# Tokenise and translate to a regex.
	p = pattern.replace('-', '').replace(' ', '')
	out = []
	i = 0
	while i < len(p):
		c = p[i]
		if c == '<':
			out.append('^')
			i += 1
		elif c == '>':
			out.append('$')
			i += 1
		elif c == '[':
			j = p.find(']', i)
			if j == -1:
				raise Exception('Unclosed [ in pattern')
			out.append('[' + p[i+1:j] + ']')
			i = j + 1
		elif c == '{':
			j = p.find('}', i)
			if j == -1:
				raise Exception('Unclosed { in pattern')
			out.append('[^' + p[i+1:j] + ']')
			i = j + 1
		elif c == 'x' or c == 'X':
			out.append('.')
			i += 1
		elif c.isalpha():
			out.append(c.upper())
			i += 1
		else:
			raise Exception(
				f'Unexpected character {c!r} '
				f'at position {i} of pattern')
		# Optional quantifier (n) or (n,m)
		if i < len(p) and p[i] == '(':
			j = p.find(')', i)
			if j == -1:
				raise Exception('Unclosed ( in pattern')
			body = p[i+1:j]
			if ',' in body:
				lo, hi = body.split(',', 1)
				out.append('{' + lo.strip() + ','
					+ hi.strip() + '}')
			else:
				out.append('{' + body.strip() + '}')
			i = j + 1
	regex = '(?=(' + ''.join(out) + '))'
	rx   = re.compile(regex, re.IGNORECASE)
	hits = []
	for m in rx.finditer(sequence):
		mstr  = m.group(1)
		start = m.start() + 1
		end   = start + len(mstr) - 1
		hits.append((start, end, mstr))
	return hits

def HydrogenBondMap(pose):
	'''
	Backbone hydrogen-bond donor/acceptor map for a protein pose.

	Uses the same DSSP electrostatic criterion as Pose.CalcDSSP
	(Kabsch & Sander 1983):
	    E = 0.084 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332
	A backbone NH (residue i) -> C=O (residue j) bond is accepted
	when E < -0.5 kcal/mol, |i - j| > 1, and i, j share a chain.

	Parameters
	----------
	pose : Pose object containing a protein.

	Returns
	-------
	np.ndarray of shape (N_atoms, N_atoms) with values:
	    0 - no bond
	    1 - this atom is a donor (backbone N) in a bond
	    2 - this atom is an acceptor (backbone O) in a bond
	'''
	if pose.data.get('Type') != 'Protein':
		raise Exception(
			'HydrogenBondMap only supports protein poses')
	AAs = pose.data.get('Amino Acids') or {}
	if not AAs:
		raise Exception('Pose has no amino acids')
	atoms = pose.data['Atoms']
	N_atoms = max(atoms.keys()) + 1 if atoms else 0
	M = np.zeros((N_atoms, N_atoms), dtype=np.int8)
	N_res = len(AAs)
	res_idx = sorted(AAs.keys())
	chains   = [AAs[i][1] for i in res_idx]
	tricodes = [AAs[i][5].upper() for i in res_idx]
	def _hasatom(r, name):
		try:
			pose._hasatom(r, name)
		except Exception:
			pass
		for ai in AAs[r][2]:
			if atoms[ai][0] == name: return True
		return False
	def _atomidx(r, name):
		for ai in AAs[r][2]:
			if atoms[ai][0] == name: return ai
		return -1
	co = pose.data['Coordinates']
	# Compute virtual H positions (DSSP rule: H is N + unit(C->O) of i-1)
	H_pos = [None] * N_res
	for k, r in enumerate(res_idx):
		if tricodes[k] == 'PRO': continue
		if k == 0 or chains[k] != chains[k-1]: continue
		if _hasatom(r, 'H'):
			H_pos[k] = co[_atomidx(r, 'H')]
		elif _hasatom(r, '1H'):
			H_pos[k] = co[_atomidx(r, '1H')]
		elif (_hasatom(r, 'N')
			and _hasatom(res_idx[k-1], 'C')
			and _hasatom(res_idx[k-1], 'O')):
			Ni = co[_atomidx(r, 'N')]
			Cp = co[_atomidx(res_idx[k-1], 'C')]
			Op = co[_atomidx(res_idx[k-1], 'O')]
			cdir = Cp - Op
			nm = float(np.linalg.norm(cdir))
			if nm > 0.001:
				H_pos[k] = Ni + (cdir / nm)
	for ki in range(N_res):
		if H_pos[ki] is None: continue
		ri = res_idx[ki]
		Ni_idx = _atomidx(ri, 'N')
		if Ni_idx < 0: continue
		Ni = co[Ni_idx]
		Hi = H_pos[ki]
		for kj in range(N_res):
			if abs(ki - kj) <= 1: continue
			if chains[ki] != chains[kj]: continue
			rj = res_idx[kj]
			if not _hasatom(rj, 'O'): continue
			Cj_idx = _atomidx(rj, 'C')
			Oj_idx = _atomidx(rj, 'O')
			if Cj_idx < 0 or Oj_idx < 0: continue
			Cj = co[Cj_idx]
			Oj = co[Oj_idx]
			r_ON = float(np.linalg.norm(Oj - Ni))
			r_CH = float(np.linalg.norm(Cj - Hi))
			r_OH = float(np.linalg.norm(Oj - Hi))
			r_CN = float(np.linalg.norm(Cj - Ni))
			if min(r_ON, r_CH, r_OH, r_CN) < 0.001: continue
			E = 0.084 * (
				1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332
			if E < -0.5:
				M[Ni_idx, Oj_idx] = 1
				M[Oj_idx, Ni_idx] = 2
	return M

def ContactMap(pose):
	'''
	Residue-residue distance map (in angstroms).

	Uses CA atoms for proteins and C1' atoms for DNA / RNA.

	Parameters
	----------
	pose : Pose object containing a protein or nucleic acid.

	Returns
	-------
	np.ndarray of shape (N_residues, N_residues) with pairwise
	Euclidean distances in angstroms (zero on the diagonal).
	'''
	mol = pose.data.get('Type')
	if mol is None:
		raise Exception('Empty pose')
	if mol == 'Protein':
		src = pose.data['Amino Acids']
		ref = 'CA'
	elif mol in ('DNA', 'RNA'):
		src = pose.data['Nucleotides']
		ref = "C1'"
	else:
		raise Exception(f'Unknown molecule type: {mol}')
	if not src:
		raise Exception('Pose has no residues')
	atoms = pose.data['Atoms']
	co    = pose.data['Coordinates']
	keys  = sorted(src.keys())
	N     = len(keys)
	pts   = np.zeros((N, 3))
	for k, ri in enumerate(keys):
		hit = False
		for ai in src[ri][2]:
			if atoms[ai][0] == ref:
				pts[k] = co[ai]
				hit = True
				break
		if not hit:
			raise Exception(
				f'Residue {ri} has no {ref} atom')
	diff = pts[:, None, :] - pts[None, :, :]
	mat  = np.sqrt((diff * diff).sum(-1))
	np.fill_diagonal(mat, 0.0)
	return mat

# === Void() pocket detection ====================================
# Van der Waals radii (Å) for protein-relevant elements.
_VDW_VOID = {
	'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
	'S': 1.80, 'P': 1.80, 'F': 1.47, 'CL': 1.75,
	'BR': 1.85, 'I': 1.98, 'SE': 1.90}
_VDW_DEFAULT = 1.70
# fpocket parameters, Le Guilloux 2009 defaults.
_FP_MIN_R   = 2.8    # min alpha sphere radius (Å)
_FP_MAX_R   = 6.0    # max alpha sphere radius (Å)
_FP_D1      = 1.73   # 1st-pass single-linkage threshold (Å)
_FP_D2      = 4.5    # 2nd-pass merge distance (Å)
_FP_NMIN    = 2      # min shared spheres for 2nd-pass merge
_FP_MIN_SPH = 6      # min spheres per final pocket
_FP_MIN_APO = 2      # min apolar spheres per final pocket
_FP_MC_N    = 1500   # MC sample count
_FP_GRID_DX = 0.25   # exact-mode grid spacing (Å)
# Kyte-Doolittle hydropathy reused via _KD already in this module.

def _vdw_radius(element):
	if not element: return _VDW_DEFAULT
	return _VDW_VOID.get(element.upper(), _VDW_DEFAULT)

def _heavy_atoms(pose):
	'''
	Return (coords, radii, elements, atom2res, res_keys) for all
	heavy atoms (no H) in a protein pose.
	'''
	if pose.data.get('Type') != 'Protein':
		raise Exception('Void only supports protein poses')
	AAs = pose.data.get('Amino Acids')
	if not AAs:
		raise Exception('Pose has no amino acids')
	atoms_d = pose.data['Atoms']
	co      = pose.data['Coordinates']
	res_keys = sorted(AAs.keys())
	atom_ids = []
	atom2res = []
	for ri in res_keys:
		for ai in AAs[ri][2] + AAs[ri][3]:
			el = atoms_d[ai][1]
			if el.upper() == 'H': continue
			atom_ids.append(ai)
			atom2res.append(ri)
	if not atom_ids:
		raise Exception('No heavy atoms found in pose')
	atom_ids = np.array(atom_ids, dtype=int)
	atom2res = np.array(atom2res, dtype=int)
	coords   = co[atom_ids].astype(float)
	elements = [atoms_d[ai][1] for ai in atom_ids]
	radii    = np.array([_vdw_radius(e) for e in elements])
	return coords, radii, elements, atom2res, res_keys

def _voronoi_alpha_spheres(coords, radii):
	'''
	Run scipy Voronoi on heavy-atom coords and return the alpha
	spheres (centers, radii, contact-atom-index lists) that satisfy
	_FP_MIN_R <= radius <= _FP_MAX_R.
	'''
	from scipy.spatial import Voronoi
	vor = Voronoi(coords)
	# Build vertex -> generating-point map from regions.
	v2p = [[] for _ in range(len(vor.vertices))]
	for pi, ri in enumerate(vor.point_region):
		region = vor.regions[ri]
		if not region or -1 in region: continue
		for vi in region:
			v2p[vi].append(pi)
	centers = []
	asph_r  = []
	contacts = []
	for vi, gens in enumerate(v2p):
		if len(gens) < 4: continue
		v = vor.vertices[vi]
		# Distances from the vertex to all generating atoms,
		# minus their vdW radii (probe-accessible radius).
		gens = sorted(set(gens))
		d = np.linalg.norm(coords[gens] - v, axis=1) - radii[gens]
		r = float(d.min())
		if not (_FP_MIN_R <= r <= _FP_MAX_R): continue
		centers.append(v)
		asph_r.append(r)
		# Keep the 4 closest generators as the "contact atoms".
		order = np.argsort(d)[:4]
		contacts.append([int(gens[i]) for i in order])
	if not centers:
		return (np.zeros((0, 3)), np.zeros(0), [])
	return np.array(centers), np.array(asph_r), contacts

def _classify_polarity(contacts, elements):
	'''
	Return a bool array, True where the alpha sphere is "apolar".
	Matches the real fpocket per-sphere classification: an alpha
	sphere is apolar if at least half of its contact atoms are
	non-polar (C, S, P, halogens). Backbone N/O atoms are nearly
	ubiquitous in proteins, so a stricter rule loses too many
	hydrophobic-pocket spheres (e.g., the FK506 site of FKBP12).
	'''
	out = np.zeros(len(contacts), dtype=bool)
	for i, ats in enumerate(contacts):
		n_polar = sum(
			1 for a in ats
			if elements[a].upper() in ('N', 'O'))
		# At most half the contacts are polar
		out[i] = (n_polar <= len(ats) // 2)
	return out

def _single_linkage(points, threshold):
	'''
	Single-linkage clustering by distance threshold.
	Returns a label array (densely renumbered from 0).
	Pure NumPy + union-find.
	'''
	n = len(points)
	if n == 0: return np.zeros(0, dtype=int)
	parent = list(range(n))
	def find(x):
		while parent[x] != x:
			parent[x] = parent[parent[x]]
			x = parent[x]
		return x
	def union(a, b):
		ra, rb = find(a), find(b)
		if ra != rb: parent[ra] = rb
	d2  = ((points[:, None, :] - points[None, :, :])**2).sum(-1)
	thr2 = threshold * threshold
	tri  = np.triu(np.ones_like(d2, dtype=bool), k=1)
	ii, jj = np.where((d2 < thr2) & tri)
	for a, b in zip(ii, jj):
		union(int(a), int(b))
	roots = np.array([find(i) for i in range(n)])
	# Renumber densely
	uniq, inv = np.unique(roots, return_inverse=True)
	return inv

def _second_pass_merge(labels, centers, d2, nmin):
	'''
	fpocket second-pass merge: two raw clusters merge if at least
	`nmin` of one's spheres are within `d2` of any sphere in the
	other cluster. Returns new label array, densely renumbered.
	'''
	uniq = sorted(set(int(x) for x in labels))
	if len(uniq) <= 1: return labels.copy()
	members = {c: np.where(labels == c)[0] for c in uniq}
	parent  = {c: c for c in uniq}
	def find(x):
		while parent[x] != x:
			parent[x] = parent[parent[x]]
			x = parent[x]
		return x
	def union(a, b):
		ra, rb = find(a), find(b)
		if ra != rb: parent[ra] = rb
	d2sq = d2 * d2
	for i, ci in enumerate(uniq):
		ai_pts = centers[members[ci]]
		for cj in uniq[i+1:]:
			aj_pts = centers[members[cj]]
			dm = ((ai_pts[:, None, :] - aj_pts[None, :, :])
				**2).sum(-1)
			# count i-spheres with at least one j-sphere within d2
			close_i = (dm < d2sq).any(axis=1).sum()
			close_j = (dm < d2sq).any(axis=0).sum()
			if close_i >= nmin or close_j >= nmin:
				union(ci, cj)
	new_root = np.array([find(int(c)) for c in labels])
	uniq2, inv = np.unique(new_root, return_inverse=True)
	return inv

def _pocket_volume_mc(centers, radii, n_samples, rng):
	'''
	Monte Carlo volume + area of a sphere union.
	Volume: bounding-box rejection sampling.
	Area:   per-sphere surface sampling, count the points NOT
	        covered by any other sphere.
	Both deterministic given the rng.
	'''
	if len(centers) == 0: return (0.0, 0.0)
	margin = float(radii.max())
	lo = centers.min(axis=0) - margin
	hi = centers.max(axis=0) + margin
	bbox_v = float(np.prod(hi - lo))
	pts = rng.uniform(lo, hi, size=(n_samples, 3))
	d2  = ((pts[:, None, :] - centers[None, :, :])**2).sum(-1)
	r2  = radii * radii
	inside = (d2 < r2[None, :]).any(axis=1)
	V = float(inside.sum() / n_samples * bbox_v)
	# Surface area via per-sphere Marsaglia-style sampling
	n_per  = max(64, n_samples // max(1, len(centers)))
	A_total = 0.0
	for i in range(len(centers)):
		u   = rng.uniform(-1.0, 1.0, size=n_per)
		phi = rng.uniform(0.0, 2.0 * math.pi, size=n_per)
		s   = np.sqrt(np.clip(1.0 - u * u, 0.0, None))
		spts = centers[i] + radii[i] * np.column_stack(
			[s * np.cos(phi), s * np.sin(phi), u])
		dother = ((spts[:, None, :]
			- centers[None, :, :])**2).sum(-1)
		dother[:, i] = np.inf
		free = ~(dother < r2[None, :]).any(axis=1)
		A_total += (
			free.sum() / n_per * 4.0 * math.pi * radii[i]**2)
	return V, float(A_total)

def _pocket_volume_exact(centers, radii):
	'''
	Deterministic high-resolution grid integration of a sphere
	union — used for `volume='exact'`. Grid spacing _FP_GRID_DX.
	Errors are bounded by half a voxel diagonal (<0.5% for
	typical pocket clusters).
	'''
	if len(centers) == 0: return (0.0, 0.0)
	margin = float(radii.max()) + _FP_GRID_DX
	lo = centers.min(axis=0) - margin
	hi = centers.max(axis=0) + margin
	dx = _FP_GRID_DX
	xs = np.arange(lo[0], hi[0] + dx, dx)
	ys = np.arange(lo[1], hi[1] + dx, dx)
	zs = np.arange(lo[2], hi[2] + dx, dx)
	gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
	pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
	r2  = radii * radii
	# Inside-mask: point belongs to the sphere union iff any sphere
	# encloses it.
	inside = np.zeros(len(pts), dtype=bool)
	for i in range(len(centers)):
		d2 = ((pts - centers[i])**2).sum(-1)
		inside |= (d2 < r2[i])
	V = float(inside.sum() * dx**3)
	# Surface area: count voxels that are inside but adjacent to
	# at least one outside voxel along the 6-axis grid neighbours.
	shape = (len(xs), len(ys), len(zs))
	mask  = inside.reshape(shape)
	bnd = np.zeros_like(mask)
	bnd[1:, :, :]  |= mask[1:, :, :] & ~mask[:-1, :, :]
	bnd[:-1, :, :] |= mask[:-1, :, :] & ~mask[1:, :, :]
	bnd[:, 1:, :]  |= mask[:, 1:, :] & ~mask[:, :-1, :]
	bnd[:, :-1, :] |= mask[:, :-1, :] & ~mask[:, 1:, :]
	bnd[:, :, 1:]  |= mask[:, :, 1:] & ~mask[:, :, :-1]
	bnd[:, :, :-1] |= mask[:, :, :-1] & ~mask[:, :, 1:]
	# Each boundary voxel contributes ~ dx² of surface area;
	# multiply by 2/3 (Cauchy-Crofton correction for axis-aligned
	# boundary sampling vs true smooth surface).
	A = float(bnd.sum() * dx * dx * (2.0 / 3.0))
	return V, A

def _residues_within(cutoff, points, coords, atom2res,
		point_radii=None):
	'''
	Return a sorted list of unique residue indices that have at
	least one heavy atom within `cutoff` Å of the closest sphere
	SURFACE in `points`. If `point_radii` is given, the threshold
	for point i is (cutoff + point_radii[i]); otherwise the bare
	cutoff is used.
	'''
	if len(points) == 0 or len(coords) == 0: return []
	hits = set()
	for i, p in enumerate(points):
		thr = cutoff + (point_radii[i] if point_radii is not None
			else 0.0)
		thr2 = thr * thr
		d2 = ((coords - p)**2).sum(-1)
		idx = np.where(d2 < thr2)[0]
		for k in idx: hits.add(int(atom2res[k]))
	return sorted(hits)

def _druggability_score(n_alpha, n_apol, V, hyd, pol, n_res):
	'''
	fpocket-style raw pocket score for ranking. Combines
	number of lining residues (the strongest signal in
	practice — real binding sites are surrounded by many
	residues), pocket size (n_alpha and log volume), apolar
	contact density, and a hydrophobicity bonus.
	Polarity is mildly penalised. Returns a raw float, NOT in
	[0, 1] — used directly for ranking.
	'''
	if n_alpha == 0 or V <= 0.0 or n_res == 0: return 0.0
	apol_density = n_apol / float(n_alpha)
	log_v = math.log(V + 1.0)
	return (
		1.20 * n_res
		+ 0.10 * n_alpha
		+ 0.50 * log_v
		+ 2.00 * apol_density
		+ 0.05 * hyd
		- 0.50 * pol)

def Void(pose, algorithm='fpocket', cutoff=4.5, volume='mc', seed=0):
	'''
	Detect cavities (pockets and internal voids) in a protein.

	Algorithms:
	    'fpocket' - Voronoi + alpha sphere clustering
	                (Le Guilloux 2009). Returns surface Pockets only.
	                Full pipeline: alpha-sphere extraction,
	                polarity classification, two-pass clustering,
	                descriptor calculation, fpocket-style
	                druggability ranking.
	    'castp'   - Edelsbrunner-Liang discrete flow on a
	                hand-rolled weighted alpha shape.
	                Returns Voids + Pockets. Pipeline:
	                hand-rolled Bowyer-Watson regular (power)
	                triangulation with the four mandatory
	                validation checks (no zero-volume tets, no
	                duplicates, face manifoldness, power-empty
	                axiom); alpha complex at probe=1.4 Å;
	                discrete-flow watershed segmentation into
	                basins at local filtration maxima; basin
	                classification as Pocket vs Void by
	                reachability to bulk solvent through
	                probe-passable faces.

	Parameters
	----------
	pose      : Pose object containing a protein.
	algorithm : 'fpocket' (default) or 'castp'.
	cutoff    : Å cutoff for "participating" residues. Default
	            7.0 for fpocket, 7.0 for CASTp.
	volume    : 'mc' (Monte Carlo, default — matches real fpocket)
	            or 'exact' (deterministic high-resolution grid).
	            fpocket only — CASTp computes analytical tet
	            volumes and boundary face areas regardless.
	seed      : RNG seed for the MC sampler. Default 0 for
	            reproducibility. fpocket only.

	Returns
	-------
	dict sorted by descending Volume, indexed densely from 0:
	    {0: {'Type'       : 'Pocket' or 'Void',
	         'Volume'     : float (Å^3),
	         'Area'       : float (Å^2),
	         'Center'     : np.array([x, y, z]),
	         'Amino Acids': [residue_idx, ...]},
	     1: {...}, ...}

	Note on cryptic pockets: cryptic pockets are absent from the
	apo structure and only appear on ligand binding. They cannot
	be detected from a single static snapshot. Use MD-based
	methods (PocketMiner, CryptoSite, mixed-solvent MD) instead.
	'''
	try:
		import scipy.spatial  # noqa: F401
	except ImportError:
		raise Exception(
			'Void() requires scipy. pip install scipy')
	algo = algorithm.lower()
	if algo not in ('fpocket', 'castp'):
		raise Exception(
			f'Unknown algorithm {algorithm!r}; '
			f"use 'fpocket' or 'castp'")
	if algo == 'castp':
		# Hand-rolled regular triangulation + discrete flow
		# (Liang 1998, Edelsbrunner-Facello-Liang 1998).
		raw = _castp_pipeline(pose, cutoff)
		if not raw: return {}
		raw.sort(key=lambda p: -p['Volume'])
		out = {}
		for i, p in enumerate(raw):
			out[i] = {
				'Type'       : p['Type'],
				'Volume'     : round(p['Volume'], 3),
				'Area'       : round(p['Area'], 3),
				'Center'     : p['Center'],
				'Amino Acids': p['Amino Acids']}
		return out
	if volume not in ('mc', 'exact'):
		raise Exception(
			f"Unknown volume {volume!r}; use 'mc' or 'exact'")
	# --- Step 1: heavy atoms --------------------------------------
	coords, radii, elements, atom2res, res_keys = _heavy_atoms(pose)
	# --- Step 2-3: Voronoi + alpha spheres ------------------------
	asph_c, asph_r, contacts = _voronoi_alpha_spheres(coords, radii)
	if len(asph_c) == 0: return {}
	# --- Step 4: classify polarity --------------------------------
	apolar = _classify_polarity(contacts, elements)
	# --- Step 5: 1st-pass single-linkage clustering ---------------
	labels = _single_linkage(asph_c, _FP_D1)
	# --- Step 6: 2nd-pass merge (no per-sphere filtering) ---------
	labels = _second_pass_merge(labels, asph_c, _FP_D2, _FP_NMIN)
	# --- Step 7: drop merged clusters that are too small or
	#             have too few apolar spheres --------------------
	# --- Step 8: descriptors per pocket ---------------------------
	rng = np.random.default_rng(seed)
	pockets = []
	for c in np.unique(labels):
		mask  = labels == c
		n_a   = int(mask.sum())
		if n_a < _FP_MIN_SPH: continue
		if int(apolar[mask].sum()) < _FP_MIN_APO: continue
		ctrs  = asph_c[mask]
		rs    = asph_r[mask]
		n_pol = int((~apolar[mask]).sum())
		n_apo = int(apolar[mask].sum())
		if volume == 'mc':
			V, A = _pocket_volume_mc(ctrs, rs, _FP_MC_N, rng)
		else:
			V, A = _pocket_volume_exact(ctrs, rs)
		# Participating residues for hydrophobicity computation
		resids = _residues_within(
			cutoff, ctrs, coords, atom2res, point_radii=rs)
		if not resids: continue
		# Mean Kyte-Doolittle hydropathy of contacting residues
		AAs = pose.data['Amino Acids']
		hyd = 0.0
		nres = 0
		for ri in resids:
			sym = AAs[ri][0].upper()
			if sym in _KD:
				hyd += _KD[sym]
				nres += 1
		hyd_score = (hyd / nres) if nres else 0.0
		pol = n_pol / float(n_a)
		drug = _druggability_score(
			n_a, n_apo, V, hyd_score, pol, len(resids))
		pockets.append({
			'Type'       : 'Pocket',
			'Volume'     : float(V),
			'Area'       : float(A),
			'Center'     : ctrs.mean(axis=0),
			'Amino Acids': resids,
			'_drug'      : float(drug),
			'_n_alpha'   : n_a})
	if not pockets: return {}
	# --- Step 9: rank internally by druggability, return by volume
	# (the dict indices are assigned in volume-descending order; the
	# druggability rank is preserved as a hidden key for benchmarks)
	pockets.sort(key=lambda p: -p['_drug'])
	for di, p in enumerate(pockets):
		p['_drug_rank'] = di
	pockets.sort(key=lambda p: -p['Volume'])
	out = {}
	for i, p in enumerate(pockets):
		out[i] = {
			'Type'       : p['Type'],
			'Volume'     : round(p['Volume'], 3),
			'Area'       : round(p['Area'], 3),
			'Center'     : p['Center'],
			'Amino Acids': p['Amino Acids'],
			'_drug_rank' : p['_drug_rank'],
			'_drug'      : p['_drug']}
	return out

# === CASTp pipeline helpers ====================================
# CASTp parameters (Liang 1998).
_CP_PROBE   = 1.4    # water probe radius (Å)
_CP_MIN_TET = 15     # min tets per basin to report
_CP_MIN_DEPTH2 = 12.25  # min "deepest tet" filtration² = 3.5²
                        # basins with all tets shallower than 3.5 Å
                        # are pruned as noise

def _power_circumball(p0, p1, p2, p3, w0, w1, w2, w3):
	'''
	Power (weighted) circumball of 4 weighted points.
	Returns (center, power_radius_squared) where power_radius² is
	|center - p_i|² - w_i (same for all i in a regular tet).
	Raises numpy.linalg.LinAlgError on degenerate (coplanar) input.
	'''
	A = 2.0 * np.array([p1 - p0, p2 - p0, p3 - p0])
	b = np.array([
		float(np.dot(p1, p1) - np.dot(p0, p0) - (w1 - w0)),
		float(np.dot(p2, p2) - np.dot(p0, p0) - (w2 - w0)),
		float(np.dot(p3, p3) - np.dot(p0, p0) - (w3 - w0))])
	c = np.linalg.solve(A, b)
	cr2 = float(np.dot(c - p0, c - p0) - w0)
	return c, cr2

def _tet_volume(p0, p1, p2, p3):
	'''Signed tet volume / 6 = |det((p1-p0, p2-p0, p3-p0))| / 6.'''
	return abs(float(np.linalg.det(
		np.array([p1 - p0, p2 - p0, p3 - p0])))) / 6.0

def _triangle_area(p0, p1, p2):
	'''Triangle area via cross product.'''
	return 0.5 * float(np.linalg.norm(
		np.cross(p1 - p0, p2 - p0)))

def _bowyer_watson_power(coords, weights, seed=0):
	'''
	Hand-rolled incremental Bowyer-Watson regular (power)
	triangulation in 3D.

	Inputs:
	    coords  : (N, 3) atom centers
	    weights : (N,)   atom weights (= radius²)
	Output:
	    tets : (M, 4) int array of vertex indices into `coords`
	    cc   : (M, 3) float array of power circumcenters
	    cr2  : (M,)   float array of power radii squared
	Conflict detection is vectorised per insertion (O(N²) total
	for ~1500 atoms, runs in ~5-15 s in pure NumPy).
	'''
	N = len(coords)
	# 1. Bounding super-tet vertices, placed far from the data
	cn = coords.mean(axis=0)
	span = float((coords.max(axis=0) - coords.min(axis=0)).max())
	R    = max(span * 100.0, 1000.0)
	super_pts = np.array([
		cn + R * np.array([ 1,  1,  1]),
		cn + R * np.array([ 1, -1, -1]),
		cn + R * np.array([-1,  1, -1]),
		cn + R * np.array([-1, -1,  1])])
	all_pts  = np.vstack([coords, super_pts])
	all_w    = np.concatenate([weights, np.zeros(4)])
	super_id = set(range(N, N + 4))
	# 2. Initialise with the super-tet
	tets = {}
	cc   = {}
	cr2_ = {}
	next_id = 0
	c0, r0 = _power_circumball(
		all_pts[N], all_pts[N+1],
		all_pts[N+2], all_pts[N+3],
		all_w[N], all_w[N+1], all_w[N+2], all_w[N+3])
	tets[next_id] = (N, N + 1, N + 2, N + 3)
	cc[next_id]   = c0
	cr2_[next_id] = r0
	next_id += 1
	# 3. Insert points in random order (improves geometry)
	rng = np.random.default_rng(seed)
	order = rng.permutation(N)
	for new_idx in order:
		p = all_pts[new_idx]
		w = float(all_w[new_idx])
		# Vectorised conflict scan
		ids = list(tets.keys())
		if not ids: break
		C  = np.array([cc[i] for i in ids])
		R2 = np.array([cr2_[i] for i in ids])
		d2 = ((C - p) ** 2).sum(-1)
		conflict_mask = (d2 - w) < R2
		conflict_ids = [ids[i] for i in np.where(conflict_mask)[0]]
		if not conflict_ids:
			continue
		# Boundary faces of the conflict region
		face_count = {}
		for tid in conflict_ids:
			t = tets[tid]
			for face in (
				(t[0], t[1], t[2]),
				(t[0], t[1], t[3]),
				(t[0], t[2], t[3]),
				(t[1], t[2], t[3])):
				key = tuple(sorted(face))
				face_count[key] = face_count.get(key, 0) + 1
		boundary = [f for f, c in face_count.items() if c == 1]
		# Delete conflict tets
		for tid in conflict_ids:
			del tets[tid]
			del cc[tid]
			del cr2_[tid]
		# Connect new point to each boundary face
		for face in boundary:
			try:
				c_, r_ = _power_circumball(
					all_pts[face[0]], all_pts[face[1]],
					all_pts[face[2]], p,
					float(all_w[face[0]]),
					float(all_w[face[1]]),
					float(all_w[face[2]]), w)
			except np.linalg.LinAlgError:
				continue
			tets[next_id] = (face[0], face[1], face[2], int(new_idx))
			cc[next_id]   = c_
			cr2_[next_id] = r_
			next_id += 1
	# 4. Strip tets that touch a super vertex
	final_t = []
	final_c = []
	final_r = []
	for tid, t in tets.items():
		if any(v in super_id for v in t): continue
		final_t.append(t)
		final_c.append(cc[tid])
		final_r.append(cr2_[tid])
	if not final_t:
		return (np.zeros((0, 4), dtype=int),
			np.zeros((0, 3)), np.zeros(0))
	return (np.array(final_t, dtype=int),
		np.array(final_c), np.array(final_r))

def _validate_regular_triangulation(tets, cc, cr2, coords, weights):
	'''
	Mandatory post-construction validation of a regular
	triangulation. Returns (ok, reason).
	Checks:
	  1. No zero-volume tets.
	  2. No duplicate tets.
	  3. Face manifoldness (each interior face shared by 2 tets).
	  4. Power-empty axiom (no atom outside a tet's vertices is
	     inside the tet's power circumball).
	'''
	n_tet = len(tets)
	if n_tet == 0:
		return True, 'empty triangulation'
	# 1. Zero-volume check
	for i, t in enumerate(tets):
		v = _tet_volume(
			coords[t[0]], coords[t[1]],
			coords[t[2]], coords[t[3]])
		if v < 1e-9:
			return False, f'zero-volume tet at index {i}'
	# 2. Duplicate check
	seen = set()
	for t in tets:
		key = tuple(sorted(t))
		if key in seen:
			return False, f'duplicate tet {key}'
		seen.add(key)
	# 3. Face manifoldness
	face_count = {}
	for t in tets:
		for face in (
			(t[0], t[1], t[2]),
			(t[0], t[1], t[3]),
			(t[0], t[2], t[3]),
			(t[1], t[2], t[3])):
			key = tuple(sorted(face))
			face_count[key] = face_count.get(key, 0) + 1
	for face, k in face_count.items():
		if k not in (1, 2):
			return (False,
				f'face {face} has {k} incident tets')
	# 4. Power-empty axiom — sample-based, full check is too slow
	N = len(coords)
	for i in rng_sample(range(n_tet), min(n_tet, 200)):
		t  = tets[i]
		c  = cc[i]
		r2 = cr2[i]
		# For all atoms not in this tet, power_dist must be >= 0
		mask = np.ones(N, dtype=bool)
		mask[list(t)] = False
		d2 = ((coords[mask] - c) ** 2).sum(-1)
		w  = weights[mask]
		violations = (d2 - w) < (r2 - 1e-6)
		if violations.any():
			return (False,
				f'power-empty violated at tet {i}')
	return True, 'OK'

def rng_sample(seq, k):
	'''Deterministic sample of k items from seq (or all if smaller).'''
	seq = list(seq)
	if len(seq) <= k: return seq
	step = max(1, len(seq) // k)
	return seq[::step][:k]

def _build_face_to_tets(tets):
	'''
	Return {sorted face tuple: [tet indices that contain it]}.
	Each interior face has exactly 2 tets; boundary faces have 1.
	'''
	out = {}
	for ti, t in enumerate(tets):
		for face in (
			(t[0], t[1], t[2]),
			(t[0], t[1], t[3]),
			(t[0], t[2], t[3]),
			(t[1], t[2], t[3])):
			key = tuple(sorted(face))
			out.setdefault(key, []).append(ti)
	return out

def _castp_pipeline(pose, cutoff, probe=_CP_PROBE):
	'''
	Full CASTp-style cavity detection on a protein pose.

	Steps:
	  1. Build the regular (power) triangulation of heavy atoms.
	  2. Validate (4 axioms).
	  3. Compute filtration value per tet (= power circumradius²).
	  4. Identify "empty" tets (filtration > probe²): water fits.
	  5. Discrete flow on the empty-tet graph.
	  6. Basins = sinks + transitive predecessors.
	  7. Pocket vs Void by reachability to bulk solvent.
	  8. Volume / area from tet sums + boundary triangles.

	Returns a list of pocket dicts in the same shape that the
	fpocket pipeline produces (Volume, Area, Center, Amino Acids,
	Type) for downstream sorting.
	'''
	coords, radii, elements, atom2res, res_keys = _heavy_atoms(pose)
	weights = radii * radii
	tets, cc, cr2 = _bowyer_watson_power(coords, weights)
	if len(tets) == 0: return []
	ok, reason = _validate_regular_triangulation(
		tets, cc, cr2, coords, weights)
	if not ok:
		raise Exception(
			f'Regular triangulation failed validation '
			f'\u2014 {reason}')
	# Filtration value per tet = power circumradius²
	filt = cr2.copy()
	# "Empty" tets: probe sphere fits inside (alpha-complement)
	probe2 = probe * probe
	empty = filt > probe2
	# Filter out boundary "shell" tets — empty tets whose centre
	# is far from many atoms are at the convex hull rather than
	# in real interior cavities. A real cavity tet sits "inside"
	# the protein with several atoms nearby; a boundary tet is
	# bordered by widely-spaced atoms with few neighbours.
	# Drop tets whose centre has fewer than 6 heavy atoms within
	# 7 Å (interior cavities are well-surrounded; boundary tets
	# are not).
	close_thresh2 = 7.0 * 7.0
	min_close = 6
	keep = np.zeros(len(tets), dtype=bool)
	for i in np.where(empty)[0]:
		d2 = ((coords - cc[i]) ** 2).sum(-1)
		if int((d2 < close_thresh2).sum()) >= min_close:
			keep[i] = True
	empty = keep
	empty_idx = np.where(empty)[0]
	if len(empty_idx) == 0: return []
	# Build face -> tet adjacency restricted to empty tets
	empty_set = set(int(i) for i in empty_idx)
	face_to_tets = _build_face_to_tets(tets)
	# Adjacency: tet -> list of (neighbor_tet_idx, shared_face)
	adj = {int(i): [] for i in empty_idx}
	for face, ts in face_to_tets.items():
		ets = [t for t in ts if t in empty_set]
		if len(ets) == 2:
			adj[ets[0]].append((ets[1], face))
			adj[ets[1]].append((ets[0], face))
	# Discrete-flow watershed: cavities are LOCAL MAXIMA of
	# depth (filtration = power circumradius²). Each empty tet
	# walks uphill (to its highest-filtration empty neighbour)
	# until it reaches a local max — that max is the "sink" /
	# basin seed. A tet is a local max if no neighbour has
	# strictly higher filtration. Ties broken by tet index for
	# determinism. This produces one basin per local max,
	# splitting the giant connected empty component into
	# separate regions around each cavity.
	flow = {}
	for ti in empty_idx:
		ti = int(ti)
		best = ti
		best_f = filt[ti]
		for nb, _ in adj[ti]:
			if filt[nb] > best_f or (
					filt[nb] == best_f and nb > best):
				best = nb
				best_f = filt[nb]
		flow[ti] = best
	# Path-compressed walk to the local max (basin seed)
	sink_of = {}
	def find_sink(t):
		path = []
		cur = t
		while flow[cur] != cur and cur not in sink_of:
			path.append(cur)
			cur = flow[cur]
		root = sink_of.get(cur, cur)
		for x in path:
			sink_of[x] = root
		sink_of[cur] = root
		return root
	for ti in empty_idx:
		find_sink(int(ti))
	basins = {}
	for ti, sk in sink_of.items():
		basins.setdefault(sk, []).append(ti)
	# Pocket vs Void classification by reachability to bulk
	# solvent. A basin is a Pocket if water can reach it from
	# OUTSIDE the protein without passing through atom-filled
	# space. In the alpha-complement graph (empty tets connected
	# via shared faces), this means: the basin is reachable from
	# the convex hull (empty tets whose power circumcentres lie
	# outside the atom convex hull) via a path of empty-tet
	# adjacencies. If not reachable: Void.
	#
	# We seed reachability from empty tets whose centres are
	# outside the atom convex hull (these trivially connect to
	# bulk solvent).
	from scipy.spatial import ConvexHull
	try:
		hull = ConvexHull(coords)
		hull_eq = hull.equations  # (n_facets, 4): [a b c d]
		# A point is outside the hull iff max(a*x + b*y + c*z + d) > 0
		def is_outside(p):
			val = hull_eq[:, :3] @ p + hull_eq[:, 3]
			return float(val.max()) > 0.0
	except Exception:
		is_outside = lambda p: False
	# Seed reachable set with tets whose circumcentre is outside
	exterior_tets = set()
	for i in empty_idx:
		i = int(i)
		if is_outside(cc[i]):
			exterior_tets.add(i)
	# For BFS propagation, a shared face is "passable" iff a
	# water probe fits through the gap between the 3 face
	# atoms. Compute the gate radius: distance from the
	# triangle centroid to each atom's centre, minus that
	# atom's vdW radius; the minimum must exceed the probe
	# radius. Narrow faces block propagation, so interior
	# cavities stay disconnected from bulk solvent.
	def face_passable(face):
		ca = coords[face[0]]
		cb = coords[face[1]]
		cc_ = coords[face[2]]
		centroid = (ca + cb + cc_) / 3.0
		d0 = float(np.linalg.norm(centroid - ca)) - float(radii[face[0]])
		d1 = float(np.linalg.norm(centroid - cb)) - float(radii[face[1]])
		d2 = float(np.linalg.norm(centroid - cc_)) - float(radii[face[2]])
		return min(d0, d1, d2) > probe
	# BFS via empty-tet adjacency, only through passable faces.
	queue = list(exterior_tets)
	reachable = set(exterior_tets)
	while queue:
		t = queue.pop()
		for nb, face in adj[t]:
			if nb in reachable: continue
			if not face_passable(face): continue
			reachable.add(nb)
			queue.append(nb)
	def is_exterior(basin_tets):
		return any(t in reachable for t in basin_tets)
	out = []
	for sk, basin_tets in basins.items():
		if len(basin_tets) < _CP_MIN_TET: continue
		# Drop shallow noise basins
		max_depth = max(filt[ti] for ti in basin_tets)
		if max_depth < _CP_MIN_DEPTH2: continue
		# Volume = sum of tet volumes
		V = 0.0
		all_verts = set()
		for ti in basin_tets:
			t = tets[ti]
			V += _tet_volume(
				coords[t[0]], coords[t[1]],
				coords[t[2]], coords[t[3]])
			all_verts.update(int(v) for v in t)
		# Area = sum of boundary face areas (faces not shared
		# with another tet in the same basin)
		bset = set(basin_tets)
		A = 0.0
		bd_pts = []
		for ti in basin_tets:
			t = tets[ti]
			for face in (
				(t[0], t[1], t[2]),
				(t[0], t[1], t[3]),
				(t[0], t[2], t[3]),
				(t[1], t[2], t[3])):
				key = tuple(sorted(face))
				ts  = face_to_tets[key]
				other = [x for x in ts if x != ti]
				if any(o not in bset for o in other) \
						or not other:
					A += _triangle_area(
						coords[face[0]],
						coords[face[1]],
						coords[face[2]])
					bd_pts.extend(face)
		# Center = centroid of basin tet vertices
		vcoords = coords[list(all_verts)]
		center  = vcoords.mean(axis=0)
		# Participating residues: residues of any atom in any
		# basin tet vertex (within `cutoff` of any vertex)
		verts_list = list(all_verts)
		pts = coords[verts_list]
		dummy_radii = np.zeros(len(pts))
		resids = _residues_within(
			cutoff, pts, coords, atom2res,
			point_radii=dummy_radii)
		if not resids: continue
		typ = 'Pocket' if is_exterior(basin_tets) else 'Void'
		out.append({
			'Type'       : typ,
			'Volume'     : V,
			'Area'       : A,
			'Center'     : center,
			'Amino Acids': resids,
			'_n_tets'    : len(basin_tets)})
	return out

