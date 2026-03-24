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
