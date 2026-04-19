import re
import os
import json
import math
import numpy as np
from collections import defaultdict, deque

def Parameterise(filename, unicode, tricode):
	'''
	Add a new amino acid entry to database.json
	Arguments:
	----------
		filename : str - Path to CIF file (download from RCSB Chemical Sketch)
		unicode  : str - Single-letter key to use in AminoAcids.json (e.g. 'J')
		tricode  : str - Three-letter residue code (e.g. 'MSE')
	Return:
	-------
		Updated database.json
		Notice   : str - What got updated or replaced
	'''
	# 1. ALA reference frame (N at origin): N, H1-3, CA, HA, CB, 1HB-3HB, C, O, OXT.
	ALA = np.array([
		[ 0.000,  0.000,  0.000], [-0.334, -0.943,  0.000],
		[-0.334,  0.471,  0.816], [-0.334,  0.471, -0.816],
		[ 1.458,  0.000,  0.000], [ 1.822, -0.535,  0.877],
		[ 1.988, -0.773, -1.199], [ 3.078, -0.764, -1.185],
		[ 1.633, -1.802, -1.154], [ 1.633, -0.307, -2.117],
		[ 2.009,  1.420,  0.000], [ 2.058,  2.045,  1.023],
		[ 2.394,  1.914, -1.023]])
	unicode, tricode = unicode.upper(), tricode.upper()
	# 2. Parse CIF: atom rows have >=18 tokens (coords at [15:18] or fallback [12:15]); bond rows have 7 tokens.
	COORD_RAW, ATOMS_RAW, BONDS = [], [], []
	with open(filename) as fh:
		for line in fh:
			t = line.strip().split()
			if not t or t[0] != tricode: continue
			if len(t) == 7 and t[3] in ('SING','DOUB','TRIP','AROM'):
				BONDS.append((t[1], t[2], t[3], t[4]))
			elif len(t) >= 18:
				try:
					try:    c = [float(t[i]) for i in (15, 16, 17)]
					except (ValueError, IndexError):
						c = [float(t[i]) for i in (12, 13, 14)]
					COORD_RAW.append(c)
					ATOMS_RAW.append({'id': t[1],
						'elem': t[3].capitalize(),
						'bb': (t[9] == 'Y')})
				except (ValueError, IndexError): pass
	COORD   = np.array(COORD_RAW)
	CIF_IDS = [a['id'] for a in ATOMS_RAW]
	if 'CB' not in CIF_IDS:
		raise ValueError(f'No CB atom found in {filename}. '
			'Only standard amino acids (not GLY) are supported.')
	bb_set = {a['id'] for a in ATOMS_RAW if a['bb']} or {
		'N','CA','C','O','OXT','H','H1','H2','H3',
		'HA','HA2','HA3','HXT'}
	elem    = {a['id']: a['elem'] for a in ATOMS_RAW}
	cif_ord = {a['id']: i for i, a in enumerate(ATOMS_RAW)}
	# 3. Superimpose onto ALA backbone frame via rigid motion (N, CA, CB, C) with CA as origin.
	try:
		Ni, CAi = CIF_IDS.index('N'),  CIF_IDS.index('CA')
		CBi, Ci = CIF_IDS.index('CB'), CIF_IDS.index('C')
	except ValueError as e:
		raise ValueError(f'Missing backbone atom in {filename}: {e}')
	A = np.c_[ALA,   np.ones(len(ALA))]
	B = np.c_[COORD, np.ones(len(COORD))]
	AL = np.array([A[0]-A[4], A[6]-A[4], A[-3]-A[4], A[4]])
	BL = np.array([B[Ni]-B[CAi], B[CBi]-B[CAi], B[Ci]-B[CAi], B[CAi]])
	COORD = (B @ (np.linalg.inv(BL) @ AL))[:, :3]
	# 4. Build undirected bond graph indexed by atom id.
	adj = defaultdict(set)
	for a1, a2, _v, _r in BONDS:
		adj[a1].add(a2); adj[a2].add(a1)
	# 5. BFS sidechain from CB (heavy-atom queue; H neighbours land next to their parent in CIF order).
	ordered = []
	seen    = set(bb_set) | {'CB'}
	q       = deque(['CB'])
	while q:
		atom = q.popleft()
		ordered.append(atom)
		for n in sorted(adj[atom], key=lambda m: cif_ord.get(m, 9999)):
			if n in seen: continue
			seen.add(n)
			(ordered if elem.get(n, '').upper() in ('H', 'D') else q).append(n)
	# 6. Rename atoms: CIF HB2 becomes Pose 1HB (per-base counter on H/D atoms).
	name_map, counter = {}, defaultdict(int)
	for name in ordered:
		m = re.match(r'^([A-Z]+)(\d+)$', name)
		if m and elem.get(name, '').upper() in ('H', 'D'):
			counter[m.group(1)] += 1
			name_map[name] = f'{counter[m.group(1)]}{m.group(1)}'
		else:
			name_map[name] = name
	# 7. Detect fused sidechain: any sidechain atom bonded back to N (e.g. PRO).
	fused_atom = next((sc for sc in ordered if 'N' in adj[sc]), None)
	fused = fused_atom is not None
	# 8. Sidechain bond graph on new indices; -5 sentinel stands in for the backbone N of a fused ring.
	sc_set  = set(ordered)
	new_idx = {n: i for i, n in enumerate(ordered)}
	sc_bonds, sc_orders, bo_lookup = (
		defaultdict(list), defaultdict(list), {})
	for a1, a2, vo, ar in BONDS:
		u = vo.upper()
		if   ar == 'Y':   bo = 1.5
		elif u == 'SING': bo = 1
		elif u == 'DOUB': bo = 2
		elif u == 'TRIP': bo = 3
		elif u == 'AROM': bo = 1.5
		else:
			print(f'Warning: unknown bond order {vo!r} for '
				f'{a1}-{a2} in {tricode}, defaulting to 1')
			bo = 1
		bo_lookup[(a1, a2)] = bo_lookup[(a2, a1)] = bo
	for a1, a2, _v, _r in BONDS:
		if a1 in sc_set and a2 in sc_set:
			i1, i2 = new_idx[a1], new_idx[a2]
			sc_bonds[i1].append(i2);  sc_bonds[i2].append(i1)
			bo = bo_lookup[(a1, a2)]
			sc_orders[i1].append(bo); sc_orders[i2].append(bo)
	if fused:
		fi = new_idx[fused_atom]
		sc_bonds[fi].append(-5);  sc_bonds[-5].append(fi)
		sc_orders[fi].append(1);  sc_orders[-5].append(1)
	# 9. Two-pass aromaticity: C with >=2 O/N neighbours and at least one double bond gets resonance (all C-O/N -> 1.5).
	elem_at_idx = {new_idx[n]: elem[n] for n in ordered}
	for _p in range(2):
		for i in list(sc_bonds.keys()):
			if i < 0 or elem_at_idx.get(i, '') != 'C': continue
			xs = [(k, nb, bo) for k, (nb, bo) in enumerate(
					zip(sc_bonds[i], sc_orders[i]))
				if nb >= 0 and elem_at_idx.get(nb, '') in ('O','N')]
			if len(xs) < 2 or not any(bo >= 2 for _, _, bo in xs): continue
			for k, nb, bo in xs:
				if bo == 1.5: continue
				sc_orders[i][k] = 1.5
				for kk, mnb in enumerate(sc_bonds[nb]):
					if mnb == i:
						sc_orders[nb][kk] = 1.5
						break
	# 10. Final bond dicts with sorted neighbours for determinism; -5 sentinel kept last if fused.
	pos_keys     = sorted(k for k in sc_bonds if k >= 0)
	final_bonds  = {k: sorted(sc_bonds[k]) for k in pos_keys}
	final_orders = {k: [dict(zip(sc_bonds[k], sc_orders[k]))[nb]
		for nb in final_bonds[k]] for k in pos_keys}
	if fused:
		final_bonds[-5]  = sorted(sc_bonds[-5])
		final_orders[-5] = [dict(zip(sc_bonds[-5], sc_orders[-5]))[nb]
			for nb in final_bonds[-5]]
	# 11. Chi angles: trace main chain from CB by CIF-ordinal preference; 4-atom window over [N, CA, *mc].
	mc, visited, cur = [], set(bb_set) | {'CA'}, 'CB'
	while cur is not None:
		mc.append(cur); visited.add(cur)
		hvs = [n for n in adj[cur] if n not in visited
			and elem.get(n, '').upper() not in ('H', 'D')]
		cur = min(hvs, key=lambda n: cif_ord.get(n, 9999)) if hvs else None
	full_chain = ['N', 'CA'] + mc
	chis = [full_chain[i:i+4] for i in range(len(full_chain) - 3)]
	if fused and len(full_chain) >= 5:
		chis.append(full_chain[-3:] + ['N'])
		chis.append(full_chain[-2:] + ['N', 'CA'])
	# 12. Assemble the new entry in the same field order as existing AAs.
	id_to_i = {cid: i for i, cid in enumerate(CIF_IDS)}
	entry = {
		'Vectors':         [COORD[id_to_i[n]].tolist() for n in ordered],
		'Tricode':         tricode,
		'Fused':           fused,
		'Sidechain Atoms': [[name_map[n], elem[n], 0, 1.0, 0] for n in ordered],
		'Chi Angle Atoms': chis,
		'Bonds':           {str(k): v for k, v in final_bonds.items()},
		'BondOrders':      {str(k): v for k, v in final_orders.items()}}
	# 13. Merge into database.json.
	db_path = os.path.join(
		os.path.dirname(os.path.abspath(__file__)), 'database.json')
	with open(db_path) as fh: db = json.load(fh)
	if unicode in db['Amino Acids']:
		print(f'Warning: "{unicode}" already exists... overwriting.')
	db['Amino Acids'][unicode] = entry
	# 14. Validate Bonds/BondOrders symmetry across the whole DB before
	#     writing anything. Fails loudly on any malformed entry so the
	#     hot paths in Pose (_bondtree, Import) can stay guard-free.
	def _validate_db(db):
		for section in ('Amino Acids', 'Nucleotides'):
			for ekey, e in db.get(section, {}).items():
				if 'Bonds' not in e: continue
				bonds = e['Bonds']
				if 'BondOrders' not in e:
					raise ValueError(
						f'{section}[{ekey!r}]: '
						f'has Bonds but no BondOrders')
				bo = e['BondOrders']
				for k, nbrs in bonds.items():
					if k not in bo:
						raise ValueError(
							f'{section}[{ekey!r}]: '
							f'BondOrders missing key {k!r}')
					if len(bo[k]) != len(nbrs):
						raise ValueError(
							f'{section}[{ekey!r}][{k!r}]: '
							f'Bonds has {len(nbrs)} entries but '
							f'BondOrders has {len(bo[k])}')
	_validate_db(db)
	# 15. Serialise preserving the compact layout; atomic write via rename.
	def _fmt_field(field, val):
		'''Format one database.json field as a list of pre-indented
		lines (no trailing comma; the enclosing entry closes the dict).
		Unknown fields fall back to json.dumps so no data is dropped.'''
		if field == 'Vectors':
			out = ['        "Vectors": [']
			n = len(val) - 1
			for vi, v in enumerate(val):
				tail = ',' if vi < n else ']'
				body = '[' + ', '.join(
					json.dumps(round(float(x), 3)) for x in v) + ']'
				out.append('            ' + body + tail)
			return out
		if field in ('Tricode', 'Type'):
			return [f'        "{field}": "{val}"']
		if field == 'Fused':
			return ['        "Fused": ' + ('true' if val else 'false')]
		if field in ('Sidechain Atoms', 'Backbone Atoms', 'Base Atoms'):
			out = [f'        "{field}": [']
			n = len(val) - 1
			for ai, a in enumerate(val):
				if len(a) == 5:
					body = (f'["{a[0]}", "{a[1]}", '
						f'{float(a[2]):.1f}, {float(a[3]):.1f}, '
						f'{float(a[4]):.1f}]')
				else:
					body = (f'["{a[0]}", "{a[1]}", '
						f'{int(a[2])}, {int(a[3])}]')
				tail = ',' if ai < n else ']'
				out.append('            ' + body + tail)
			return out
		if field == 'Chi Angle Atoms':
			if not val:
				return [f'        "{field}": []']
			if isinstance(val[0], str):
				inner = ', '.join(f'"{x}"' for x in val)
				return [f'        "{field}": [{inner}]']
			out = [f'        "{field}": [']
			n = len(val) - 1
			for ci, c in enumerate(val):
				tail = ',' if ci < n else ']'
				inner = ', '.join(f'"{x}"' for x in c)
				out.append('            [' + inner + ']' + tail)
			return out
		if field in ('Bonds', 'BondOrders'):
			out = [f'        "{field}": {{']
			bi = list(val.items()); n = len(bi) - 1
			for k, (bk, bv) in enumerate(bi):
				inner = ', '.join(
					(f'{x:g}' if isinstance(x, float) else str(x))
					for x in bv)
				tail = ',' if k < n else '}'
				out.append(
					'            "' + str(bk) + '":[' + inner + ']' + tail)
			return out
		encoded = json.dumps(val, indent=4)
		enc_lines = encoded.split('\n')
		out = [f'        "{field}": {enc_lines[0]}']
		for el in enc_lines[1:]:
			out.append('        ' + el)
		return out
	def _fmt_entry(entry_key, entry):
		out = [f'    "{entry_key}": {{']
		blocks = [_fmt_field(f, v) for f, v in entry.items()]
		for bi, block in enumerate(blocks):
			if bi < len(blocks) - 1:
				block[-1] = block[-1] + ','
			out.extend(block)
		out[-1] = out[-1] + '}'
		return out
	def _fmt_db(db):
		L = ['{']
		sections = list(db.items())
		for si, (sname, entries) in enumerate(sections):
			L.append(f'"{sname}": {{')
			items = list(entries.items())
			for ei, (ekey, e) in enumerate(items):
				block = _fmt_entry(ekey, e)
				if ei < len(items) - 1:
					block[-1] = block[-1] + ','
				L.extend(block)
				if ei < len(items) - 1:
					L.append('')
			L[-1] = L[-1] + '}'
			if si < len(sections) - 1:
				L[-1] = L[-1] + ','
				L.append('')
		L.append('}')
		return '\n'.join(L) + '\n'
	tmp_path = db_path + '.tmp'
	with open(tmp_path, 'w') as fh: fh.write(_fmt_db(db))
	os.replace(tmp_path, db_path)
	print(f'Added {tricode} as "{unicode}" to database.json')

def RMSD(pose1, pose2, alg='align', export=None):
	'''
	Calculate RMSD between two poses (protein or nucleic acid)
	Arguments:
	----------
		pose1  : Pose - First pose (protein or nucleic acid)
		pose2  : Pose - Second pose (must be same Type as pose1)
		alg    : str  - 'align' (default), 'kabsch', 'quaternion', 'simple'
		export : str  - Output filename for aligned PDB pair; None skips export
	Return:
	-------
		float : RMSD value in angstroms, rounded to 5 decimals
	'''
	# 1. Validate algorithm and check both poses are the same molecule type.
	if alg not in ('align', 'kabsch', 'quaternion', 'simple'):
		raise Exception('Unknown algorithm: ' + str(alg))
	t1, t2 = pose1.data['Type'], pose2.data['Type']
	if (t1 == 'Protein') != (t2 == 'Protein'):
		raise Exception(f'Cannot align {t1} with {t2}: '
			'cannot mix protein and nucleic acid')
	# 2. Resolve molecule-specific residue-key and reference-atom name.
	is_pro = (t1 == 'Protein')
	rk     = 'Amino Acids' if is_pro else 'Nucleotides'
	ra     = 'CA' if is_pro else "C1'"
	atoms1, co1, res1 = (pose1.data['Atoms'],
		pose1.data['Coordinates'], pose1.data[rk])
	atoms2, co2, res2 = (pose2.data['Atoms'],
		pose2.data['Coordinates'], pose2.data[rk])
	if alg == 'align':
		# 3. Needleman-Wunsch DP with BLOSUM62 (proteins) or +1/-0.5 (nucleic); gap = -1.
		rk1, rk2 = sorted(res1.keys()), sorted(res2.keys())
		seq1 = ''.join(res1[k][0].upper() for k in rk1)
		seq2 = ''.join(res2[k][0].upper() for k in rk2)
		m, n, gap = len(seq1), len(seq2), -1.0
		dp = np.zeros((m + 1, n + 1))
		dp[:, 0] = np.arange(m + 1) * gap
		dp[0, :] = np.arange(n + 1) * gap
		for i in range(1, m + 1):
			a = seq1[i-1]
			for j in range(1, n + 1):
				b = seq2[j-1]
				s = (_blosum(a, b) if is_pro
					else (1.0 if a == b else -0.5))
				dp[i, j] = max(dp[i-1, j-1] + s,
					dp[i-1, j] + gap, dp[i, j-1] + gap)
		# 4. Traceback the optimal alignment path to recover residue pairs.
		pairs, i, j = [], m, n
		while i > 0 and j > 0:
			a, b = seq1[i-1], seq2[j-1]
			s = (_blosum(a, b) if is_pro
				else (1.0 if a == b else -0.5))
			if   abs(dp[i, j] - (dp[i-1, j-1] + s)) < 1e-9:
				pairs.append((i - 1, j - 1))
				i -= 1; j -= 1
			elif abs(dp[i, j] - (dp[i-1, j]   + gap)) < 1e-9:
				i -= 1
			else:
				j -= 1
		pairs.reverse()
		if len(pairs) < 3:
			raise Exception('Too few aligned residue pairs')
		# 5. Gather reference-atom coordinates for each aligned pair.
		P_aln = np.array([next(co1[ai].copy().astype(float)
			for ai in res1[rk1[ii]][2]
			if atoms1[ai][0] == ra) for ii, _ in pairs])
		Q_aln = np.array([next(co2[ai].copy().astype(float)
			for ai in res2[rk2[jj]][2]
			if atoms2[ai][0] == ra) for _, jj in pairs])
		# 6. Iterative Kabsch with 2.0 A outlier rejection (5 rounds + 1 final fit).
		mask = np.ones(len(pairs), dtype=bool)
		for _ in range(6):
			Pm, Qm   = P_aln[mask], Q_aln[mask]
			t_P, t_Q = Pm.mean(axis=0), Qm.mean(axis=0)
			P, Q     = Pm - t_P, Qm - t_Q
			U, _, Vt = np.linalg.svd(P.T @ Q)
			d = np.sign(np.linalg.det(Vt.T @ U.T))
			R = Vt.T @ np.diag(np.array([1.0, 1.0, d])) @ U.T
			dists = np.sqrt((((P_aln - t_P)
				- (Q_aln - t_Q) @ R) ** 2).sum(axis=1))
			new_mask = dists < 2.0
			if (np.array_equal(new_mask, mask)
					or new_mask.sum() < 3):
				break
			mask = new_mask
	else:
		# 3. Gather all ref-atom coords (skipping residues that lack it), truncate to shorter pose.
		coords1 = [c for c in (next(
			(co1[ai].copy().astype(float)
				for ai in res1[ri][2] if atoms1[ai][0] == ra),
			None) for ri in sorted(res1.keys()))
			if c is not None]
		coords2 = [c for c in (next(
			(co2[ai].copy().astype(float)
				for ai in res2[ri][2] if atoms2[ai][0] == ra),
			None) for ri in sorted(res2.keys()))
			if c is not None]
		if not coords1 or not coords2:
			raise Exception(
				f'No {ra} atoms found in one or both poses')
		n = min(len(coords1), len(coords2))
		P, Q     = np.array(coords1[:n]), np.array(coords2[:n])
		t_P, t_Q = P.mean(axis=0), Q.mean(axis=0)
		P, Q     = P - t_P, Q - t_Q
		# 4. Compute rotation matrix via the selected algorithm (Horn 1987 for quaternion).
		if alg == 'simple':
			R = np.eye(3)
		elif alg == 'kabsch':
			U, _, Vt = np.linalg.svd(P.T @ Q)
			d = np.sign(np.linalg.det(Vt.T @ U.T))
			R = Vt.T @ np.diag(np.array([1.0, 1.0, d])) @ U.T
		else:
			H  = P.T @ Q
			a, b, c = H[0]; d, e, f = H[1]; g, h, k = H[2]
			F = np.array([
				[a+e+k,   f-h,     g-c,     b-d    ],
				[f-h,     a-e-k,   b+d,     c+g    ],
				[g-c,     b+d,    -a+e-k,   f+h    ],
				[b-d,     c+g,     f+h,    -a-e+k  ]])
			q0, q1, q2, q3 = np.linalg.eigh(F)[1][:, -1]
			R = np.array([
				[q0*q0+q1*q1-q2*q2-q3*q3,
					2*(q1*q2-q0*q3),         2*(q1*q3+q0*q2)],
				[2*(q1*q2+q0*q3),
					q0*q0-q1*q1+q2*q2-q3*q3, 2*(q2*q3-q0*q1)],
				[2*(q1*q3-q0*q2),
					2*(q2*q3+q0*q1),         q0*q0-q1*q1-q2*q2+q3*q3]])
	# 7. Compute RMSD from centred coordinate residuals.
	diff = P - Q @ R
	rmsd = np.sqrt(np.mean((diff ** 2).sum(axis=1)))
	# 8. Optionally export aligned pose pair as PDB files.
	if export is not None:
		orig = pose2.data['Coordinates'].copy()
		pose2.data['Coordinates'] = (orig - t_Q) @ R + t_P
		fn, ext = export[:-4], export[-4:]
		pose1.Export(fn + '_1' + ext)
		pose2.Export(fn + '_2' + ext)
		pose2.data['Coordinates'] = orig
	return round(float(rmsd), 5)

# BLOSUM62 scoring matrix — shared by BLAST() and MSA()
_aa  = 'ARNDCQEGHILKMFPSTWYV'
_idx = {c: i for i, c in enumerate(_aa)}
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
	[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]]

def _blosum(a, b):
	'''
	BLOSUM62 pairwise amino-acid substitution score
	Arguments:
	----------
		a : str - First amino acid one-letter code (must be uppercase)
		b : str - Second amino acid one-letter code (must be uppercase)
	Return:
	-------
		int : BLOSUM62 score; falls back to +4 for match / -1 for mismatch if unknown
	'''
	# 1. Resolve alphabet indices for both residues.
	ia, ib = _idx.get(a, -1), _idx.get(b, -1)
	# 2. Fallback when either residue is outside the BLOSUM62 alphabet.
	if ia < 0 or ib < 0: return(4 if a == b else -1)
	# 3. Look up the canonical BLOSUM62 score.
	return(_bm[ia][ib])

def BLAST(seq1, seq2):
	'''
	Pairwise protein alignment via Smith-Waterman with BLOSUM62 and Karlin-Altschul E-value
	Arguments:
	----------
		seq1 : str - FASTA sequence of the first (query) protein
		seq2 : str - FASTA sequence of the second (subject) protein
	Return:
	-------
		str   : BLAST-like formatted alignment report
		float : Percent identity over the aligned region
		float : Karlin-Altschul expect value
	'''
	# 1. Uppercase both sequences and capture lengths.
	seq1, seq2 = seq1.upper(), seq2.upper()
	m, n = len(seq1), len(seq2)
	# 2. Smith-Waterman DP with affine gaps (NCBI BLASTP defaults: open=11, extend=1).
	go, ge, INF = 11, 1, float('-inf')
	H  = np.zeros((m+1, n+1))
	E  = np.full((m+1, n+1), INF)
	F  = np.full((m+1, n+1), INF)
	tb = np.zeros((m+1, n+1), dtype=np.int8)
	# 2a. Precompute (m, n) substitution score matrix once, using BLOSUM62
	# for alphabet residues and (+4 self-match / -1 mismatch) fallback for
	# anything outside the 20-letter alphabet.
	BM   = np.array(_bm, dtype=float)
	idx1 = np.array([_idx.get(c, -1) for c in seq1], dtype=np.int64)
	idx2 = np.array([_idx.get(c, -1) for c in seq2], dtype=np.int64)
	valid = (idx1[:, None] >= 0) & (idx2[None, :] >= 0)
	S = np.where(valid,
		BM[np.clip(idx1[:, None], 0, 19),
		   np.clip(idx2[None, :], 0, 19)],
		0.0)
	arr1 = np.array(list(seq1))
	arr2 = np.array(list(seq2))
	eq   = arr1[:, None] == arr2[None, :]
	S    = np.where(valid, S, np.where(eq, 4.0, -1.0))
	best, bi, bj = 0.0, 0, 0
	for i in range(1, m+1):
		# 2b. Vectorised F column update: depends only on H[i-1], F[i-1].
		F[i, 1:] = np.maximum(H[i-1, 1:] - go - ge, F[i-1, 1:] - ge)
		for j in range(1, n+1):
			diag    = H[i-1, j-1] + S[i-1, j-1]
			E[i, j] = max(H[i, j-1] - go - ge, E[i, j-1] - ge)
			h       = max(0.0, diag, E[i, j], F[i, j])
			H[i, j] = h
			if h > best: best, bi, bj = h, i, j
			tb[i, j] = (0 if h == 0 else 1 if h == diag
				else 2 if h == F[i, j] else 3)
	if best == 0:
		raise Exception('No alignment found between the sequences')
	# 3. Traceback from the highest-scoring cell to recover aligned strings.
	aq, as_, i, j = [], [], bi, bj
	while i > 0 and j > 0 and H[i, j] > 0:
		t = int(tb[i, j])
		if t == 1:
			aq.append(seq1[i-1]); as_.append(seq2[j-1])
			i -= 1; j -= 1
		elif t == 2:
			aq.append(seq1[i-1]); as_.append('-'); i -= 1
		else:
			aq.append('-'); as_.append(seq2[j-1]); j -= 1
	aq, as_ = ''.join(reversed(aq)), ''.join(reversed(as_))
	# 4. Compute identity, similarity, gap statistics over the aligned region.
	qs, ss, aln_len = i + 1, j + 1, len(aq)
	n_id  = sum(1 for a, b in zip(aq, as_) if a == b and a != '-')
	n_pos = sum(1 for a, b in zip(aq, as_)
		if a != '-' and b != '-' and _blosum(a, b) > 0)
	n_gap = aq.count('-') + as_.count('-')
	pct     = round(n_id  / aln_len * 100, 2)
	pct_pos = round(n_pos / aln_len * 100, 1)
	pct_gap = round(n_gap / aln_len * 100, 1)
	# 5. Karlin-Altschul E-value (lam=0.270, K=0.041 for BLOSUM62 with gap 11/1).
	lam, K  = 0.270, 0.041
	e_value = K * m * n * math.exp(-lam * best)
	bits    = (lam * best - math.log(K)) / math.log(2)
	# 6. Build the per-column match-symbol line: | identical, + similar, ' ' otherwise.
	mid = ''.join(
		' ' if a == '-' or b == '-'
		else '|' if a == b
		else '+' if _blosum(a, b) > 0
		else ' '
		for a, b in zip(aq, as_))
	# 7. Format the header and stats lines of the BLAST-style report.
	out = [
		f'Query length={m}  Subject length={n}', '',
		(f'Score: {bits:.1f} bits ({int(best)}), '
			f'E-value: {e_value:.3e}'),
		(f'Identities: {n_id}/{aln_len} ({pct}%), '
			f'Positives: {n_pos}/{aln_len} ({pct_pos}%), '
			f'Gaps: {n_gap}/{aln_len} ({pct_gap}%)'), '']
	# 8. Emit 60-column aligned blocks with Query / midline / Sbjct tracks.
	qp, sp, w = qs, ss, 60
	for st in range(0, aln_len, w):
		bq, bm, bs = aq[st:st+w], mid[st:st+w], as_[st:st+w]
		qr, sr = len(bq) - bq.count('-'), len(bs) - bs.count('-')
		out += [
			f'Query  {qp:>6}  {bq}  {qp+qr-1}',
			f'       {"":>6}  {bm}',
			f'Sbjct  {sp:>6}  {bs}  {sp+sr-1}', '']
		qp += qr; sp += sr
	return '\n'.join(out), pct, e_value

def MSA(sequences):
	'''
	Progressive multiple sequence alignment (ClustalW-like) with BLOSUM62 and mean-field DCA
	Arguments:
	----------
		sequences : list[str] - FASTA sequences to align (at least 2 required)
	Return:
	-------
		str        : ClustalW-style formatted alignment text
		list[str]  : Gap-padded aligned sequences in input order
		list[float]: Per-column conservation score = 1 - H/log2(20), range [0, 1]
		list[float]: Per-column Shannon entropy in bits
		np.ndarray : PSSM of shape (L, 20) in AA order 'ARNDCQEGHILKMFPSTWYV'
		np.ndarray : APC-corrected mean-field DCA direct-information matrix (L, L)
	'''
	# 1. Validate input count and normalise sequences to uppercase.
	n = len(sequences)
	if n < 2:
		raise Exception('MSA requires at least 2 sequences')
	seqs   = [s.upper() for s in sequences]
	labels = [f'Seq{i+1}' for i in range(n)]
	go, ge, INF = 11, 1, float('-inf')
	# 2. Pairwise distances via BLAST (1 - pct/100, clipped to 1 on error).
	dist = np.zeros((n, n))
	for i in range(n):
		for j in range(i+1, n):
			try:
				_, pct, _ = BLAST(seqs[i], seqs[j])
				dd = 1.0 - pct / 100.0
			except Exception:
				dd = 1.0
			dist[i, j] = dist[j, i] = dd
	# 3. UPGMA guide tree: repeatedly merge closest active clusters.
	sizes  = {k: 1 for k in range(n)}
	active = list(range(n))
	d      = dist.copy()
	merge_order = []
	for _ in range(n - 1):
		bi, bj, best = -1, -1, float('inf')
		for x in range(len(active)):
			for y in range(x + 1, len(active)):
				ii, jj = active[x], active[y]
				if d[ii, jj] < best:
					best, bi, bj = d[ii, jj], ii, jj
		merge_order.append((bi, bj))
		ni, nj = sizes[bi], sizes[bj]
		for k in active:
			if k == bi or k == bj: continue
			d[bi, k] = d[k, bi] = (
				ni * d[bi, k] + nj * d[bj, k]) / (ni + nj)
		sizes[bi] += sizes[bj]
		active.remove(bj)
	# 4. Progressive profile-to-profile Needleman-Wunsch with affine gaps and BLOSUM62 column scoring.
	BM_aa = np.array(_bm, dtype=float)
	def _profile_freq(profile):
		'''Per-column frequency vector (L, 20) over the 20-letter
		BLOSUM alphabet, normalised by non-gap count. Residues outside
		the alphabet and gaps contribute zero weight.'''
		L = len(profile[0])
		F = np.zeros((L, 20))
		for row in profile:
			for ci, c in enumerate(row):
				k = _idx.get(c, -1)
				if k >= 0: F[ci, k] += 1
		denom = F.sum(axis=1, keepdims=True)
		with np.errstate(divide='ignore', invalid='ignore'):
			return np.divide(F, denom, where=(denom > 0),
				out=np.zeros_like(F))
	profiles = {k: [seqs[k]] for k in range(n)}
	for (ci, cj) in merge_order:
		p1, p2 = profiles[ci], profiles[cj]
		L1, L2 = len(p1[0]), len(p2[0])
		H  = np.zeros((L1+1, L2+1))
		E  = np.full((L1+1, L2+1), INF)
		F  = np.full((L1+1, L2+1), INF)
		tb = np.zeros((L1+1, L2+1), dtype=np.int8)
		for i in range(1, L1+1):
			H[i, 0], tb[i, 0] = -(go + ge * i), 2
		for j in range(1, L2+1):
			H[0, j], tb[0, j] = -(go + ge * j), 3
		# Precompute profile-profile BLOSUM column scores in one shot.
		Fp1 = _profile_freq(p1)
		Fp2 = _profile_freq(p2)
		with np.errstate(all='ignore'):
			CS = Fp1 @ BM_aa @ Fp2.T     # shape (L1, L2)
		for i in range(1, L1+1):
			# Vectorised F column update: depends only on H[i-1], F[i-1].
			F[i, 1:] = np.maximum(H[i-1, 1:] - go - ge, F[i-1, 1:] - ge)
			for j in range(1, L2+1):
				diag    = H[i-1, j-1] + CS[i-1, j-1]
				E[i, j] = max(H[i, j-1] - go - ge, E[i, j-1] - ge)
				h       = max(diag, E[i, j], F[i, j])
				H[i, j] = h
				tb[i, j] = (1 if h == diag
					else 2 if h == F[i, j] else 3)
		np1 = [[] for _ in p1]
		np2 = [[] for _ in p2]
		i, j = L1, L2
		while i > 0 or j > 0:
			if i == 0:
				for k in range(len(p1)): np1[k].append('-')
				for k, r in enumerate(p2): np2[k].append(r[j-1])
				j -= 1
			elif j == 0:
				for k, r in enumerate(p1): np1[k].append(r[i-1])
				for k in range(len(p2)): np2[k].append('-')
				i -= 1
			else:
				t = int(tb[i, j])
				if t == 1:
					for k, r in enumerate(p1): np1[k].append(r[i-1])
					for k, r in enumerate(p2): np2[k].append(r[j-1])
					i -= 1; j -= 1
				elif t == 2:
					for k, r in enumerate(p1): np1[k].append(r[i-1])
					for k in range(len(p2)): np2[k].append('-')
					i -= 1
				else:
					for k in range(len(p1)): np1[k].append('-')
					for k, r in enumerate(p2): np2[k].append(r[j-1])
					j -= 1
		a1 = [''.join(reversed(row)) for row in np1]
		a2 = [''.join(reversed(row)) for row in np2]
		profiles[ci] = a1 + a2
		del profiles[cj]
	final = list(profiles.values())[0]
	L   = len(final[0])
	lw  = max(max(len(lb) for lb in labels), 4)
	# 5. Per-column conservation symbol: * (all identical), : (all similar), . (mean>0), or space.
	con = []
	for ci in range(L):
		col = [final[k][ci] for k in range(n)]
		ng  = [c for c in col if c != '-']
		if not ng:
			con.append(' ')
		elif len(ng) == n and all(c == ng[0] for c in ng):
			con.append('*')
		else:
			pairs = [_blosum(a, b) for x, a in enumerate(ng)
				for b in ng[x+1:]]
			if not pairs:
				con.append('*' if len(ng) == 1 else ' ')
			elif all(s > 0 for s in pairs):
				con.append(':')
			elif sum(pairs) / len(pairs) > 0:
				con.append('.')
			else:
				con.append(' ')
	con = ''.join(con)
	# 6. ClustalW-style output block, 60 columns per block with running residue counts.
	out = [f'Multiple Sequence Alignment ({n} sequences, {L} columns)',
		'']
	pos, w = [0] * n, 60
	for st in range(0, L, w):
		for k, lb in enumerate(labels):
			blk = final[k][st:st+w]
			pos[k] += len(blk) - blk.count('-')
			out.append(f'{lb:<{lw}}  {blk}  {pos[k]}')
		out.append(f'{"":>{lw}}  {con[st:st+w]}')
		out.append('')
	# 7. Encode the MSA as an integer matrix (gap=0, AA=1..20) for downstream stats.
	alphabet = '-' + _aa
	q, B = len(alphabet), n
	a2i  = {c: i for i, c in enumerate(alphabet)}
	M = np.zeros((B, L), dtype=np.int8)
	for bi, s in enumerate(final):
		for ci, ch in enumerate(s):
			M[bi, ci] = a2i.get(ch, 0)
	# 8. Shannon entropy and normalised conservation (1 - H/log2(20)) per column.
	log2_20 = math.log2(20)
	entropy, conservation = [], []
	for ci in range(L):
		nz = M[:, ci][M[:, ci] != 0]
		if len(nz) == 0:
			entropy.append(0.0); conservation.append(0.0); continue
		p = np.bincount(nz, minlength=q)[1:] / len(nz)
		nzp = p[p > 0]
		Hc = float(-np.sum(nzp * np.log2(nzp)))
		entropy.append(round(Hc, 4))
		conservation.append(round(1.0 - Hc / log2_20, 4))
	# 9. Position-specific scoring matrix with Laplace pseudocount against uniform 1/20 background.
	pssm = np.zeros((L, 20), dtype=float)
	for ci in range(L):
		nz = M[:, ci][M[:, ci] != 0]
		counts = np.bincount(nz, minlength=q)[1:]
		pssm[ci] = np.log2((counts + 1.0) / (counts.sum() + 20.0) * 20.0)
	# 10. DCA sequence reweighting by identity clustering (theta=0.2, 80% similarity threshold).
	theta, weights = 0.2, np.ones(B)
	if B > 1:
		simthr = (1.0 - theta) * L
		eq_count = np.zeros(B)
		for a in range(B):
			for b in range(a, B):
				if a == b:
					eq_count[a] += 1; continue
				if int((M[a] == M[b]).sum()) >= simthr:
					eq_count[a] += 1; eq_count[b] += 1
		weights = 1.0 / eq_count
	Beff = float(weights.sum())
	# 11. Single-site and two-site frequencies (Beff-weighted) with lambda=0.5 pseudocount.
	Pi = np.zeros((L, q))
	for bi in range(B):
		for ci in range(L):
			Pi[ci, M[bi, ci]] += weights[bi]
	Pi /= Beff
	lam   = 0.5
	Pi_pc = (1.0 - lam) * Pi + lam / q
	def _pij_pc(i, j):
		'''On-demand q-by-q pair frequency with pseudocount and
		diagonal reset. Replaces the full (L,L,q,q) tensor.'''
		pij = np.zeros((q, q))
		np.add.at(pij, (M[:, i], M[:, j]), weights)
		pij /= Beff
		pij = (1.0 - lam) * pij + lam / (q * q)
		if i == j:
			pij[:] = 0.0
			for a in range(q):
				pij[a, a] = Pi_pc[i, a]
		return pij
	# 12. Covariance matrix with last state dropped as gauge, then invert (pseudo-inverse on failure).
	qm = q - 1
	C = np.zeros((L * qm, L * qm))
	for i in range(L):
		for j in range(L):
			pij = _pij_pc(i, j)
			block = (pij[:qm, :qm]
				- np.outer(Pi_pc[i, :qm], Pi_pc[j, :qm]))
			C[i*qm:(i+1)*qm, j*qm:(j+1)*qm] = block
	try:
		invC = np.linalg.inv(C)
	except np.linalg.LinAlgError:
		invC = np.linalg.pinv(C)
	# 13. Direct-information per residue pair via mean-field fixed-point (tolerance 1e-6, 100-iter cap).
	dca_raw = np.zeros((L, L))
	for i in range(L):
		for j in range(i + 1, L):
			W = np.ones((q, q))
			for a in range(qm):
				for b in range(qm):
					W[a, b] = math.exp(-invC[i*qm + a, j*qm + b])
			mu1, mu2 = np.ones(q) / q, np.ones(q) / q
			pi_i, pi_j = Pi_pc[i], Pi_pc[j]
			for _ in range(100):
				new_mu1 = pi_i / (mu2 @ W.T)
				new_mu2 = pi_j / (mu1 @ W)
				new_mu1 /= new_mu1.sum()
				new_mu2 /= new_mu2.sum()
				if (np.max(np.abs(new_mu1 - mu1)) < 1e-6
						and np.max(np.abs(new_mu2 - mu2)) < 1e-6):
					mu1, mu2 = new_mu1, new_mu2
					break
				mu1, mu2 = new_mu1, new_mu2
			Pdir  = W * np.outer(mu1, mu2)
			Pdir /= Pdir.sum()
			Pfac  = np.outer(pi_i, pi_j)
			mask  = (Pdir > 1e-12) & (Pfac > 1e-12)
			di = float(np.sum(
				Pdir[mask] * np.log(Pdir[mask] / Pfac[mask])))
			dca_raw[i, j] = dca_raw[j, i] = di
	# 14. Apply Average Product Correction (APC) to deflate phylogenetic and compositional bias.
	dca = np.zeros((L, L))
	if L > 1:
		row_mean   = dca_raw.sum(axis=1) / (L - 1)
		total_mean = dca_raw.sum() / (L * (L - 1))
		if total_mean > 0:
			for i in range(L):
				for j in range(L):
					if i == j: continue
					dca[i, j] = dca_raw[i, j] - (
						row_mean[i] * row_mean[j] / total_mean)
		else:
			dca = dca_raw.copy()
		np.fill_diagonal(dca, 0.0)
	return '\n'.join(out), final, conservation, entropy, pssm, dca

def Isoelectric(sequence):
	'''
	Isoelectric point (pI) of a protein via EMBOSS pKa and bisection on [0, 14]
	Arguments:
	----------
		sequence : str - Protein FASTA sequence (one-letter codes)
	Return:
	-------
		float : pH at which the protein has zero net charge, rounded to 2 decimals
	'''
	# 1. Validate input and uppercase the sequence.
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	# 2. Count titratable residues (EMBOSS pKa set: K/R/H positive, D/E/C/Y negative).
	nK, nR, nH = seq.count('K'), seq.count('R'), seq.count('H')
	nD, nE     = seq.count('D'), seq.count('E')
	nC, nY     = seq.count('C'), seq.count('Y')
	# 3. Bisect net charge on [0, 14] using pKa_NT=8.6, pKa_CT=3.6.
	lo, hi = 0.0, 14.0
	for _ in range(100):
		mid = (lo + hi) / 2.0
		pos = 1.0 / (1.0 + 10 ** (mid - 8.6))
		if nK: pos += nK / (1.0 + 10 ** (mid - 10.53))
		if nR: pos += nR / (1.0 + 10 ** (mid - 12.48))
		if nH: pos += nH / (1.0 + 10 ** (mid -  6.00))
		neg = 1.0 / (1.0 + 10 ** (3.6 - mid))
		if nD: neg += nD / (1.0 + 10 ** ( 3.65 - mid))
		if nE: neg += nE / (1.0 + 10 ** ( 4.25 - mid))
		if nC: neg += nC / (1.0 + 10 ** ( 8.33 - mid))
		if nY: neg += nY / (1.0 + 10 ** (10.07 - mid))
		c = pos - neg
		if abs(c) < 1e-4: break
		if c > 0: lo = mid
		else:     hi = mid
	return round(mid, 2)

def Hydrophobicity(sequence, window=9, scale='eisenberg'):
	'''
	Sliding-window hydrophobicity profile (ProtScale-style)
	Arguments:
	----------
		sequence : str - Protein FASTA sequence
		window   : int - Odd window size (default 9)
		scale    : str - Scale name: 'eisenberg', 'kyte-doolittle', 'hopp-woods', or 'engelman'
	Return:
	-------
		list[int]  : 0-indexed centre position of each window
		list[float]: Mean hydrophobicity score in each window, rounded to 3 decimals
	'''
	# 1. Declare the four supported ProtScale hydrophobicity tables.
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
	# 2. Validate window size against sequence length and pick the scale table.
	seq, L = sequence.upper(), len(sequence)
	if window < 1: raise Exception('window must be >= 1')
	if window > L: raise Exception(
		f'window ({window}) larger than sequence ({L})')
	tbl = _HPHOB_SCALES.get(scale.lower())
	if tbl is None: raise Exception(
		f'Unknown scale {scale!r}; choose from '
		f'{list(_HPHOB_SCALES)}')
	# 3. Slide the window, emitting the centre position and mean score per window.
	half, n = (window - 1) // 2, L - window + 1
	return([i + half for i in range(n)],
		[round(sum(tbl.get(seq[i+k], 0.0)
			for k in range(window)) / window, 3) for i in range(n)])

def Aliphatic(sequence):
	'''
	Aliphatic index AI = X(A) + 2.9*X(V) + 3.9*(X(I) + X(L)) from mole percentages (Ikai 1980)
	Arguments:
	----------
		sequence : str - Protein FASTA sequence
	Return:
	-------
		float : Aliphatic index, rounded to 2 decimals
	'''
	# 1. Validate input and uppercase the sequence.
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	# 2. Compute mole percentages of aliphatic residues A, V, I, L.
	xA, xV, xI, xL = (100.0 * seq.count(a) / len(seq) for a in 'AVIL')
	# 3. Weighted sum per Ikai 1980.
	return round(xA + 2.9 * xV + 3.9 * (xI + xL), 2)

def ExtinctCoeff(sequence, reduced=True):
	'''
	Molar extinction coefficient at 280 nm via eps = nW*5500 + nY*1490 + (nC/2)*125 (Pace 1995)
	Arguments:
	----------
		sequence : str  - Protein FASTA sequence
		reduced  : bool - True (default) treats Cys as reduced (no contribution); False as cystines
	Return:
	-------
		int : Molar extinction coefficient in M^-1 cm^-1
	'''
	# 1. Validate input and uppercase the sequence.
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	# 2. Sum W and Y contributions; add C/2 contribution only when oxidised.
	eps = (seq.count('W') * 5500 + seq.count('Y') * 1490
		+ (0 if reduced else (seq.count('C') // 2) * 125))
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
	Instability index II = (10/L)*sum DIWV(seq[i], seq[i+1]); <40 suggests stable (Guruprasad 1990)
	Arguments:
	----------
		sequence : str - Protein FASTA sequence
	Return:
	-------
		float : Instability index, rounded to 2 decimals; 0.0 for single-residue input
	'''
	# 1. Validate input, uppercase, and short-circuit on single-residue sequences.
	if not sequence: raise Exception('Empty sequence')
	seq, L = sequence.upper(), len(sequence)
	if L < 2: return 0.0
	# 2. Sum DIWV dipeptide values across the sequence; unknown dipeptides contribute 0.
	total = sum(_DIWV.get(seq[i], {}).get(seq[i+1], 0) for i in range(L - 1))
	# 3. Normalise by length and scale by 10.
	return round(10.0 * total / L, 2)

# Kyte-Doolittle hydropathy (used by GRAVY)
_KD = {
	'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
	'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
	'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
	'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2}

def GRAVY(sequence):
	'''
	Grand average of hydropathy (mean Kyte-Doolittle hydropathy, Kyte & Doolittle 1982)
	Arguments:
	----------
		sequence : str - Protein FASTA sequence
	Return:
	-------
		float : Mean Kyte-Doolittle hydropathy, rounded to 3 decimals
	'''
	# 1. Validate input.
	if not sequence: raise Exception('Empty sequence')
	# 2. Mean KD hydropathy over the uppercased sequence (unknown residues contribute 0).
	return round(sum(_KD.get(a, 0.0)
		for a in sequence.upper()) / len(sequence), 3)

def Split(pose, chain=None, start=None, end=None):
	'''
	Extract a slice of a Pose (by chain or residue range) into a new densely-renumbered Pose
	Arguments:
	----------
		pose  : Pose - Source protein, DNA, or RNA pose
		chain : str  - Chain ID to extract (mutually exclusive with start/end)
		start : int  - First residue index to keep (inclusive, zero-based)
		end   : int  - Last residue index to keep (inclusive, zero-based)
	Return:
	-------
		Pose : New pose with atoms, residues, bonds, and coordinates renumbered from zero
	'''
	# 1. Import Pose locally to avoid a circular import at module load time.
	try:    from .pose import Pose
	except ImportError: from pose import Pose
	# 2. Reject ambiguous arg combos: exactly one of chain= or (start=, end=) must be given.
	if (chain is None) == (start is None and end is None):
		raise Exception("Split requires either chain= OR (start=, end=)")
	# 3. Resolve molecule type and fetch the residue table.
	mol = pose.data.get('Type')
	if mol is None: raise Exception('Source pose is empty')
	is_pro = (mol == 'Protein')
	rk  = 'Amino Acids' if is_pro else 'Nucleotides'
	src = pose.data[rk]
	if not src: raise Exception(f'Source pose has no {rk}')
	all_idx = sorted(src.keys())
	# 4. Select the residue indices to retain, based on chain or range mode.
	if chain is not None:
		keep_res = [i for i in all_idx if src[i][1] == chain]
		if not keep_res:
			raise Exception(f'Chain {chain!r} not in pose')
	else:
		if start is None or end is None:
			raise Exception('Split needs both start and end for range mode')
		if start > end:
			raise Exception(f'start ({start}) > end ({end})')
		keep_res = [i for i in all_idx if start <= i <= end]
		if not keep_res:
			raise Exception(f'Range [{start}, {end}] selects no residues')
	# 5. Collect kept atom indices and build dense remaps for atoms and residues.
	keep_atoms = sorted({ai for ri in keep_res
		for ai in src[ri][2] + src[ri][3]})
	a_remap = {old: new for new, old in enumerate(keep_atoms)}
	r_remap = {old: new for new, old in enumerate(keep_res)}
	src_atoms, src_bonds, src_co = (pose.data['Atoms'],
		pose.data['Bonds'], pose.data['Coordinates'])
	# 6. Build the new pose's data skeleton with remapped atoms, bonds, and coordinates.
	new = Pose()
	new.data = {
		'Type':        mol,  'Energy': 0, 'Rg': 0, 'Mass': 0,
		'Size':        {},   'FASTA':  {}, 'SS': {},
		'Nucleotides': None if is_pro else {},
		'Amino Acids': {} if is_pro else None,
		'Atoms':       {a_remap[o]: list(src_atoms[o])
			for o in keep_atoms},
		'Bonds':       {a_remap[o]: sorted(a_remap[ob]
			for ob in src_bonds.get(o, []) if ob in a_remap)
			for o in keep_atoms},
		'Coordinates': np.array([src_co[o] for o in keep_atoms],
			dtype=float) if keep_atoms else np.zeros((0, 3))}
	# 7. Copy residue rows with atom lists translated to the new indices.
	tgt = new.data[rk]
	for old_ri in keep_res:
		row = list(src[old_ri])
		row[2] = [a_remap[a] for a in row[2] if a in a_remap]
		row[3] = [a_remap[a] for a in row[3] if a in a_remap]
		tgt[r_remap[old_ri]] = row
	# 8. Refresh derived fields (Size, FASTA, SS, Mass, Rg) and return.
	new._update()
	return new

def Concatenate(pose1, pose2, fuse=False):
	'''
	Combine two poses of the same Type by chain-appending or by rebuilding a fused polymer
	Arguments:
	----------
		pose1 : Pose - First pose (protein, DNA, or RNA)
		pose2 : Pose - Second pose; must share Type with pose1
		fuse  : bool - False appends pose2 chains (colliding IDs renamed); True rebuilds as one polymer
	Return:
	-------
		Pose : New combined pose; fuse=True discards original coordinates and idealises geometry
	'''
	# 1. Import Pose locally to avoid a circular import at module load time.
	try:    from .pose import Pose
	except ImportError: from pose import Pose
	# 2. Validate both poses are non-empty and share the same molecule type.
	t1, t2 = pose1.data.get('Type'), pose2.data.get('Type')
	if t1 is None or t2 is None:
		raise Exception('Concatenate: empty pose given')
	if t1 != t2:
		raise Exception(f'Cannot concatenate {t1} with {t2}')
	is_pro = (t1 == 'Protein')
	rk     = 'Amino Acids' if is_pro else 'Nucleotides'
	# 3. Fuse mode: rebuild a single idealised polymer from the concatenated FASTA.
	if fuse:
		f1, f2 = pose1.data['FASTA'], pose2.data['FASTA']
		new = Pose()
		new.Build(''.join(f1[c] for c in sorted(f1))
			+ ''.join(f2[c] for c in sorted(f2)), fmt=t1)
		return new
	# 4. Append mode: initialise the new pose's data skeleton.
	new = Pose()
	new.data = {
		'Type': t1, 'Energy': 0, 'Rg': 0, 'Mass': 0,
		'Size': {}, 'FASTA': {}, 'SS': {},
		'Nucleotides': None if is_pro else {},
		'Amino Acids': {} if is_pro else None,
		'Atoms': {}, 'Bonds': {}, 'Coordinates': np.zeros((0, 3))}
	# 5. Two-pass copy: pose1 first, then compute pose2 chain-collision remap, then pose2.
	coords_all = []
	ai_off, ri_off = 0, 0
	ch_remap = {}
	for step, src_pose in enumerate((pose1, pose2)):
		if step == 1:
			taken = {v[1] for v in new.data[rk].values()}
			for c in sorted({v[1] for v in pose2.data[rk].values()}):
				if c not in taken:
					taken.add(c); continue
				for cand in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
					if cand not in taken:
						taken.add(cand); ch_remap[c] = cand; break
				else:
					raise Exception('Ran out of chain letters')
		src_at, src_bd = src_pose.data['Atoms'], src_pose.data['Bonds']
		src_co, src_aa = src_pose.data['Coordinates'], src_pose.data[rk]
		old_a = sorted(src_at.keys())
		a_map = {oa: ai_off + i for i, oa in enumerate(old_a)}
		for oa in old_a:
			new.data['Atoms'][a_map[oa]] = list(src_at[oa])
			coords_all.append(src_co[oa])
		for oa in old_a:
			new.data['Bonds'][a_map[oa]] = sorted(
				a_map[ob] for ob in src_bd.get(oa, []) if ob in a_map)
		old_r = sorted(src_aa.keys())
		for i, ori in enumerate(old_r):
			row = list(src_aa[ori])
			row[1] = ch_remap.get(row[1], row[1])
			row[2] = [a_map[a] for a in row[2] if a in a_map]
			row[3] = [a_map[a] for a in row[3] if a in a_map]
			new.data[rk][ri_off + i] = row
		ai_off += len(old_a)
		ri_off += len(old_r)
	# 6. Finalise coordinates array and refresh derived fields.
	new.data['Coordinates'] = (np.array(coords_all, dtype=float)
		if coords_all else np.zeros((0, 3)))
	new._update()
	return new

def _has_hairpin(seq, cmap, min_stem=4, min_loop=3):
	'''True if seq has an internal inverted repeat of length >= min_stem
	separated by a loop of >= min_loop nt. Crude proxy for hairpin stability.'''
	L = len(seq)
	for stem in range(min_stem, L // 2 + 1):
		for i in range(L - 2 * stem - min_loop + 1):
			target = seq[i:i+stem][::-1].translate(cmap)
			j = seq.find(target, i + stem + min_loop)
			if j != -1:
				return True
	return False

def _has_3p_selfdimer(seq, cmap, window=5):
	'''True if the 3' last `window` bases of seq pair elsewhere in seq
	(ignoring the trivial self-overlap at the end).'''
	tail_rc = seq[-window:][::-1].translate(cmap)
	hit = seq.find(tail_rc)
	return hit != -1 and hit + window <= len(seq) - 1

def _has_cross_dimer(fwd, rev, cmap, window=5):
	'''True if the 3' tail of either primer pairs with the other primer.'''
	a = fwd[-window:][::-1].translate(cmap)
	b = rev[-window:][::-1].translate(cmap)
	return (a in rev) or (b in fwd)

def PCR(dna_sequence):
	'''
	Design forward and reverse PCR primers for a DNA template via a 5-tier relaxation search
	Arguments:
	----------
		dna_sequence : str - Template DNA sequence (A/C/G/T only, length >= 36 bp)
	Return:
	-------
		str : Forward primer (5' end of template)
		str : Reverse primer (reverse complement of 3' end of template)
		str : Suboptimal-tier warning, or None if the Ideal tier was satisfied
	'''
	# 1. Validate: uppercase, reject illegal bases, require at least 36 bp template.
	seq = dna_sequence.upper()
	for ch in seq:
		if ch not in 'ACGT':
			raise Exception(f'Illegal base {ch!r} in template')
	if len(seq) < 36:
		raise Exception('Template too short for primer design (<36 bp)')
	# 2. Reverse-complement the template via str.translate (replaces _revcomp helper).
	cmap = str.maketrans('ACGTN', 'TGCAN')
	rc = seq[::-1].translate(cmap)
	# 3. SantaLucia 1998 nearest-neighbor thermodynamics (dH kcal/mol, dS cal/mol/K).
	DH = {'AA':-7.9,'TT':-7.9,'AT':-7.2,'TA':-7.2,
		'CA':-8.5,'TG':-8.5,'GT':-8.4,'AC':-8.4,
		'CT':-7.8,'AG':-7.8,'GA':-8.2,'TC':-8.2,
		'CG':-10.6,'GC':-9.8,'GG':-8.0,'CC':-8.0}
	DS = {'AA':-22.2,'TT':-22.2,'AT':-20.4,'TA':-21.3,
		'CA':-22.7,'TG':-22.7,'GT':-22.4,'AC':-22.4,
		'CT':-21.0,'AG':-21.0,'GA':-22.2,'TC':-22.2,
		'CG':-27.2,'GC':-24.4,'GG':-19.9,'CC':-19.9}
	# 4. Relaxation tier table: Ideal -> Good -> Fair -> Poor -> Last resort.
	tiers = [
		{'label':'Ideal',      'len':(18,25),'gc':(40.0,60.0),
			'tm':(55.0,65.0),'clamp':True, 'max_run':4,
			'no_hairpin':True, 'no_dimer':True,
			'no_cross_dimer':True, 'dtm':2.0},
		{'label':'Good',       'len':(18,28),'gc':(35.0,65.0),
			'tm':(50.0,68.0),'clamp':True, 'max_run':5,
			'no_hairpin':True, 'no_dimer':True,
			'no_cross_dimer':True, 'dtm':3.0},
		{'label':'Fair',       'len':(18,30),'gc':(25.0,75.0),
			'tm':(45.0,72.0),'clamp':False,'max_run':5,
			'no_hairpin':False,'no_dimer':True,
			'no_cross_dimer':False,'dtm':5.0},
		{'label':'Poor',       'len':(18,30),'gc':None,'tm':None,
			'clamp':False,'max_run':None,
			'no_hairpin':False,'no_dimer':False,
			'no_cross_dimer':False,'dtm':8.0},
		{'label':'Last resort','len':(18,30),'gc':None,'tm':None,
			'clamp':False,'max_run':None,
			'no_hairpin':False,'no_dimer':False,
			'no_cross_dimer':False,'dtm':float('inf')}]
	# 5. Walk tiers from strict to permissive, building candidate pools and pairing primers.
	max_off = max(0, min(60, len(seq) - 18))
	chosen, chosen_tier = None, None
	for ti, tier in enumerate(tiers):
		# 5a. Build fwd_pool (from seq) and rev_pool (from rc); all helper logic inlined.
		fwd_pool, rev_pool = [], []
		lo, hi = tier['len']
		for source, pool in ((seq, fwd_pool), (rc, rev_pool)):
			for off in range(max_off + 1):
				region = source[off:]
				for L in range(lo, hi + 1):
					if L > len(region): continue
					cand = region[:L]
					if tier['clamp'] and cand[-1] not in 'GC':
						continue
					gc = 100.0 * (cand.count('G') + cand.count('C')) / L
					if tier['gc'] is not None:
						glo, ghi = tier['gc']
						if not (glo <= gc <= ghi): continue
					mr = tier['max_run']
					if mr is not None and any(
							b * mr in cand for b in 'ACGT'):
						continue
					if tier['no_hairpin'] and _has_hairpin(cand, cmap):
						continue
					if tier['no_dimer'] and _has_3p_selfdimer(cand, cmap):
						continue
					# 5b. Tm via SantaLucia 1998 with Owczarzy 2004 salt correction.
					dH = dS = 0.0
					for i in range(L - 1):
						nn = cand[i:i+2]
						dH += DH.get(nn, 0.0)
						dS += DS.get(nn, 0.0)
					if cand[0]  in 'GC': dH += 0.1; dS += -2.8
					else:                dH += 2.3; dS +=  4.1
					if cand[-1] in 'GC': dH += 0.1; dS += -2.8
					else:                dH += 2.3; dS +=  4.1
					dS_salt = dS + 0.368 * (L - 1) * math.log(0.05)
					tm = ((dH * 1000.0) / (dS_salt
						+ 1.987 * math.log(250e-9 / 4.0))) - 273.15
					if tier['tm'] is not None:
						tlo, thi = tier['tm']
						if not (tlo <= tm <= thi): continue
					pool.append((off, cand, tm, gc))
		if not fwd_pool or not rev_pool: continue
		# 5c. Score every fwd/rev pair under the tier's dTm gate; keep the best.
		best, best_score = None, float('inf')
		dtm_max = tier['dtm']
		no_xd = tier['no_cross_dimer']
		for off1, fwd, tmf, gcf in fwd_pool:
			for off2, rev, tmr, gcr in rev_pool:
				dT = abs(tmf - tmr)
				if dT > dtm_max: continue
				if no_xd and _has_cross_dimer(fwd, rev, cmap): continue
				score = (dT * 5.0 + abs(tmf - 60.0) + abs(tmr - 60.0)
					+ abs(gcf - 50.0) * 0.1 + abs(gcr - 50.0) * 0.1
					+ (off1 + off2) * 0.05)
				if score < best_score:
					best_score = score
					best = (fwd, rev, tmf, tmr, gcf, gcr)
		if best is not None:
			chosen, chosen_tier = best, ti
			break
	if chosen is None:
		raise Exception('No primer pair found even at last-resort tier')
	# 6. Build a suboptimal-tier warning message if the Ideal tier failed.
	fwd, rev, tmf, tmr, gcf, gcr = chosen
	msg = None
	if chosen_tier > 0:
		reasons = []
		if not (40.0 <= gcf <= 60.0 and 40.0 <= gcr <= 60.0):
			reasons.append('GC% outside 40-60')
		if not (55.0 <= tmf <= 65.0 and 55.0 <= tmr <= 65.0):
			reasons.append('Tm outside 55-65 \u00b0C')
		if abs(tmf - tmr) > 2.0:
			reasons.append('|\u0394Tm| > 2 \u00b0C')
		if fwd[-1] not in 'GC' or rev[-1] not in 'GC':
			reasons.append('GC clamp missing')
		if _has_cross_dimer(fwd, rev, cmap):
			reasons.append('primer-pair cross-dimer')
		reason = '; '.join(reasons) if reasons else 'gates relaxed'
		msg = (f'Warning: Suboptimal PCR primers '
			f'({tiers[chosen_tier]["label"]} tier) \u2014 {reason}')
	return (fwd, rev, msg)

def Translate(sequence, fmt='protein', organism='ecoli'):
	'''
	Translate between DNA, RNA, and protein with auto-detected source alphabet
	Arguments:
	----------
		sequence : str - Input sequence (alphabet auto-detected: DNA, RNA, or protein)
		fmt      : str - Target alphabet: 'protein' (default), 'dna', or 'rna'
		organism : str - Codon usage for back-translation: 'ecoli' (default) or 'human'
	Return:
	-------
		str : Translated sequence (uppercase, with gaps and spaces stripped)
	'''
	# 1. Validate input and target format.
	if not sequence: raise Exception('Empty sequence')
	tgt = fmt.lower()
	if tgt not in ('protein', 'dna', 'rna'):
		raise Exception(f'Unknown target fmt: {fmt}')
	# 2. Detect source alphabet by character set (gap, *, N excluded from the test).
	chars = set(sequence.upper()) - {'-', '*', 'N'}
	if not chars: src = 'protein'
	elif chars <= set('ACGT'): src = 'dna'
	elif chars <= set('ACGU'): src = 'rna'
	elif chars <= set('ACDEFGHIKLMNPQRSTVWY'): src = 'protein'
	elif chars - set('ACGT') - set('ACGU'): src = 'protein'
	else: src = 'dna'
	# 3. Normalise: uppercase, strip gaps and spaces.
	s = sequence.upper().replace('-', '').replace(' ', '')
	# 4. Identity and DNA<->RNA alphabet swaps.
	if src == tgt: return s
	if src == 'dna' and tgt == 'rna': return s.replace('T', 'U')
	if src == 'rna' and tgt == 'dna': return s.replace('U', 'T')
	# 5. Nucleotide -> protein via the standard genetic code ('*' = stop, unknown codons -> 'X').
	if src in ('dna', 'rna') and tgt == 'protein':
		CODON = {
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
		dna = s.replace('U', 'T')
		dna = dna[:len(dna) - len(dna) % 3]
		return ''.join(CODON.get(dna[i:i+3], 'X')
			for i in range(0, len(dna), 3))
	# 6. Protein -> nucleotide via highest-weight Kazusa codon per amino acid.
	if src == 'protein' and tgt in ('dna', 'rna'):
		BEST = {
			'ecoli': {'F':'TTT','L':'CTG','I':'ATT','M':'ATG','V':'GTG',
				'S':'AGC','P':'CCG','T':'ACC','A':'GCG','Y':'TAT',
				'*':'TAA','H':'CAT','Q':'CAG','N':'AAC','K':'AAA',
				'D':'GAT','E':'GAA','C':'TGC','W':'TGG','R':'CGC','G':'GGC'},
			'human': {'F':'TTC','L':'CTG','I':'ATC','M':'ATG','V':'GTG',
				'S':'AGC','P':'CCC','T':'ACC','A':'GCC','Y':'TAC',
				'*':'TGA','H':'CAC','Q':'CAG','N':'AAC','K':'AAG',
				'D':'GAC','E':'GAG','C':'TGC','W':'TGG','R':'CGG','G':'GGC'}
			}.get(organism.lower())
		if BEST is None:
			raise Exception(
				f"Unknown organism {organism!r}; use 'ecoli' or 'human'")
		out = []
		for aa in s:
			c = BEST.get(aa)
			if c is None:
				raise Exception(f'No codon for residue {aa!r}')
			out.append(c)
		dna = ''.join(out)
		return dna if tgt == 'dna' else dna.replace('T', 'U')
	raise Exception(f'Unsupported translation {src} -> {tgt}')

def PROSITE(sequence, pattern):
	'''
	Search a protein sequence for a PROSITE-style pattern (subset grammar: [..], {..}, x(n,m), < >)
	Arguments:
	----------
		sequence : str - Protein sequence to search
		pattern  : str - PROSITE pattern using literals, [ABC], {ABC}, x, x(n), x(n,m), < anchors >
	Return:
	-------
		list[tuple] : Each hit is (start, end, match) with 1-based inclusive positions
	'''
	# 1. Validate: empty pattern is fatal; empty sequence yields no hits.
	if not pattern: raise Exception('Empty pattern')
	if not sequence: return []
	# 2. Tokenise PROSITE pattern into regex: [..]/{..} -> char classes, x -> '.', < > -> ^ $.
	p = pattern.replace('-', '').replace(' ', '')
	out, i = [], 0
	while i < len(p):
		c = p[i]
		if   c == '<':       out.append('^'); i += 1
		elif c == '>':       out.append('$'); i += 1
		elif c in '[{':
			close = ']' if c == '[' else '}'
			j = p.find(close, i)
			if j == -1:
				raise Exception(f'Unclosed {c} in pattern')
			out.append(('[' if c == '[' else '[^') + p[i+1:j] + ']')
			i = j + 1
		elif c in 'xX':      out.append('.'); i += 1
		elif c.isalpha():    out.append(c.upper()); i += 1
		else:
			raise Exception(
				f'Unexpected character {c!r} at position {i} of pattern')
		if i < len(p) and p[i] == '(':
			j = p.find(')', i)
			if j == -1: raise Exception('Unclosed ( in pattern')
			body = p[i+1:j]
			out.append('{' + ','.join(
				s.strip() for s in body.split(',', 1)) + '}')
			i = j + 1
	# 3. Compile as a zero-width lookahead so overlapping hits are found; scan the sequence.
	rx = re.compile('(?=(' + ''.join(out) + '))', re.IGNORECASE)
	return [(m.start() + 1, m.start() + len(m.group(1)), m.group(1))
		for m in rx.finditer(sequence)]

def HydrogenBondMap(pose):
	'''
	Backbone H-bond donor/acceptor map via the DSSP electrostatic criterion (Kabsch-Sander 1983)
	Arguments:
	----------
		pose : Pose - Protein pose with backbone N, C, O atoms
	Return:
	-------
		np.ndarray : (N_atoms, N_atoms) int8 matrix; 0 = no bond, 1 = donor N, 2 = acceptor O
	'''
	# 1. Validate molecule type and presence of amino-acid residues.
	if pose.data.get('Type') != 'Protein':
		raise Exception('HydrogenBondMap only supports protein poses')
	AAs = pose.data.get('Amino Acids') or {}
	if not AAs:
		raise Exception('Pose has no amino acids')
	# 2. Allocate output matrix and gather residue indices, chain IDs, and tricodes.
	atoms = pose.data['Atoms']
	co    = pose.data['Coordinates']
	N_atoms = max(atoms.keys()) + 1 if atoms else 0
	M = np.zeros((N_atoms, N_atoms), dtype=np.int8)
	res_idx  = sorted(AAs.keys())
	N_res    = len(res_idx)
	chains   = [AAs[r][1] for r in res_idx]
	tricodes = [AAs[r][5].upper() for r in res_idx]
	# 3. Precompute per-residue backbone atom-name -> atom-index lookup.
	ai_of = {r: {atoms[ai][0]: ai for ai in AAs[r][2]} for r in res_idx}
	# 4. Place virtual amide-H: use explicit H/1H when available, else N + unit(C_{i-1}->O_{i-1}).
	H_pos = [None] * N_res
	for k, r in enumerate(res_idx):
		if tricodes[k] == 'PRO': continue
		if k == 0 or chains[k] != chains[k-1]: continue
		idx = ai_of[r]
		ah = idx.get('H', idx.get('1H'))
		if ah is not None:
			H_pos[k] = co[ah]
			continue
		prev = ai_of[res_idx[k-1]]
		if 'N' in idx and 'C' in prev and 'O' in prev:
			cdir = co[prev['C']] - co[prev['O']]
			nm = float(np.linalg.norm(cdir))
			if nm > 0.001:
				H_pos[k] = co[idx['N']] + cdir / nm
	# 5. For every (i, j) pair on the same chain with |i-j|>1, apply DSSP energy threshold E<-0.5.
	for ki in range(N_res):
		if H_pos[ki] is None: continue
		Ni_idx = ai_of[res_idx[ki]].get('N', -1)
		if Ni_idx < 0: continue
		Ni, Hi = co[Ni_idx], H_pos[ki]
		for kj in range(N_res):
			if abs(ki - kj) <= 1 or chains[ki] != chains[kj]: continue
			idxj = ai_of[res_idx[kj]]
			if 'O' not in idxj or 'C' not in idxj: continue
			Cj_idx, Oj_idx = idxj['C'], idxj['O']
			Cj, Oj = co[Cj_idx], co[Oj_idx]
			r_ON = float(np.linalg.norm(Oj - Ni))
			r_CH = float(np.linalg.norm(Cj - Hi))
			r_OH = float(np.linalg.norm(Oj - Hi))
			r_CN = float(np.linalg.norm(Cj - Ni))
			if min(r_ON, r_CH, r_OH, r_CN) < 0.001: continue
			if 0.084 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332 < -0.5:
				M[Ni_idx, Oj_idx] = 1
				M[Oj_idx, Ni_idx] = 2
	return M

def ContactMap(pose):
	'''
	Residue-residue Euclidean distance map (angstroms) using CA for protein, C1' for DNA/RNA
	Arguments:
	----------
		pose : Pose - Protein or nucleic-acid pose with a non-empty residue table
	Return:
	-------
		np.ndarray : (N_residues, N_residues) pairwise distances, zero on the diagonal
	'''
	# 1. Resolve molecule type, residue table, and reference-atom name.
	mol = pose.data.get('Type')
	if mol is None: raise Exception('Empty pose')
	if   mol == 'Protein':      src, ref = pose.data['Amino Acids'], 'CA'
	elif mol in ('DNA', 'RNA'): src, ref = pose.data['Nucleotides'], "C1'"
	else: raise Exception(f'Unknown molecule type: {mol}')
	if not src: raise Exception('Pose has no residues')
	# 2. Gather reference-atom coordinates for every residue (must exist per residue).
	atoms, co = pose.data['Atoms'], pose.data['Coordinates']
	keys = sorted(src.keys())
	pts  = np.zeros((len(keys), 3))
	for k, ri in enumerate(keys):
		pos = next((co[ai] for ai in src[ri][2]
			if atoms[ai][0] == ref), None)
		if pos is None:
			raise Exception(f'Residue {ri} has no {ref} atom')
		pts[k] = pos
	# 3. Broadcast pairwise difference, take Euclidean norm, zero the diagonal.
	diff = pts[:, None, :] - pts[None, :, :]
	mat  = np.sqrt((diff * diff).sum(-1))
	np.fill_diagonal(mat, 0.0)
	return mat


