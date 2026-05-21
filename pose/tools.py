import re
import os
import sys
import json
import math
import time
import shutil
import itertools
import numpy as np
import urllib.request
import xml.etree.ElementTree as ET
from .pose import DBLoad
from .energy import ForceField
from collections import defaultdict, deque

def _validate_rot_entry(rot_entry, expected_tricode):
	'''
	Validate a rotamer JSON against the Dunbrack BBDEP2010 schema.

	The JSON must come from the nnca_pipeline (build_*_rotamer.py) or
	an equivalent producer, with top-level keys
	{tricode, n_chi, rotamers, method}. The method.chi_axes field is
	required because Parameterise() uses it as the source of truth for
	the Amino Acids entry's "Chi Angle Atoms" field.

	Arguments:
	----------
		rot_entry         : dict - parsed JSON content
		expected_tricode  : str  - tricode the caller is asking us to
		                    insert; must match rot_entry['tricode']
	Returns:
	--------
		None - raises ValueError on any schema violation with explicit
		context.
	'''
	REQUIRED_TOP = ('tricode', 'n_chi', 'rotamers')
	REQUIRED_ROT = ('columns', 'table', 'bin_offsets', 'top_chi')
	PHI_N, PSI_N = 36, 36
	missing_top = [k for k in REQUIRED_TOP if k not in rot_entry]
	if missing_top:
		raise ValueError(
			f'rotamer JSON missing required keys: {missing_top}')
	tri = rot_entry['tricode']
	if not isinstance(tri, str) or len(tri) != 3:
		raise ValueError(
			f'rotamer JSON tricode must be a 3-letter str, got {tri!r}')
	if tri.upper() != expected_tricode.upper():
		raise ValueError(
			f'rotamer JSON tricode {tri!r} does not match argument '
			f'{expected_tricode!r}')
	n_chi = int(rot_entry['n_chi'])
	if n_chi < 1 or n_chi > 8:
		raise ValueError(f'n_chi out of range (1-8): {n_chi}')
	if ('method' not in rot_entry
			or 'chi_axes' not in rot_entry['method']):
		raise ValueError(
			'rotamer JSON missing method.chi_axes (required as the '
			'source of truth for Amino Acids "Chi Angle Atoms")')
	chi_axes = rot_entry['method']['chi_axes']
	if len(chi_axes) != n_chi:
		raise ValueError(
			f'method.chi_axes has {len(chi_axes)} axes but '
			f'n_chi={n_chi}')
	for k, axis in enumerate(chi_axes):
		if len(axis) != 4:
			raise ValueError(
				f'chi_axes[{k}] has {len(axis)} atoms (need 4): {axis}')
	rot = rot_entry['rotamers']
	missing_rot = [k for k in REQUIRED_ROT if k not in rot]
	if missing_rot:
		raise ValueError(f'rotamers missing keys: {missing_rot}')
	expect_cols = (['count', 'prob']
		+ [f'chi{k+1}' for k in range(n_chi)]
		+ [f'sig{k+1}' for k in range(n_chi)])
	if rot['columns'] != expect_cols:
		raise ValueError(
			f'rotamer columns mismatch.\n'
			f'  got:      {rot["columns"]}\n'
			f'  expected: {expect_cols}')
	bo_off = rot['bin_offsets']
	if len(bo_off) != PHI_N * PSI_N + 1:
		raise ValueError(
			f'bin_offsets length {len(bo_off)} != '
			f'{PHI_N * PSI_N + 1}')
	tc = rot['top_chi']
	if len(tc) != PHI_N:
		raise ValueError(
			f'top_chi outer length {len(tc)} != {PHI_N}')
	if any(len(r) != PSI_N for r in tc):
		raise ValueError(
			f'top_chi inner length != {PSI_N}')

def _clamp_sigmas(rot_entry, floor=0.5):
	'''
	Clamp sigma columns of the rotamer table to >= floor degrees.

	Some BGMM bins emit zero-width sigmas when the underlying data is
	a single delta; the unified-DB schema requires sigmas >= 0.5 deg
	to avoid divide-by-zero in Score._rotamer_prior.

	Arguments:
	----------
		rot_entry : dict
		floor     : float - minimum sigma in degrees (default 0.5)
	Returns:
	--------
		int : count of values clamped (informational only)
	'''
	n_chi = int(rot_entry['n_chi'])
	sig_col0 = 2 + n_chi
	table = rot_entry['rotamers']['table']
	n_clamped = 0
	for row in table:
		for k in range(n_chi):
			v = float(row[sig_col0 + k])
			if v < floor:
				row[sig_col0 + k] = floor
				n_clamped += 1
	return n_clamped

def Parameterise(cif_file, rotamer_json_file, tricode, unicode,
		backup=True):
	'''
	Add a non-canonical amino acid (NCAA) to Pose's unified
	database.json.

	Builds the entry under "Amino Acids"[unicode] from cif_file
	(verified RCSB Chemical Component Dictionary CIF) and inserts the
	matching backbone-dependent rotamer library under
	"Rotamer Library"["residues"][tricode] from rotamer_json_file
	(Dunbrack BBDEP2010-format JSON produced by the nnca_pipeline at
	/Users/slurm/Desktop/Research/nnca_pipeline/). Both insertions
	land in a single atomic write.

	Arguments:
	----------
		cif_file          : str  - Path to RCSB CCD CIF
		rotamer_json_file : str  - Path to Dunbrack-format rotamer JSON
		tricode           : str  - Three-letter residue code, e.g. 'PTR'
		unicode           : str  - Single-letter key for db['Amino Acids']
		backup            : bool - If True (default), copy database.json
		                    to database.json.bak.<YYYYMMDD-HHMMSS>
		                    before modifying. Set False for batch / CI
		                    runs that handle backups externally.
	Behaviour on existing keys:
	---------------------------
		If `unicode` is already a key in db['Amino Acids'], or `tricode`
		is already in db['Rotamer Library']['residues'], a warning is
		logged to stderr identifying the old entry, and the new entries
		overwrite the old.
	Returns:
	--------
		None - database.json is updated in place; DBLoad cache is cleared
		so subsequently constructed Pose / ForceField / Score / Rotamers
		instances see the new residue without restart.
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
	# 2. Load + validate the rotamer JSON FIRST. Failing fast on a bad
	#    schema means we never half-write a CIF-derived entry into the
	#    DB without a matching rotamer library.
	with open(rotamer_json_file) as fh:
		rot_entry = json.load(fh)
	_validate_rot_entry(rot_entry, tricode)
	n_clamped = _clamp_sigmas(rot_entry)
	if n_clamped:
		print(f'Note: clamped {n_clamped} rotamer sigma values to '
			f'>=0.5 deg floor.')
	chi_axes_from_json = rot_entry['method']['chi_axes']
	# 3. Parse CIF: atom rows have >=18 tokens (coords at [15:18] or fallback [12:15]); bond rows have 7 tokens.
	COORD_RAW, ATOMS_RAW, BONDS = [], [], []
	with open(cif_file) as fh:
		for line in fh:
			t = line.strip().split()
			if not t or t[0] != tricode: continue
			if len(t) == 7 and t[3] in ('SING','DOUB','TRIP','AROM'):
				BONDS.append((t[1], t[2], t[3], t[4]))
			elif len(t) >= 18:
				try:
					try:c = [float(t[i]) for i in (15, 16, 17)]
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
		raise ValueError(f'No CB atom found in {cif_file}. '
			'Only standard amino acids (not GLY) are supported.')
	# 4. Validate every chi-axis atom from the JSON exists in the CIF.
	#    Chi axes only ever reference heavy atoms, so a CIF-name lookup
	#    is sufficient (no need to canonicalise H names yet).
	cif_atom_set = set(CIF_IDS)
	for k, axis in enumerate(chi_axes_from_json):
		for an in axis:
			if an not in cif_atom_set:
				raise ValueError(
					f'chi axis {k+1} references atom {an!r} which '
					f'does not exist in {cif_file}. CIF atoms: '
					f'{sorted(cif_atom_set)}')
	bb_set = {a['id'] for a in ATOMS_RAW if a['bb']} or {
		'N','CA','C','O','OXT','H','H1','H2','H3',
		'HA','HA2','HA3','HXT'}
	elem    = {a['id']: a['elem'] for a in ATOMS_RAW}
	cif_ord = {a['id']: i for i, a in enumerate(ATOMS_RAW)}
	# 5. Superimpose onto ALA backbone frame via rigid motion (N, CA, CB, C) with CA as origin.
	try:
		Ni, CAi = CIF_IDS.index('N'),  CIF_IDS.index('CA')
		CBi, Ci = CIF_IDS.index('CB'), CIF_IDS.index('C')
	except ValueError as e:
		raise ValueError(f'Missing backbone atom in {cif_file}: {e}')
	A = np.c_[ALA,   np.ones(len(ALA))]
	B = np.c_[COORD, np.ones(len(COORD))]
	AL = np.array([A[0]-A[4], A[6]-A[4], A[-3]-A[4], A[4]])
	BL = np.array([B[Ni]-B[CAi], B[CBi]-B[CAi], B[Ci]-B[CAi], B[CAi]])
	COORD = (B @ (np.linalg.inv(BL) @ AL))[:, :3]
	# 6. Build undirected bond graph indexed by atom id.
	adj = defaultdict(set)
	for a1, a2, _v, _r in BONDS:
		adj[a1].add(a2); adj[a2].add(a1)
	# 7. BFS sidechain from CB (heavy-atom queue; H neighbours land next to their parent in CIF order).
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
	# 8. Rename atoms: CIF HB2 becomes Pose 1HB (per-base counter on H/D atoms).
	name_map, counter = {}, defaultdict(int)
	for name in ordered:
		m = re.match(r'^([A-Z]+)(\d+)$', name)
		if m and elem.get(name, '').upper() in ('H', 'D'):
			counter[m.group(1)] += 1
			name_map[name] = f'{counter[m.group(1)]}{m.group(1)}'
		else:
			name_map[name] = name
	# 9. Detect fused sidechain: any sidechain atom bonded back to N (e.g. PRO).
	fused_atom = next((sc for sc in ordered if 'N' in adj[sc]), None)
	fused = fused_atom is not None
	# 10. Sidechain bond graph on new indices; -5 sentinel stands in for the backbone N of a fused ring.
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
	# 11. Two-pass aromaticity: C with >=2 O/N neighbours and at least one double bond gets resonance (all C-O/N -> 1.5).
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
	# 12. Final bond dicts with sorted neighbours for determinism; -5 sentinel kept last if fused.
	pos_keys     = sorted(k for k in sc_bonds if k >= 0)
	final_bonds  = {k: sorted(sc_bonds[k]) for k in pos_keys}
	final_orders = {k: [dict(zip(sc_bonds[k], sc_orders[k]))[nb]
		for nb in final_bonds[k]] for k in pos_keys}
	if fused:
		final_bonds[-5]  = sorted(sc_bonds[-5])
		final_orders[-5] = [dict(zip(sc_bonds[-5], sc_orders[-5]))[nb]
			for nb in final_bonds[-5]]
	# 13. Chi axes come from the rotamer JSON's method.chi_axes (the
	#     plan's user-confirmed source of truth). The CIF walker that
	#     this section used to do is removed -- it was fragile for
	#     NCAAs with non-standard atom orderings, and any drift between
	#     the Amino Acids "Chi Angle Atoms" field and the rotamer
	#     library's chi convention silently corrupts the rotamer prior.
	chis = [list(a) for a in chi_axes_from_json]
	# 14. Assemble the new entry in the same field order as existing AAs.
	id_to_i = {cid: i for i, cid in enumerate(CIF_IDS)}
	def _infer_hybridisation(elem, bond_orders):
		'''
		Classify an atom's hybridization from its element and the list of
		bond orders incident on it. Relies on aromatic/resonance bonds
		having been encoded as order 1.5 by the caller (true of Parameterise
		after the two-pass aromaticity rewrite, and of Molecule.Import after
		bond-order inference).
		Arguments:
		----------
			elem:        str, element symbol (case-insensitive)
			bond_orders: iterable of numeric bond orders on this atom
		Returns:
		--------
			str: one of 's', 'sp', 'sp2', 'sp3'
		'''
		if elem and elem.upper() == 'H': return 's'
		bos = list(bond_orders)
		if any(bo == 3 for bo in bos):   return 'sp'
		if any(bo >= 1.5 for bo in bos): return 'sp2'
		return 'sp3'
	entry = {
		'Vectors':         [COORD[id_to_i[n]].tolist() for n in ordered],
		'Tricode':         tricode,
		'Fused':           fused,
		'Sidechain Atoms': [[name_map[n], elem[n], 0, 1.0, 0,
			_infer_hybridisation(elem[n], sc_orders[new_idx[n]])]
			for n in ordered],
		'Chi Angle Atoms': chis,
		'Bonds':           {str(k): v for k, v in final_bonds.items()},
		'BondOrders':      {str(k): v for k, v in final_orders.items()}}
	# 15. Load the existing database.
	db_path = os.path.join(
		os.path.dirname(os.path.abspath(__file__)), 'database.json')
	with open(db_path) as fh: db = json.load(fh)
	# 16. Warn-and-overwrite on key collisions, per user-confirmed plan.
	#     Both single-letter unicode (Amino Acids) and 3-letter tricode
	#     (Rotamer Library.residues) are checked independently.
	if unicode in db.get('Amino Acids', {}):
		old_tri = db['Amino Acids'][unicode].get('Tricode', '?')
		print(f'Warning: db["Amino Acids"]["{unicode}"] already '
			f'exists (was Tricode={old_tri}); overwriting with '
			f'Tricode={tricode}.', file=sys.stderr)
	rl       = db.setdefault('Rotamer Library', {})
	rl_resid = rl.setdefault('residues', {})
	if tricode in rl_resid:
		print(f'Warning: db["Rotamer Library"]["residues"]'
			f'["{tricode}"] already exists; overwriting.',
			file=sys.stderr)
	# 17. Insert both entries. The Rotamer Library form keeps only
	#     n_chi/rotamers/densities (matching merge_into_database.py);
	#     the method/metadata fields are stripped on insertion to keep
	#     database.json compact.
	db.setdefault('Amino Acids', {})[unicode] = entry
	rl_resid[tricode] = {
		'n_chi':     int(rot_entry['n_chi']),
		'rotamers':  rot_entry['rotamers'],
		'densities': rot_entry.get('densities'),
	}
	# 18. Validate Bonds/BondOrders symmetry across the whole DB before
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
	# 19. Optional timestamped backup before atomic write.
	if backup:
		ts = time.strftime('%Y%m%d-%H%M%S')
		bak_path = db_path + f'.bak.{ts}'
		shutil.copy2(db_path, bak_path)
		print(f'Backup: {bak_path}')
	# 20. Compact atomic write (no whitespace -- matches the rest of
	#     the unified-DB infrastructure).
	tmp_path = db_path + '.tmp'
	try:
		with open(tmp_path, 'w') as fh:
			json.dump(db, fh, separators=(',', ':'))
		os.replace(tmp_path, db_path)
	except BaseException:
		if os.path.exists(tmp_path):
			os.remove(tmp_path)
		raise
	# 21. Invalidate the DBLoad cache so subsequently constructed Pose
	#     / ForceField / Score / Rotamers instances see the new residue
	#     without restart.
	DBLoad.cache_clear()
	print(f'Added {tricode} as "{unicode}" to database.json '
		f'(Amino Acids + Rotamer Library)')

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
	# 5. For every (i, j) pair on the same chain with |i-j|>1, apply DSSP energy threshold E<-2.092.
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
			if 0.084*(1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 1389.35458 < -2.092:
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

def _rotlib_lookup(rotlib_root, three_letter, phi_deg, psi_deg):
	'''
	Slice the Rotamer Library CSR table for one (residue, phi, psi) cell
	Arguments:
	----------
		rotlib_root: dict - database['Rotamer Library']
		three_letter: str - 3-letter residue code (uppercase, L-form)
		phi_deg, psi_deg: float - backbone angles in degrees
	Returns:
	--------
		(n_chi, table_slice) where table_slice is a list of rotamer rows;
		(0, []) if the residue has no entry, or rows is empty if the cell
		has no rotamers.
	'''
	residues = rotlib_root.get('residues', {}) if rotlib_root else {}
	entry = residues.get(three_letter)
	if entry is None: return 0, []
	phi_start = float(rotlib_root.get('phi_start', -180.0))
	phi_step  = float(rotlib_root.get('phi_step',   10.0))
	phi_n     = int  (rotlib_root.get('phi_n',     36))
	psi_start = float(rotlib_root.get('psi_start', -180.0))
	psi_step  = float(rotlib_root.get('psi_step',   10.0))
	psi_n     = int  (rotlib_root.get('psi_n',     36))
	rot         = entry['rotamers']
	bin_offsets = rot['bin_offsets']
	i_phi = int(math.floor((phi_deg - phi_start) / phi_step)) % phi_n
	i_psi = int(math.floor((psi_deg - psi_start) / psi_step)) % psi_n
	bidx  = i_phi * psi_n + i_psi
	start = bin_offsets[bidx]
	end   = bin_offsets[bidx + 1]
	return int(entry['n_chi']), rot['table'][start:end]

def Rotamers(index, pose):
	'''
	Single-amino-acid rotamer packer: set every chi of one residue to the
	dominant (most-populated) rotamer from the Rotamer Library at that
	residue's current backbone (phi, psi).

	Algorithm (production):
	  1. Look up the residue's 3-letter code; bail out if it has no chis.
	  2. Read backbone phi, psi; bail out if either is undefined (chain end).
	  3. Snap to the nearest (phi, psi) grid cell in the Rotamer Library.
	  4. Pick the rotamer k* with maximum P_k in that cell.
	  5. Apply mu_k*_chi[c] for c = 1..n_chi via pose.RotateDihedral.

	D-amino acids (lowercase 1-letter codes) are handled via the standard
	chi/backbone mirror trick: lookup with negated phi/psi, negate predicted
	mu values when applying.
	Arguments:
	----------
		index : int - residue index in pose.data['Amino Acids']
		pose  : Pose - protein pose with a non-empty residue table
	Return:
	-------
		None - mutates the pose in place. No-op if the residue has no chis,
		undefined backbone, or no rotamer-library entry for its type.
	'''
	info  = pose.data.get('Amino Acids', {}).get(index)
	if info is None: return
	c     = info[0]
	aa_u  = c.upper()
	aa_db = pose.aminoacids.get(aa_u, {})
	chi_atoms = aa_db.get('Chi Angle Atoms') or []
	if not chi_atoms: return                # Gly, Ala -- no chis
	three = aa_db.get('Tricode')
	if not three: return
	phi = pose.GetDihedral(index, 'PHI')
	psi = pose.GetDihedral(index, 'PSI')
	if math.isnan(phi) or math.isnan(psi): return  # chain ends
	flip = (c != aa_u)
	phi_q = -phi if flip else phi
	psi_q = -psi if flip else psi
	rotlib = DBLoad().get('Rotamer Library')
	n_chi, rows = _rotlib_lookup(rotlib, three, phi_q, psi_q)
	if n_chi == 0 or not rows: return
	# Find argmax_k by P_k. Column layout: [count, prob, chi1..N, sig1..N]
	prob_i = 1
	chi_i  = 2
	best   = max(rows, key=lambda row: row[prob_i])
	for ci in range(n_chi):
		mu = best[chi_i + ci]
		if flip: mu = -mu
		pose.RotateDihedral(index, float(mu), 'CHI', ci + 1)

def Minimise(pose, ff=None, max_steps=500, ftol=1.0, dt_fs=0.5,
		dt_max_fs=1.0, step_max=0.2, etol=1e-6, stall_k=10, box=None):
	'''
	Relax pose coordinates with FIRE2 (Guenole et al. 2020). A
	trust-region cap bounds the per-atom displacement; a step that turns
	non-finite or strongly uphill is rejected, dt shrunk and retried;
	and the lowest-|force| frame ever seen is restored before returning,
	so a force-field singularity (e.g. an uncovered atom with no bond)
	can never fling atoms away or corrupt the returned structure
	Arguments:
	----------
		pose:      Pose - molecule source protein, DNA, RNA, or Molecule
		ff:        ForceField - reusable evaluator; created if None
		max_steps: int - maximum number of FIRE2 iterations
		ftol:      float - convergence on max|force| (L_inf) in kJ/mol/A
		dt_fs:     float - initial integrator step in femtoseconds
		dt_max_fs: float - upper bound on the adaptive step in fs
		step_max:  float - trust-region cap on per-atom displacement in A
		etol:      float - energy-stall tolerance in kJ/mol
		stall_k:   int - consecutive stalled steps that trigger early stop
		box:       None for no PBC; (3,) orthorhombic; (3, 3) triclinic
	Returns:
	--------
		tuple: (float, dict) - energy of the best frame in kJ/mol and a
		per-step log ('energies', 'fmax', 'max_step', 'converged',
		'n_steps')
	'''
	if ff is None: ff = ForceField()
	N_MIN, F_INC, F_DEC = 5, 1.1, 0.5
	A_START, F_ALPHA = 0.1, 0.99
	AKMA_FS = 23.91888086
	atoms = pose.data['Atoms']
	m = np.array([pose.masses[atoms[i][1]] for i in sorted(atoms)],
		dtype=np.float64)[:, None]
	v = np.zeros_like(pose.data['Coordinates'], dtype=np.float64)
	dt     = float(dt_fs) / AKMA_FS
	dt_max = float(dt_max_fs) / AKMA_FS
	dt_min = dt * 1e-3
	alpha, n_pos = float(A_START), 0
	energies, fmaxes, max_steps_log = [], [], []
	E, F = ff(pose, grad=True, box=box)
	E = float(E)
	best_fmax   = float(np.max(np.abs(F)))
	best_coords = pose.data['Coordinates'].copy()
	converged, steps_done, stall = False, 0, 0
	for step in range(int(max_steps)):
		fmax = float(np.max(np.abs(F)))
		energies.append(E); fmaxes.append(fmax)
		steps_done = step + 1
		# Remember the lowest-|force| frame; it is restored at the end.
		if np.isfinite(fmax) and fmax < best_fmax:
			best_fmax   = fmax
			best_coords = pose.data['Coordinates'].copy()
		if fmax < ftol or stall >= stall_k:
			converged = True
			break
		# Stop a clear divergence early (a FF singularity); the best
		# frame is restored below, so the returned pose stays intact.
		if (not np.isfinite(fmax)) or (fmax > 1e4
				and fmax > 1e3 * best_fmax):
			break
		# FIRE: mix the velocity toward the force when the power
		# P = F . v is positive (downhill); zero it and shrink dt on
		# an uphill power.
		P = float(np.sum(F * v))
		if P > 0.0:
			f_norm = float(np.linalg.norm(F))
			v_norm = float(np.linalg.norm(v))
			mix = (alpha * v_norm / f_norm) if f_norm > 1e-12 else 0.0
			v = (1.0 - alpha) * v + mix * F
			n_pos += 1
			if n_pos > N_MIN:
				dt = min(dt * F_INC, dt_max)
				alpha *= F_ALPHA
		else:
			v = np.zeros_like(v)
			dt = max(dt * F_DEC, dt_min)
			alpha, n_pos = A_START, 0
		# Semi-implicit Euler step; clamp the per-atom DISPLACEMENT
		# (not the velocity, which carries FIRE's persistent momentum).
		v = v + dt * F / m
		dr = dt * v
		nrm = np.linalg.norm(dr, axis=1, keepdims=True)
		dr = dr * np.minimum(1.0, step_max / np.maximum(nrm, 1e-12))
		max_steps_log.append(float(np.max(np.abs(dr))))
		x_old = pose.data['Coordinates']
		pose.data['Coordinates'] = x_old + dr
		E_new, F_new = ff(pose, grad=True, box=box)
		E_new = float(E_new)
		fmax_new = float(np.max(np.abs(F_new)))
		# Safeguard: undo a step that is non-finite, strongly uphill, or
		# whose force explodes (a downhill run into a FF singularity);
		# zero the velocity and shrink dt so the retry is smaller.
		bad = (not np.isfinite(E_new)
			or not np.isfinite(F_new).all()
			or E_new > E + 1.0 + 0.05 * abs(E)
			or (fmax_new > 1e3 and fmax_new > 100.0 * max(fmax, 1.0)))
		if bad:
			pose.data['Coordinates'] = x_old
			v = np.zeros_like(v)
			dt = max(dt * F_DEC, dt_min)
			alpha, n_pos = A_START, 0
			continue
		stall = stall + 1 if abs(E_new - E) < etol else 0
		E, F = E_new, F_new
	# Restore the lowest-|force| frame and report its energy.
	pose.data['Coordinates'] = best_coords
	E, F = ff(pose, grad=True, box=box)
	log = {
		'energies':  np.asarray(energies,      dtype=np.float64),
		'fmax':      np.asarray(fmaxes,        dtype=np.float64),
		'max_step':  np.asarray(max_steps_log, dtype=np.float64),
		'converged': bool(converged),
		'n_steps':   int(steps_done)}
	return float(E), log

def Anneal(pose, ff=None, n_steps=10000, T_start=2000.0, T_end=10.0,
		sigma_small=5.0, sigma_large=30.0, p_large=0.2, p_shear=0.5,
		target_acc=0.30, adapt_window=100, seed=None, box=None):
	'''
	Simulated annealing with shear+single moves and adaptive small sigma
	Arguments:
	----------
		pose:         Pose - protein pose with Amino Acids dict
		ff:           ForceField - reusable evaluator; created if None
		n_steps:      int - total Metropolis steps in the cooling schedule
		T_start:      float - starting temperature in Kelvin
		T_end:        float - final temperature in Kelvin
		sigma_small:  float - initial small-move std-dev in degrees
		sigma_large:  float - large-move std-dev in degrees (fixed)
		p_large:      float - probability of choosing a large move
		p_shear:      float - probability of choosing a shear move
		target_acc:   float - target acceptance ratio for small moves
		adapt_window: int - small moves between sigma_small updates
		seed:         int or None - RNG seed for reproducibility
		box:          None for no PBC; (3,) ortho; (3, 3) triclinic
	Returns:
	--------
		tuple: (float, dict) - best energy seen and per-step log
	'''
	if ff is None: ff = ForceField()
	if pose.data.get('Amino Acids') is None:
		raise ValueError('Anneal requires a protein pose with Amino Acids')
	GAIN, SIGMA_MIN, SIGMA_MAX = 0.5, 0.5, 60.0
	rng = np.random.default_rng(seed)
	res_ids = np.array(sorted(pose.data['Amino Acids']), dtype=np.int64)
	n_res = len(res_ids)
	kB = 8.31446262e-3
	T_arr = T_start * (T_end / T_start) ** (
		np.arange(n_steps) / max(n_steps - 1, 1))
	res_arr   = res_ids[rng.integers(0, n_res, size=n_steps)]
	kind_arr  = np.where(rng.integers(0, 2, size=n_steps) == 0, 'PHI', 'PSI')
	shear_arr = rng.random(size=n_steps) < p_shear
	large_arr = rng.random(size=n_steps) < p_large
	noise_arr = rng.standard_normal(size=n_steps)
	uni_arr   = rng.random(size=n_steps)
	def try_single(res, kind, delta):
		theta_old = pose.GetDihedral(res, kind)
		if math.isnan(theta_old): return False
		pose.RotateDihedral(res, theta_old + delta, kind)
		return True
	def try_shear(res, delta):
		psi_old = pose.GetDihedral(res, 'PSI')
		phi_next = pose.GetDihedral(res + 1, 'PHI') \
			if (res + 1) in pose.data['Amino Acids'] else float('nan')
		if math.isnan(psi_old) or math.isnan(phi_next): return False
		pose.RotateDihedral(res, psi_old + delta, 'PSI')
		pose.RotateDihedral(res + 1, phi_next - delta, 'PHI')
		return True
	E_curr = float(ff(pose, grad=False, box=box))
	E_best = E_curr
	coords_best = pose.data['Coordinates'].copy()
	energies   = np.empty(n_steps, dtype=np.float64)
	accepted   = np.zeros(n_steps, dtype=bool)
	move_types = np.full(n_steps, 2, dtype=np.int8)  # 0=single,1=shear,2=invalid
	sigma_history = [float(sigma_small)]
	small_count, small_acc, best_step = 0, 0, 0
	for s in range(int(n_steps)):
		sigma = sigma_large if large_arr[s] else sigma_small
		delta = float(noise_arr[s] * sigma)
		res = int(res_arr[s])
		coords_old = pose.data['Coordinates'].copy()
		applied = (try_shear(res, delta) if shear_arr[s]
			else try_single(res, str(kind_arr[s]), delta))
		mtype = 1 if shear_arr[s] else 0
		if not applied and shear_arr[s]:
			applied = try_single(res, str(kind_arr[s]), delta)
			mtype = 0
		if not applied:
			energies[s] = E_curr
			continue
		E_new = float(ff(pose, grad=False, box=box))
		dE = E_new - E_curr
		RT = kB * float(T_arr[s])
		boltz = math.exp(-dE / RT) if (dE > 0.0 and RT > 0.0) else 1.0
		accept = (dE <= 0.0) or (uni_arr[s] < boltz)
		move_types[s] = mtype
		if accept:
			E_curr = E_new
			accepted[s] = True
			if E_curr < E_best:
				E_best = E_curr
				coords_best = pose.data['Coordinates'].copy()
				best_step = s
		else:
			pose.data['Coordinates'] = coords_old
		energies[s] = E_curr
		if not large_arr[s]:
			small_count += 1
			small_acc += int(accept)
			if small_count >= adapt_window:
				rate = small_acc / small_count
				sigma_small *= math.exp(GAIN * (rate - target_acc))
				sigma_small = max(SIGMA_MIN, min(sigma_small, SIGMA_MAX))
				sigma_history.append(float(sigma_small))
				small_count, small_acc = 0, 0
	pose.data['Coordinates'] = coords_best
	log = {
		'energies':      energies,
		'temperatures':  T_arr,
		'accepted':      accepted,
		'move_types':    move_types,
		'sigma_history': np.asarray(sigma_history, dtype=np.float64),
		'best_step':     int(best_step)}
	return float(E_best), log

def Pack(pose, score=None, ff=None, n_steps=2000, T_start=10.0, T_end=0.1,
		patience=400, seed=None, box=None):
	'''
	Sidechain repacking via simulated annealing on the full Rotamer Library
	ensemble at each residue's current backbone (phi, psi).

	Algorithm (production):
	  1. For each residue with chis and a defined (phi, psi), build the static
	     candidate set = list of (mu_chi_tuple, prob) from the rotamer library
	     cell at that residue's backbone.
	  2. Initialise from the pose's current chi configuration and score it.
	  3. SA loop with geometric cooling T = T_start * (T_end/T_start)^(t/N):
	     - pick a random repackable residue
	     - propose one of its rotamers k weighted by prob (so dominant
	       rotamers are explored more often, but rare ones remain reachable)
	     - apply trial chis; rescore
	     - accept if dE <= 0 or random() < exp(-dE / T); else revert
	     - track best-so-far
	  4. Early-exit if no accepted move in `patience` consecutive steps.
	  5. Restore best-found configuration; return its energy.

	D-amino acids: looked up against the L-form table with mirrored phi/psi,
	mu values negated when applied (same convention as Rotamers / _rotamer_prior).

	Arguments:
	----------
		pose:    Pose - protein pose with Amino Acids dict
		score:   Score - reusable; built from `ff` if None
		ff:      ForceField - used only when `score` is None
		n_steps: int - max number of SA proposals
		T_start: float - initial temperature (in score units, typically kJ/mol)
		T_end:   float - final temperature
		patience:int - early-exit if no acceptance in this many consecutive steps
		seed:    int or None - RNG seed for reproducibility
		box:     None for no PBC; (3,) orthorhombic; (3, 3) triclinic
	Returns:
	--------
		tuple: (E_best, log) where log contains 'energies', 'temperatures',
		       'accepts', 'best_E', 'steps_run', 'converged', 'n_residues'.
	'''
	if score is None:
		from .energy import Score
		score = Score(ff=ff, box=box)
	if pose.data.get('Amino Acids') is None:
		raise ValueError('Pack requires a protein pose with Amino Acids')
	rng = np.random.default_rng(seed)
	rotlib = DBLoad().get('Rotamer Library')
	# Step 1: build candidate sets per repackable residue.
	# Each entry: (mus (K, n_chi), probs (K,) normalised, n_chi)
	candidates = {}
	for r, info in sorted(pose.data['Amino Acids'].items()):
		c = info[0]
		aa_u = c.upper()
		aa_db = pose.aminoacids.get(aa_u, {})
		chi_atoms = aa_db.get('Chi Angle Atoms') or []
		if not chi_atoms: continue
		three = aa_db.get('Tricode')
		if not three: continue
		phi = pose.GetDihedral(r, 'PHI')
		psi = pose.GetDihedral(r, 'PSI')
		if math.isnan(phi) or math.isnan(psi): continue
		flip = (c != aa_u)
		phi_q = -phi if flip else phi
		psi_q = -psi if flip else psi
		n_chi, rows = _rotlib_lookup(rotlib, three, phi_q, psi_q)
		if n_chi == 0 or not rows: continue
		# Column layout: [count, prob, chi1..N, sig1..N]
		prob_i = 1
		chi_i  = 2
		K = len(rows)
		mus   = np.empty((K, n_chi), dtype=np.float64)
		probs = np.empty(K,          dtype=np.float64)
		for k, row in enumerate(rows):
			probs[k] = max(float(row[prob_i]), 0.0)
			for ci in range(n_chi):
				m = float(row[chi_i + ci])
				mus[k, ci] = -m if flip else m
		s = probs.sum()
		if s <= 0.0: continue
		probs /= s
		candidates[r] = (mus, probs, n_chi)
	if not candidates:
		E0 = float(score(pose))
		return E0, {
			'energies': np.array([E0]), 'temperatures': np.array([T_start]),
			'accepts': np.array([], dtype=bool), 'best_E': E0, 'steps_run': 0,
			'converged': True, 'n_residues': 0}
	res_ids = list(candidates.keys())
	# Step 2: initial energy + best-state snapshot.
	def _snapshot():
		return {r: tuple(pose.GetDihedral(r, 'CHI', chi_type=ci+1)
			for ci in range(candidates[r][2])) for r in res_ids}
	def _restore(snap):
		for r, chis in snap.items():
			n_chi = candidates[r][2]
			for ci in range(n_chi):
				pose.RotateDihedral(r, float(chis[ci]), 'CHI', ci+1)
	E_curr  = float(score(pose))
	E_best  = E_curr
	best_state = _snapshot()
	# Step 3: SA loop.
	N = max(1, int(n_steps))
	energies     = np.empty(N, dtype=np.float64)
	temperatures = np.empty(N, dtype=np.float64)
	accepts      = np.empty(N, dtype=bool)
	last_accept  = 0
	step         = 0
	for step in range(N):
		T = T_start * (T_end / T_start) ** (step / max(1, N - 1))
		# Pick residue uniformly among repackable.
		r = res_ids[int(rng.integers(0, len(res_ids)))]
		mus, probs, n_chi = candidates[r]
		# Sample rotamer k weighted by prob.
		k = int(rng.choice(len(probs), p=probs))
		# Snapshot current chis for revert.
		snap = tuple(pose.GetDihedral(r, 'CHI', chi_type=ci+1)
			for ci in range(n_chi))
		# Apply trial.
		for ci in range(n_chi):
			pose.RotateDihedral(r, float(mus[k, ci]), 'CHI', ci+1)
		E_trial = float(score(pose))
		dE = E_trial - E_curr
		if dE <= 0.0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
			E_curr = E_trial
			last_accept = step
			accepts[step] = True
			if E_curr < E_best:
				E_best = E_curr
				best_state = _snapshot()
		else:
			# Revert.
			for ci in range(n_chi):
				pose.RotateDihedral(r, float(snap[ci]), 'CHI', ci+1)
			accepts[step] = False
		energies[step]     = E_curr
		temperatures[step] = T
		# Step 4: early-exit on stagnation.
		if step - last_accept >= patience: break
	steps_run = step + 1
	# Step 5: restore best-found state.
	_restore(best_state)
	E_final = float(score(pose))
	# Sanity: best_state may slightly differ from E_best due to caching; trust E_final.
	log = {
		'energies':     energies[:steps_run],
		'temperatures': temperatures[:steps_run],
		'accepts':      accepts[:steps_run],
		'best_E':       float(E_best),
		'steps_run':    int(steps_run),
		'converged':    bool(steps_run < N),
		'n_residues':   len(res_ids)}
	return E_final, log

def MolecularDynamics(pose, ff=None, n_steps=1000, dt_fs=2.0, T=300.0,
		thermostat='nve', friction_ps=1.0, constraints='hbonds',
		shake_tol=1e-8, shake_max=100, seed=None,
		trajectory_every=0, box=None):
	'''
	Velocity-Verlet NVE or BAOAB Langevin NVT MD with SHAKE/RATTLE
	Arguments:
	----------
		pose:             Pose - molecule source pose
		ff:               ForceField - reusable evaluator; created if None
		n_steps:          int - number of integration steps
		dt_fs:            float - integration step in femtoseconds
		T:                float - temperature in Kelvin (initial + bath)
		thermostat:       str - 'nve' or 'langevin'
		friction_ps:      float - Langevin friction in ps^-1
		constraints:      str - 'hbonds' constrains every X-H bond; 'none'
		shake_tol:        float - relative tolerance on |d^2 - r0^2|/r0^2
		shake_max:        int - max iterations for SHAKE/RATTLE projection
		seed:             int or None - RNG seed for reproducibility
		trajectory_every: int - snapshot stride; 0 disables snapshots
		box:              None for no PBC; (3,) ortho; (3, 3) triclinic
	Returns:
	--------
		tuple: (float, dict) - final potential energy and trajectory log
	'''
	if ff is None: ff = ForceField()
	if thermostat not in ('nve', 'langevin'):
		raise ValueError("thermostat must be 'nve' or 'langevin'")
	if constraints not in ('hbonds', 'none'):
		raise ValueError("constraints must be 'hbonds' or 'none'")
	rng = np.random.default_rng(seed)
	atoms = pose.data['Atoms']
	sorted_ids = sorted(atoms)
	m = np.array([pose.masses[atoms[i][1]] for i in sorted_ids],
		dtype=np.float64)
	n = len(m)
	m_col = m[:, None]
	inv_m = 1.0 / m
	inv_m_col = inv_m[:, None]
	AKMA_FS = 23.91888086
	kB = 8.31446262e-3
	dt = float(dt_fs) / AKMA_FS
	gamma = float(friction_ps) * AKMA_FS / 1000.0
	c1 = math.exp(-gamma * dt)
	c2 = np.sqrt((1.0 - c1 * c1) * kB * float(T) / m)[:, None]
	if ff._cache is None or ff._cache_hash != ff._topologyhash(pose):
		ff._prepare(pose)
	cache = ff._cache
	is_h = np.array([atoms[i][1] == 'H' for i in sorted_ids], dtype=bool)
	if constraints == 'hbonds' and len(cache['pairs']):
		cmask = is_h[cache['pairs'][:, 0]] | is_h[cache['pairs'][:, 1]]
		con = cache['pairs'][cmask]
		r0  = cache['bond_r0'][cmask]
	else:
		con = np.empty((0, 2), dtype=np.int64)
		r0  = np.empty((0,),   dtype=np.float64)
	K = len(con)
	i_c, j_c = con[:, 0], con[:, 1]
	r0sq = r0 * r0
	inv_red = inv_m[i_c] + inv_m[j_c] if K else np.empty(0)
	r0sq_max = float(r0sq.max()) if K else 1.0
	def shake(x_new, x_old, vel, dt_eff):
		if K == 0: return
		r_old = x_old[i_c] - x_old[j_c]
		for _ in range(int(shake_max)):
			r = x_new[i_c] - x_new[j_c]
			d2 = np.einsum('ij,ij->i', r, r)
			sigma = d2 - r0sq
			if float(np.max(np.abs(sigma))) < shake_tol * r0sq_max:
				return
			rdot = np.einsum('ij,ij->i', r, r_old)
			lam  = sigma / (2.0 * inv_red * rdot)
			delta = lam[:, None] * r_old
			np.add.at(x_new, i_c, -delta * inv_m_col[i_c])
			np.add.at(x_new, j_c,  delta * inv_m_col[j_c])
			np.add.at(vel,   i_c, -(delta / dt_eff) * inv_m_col[i_c])
			np.add.at(vel,   j_c,  (delta / dt_eff) * inv_m_col[j_c])
	def rattle(x, vel):
		if K == 0: return
		for _ in range(int(shake_max)):
			r = x[i_c] - x[j_c]
			v_rel = vel[i_c] - vel[j_c]
			rv = np.einsum('ij,ij->i', r, v_rel)
			d2 = np.einsum('ij,ij->i', r, r)
			if float(np.max(np.abs(rv))) < shake_tol * r0sq_max:
				return
			mu = rv / (d2 * inv_red)
			delta_v = mu[:, None] * r
			np.add.at(vel, i_c, -delta_v * inv_m_col[i_c])
			np.add.at(vel, j_c,  delta_v * inv_m_col[j_c])
	sigma_v = np.sqrt(kB * float(T) / m)[:, None]
	v = rng.standard_normal(size=(n, 3)) * sigma_v
	v -= ((m_col * v).sum(axis=0) / m.sum())[None, :]
	rattle(pose.data['Coordinates'], v)
	E, F = ff(pose, grad=True, box=box)
	dof = max(3 * n - K - 3, 1)
	energies = np.empty(int(n_steps), dtype=np.float64)
	kinetics = np.empty(int(n_steps), dtype=np.float64)
	temps    = np.empty(int(n_steps), dtype=np.float64)
	frames = []
	use_langevin = (thermostat == 'langevin')
	for step in range(int(n_steps)):
		if use_langevin:
			v += 0.5 * dt * F / m_col
			x_old = pose.data['Coordinates'].copy()
			pose.data['Coordinates'] = x_old + 0.5 * dt * v
			shake(pose.data['Coordinates'], x_old, v, 0.5 * dt)
			v = c1 * v + c2 * rng.standard_normal(size=(n, 3))
			rattle(pose.data['Coordinates'], v)
			x_old = pose.data['Coordinates'].copy()
			pose.data['Coordinates'] = x_old + 0.5 * dt * v
			shake(pose.data['Coordinates'], x_old, v, 0.5 * dt)
			E, F = ff(pose, grad=True, box=box)
			v += 0.5 * dt * F / m_col
			rattle(pose.data['Coordinates'], v)
		else:
			v += 0.5 * dt * F / m_col
			x_old = pose.data['Coordinates'].copy()
			pose.data['Coordinates'] = x_old + dt * v
			shake(pose.data['Coordinates'], x_old, v, dt)
			E, F = ff(pose, grad=True, box=box)
			v += 0.5 * dt * F / m_col
			rattle(pose.data['Coordinates'], v)
		KE = 0.5 * float(np.sum(m_col * v * v))
		energies[step] = float(E)
		kinetics[step] = KE
		temps[step] = 2.0 * KE / (dof * kB)
		if trajectory_every > 0 and (step + 1) % trajectory_every == 0:
			frames.append(pose.data['Coordinates'].copy())
	log = {
		'energies':     energies,
		'kinetic':      kinetics,
		'temperatures': temps,
		'frames':       frames,
		'n_constraints': int(K),
		'dof':           int(dof)}
	return float(E), log

def Port(name='openff'):
	'''
	Port one force field into database.json and optionally verify it
	Arguments:
	----------
		name:   str - which force field to port; 'openff', 'ff19SB' or
			'charmm36', matched case-insensitively
		verify: bool - if True, re-import the force field's benchmark
			structures and compare each energy against its reference
	Returns:
	--------
		bool: True if the port (and verification, when requested)
			succeeded; False if any benchmark deviates by > 1e-3 relative
	'''
	key     = str(name).upper()
	db_path = './database.json'
	def download(url):
		'''
		Fetch the text of a pinned GitHub raw URL
		Arguments:
		----------
			url: str - a raw.githubusercontent.com URL on a fixed commit
		Returns:
		--------
			str: the decoded file contents
		'''
		print(f'[port] downloading {url}', file=sys.stderr)
		try:
			with urllib.request.urlopen(url, timeout=120) as resp:
				return resp.read().decode('utf-8')
		except Exception as err:
			raise RuntimeError(f'port: could not download {url}: {err}')
	def cidof(rec, i):
		'''
		Read the namespaced atom identifier at slot i of a bonded record
		Arguments:
		----------
			rec: dict - the XML element's attributes
			i:   int  - 1-based slot index
		Returns:
		--------
			str: the class/type identifier, '' for an unset wildcard slot
		'''
		v = rec.get('class%d' % i)
		if v is None: v = rec.get('type%d' % i)
		return v if v is not None else ''
	def qval(qstr, target):
		'''
		Convert a SMIRNOFF quantity string to a target unit
		Arguments:
		----------
			qstr:   str - a quantity, e.g. '1.5 * angstrom ** 1'
			target: str - the desired unit expression, e.g.
				'kilojoule_per_mole * angstrom ** -2'
		Returns:
		--------
			float: the magnitude of qstr expressed in the target unit
		'''
		units = {
			'angstrom':             (1.0,           {'L': 1}),
			'nanometer':            (10.0,          {'L': 1}),
			'degree':               (1.0,           {'A': 1}),
			'radian':               (180.0/math.pi, {'A': 1}),
			'mole':                 (1.0,           {'N': 1}),
			'kilojoule':            (1.0,           {'E': 1}),
			'kilocalorie':          (4.184,         {'E': 1}),
			'kilojoule_per_mole':   (1.0,           {'E': 1, 'N': -1}),
			'kilocalorie_per_mole': (4.184,         {'E': 1, 'N': -1}),
			'elementary_charge':    (1.0,           {'Q': 1})}
		def reduce(text):
			'''Reduce a unit expression to (factor, {dimension: power}).'''
			factor, dims = 1.0, {}
			for tok in text.strip().replace('**', '^').split('*'):
				tok = tok.strip()
				if not tok: continue
				if '^' in tok:
					nm, _, ex = tok.partition('^')
					nm, ex = nm.strip(), int(ex.strip())
				else:
					nm, ex = tok, 1
				try:
					factor *= float(nm) ** ex
					continue
				except ValueError:
					pass
				if nm not in units:
					raise ValueError(
						f'port: unknown unit {nm!r} in {text!r}')
				f, d = units[nm]
				factor *= f ** ex
				for k, v in d.items():
					dims[k] = dims.get(k, 0) + v * ex
			return factor, {k: v for k, v in dims.items() if v}
		fq, dq = reduce(qstr)
		ft, dt = reduce(target)
		if dq != dt:
			raise ValueError(
				f'port: cannot convert {qstr!r} to {target!r} '
				f'(dimension mismatch)')
		return fq / ft
	def converttorsions(section):
		'''
		Convert a SMIRNOFF torsion section to Pose's component schema
		Arguments:
		----------
			section: xml.etree Element - a <ProperTorsions> or
				<ImproperTorsions> SMIRNOFF section
		Returns:
		--------
			dict: {SMIRKS: {id, components: [{n, phi_0, K_phi, idivf}]}}
		'''
		out = {}
		for p in section:
			a = p.attrib
			comps, i = [], 1
			while ('k%d' % i) in a:
				idivf = a.get('idivf%d' % i)
				comps.append({
					'n':     int(a['periodicity%d' % i]),
					'phi_0': qval(a['phase%d' % i], 'degree'),
					'K_phi': qval(a['k%d' % i], 'kilojoule_per_mole'),
					'idivf': float(idivf) if idivf is not None else 1.0})
				i += 1
			out[a['smirks']] = {'id': a.get('id'), 'components': comps}
		return out
	def charmmtypes(root):
		'''
		Rebuild per-residue templates (atom name / element / class /
		charge plus the intra-residue bond list), with the N/C-terminal
		and disulfide variants, from charmm36.xml Residues + Patches --
		a pure-stdlib replacement for the OpenMM template/patch engine
		Arguments:
		----------
			root: xml.etree Element - the parsed charmm36.xml root
		Returns:
		--------
			dict: {variant: {atoms: [[name, element, class, charge]],
				bonds: [[name, name]]}}
		'''
		at_elem = {t.attrib['name']: t.attrib.get('element', '')
			for t in root.find('AtomTypes')}
		res = {}
		for rr in root.find('Residues'):
			atoms, bonds = {}, []
			for c in rr:
				if c.tag == 'Atom':
					atoms[c.attrib['name']] = [c.attrib['type'],
						float(c.attrib['charge'])]
				elif c.tag == 'Bond':
					bonds.append((c.attrib['atomName1'],
						c.attrib['atomName2']))
			res[rr.attrib['name']] = (atoms, bonds)
		patch = {}
		for pp in root.find('Patches'):
			d = {'change': {}, 'add': {}, 'remove': [],
				'addbond': [], 'rmbond': []}
			for c in pp:
				a = c.attrib
				if   c.tag == 'ChangeAtom':
					d['change'][a['name']] = [a['type'],
						float(a['charge'])]
				elif c.tag == 'AddAtom':
					d['add'][a['name']] = [a['type'],
						float(a['charge'])]
				elif c.tag == 'RemoveAtom':
					d['remove'].append(a['name'])
				elif c.tag == 'AddBond':
					d['addbond'].append((a['atomName1'],
						a['atomName2']))
				elif c.tag == 'RemoveBond':
					d['rmbond'].append((a['atomName1'],
						a['atomName2']))
			patch[pp.attrib['name']] = d
		def patchside(nm):
			'''Strip a 2-residue patch prefix: "1:CB" -> ("1", "CB").'''
			if len(nm) > 2 and nm[1] == ':': return nm[0], nm[2:]
			return None, nm
		def applypatch(base, pname):
			'''Apply one patch (residue-1 side) to a (atoms, bonds) pair.'''
			atoms = {k: list(v) for k, v in base[0].items()}
			bonds = list(base[1])
			d = patch[pname]
			def keep(nm):
				s, real = patchside(nm)
				return real if s in (None, '1') else None
			for nm, v in d['change'].items():
				real = keep(nm)
				if real is not None and real in atoms:
					atoms[real] = list(v)
			for nm, v in d['add'].items():
				real = keep(nm)
				if real is not None: atoms[real] = list(v)
			rem = {keep(nm) for nm in d['remove']} - {None}
			atoms = {k: v for k, v in atoms.items() if k not in rem}
			bonds = [b for b in bonds
				if b[0] not in rem and b[1] not in rem]
			rmb = set()
			for x, y in d['rmbond']:
				rx, ry = keep(x), keep(y)
				if rx is not None and ry is not None:
					rmb.add(frozenset((rx, ry)))
			bonds = [b for b in bonds if frozenset(b) not in rmb]
			for x, y in d['addbond']:
				rx, ry = keep(x), keep(y)
				if rx is not None and ry is not None:
					bonds.append((rx, ry))
			return (atoms, bonds)
		npatch = {'GLY': 'GLYP', 'PRO': 'PROP'}
		protein = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
			'GLY', 'HSD', 'HSE', 'HSP', 'ILE', 'LEU', 'LYS', 'MET',
			'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
		variants = {}
		for rn in protein:
			if rn not in res: continue
			variants[rn]       = res[rn]
			variants['N' + rn] = applypatch(res[rn],
				npatch.get(rn, 'NTER'))
			variants['C' + rn] = applypatch(res[rn], 'CTER')
		if 'CYS' in res:
			cyx = applypatch(res['CYS'], 'DISU')
			variants['CYX']  = cyx
			variants['NCYX'] = applypatch(cyx, 'NTER')
			variants['CCYX'] = applypatch(cyx, 'CTER')
		templates = {}
		for vn, (atoms, bonds) in variants.items():
			templates[vn] = {
				'atoms': [[nm, at_elem.get(cls, ''), cls, chg]
					for nm, (cls, chg) in atoms.items()],
				'bonds': [[a, b] for a, b in bonds]}
		return templates
	with open(db_path) as f: db = json.load(f)
	ep = db.setdefault('Energy Parameters', {})
	# ============================================================
	# OpenFF Sage 2.3.0
	# ============================================================
	if key == 'OPENFF':
		commit = 'edd7724103a558328c358a9e35462334c4a45b6f'
		url = ('https://raw.githubusercontent.com/openforcefield/'
			'openff-forcefields/' + commit
			+ '/openforcefields/offxml/openff-2.3.0.offxml')
		root = ET.fromstring(download(url))
		rmin_factor = 2.0 / (2.0 ** (1.0 / 6.0))
		bonds = {}
		for p in root.find('Bonds'):
			a = p.attrib
			bonds[a['smirks']] = {'id': a.get('id'),
				'r_0': qval(a['length'], 'angstrom'),
				'K_b': qval(a['k'],
					'kilojoule_per_mole * angstrom ** -2')}
		angles = {}
		for p in root.find('Angles'):
			a = p.attrib
			angles[a['smirks']] = {'id': a.get('id'),
				'theta_0': qval(a['angle'], 'degree'),
				'K_theta': qval(a['k'],
					'kilojoule_per_mole * radian ** -2')}
		propers   = converttorsions(root.find('ProperTorsions'))
		impropers = converttorsions(root.find('ImproperTorsions'))
		vdw = {}
		for p in root.find('vdW'):
			a = p.attrib
			if 'sigma' in a:
				r = qval(a['sigma'], 'angstrom') / rmin_factor
			else:
				r = qval(a['rmin_half'], 'angstrom')
			vdw[a['smirks']] = {'id': a.get('id'),
				'epsilon': qval(a['epsilon'], 'kilojoule_per_mole'),
				'r': r, 'alpha': 0.0}
		charges = {}
		for p in root.find('LibraryCharges'):
			a = p.attrib
			qs, i = [], 1
			while ('charge%d' % i) in a:
				qs.append(qval(a['charge%d' % i], 'elementary_charge'))
				i += 1
			charges[a['smirks']] = {'id': a.get('id'), 'q': qs}
		constraints = {}
		for p in root.find('Constraints'):
			constraints[p.attrib['smirks']] = {'id': p.attrib.get('id')}
		prev = ep.get('OpenFF') or ep.get('openFF') or {}
		nagl = prev.get('AM1BCC') or ep.get('AM1BCC')
		block = {
			'Constants': {'epsilon_r': 1.0, 'f_lj': 0.5,
				'f_elec': 5.0 / 6.0},
			'Constraints':      constraints,
			'Bonds':            bonds,
			'Angles':           angles,
			'UB':               prev.get('UB', {}),
			'ProperTorsions':   propers,
			'ImproperTorsions': impropers,
			'vdW':              vdw,
			'Electrostatic':    charges,
			'CMAP':             prev.get('CMAP', prev.get('cmap', {})),
			'Terms': [
				['BondPotential',            {'alg': 'harmonic'}],
				['AnglePotential',           {}],
				['ProperTorsionPotential',   {}],
				['ImproperTorsionPotential', {'alg': 'fourier'}],
				['VDWPotential',             {'alg': '12-6'}],
				['ElectrostaticPotential',   {'alg': 'constant'}],
			],
		}
		if nagl is not None: block['AM1BCC'] = nagl
		ep.pop('OpenFF', None)
		db_key  = 'OpenFF'
		suffix  = '.sdf'
		targets = {'CFF': -526.798, 'AMX': 288.644, 'SUR': 617.480,
			'GLU': 240.001, 'PEN': 165.780}
	# ============================================================
	# AMBER ff19SB  (proteins + DNA OL15 + RNA OL3)
	# ============================================================
	elif key == 'FF19SB':
		commit = 'f7fa0c27c1f8d943c339d67b3bf22f026d0bd8b5'
		base = ('https://raw.githubusercontent.com/openmm/openmm/'
			+ commit + '/wrappers/python/openmm/app/data/')
		xml_urls = [base + 'amber19/protein.ff19SB.xml',
			base + 'amber14/DNA.OL15.xml',
			base + 'amber14/RNA.OL3.xml']
		bonds, angles, propers, impropers = {}, {}, {}, {}
		vdw, templates, cmap = {}, {}, {}
		for url in xml_urls:
			root = ET.fromstring(download(url))
			type2class, type2elem = {}, {}
			at = root.find('AtomTypes')
			if at is not None:
				for t in at:
					type2class[t.attrib['name']] = \
						t.attrib.get('class', t.attrib['name'])
					type2elem[t.attrib['name']] = \
						t.attrib.get('element', '')
			hbf = root.find('HarmonicBondForce')
			by_class = (hbf is not None and len(hbf) > 0
				and 'class1' in hbf[0].attrib)
			res = root.find('Residues')
			if res is not None:
				for r in res:
					ratoms, rbonds = [], []
					for c in r:
						if c.tag == 'Atom':
							tp = c.attrib['type']
							tid = (type2class.get(tp, tp)
								if by_class else tp)
							ratoms.append([c.attrib['name'],
								type2elem.get(tp, ''), tid,
								float(c.attrib.get('charge', 0.0))])
						elif c.tag == 'Bond':
							rbonds.append([c.attrib['atomName1'],
								c.attrib['atomName2']])
					templates[r.attrib['name']] = {
						'atoms': ratoms, 'bonds': rbonds}
			if hbf is not None:
				for b in hbf:
					c1, c2 = cidof(b, 1), cidof(b, 2)
					bonds[f'<at={c1},{c2}>[*:1]~[*:2]'] = {
						'r_0': float(b.attrib['length']) * 10.0,
						'K_b': float(b.attrib['k']) * 0.01}
			haf = root.find('HarmonicAngleForce')
			if haf is not None:
				for a in haf:
					c1, c2, c3 = (cidof(a, 1), cidof(a, 2), cidof(a, 3))
					angles[f'<at={c1},{c2},{c3}>[*:1]~[*:2]~[*:3]'] = {
						'theta_0': math.degrees(
							float(a.attrib['angle'])),
						'K_theta': float(a.attrib['k'])}
			ptf = root.find('PeriodicTorsionForce')
			if ptf is not None:
				for t in ptf:
					cs = [cidof(t, i) for i in (1, 2, 3, 4)]
					comps, k = [], 1
					while ('k%d' % k) in t.attrib:
						comps.append({
							'n': int(t.attrib['periodicity%d' % k]),
							'phi_0': -math.degrees(
								float(t.attrib['phase%d' % k])),
							'K_phi': float(t.attrib['k%d' % k]),
							'idivf': 1.0})
						k += 1
					if t.tag == 'Improper':
						ro = [cs[1], cs[0], cs[2], cs[3]]
						tag = ','.join('*' if x == '' else x
							for x in ro)
						impropers[f'<at={tag}>'
							'[*:1]~[*:2](~[*:3])~[*:4]'] = {
							'components': comps}
					else:
						tag = ','.join('*' if x == '' else x
							for x in cs)
						propers[f'<at={tag}>[*:1]~[*:2]~[*:3]~[*:4]'] = \
							{'components': comps}
			nbf = root.find('NonbondedForce')
			if nbf is not None:
				for a in nbf:
					if a.tag != 'Atom': continue
					tid = a.attrib.get('class') or a.attrib.get('type')
					vdw[f'<at={tid}>[*:1]'] = {
						'epsilon': float(a.attrib.get('epsilon', 0.0)),
						'sigma': float(a.attrib.get('sigma', 0.0)) * 10.0,
						'alpha': 0.0}
			cmf = root.find('CMAPTorsionForce')
			if cmf is not None:
				maps, ctors = [], []
				for c in cmf:
					if c.tag == 'Map':
						g = [float(x) for x in c.text.split()]
						m = int(round(len(g) ** 0.5))
						maps.append(np.asarray(g,
							dtype=np.float64).reshape(m, m))
					elif c.tag == 'Torsion':
						ctors.append(c.attrib)
				for tr in ctors:
					idx = int(tr.get('map', 0))
					if idx >= len(maps): continue
					parts = (tr.get('type3', '') or '').split('-')
					if len(parts) >= 2 and parts[0] == 'cmap':
						cmap[parts[1]] = maps[idx].tolist()
		block = {
			'Constants': {'epsilon_r': 1.0, 'f_lj': 0.5,
				'f_elec': 0.8333333333333334},
			'improper_style':    'amber',
			'proper_precedence': 'openmm',
			'Constraints':      {'<residue_templates>': templates},
			'Bonds':            bonds,
			'Angles':           angles,
			'UB':               {},
			'ProperTorsions':   propers,
			'ImproperTorsions': impropers,
			'vdW':              vdw,
			'Electrostatic':    {},
			'CMAP':             cmap,
			'Terms': [
				['BondPotential',            {'alg': 'harmonic'}],
				['AnglePotential',           {}],
				['ProperTorsionPotential',   {}],
				['ImproperTorsionPotential', {'alg': 'fourier'}],
				['VDWPotential',             {'alg': '12-6'}],
				['ElectrostaticPotential',   {'alg': 'constant'}],
				['CMAPPotential',            {'alg': 'openmm'}],
			],
		}
		ep.pop('AMBER ff19SB', None)
		db_key  = 'ff19SB'
		suffix  = '.pdb'
		targets = {'1YN3': -3749.609, '1UBQ': -7251.245,
			'1L2Y': -1680.192, '1CRN': -4169.046, '2GB1': -103.132,
			'1BNA': 5927.131, '1RNA': 19196.876}
	# ============================================================
	# CHARMM36  (proteins)
	# ============================================================
	elif key == 'CHARMM36':
		commit = 'f7fa0c27c1f8d943c339d67b3bf22f026d0bd8b5'
		xml_url = ('https://raw.githubusercontent.com/openmm/openmm/'
			+ commit + '/wrappers/python/openmm/app/data/charmm36.xml')
		root = ET.fromstring(download(xml_url))
		bonds = {}
		hbf = root.find('HarmonicBondForce')
		if hbf is not None:
			for b in hbf:
				c1, c2 = cidof(b, 1), cidof(b, 2)
				bonds[f'<at={c1},{c2}>[*:1]~[*:2]'] = {
					'r_0': float(b.attrib['length']) * 10.0,
					'K_b': float(b.attrib['k']) * 0.01}
		angles = {}
		haf = root.find('HarmonicAngleForce')
		if haf is not None:
			for a in haf:
				c1, c2, c3 = cidof(a, 1), cidof(a, 2), cidof(a, 3)
				angles[f'<at={c1},{c2},{c3}>[*:1]~[*:2]~[*:3]'] = {
					'theta_0': math.degrees(float(a.attrib['angle'])),
					'K_theta': float(a.attrib['k'])}
		ub = {}
		ubf = root.find('AmoebaUreyBradleyForce')
		if ubf is not None:
			for u in ubf:
				c1, c2, c3 = cidof(u, 1), cidof(u, 2), cidof(u, 3)
				ub[f'<at={c1},{c2},{c3}>[*:1]~[*:2]~[*:3]'] = {
					's_0':  float(u.attrib['d']) * 10.0,
					'K_ub': float(u.attrib['k']) * 0.01}
		propers = {}
		ptf = root.find('PeriodicTorsionForce')
		if ptf is not None:
			for t in ptf:
				if t.tag != 'Proper': continue
				cs = [cidof(t, i) for i in (1, 2, 3, 4)]
				comps, k = [], 1
				while ('k%d' % k) in t.attrib:
					comps.append({
						'n': int(t.attrib['periodicity%d' % k]),
						'phi_0': -math.degrees(
							float(t.attrib['phase%d' % k])),
						'K_phi': float(t.attrib['k%d' % k]),
						'idivf': 1.0})
					k += 1
				tag = ','.join('*' if x == '' else x for x in cs)
				sm = f'<at={tag}>[*:1]~[*:2]~[*:3]~[*:4]'
				if sm not in propers: propers[sm] = {'components': comps}
		impropers = {}
		ctf = root.find('CustomTorsionForce')
		if ctf is not None:
			for t in ctf:
				if t.tag != 'Improper': continue
				cs = [cidof(t, i) for i in (1, 2, 3, 4)]
				tag = ','.join('*' if x == '' else x for x in cs)
				sm = f'<at={tag}>[*:1](~[*:2])(~[*:3])~[*:4]'
				if sm in impropers: continue
				impropers[sm] = {'components': [{
					'n': 2,
					'phi_0': -math.degrees(
						float(t.attrib.get('theta0', 0.0))),
					'K_phi': float(t.attrib.get('k', 0.0)),
					'idivf': 1.0}]}
		vdw = {}
		ljf = root.find('LennardJonesForce')
		if ljf is not None:
			for a in ljf:
				if a.tag != 'Atom': continue
				tid = a.attrib.get('type') or a.attrib.get('class')
				sig = float(a.attrib.get('sigma', 0.0)) * 10.0
				eps = float(a.attrib.get('epsilon', 0.0))
				s14 = (float(a.attrib['sigma14']) * 10.0
					if 'sigma14' in a.attrib else sig)
				e14 = (float(a.attrib['epsilon14'])
					if 'epsilon14' in a.attrib else eps)
				vdw[f'<at={tid}>[*:1]'] = {'epsilon': eps, 'sigma': sig,
					'epsilon14': e14, 'sigma14': s14, 'alpha': 0.0}
		cmap = {}
		cmf = root.find('CMAPTorsionForce')
		if cmf is not None:
			maps, ctors = [], []
			for c in cmf:
				if c.tag == 'Map':
					g = [float(x) for x in c.text.split()]
					m = int(round(len(g) ** 0.5))
					maps.append(np.asarray(g,
						dtype=np.float64).reshape(m, m))
				elif c.tag == 'Torsion':
					ctors.append(c.attrib)
			standard = list('ARNDCQEHILKMFSTWYV')
			for tr in ctors:
				if (tr.get('type5', '') or '') == 'N': continue
				idx = int(tr.get('map', 0))
				if idx >= len(maps): continue
				t2, t3 = tr.get('type2', ''), tr.get('type3', '')
				if   t3 == 'CT1' and t2 == 'NH1': letters = standard
				elif t3 == 'CT2' and t2 == 'NH1': letters = ['G']
				elif t3 == 'CP1' and t2 == 'N':   letters = ['P']
				else: continue
				grid = maps[idx].tolist()
				for one in letters: cmap[one] = grid
		templates = charmmtypes(root)
		block = {
			'Constants': {'epsilon_r': 1.0, 'f_lj': 1.0,
				'f_elec': 1.0},
			'improper_style':    'charmm',
			'proper_precedence': 'openmm',
			'Constraints':      {'<residue_templates>': templates},
			'Bonds':            bonds,
			'Angles':           angles,
			'UB':               ub,
			'ProperTorsions':   propers,
			'ImproperTorsions': impropers,
			'vdW':              vdw,
			'Electrostatic':    {},
			'CMAP':             cmap,
			'Terms': [
				['BondPotential',            {'alg': 'harmonic'}],
				['AnglePotential',           {}],
				['UBPotential',              {}],
				['ProperTorsionPotential',   {}],
				['ImproperTorsionPotential', {'alg': 'harmonic'}],
				['VDWPotential',             {'alg': '12-6'}],
				['ElectrostaticPotential',   {'alg': 'constant'}],
				['CMAPPotential',            {'alg': 'openmm'}],
			],
		}
		db_key  = 'CHARMM36'
		suffix  = '.pdb'
		targets = {'1YN3': 4104.856, '1UBQ': -2687.538,
			'1L2Y': 140.922, '1CRN': -401.309, '2GB1': 4225.107}
	else:
		raise ValueError(
			"port: name must be 'openff', 'ff19SB' or 'charmm36' "
			f'(got {name!r})')
	ep[db_key] = block
	with open(db_path, 'w') as f:
		json.dump(db, f, separators=(',', ':'))
	print(f'[port] wrote {db_key} block: {len(block.get("Bonds", {}))} '
		f'bonds, {len(block.get("Angles", {}))} angles, '
		f'{len(block.get("ProperTorsions", {}))} propers, '
		f'{len(block.get("ImproperTorsions", {}))} impropers, '
		f'{len(block.get("vdW", {}))} vdW', file=sys.stderr)
	return True
