import numpy as np

Parameters = {
	'bonds': {
		#i     j       Kb    De     a     r0
		('C',  'CA'): (317.0, 88.0, 1.91, 1.522),
		('C',  'N' ): (490.0,143.0, 2.05, 1.335),
		('C',  'O' ): (570.0,170.0, 2.20, 1.229),
		('CA', 'CB'): (310.0, 85.0, 1.91, 1.526),
		('CA', 'HA'): (340.0, 97.0, 1.84, 1.090),
		('CA', 'N' ): (337.0, 95.0, 1.95, 1.449),
		('CB', 'HB'): (340.0, 97.0, 1.84, 1.090),
		('H',  'N' ): (434.0,110.0, 2.07, 1.010),
		# Generic element-pair stand-ins so unknown types still
		('C',  'C' ): (310.0, 85.0, 1.91, 1.526),
		('C',  'H' ): (340.0, 97.0, 1.84, 1.090),
		('C',  'S' ): (227.0, 60.0, 1.80, 1.810),
		('H',  'O' ): (553.0,115.0, 2.27, 0.960),
		('H',  'S' ): (274.0, 80.0, 1.85, 1.336),
		('O',  'P' ): (525.0, 90.0, 2.00, 1.610),
		'default': (300.0,90.0, 2.00,  1.500)},
	'angles': {
		#i     j     k      theta0 K theta K_ub   S0
		('C',  'CA', 'N' ): (63.0, 110.1, 0.0,  1.500),
		('CA', 'C',  'N' ): (70.0, 116.6, 50.0, 2.450),
		('CA', 'C',  'O' ): (80.0, 120.4, 50.0, 2.388),
		('N',  'C',  'O' ): (80.0, 122.9, 50.0, 2.250),
		('C',  'N',  'CA'): (50.0, 121.9, 50.0, 2.453),
		('CA', 'N',  'H' ): (50.0, 118.0, 0.0,  1.500),
		('C',  'N',  'H' ): (50.0, 119.8, 0.0,  1.500),
		('CB', 'CA', 'N' ): (80.0, 109.7, 50.0, 2.510),
		('C',  'CA', 'CB'): (63.0, 111.1, 50.0, 2.561),
		('CB', 'CA', 'HA'): (50.0, 109.5, 0.0,  1.500),
		('C',  'CA', 'HA'): (50.0, 109.5, 0.0,  1.500),
		('HA', 'CA', 'N' ): (50.0, 109.5, 0.0,  1.500),
		('CA', 'CB', 'HB'): (50.0, 109.5, 0.0,  1.500),
		('HB', 'CB', 'HB'): (35.0, 109.5, 0.0,  1.500),
		# Generic element-pair stand-ins so unknown types still resolve.
		('C',  'C',  'C' ): (63.0, 111.0, 0.0,  1.500),
		('C',  'C',  'H' ): (50.0, 109.5, 0.0,  1.500),
		('H',  'C',  'H' ): (35.0, 109.5, 0.0,  1.500),
		('C',  'C',  'N' ): (80.0, 110.0, 0.0,  1.500),
		('C',  'C',  'O' ): (80.0, 120.0, 0.0,  1.500),
		('H',  'N',  'H' ): (35.0, 109.5, 0.0,  1.500),
		('C',  'O',  'H' ): (55.0, 108.5, 0.0,  1.500),
		('CA', 'CB', 'H' ): (50.0, 109.5, 0.0,  1.500),
		('H',  'CB', 'H' ): (35.0, 109.5, 0.0,  1.500),
		('C',  'CB', 'H' ): (50.0, 109.5, 0.0,  1.500),
		'default': (50.0, 109.5, 0.0, 1.500)},
	'lennard_jones': {
		#i     sigma  epsilon alpha
		'H' : (2.886, 0.044, 0.496),
		'C' : (3.851, 0.105, 1.334),
		'N' : (3.660, 0.069, 1.073),
		'O' : (3.500, 0.060, 0.837),
		'F' : (3.364, 0.050, 0.444),
		'P' : (4.147, 0.305, 1.828),
		'S' : (4.035, 0.274, 2.500),
		'Cl': (3.947, 0.227, 2.315),
		'Br': (4.189, 0.251, 3.013),
		'I' : (4.500, 0.339, 4.692),
		'Se': (4.205, 0.291, 2.700),
		'default': (3.851, 0.105, 1.000)},
	'electrostatic': {
		'epsilon_r': 1.0},
	'scaling_14': {
		'f_lj'  : 0.5,
		'f_elec': 1.0 / 1.2},
	'dihedrals': {
		#i     j     k     l       k_phi  n  phi0_deg
		('CA', 'C',  'N',  'CA'): [(2.50, 2, 180.0)],
		('C',  'N',  'CA', 'C' ): [(0.20, 1, 180.0), (0.20, 2,   0.0)],
		('N',  'CA', 'C',  'N' ): [(0.45, 2, 180.0)],
		('N',  'CA', 'CB', 'HB'): [(0.16, 3,   0.0)],
		('C',  'CA', 'CB', 'HB'): [(0.16, 3,   0.0)],
		('HA', 'CA', 'CB', 'HB'): [(0.16, 3,   0.0)],
		# Generic element-pair stand-ins.
		('C',  'C',  'C',  'C' ): [(0.18, 3,   0.0)],
		('C',  'C',  'C',  'H' ): [(0.16, 3,   0.0)],
		('H',  'C',  'C',  'H' ): [(0.15, 3,   0.0)],
		'default': [(0.0, 1, 0.0)]},
	'impropers': {
		#a     b     c     d      k_phi n  phi0_deg
		('C', 'CA', 'N',  'O' ): (1.10, 2, 180.0),
		('N', 'C',  'CA', 'H' ): (1.10, 2, 180.0),
		'default': (1.10, 2, 180.0)},
	'cmap': {
		# Per-residue 1-letter code → 24×24 grid of CMAP energies in
		# kcal/mol over (φ, ψ) ∈ [-π, π)². Replace 'default' with real
		# per-residue tables when populating the FF.
		# in the future should be instead of default 'G':[[PHI/PSI 24x24 matrix]]
		'default': 0.1 * np.cos(
			np.linspace(-np.pi, np.pi, 24, endpoint=False))[:, None] * np.cos(
			np.linspace(-np.pi, np.pi, 24, endpoint=False))[None, :]},
}

def _wrap(dvec, box):
	'''
	Apply minimum-image convention to displacement vectors for PBC.
	Arguments:
	----------
		dvec: ndarray with last axis = 3 (any other shape passes through)
		box: None, shape (3,) orthorhombic, or shape (3, 3) triclinic.
	Returns:
	--------
		dvec wrapped to its minimum-image representation.
	'''
	if box is None: return dvec
	box = np.asarray(box, dtype=np.float64)
	if box.ndim == 1:
		return dvec - box * np.round(dvec / box)
	inv_B = np.linalg.inv(box)
	f = dvec @ inv_B
	f -= np.round(f)
	return f @ box

def _atomtype(atom_index):
	'''
	Converts atom index to atom name
	Arguments:
	----------
		atom_index: atom index
	Returns:
	--------
		list: [name, element q?, q?, q?, hybridisation]
	'''
	name, element = atom_index[0], atom_index[1]
	backbone = {'N', 'CA', 'C', 'O', 'H', 'HA', 'CB', 'HB'}
	return name if name in backbone else element

def bond_potential(pose, alg='harmonic', grad=True, box=None):
	'''
	Calculates the Bond stretching potential energy for all bonded atom pairs
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		alg:  Str algorithm type either 'harmonic' or 'morse'
		grad: bool - if True, also return per-atom forces (N, 3) array
		box:  None for no PBC; (3,) array for orthorhombic; (3, 3) for triclinic
	Returns:
	--------
		float: potential energy in kcal/mol  (when grad=False)
		(float, ndarray): energy and (N, 3) forces  (when grad=True)
	'''
	atoms = pose.data['Atoms']
	n = len(atoms)
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	idx = np.array(
		[(int(k), int(j)) for k, vs in pose.data['Bonds'].items()
		for j in vs], dtype=np.int64).reshape(-1, 2)
	idx.sort(axis=1)
	pairs = np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
	if len(pairs) == 0:
		return (0.0, np.zeros((n, 3))) if grad else 0.0
	i_idx, j_idx = pairs[:, 0], pairs[:, 1]
	dvec = _wrap(coords[i_idx] - coords[j_idx], box)
	r = np.linalg.norm(dvec, axis=1)
	P  = Parameters['bonds']
	df = P['default']
	params = np.array([P.get(tuple(sorted((
		_atomtype(atoms[int(i)]),
		_atomtype(atoms[int(j)])))),
		df) for i, j in pairs], dtype=np.float64).reshape(-1, 4)
	Kb, De, a, r0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
	if   alg.upper() == 'HARMONIC':
		dr = r - r0
		energy = float(np.sum(Kb * dr**2))
		if not grad: return energy
		coef = -2.0 * Kb * dr / r
	elif alg.upper() == 'MORSE':
		dr = r - r0
		e_decay = np.exp(-a * dr)
		energy = float(np.sum(De * (1 - e_decay)**2))
		if not grad: return energy
		coef = -2.0 * De * (1 - e_decay) * a * e_decay / r
	else:
		raise ValueError('Algorithm not supported, choose (harmonic or morse)')
	forces = np.zeros((n, 3), dtype=np.float64)
	fij = coef[:, None] * dvec
	np.add.at(forces, i_idx, fij)
	np.add.at(forces, j_idx, -fij)
	return energy, forces

def angle_potential(pose, grad=True, box=None):
	'''
	Calculates the Harmonic Angle potential for every bonded triplet
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		grad: bool - if True, also return per-atom forces (N, 3) array
		box:  None for no PBC; (3,) array for orthorhombic; (3, 3) for triclinic
	Returns:
	--------
		float: potential energy in kcal/mol  (when grad=False)
		(float, ndarray): energy and (N, 3) forces  (when grad=True)
	'''
	atoms = pose.data['Atoms']
	n = len(atoms)
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	idx = np.array(
		[(int(k), int(j)) for k, vs in pose.data['Bonds'].items()
		for j in vs], dtype=np.int64).reshape(-1, 2)
	idx.sort(axis=1)
	pairs = np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
	flat = np.concatenate([pairs, pairs[:, ::-1]])
	nbrs = {int(a): np.sort(flat[flat[:, 0] == a, 1])
		for a in np.unique(flat[:, 0])}
	triplets = np.array(
		[(int(i), j, int(k)) for j, ns in nbrs.items()
		for p, i in enumerate(ns) for k in ns[p+1:]],
		dtype=np.int64).reshape(-1, 3)
	if len(triplets) == 0:
		return (0.0, np.zeros((n, 3))) if grad else 0.0
	i_idx, j_idx, k_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
	v1 = _wrap(coords[i_idx] - coords[j_idx], box)
	v2 = _wrap(coords[k_idx] - coords[j_idx], box)
	mag1 = np.linalg.norm(v1, axis=1)
	mag2 = np.linalg.norm(v2, axis=1)
	cos = np.einsum('ij,ij->i', v1, v2) / (mag1 * mag2)
	cos = np.clip(cos, -1.0, 1.0)
	theta = np.arccos(cos)
	P  = Parameters['angles']
	df = P['default']
	params = np.array([P.get((
		min(_atomtype(atoms[int(i)]), _atomtype(atoms[int(k)])),
		_atomtype(atoms[int(j)]),
		max(_atomtype(atoms[int(i)]), _atomtype(atoms[int(k)]))),
		df) for i, j, k in triplets], dtype=np.float64).reshape(-1, 4)
	K_theta, theta0 = params[:, 0], np.deg2rad(params[:, 1])
	energy = float(np.sum(K_theta * (theta - theta0)**2))
	if not grad: return energy
	forces = np.zeros((n, 3), dtype=np.float64)
	dU_dth = 2.0 * K_theta * (theta - theta0)
	sin_th = np.sqrt(np.clip(1.0 - cos**2, 1e-12, None))
	u1 = v1 / mag1[:, None]
	u2 = v2 / mag2[:, None]
	factor_i = (dU_dth / (sin_th * mag1))[:, None]
	factor_k = (dU_dth / (sin_th * mag2))[:, None]
	Fi = factor_i * (u2 - cos[:, None] * u1)
	Fk = factor_k * (u1 - cos[:, None] * u2)
	Fj = -(Fi + Fk)
	np.add.at(forces, i_idx, Fi)
	np.add.at(forces, j_idx, Fj)
	np.add.at(forces, k_idx, Fk)
	return energy, forces

def lj_potential(pose, alg='12-6', grad=True, box=None):
	'''
	Calculates the total Lennard-Jones non-bonded potential for all atom pairs
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		alg:  Str algorithm type either '12-6' or '9-6'
		grad: bool - if True, also return per-atom forces (N, 3) array
		box:  None for no PBC; (3,) array for orthorhombic; (3, 3) for triclinic
	Returns:
	--------
		float: potential energy in kcal/mol  (when grad=False)
		(float, ndarray): energy and (N, 3) forces  (when grad=True)
	'''
	atoms = pose.data['Atoms']
	n = len(atoms)
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	idx = np.array(
		[(int(k), int(j)) for k, vs in pose.data['Bonds'].items()
		for j in vs], dtype=np.int64).reshape(-1, 2)
	idx.sort(axis=1)
	pairs = np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
	flat = np.concatenate([pairs, pairs[:, ::-1]])
	nbrs = {int(a): np.sort(flat[flat[:, 0] == a, 1])
		for a in np.unique(flat[:, 0])}
	excl_13 = np.array(
		[(int(i), int(k)) for j, ns in nbrs.items()
		for p, i in enumerate(ns) for k in ns[p+1:]],
		dtype=np.int64).reshape(-1, 2)
	excl_14 = np.array(
		[(int(i), int(l)) for j, k in pairs
		for i in nbrs[int(j)] if i != k
		for l in nbrs[int(k)] if l != j and l != i],
		dtype=np.int64).reshape(-1, 2)
	excl_14.sort(axis=1)
	excl_14 = np.unique(excl_14[excl_14[:, 0] != excl_14[:, 1]], axis=0)
	P  = Parameters['lennard_jones']
	df = P['default']
	sig = np.array([P.get(_atomtype(atoms[i]), P.get(atoms[i][1], df))[0]
		for i in range(n)], dtype=np.float64)
	sigma   = 0.5 * (sig[:, None] + sig[None, :])
	eps = np.array([P.get(_atomtype(atoms[i]), P.get(atoms[i][1], df))[1]
		for i in range(n)], dtype=np.float64)
	epsilon = np.sqrt(eps[:, None] * eps[None, :])
	dvec = _wrap(coords[:, None, :] - coords[None, :, :], box)
	r = np.linalg.norm(dvec, axis=-1)
	np.fill_diagonal(r, 1.0)
	excl = np.eye(n, dtype=bool)
	excl[pairs[:, 0], pairs[:, 1]] = True
	excl[pairs[:, 1], pairs[:, 0]] = True
	excl[excl_13[:, 0], excl_13[:, 1]] = True
	excl[excl_13[:, 1], excl_13[:, 0]] = True
	scal14 = np.zeros((n, n), dtype=bool)
	scal14[excl_14[:, 0], excl_14[:, 1]] = True
	scal14[excl_14[:, 1], excl_14[:, 0]] = True
	scal14 &= ~excl
	upper = np.triu(np.ones((n, n), dtype=bool), k=1)
	mask14  = scal14 & upper
	mask_far = (~excl) & (~scal14) & upper
	f_lj = Parameters['scaling_14']['f_lj']
	if   alg == '12-6':
		ratio_6  = (sigma / r)**6
		ratio_12 = ratio_6**2
		lj = 4.0 * epsilon * (ratio_12 - ratio_6)
		dU_dr = -24.0 * epsilon * (2*ratio_12 - ratio_6) / r
	elif alg == '9-6':
		ratio_6 = (sigma / r)**6
		ratio_9 = (sigma / r)**9
		lj = epsilon * (2*ratio_9 - 3*ratio_6)
		dU_dr = -18.0 * epsilon * (ratio_9 - ratio_6) / r
	else:
		raise ValueError('Algorithm not supported, choose (12-6 or 9-6)')
	energy = float(np.sum(lj[mask_far]) + f_lj * np.sum(lj[mask14]))
	if not grad: return energy
	weight = np.where(excl, 0.0, np.where(scal14, f_lj, 1.0))
	coef = -dU_dr / r * weight
	fij_per_pair = coef[:, :, None] * dvec
	forces = np.sum(fij_per_pair, axis=1)
	return energy, forces

def electrostatic_potential(pose, alg='constant', grad=True, box=None):
	'''
	Calculates the total Electrostatic non-bonded potential for all atom pairs
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		alg:  Str algorithm type either 'constant' (uniform εr) or 'ddd'
			(distance-dependent dielectric, ε(r) = εr·r)
		grad: bool - if True, also return per-atom forces (N, 3) array
		box:  None for no PBC; (3,) array for orthorhombic; (3, 3) for triclinic
	Returns:
	--------
		float: potential energy in kcal/mol  (when grad=False)
		(float, ndarray): energy and (N, 3) forces  (when grad=True)
	'''
	atoms = pose.data['Atoms']
	n = len(atoms)
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	idx = np.array(
		[(int(k), int(j)) for k, vs in pose.data['Bonds'].items()
		for j in vs], dtype=np.int64).reshape(-1, 2)
	idx.sort(axis=1)
	pairs = np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
	flat = np.concatenate([pairs, pairs[:, ::-1]])
	nbrs = {int(a): np.sort(flat[flat[:, 0] == a, 1])
		for a in np.unique(flat[:, 0])}
	excl_13 = np.array(
		[(int(i), int(k)) for j, ns in nbrs.items()
		for p, i in enumerate(ns) for k in ns[p+1:]],
		dtype=np.int64).reshape(-1, 2)
	excl_14 = np.array(
		[(int(i), int(l)) for j, k in pairs
		for i in nbrs[int(j)] if i != k
		for l in nbrs[int(k)] if l != j and l != i],
		dtype=np.int64).reshape(-1, 2)
	excl_14.sort(axis=1)
	excl_14 = np.unique(excl_14[excl_14[:, 0] != excl_14[:, 1]], axis=0)
	q = np.array([atoms[i][2] for i in range(n)], dtype=np.float64)
	qq = q[:, None] * q[None, :]
	dvec = _wrap(coords[:, None, :] - coords[None, :, :], box)
	r = np.linalg.norm(dvec, axis=-1)
	np.fill_diagonal(r, 1.0)
	excl = np.eye(n, dtype=bool)
	excl[pairs[:, 0], pairs[:, 1]] = True
	excl[pairs[:, 1], pairs[:, 0]] = True
	excl[excl_13[:, 0], excl_13[:, 1]] = True
	excl[excl_13[:, 1], excl_13[:, 0]] = True
	scal14 = np.zeros((n, n), dtype=bool)
	scal14[excl_14[:, 0], excl_14[:, 1]] = True
	scal14[excl_14[:, 1], excl_14[:, 0]] = True
	scal14 &= ~excl
	upper = np.triu(np.ones((n, n), dtype=bool), k=1)
	mask_far = (~excl) & (~scal14) & upper
	mask_14  = scal14 & upper
	epsilon_r = Parameters['electrostatic']['epsilon_r']
	if   alg == 'constant':
		elec = (332.06 * qq) / (epsilon_r * r)
		dU_dr = -elec / r
	elif alg == 'ddd':
		elec = (332.06 * qq) / (epsilon_r * r * r)
		dU_dr = -2.0 * elec / r
	else:
		raise ValueError('Algorithm not supported, choose (constant or ddd)')
	f_elec = Parameters['scaling_14']['f_elec']
	energy = float(np.sum(elec[mask_far]) + f_elec * np.sum(elec[mask_14]))
	if not grad: return energy
	weight = np.where(excl, 0.0, np.where(scal14, f_elec, 1.0))
	coef = -dU_dr / r * weight
	fij_per_pair = coef[:, :, None] * dvec
	forces = np.sum(fij_per_pair, axis=1)
	return energy, forces

def dihedral_potential(pose, grad=True, box=None):
	'''
	Calculates the total Proper Dihedral (torsion) potential for i-j-k-l atoms
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		grad: bool - if True, also return per-atom forces (N, 3) array
		box:  None for no PBC; (3,) array for orthorhombic; (3, 3) for triclinic
	Returns:
	--------
		float: potential energy in kcal/mol  (when grad=False)
		(float, ndarray): energy and (N, 3) forces  (when grad=True)
	'''
	atoms = pose.data['Atoms']
	n = len(atoms)
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	idx = np.array(
		[(int(k), int(j)) for k, vs in pose.data['Bonds'].items()
		for j in vs], dtype=np.int64).reshape(-1, 2)
	idx.sort(axis=1)
	pairs = np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
	flat = np.concatenate([pairs, pairs[:, ::-1]])
	nbrs = {int(a): np.sort(flat[flat[:, 0] == a, 1])
		for a in np.unique(flat[:, 0])}
	quartets = np.array(
		[(int(i), int(j), int(k), int(l)) for j, k in pairs
		for i in nbrs[int(j)] if i != k
		for l in nbrs[int(k)] if l != j and l != i],
		dtype=np.int64).reshape(-1, 4)
	rev = quartets[:, ::-1]
	swap = (quartets[:, 0] > rev[:, 0]) | (
		(quartets[:, 0] == rev[:, 0]) & (quartets[:, 1] > rev[:, 1]))
	quartets = np.where(swap[:, None], rev, quartets)
	quartets = np.unique(quartets, axis=0)
	if len(quartets) == 0:
		return (0.0, np.zeros((n, 3))) if grad else 0.0
	i_idx, j_idx, k_idx, l_idx = quartets.T
	b1 = _wrap(coords[j_idx] - coords[i_idx], box)
	b2 = _wrap(coords[k_idx] - coords[j_idx], box)
	b3 = _wrap(coords[l_idx] - coords[k_idx], box)
	n1 = np.cross(b1, b2)
	n2 = np.cross(b2, b3)
	b2_mag = np.linalg.norm(b2, axis=1)
	b2n = b2 / b2_mag[:, None]
	phi = np.arctan2(
		np.einsum('ij,ij->i', np.cross(n1, b2n), n2),
		np.einsum('ij,ij->i', n1, n2))
	P  = Parameters['dihedrals']
	df = P['default']
	component_lists = []
	for i, j, k, l in quartets:
		t = (_atomtype(atoms[int(i)]), _atomtype(atoms[int(j)]),
			_atomtype(atoms[int(k)]), _atomtype(atoms[int(l)]))
		if t > t[::-1]: t = t[::-1]
		component_lists.append(P.get(t, df))
	counts = np.array([len(c) for c in component_lists], dtype=np.int64)
	phi_flat = np.repeat(phi, counts)
	flat_p = np.array([row for cl in component_lists for row in cl],
		dtype=np.float64).reshape(-1, 3)
	k_phi, n_mult, phi0 = flat_p[:, 0], flat_p[:, 1], np.deg2rad(flat_p[:, 2])
	energy = float(np.sum(k_phi * (1 + np.cos(n_mult * phi_flat - phi0))))
	if not grad: return energy
	# Per-component dU/dφ summed back to per-quartet via add.at scatter.
	dU_dphi_flat = -k_phi * n_mult * np.sin(n_mult * phi_flat - phi0)
	dU_dphi = np.zeros(len(quartets), dtype=np.float64)
	# quartet index for each component row
	q_idx = np.repeat(np.arange(len(quartets)), counts)
	np.add.at(dU_dphi, q_idx, dU_dphi_flat)
	forces = np.zeros((n, 3), dtype=np.float64)
	n1_sq = np.einsum('ij,ij->i', n1, n1)
	n2_sq = np.einsum('ij,ij->i', n2, n2)
	Fi = -(dU_dphi * b2_mag / n1_sq)[:, None] * n1
	Fl =  (dU_dphi * b2_mag / n2_sq)[:, None] * n2
	b1_dot_b2 = np.einsum('ij,ij->i', b1, b2)
	b3_dot_b2 = np.einsum('ij,ij->i', b3, b2)
	b2_sq = b2_mag**2
	Fj = -((b1_dot_b2/b2_sq+1.0)[:,None]*Fi) + (b3_dot_b2/b2_sq)[:,None]*Fl
	Fk = -(Fi + Fj + Fl)
	np.add.at(forces, i_idx, Fi)
	np.add.at(forces, j_idx, Fj)
	np.add.at(forces, k_idx, Fk)
	np.add.at(forces, l_idx, Fl)
	return energy, forces

def improper_dihedral_potential(pose, alg='harmonic', grad=True, box=None):
	'''
	Calculates the total Improper Dihedral potential energy
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		alg:  Str algorithm type either 'harmonic' or 'fourier'
		grad: bool - if True, also return per-atom forces (N, 3) array
		box:  None for no PBC; (3,) array for orthorhombic; (3, 3) for triclinic
	Returns:
	--------
		float: potential energy in kcal/mol  (when grad=False)
		(float, ndarray): energy and (N, 3) forces  (when grad=True)
	'''
	atoms = pose.data['Atoms']
	n = len(atoms)
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	idx = np.array(
		[(int(k), int(j)) for k, vs in pose.data['Bonds'].items()
		for j in vs], dtype=np.int64).reshape(-1, 2)
	idx.sort(axis=1)
	pairs = np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
	flat = np.concatenate([pairs, pairs[:, ::-1]])
	nbrs = {int(a): np.sort(flat[flat[:, 0] == a, 1])
		for a in np.unique(flat[:, 0])}
	impropers = np.array(
		[(int(ns[0]), int(j), int(ns[1]), int(ns[2]))
		for j, ns in nbrs.items() if len(ns) == 3],
		dtype=np.int64).reshape(-1, 4)
	if len(impropers) == 0:
		return (0.0, np.zeros((n, 3))) if grad else 0.0
	i_idx, j_idx, k_idx, l_idx = impropers.T
	b1 = _wrap(coords[j_idx] - coords[i_idx], box)
	b2 = _wrap(coords[k_idx] - coords[j_idx], box)
	b3 = _wrap(coords[l_idx] - coords[k_idx], box)
	n1 = np.cross(b1, b2)
	n2 = np.cross(b2, b3)
	b2_mag = np.linalg.norm(b2, axis=1)
	b2n = b2 / b2_mag[:, None]
	psi = np.arctan2(
		np.einsum('ij,ij->i', np.cross(n1, b2n), n2),
		np.einsum('ij,ij->i', n1, n2))
	P  = Parameters['impropers']
	df = P['default']
	keys = []
	for i, j, k, l in impropers:
		nb = sorted([_atomtype(atoms[int(i)]),
			_atomtype(atoms[int(k)]),
			_atomtype(atoms[int(l)])])
		keys.append((_atomtype(atoms[int(j)]), nb[0], nb[1], nb[2]))
	params = np.array([P.get(key, df) for key in keys],
		dtype=np.float64).reshape(-1, 3)
	k_imp = params[:, 0]
	n_mult = params[:, 1]
	psi0  = np.deg2rad(params[:, 2])
	if   alg == 'harmonic':
		delta = ((psi - psi0 + np.pi) % (2 * np.pi)) - np.pi
		energy = float(np.sum(k_imp * delta**2))
		dU_dphi = 2.0 * k_imp * delta
	elif alg == 'fourier':
		energy = float(np.sum(k_imp * (1 + np.cos(n_mult * psi - psi0))))
		dU_dphi = -k_imp * n_mult * np.sin(n_mult * psi - psi0)
	else:
		raise ValueError(
		'Algorithm not supported, choose (harmonic or fourier)')
	if not grad: return energy
	forces = np.zeros((n, 3), dtype=np.float64)
	n1_sq = np.einsum('ij,ij->i', n1, n1)
	n2_sq = np.einsum('ij,ij->i', n2, n2)
	# Derived chain rule for our atan2 convention (verified against FD):
	#   ∂φ/∂x_i = +|b2|·n1/n1²,  ∂φ/∂x_l = -|b2|·n2/n2²
	# Force F_a = -(dU/dφ)·(∂φ/∂x_a):
	Fi = -(dU_dphi * b2_mag / n1_sq)[:, None] * n1
	Fl =  (dU_dphi * b2_mag / n2_sq)[:, None] * n2
	b1_dot_b2 = np.einsum('ij,ij->i', b1, b2)
	b3_dot_b2 = np.einsum('ij,ij->i', b3, b2)
	b2_sq = b2_mag**2
	# Chain rule on x_j: ∂φ/∂x_j = (1 + b1·b2/b2²)·∂φ/∂b1 + (b3·b2/b2²)·∂φ/∂b3
	# Reduces to: F_j = -(b1·b2/b2² + 1)·F_i + (b3·b2/b2²)·F_l
	Fj = -((b1_dot_b2/b2_sq+1.0)[:,None]*Fi) + (b3_dot_b2/b2_sq)[:,None]*Fl
	Fk = -(Fi + Fj + Fl)
	np.add.at(forces, i_idx, Fi)
	np.add.at(forces, j_idx, Fj)
	np.add.at(forces, k_idx, Fk)
	np.add.at(forces, l_idx, Fl)
	return energy, forces

def ub_potential(pose, grad=True, box=None):
	'''
	Calculates Urey-Bradley 1-3 stretching potential between all three atoms
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		grad: bool - if True, also return per-atom forces (N, 3) array
		box:  None for no PBC; (3,) array for orthorhombic; (3, 3) for triclinic
	Returns:
	--------
		float: potential energy in kcal/mol  (when grad=False)
		(float, ndarray): energy and (N, 3) forces  (when grad=True)
	'''
	atoms = pose.data['Atoms']
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	idx = np.array(
		[(int(k), int(j)) for k, vs in pose.data['Bonds'].items()
		for j in vs], dtype=np.int64).reshape(-1, 2)
	idx.sort(axis=1)
	pairs = np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
	flat = np.concatenate([pairs, pairs[:, ::-1]])
	nbrs = {int(a): np.sort(flat[flat[:, 0] == a, 1])
		for a in np.unique(flat[:, 0])}
	triplets = np.array(
		[(int(i), j, int(k)) for j, ns in nbrs.items()
		for p, i in enumerate(ns) for k in ns[p+1:]],
		dtype=np.int64).reshape(-1, 3)
	n = len(atoms)
	if len(triplets) == 0:
		return (0.0, np.zeros((n, 3))) if grad else 0.0
	i_idx, j_idx, k_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
	dvec = _wrap(coords[i_idx] - coords[k_idx], box)
	r = np.linalg.norm(dvec, axis=1)
	P  = Parameters['angles']
	df = P['default']
	params = np.array([P.get((
		min(_atomtype(atoms[int(i)]), _atomtype(atoms[int(k)])),
		_atomtype(atoms[int(j)]),
		max(_atomtype(atoms[int(i)]), _atomtype(atoms[int(k)]))),
		df) for i, j, k in triplets], dtype=np.float64).reshape(-1, 4)
	k_ub, s0 = params[:, 2], params[:, 3]
	energy = float(np.sum(k_ub * (r - s0)**2))
	if not grad: return energy
	forces = np.zeros((n, 3), dtype=np.float64)
	coef = -2.0 * k_ub * (r - s0) / r
	fik = coef[:, None] * dvec
	np.add.at(forces, i_idx, fik)
	np.add.at(forces, k_idx, -fik)
	return energy, forces

def polarisation_potential(pose, alg='constant', grad=True, box=None):
	'''
	Calculates total Polarisation potential energy ½·Σᵢ αᵢ·|Eᵢ|² where
	Eᵢ is the local electric field at atom i from all other permanent
	charges (excluding 1-2 and 1-3 pairs, scaling 1-4 pairs by f_elec).
	Non-iterative — uses permanent charges only, no self-consistency.
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		alg:  Str algorithm type either 'constant' or 'ddd'
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	atoms = pose.data['Atoms']
	n = len(atoms)
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	idx = np.array(
		[(int(k), int(j)) for k, vs in pose.data['Bonds'].items()
		for j in vs], dtype=np.int64).reshape(-1, 2)
	idx.sort(axis=1)
	pairs = np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
	flat = np.concatenate([pairs, pairs[:, ::-1]])
	nbrs = {int(a): np.sort(flat[flat[:, 0] == a, 1])
		for a in np.unique(flat[:, 0])}
	excl_13 = np.array(
		[(int(i), int(k)) for j, ns in nbrs.items()
		for p, i in enumerate(ns) for k in ns[p+1:]],
		dtype=np.int64).reshape(-1, 2)
	excl_14 = np.array(
		[(int(i), int(l)) for j, k in pairs
		for i in nbrs[int(j)] if i != k
		for l in nbrs[int(k)] if l != j and l != i],
		dtype=np.int64).reshape(-1, 2)
	excl_14.sort(axis=1)
	excl_14 = np.unique(excl_14[excl_14[:, 0] != excl_14[:, 1]], axis=0)
	q = np.array([atoms[i][2] for i in range(n)], dtype=np.float64)
	P  = Parameters['lennard_jones']
	df = P['default']
	alpha = np.array(
		[P.get(_atomtype(atoms[i]), P.get(atoms[i][1], df))[2]
		for i in range(n)], dtype=np.float64)
	dr = _wrap(coords[:, None, :] - coords[None, :, :], box)
	r = np.linalg.norm(dr, axis=-1)
	np.fill_diagonal(r, 1.0)
	excl = np.eye(n, dtype=bool)
	excl[pairs[:, 0], pairs[:, 1]] = True
	excl[pairs[:, 1], pairs[:, 0]] = True
	excl[excl_13[:, 0], excl_13[:, 1]] = True
	excl[excl_13[:, 1], excl_13[:, 0]] = True
	scal14 = np.zeros((n, n), dtype=bool)
	scal14[excl_14[:, 0], excl_14[:, 1]] = True
	scal14[excl_14[:, 1], excl_14[:, 0]] = True
	scal14 &= ~excl
	f_elec = Parameters['scaling_14']['f_elec']
	weight = np.where(excl, 0.0, np.where(scal14, f_elec, 1.0))
	epsilon_r = Parameters['electrostatic']['epsilon_r']
	if alg == 'constant':
		coeff = 332.06 * q[None, :] / (epsilon_r * r**3)
	elif alg == 'ddd':
		coeff = 332.06 * q[None, :] / (epsilon_r * r**4)
	else:
		raise ValueError('Algorithm not supported, choose (constant or ddd)')
	coeff = coeff * weight
	E = np.einsum('ij,ijk->ik', coeff, dr)
	E_sq = np.sum(E**2, axis=1)
	energy = float(0.5 * np.sum(alpha * E_sq))
	if not grad: return energy
	# Polarisation gradient. Define per pair:
	#   A[i,j] = α_i · coef[i,j]                          (scalar)
	#   r̂[i,j] = (x_i - x_j) / r_ij
	#   G[i,j] = E_i - p · (E_i · r̂[i,j]) · r̂[i,j]      (3-vector)
	#   M[i,j] = A[i,j] · G[i,j]                          (3-vector)
	# where p = 3 (constant) or 4 (ddd) — the exponent of (1/r) in the
	# field expression governs the [I − p·r̂⊗r̂] form of ∂E_i/∂x_a.
	# Then F_a = -Σ_j M[a,j] + Σ_i M[i,a].
	p_pow = 3.0 if alg == 'constant' else 4.0
	rhat = dr / r[:, :, None]                            # (N, N, 3)
	# E_i · r̂[i,j] for each (i, j) — note this depends on i and j
	E_dot_rhat = np.einsum('ik,ijk->ij', E, rhat)        # (N, N)
	G = E[:, None, :] - p_pow * E_dot_rhat[:, :, None] * rhat  # (N, N, 3)
	A = alpha[:, None] * coeff                           # (N, N)
	M = A[:, :, None] * G                                # (N, N, 3)
	# Diagonal cleanup — coeff is already 0 on diagonal via weight, so
	# A and M are 0 on diagonal. No extra masking needed.
	forces = -np.sum(M, axis=1) + np.sum(M, axis=0)
	return energy, forces

def cmap_potential(pose, grad=True, box=None):
	'''
	Calculates the total CMAP backbone (φ, ψ) correction energy for
	every interior protein residue. Bicubic Catmull-Rom interpolation
	of per-residue 24×24 tables stored in Parameters['cmap'].
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		grad: bool - if True, also return per-atom forces (N, 3) array
		box:  None for no PBC; (3,) orthorhombic; (3, 3) triclinic
	Returns:
	--------
		float: potential energy in kcal/mol  (when grad=False)
		(float, ndarray): energy and (N, 3) forces  (when grad=True)
	'''
	atoms = pose.data['Atoms']
	n = len(atoms)
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	aas = pose.data.get('Amino Acids')
	if aas is None or len(aas) < 3:
		return (0.0, np.zeros((n, 3))) if grad else 0.0
	# Per-residue {atom-name → atom-index} for backbone atoms.
	res_atoms = {ai: (info[0], info[1],
		{atoms[k][0]: k for k in info[2]}) for ai, info in aas.items()}
	# Enumerate (φ, ψ) quartets per interior residue with same-chain neighbours.
	phi_q, psi_q, codes = [], [], []
	for r in sorted(res_atoms.keys())[1:-1]:
		aa_curr, ch_curr, names_curr = res_atoms[r]
		_,       ch_prev, names_prev = res_atoms[r - 1]
		_,       ch_next, names_next = res_atoms[r + 1]
		if ch_curr != ch_prev or ch_curr != ch_next: continue
		try:
			C_prev  = names_prev['C']
			N_curr  = names_curr['N']
			CA_curr = names_curr['CA']
			C_curr  = names_curr['C']
			N_next  = names_next['N']
		except KeyError: continue
		phi_q.append((C_prev, N_curr, CA_curr, C_curr))
		psi_q.append((N_curr, CA_curr, C_curr, N_next))
		codes.append(aa_curr)
	if len(phi_q) == 0:
		return (0.0, np.zeros((n, 3))) if grad else 0.0
	M = len(phi_q)
	phi_q = np.asarray(phi_q, dtype=np.int64)
	psi_q = np.asarray(psi_q, dtype=np.int64)
	# Stack φ and ψ quartets — shared Bekker math via concatenation.
	quartets = np.concatenate([phi_q, psi_q], axis=0)
	i_idx, j_idx, k_idx, l_idx = quartets.T
	b1 = _wrap(coords[j_idx] - coords[i_idx], box)
	b2 = _wrap(coords[k_idx] - coords[j_idx], box)
	b3 = _wrap(coords[l_idx] - coords[k_idx], box)
	n1 = np.cross(b1, b2)
	n2 = np.cross(b2, b3)
	b2_mag = np.linalg.norm(b2, axis=1)
	b2n = b2 / b2_mag[:, None]
	ang = np.arctan2(
		np.einsum('ij,ij->i', np.cross(n1, b2n), n2),
		np.einsum('ij,ij->i', n1, n2))
	phi, psi = ang[:M], ang[M:]
	# Gather per-residue 24×24 CMAP tables.
	P  = Parameters['cmap']
	df = P['default']
	tables = np.stack([np.asarray(P.get(c, df), dtype=np.float64)
		for c in codes])                             # (M, N, N)
	N_grid = tables.shape[1]
	H = 2.0 * np.pi / N_grid
	# Map angles [-π, π) → [0, N) grid coords; find cell (i, j) and (u, v).
	x = (phi + np.pi) / H
	y = (psi + np.pi) / H
	gi = np.floor(x).astype(np.int64) % N_grid
	gj = np.floor(y).astype(np.int64) % N_grid
	u = x - np.floor(x)
	v = y - np.floor(y)
	# 4×4 stencil per residue: stencil[m, a, b]=tables[m,(gi+a-1)%N,(gj+b-1)%N].
	off = np.array([-1, 0, 1, 2])
	a_grid = (gi[:, None, None] + off[None, :, None]) % N_grid
	b_grid = (gj[:, None, None] + off[None, None, :]) % N_grid
	stencil = tables[np.arange(M)[:, None, None], a_grid, b_grid]    # (M, 4, 4)
	# Catmull-Rom basis weights and their u/v derivatives.
	uw  = 0.5 * np.stack([-u + 2*u**2 - u**3, 2 - 5*u**2 + 3*u**3,
		u + 4*u**2 - 3*u**3, -u**2 + u**3], axis=-1)
	vw  = 0.5 * np.stack([-v + 2*v**2 - v**3, 2 - 5*v**2 + 3*v**3,
		v + 4*v**2 - 3*v**3, -v**2 + v**3], axis=-1)
	duw = 0.5 * np.stack([-1 + 4*u - 3*u**2, -10*u + 9*u**2,
		1 + 8*u - 9*u**2, -2*u + 3*u**2], axis=-1)
	dvw = 0.5 * np.stack([-1 + 4*v - 3*v**2, -10*v + 9*v**2,
		1 + 8*v - 9*v**2, -2*v + 3*v**2], axis=-1)
	E_per = np.einsum('ma,mab,mb->m', uw, stencil, vw)
	dE_du = np.einsum('ma,mab,mb->m', duw, stencil, vw)
	dE_dv = np.einsum('ma,mab,mb->m', uw, stencil, dvw)
	# Cell-fraction gradient → angular gradient (cell width = H rad).
	dU_dphi = dE_du / H
	dU_dpsi = dE_dv / H
	energy = float(np.sum(E_per))
	if not grad: return energy
	# Bekker dihedral force scatter on stacked quartets — same math as
	# dihedral_potential, applied once on (2M, 4) with stacked dU/dφ + dU/dψ.
	dU_d = np.concatenate([dU_dphi, dU_dpsi])
	n1_sq = np.einsum('ij,ij->i', n1, n1)
	n2_sq = np.einsum('ij,ij->i', n2, n2)
	Fi = -(dU_d * b2_mag / n1_sq)[:, None] * n1
	Fl =  (dU_d * b2_mag / n2_sq)[:, None] * n2
	b1_dot_b2 = np.einsum('ij,ij->i', b1, b2)
	b3_dot_b2 = np.einsum('ij,ij->i', b3, b2)
	b2_sq = b2_mag**2
	Fj = -((b1_dot_b2/b2_sq+1.0)[:,None]*Fi) + (b3_dot_b2/b2_sq)[:,None]*Fl
	Fk = -(Fi + Fj + Fl)
	forces = np.zeros((n, 3), dtype=np.float64)
	np.add.at(forces, i_idx, Fi)
	np.add.at(forces, j_idx, Fj)
	np.add.at(forces, k_idx, Fk)
	np.add.at(forces, l_idx, Fl)
	return energy, forces

















def Energy(pose, alg='lennard_jones'):
	''' Main energy function where you choose which force field to use '''
	if alg.upper() == 'LENNARD_JONES':
		return lj_potential(pose, alg='12-6')
	else:
		raise Exception('Algorithm no supported')
