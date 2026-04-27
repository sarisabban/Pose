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
		#i     j     k      theta0 K theta
		('C',  'CA', 'N' ): (63.0, 110.1),
		('CA', 'C',  'N' ): (70.0, 116.6),
		('CA', 'C',  'O' ): (80.0, 120.4),
		('N',  'C',  'O' ): (80.0, 122.9),
		('C',  'N',  'CA'): (50.0, 121.9),
		('CA', 'N',  'H' ): (50.0, 118.0),
		('C',  'N',  'H' ): (50.0, 119.8),
		('CB', 'CA', 'N' ): (80.0, 109.7),
		('C',  'CA', 'CB'): (63.0, 111.1),
		('CB', 'CA', 'HA'): (50.0, 109.5),
		('C',  'CA', 'HA'): (50.0, 109.5),
		('HA', 'CA', 'N' ): (50.0, 109.5),
		('CA', 'CB', 'HB'): (50.0, 109.5),
		('HB', 'CB', 'HB'): (35.0, 109.5),
		# Generic element-pair stand-ins so unknown types still resolve.
		('C',  'C',  'C' ): (63.0, 111.0),
		('C',  'C',  'H' ): (50.0, 109.5),
		('H',  'C',  'H' ): (35.0, 109.5),
		('C',  'C',  'N' ): (80.0, 110.0),
		('C',  'C',  'O' ): (80.0, 120.0),
		('H',  'N',  'H' ): (35.0, 109.5),
		('C',  'O',  'H' ): (55.0, 108.5),
		('CA', 'CB', 'H' ): (50.0, 109.5),
		('H',  'CB', 'H' ): (35.0, 109.5),
		('C',  'CB', 'H' ): (50.0, 109.5),
		'default': (50.0, 109.5)},
	'lennard_jones': {
		#i     sigma  epsilon
		'H' : (2.886, 0.044),
		'C' : (3.851, 0.105),
		'N' : (3.660, 0.069),
		'O' : (3.500, 0.060),
		'F' : (3.364, 0.050),
		'P' : (4.147, 0.305),
		'S' : (4.035, 0.274),
		'Cl': (3.947, 0.227),
		'Br': (4.189, 0.251),
		'I' : (4.500, 0.339),
		'Se': (4.205, 0.291),
		'default': (3.851, 0.105)},
	'electrostatic': {
		'epsilon_r': 1.0},
	'scaling_14': {
		'f_lj'  : 0.5,
		'f_elec': 1.0 / 1.2},
}

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

def bond_potential(pose, alg='harmonic'):
	'''
	Calculate the Harmonic Bond potential energy between all atom pairs
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		alg:  Str algorithm type either 'harmonic' or 'morse'
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	atoms = pose.data['Atoms']
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	idx = np.array(
		[(int(k), int(j)) for k, vs in pose.data['Bonds'].items()
		for j in vs], dtype=np.int64).reshape(-1, 2)
	idx.sort(axis=1)
	pairs = np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
	i_idx, j_idx = pairs[:, 0], pairs[:, 1]
	r  = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=1)
	P  = Parameters['bonds']
	df = P['default']
	params = np.array([P.get(tuple(sorted((
		_atomtype(atoms[int(i)]),
		_atomtype(atoms[int(j)])))),
		df) for i, j in pairs], dtype=np.float64).reshape(-1, 4)
	Kb, De, a, r0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
	harmonic = float(np.sum(Kb * (r - r0)**2))
	morse = float(np.sum(De * (1 - np.exp(-a * (r - r0)))**2))
	if   alg.upper() == 'HARMONIC': return harmonic
	elif alg.upper() == 'MORSE':    return morse
	else: raise Exception('Algorithm not supported, choose (harmonic or morse)')

def angle_potential(pose):
	'''
	Calculate the Harmonic Angle potential between all three atoms
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
	Returns:
	--------
		float: potential energy in kcal/mol
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
	i_idx, j_idx, k_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
	v1 = coords[i_idx] - coords[j_idx]
	v2 = coords[k_idx] - coords[j_idx]
	cos = np.einsum('ij,ij->i', v1, v2) / (
		np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
	theta = np.arccos(np.clip(cos, -1.0, 1.0))
	P  = Parameters['angles']
	df = P['default']
	params = np.array([P.get((
		min(_atomtype(atoms[int(i)]), _atomtype(atoms[int(k)])),
		_atomtype(atoms[int(j)]),
		max(_atomtype(atoms[int(i)]), _atomtype(atoms[int(k)]))),
		df) for i, j, k in triplets], dtype=np.float64).reshape(-1, 2)
	K_theta, theta0 = params[:, 0], np.deg2rad(params[:, 1])
	return float(np.sum(K_theta * (theta - theta0)**2))

def lj_potential(pose, alg='12-6'):
	'''
	Calculates the total Lennard-Jones non-bonded potential for all atom pairs
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		alg:  Str algorithm type either '12-6' or '9-6'
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
	P  = Parameters['lennard_jones']
	df = P['default']
	sig = np.array([P.get(_atomtype(atoms[i]), P.get(atoms[i][1], df))[0]
		for i in range(n)], dtype=np.float64)
	sigma   = 0.5 * (sig[:, None] + sig[None, :])
	eps = np.array([P.get(_atomtype(atoms[i]), P.get(atoms[i][1], df))[1]
		for i in range(n)], dtype=np.float64)
	epsilon = np.sqrt(eps[:, None] * eps[None, :])
	r = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
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
	if   alg == '12-6':
		ratio_6  = (sigma / r)**6
		ratio_12 = ratio_6**2
		lj = 4.0 * epsilon * (ratio_12 - ratio_6)
	elif alg == '9-6':
		ratio_6 = (sigma / r)**6
		ratio_9 = (sigma / r)**9
		lj = epsilon * (2 * ratio_9 - 3 * ratio_6)
	else: raise Exception('Algorithm not supported, choose (12-6 or 9-6)')
	f_lj = Parameters['scaling_14']['f_lj']
	return float(np.sum(lj[mask_far]) + f_lj * np.sum(lj[mask14]))

def electrostatic_potential(pose, alg='constant'):
	'''
	Calculates the total Electrostatic non-bonded potential for all atom pairs
	Arguments:
	----------
		pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		alg:  Str algorithm type either 'constant' (uniform εr) or 'ddd'
			(distance-dependent dielectric, ε(r) = εr·r)
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
	qq = q[:, None] * q[None, :]
	r = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
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
	if   alg == 'constant': elec = (332.06 * qq) / (epsilon_r * r)
	elif alg == 'ddd':      elec = (332.06 * qq) / (epsilon_r * r * r)
	else: raise Exception('Algorithm not supported, choose (constant or ddd)')
	f_elec = Parameters['scaling_14']['f_elec']
	return float(np.sum(elec[mask_far]) + f_elec * np.sum(elec[mask_14]))



















def Lennard_Jones(r, sigma, epsilon, mask):
	ratio_12 = (sigma / r)**12
	ratio_6  = (sigma / r)**6
	lj = 4.0 * epsilon * (ratio_12 - ratio_6)
	return float(np.sum(lj[mask]))






def Energy(pose, alg='lennard_jones'):
	''' Main energy function where you choose wich force field to use '''
	if alg.upper() == 'LENNARD_JONES':
		_LJ = {
			'H':  (2.886, 0.044), 'C':  (3.851, 0.105),
			'N':  (3.660, 0.069), 'O':  (3.500, 0.060),
			'F':  (3.364, 0.050), 'P':  (4.147, 0.305),
			'S':  (4.035, 0.274), 'Cl': (3.947, 0.227),
			'Br': (4.189, 0.251), 'I':  (4.500, 0.339),
			'Se': (4.205, 0.291)}
		atoms = pose.data['Atoms']
		n = len(atoms)
		elems = [atoms[i][1] for i in range(n)]
		sig = np.array([_LJ.get(e,_LJ['C'])[0] for e in elems],dtype=np.float64)
		eps = np.array([_LJ.get(e,_LJ['C'])[1] for e in elems],dtype=np.float64)
		sigma = 0.5 * (sig[:, None] + sig[None, :])
		epsilon = np.sqrt(eps[:, None] * eps[None, :])
		nbrs = [set() for _ in range(n)]
		for k, vs in pose.data['Bonds'].items():
			i = int(k)
			for j in vs:
				nbrs[i].add(int(j))
				nbrs[int(j)].add(i)
		excl = np.eye(n, dtype=bool)
		for i in range(n):
			for j in nbrs[i]:
				excl[i, j] = True
				for kk in nbrs[j]:
					excl[i, kk] = True
		mask = (~excl) & np.triu(np.ones_like(excl), k=1).astype(bool)
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		r = coords[:, None, :] - coords[None, :, :]
		r = np.linalg.norm(r, axis=-1)
		np.fill_diagonal(r, 1.0)
		return Lennard_Jones(r, sigma, epsilon, mask)
	else:
		raise Exception('Algorithm no supported')
