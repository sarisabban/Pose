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
		# i    Sigma  eps
		'H' : (2.886, 0.044),
		'N' : (3.660, 0.069),
		'F' : (3.364, 0.050),
		'S' : (4.035, 0.274),
		'Br': (4.189, 0.251),
		'Se': (4.205, 0.291),
		'C' : (3.851, 0.105),
		'O' : (3.500, 0.060),
		'P' : (4.147, 0.305),
		'Cl': (3.947, 0.227),
		'I' : (4.500, 0.339),
		'default': (3.851, 0.105)},
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
	Calculates the Harmonic Bond Stretching potential energy between two atoms
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
'''











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
