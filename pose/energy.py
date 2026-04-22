import numpy as np

def Lennard_Jones(pose, terms):
	'''
	Total Lennard-Jones 12-6 potential energy (kcal/mol)
	Arguments:
	----------
		pose  : Pose
		terms : (sigma [Angstrom], epsilon [kcal/mol], mask [bool])
	Returns:
	--------
		float : total LJ energy in kcal/mol
	'''
	sigma, epsilon, mask = terms
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	diff = coords[:, None, :] - coords[None, :, :]
	r2 = np.einsum('ijk,ijk->ij', diff, diff)
	np.fill_diagonal(r2, 1.0)
	sr2 = (sigma * sigma) / r2
	sr6 = sr2 * sr2 * sr2
	sr12 = sr6 * sr6
	lj = 4.0 * epsilon * (sr12 - sr6)
	return float(np.sum(lj[mask]))




def Energy(pose, alg='lennard_jones', terms=None):
	''' Main energy function where you choose wich force field to use '''
	if alg.upper() == 'LENNARD_JONES':
		_LJ = {
			'H':  (2.886, 0.044), 'C':  (3.851, 0.105),
			'N':  (3.660, 0.069), 'O':  (3.500, 0.060),
			'F':  (3.364, 0.050), 'P':  (4.147, 0.305),
			'S':  (4.035, 0.274), 'Cl': (3.947, 0.227),
			'Br': (4.189, 0.251), 'I':  (4.500, 0.339),
			'Se': (4.205, 0.291)}
		if terms is None:
			atoms = pose.data['Atoms']
			n = len(atoms)
			elems = [atoms[i][1] for i in range(n)]
			sig = np.array([_LJ.get(e, _LJ['C'])[0] for e in elems],
				dtype=np.float64)
			eps = np.array([_LJ.get(e, _LJ['C'])[1] for e in elems],
				dtype=np.float64)
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
			terms = (sigma, epsilon, mask)
		return Lennard_Jones(pose, terms)
	else:
		raise Exception('Algorithm no supported')
