import numpy as np

def Lennard_Jones(pose, terms=None):
	# Universal Force Field (Rappe 1992) Lennard-Jones parameters, in units
	# of (sigma [Angstrom], epsilon [kcal/mol]). Used by `energy()` below and
	# by the BBDEP scanner inside Parameterise(). Unknown elements fall back
	# to carbon. Extend this table if you register NCAAs containing exotic
	# elements (e.g. boron, metals).
	'''
	Total Lennard-Jones 12-6 potential energy (kcal/mol) for a Pose's
	current conformation. 1-2 and 1-3 bonded atom pairs are excluded
	(standard convention: directly bonded atoms and atoms two bonds
	apart don't contribute to the non-bonded energy).
	Arguments:
	----------
		pose   : Pose - a built protein/nucleic-acid pose whose
				 current coordinates are the conformation to evaluate.
		terms : tuple or None - optional precomputed
				 (sigma_ij, epsilon_ij, pair_mask) triple for hot-loop
				 callers (e.g. Parameterise's chi scanner). When None, the
				 element-pair parameters and the 1-2/1-3 exclusion mask are
				 derived from `pose.data['Atoms']` and `pose.data['Bonds']`
				 on each call.
	Returns:
	--------
		float - total LJ energy in kcal/mol.
	This is a deliberately minimal ranker: sterics dominate side-chain
	rotamer preference, and the rotameric (+/- 60, 180) wells are
	already sampled by the scanner's starting points. Replace this
	function with a full force field (MMFF94, AMBER) when available --
	the BBDEP scanner only assumes `(pose, terms=None) -> float`.
	'''
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
	else:
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
	if alg.upper() == 'LENNARD_JONES': return Lennard_Jones(pose, terms)
	else: raise Exception('Algorithm no supported')
