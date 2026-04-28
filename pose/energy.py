import os
import json
import math
import numpy as np

class ForceField():
	'''
	Configurable molecular mechanics force field assembled from energy terms
	'''
	def __init__(self, terms=None):
		'''
		Initialise the force field with a chosen set of energy terms
		Arguments:
		----------
			terms: list of (method_name, kwargs) tuples; None uses the default
		Returns:
		--------
			None: instance is configured in-place
		'''
		self.DEFAULT_TERMS = [
			('BondPotential',          {'alg': 'harmonic'}),
			('AnglePotential',         {}),
			('UBPotential',            {}),
			('DihedralPotential',      {}),
			('ImproperPotential',      {'alg': 'harmonic'}),
			('LJPotential',            {'alg': '12-6'}),
			('ElectrostaticPotential', {'alg': 'constant'}),
			('PolarisationPotential',  {'alg': 'constant'}),
			('CMAPPotential',          {})]
		self.terms = terms if terms is not None else self.DEFAULT_TERMS
		path = os.path.join(os.path.dirname(__file__), 'parameters.json')
		with open(path) as f:
			P = json.load(f)
		for sect in ('bonds', 'angles', 'dihedrals', 'impropers'):
			P[sect] = {(tuple(k.split('-')) if k != 'default' else k): v
				for k, v in P[sect].items()}
		self.Parameters = P
		self._cache = None
		self._cache_hash = None
	def __call__(self, pose, grad=True, box=None):
		'''
		Calculates the total potential energy summed over configured terms
		Arguments:
		----------
			pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
			grad: bool - if True, also return per-atom forces (N, 3) array
			box:  None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kcal/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		h = self._topology_hash(pose)
		if self._cache is None or self._cache_hash != h:
			self._cache = self._compile(pose)
			self._cache_hash = h
		n = self._cache['n']
		E, F = 0.0, np.zeros((n, 3))
		for method_name, kwargs in self.terms:
			fn = getattr(self, method_name)
			if grad:
				e, f = fn(pose, cache=self._cache, grad=True, box=box, **kwargs)
				E += e; F += f
			else:
				E += fn(pose, cache=self._cache, grad=False, box=box, **kwargs)
		return (E, F) if grad else E
	def _prepare(self, pose):
		'''
		Compile and store topology + parameter arrays for the given pose
		Arguments:
		----------
			pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		Returns:
		--------
			None: cache and cache hash are stored on the instance
		'''
		self._cache = self._compile(pose)
		self._cache_hash = self._topology_hash(pose)
	@staticmethod
	def _topology_hash(pose):
		'''
		Deterministic hash of bond graph, atom records and AA assignments
		Arguments:
		----------
			pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		Returns:
		--------
			int: hash value used to detect cache invalidation
		'''
		bonds_key = tuple((int(k), tuple(sorted(int(j) for j in v)))
			for k, v in sorted(pose.data['Bonds'].items()))
		atoms_key = tuple((int(k), tuple(a))
			for k, a in sorted(pose.data['Atoms'].items()))
		aas = pose.data.get('Amino Acids')
		aas_key = None if aas is None else tuple(
			(int(k), info[0], info[1], tuple(info[2]))
			for k, info in sorted(aas.items()))
		return hash((bonds_key, atoms_key, aas_key))
	def _compile(self, pose):
		'''
		Build all topology and parameter arrays consumed by every term
		Arguments:
		----------
			pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		Returns:
		--------
			dict: cache of topology arrays, per-atom and per-pair parameters
		'''
		atoms = pose.data['Atoms']
		n = len(atoms)
		cache = {'n': n}
		idx = np.array([(int(k), int(j)) for k, vs in pose.data['Bonds'].items()
			for j in vs], dtype=np.int64).reshape(-1, 2)
		idx.sort(axis=1)
		pairs = np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
		cache['pairs'] = pairs
		flat = (np.concatenate([pairs, pairs[:, ::-1]])
			if len(pairs) else np.empty((0, 2), dtype=np.int64))
		nbrs = ({int(a): np.sort(flat[flat[:, 0] == a, 1])
			for a in np.unique(flat[:, 0])} if len(flat) else {})
		cache['nbrs'] = nbrs
		cache['triplets'] = np.array(
			[(int(i), j, int(k)) for j, ns in nbrs.items()
			for p, i in enumerate(ns) for k in ns[p+1:]],
			dtype=np.int64).reshape(-1, 3)
		cache['excl_13'] = np.array(
			[(int(i), int(k)) for j, ns in nbrs.items()
			for p, i in enumerate(ns) for k in ns[p+1:]],
			dtype=np.int64).reshape(-1, 2)
		quartets = np.array(
			[(int(i), int(j), int(k), int(l)) for j, k in pairs
			for i in nbrs[int(j)] if i != k
			for l in nbrs[int(k)] if l != j and l != i],
			dtype=np.int64).reshape(-1, 4)
		if len(quartets):
			rev = quartets[:, ::-1]
			swap = (quartets[:, 0] > rev[:, 0]) | (
				(quartets[:, 0] == rev[:, 0]) & (quartets[:, 1] > rev[:, 1]))
			quartets = np.where(swap[:, None], rev, quartets)
			quartets = np.unique(quartets, axis=0)
		cache['quartets'] = quartets
		excl_14 = np.array(
			[(int(i), int(l)) for j, k in pairs
			for i in nbrs[int(j)] if i != k
			for l in nbrs[int(k)] if l != j and l != i],
			dtype=np.int64).reshape(-1, 2)
		if len(excl_14):
			excl_14.sort(axis=1)
			excl_14 = np.unique(excl_14[excl_14[:, 0] != excl_14[:, 1]], axis=0)
		cache['excl_14'] = excl_14
		cache['impropers'] = np.array(
			[(int(ns[0]), int(j), int(ns[1]), int(ns[2]))
			for j, ns in nbrs.items() if len(ns) == 3],
			dtype=np.int64).reshape(-1, 4)
		Pb = self.Parameters['bonds']; df_b = Pb['default']
		bond_params = np.array([Pb.get(tuple(sorted((
			self._atomtype(atoms[int(i)]),
			self._atomtype(atoms[int(j)])))), df_b)
			for i, j in pairs], dtype=np.float64).reshape(-1, 4)
		cache['bond_Kb'] = bond_params[:, 0]
		cache['bond_De'] = bond_params[:, 1]
		cache['bond_a']  = bond_params[:, 2]
		cache['bond_r0'] = bond_params[:, 3]
		Pa = self.Parameters['angles']; df_a = Pa['default']
		triplets = cache['triplets']
		angle_params = np.array([Pa.get((
			min(self._atomtype(atoms[int(i)]), self._atomtype(atoms[int(k)])),
			self._atomtype(atoms[int(j)]),
			max(self._atomtype(atoms[int(i)]), self._atomtype(atoms[int(k)]))),
			df_a) for i, j, k in triplets], dtype=np.float64).reshape(-1, 4)
		cache['angle_K_theta'] = angle_params[:, 0]
		cache['angle_theta0']  = np.deg2rad(angle_params[:, 1])
		cache['ub_K_ub']       = angle_params[:, 2]
		cache['ub_s0']         = angle_params[:, 3]
		Pd = self.Parameters['dihedrals']; df_d = Pd['default']
		component_lists = []
		for i, j, k, l in cache['quartets']:
			t = (self._atomtype(atoms[int(i)]), self._atomtype(atoms[int(j)]),
				self._atomtype(atoms[int(k)]), self._atomtype(atoms[int(l)]))
			if t > t[::-1]: t = t[::-1]
			component_lists.append(Pd.get(t, df_d))
		counts = np.array([len(c) for c in component_lists], dtype=np.int64)
		flat_p = (np.array([row for cl in component_lists for row in cl],
			dtype=np.float64).reshape(-1, 3) if component_lists
			else np.empty((0, 3), dtype=np.float64))
		cache['dihedral_counts'] = counts
		cache['dihedral_q_idx']  = np.repeat(np.arange(len(counts)), counts)
		cache['dihedral_k_phi']  = flat_p[:, 0]
		cache['dihedral_n_mult'] = flat_p[:, 1]
		cache['dihedral_phi0']   = np.deg2rad(flat_p[:, 2])
		Pi = self.Parameters['impropers']; df_i = Pi['default']
		impropers = cache['impropers']
		keys = [(self._atomtype(atoms[int(j)]),
			*sorted([self._atomtype(atoms[int(i)]),
				self._atomtype(atoms[int(k)]),
				self._atomtype(atoms[int(l)])]))
			for i, j, k, l in impropers]
		imp_params = (np.array([Pi.get(key, df_i) for key in keys],
			dtype=np.float64).reshape(-1, 3) if keys
			else np.empty((0, 3), dtype=np.float64))
		cache['imp_k']    = imp_params[:, 0]
		cache['imp_n']    = imp_params[:, 1]
		cache['imp_psi0'] = np.deg2rad(imp_params[:, 2])
		Plj = self.Parameters['lennard_jones']; df_lj = Plj['default']
		sig = np.empty(n); eps = np.empty(n); alpha = np.empty(n)
		for i in range(n):
			v = Plj.get(self._atomtype(atoms[i]), Plj.get(atoms[i][1], df_lj))
			sig[i], eps[i], alpha[i] = v[0], v[1], v[2]
		cache['lj_sig']    = sig
		cache['lj_eps']    = eps
		cache['lj_alpha']  = alpha
		cache['lj_sigma']  = 0.5 * (sig[:, None] + sig[None, :])
		cache['lj_eps_ij'] = np.sqrt(eps[:, None] * eps[None, :])
		q = np.array([atoms[i][2] for i in range(n)], dtype=np.float64)
		cache['charges'] = q
		cache['qq']      = q[:, None] * q[None, :]
		excl = np.eye(n, dtype=bool)
		if len(pairs):
			excl[pairs[:, 0], pairs[:, 1]] = True
			excl[pairs[:, 1], pairs[:, 0]] = True
		if len(cache['excl_13']):
			excl[cache['excl_13'][:, 0], cache['excl_13'][:, 1]] = True
			excl[cache['excl_13'][:, 1], cache['excl_13'][:, 0]] = True
		scal14 = np.zeros((n, n), dtype=bool)
		if len(excl_14):
			scal14[excl_14[:, 0], excl_14[:, 1]] = True
			scal14[excl_14[:, 1], excl_14[:, 0]] = True
			scal14 &= ~excl
		upper = np.triu(np.ones((n, n), dtype=bool), k=1)
		cache['mask_far']    = (~excl) & (~scal14) & upper
		cache['mask_14']     = scal14 & upper
		f_lj   = self.Parameters['scaling_14']['f_lj']
		f_elec = self.Parameters['scaling_14']['f_elec']
		cache['weight_lj']   = np.where(excl, 0.0, np.where(scal14, f_lj,  1.0))
		cache['weight_elec'] = np.where(excl, 0.0, np.where(scal14, f_elec,1.0))
		cache['scal14_bool'] = scal14
		cache['excl_bool']   = excl
		Pcmap = self.Parameters['cmap']
		aas = pose.data.get('Amino Acids')
		phi_q, psi_q, codes = [], [], []
		if aas is not None and len(aas) >= 3:
			res_atoms = {ai: (info[0], info[1],
				{atoms[k][0]: k for k in info[2]})
				for ai, info in aas.items()}
			for r in sorted(res_atoms.keys())[1:-1]:
				aa_curr, ch_curr, names_curr = res_atoms[r]
				_, ch_prev, names_prev = res_atoms[r - 1]
				_, ch_next, names_next = res_atoms[r + 1]
				if ch_curr != ch_prev or ch_curr != ch_next: continue
				try:
					phi_q.append((names_prev['C'], names_curr['N'],
						names_curr['CA'], names_curr['C']))
					psi_q.append((names_curr['N'], names_curr['CA'],
						names_curr['C'], names_next['N']))
					codes.append(aa_curr)
				except KeyError: continue
		if phi_q:
			tables = []
			for c in codes:
				key = c.upper()
				if key not in Pcmap:
					raise KeyError(
						f"ForceField/CMAP: amino acid '{key}' missing "
						f"from parameters.json['cmap']. Add a 24x24 "
						f"per-aa grid for this residue type.")
				t = np.asarray(Pcmap[key], dtype=np.float64)
				if c != key:
					t = t[::-1, ::-1]
				tables.append(t)
			cache['cmap_phi_q']  = np.asarray(phi_q, dtype=np.int64)
			cache['cmap_psi_q']  = np.asarray(psi_q, dtype=np.int64)
			cache['cmap_tables'] = np.stack(tables)
		else:
			cache['cmap_phi_q']  = np.empty((0, 4), dtype=np.int64)
			cache['cmap_psi_q']  = np.empty((0, 4), dtype=np.int64)
			cache['cmap_tables'] = np.empty((0, 24, 24), dtype=np.float64)
		return cache
	def _wrap(self, dvec, box):
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
	def _atomtype(self, atom_record):
		'''
		Map an atom record to a parameter-lookup atom type string
		Arguments:
		----------
			atom_record: tuple - (name, element, charge, ...) per atom record
		Returns:
		--------
			str: atom name when in backbone set, otherwise the element symbol
		'''
		name, element = atom_record[0], atom_record[1]
		backbone = {'N', 'CA', 'C', 'O', 'H', 'HA', 'CB', 'HB'}
		return name if name in backbone else element
	def BondPotential(self, pose, cache, alg='harmonic', grad=True, box=None):
		'''
		Calculates the Bond stretching potential for all bonded atom pairs
		Arguments:
		----------
			pose:  Pose - molecule source protein, DNA, RNA, or Molecule pose
			cache: dict - precomputed topology + parameter from _compile()
			alg:   Str algorithm type either 'harmonic' or 'morse'
			grad:  bool - if True, also return per-atom forces (N, 3) array
			box:   None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kcal/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		n     = cache['n']
		pairs = cache['pairs']
		if len(pairs) == 0:
			return (0.0, np.zeros((n, 3))) if grad else 0.0
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		i_idx, j_idx = pairs[:, 0], pairs[:, 1]
		dvec = self._wrap(coords[i_idx] - coords[j_idx], box)
		r = np.linalg.norm(dvec, axis=1)
		Kb, De, a, r0 = (cache['bond_Kb'], cache['bond_De'],
			cache['bond_a'], cache['bond_r0'])
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
			raise ValueError(
				'Algorithm not supported, choose (harmonic or morse)')
		forces = np.zeros((n, 3), dtype=np.float64)
		fij = coef[:, None] * dvec
		np.add.at(forces, i_idx, fij)
		np.add.at(forces, j_idx, -fij)
		return energy, forces
	def AnglePotential(self, pose, cache, grad=True, box=None):
		'''
		Calculates the Harmonic Angle potential for every bonded triplet
		Arguments:
		----------
			pose:  Pose - molecule source protein, DNA, RNA, or Molecule pose
			cache: dict - precomputed topology + parameter from _compile()
			grad:  bool - if True, also return per-atom forces (N, 3) array
			box:   None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kcal/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		n        = cache['n']
		triplets = cache['triplets']
		if len(triplets) == 0:
			return (0.0, np.zeros((n, 3))) if grad else 0.0
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		i_idx, j_idx, k_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
		v1 = self._wrap(coords[i_idx] - coords[j_idx], box)
		v2 = self._wrap(coords[k_idx] - coords[j_idx], box)
		mag1 = np.linalg.norm(v1, axis=1)
		mag2 = np.linalg.norm(v2, axis=1)
		cos = np.einsum('ij,ij->i', v1, v2) / (mag1 * mag2)
		cos = np.clip(cos, -1.0, 1.0)
		theta = np.arccos(cos)
		K_theta = cache['angle_K_theta']
		theta0  = cache['angle_theta0']
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
	def LJPotential(self, pose, cache, alg='12-6', grad=True, box=None):
		'''
		Calculates the Lennard-Jones non-bonded potential for all atom pairs
		Arguments:
		----------
			pose:  Pose - molecule source protein, DNA, RNA, or Molecule pose
			cache: dict - precomputed topology + parameter from _compile()
			alg:   Str algorithm type either '12-6' or '9-6'
			grad:  bool - if True, also return per-atom forces (N, 3) array
			box:   None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kcal/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		n = cache['n']
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		sigma    = cache['lj_sigma']
		epsilon  = cache['lj_eps_ij']
		mask_far = cache['mask_far']
		mask14   = cache['mask_14']
		weight   = cache['weight_lj']
		dvec = self._wrap(coords[:, None, :] - coords[None, :, :], box)
		r = np.linalg.norm(dvec, axis=-1)
		np.fill_diagonal(r, 1.0)
		f_lj = self.Parameters['scaling_14']['f_lj']
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
		coef = -dU_dr / r * weight
		fij_per_pair = coef[:, :, None] * dvec
		forces = np.sum(fij_per_pair, axis=1)
		return energy, forces
	def ElectrostaticPotential(self,pose,cache,alg='constant',grad=True,box=None):
		'''
		Calculates the Electrostatic non-bonded potential for all atom pairs
		Arguments:
		----------
			pose:  Pose - molecule source protein, DNA, RNA, or Molecule pose
			cache: dict - precomputed topology + parameter from _compile()
			alg:   Str algorithm type either 'constant' (uniform εr) or 'ddd'
				(distance-dependent dielectric, ε(r) = εr·r)
			grad:  bool - if True, also return per-atom forces (N, 3) array
			box:   None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kcal/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		n        = cache['n']
		qq       = cache['qq']
		mask_far = cache['mask_far']
		mask_14  = cache['mask_14']
		weight   = cache['weight_elec']
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		dvec = self._wrap(coords[:, None, :] - coords[None, :, :], box)
		r = np.linalg.norm(dvec, axis=-1)
		np.fill_diagonal(r, 1.0)
		epsilon_r = self.Parameters['electrostatic']['epsilon_r']
		if alg == 'constant':
			elec = (332.06 * qq) / (epsilon_r * r)
			dU_dr = -elec / r
		elif alg == 'ddd':
			elec = (332.06 * qq) / (epsilon_r * r * r)
			dU_dr = -2.0 * elec / r
		else:
			raise ValueError(
				'Algorithm not supported, choose (constant or ddd)')
		f_elec = self.Parameters['scaling_14']['f_elec']
		energy = float(np.sum(elec[mask_far]) + f_elec * np.sum(elec[mask_14]))
		if not grad: return energy
		coef = -dU_dr / r * weight
		fij_per_pair = coef[:, :, None] * dvec
		forces = np.sum(fij_per_pair, axis=1)
		return energy, forces
	def DihedralPotential(self, pose, cache, grad=True, box=None):
		'''
		Calculates the Proper Dihedral (torsion) potential for i-j-k-l atoms
		Arguments:
		----------
			pose:  Pose - molecule source protein, DNA, RNA, or Molecule pose
			cache: dict - precomputed topology + parameter from _compile()
			grad:  bool - if True, also return per-atom forces (N, 3) array
			box:   None for no PBC; (3,)  for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kcal/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		n        = cache['n']
		quartets = cache['quartets']
		if len(quartets) == 0:
			return (0.0, np.zeros((n, 3))) if grad else 0.0
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		i_idx, j_idx, k_idx, l_idx = quartets.T
		b1 = self._wrap(coords[j_idx] - coords[i_idx], box)
		b2 = self._wrap(coords[k_idx] - coords[j_idx], box)
		b3 = self._wrap(coords[l_idx] - coords[k_idx], box)
		n1 = np.cross(b1, b2)
		n2 = np.cross(b2, b3)
		b2_mag = np.linalg.norm(b2, axis=1)
		b2n = b2 / b2_mag[:, None]
		phi = np.arctan2(
			np.einsum('ij,ij->i', np.cross(n1, b2n), n2),
			np.einsum('ij,ij->i', n1, n2))
		q_idx  = cache['dihedral_q_idx']
		k_phi  = cache['dihedral_k_phi']
		n_mult = cache['dihedral_n_mult']
		phi0   = cache['dihedral_phi0']
		phi_flat = phi[q_idx]
		energy = float(np.sum(k_phi * (1 + np.cos(n_mult * phi_flat - phi0))))
		if not grad: return energy
		dU_dphi_flat = -k_phi * n_mult * np.sin(n_mult * phi_flat - phi0)
		dU_dphi = np.zeros(len(quartets), dtype=np.float64)
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
	def ImproperPotential(self,pose,cache,alg='harmonic',grad=True,box=None):
		'''
		Calculates the total Improper Dihedral potential energy
		Arguments:
		----------
			pose:  Pose - molecule source protein, DNA, RNA, or Molecule pose
			cache: dict - precomputed topology + parameter from _compile()
			alg:   Str algorithm type either 'harmonic' or 'fourier'
			grad:  bool - if True, also return per-atom forces (N, 3) array
			box:   None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kcal/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		n         = cache['n']
		impropers = cache['impropers']
		if len(impropers) == 0:
			return (0.0, np.zeros((n, 3))) if grad else 0.0
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		i_idx, j_idx, k_idx, l_idx = impropers.T
		b1 = self._wrap(coords[j_idx] - coords[i_idx], box)
		b2 = self._wrap(coords[k_idx] - coords[j_idx], box)
		b3 = self._wrap(coords[l_idx] - coords[k_idx], box)
		n1 = np.cross(b1, b2)
		n2 = np.cross(b2, b3)
		b2_mag = np.linalg.norm(b2, axis=1)
		b2n = b2 / b2_mag[:, None]
		psi = np.arctan2(
			np.einsum('ij,ij->i', np.cross(n1, b2n), n2),
			np.einsum('ij,ij->i', n1, n2))
		k_imp  = cache['imp_k']
		n_mult = cache['imp_n']
		psi0   = cache['imp_psi0']
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
	def UBPotential(self, pose, cache, grad=True, box=None):
		'''
		Calculates Urey-Bradley 1-3 stretching potential between all three atoms
		Arguments:
		----------
			pose:  Pose - molecule source protein, DNA, RNA, or Molecule pose
			cache: dict - precomputed topology + parameter from _compile()
			grad:  bool - if True, also return per-atom forces (N, 3) array
			box:   None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kcal/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		n        = cache['n']
		triplets = cache['triplets']
		if len(triplets) == 0:
			return (0.0, np.zeros((n, 3))) if grad else 0.0
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		i_idx, j_idx, k_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
		dvec = self._wrap(coords[i_idx] - coords[k_idx], box)
		r = np.linalg.norm(dvec, axis=1)
		k_ub = cache['ub_K_ub']
		s0   = cache['ub_s0']
		energy = float(np.sum(k_ub * (r - s0)**2))
		if not grad: return energy
		forces = np.zeros((n, 3), dtype=np.float64)
		coef = -2.0 * k_ub * (r - s0) / r
		fik = coef[:, None] * dvec
		np.add.at(forces, i_idx, fik)
		np.add.at(forces, k_idx, -fik)
		return energy, forces
	def PolarisationPotential(self,pose,cache,alg='constant',grad=True,box=None):
		'''
		Calculates the induced-dipole polarisation potential for all atoms
		Arguments:
		----------
			pose:  Pose - molecule source protein, DNA, RNA, or Molecule pose
			cache: dict - precomputed topology + parameter from _compile()
			alg:   Str algorithm type either 'constant' or 'ddd'
			grad:  bool - if True, also return per-atom forces (N, 3) array
			box:   None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kcal/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		n      = cache['n']
		q      = cache['charges']
		alpha  = cache['lj_alpha']
		weight = cache['weight_elec']
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		dr = self._wrap(coords[:, None, :] - coords[None, :, :], box)
		r = np.linalg.norm(dr, axis=-1)
		np.fill_diagonal(r, 1.0)
		epsilon_r = self.Parameters['electrostatic']['epsilon_r']
		if alg == 'constant':
			coeff = 332.06 * q[None, :] / (epsilon_r * r**3)
		elif alg == 'ddd':
			coeff = 332.06 * q[None, :] / (epsilon_r * r**4)
		else:
			raise ValueError(
				'Algorithm not supported, choose (constant or ddd)')
		coeff = coeff * weight
		E = np.einsum('ij,ijk->ik', coeff, dr)
		E_sq = np.sum(E**2, axis=1)
		energy = float(0.5 * np.sum(alpha * E_sq))
		if not grad: return energy
		p_pow = 3.0 if alg == 'constant' else 4.0
		rhat = dr / r[:, :, None]
		E_dot_rhat = np.einsum('ik,ijk->ij', E, rhat)
		G = E[:, None, :] - p_pow * E_dot_rhat[:, :, None] * rhat
		A = alpha[:, None] * coeff
		M = A[:, :, None] * G
		forces = -np.sum(M, axis=1) + np.sum(M, axis=0)
		return energy, forces
	def CMAPPotential(self, pose, cache, grad=True, box=None):
		'''
		Calculates the CMAP backbone (phi, psi) cross-term correction energy
		Arguments:
		----------
			pose:  Pose - molecule source protein, DNA, RNA, or Molecule pose
			cache: dict - precomputed topology + parameter from _compile()
			grad:  bool - if True, also return per-atom forces (N, 3) array
			box:   None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kcal/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		n      = cache['n']
		phi_q  = cache['cmap_phi_q']
		psi_q  = cache['cmap_psi_q']
		tables = cache['cmap_tables']
		if len(phi_q) == 0:
			return (0.0, np.zeros((n, 3))) if grad else 0.0
		M = len(phi_q)
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		quartets = np.concatenate([phi_q, psi_q], axis=0)
		i_idx, j_idx, k_idx, l_idx = quartets.T
		b1 = self._wrap(coords[j_idx] - coords[i_idx], box)
		b2 = self._wrap(coords[k_idx] - coords[j_idx], box)
		b3 = self._wrap(coords[l_idx] - coords[k_idx], box)
		n1 = np.cross(b1, b2)
		n2 = np.cross(b2, b3)
		b2_mag = np.linalg.norm(b2, axis=1)
		b2n = b2 / b2_mag[:, None]
		ang = np.arctan2(
			np.einsum('ij,ij->i', np.cross(n1, b2n), n2),
			np.einsum('ij,ij->i', n1, n2))
		phi, psi = ang[:M], ang[M:]
		N_grid = tables.shape[1]
		H = 2.0 * np.pi / N_grid
		x = (phi + np.pi) / H
		y = (psi + np.pi) / H
		gi = np.floor(x).astype(np.int64) % N_grid
		gj = np.floor(y).astype(np.int64) % N_grid
		u = x - np.floor(x)
		v = y - np.floor(y)
		off = np.array([-1, 0, 1, 2])
		a_grid = (gi[:, None, None] + off[None, :, None]) % N_grid
		b_grid = (gj[:, None, None] + off[None, None, :]) % N_grid
		stencil = tables[np.arange(M)[:, None, None], a_grid, b_grid]
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
		dU_dphi = dE_du / H
		dU_dpsi = dE_dv / H
		energy = float(np.sum(E_per))
		if not grad: return energy
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

class Score():
	'''
	Hybrid physics+statistical score for protein design (L/D/non-canonical)
	'''
	def __init__(self, ff=None, box=None):
		'''
		Initialise an 8-term protein-design score function
		Arguments:
		----------
			ff:  ForceField - reusable physics evaluator; created if None
			box: None for no PBC; (3,) ortho; (3, 3) triclinic
		Returns:
		--------
			None: instance is configured in place
		'''
		if ff is None: ff = ForceField()
		self.ff = ff
		self.box = box
		path = os.path.join(os.path.dirname(__file__), 'parameters.json')
		with open(path) as f: P = json.load(f)
		self.weights     = P['weights']
		self.ref_state   = P['ref_state']
		self.lk          = P['lk_solvation']
		self.hb          = P['hbond']
		self.kbp         = P['kbp']
		self.rot_sigma   = float(P['rotamer']['sigma_chi_deg'])
		self._kbp_table  = np.asarray(self.kbp['table'], dtype=np.float64)
		self._lk_types   = None
		self._kbp_types  = None
		self._lk_dG      = None
		self._lk_lam     = None
		self._lk_V       = None
		self._cache_hash = None
	def __call__(self, pose, decompose=False):
		'''
		Evaluate the design score; optionally return per-term breakdown
		Arguments:
		----------
			pose:      Pose - molecule source pose
			decompose: bool - if True, return (total, per_term_dict)
		Returns:
		--------
			float OR (float, dict): total score (and per-term values)
		'''
		h = self.ff._topology_hash(pose)
		if self.ff._cache is None or self.ff._cache_hash != h:
			self.ff._prepare(pose)
			self._cache_hash = None
		cache = self.ff._cache
		if self._cache_hash != h:
			self._build_typing(pose)
			self._cache_hash = h
		E_lj   = self.ff.LJPotential(pose, cache=cache, grad=False,
			box=self.box)
		E_elec = self.ff.ElectrostaticPotential(pose, cache=cache,
			grad=False, box=self.box)
		E_cmap = self.ff.CMAPPotential(pose, cache=cache, grad=False,
			box=self.box)
		E_lk   = self._lk_solvation(pose, cache)
		E_hb   = self._hbond_geom(pose, cache)
		E_rot  = self._rotamer_prior(pose)
		E_ref  = self._reference_state(pose)
		E_kbp  = self._kbp_score(pose, cache)
		w = self.weights
		total = (w['LJ']*E_lj + w['Electrostatic']*E_elec
			+ w['LK']*E_lk + w['Hbond']*E_hb
			+ w['CMAP']*E_cmap + w['Rotamer']*E_rot
			+ w['Reference']*E_ref + w['KBP']*E_kbp)
		if decompose:
			return float(total), {
				'LJ': float(E_lj), 'Electrostatic': float(E_elec),
				'LK': float(E_lk), 'Hbond': float(E_hb),
				'CMAP': float(E_cmap), 'Rotamer': float(E_rot),
				'Reference': float(E_ref), 'KBP': float(E_kbp)}
		return float(total)
	def _build_typing(self, pose):
		'''
		Build per-atom LK and KBP type arrays plus LK parameter vectors
		Arguments:
		----------
			pose: Pose - molecule source pose
		Returns:
		--------
			None: stores arrays on the instance
		'''
		atoms = pose.data['Atoms']
		sorted_ids = sorted(atoms)
		lk_map  = self.lk['atom_types']
		kbp_map = self.kbp['atom_types']
		lk_types_str = []
		kbp_idx = []
		dG_list, lam_list, V_list = [], [], []
		for i in sorted_ids:
			a = atoms[i]
			composite = f"{a[0]}-{a[1]}"
			if composite in lk_map: lk_key = composite
			elif a[1] in lk_map:    lk_key = a[1]
			else:
				raise KeyError(
					f"Score: atom type '{composite}' "
					f"(atom #{i}, name='{a[0]}', element='{a[1]}') "
					f"missing from "
					f"parameters.json['lk_solvation']['atom_types']. "
					f"Add an entry [dG_free, lambda, volume] for "
					f"this type or its element fallback.")
			vals = lk_map[lk_key]
			dG_list.append(vals[0]); lam_list.append(vals[1])
			V_list.append(vals[2])
			lk_types_str.append(lk_key)
			if composite in kbp_map: kbp_idx.append(kbp_map[composite])
			elif a[1] in kbp_map:    kbp_idx.append(kbp_map[a[1]])
			else:
				raise KeyError(
					f"Score: atom type '{composite}' "
					f"(atom #{i}, name='{a[0]}', element='{a[1]}') "
					f"missing from "
					f"parameters.json['kbp']['atom_types']. "
					f"Add an integer type index for this type or its "
					f"element fallback.")
		self._lk_dG  = np.asarray(dG_list,  dtype=np.float64)
		self._lk_lam = np.asarray(lam_list, dtype=np.float64)
		self._lk_V   = np.asarray(V_list,   dtype=np.float64)
		self._lk_types  = np.asarray(lk_types_str, dtype=object)
		self._kbp_types = np.asarray(kbp_idx, dtype=np.int64)
	def _reference_state(self, pose):
		'''
		Per-residue reference (unfolded baseline) energy summed over pose
		Arguments:
		----------
			pose: Pose - molecule source pose
		Returns:
		--------
			float: sum of ref_state[aa] over every residue, in kcal/mol
		'''
		aas = pose.data.get('Amino Acids')
		if aas is None: return 0.0
		d = self.ref_state
		total = 0.0
		for r, info in aas.items():
			key = info[0].upper()
			if key not in d:
				raise KeyError(
					f"Score: amino acid '{key}' "
					f"(residue #{r}) missing from "
					f"parameters.json['ref_state']. "
					f"Add a per-aa reference energy. Known canonical "
					f"codes: ACDEFGHIKLMNPQRSTVWY; non-canonical: "
					f"BJOUXZ.")
			total += d[key]
		return total
	def _rotamer_prior(self, pose):
		'''
		Gaussian rotamer prior centered at BBDEP-predicted mean chi
		Arguments:
		----------
			pose: Pose - molecule source pose
		Returns:
		--------
			float: 0.5 * sum (Delta_chi / sigma)^2 over interior chis
		'''
		aas = pose.data.get('Amino Acids')
		if aas is None: return 0.0
		inv_sigma_sq = 0.5 / (self.rot_sigma ** 2)
		total = 0.0
		for r, info in aas.items():
			c = info[0]
			aa = c.upper()
			db = pose.aminoacids.get(aa, {})
			chi_atoms = db.get('Chi Angle Atoms') or []
			if not chi_atoms: continue
			grids = db.get('BBDEP')
			if not grids:
				raise KeyError(
					f"Score: amino acid '{aa}' has "
					f"{len(chi_atoms)} chi angle(s) but no BBDEP "
					f"grids in pose.aminoacids. Database inconsistency.")
			n_chi = min(len(chi_atoms), len(grids) // 2)
			phi = pose.GetDihedral(r, 'PHI')
			psi = pose.GetDihedral(r, 'PSI')
			if math.isnan(phi) or math.isnan(psi): continue
			flip = (c != aa)
			phi_q = -phi if flip else phi
			psi_q = -psi if flip else psi
			fx = ((phi_q + 180.0) % 360.0) / 10.0
			fy = ((psi_q + 180.0) % 360.0) / 10.0
			i0 = int(math.floor(fx)) % 36
			j0 = int(math.floor(fy)) % 36
			i1 = (i0 + 1) % 36; j1 = (j0 + 1) % 36
			u = fx - math.floor(fx); v = fy - math.floor(fy)
			w = ((1-u)*(1-v), u*(1-v), (1-u)*v, u*v)
			for ci in range(n_chi):
				sg = grids[2*ci]; cg = grids[2*ci+1]
				s = 1e-4 * (w[0]*sg[i0,j0] + w[1]*sg[i1,j0]
					+ w[2]*sg[i0,j1] + w[3]*sg[i1,j1])
				cv = 1e-4 * (w[0]*cg[i0,j0] + w[1]*cg[i1,j0]
					+ w[2]*cg[i0,j1] + w[3]*cg[i1,j1])
				chi_pred = math.degrees(math.atan2(s, cv))
				if flip: chi_pred = -chi_pred
				chi_now = pose.GetDihedral(r, 'CHI', chi_type=ci+1)
				if math.isnan(chi_now): continue
				d = ((chi_now - chi_pred + 180.0) % 360.0) - 180.0
				total += inv_sigma_sq * d * d
		return total
	def _lk_solvation(self, pose, cache):
		'''
		Lazaridis-Karplus EEF1 implicit solvation summed over atom pairs
		Arguments:
		----------
			pose:  Pose - molecule source pose
			cache: dict - ff topology cache providing excl_bool, lj_sigma
		Returns:
		--------
			float: solvation free energy in kcal/mol
		'''
		dG  = self._lk_dG
		lam = self._lk_lam
		V   = self._lk_V
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		dvec = self.ff._wrap(
			coords[:, None, :] - coords[None, :, :], self.box)
		r = np.linalg.norm(dvec, axis=-1)
		np.fill_diagonal(r, 1.0)
		excl = cache['excl_bool']
		mask = (~excl) & (r < 9.0)
		R_min = 0.5 * cache['lj_sigma']
		with np.errstate(divide='ignore', invalid='ignore'):
			gauss = np.exp(-((r - R_min) / lam[:, None])**2)
			pre = (2.0 * V[None, :]) / (np.pi**1.5 * lam[:, None])
			E_ij = pre * (dG[:, None] / (r * r)) * gauss
		E_self = float(dG.sum())
		E_pair = float(np.sum(np.where(mask, E_ij, 0.0)))
		return E_self - E_pair
	def _hbond_geom(self, pose, cache):
		'''
		Geometric hydrogen-bond term over donor-H-acceptor-base quartets
		Arguments:
		----------
			pose:  Pose - molecule source pose
			cache: dict - ff topology cache providing nbrs adjacency
		Returns:
		--------
			float: H-bond energy contribution in kcal/mol
		'''
		atoms = pose.data['Atoms']
		nbrs = cache['nbrs']
		donors, acceptors = [], []
		for i, a in atoms.items():
			elem = a[1]
			ns = nbrs.get(i, [])
			if elem in ('N', 'O'):
				hs = [int(j) for j in ns if atoms[int(j)][1] == 'H']
				heavy = [int(j) for j in ns if atoms[int(j)][1] != 'H']
				for h in hs: donors.append((i, h))
				if heavy: acceptors.append((i, heavy[0]))
		if not donors or not acceptors: return 0.0
		D = np.asarray(donors,    dtype=np.int64)
		A = np.asarray(acceptors, dtype=np.int64)
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		dvec_HA = self.ff._wrap(
			coords[D[:, 1]][:, None, :] - coords[A[:, 0]][None, :, :],
			self.box)
		r_HA = np.linalg.norm(dvec_HA, axis=-1)
		mask_r = (r_HA > 1.4) & (r_HA < 3.0)
		dvec_HD = (coords[D[:, 0]] - coords[D[:, 1]])
		nrm_HD = np.linalg.norm(dvec_HD, axis=-1, keepdims=True)
		uHD = dvec_HD / np.maximum(nrm_HD, 1e-12)
		uHA = dvec_HA / np.maximum(r_HA[:, :, None], 1e-12)
		cos_DHA = np.einsum('ik,ijk->ij', uHD, -uHA)
		dvec_AB = (coords[A[:, 1]] - coords[A[:, 0]])
		nrm_AB = np.linalg.norm(dvec_AB, axis=-1, keepdims=True)
		uAB = dvec_AB / np.maximum(nrm_AB, 1e-12)
		cos_HAB = np.einsum('ijk,jk->ij', uHA, uAB)
		E_r = self.hb['well_depth'] * np.exp(
			-((r_HA - self.hb['r_opt'])**2) / (self.hb['r_sigma']**2))
		F_DHA = np.maximum(0.0, -cos_DHA)**2
		F_HAB = np.maximum(0.0,  cos_HAB)**2
		E = -E_r * F_DHA * F_HAB
		return float(np.sum(np.where(mask_r, E, 0.0)))
	def _kbp_score(self, pose, cache):
		'''
		Knowledge-based pair potential summed over far-pair atom pairs
		Arguments:
		----------
			pose:  Pose - molecule source pose
			cache: dict - ff topology cache providing mask_far
		Returns:
		--------
			float: KBP energy contribution in kcal/mol
		'''
		t = self._kbp_types
		table = self._kbp_table
		cutoff = float(self.kbp['cutoff'])
		bin_w  = float(self.kbp['bin_width'])
		N_bins = table.shape[2]
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		dvec = self.ff._wrap(coords[:, None, :] - coords[None, :, :], self.box)
		r = np.linalg.norm(dvec, axis=-1)
		mask = cache['mask_far'] & (r > 0.0) & (r < cutoff)
		I, J = np.where(mask)
		if len(I) == 0: return 0.0
		bins = np.minimum((r[I, J] / bin_w).astype(np.int64), N_bins - 1)
		E = table[t[I], t[J], bins]
		return float(E.sum())