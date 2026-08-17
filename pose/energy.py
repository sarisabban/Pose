#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
import json
import math
import copy
import base64
import warnings
import numpy as np
from . import tools
from .pose import DBLoad

class ForceField():
	'''
	Configurable molecular mechanics force field assembled from energy terms
	'''
	def __init__(self, name='Default', strict=False):
		'''
		Initialise the force field with a named parameter set from database.json
		Arguments:
		----------
			name:   str - key into database.json['Energy Parameters']
				(e.g. 'Default', 'OpenFF'); matched case-insensitively
				(e.g. 'default', 'OPENFF', 'oPeNfF' all resolve correctly).
				Selects both the SMIRKS-keyed parameter sections and the
				list of potential methods to evaluate (from `Terms`
				sub-key)
			strict: if True, raise RuntimeError on any SMIRKS coverage gap
				(unmatched bond/angle/torsion/improper centre/atom). If
				False (default), warn but continue with K=0 fall-through.
		Returns:
		--------
			None: instance is configured in-place
		'''
		self.strict = strict
		EP = DBLoad()['Energy Parameters']
		key_map = {k.upper(): k for k in EP}
		name_upper = name.upper()
		if name_upper not in key_map:
			raise ValueError(
				'ForceField: unknown name=%r (available: %r)'
				% (name, sorted(EP)))
		self.name = key_map[name_upper]
		ff_db = copy.deepcopy(EP[self.name])
		if 'Terms' not in ff_db:
			raise ValueError(
				"ForceField: '%s' is missing the 'Terms' key in database.json"
				% (name,))
		self.terms = [(t[0], dict(t[1])) for t in ff_db['Terms']]
		self.DEFAULT_TERMS = self.terms
		MOL_KEYS = ('Constraints', 'Bonds', 'Angles', 'UB',
			'ProperTorsions', 'ImproperTorsions', 'vdW', 'Electrostatic')
		ff = {k: ff_db[k] for k in MOL_KEYS if k in ff_db}
		if 'Electrostatic' in ff:
			ff['LibraryCharges'] = ff['Electrostatic']
		for k in ('improper_style', 'proper_precedence'):
			if k in ff_db: ff[k] = ff_db[k]
		for sm, par in ff.get('Bonds', {}).items():
			par['K_b'] = par['K_b'] * 0.5
		for sm, par in ff.get('Angles', {}).items():
			par['K_theta'] = par['K_theta'] * 0.5
		self.mol = ff if ff else None
		self.Parameters = ff_db
		self._cache = None
		self._cache_hash = None
		self._warned_poses = set()
		self._EPS = 1e-12
	def __call__(self, pose, grad=False, box=None, v=False):
		'''
		Calculates the total potential energy summed over configured terms
		Arguments:
		----------
			pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
			grad: bool - if True, also return per-atom forces (N, 3) array
			box:  None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
			v:    verbosity, if True will print error for missing SMIRKS
		Returns:
		--------
			float: potential energy in kJ/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		if len(pose.data.get('Atoms', {})) == 0:
			return (0.0, np.zeros((0, 3))) if grad else 0.0
		c = np.asarray(pose.data['Coordinates'], float)
		if not np.isfinite(c).all():
			bad = int(np.flatnonzero(~np.isfinite(c).all(1))[0])
			raise FloatingPointError(f'Non-finite coordinate at atom {bad}')
		self._repairbonds(pose)
		bonds_key = tuple((int(k), tuple(sorted(int(j) for j in v)))
			for k, v in sorted(pose.data['Bonds'].items()))
		atoms_key = tuple((int(k), tuple(a))
			for k, a in sorted(pose.data['Atoms'].items()))
		aas = pose.data.get('Amino Acids')
		aas_key = None if aas is None else tuple(
			(int(k), info[0], info[1], tuple(info[2]))
			for k, info in sorted(aas.items()))
		h = hash((bonds_key, atoms_key, aas_key))
		if self._cache is None or self._cache_hash != h:
			self._cache = self._buildcache(pose, v)
			self._cache_hash = h
		n = self._cache['n']
		E, F = 0.0, np.zeros((n, 3))
		with np.errstate(over='ignore', invalid='ignore',
			divide='ignore'):
			for method_name, kwargs in self.terms:
				fn = getattr(self, method_name)
				if grad:
					e, f = fn(pose, cache=self._cache, grad=True,
						box=box, **kwargs)
					E += e; F += f
				else:
					E += fn(pose, cache=self._cache, grad=False,
						box=box, **kwargs)
		return (E, F) if grad else E
	def _buildcache(self, pose, v):
		'''
		Build the topology and parameter cache for a pose
		Arguments:
		----------
			pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
			v:    bool - verbosity, if True print unmatched SMIRKS warnings
		Returns:
		--------
			dict: the cache consumed by every potential method
		'''
		atoms = pose.data['Atoms']
		n = len(atoms)
		cache = {'n': n}
		idx = np.array([(int(k), int(j))
			for k, vs in pose.data['Bonds'].items()
			for j in vs], dtype=np.int64).reshape(-1, 2)
		idx.sort(axis=1)
		pairs = (np.unique(idx[idx[:, 0] != idx[:, 1]], axis=0)
			if len(idx) else np.empty((0, 2), dtype=np.int64))
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
				(quartets[:,0] == rev[:,0]) & (quartets[:,1] > rev[:,1]))
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
			excl_14 = np.unique(
				excl_14[excl_14[:, 0] != excl_14[:, 1]], axis=0)
		cache['excl_14'] = excl_14
		with warnings.catch_warnings():
			warnings.simplefilter('always' if v else 'ignore')
			assigns = tools.SMIRKSMatch(pose, self.mol)
		atoms_set = set(pose.data['Atoms'].keys())
		bonds_dict = pose.data['Bonds']
		nbr_local = {i: [j for j in bonds_dict.get(i, [])
			if j in atoms_set and j != i] for i in atoms_set}
		atoms = pose.data['Atoms']
		nms = {i: '%s/%s[%s]' % (atoms[i][0], atoms[i][1],
			''.join(sorted(atoms[j][1] for j in nbr_local[i])))
			for i in atoms_set}
		gaps = []
		for i in atoms_set:
			for j in nbr_local[i]:
				if i >= j: continue
				if (int(i), int(j)) not in assigns['bonds']:
					gaps.append(f'bond {nms[i]}-{nms[j]}')
		matched_angles = {(min(t[0], t[2]), t[1], max(t[0], t[2]))
			for t in assigns['angles']}
		for j in atoms_set:
			ns = nbr_local[j]
			for x in range(len(ns)):
				for y in range(x + 1, len(ns)):
					i, k = ns[x], ns[y]
					tup = (min(i, k), j, max(i, k))
					if tup not in matched_angles:
						gaps.append(f'angle {nms[i]}-{nms[j]}-{nms[k]}')
		matched_propers = set()
		for tup in assigns['propers']:
			ti, tj, tk, tl = tup
			matched_propers.add((ti, tj, tk, tl) if tj < tk
				else (tl, tk, tj, ti))
		for i in atoms_set:
			for j in nbr_local[i]:
				if i >= j: continue
				for x in nbr_local[i]:
					if x == j: continue
					for y in nbr_local[j]:
						if y == i or y == x: continue
						quad = (x, i, j, y) if i < j else (y, j, i, x)
						if quad not in matched_propers:
							gaps.append(f'torsion {nms[x]}-{nms[i]}-{nms[j]}-{nms[y]}')
		if self.Parameters.get('improper_style',
			'smirnoff') == 'smirnoff':
			matched_centres = {tup[0]
				for tup in assigns['impropers']}
			matched_centres |= {tup[2]
				for tup in assigns['impropers']}
			for c in atoms_set:
				if (len(nbr_local[c]) == 3
					and c not in matched_centres):
					gaps.append(f'improper centre {nms[c]}')
		for i in atoms_set:
			if assigns['vdw'].get(i) is None:
				gaps.append(f'vdW atom {nms[i]}')
		if gaps:
			n_h = sum(1 for i in atoms_set if atoms[i][1] == 'H')
			if n_h == 0:
				msg=(f'Force field is missing ~{len(gaps)} H bonded terms. '
					f'Call ReBuild() after Import() to add hydrogens.')
			else:
				first_few = ', '.join(gaps[:5])
				msg = (f'{len(gaps)} internal coordinate(s) not covered '
					f'by the SMIRKS database; energy includes K=0 for '
					f'these terms. First few: {first_few}.')
			if self.strict:
				raise RuntimeError(msg)
			if id(pose) not in self._warned_poses:
				if v: print(msg)
				self._warned_poses.add(id(pose))
		constraints = assigns.get('constraints', set())
		bond_Kb = np.zeros(len(pairs)); bond_r0 = np.zeros(len(pairs))
		for p, (a, b) in enumerate(pairs):
			par = assigns['bonds'].get((int(a), int(b)))
			if par is not None:
				bond_r0[p] = par[0]
				if (int(a), int(b)) not in constraints:
					bond_Kb[p] = par[1]
		cache['bond_Kb'] = bond_Kb
		cache['bond_r0'] = bond_r0
		cache['bond_De'] = np.zeros(len(pairs))
		cache['bond_a']  = np.zeros(len(pairs))
		triplets = cache['triplets']
		angle_Kt = np.zeros(len(triplets))
		angle_t0 = np.zeros(len(triplets))
		for p, (i, j, k) in enumerate(triplets):
			ii = (int(i), int(j), int(k))
			canon = (min(ii[0], ii[2]), ii[1], max(ii[0], ii[2]))
			par = assigns['angles'].get(canon)
			if par is not None:
				angle_t0[p] = par[0]
				if (canon[0], canon[2]) not in constraints:
					angle_Kt[p] = par[1]
		cache['angle_K_theta'] = angle_Kt
		cache['angle_theta0']  = np.deg2rad(angle_t0)
		ub_assigns = assigns.get('ub', {})
		ub_K_ub = np.zeros(len(triplets))
		ub_s0   = np.zeros(len(triplets))
		for p, (i, j, k) in enumerate(triplets):
			canon = (min(int(i), int(k)), int(j), max(int(i), int(k)))
			par = ub_assigns.get(canon)
			if par is not None:
				ub_s0[p]   = par[0]
				ub_K_ub[p] = par[1]
		cache['ub_K_ub'] = ub_K_ub
		cache['ub_s0']   = ub_s0
		comp_lists = []
		for q in cache['quartets']:
			i, j, k, l = (int(q[0]), int(q[1]), int(q[2]), int(q[3]))
			canon = (i, j, k, l) if (i, j, k, l) <= (l, k, j, i) \
				else (l, k, j, i)
			comp = assigns['propers'].get(canon)
			comp_lists.append(comp if comp is not None
				else [[1, 0.0, 0.0, 1.0]])
		counts = np.array([len(c) for c in comp_lists], dtype=np.int64)
		flat_p = (np.array([row for cl in comp_lists for row in cl],
			dtype=np.float64).reshape(-1, 4) if comp_lists
			else np.empty((0, 4), dtype=np.float64))
		cache['dihedral_counts'] = counts
		cache['dihedral_q_idx']  = np.repeat(np.arange(len(counts)), counts)
		cache['dihedral_k_phi']  = flat_p[:, 2] if len(flat_p) \
			else np.zeros(0)
		cache['dihedral_n_mult'] = flat_p[:, 0] if len(flat_p) \
			else np.zeros(0)
		cache['dihedral_phi0']   = (np.deg2rad(flat_p[:, 1])
			if len(flat_p) else np.zeros(0))
		cache['dihedral_idivf']  = (flat_p[:, 3] if len(flat_p)
			else np.ones(0))
		imps = assigns['impropers']
		imp_arr = (np.array([(t[0], t[1], t[2], t[3]) for t in imps],
			dtype=np.int64).reshape(-1, 4) if imps
			else np.empty((0, 4), dtype=np.int64))
		cache['impropers'] = imp_arr
		cache['imp_k']    = np.array([t[6] for t in imps],
			dtype=np.float64) if imps else np.zeros(0)
		cache['imp_n']    = np.array([t[4] for t in imps],
			dtype=np.float64) if imps else np.zeros(0)
		cache['imp_psi0'] = (np.deg2rad(np.array([t[5] for t in imps],
			dtype=np.float64)) if imps else np.zeros(0))
		sig = np.zeros(n); eps = np.zeros(n)
		for i in range(n):
			par = assigns['vdw'].get(i)
			if par is not None:
				eps[i], sig[i] = par[0], par[1]
		cache['lj_sig']    = sig
		cache['lj_eps']    = eps
		pol_assigns = assigns.get('polarisation', {})
		alpha = np.zeros(n)
		for i in range(n):
			a = pol_assigns.get(i)
			if a is not None:
				alpha[i] = a
		cache['lj_alpha']  = alpha
		cache['lj_sigma']  = 0.5 * (sig[:, None] + sig[None, :])
		cache['lj_eps_ij'] = np.sqrt(eps[:, None] * eps[None, :])
		try: nagl_q = self.NAGLCharges(pose)
		except Exception: nagl_q = None
		q = np.zeros(n, dtype=np.float64)
		used_nagl = False
		n_fallback = 0
		fallback_mask = np.zeros(n, dtype=bool)
		for i in range(n):
			if assigns['charges'][i] is not None:
				q[i] = assigns['charges'][i]
			elif nagl_q is not None and i < len(nagl_q):
				q[i] = nagl_q[i]; used_nagl = True
			else:
				q[i] = atoms[i][2]
				fallback_mask[i] = True
				n_fallback += 1
		if not used_nagl and n_fallback > 0:
			fc_dict = getattr(pose, '_formal_charges', {}) or {}
			Q = float(sum(int(fc_dict.get(i, 0)) for i in atoms))
			shift = (Q - float(q.sum())) / n_fallback
			q[fallback_mask] += shift
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
		f_lj   = self.Parameters['Constants']['f_lj']
		f_elec = self.Parameters['Constants']['f_elec']
		cache['weight_lj']   = np.where(excl, 0.0,
			np.where(scal14, f_lj,  1.0))
		cache['weight_elec'] = np.where(excl, 0.0,
			np.where(scal14, f_elec, 1.0))
		cache['scal14_bool'] = scal14
		cache['excl_bool']   = excl
		vdw14 = assigns.get('vdw14', {})
		if vdw14:
			sig14 = sig.copy(); eps14 = eps.copy()
			for i in range(n):
				p14 = vdw14.get(i)
				if p14 is not None:
					eps14[i], sig14[i] = p14[0], p14[1]
			ls14 = 0.5 * (sig14[:, None] + sig14[None, :])
			le14 = np.sqrt(eps14[:, None] * eps14[None, :])
			cache['lj_sigma']  = np.where(scal14, ls14,
				cache['lj_sigma'])
			cache['lj_eps_ij'] = np.where(scal14, le14,
				cache['lj_eps_ij'])
		cache['cmap_phi_q']  = np.empty((0, 4), dtype=np.int64)
		cache['cmap_psi_q']  = np.empty((0, 4), dtype=np.int64)
		cache['cmap_tables'] = np.empty((0, 24, 24), dtype=np.float64)
		if pose.data.get('Type') == 'Protein':
			aas = pose.data.get('Amino Acids', {}) or {}
			cmap_section = self.Parameters.get('CMAP', {}) or {}
			bb_per_res = {}
			for ri, rec in aas.items():
				code, chain, bb = rec[0], rec[1], rec[2]
				name_to_idx = {atoms[idx][0]: idx for idx in bb
					if idx in atoms}
				if not all(nm in name_to_idx for nm in ('N','CA','C')):
					continue
				bb_per_res[ri] = (chain, code,
					name_to_idx['N'], name_to_idx['CA'],
					name_to_idx['C'])
			res_order = sorted(bb_per_res.keys())
			bonds_g = pose.data.get('Bonds', {}) or {}
			c_of = {bb_per_res[r][4]: r for r in bb_per_res}
			n_of = {bb_per_res[r][2]: r for r in bb_per_res}
			phi_q_list = []; psi_q_list = []; grids = []
			for ri in res_order:
				Ni0 = bb_per_res[ri][2]; Ci0 = bb_per_res[ri][4]
				prev_ri = next((c_of[j] for j in bonds_g.get(Ni0, [])
					if j in c_of and c_of[j] != ri), None)
				next_ri = next((n_of[j] for j in bonds_g.get(Ci0, [])
					if j in n_of and n_of[j] != ri), None)
				if prev_ri is None or next_ri is None:
					continue
				chain  = bb_per_res[ri][0]
				if (bb_per_res[prev_ri][0] != chain or
					bb_per_res[next_ri][0] != chain):
					continue
				restri = assigns.get('restri', {})
				code1 = bb_per_res[ri][1]
				grid = (cmap_section.get(restri.get(ri))
					or cmap_section.get(aas[ri][5])
					or cmap_section.get(code1)
					or cmap_section.get(code1.upper()))
				if grid is None: continue
				g = np.asarray(grid, dtype=np.float64)
				if g.shape != (24, 24): continue
				if code1.islower():
					g = np.roll(g[::-1, ::-1], (1, 1), axis=(0, 1))
				_, _, Ni, CAi, Ci = bb_per_res[ri]
				Cm1 = bb_per_res[prev_ri][4]
				Np1 = bb_per_res[next_ri][2]
				phi_q_list.append((Cm1, Ni, CAi, Ci))
				psi_q_list.append((Ni, CAi, Ci, Np1))
				grids.append(g)
			if phi_q_list:
				cache['cmap_phi_q']  = np.asarray(phi_q_list,
					dtype=np.int64)
				cache['cmap_psi_q']  = np.asarray(psi_q_list,
					dtype=np.int64)
				cache['cmap_tables'] = np.stack(grids)
		T  = cache['cmap_tables']
		Ng = T.shape[1]
		eye = np.eye(Ng)
		Am = 4.0 * eye + np.roll(eye, 1, 0) + np.roll(eye, -1, 0)
		Bm = np.roll(eye, -1, 0) - np.roll(eye, 1, 0)
		D  = (3.0 * Ng / (2.0 * np.pi)) * np.linalg.solve(Am, Bm)
		cache['cmap_d1']  = np.einsum('ab,mbc->mac', D, T)
		cache['cmap_d2']  = np.einsum('mab,cb->mac', T, D)
		cache['cmap_d12'] = np.einsum('mab,cb->mac',
			cache['cmap_d1'], D)
		return cache
	def _topologyhash(self, pose):
		'''
		Deterministic hash of bond graph, atom records and AA assignments
		Arguments:
		----------
			pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
		Returns:
		--------
			int: hash used by tools.py callers to detect cache invalidation
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
	def _repairbonds(self, pose):
		'''
		Complete an under-specified bond graph in place: bond every
		orphan hydrogen to its nearest atom and every disulfide SG-SG
		pair, using the (exact) imported coordinates. A no-op when the
		graph is already complete, so it never disturbs poses built from
		SDF files or from sequence.
		Arguments:
		----------
			pose: Pose - the pose whose data['Bonds'] may be incomplete
		Returns:
		--------
			int: number of bonds added (0 when nothing needed repair)
		'''
		atoms  = pose.data['Atoms']
		bonds  = pose.data['Bonds']
		orders = pose.data.setdefault('BondOrders', {})
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		ids = sorted(atoms.keys())
		deg = {i: len(bonds.get(i, [])) for i in ids}
		bondset = set()
		for i in ids:
			for j in bonds.get(i, []):
				bondset.add((min(i, j), max(i, j)))
		added = 0
		orphans = [i for i in ids
			if deg[i] == 0 and atoms[i][1] == 'H']
		for i in orphans:
			d = np.linalg.norm(coords - coords[i], axis=1)
			d[i] = 1e18
			j = int(np.argmin(d))
			if d[j] <= 1.3:
				self._addbond(bonds, orders, bondset, i, j)
				added += 1
		if pose.data.get('Type') == 'Protein':
			sg = [i for i in ids if atoms[i][0] == 'SG']
			for a in range(len(sg)):
				for b in range(a + 1, len(sg)):
					i, j = sg[a], sg[b]
					if (min(i, j), max(i, j)) in bondset: continue
					if np.linalg.norm(coords[i] - coords[j]) <= 2.5:
						self._addbond(bonds, orders, bondset, i, j)
						added += 1
		return added
	def _addbond(self, bonds, orders, bondset, i, j):
		'''
		Add a bond-order-1.0 edge between atoms i and j in place
		Arguments:
		----------
			bonds:   dict - atom index to neighbour list, edited in place
			orders:  dict - atom index to bond-order list, edited in place
			bondset: set - canonical (low, high) pairs, edited in place
			i: int - atom index
			j: int - atom index
		Returns:
		--------
			No return value, the three containers are edited in place
		'''
		bonds.setdefault(i, []).append(j)
		bonds.setdefault(j, []).append(i)
		orders.setdefault(i, []).append(1.0)
		orders.setdefault(j, []).append(1.0)
		bondset.add((min(i, j), max(i, j)))
	def _prepare(self, pose):
		'''
		Force a cache build for the given pose (used by tools.py)
		Arguments:
		----------
			pose: Pose - any pose
		Returns:
		--------
			None: side effect is self._cache + self._cache_hash populated
		'''
		self._cache = None
		self(pose, grad=False)
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
	def NAGLCharges(self, pose):
		'''
		NAGL AM1-BCC partial charges, NumPy reimplementation of the
		AM1-BCC graph-NN forward pass; weights load from
		database.json['Energy Parameters']['AM1BCC']
		Arguments:
		----------
			pose: Pose - molecule, protein, DNA, or RNA pose
		Returns:
		--------
			ndarray of length max(atom_id)+1: per-atom partial charges in
				elementary charge units, summing to the molecule's total
				formal charge. Bit-equivalent to NAGL float32 inference.
		'''
		nagl = (DBLoad()['Energy Parameters']
			.get(self.name, {}).get('AM1BCC') or {})
		if 'gcn_layers' not in nagl or 'readout' not in nagl:
			raise RuntimeError(
				'AM1BCC weights missing from database.json. '
				'Run database_nagl_extract.py to install them.')
		atoms = pose.data['Atoms']
		bonds = pose.data['Bonds']
		sorted_ids = sorted(atoms.keys())
		n = len(sorted_ids)
		if n == 0:
			return np.zeros(1, dtype=np.float64)
		nbr = {i: [] for i in sorted_ids}
		for i in sorted_ids:
			for j in bonds.get(i, []):
				if j in atoms and j != i and j not in nbr[i]:
					nbr[i].append(j)
		table = getattr(self, 'naglutable', None)
		if table is None:
			table = {}
			for e in (nagl.get('lookup') or []):
				g = self._parsemapped(e['smiles'])
				if g is None: continue
				table.setdefault((tuple(sorted(g[0])), sum(g[1])),
					[]).append((g[0], g[1], g[2], e['q']))
			self.naglutable = table
		fcs = getattr(pose, '_formal_charges', {}) or {}
		q_el = [atoms[i][1] for i in sorted_ids]
		q_ch = [int(fcs.get(i, 0)) for i in sorted_ids]
		pos = {i: k for k, i in enumerate(sorted_ids)}
		bords = pose.data.get('BondOrders', {}) or {}
		q_adj = {}
		for i in sorted_ids:
			ns = list(bonds.get(i, []))
			os_ = list(bords.get(i, []))
			if len(os_) != len(ns): os_ = [1.0] * len(ns)
			q_adj[pos[i]] = {pos[j]: float(o) for j, o in zip(ns, os_)
				if j in atoms and j != i}
		for te, tc, ta, tq in table.get(
				(tuple(sorted(q_el)), sum(q_ch)), []):
			mp = self._graphmatch(q_el, q_ch, q_adj, te, tc, ta)
			if mp is None: continue
			out = np.zeros(max(sorted_ids) + 1, dtype=np.float64)
			for k, i in enumerate(sorted_ids): out[i] = float(tq[mp[k]])
			return out
		rings = self._findrings(nbr, sorted_ids)
		in_ring_sizes = {i: set() for i in sorted_ids}
		for r in rings:
			for a in r: in_ring_sizes[a].add(len(r))
		ELEM_IDX = {'C':0,'O':1,'H':2,'N':3,'S':4,'F':5,
			'Br':6,'Cl':7,'I':8,'P':9}
		fc_dict = getattr(pose, '_formal_charges', {}) or {}
		h = np.zeros((n, 22), dtype=np.float32)
		for k, i in enumerate(sorted_ids):
			elem = atoms[i][1]
			if elem in ELEM_IDX:
				h[k, ELEM_IDX[elem]] = 1.0
			deg = len(nbr[i])
			if 0 <= deg <= 6:
				h[k, 10 + deg] = 1.0
			h[k, 17] = float(fc_dict.get(i, 0))
			rs = in_ring_sizes[i]
			if 3 in rs: h[k, 18] = 1.0
			if 4 in rs: h[k, 19] = 1.0
			if 5 in rs: h[k, 20] = 1.0
			if 6 in rs: h[k, 21] = 1.0
		idx_of = {i: k for k, i in enumerate(sorted_ids)}
		A_mean = np.zeros((n, n), dtype=np.float32)
		for i in sorted_ids:
			ki = idx_of[i]; deg = len(nbr[i])
			if deg == 0: continue
			inv = 1.0 / float(deg)
			for j in nbr[i]:
				A_mean[ki, idx_of[j]] = inv
		with np.errstate(over='ignore', under='ignore', divide='ignore',
				invalid='ignore'):
			for layer in nagl['gcn_layers']:
				W_neigh = self._loadtensor(layer['fc_neigh_w'])
				W_self  = self._loadtensor(layer['fc_self_w'])
				b_self  = self._loadtensor(layer['fc_self_b'])
				h_avg        = A_mean @ h
				h_self_proj  = h @ W_self.T + b_self
				h_neigh_proj = h_avg @ W_neigh.T
				h = h_self_proj + h_neigh_proj
				np.maximum(h, 0, out=h)
		W0 = self._loadtensor(nagl['readout']['linear_0_w'])
		b0 = self._loadtensor(nagl['readout']['linear_0_b'])
		W1 = self._loadtensor(nagl['readout']['linear_1_w'])
		b1 = self._loadtensor(nagl['readout']['linear_1_b'])
		with np.errstate(over='ignore', under='ignore', divide='ignore',
				invalid='ignore'):
			z = h @ W0.T + b0
			z = 1.0 / (1.0 + np.exp(-z))
			pred = z @ W1.T + b1
		q_prior = pred[:, 0].astype(np.float64)
		chi     = pred[:, 1].astype(np.float64)
		eta     = pred[:, 2].astype(np.float64)
		Q_total = float(sum(int(fc_dict.get(i, 0)) for i in sorted_ids))
		s     = 1.0 / eta
		chi_s = chi * s
		phi   = float(q_prior.sum()) - Q_total - float(chi_s.sum())
		denom = float(s.sum())
		if abs(denom) < 1e-12: denom = 1e-12
		frac    = s * (phi / denom)
		q_final = q_prior - chi_s - frac
		out = np.zeros(max(sorted_ids) + 1, dtype=np.float64)
		for k, i in enumerate(sorted_ids):
			out[i] = float(q_final[k])
		return out
	def _parsemapped(self, smi):
		'''
		Parse an all-bracketed, atom-mapped SMILES into a graph
		Arguments:
		----------
			smi: str - mapped SMILES, every atom bracketed and tagged
				with an atom map number
		Returns:
		--------
			tuple: (elements, formal charges, {i: {j: bond order}}),
				each indexed by the atom map number minus one, or
				None when the string is outside the supported subset
		'''
		bord = {'-': 1.0, '=': 2.0, '#': 3.0}
		toks = re.findall(r'\[[^\]]*\]|[()\-=#]|\d', smi)
		el = {}; ch = {}; adj = {}; stack = []; ring = {}
		prev = None; order = 1.0
		for t in toks:
			if t == '(': stack.append(prev); continue
			if t == ')': prev = stack.pop(); continue
			if t in bord: order = bord[t]; continue
			if t.isdigit():
				if t not in ring: ring[t] = (prev, order)
				else:
					a, o = ring.pop(t)
					adj.setdefault(a, {})[prev] = o
					adj.setdefault(prev, {})[a] = o
				order = 1.0; continue
			m = re.match(
				r'\[([A-Z][a-z]?)((?:[+-]\d*)?)[^\]:]*:(\d+)\]', t)
			if m is None: return None
			k = int(m.group(3)) - 1; sgn = m.group(2)
			el[k] = m.group(1)
			ch[k] = 0 if not sgn else (
				int(sgn[1:]) if len(sgn) > 1 else 1) * (
				1 if sgn[0] == '+' else -1)
			adj.setdefault(k, {})
			if prev is not None:
				adj[k][prev] = order; adj[prev][k] = order
			order = 1.0; prev = k
		m = len(el)
		if sorted(el) != list(range(m)): return None
		return ([el[i] for i in range(m)], [ch[i] for i in range(m)],
			{i: adj.get(i, {}) for i in range(m)})
	def _graphmatch(self, qe, qc, qa, te, tc, ta):
		'''
		Find a graph isomorphism between a query and a table entry
		Arguments:
		----------
			qe, qc, qa: query elements, formal charges, adjacency
			te, tc, ta: entry elements, formal charges, adjacency
		Returns:
		--------
			dict: query atom index to entry atom index, or None when
				the two graphs are not isomorphic
		'''
		m = len(qe)
		if m != len(te) or sorted(qe) != sorted(te): return None
		if sorted(qc) != sorted(tc): return None
		qs = [(qe[i], qc[i], tuple(sorted(qa[i].values())))
			for i in range(m)]
		ts = [(te[j], tc[j], tuple(sorted(ta[j].values())))
			for j in range(m)]
		cand = {i: [j for j in range(m) if qs[i] == ts[j]]
			for i in range(m)}
		if any(not v for v in cand.values()): return None
		rank = sorted((len(cand[i]), -len(qa[i]), i) for i in range(m))
		seq = [i for _, _, i in rank]
		mp = {}; used = set()
		if not self._walk(0, m, seq, cand, qa, ta, mp, used):
			return None
		for i in range(m):
			if {(mp[k], v) for k, v in qa[i].items()} != set(
					ta[mp[i]].items()): return None
		return mp
	def _walk(self, k, m, seq, cand, qa, ta, mp, used):
		'''
		Backtracking search over the candidate atom assignments
		Arguments:
		----------
			k:    int - position in the search order seq
			m:    int - number of atoms in the query graph
			seq:  list - query atom indices in search order
			cand: dict - query atom index to allowed entry indices
			qa:   dict - query adjacency, atom index to neighbour orders
			ta:   dict - entry adjacency, atom index to neighbour orders
			mp:   dict - partial mapping so far, edited in place
			used: set - entry indices already taken, edited in place
		Returns:
		--------
			bool: True when a complete consistent mapping was reached
		'''
		if k == m: return True
		i = seq[k]
		for j in cand[i]:
			clash = [1 for nb, o in qa[i].items()
				if nb in mp and ta[j].get(mp[nb]) != o]
			if j in used or clash: continue
			mp[i] = j; used.add(j)
			if self._walk(k + 1, m, seq, cand, qa, ta, mp, used):
				return True
			del mp[i]; used.discard(j)
		return False
	def _findrings(self, nbr, sorted_ids):
		'''
		SSSR via shortest cycle per edge
		Arguments:
		----------
			No arguments taken (closes over nbr, sorted_ids)
		Returns:
		--------
			list: each ring as a tuple of atom indices
		'''
		edges = sorted({(min(i, j), max(i, j))
			for i in sorted_ids for j in nbr[i]})
		seen = set(); out = []
		for u, v in edges:
			parent = {u: None}; q = [u]
			while q:
				nq = []
				for x in q:
					for y in nbr[x]:
						if (min(x, y), max(x, y)) == (u, v): continue
						if y in parent: continue
						parent[y] = x
						if y == v: q = []; break
						nq.append(y)
					if not q: break
				q = nq
			if v not in parent: continue
			path = [v]; cur = v
			while parent[cur] is not None:
				cur = parent[cur]; path.append(cur)
			ring = tuple(path)
			mn = min(ring); i0 = ring.index(mn)
			rotated = ring[i0:] + ring[:i0]
			canon = min(rotated, (rotated[0],) + rotated[:0:-1])
			if canon in seen: continue
			seen.add(canon); out.append(canon)
		return out
	def _loadtensor(self, d):
		'''
		Decode a base64-encoded float32 tensor from database.json
		Arguments:
		----------
			d: dict with keys 'shape' (list of ints) and 'data' (base64 str)
		Returns:
		--------
			ndarray of dtype float32 with the requested shape
		'''
		raw = base64.b64decode(d['data'])
		return np.frombuffer(raw, dtype=np.float32).reshape(d['shape'])
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
			float: potential energy in kJ/mol  (when grad=False)
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
			coef = -2.0 * Kb * dr / np.maximum(r, self._EPS)
		elif alg.upper() == 'MORSE':
			dr = r - r0
			e_decay = np.exp(-a * dr)
			energy = float(np.sum(De * (1 - e_decay)**2))
			if not grad: return energy
			coef = -2.0 * De * (1 - e_decay) * a * e_decay \
				/ np.maximum(r, self._EPS)
		else:
			raise ValueError(
				"BondPotential: unknown alg=%r (allowed: 'harmonic', 'morse')"
				% (alg,))
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
			float: potential energy in kJ/mol  (when grad=False)
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
		mag1 = np.maximum(np.linalg.norm(v1, axis=1), self._EPS)
		mag2 = np.maximum(np.linalg.norm(v2, axis=1), self._EPS)
		cos = np.einsum('ij,ij->i', v1, v2) / (mag1 * mag2)
		cos = np.clip(cos, -1.0, 1.0)
		theta = np.arccos(cos)
		K_theta = cache['angle_K_theta']
		theta0  = cache['angle_theta0']
		energy = float(np.sum(K_theta * (theta - theta0)**2))
		if not grad: return energy
		forces = np.zeros((n, 3), dtype=np.float64)
		dU_dth = 2.0 * K_theta * (theta - theta0)
		sin_th = np.sqrt(np.clip(1.0 - cos**2, self._EPS, None))
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
	def VDWPotential(self, pose, cache, alg='12-6', grad=True, box=None):
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
			float: potential energy in kJ/mol  (when grad=False)
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
		r = np.maximum(r, self._EPS)
		f_lj = self.Parameters['Constants']['f_lj']
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
			raise ValueError(
				"VDWPotential: unknown alg=%r (allowed: '12-6', '9-6')"
				% (alg,))
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
			float: potential energy in kJ/mol  (when grad=False)
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
		r = np.maximum(r, self._EPS)
		epsilon_r = self.Parameters['Constants']['epsilon_r']
		if alg == 'constant':
			elec = (1389.35458 * qq) / (epsilon_r * r)
			dU_dr = -elec / r
		elif alg == 'ddd':
			elec = (1389.35458 * qq) / (epsilon_r * r * r)
			dU_dr = -2.0 * elec / r
		else:
			raise ValueError(
				"ElectrostaticPotential: unknown alg=%r "
				"(allowed: 'constant', 'ddd')" % (alg,))
		f_elec = self.Parameters['Constants']['f_elec']
		energy = float(np.sum(elec[mask_far]) + f_elec * np.sum(elec[mask_14]))
		if not grad: return energy
		coef = -dU_dr / r * weight
		fij_per_pair = coef[:, :, None] * dvec
		forces = np.sum(fij_per_pair, axis=1)
		return energy, forces
	def ProperTorsionPotential(self, pose, cache, grad=True, box=None):
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
			float: potential energy in kJ/mol  (when grad=False)
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
		b2_mag = np.maximum(np.linalg.norm(b2, axis=1), self._EPS)
		b2n = b2 / b2_mag[:, None]
		phi = np.arctan2(
			np.einsum('ij,ij->i', np.cross(n1, b2n), n2),
			np.einsum('ij,ij->i', n1, n2))
		q_idx  = cache['dihedral_q_idx']
		k_phi  = cache['dihedral_k_phi']
		n_mult = cache['dihedral_n_mult']
		phi0   = cache['dihedral_phi0']
		idivf  = cache.get('dihedral_idivf')
		k_eff  = k_phi / idivf if idivf is not None else k_phi
		phi_flat = phi[q_idx]
		energy = float(np.sum(k_eff * (1 + np.cos(n_mult * phi_flat - phi0))))
		if not grad: return energy
		dU_dphi_flat = -k_eff * n_mult * np.sin(n_mult * phi_flat - phi0)
		dU_dphi = np.zeros(len(quartets), dtype=np.float64)
		np.add.at(dU_dphi, q_idx, dU_dphi_flat)
		forces = np.zeros((n, 3), dtype=np.float64)
		n1_sq = np.maximum(np.einsum('ij,ij->i', n1, n1), self._EPS)
		n2_sq = np.maximum(np.einsum('ij,ij->i', n2, n2), self._EPS)
		Fi = -(dU_dphi * b2_mag / n1_sq)[:, None] * n1
		Fl =  (dU_dphi * b2_mag / n2_sq)[:, None] * n2
		b1_dot_b2 = np.einsum('ij,ij->i', b1, b2)
		b3_dot_b2 = np.einsum('ij,ij->i', b3, b2)
		b2_sq = np.maximum(b2_mag**2, self._EPS)
		Fj = -((b1_dot_b2/b2_sq+1.0)[:,None]*Fi) + (b3_dot_b2/b2_sq)[:,None]*Fl
		Fk = -(Fi + Fj + Fl)
		np.add.at(forces, i_idx, Fi)
		np.add.at(forces, j_idx, Fj)
		np.add.at(forces, k_idx, Fk)
		np.add.at(forces, l_idx, Fl)
		return energy, forces
	def ImproperTorsionPotential(self, pose, cache, alg='harmonic',
			grad=True, box=None):
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
			float: potential energy in kJ/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		n = cache['n']
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
		b2_mag = np.maximum(np.linalg.norm(b2, axis=1), self._EPS)
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
				"ImproperTorsionPotential: unknown alg=%r "
				"(allowed: 'harmonic', 'fourier')" % (alg,))
		if not grad: return energy
		forces = np.zeros((n, 3), dtype=np.float64)
		n1_sq = np.maximum(np.einsum('ij,ij->i', n1, n1), self._EPS)
		n2_sq = np.maximum(np.einsum('ij,ij->i', n2, n2), self._EPS)
		Fi = -(dU_dphi * b2_mag / n1_sq)[:, None] * n1
		Fl =  (dU_dphi * b2_mag / n2_sq)[:, None] * n2
		b1_dot_b2 = np.einsum('ij,ij->i', b1, b2)
		b3_dot_b2 = np.einsum('ij,ij->i', b3, b2)
		b2_sq = np.maximum(b2_mag**2, self._EPS)
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
			float: potential energy in kJ/mol  (when grad=False)
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
		coef = -2.0 * k_ub * (r - s0) / np.maximum(r, self._EPS)
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
			float: potential energy in kJ/mol  (when grad=False)
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
		r = np.maximum(r, self._EPS)
		epsilon_r = self.Parameters['Constants']['epsilon_r']
		if alg == 'constant':
			coeff = 1389.35458 * q[None, :] / (epsilon_r * r**3)
		elif alg == 'ddd':
			coeff = 1389.35458 * q[None, :] / (epsilon_r * r**4)
		else:
			raise ValueError(
				"PolarisationPotential: unknown alg=%r "
				"(allowed: 'constant', 'ddd')" % (alg,))
		coeff = coeff * weight
		E = np.einsum('ij,ijk->ik', coeff, dr)
		E_sq = np.sum(E**2, axis=1)
		energy = float(-0.5 * np.sum(alpha * E_sq) / 1389.35458)
		if not grad: return energy
		p_pow = 3.0 if alg == 'constant' else 4.0
		rhat = dr / r[:, :, None]
		E_dot_rhat = np.einsum('ik,ijk->ij', E, rhat)
		G = E[:, None, :] - p_pow * E_dot_rhat[:, :, None] * rhat
		A = alpha[:, None] * coeff
		M = A[:, :, None] * G
		forces = (np.sum(M, axis=1) - np.sum(M, axis=0)) / 1389.35458
		return energy, forces
	def CMAPPotential(self, pose, cache, alg='catmullrom', grad=True, box=None):
		'''
		Calculates the CMAP backbone (phi, psi) cross-term correction energy
		Arguments:
		----------
			pose:  Pose - molecule source protein, DNA, RNA, or Molecule pose
			cache: dict - precomputed topology + parameter from _compile()
			alg:   Str algorithm type, 'catmullrom' (centred-difference
				bicubic) or 'openmm' (periodic-cubic-spline bicubic,
				bit-exact to OpenMM's CMAPTorsionForce)
			grad:  bool - if True, also return per-atom forces (N, 3) array
			box:   None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kJ/mol  (when grad=False)
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
		b2_mag = np.maximum(np.linalg.norm(b2, axis=1), self._EPS)
		b2n = b2 / b2_mag[:, None]
		ang = np.arctan2(
			np.einsum('ij,ij->i', np.cross(n1, b2n), n2),
			np.einsum('ij,ij->i', n1, n2))
		phi, psi = ang[:M], ang[M:]
		N_grid = tables.shape[1]
		H = 2.0 * np.pi / N_grid
		if alg == 'openmm':
			d1, d2, d12 = (cache['cmap_d1'], cache['cmap_d2'],
				cache['cmap_d12'])
			WT = np.array([
				[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
				[-3,0,0,3,0,0,0,0,-2,0,0,-1,0,0,0,0],
				[2,0,0,-2,0,0,0,0,1,0,0,1,0,0,0,0],
				[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
				[0,0,0,0,-3,0,0,3,0,0,0,0,-2,0,0,-1],
				[0,0,0,0,2,0,0,-2,0,0,0,0,1,0,0,1],
				[-3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,-3,3,0,0,-2,-1,0,0],
				[9,-9,9,-9,6,3,-3,-6,6,-6,-3,3,4,2,1,2],
				[-6,6,-6,6,-4,-2,2,4,-3,3,3,-3,-2,-1,-1,-2],
				[2,-2,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,2,-2,0,0,1,1,0,0],
				[-6,6,-6,6,-3,-3,3,3,-4,4,2,-2,-2,-2,-1,-1],
				[4,-4,4,-4,2,2,-2,-2,2,-2,-2,2,1,1,1,1]],
				dtype=np.float64)
			xa = np.mod(-psi, 2.0 * np.pi) / H
			ya = np.mod(-phi, 2.0 * np.pi) / H
			gi = np.floor(xa).astype(np.int64) % N_grid
			gj = np.floor(ya).astype(np.int64) % N_grid
			t  = xa - np.floor(xa)
			u  = ya - np.floor(ya)
			mm  = np.arange(M)[:, None]
			ci  = np.stack([gi, (gi + 1) % N_grid,
				(gi + 1) % N_grid, gi], axis=1)
			cj  = np.stack([gj, gj, (gj + 1) % N_grid,
				(gj + 1) % N_grid], axis=1)
			f   = tables[mm, ci, cj]
			fx  = d1[mm, ci, cj] * H
			fy  = d2[mm, ci, cj] * H
			fxy = d12[mm, ci, cj] * H * H
			rhs = np.concatenate([f, fx, fy, fxy], axis=1)
			c   = (rhs @ WT.T).reshape(M, 4, 4)
			pu  = [((c[:, i, 3] * u + c[:, i, 2]) * u
				+ c[:, i, 1]) * u + c[:, i, 0] for i in range(4)]
			dpu = [(3 * c[:, i, 3] * u + 2 * c[:, i, 2]) * u
				+ c[:, i, 1] for i in range(4)]
			E_per = ((pu[3] * t + pu[2]) * t + pu[1]) * t + pu[0]
			dE_dt = pu[1] + 2 * t * pu[2] + 3 * t * t * pu[3]
			dE_du = ((dpu[3] * t + dpu[2]) * t + dpu[1]) * t + dpu[0]
			dU_dpsi = -dE_dt / H
			dU_dphi = -dE_du / H
			energy = float(np.sum(E_per))
		else:
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
		n1_sq = np.maximum(np.einsum('ij,ij->i', n1, n1), self._EPS)
		n2_sq = np.maximum(np.einsum('ij,ij->i', n2, n2), self._EPS)
		Fi = -(dU_d * b2_mag / n1_sq)[:, None] * n1
		Fl =  (dU_d * b2_mag / n2_sq)[:, None] * n2
		b1_dot_b2 = np.einsum('ij,ij->i', b1, b2)
		b3_dot_b2 = np.einsum('ij,ij->i', b3, b2)
		b2_sq = np.maximum(b2_mag**2, self._EPS)
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
	Configurable scoring function for protein design and docking
	'''
	def __init__(self, name='Default', strict=False):
		'''
		Initialise a named scoring function from database.json
		Arguments:
		----------
			name:   str - parameter set under ['Score Parameters']
				(e.g. 'Default'); case-insensitive
			strict: bool - reserved for future use
		Returns:
		--------
			None: instance is configured in place
		'''
		self.strict = strict
		SP = DBLoad().get('Score Parameters', {}) or {}
		if not SP:
			raise ValueError(
				'Score: database.json has no "Score Parameters" key. '
				'Run vina.py / ref15.py first to populate.')
		key_map = {k.upper(): k for k in SP}
		if name.upper() not in key_map:
			raise ValueError(
				'Score: unknown name=%r (available: %r)'
				% (name, sorted(SP)))
		self.name = key_map[name.upper()]
		self.Parameters = copy.deepcopy(SP[self.name])
		if 'Terms' not in self.Parameters:
			raise ValueError(
				"Score: '%s' is missing the 'Terms' key" % name)
		self.terms = [(t[0], dict(t[1]))
			for t in self.Parameters['Terms']]
		self.scale = float(
			self.Parameters.get('Constants', {}).get('scale', 1.0))
		self._cache = None
		self._topo_cache = None
		self._topo_hash = None
		self._topo_refX = None
		self._skin = 1.5
		# build the cache pair list with a Verlet skin (master cutoff +
		# skin). Energy is unchanged (terms apply their own cutoffs); the
		# skin lets the pair list be reused across small coordinate moves.
		c = self.Parameters.setdefault('Constants', {})
		if 'fa_max_dis' in c:
			c['fa_max_dis'] = float(c['fa_max_dis']) + self._skin
	def __call__(self, pose, ligand=None, decompose=False,
			xs_override=None, nrot_override=None):
		'''
		Evaluate the score function for a pose (optionally with a ligand)
		Arguments:
		----------
			pose:          Pose or Molecule - receptor / source pose
			ligand:        Molecule or None - optional ligand
			decompose:     bool - if True, return (total, per_term dict)
			xs_override:   dict or None - validation hook; combined-index
				to XS type name, bypassing derived typing
			nrot_override: int or None - validation hook
		Returns:
		--------
			float OR (float, dict): total score in the score\'s native
			unit (REU, kcal/mol, or dimensionless); when decompose=True
			also returns a per-term breakdown
		'''
		# PERF (item 1): topology/geometry split. The cache is keyed on a
		# topology-only hash (bonds + atom records + AA info, NOT coords).
		# On a re-score with the same topology, reuse the cached per-atom
		# typing/charges/params/BFS/pair-list and refresh only geometry
		# (coords + pair distances) — Verlet-style pair-list reuse.
		plain = (ligand is None and xs_override is None
			and nrot_override is None)
		h = None
		if plain:
			bonds_key = tuple((int(k), tuple(sorted(int(j) for j in v)))
				for k, v in sorted(pose.data['Bonds'].items()))
			atoms_key = tuple((int(k), tuple(a))
				for k, a in sorted(pose.data['Atoms'].items()))
			aas = pose.data.get('Amino Acids')
			aas_key = None if aas is None else tuple(
				(int(k), info[0], info[1], tuple(info[2]))
				for k, info in sorted(aas.items()))
			h = hash((bonds_key, atoms_key, aas_key))
		X = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		reuse = False
		if plain and self._topo_cache is not None and self._topo_hash == h:
			disp = np.sqrt(((X - self._topo_refX) ** 2).sum(1)).max()
			reuse = disp < 0.5 * self._skin   # Verlet: safe within skin/2
		if reuse:
			cache = self._topo_cache
			# refresh geometry from live coords: recompute distances and
			# LkBall waters. Coordinates are taken as given, matching the
			# cache-build path and Rosetta itself.
			Xc = np.asarray(pose.data['Coordinates'], dtype=np.float64)
			cache['coords'] = Xc
			cache['pair_d'] = np.linalg.norm(
				Xc[cache['pairs_i']] - Xc[cache['pairs_j']], axis=1)
			wx, woff, wcnt = cache['place_waters'](Xc)
			cache['lkb_water_xyz'] = wx
			cache['lkb_water_off'] = woff
			cache['lkb_water_cnt'] = wcnt
			self._cache = cache
		else:
			self._cache = tools.ScoreMatch(pose, self.Parameters, ligand,
				xs_override, nrot_override)
			if plain:
				self._topo_cache = self._cache
				self._topo_hash = h
				self._topo_refX = X.copy()
		# per-score HBond memo: fullatomhbond is requested once by each of
		# the 4 HBond terms; clear so it is recomputed for these coords,
		# then computed once and shared across the 4 terms this call.
		self._cache['_hbond_memo'] = None
		self._cache['_dihedral_memo'] = {}
		per_term = {}
		torsional = False
		for method_name, kwargs in self.terms:
			if method_name == 'TorsionalPenalty':
				torsional = True
				continue
			fn = getattr(self, method_name, None)
			if fn is None:
				raise Exception(
					'Score: method %s not found' % method_name)
			out = fn(pose, cache=self._cache, ligand=ligand, **kwargs)
			per_term[method_name] = out
		inter_kj = sum(v.get('inter_weighted', 0.0)
			for v in per_term.values())
		intra_kj = sum(v.get('intra_weighted', 0.0)
			for v in per_term.values())
		if torsional:
			nrot_w = float(
				self.Parameters['Constants'].get('nrot_w', 0.0))
			nrot = float(self._cache.get('nrot', 0))
			denom = 1.0 + nrot_w * nrot
			affinity_kj = inter_kj / denom if denom != 0 else inter_kj
			total_native = affinity_kj * self.scale
			per_term['_summary'] = {
				'inter_total_kJ': inter_kj,
				'intra_total_kJ': intra_kj,
				'nrot': nrot, 'denom': denom,
				'affinity_native': total_native}
		else:
			total_native = (inter_kj + intra_kj) * self.scale
			per_term['_summary'] = {
				'inter_total_kJ': inter_kj,
				'intra_total_kJ': intra_kj,
				'total_native': total_native}
		if decompose:
			return float(total_native), per_term
		return float(total_native)
	def Gauss1Potential(self, pose, cache, ligand=None, **kw):
		'''
		Small-molecule pair Gaussian centred at d=0 Å, exp(-(d/0.5)^2)
		Arguments:
		----------
			pose:   Pose or Molecule - source pose
			cache:  dict - PatternSearch result
			ligand: Molecule or None - optional
		Returns:
		--------
			dict: term result with inter/intra raw and weighted
		'''
		return cache['gausspair'](cache, 'Gauss1')
	def Gauss2Potential(self, pose, cache, ligand=None, **kw):
		'''
		Small-molecule pair Gaussian centred at d=3 Å (long-range), exp(-((d-3)/2)^2)
		Arguments:
		----------
			pose:   Pose or Molecule - source pose
			cache:  dict - PatternSearch result
			ligand: Molecule or None - optional
		Returns:
		--------
			dict: term result with inter/intra raw and weighted
		'''
		return cache['gausspair'](cache, 'Gauss2')
	def RepulsionPotential(self, pose, cache, ligand=None, **kw):
		'''
		Small-molecule pair repulsion: d^2 where d < 0 (atomic overlap, quadratic penalty)
		Arguments:
		----------
			pose:   Pose or Molecule - source pose
			cache:  dict - PatternSearch result
			ligand: Molecule or None - optional
		Returns:
		--------
			dict: term result with inter/intra raw and weighted
		'''
		p = self.Parameters['Repulsion']
		offset = float(p['offset']); cutoff = float(p['cutoff'])
		weight = float(p['weight'])
		radii = cache['xs_radii_arr']; xs = cache['xs_types']
		def fn(ai, aj, rij, c):
			'''
			Per-pair kernel for the RepulsionPotential typed-pair sum
			Arguments:
			----------
				ai: np.ndarray - per-pair first-atom indices
				aj: np.ndarray - per-pair second-atom indices
				rij: np.ndarray - per-pair distance
				c: np.ndarray - per-pair connectivity weight
			Returns:
			--------
				np.ndarray: per-pair repulsion contribution
			'''
			ri = radii[xs[ai]]; rj = radii[xs[aj]]
			d = rij - (ri + rj + offset)
			gate = ((xs[ai] >= 0) & (xs[aj] >= 0) & (rij < cutoff))
			return np.where(gate & (d < 0), d * d, 0.0)
		inter_raw, intra_raw = cache['evalpairs'](cache, 'both', fn)
		return cache['termresult'](inter_raw, intra_raw, weight)
	def HydrophobicPotential(self, pose, cache, ligand=None, **kw):
		'''
		Small-molecule hydrophobic-pair slope-step contact (XS-typed hydrophobic pairs)
		Arguments:
		----------
			pose:   Pose or Molecule - source pose
			cache:  dict - PatternSearch result
			ligand: Molecule or None - optional
		Returns:
		--------
			dict: term result with inter/intra raw and weighted
		'''
		return cache['slopestep'](cache, 'Hydrophobic', 'hydrophobic')
	def HBondPotential(self, pose, cache, ligand=None, **kw):
		'''
		Small-molecule donor-acceptor slope-step hydrogen bond (non-directional)
		Arguments:
		----------
			pose:   Pose or Molecule - source pose
			cache:  dict - PatternSearch result
			ligand: Molecule or None - optional
		Returns:
		--------
			dict: term result with inter/intra raw and weighted
		'''
		return cache['slopestep'](cache, 'HBond', 'hbond')
	def TorsionalPenalty(self, pose, cache, ligand=None, **kw):
		'''
		Marker term; the actual division is applied in __call__ after the
		other terms have summed the intermolecular total
		Arguments:
		----------
			pose:   Pose or Molecule - source pose
			cache:  dict - PatternSearch result
			ligand: Molecule or None - optional
		Returns:
		--------
			dict: zero contributions (the term acts on the running total
			in __call__, not as a per-pair sum)
		'''
		return {'inter_raw': 0.0, 'intra_raw': 0.0,
			'inter_weighted': 0.0, 'intra_weighted': 0.0}
	def FaAtrPotential(self, pose, cache, ligand=None, **kw):
		'''
		Fa_atr - inter-residue LJ attractive split
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		raw, _ = cache['fullatomljraw'](cache, same_res=False)
		weight = float(self.Parameters['FaAtr']['weight'])
		return {'inter_raw': raw, 'intra_raw': 0.0,
			'inter_weighted': raw * weight, 'intra_weighted': 0.0,
			'raw': raw}
	def FaRepPotential(self, pose, cache, ligand=None, **kw):
		'''
		Fa_rep - inter-residue LJ repulsive split
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		_, raw = cache['fullatomljraw'](cache, same_res=False)
		weight = float(self.Parameters['FaRep']['weight'])
		return {'inter_raw': raw, 'intra_raw': 0.0,
			'inter_weighted': raw * weight, 'intra_weighted': 0.0,
			'raw': raw}
	def FaSolPotential(self, pose, cache, ligand=None, **kw):
		'''
		Fa_sol - inter-residue Lazaridis-Karplus solvation
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		raw = cache['fullatomsolraw'](cache, same_res=False)
		weight = float(self.Parameters['FaSol']['weight'])
		return {'inter_raw': raw, 'intra_raw': 0.0,
			'inter_weighted': raw * weight, 'intra_weighted': 0.0,
			'raw': raw}
	def FaIntraRepPotential(self, pose, cache, ligand=None, **kw):
		'''
		Fa_intra_rep - intra-residue LJ repulsive split with CP3
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		pi, pj, r, w = cache['fullatompairs'](cache, same_res=True, cp='cp3')
		weight = float(self.Parameters['FaIntraRep']['weight'])
		if len(pi) == 0:
			return {'inter_raw': 0.0, 'intra_raw': 0.0,
				'inter_weighted': 0.0, 'intra_weighted': 0.0,
				'raw': 0.0}
		_, repE = cache['ljpair'](cache, pi, pj, r)
		raw = float(np.sum(w * repE))
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * weight,
			'raw': raw}
	def FaIntraSolXover4Potential(self, pose, cache, ligand=None, **kw):
		'''
		Fa_intra_sol_xover4 - intra-residue LK solvation, only
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		raw = cache['fullatomsolraw'](cache, same_res=True)
		weight = float(self.Parameters['FaIntraSolXover4']['weight'])
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * weight,
			'raw': raw}
	def FaElecPotential(self, pose, cache, ligand=None, **kw):
		'''
		Fa_elec - Coulomb with sigmoidal distance-dependent
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		weight = float(self.Parameters['FaElec']['weight'])
		C = self.Parameters['Constants']
		C0 = float(C['coulomb_C0'])
		D = float(C['sigmoidal_D'])
		D0 = float(C['sigmoidal_D0'])
		S = float(C['sigmoidal_S'])
		d_min = float(C['fa_elec_min_dis'])
		d_max = float(C['fa_elec_max_dis'])
		q = cache['charges']
		def diel(d):
			'''
			Sigmoidal distance-dependent dielectric used by FaElec
			Arguments:
			----------
				r: np.ndarray - per-pair distance
			Returns:
			--------
				np.ndarray: per-pair dielectric value
			'''
			rS = d * S
			return D - 0.5 * (D - D0) * (
				2 + 2 * rS + rS * rS) * np.exp(-rS)
		def ddiel_dr(d):
			'''
			Derivative of the sigmoidal dielectric with respect to r
			Arguments:
			----------
				r: np.ndarray - per-pair distance
			Returns:
			--------
				np.ndarray: per-pair d(dielectric)/dr
			'''
			rS = d * S
			emr = np.exp(-rS)
			term1 = 2.0 * S + 2.0 * d * S * S
			term2 = (2.0 + 2.0 * rS + rS * rS) * (-S)
			return -0.5 * (D - D0) * (term1 + term2) * emr
		low_start = d_min - 0.25
		low_end = d_min + 0.25
		hi_start = d_max - 1.0
		hi_end = d_max
		def pair_sum(pi, pj, r, w):
			'''
			Per-pair Coulomb summation kernel for FaElec
			Arguments:
			----------
				ai: np.ndarray - per-pair first-atom indices
				aj: np.ndarray - per-pair second-atom indices
				rij: np.ndarray - per-pair distance
				c: np.ndarray - per-pair connectivity weight
			Returns:
			--------
				np.ndarray: per-pair electrostatic contribution
			'''
			if len(pi) == 0: return 0.0
			qq = q[pi] * q[pj]
			eps_r = diel(r)
			base = C0 * qq / (eps_r * np.maximum(r, 1e-9))
			base_at_max = C0 * qq / (diel(d_max) * d_max)
			e = base - base_at_max
			e = np.where(r >= d_max, 0.0, e)
			e_min_clamp = (C0 * qq / (diel(d_min) * d_min)
				- base_at_max)
			e = np.where(r < d_min, e_min_clamp, e)
			in_low = (r >= low_start) & (r < low_end)
			if np.any(in_low):
				h_low = low_end - low_start
				eps_le = diel(low_end)
				deps_le = ddiel_dr(low_end)
				v0_low = e_min_clamp
				v1_low = C0 * qq / (eps_le * low_end) - base_at_max
				d1_low = -C0 * qq * (eps_le + low_end * deps_le) \
					/ (low_end * low_end * eps_le * eps_le)
				t = (r - low_start) / h_low
				t2 = t * t; t3 = t2 * t
				H = ((2*t3 - 3*t2 + 1) * v0_low
					+ (-2*t3 + 3*t2) * v1_low
					+ (t3 - t2) * h_low * d1_low)
				e = np.where(in_low, H, e)
			in_hi = (r >= hi_start) & (r < hi_end)
			if np.any(in_hi):
				h_hi = hi_end - hi_start
				eps_hs = diel(hi_start)
				deps_hs = ddiel_dr(hi_start)
				v0_hi = C0 * qq / (eps_hs * hi_start) - base_at_max
				d0_hi = -C0 * qq * (eps_hs + hi_start * deps_hs) \
					/ (hi_start * hi_start * eps_hs * eps_hs)
				t = (r - hi_start) / h_hi
				t2 = t * t; t3 = t2 * t
				H = ((2*t3 - 3*t2 + 1) * v0_hi
					+ (t3 - 2*t2 + t) * h_hi * d0_hi)
				e = np.where(in_hi, H, e)
			return float(np.sum(w * e))
		pi, pj, r, w = cache['fullatompairs'](cache, same_res=False, cp='cp4',
			use_cp_rep=True)
		raw = pair_sum(pi, pj, r, w)
		return {'inter_raw': raw, 'intra_raw': 0.0,
			'inter_weighted': raw * weight, 'intra_weighted': 0.0,
			'raw': raw}
	def LkBallWtdPotential(self, pose, cache, ligand=None, **kw):
		'''
		Lk_ball_wtd - anisotropic LK solvation. For each polar
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		weight = float(self.Parameters['LkBallWtd']['weight'])
		pi, pj, r, w = cache['fullatompairs'](cache, same_res=False, cp='cp4')
		if len(pi) == 0:
			return {'inter_raw': 0.0, 'intra_raw': 0.0,
				'inter_weighted': 0.0, 'intra_weighted': 0.0,
				'raw': 0.0}
		isH = cache['is_H']
		heavy = ~(isH[pi] | isH[pj])
		if not np.any(heavy):
			return {'inter_raw': 0.0, 'intra_raw': 0.0,
				'inter_weighted': 0.0, 'intra_weighted': 0.0,
				'raw': 0.0}
		pi = pi[heavy]; pj = pj[heavy]; r = r[heavy]; w = w[heavy]
		w_iso = cache['lkb_w_iso']; w_ball = cache['lkb_w_ball']
		d2_low = cache['lkb_d2_low']
		water_xyz = cache['lkb_water_xyz']
		water_off = cache['lkb_water_off']
		water_cnt = cache['lkb_water_cnt']
		ramp_w2 = float(cache['lkb_ramp_w2'])
		X = cache['coords']
		lk_iso_i, lk_iso_j = cache['lkisopair'](cache, pi, pj, r)
		lk_iso_i = lk_iso_i * w
		lk_iso_j = lk_iso_j * w
		def _frac(p_polar, p_other):
			'''
			Fractional water-occupancy weight for one LkBallWtd water site
			Arguments:
			----------
				rsq: float - squared distance from heavy atom to water site
			Returns:
			--------
				float: occupancy fraction in [0, 1]
			'''
			out = np.zeros(len(p_polar), dtype=np.float64)
			cnt = water_cnt[p_polar]
			if not np.any(cnt > 0):
				return out
			has = cnt > 0
			idx = np.where(has)[0]
			other_xyz = X[p_other[idx]]
			polar_off = water_off[p_polar[idx]]
			polar_cnt = water_cnt[p_polar[idx]]
			d2_low_other = d2_low[p_other[idx]]
			MFADE = 1.0
			frac_loc = np.zeros(len(idx), dtype=np.float64)
			for k in range(len(idx)):
				off = int(polar_off[k])
				n_w = int(polar_cnt[k])
				ws = water_xyz[off:off + n_w]
				diff = ws - other_xyz[k]
				d2_arr = np.einsum('ij,ij->i', diff, diff)
				weighted = -MFADE * math.log(
					float(np.sum(np.exp(
						-(d2_arr - d2_low_other[k]) / MFADE))))
				if weighted >= ramp_w2:
					frac_loc[k] = 0.0
				elif weighted <= 0.0:
					frac_loc[k] = 1.0
				else:
					xprime = weighted / ramp_w2
					frac_loc[k] = (1 - xprime * xprime) ** 2
			out[idx] = frac_loc
			return out
		w_iso_i = w_iso[pi]; w_ball_i = w_ball[pi]
		nonzero_i = water_cnt[pi] > 0
		w_iso_j = w_iso[pj]; w_ball_j = w_ball[pj]
		nonzero_j = water_cnt[pj] > 0
		total = 0.0
		if np.any(nonzero_i):
			frac_i = _frac(pi, pj)
			total += float(np.sum(
				w_iso_i * lk_iso_i + w_ball_i * lk_iso_i * frac_i))
		if np.any(nonzero_j):
			frac_j = _frac(pj, pi)
			total += float(np.sum(
				w_iso_j * lk_iso_j + w_ball_j * lk_iso_j * frac_j))
		raw = total
		return {'inter_raw': raw, 'intra_raw': 0.0,
			'inter_weighted': raw * weight, 'intra_weighted': 0.0,
			'raw': raw}
	def FaDunPotential(self, pose, cache, ligand=None, **kw):
		'''
		Fa_dun - Dunbrack rotamer probability.
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		weight = float(self.Parameters['FaDun']['weight'])
		per_res = kw.get('per_res')
		try:
			from .pose import DBLoad
			rl = DBLoad().get('Rotamer Library', {}) or {}
		except Exception:
			rl = {}
		residues_db = rl.get('residues', {})
		if not residues_db:
			return cache['fullatomstubterm']('FaDun')
		phi_start = float(rl.get('phi_start', -180.0))
		phi_step = float(rl.get('phi_step', 10.0))
		phi_n = int(rl.get('phi_n', 36))
		psi_start = float(rl.get('psi_start', -180.0))
		psi_step = float(rl.get('psi_step', 10.0))
		psi_n = int(rl.get('psi_n', 36))
		aas = pose.data.get('Amino Acids') or {}
		SIG_MIN = 0.5
		LOG_2PI = math.log(2.0 * math.pi)
		raw = 0.0
		for ri, info in aas.items():
			tri = info[5] if len(info) >= 6 else None
			if tri == 'HIS_D': tri = 'HIS'
			entry = residues_db.get(tri)
			if entry is None: continue
			n_chi = int(entry.get('n_chi', 0))
			if n_chi <= 0: continue
			try:
				phi = cache['cdih'](pose, int(ri), 'PHI')
				psi = cache['cdih'](pose, int(ri), 'PSI')
			except Exception:
				phi = float('nan'); psi = float('nan')
			if math.isnan(phi): phi = -90.0
			if math.isnan(psi): psi = 130.0
			fp = (phi - phi_start) / phi_step
			fs = (psi - psi_start) / psi_step
			ip0 = int(math.floor(fp))
			js0 = int(math.floor(fs))
			tp = fp - ip0
			ts = fs - js0
			rot = entry['rotamers']
			offs = rot['bin_offsets']
			tbl = rot['table']
			chi_now = []
			bad = False
			for ci in range(n_chi):
				try:
					v = pose.GetDihedral(int(ri),
						'CHI', chi_type=ci+1)
				except Exception:
					bad = True; break
				if math.isnan(v): bad = True; break
				chi_now.append(v)
			if bad: continue
			def binchi(c):
				'''
				Place a chi angle into one of the rotamer wells defined by the FaDun table
				Arguments:
				----------
					chi: float - chi angle in degrees
					wells: list - per-row rotamer-well centre angles
				Returns:
				--------
					int: index of the matching well
				'''
				c = ((c + 180.0) % 360.0) - 180.0
				if 0.0 <= c <= 120.0: return 1
				if abs(c) >= 120.0: return 2
				return 3
			SEMI_ROT = ('ASP','ASN','GLU','GLN','PHE','TYR','TRP','HIS')
			n_rot = n_chi - 1 if tri in SEMI_ROT else n_chi
			if tri in SEMI_ROT:
				nrdata = cache['fadun_nrchi_data'](tri)
				if nrdata:
					rot_bins = tuple(binchi(chi_now[k])
						for k in range(n_rot))
					chi_last = chi_now[n_chi - 1]
					clow = nrdata['chi_last_low']
					cstep = nrdata['chi_last_step']
					cn = nrdata['chi_last_n']
					chigh = clow + cn * cstep
					if (chigh - clow) < 360.0 - 1e-6:
						while chi_last < clow:
							chi_last += 180.0
						while chi_last >= chigh:
							chi_last -= 180.0
					else:
						chi_last = ((chi_last + 180.0) % 360.0) - 180.0
					nlr, mus_v, sigs_v, nld = \
						cache['fadun_nrchi_eval'](tri, rot_bins,
							phi, psi, chi_last)
					if nlr is not None:
						dev_v = 0.0
						for ci in range(n_rot):
							d = ((chi_now[ci] - mus_v[ci] + 180.0)
								% 360.0) - 180.0
							dev_v += (d / sigs_v[ci]) ** 2
						contrib = nlr + nld + 0.5 * dev_v
						raw += contrib
						if per_res is not None:
							per_res[int(ri)] = contrib
						continue
			if tri not in SEMI_ROT:
				grids = cache['fadun_rotwell_grid'](
					tri, n_chi, residues_db)
				if tri == 'PRO':
					rot_bins = [
						(1 if chi_now[k] > 0 else 2) if k == 0 else 1
						for k in range(n_chi)]
				else:
					rot_bins = [binchi(chi_now[k])
						for k in range(n_chi)]
				rot_idx_now = sum(rot_bins[k] * (10 ** (3 - k))
					for k in range(n_chi))
				grid = grids.get(rot_idx_now)
				if grid is not None and grid['has_data'].any():
					MAXE_NL = math.log(1e6)
					neg_log_P_v = cache['fadun_spline_eval'](
						grid['neglogP'],
						grid['neglogP_ypp_psi'], fp, fs)
					neg_log_P_v = min(MAXE_NL, neg_log_P_v)
					ip0 = int(math.floor(fp))
					js0 = int(math.floor(fs))
					tp = fp - ip0
					ts = fs - js0
					w00 = (1 - tp) * (1 - ts)
					w10 = tp * (1 - ts)
					w01 = (1 - tp) * ts
					w11 = tp * ts
					ip0m = ip0 % 36; ip1m = (ip0 + 1) % 36
					js0m = js0 % 36; js1m = (js0 + 1) % 36
					mus_v = []
					sigs_v = []
					for ci in range(n_chi):
						mg = grid['mu'][ci]
						sg = grid['sd'][ci]
						a = mg[ip0m, js0m]
						b = mg[ip1m, js0m]
						c = mg[ip0m, js1m]
						d = mg[ip1m, js1m]
						b_u = a + ((b - a + 180.0) % 360.0 - 180.0)
						c_u = a + ((c - a + 180.0) % 360.0 - 180.0)
						d_u = a + ((d - a + 180.0) % 360.0 - 180.0)
						mus_v.append(w00 * a + w10 * b_u
							+ w01 * c_u + w11 * d_u)
						sigs_v.append(max(
							w00 * sg[ip0m, js0m]
							+ w10 * sg[ip1m, js0m]
							+ w01 * sg[ip0m, js1m]
							+ w11 * sg[ip1m, js1m], SIG_MIN))
					dev_v = 0.0
					for ci in range(n_chi):
						d = ((chi_now[ci] - mus_v[ci] + 180.0)
							% 360.0) - 180.0
						dev_v += (d / sigs_v[ci]) ** 2
					contrib = neg_log_P_v + 0.5 * dev_v
					raw += contrib
					if per_res is not None:
						per_res[int(ri)] = contrib
					continue
			if tri == 'PRO':
				rotwell_now = tuple(
					(1 if chi_now[k] > 0 else 2) if k == 0 else 1
					for k in range(n_rot))
			else:
				rotwell_now = tuple(binchi(chi_now[k])
					for k in range(n_rot))
			def cellrows(i_phi, i_psi):
				'''
				Return FaDun rotamer-table rows for the (phi_cell, psi_cell) bin
				Arguments:
				----------
					phi_cell: int - phi grid index
					psi_cell: int - psi grid index
				Returns:
				--------
					list: rotamer rows associated with the bin
				'''
				bi = (i_phi % phi_n) * psi_n + (i_psi % psi_n)
				if bi + 1 >= len(offs): return []
				return tbl[offs[bi]:offs[bi+1]]
			def rowwell(r2):
				'''
				Pull rotamer-well mu and sigma values for one row of the FaDun table
				Arguments:
				----------
					row: dict - one FaDun row entry
				Returns:
				--------
					tuple: (mu_arr, sigma_arr) over the row chi angles
				'''
				rw = []
				for ci in range(n_rot):
					mu = r2[2 + ci]
					if tri == 'PRO':
						b = (1 if mu > 0 else 2) if ci == 0 else 1
					else:
						b = binchi(mu)
					rw.append(b)
				return tuple(rw)
			def cellmatch(rows_):
				'''
				Match a chi vector to the closest rotamer well in a (phi, psi) cell
				Arguments:
				----------
					cell_rows: list - rotamer rows for this cell
					chi: np.ndarray - observed chi angles
				Returns:
				--------
					tuple: (best_row, best_well_index, distance)
				'''
				if n_rot == n_chi:
					ent = 0.0
					for r2 in rows_:
						if r2[1] > 0.0:
							ent += r2[1] * math.log(r2[1])
					cand = [r2 for r2 in rows_
						if r2[1] > 0.0
						and rowwell(r2) == rotwell_now]
					if not cand: return (0.0, None, None, ent)
					match = max(cand, key=lambda r2: r2[1])
					mus = [match[2 + ci] for ci in range(n_chi)]
					sigs = [match[2 + n_chi + ci]
						for ci in range(n_chi)]
					return (match[1], mus, sigs, ent)
				groups = {}
				for r2 in rows_:
					if r2[1] <= 0.0: continue
					groups.setdefault(rowwell(r2), []).append(r2)
				ent = 0.0
				for grows in groups.values():
					Pg = sum(r2[1] for r2 in grows)
					if Pg > 0.0: ent += Pg * math.log(Pg)
				if rotwell_now not in groups:
					return (0.0, None, None, ent)
				grp = groups[rotwell_now]
				P_rotwell = sum(r2[1] for r2 in grp)
				chi_last = chi_now[n_chi - 1]
				def unwrap_to(c, ref):
					'''
					Add multiples of 360 to bring angle close to an anchor
					Arguments:
					----------
						x: float - angle to unwrap (degrees)
						anchor: float - reference angle
					Returns:
					--------
						float: x +/- k*360 closest to anchor
					'''
					return ref + ((c - ref + 180.0) % 360.0 - 180.0)
				pts = sorted(((unwrap_to(r2[2 + (n_chi-1)],
					chi_last), r2) for r2 in grp),
					key=lambda x: x[0])
				cls = [pt[0] for pt in pts]
				prs = [pt[1] for pt in pts]
				P_eff = None
				for k in range(len(cls) - 1):
					if cls[k] <= chi_last <= cls[k+1]:
						span = cls[k+1] - cls[k]
						if span <= 0:
							P_eff = prs[k][1]
						else:
							t = (chi_last - cls[k]) / span
							P_eff = (1 - t) * prs[k][1] \
								+ t * prs[k+1][1]
						break
				if P_eff is None:
					P_eff = prs[0][1] if abs(chi_last - cls[0]) < \
						abs(chi_last - cls[-1]) else prs[-1][1]
				match = max(grp, key=lambda r2: r2[1])
				mus = [match[2 + ci] for ci in range(n_chi)]
				sigs = [match[2 + n_chi + ci]
					for ci in range(n_chi)]
				return (P_eff, mus, sigs, ent)
			def crom(p0, p1, p2, p3, t):
				'''
				1D Catmull-Rom interpolation over four equally spaced samples
				Arguments:
				----------
					p0: float - sample at t = -1
					p1: float - sample at t = 0
					p2: float - sample at t = +1
					p3: float - sample at t = +2
					t: float - interpolation parameter in [0, 1]
				Returns:
				--------
					float: interpolated value at t
				'''
				return 0.5 * ((2 * p1) + (-p0 + p2) * t
					+ (2*p0 - 5*p1 + 4*p2 - p3) * t * t
					+ (-p0 + 3*p1 - 3*p2 + p3) * t * t * t)
			samples = [[None]*4 for _ in range(4)]
			for di in range(4):
				for dj in range(4):
					ip = ip0 - 1 + di
					js = js0 - 1 + dj
					samples[di][dj] = cellmatch(cellrows(ip, js))
			ref_p, ref_mu, ref_sig, _ = samples[1][1]
			if ref_mu is None:
				for di in range(4):
					for dj in range(4):
						if samples[di][dj][1] is not None:
							ref_mu = samples[di][dj][1]
							break
					if ref_mu is not None: break
			if ref_mu is None:
				best_dev = None; best_row = None
				for r2 in cellrows(ip0, js0):
					if r2[1] <= 0.0: continue
					dv = 0.0
					for ci in range(n_rot):
						mu = r2[2 + ci]
						sig = r2[2 + n_chi + ci]
						if sig < SIG_MIN: sig = SIG_MIN
						d = ((chi_now[ci] - mu + 180.0)
							% 360.0) - 180.0
						dv += (d / sig) ** 2
					if best_dev is None or dv < best_dev:
						best_dev = dv; best_row = r2
				if best_row is None: continue
				P_k = best_row[1]
				dev = best_dev
				cent = 0.0
				cgroups = {}
				for r2 in cellrows(ip0, js0):
					if r2[1] <= 0.0: continue
					cgroups.setdefault(rowwell(r2), 0.0)
					cgroups[rowwell(r2)] += r2[1]
				for Pg in cgroups.values():
					if Pg > 0: cent += Pg * math.log(Pg)
				contrib = -math.log(P_k) + 0.5 * dev
				raw += contrib
				if per_res is not None:
					per_res[int(ri)] = contrib
				continue
			def unwrap(mu_arr, ref):
				'''
				Unwrap a 1D periodic angle sequence so consecutive samples are within +/-180 degrees
				Arguments:
				----------
					arr: np.ndarray - 1D array of angles in degrees
				Returns:
				--------
					np.ndarray: unwrapped copy of arr
				'''
				out = []
				for m in mu_arr:
					d = ((m - ref + 180.0) % 360.0) - 180.0
					out.append(ref + d)
				return out
			MAXE = 18.0
			neg_log_P = [[MAXE]*4 for _ in range(4)]
			mu_grid = [[None]*4 for _ in range(4)]
			sig_grid = [[None]*4 for _ in range(4)]
			ent_grid = [[0.0]*4 for _ in range(4)]
			for di in range(4):
				for dj in range(4):
					Pk, mus, sigs, ent = samples[di][dj]
					ent_grid[di][dj] = ent
					if Pk > 0.0:
						neg_log_P[di][dj] = -math.log(Pk)
						# unwrap mus toward ref
						mu_grid[di][dj] = unwrap(mus, ref_mu[0]) \
							if ref_mu else mus
						sig_grid[di][dj] = sigs
					else:
						mu_grid[di][dj] = list(ref_mu) \
							if ref_mu else [0.0]*n_chi
						sig_grid[di][dj] = list(ref_sig) \
							if ref_sig else [SIG_MIN]*n_chi
			def crom2d(grid, tp, ts):
				'''
				2D bicubic Catmull-Rom interpolation over a 4x4 sample grid
				Arguments:
				----------
					grid: np.ndarray - 4x4 sample values
					tx: float - x-direction parameter in [0, 1]
					ty: float - y-direction parameter in [0, 1]
				Returns:
				--------
					float: interpolated value at (tx, ty)
				'''
				cols = []
				for di in range(4):
					row = grid[di]
					cols.append(crom(row[0], row[1], row[2], row[3], ts))
				return crom(cols[0], cols[1], cols[2], cols[3], tp)
			neg_log_P_i = crom2d(neg_log_P, tp, ts)
			ent_i = crom2d(ent_grid, tp, ts)
			mu_i = []
			sig_i = []
			for ci in range(n_chi):
				gm = [[mu_grid[di][dj][ci]
					for dj in range(4)] for di in range(4)]
				gs = [[sig_grid[di][dj][ci]
					for dj in range(4)] for di in range(4)]
				mu_i.append(crom2d(gm, tp, ts))
				sig_i.append(max(crom2d(gs, tp, ts), SIG_MIN))
			dev = 0.0
			for ci in range(n_rot):
				d = ((chi_now[ci] - mu_i[ci] + 180.0)
					% 360.0) - 180.0
				dev += (d / sig_i[ci]) ** 2
			E_r = neg_log_P_i + 0.5 * dev
			raw += E_r
			if per_res is not None:
				per_res[int(ri)] = E_r
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * weight,
			'raw': raw}
	def RamaPreProTermPotential(self, pose, cache, ligand=None, **kw):
		'''
		Rama_prepro - phi/psi Ramachandran propensity.
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		weight = float(self.Parameters['RamaPreProTerm']['weight'])
		rd = self.Parameters.get('Rama_data') or {}
		all_t = rd.get('all', {})
		pre_t = rd.get('prepro', {})
		if not all_t:
			return cache['fullatomstubterm']('RamaPreProTerm')
		aas = pose.data.get('Amino Acids') or {}
		raw = 0.0
		sorted_ris = sorted(int(r) for r in aas.keys())
		next_tri = {}
		for k, ri in enumerate(sorted_ris):
			if k + 1 < len(sorted_ris):
				nxt = sorted_ris[k+1]
				nxt_info = aas.get(nxt)
				if nxt_info and len(nxt_info) >= 6:
					next_tri[ri] = nxt_info[5]
		for ri in sorted_ris:
			info = aas.get(ri)
			if info is None: continue
			tri = info[5] if len(info) >= 6 else None
			if tri == 'HIS_D': tri = 'HIS'
			try:
				phi = cache['cdih'](pose, ri, 'PHI')
				psi = cache['cdih'](pose, ri, 'PSI')
			except Exception: continue
			if math.isnan(phi) or math.isnan(psi): continue
			use_pre = (next_tri.get(ri) == 'PRO'
				and tri in pre_t and pre_t[tri])
			table = pre_t[tri] if use_pre else all_t.get(tri)
			if table is None: continue
			cache_key = id(table)
			if not hasattr(self, '_rama_entropy'):
				self._rama_entropy = {}
				self._rama_logshift = {}
			ent_cache = self._rama_entropy
			shift_cache = self._rama_logshift
			ent = ent_cache.get(cache_key)
			if ent is None:
				S = 0.0
				for row in table:
					for nE in row:
						S += math.exp(-nE)
				logS = math.log(S) if S > 0 else 0.0
				shift_cache[cache_key] = logS
				ent = 0.0
				for row in table:
					for nE in row:
						p_norm = math.exp(-(nE + logS))
						if p_norm > 0:
							ent += p_norm * math.log(p_norm)
				ent_cache[cache_key] = ent
			log_shift = shift_cache[cache_key]
			fp = (phi + 180.0) / 10.0
			fs = (psi + 180.0) / 10.0
			e = cache['rama_spline_eval'](table, fp, fs)
			raw += (e + log_shift + ent)
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * weight,
			'raw': raw}
	def PAaPpPotential(self, pose, cache, ligand=None, **kw):
		'''
		P_aa_pp - P(aa|phi,psi) propensity.
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		weight = float(self.Parameters['PAaPp']['weight'])
		paa = self.Parameters.get('P_AA') or {}
		paapp = self.Parameters.get('P_AA_pp') or {}
		if not paa or not paapp:
			return cache['fullatomstubterm']('PAaPp')
		aas = pose.data.get('Amino Acids') or {}
		nterm = set(); cterm = set()
		if aas:
			ch_map = {}
			for ri, info in aas.items():
				ch = info[1] if len(info) > 1 else ''
				ch_map.setdefault(ch, []).append(int(ri))
			for ris in ch_map.values():
				ris.sort()
				if ris:
					nterm.add(ris[0]); cterm.add(ris[-1])
		raw = 0.0
		if not hasattr(self, '_paapp_spline_cache'):
			self._paapp_spline_cache = {}
		cache_pp = self._paapp_spline_cache
		if not cache_pp:
			MAXE = math.log(1e6)
			for aa, tbl in paapp.items():
				grid = np.zeros((36, 36))
				for i in range(36):
					for j in range(36):
						v = tbl[i][j]
						grid[i, j] = (-math.log(v)
							if v > 0 else MAXE)
				ypp_psi = np.stack(
					[cache['periodic_cubic_spline'](grid[i])
						for i in range(36)])
				cache_pp[aa] = (grid, ypp_psi)
		for ri, info in aas.items():
			if int(ri) in nterm or int(ri) in cterm: continue
			tri = info[5] if len(info) >= 6 else None
			if tri == 'HIS_D': tri = 'HIS'
			if tri not in cache_pp: continue
			try:
				phi = cache['cdih'](pose, int(ri), 'PHI')
				psi = cache['cdih'](pose, int(ri), 'PSI')
			except Exception: continue
			if math.isnan(phi) or math.isnan(psi): continue
			fp = (phi + 175.0) / 10.0
			fs = (psi + 175.0) / 10.0
			grid, ypp_psi = cache_pp[tri]
			neg_log_pp = cache['fadun_spline_eval'](grid, ypp_psi, fp, fs)
			raw += neg_log_pp + math.log(paa.get(tri, 1.0))
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * weight,
			'raw': raw}
	def OmegaPotential(self, pose, cache, ligand=None, **kw):
		'''
		Omega - peptide-bond omega tether (OmegaTether term).
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		# OmegaTether's per-residue weight, from Port('ref15')
		omega_k = float(self.Parameters['Omega']['tether_k'])
		weight = float(self.Parameters['Omega']['weight'])
		omega_tab = self.Parameters.get('Omega_tables') or {}
		if not omega_tab:
			return cache['fullatomstubterm']('Omega')
		aas = pose.data.get('Amino Acids') or {}
		cterm = set()
		if aas:
			ch_map = {}
			for ri, info in aas.items():
				ch = info[1] if len(info) > 1 else ''
				ch_map.setdefault(ch, []).append(int(ri))
			for ris in ch_map.values():
				ris.sort()
				if ris: cterm.add(ris[-1])
		raw = 0.0
		normalization = math.log(1.0 / (6.0 * math.sqrt(2 * math.pi)))
		if not hasattr(self, '_omega_spline_cache'):
			self._omega_spline_cache = {}
		cache_o = self._omega_spline_cache
		if not cache_o:
			for key, t in omega_tab.items():
				mu_g = np.array(t['mu'])
				sig_g = np.array(t['sigma'])
				mu_ypp = np.stack(
					[cache['periodic_cubic_spline'](mu_g[i])
						for i in range(36)])
				sig_ypp = np.stack(
					[cache['periodic_cubic_spline'](sig_g[i])
						for i in range(36)])
				cache_o[key] = (mu_g, mu_ypp, sig_g, sig_ypp)
		for ri in sorted(aas):
			if int(ri) in cterm: continue
			tri = aas[ri][5] if len(aas[ri]) >= 6 else None
			if tri == 'HIS_D': tri = 'HIS'
			try:
				om = pose.GetDihedral(int(ri), 'OMEGA')
				phi = cache['cdih'](pose, int(ri), 'PHI')
				psi = cache['cdih'](pose, int(ri), 'PSI')
			except Exception: continue
			if math.isnan(om): continue
			om_nn = om
			while om_nn < 0: om_nn += 360
			while om_nn >= 360: om_nn -= 360
			om_p = om_nn
			while om_p < -90: om_p += 360
			while om_p > 270: om_p -= 360
			if om_p < 90:
				dangle = ((om_p - 0 + 180) % 360) - 180
				raw += omega_k * dangle * dangle
				continue
			und = float(self.Parameters['Omega']['undefined_torsion'])
			if math.isnan(phi): phi = und
			if math.isnan(psi): psi = und
			if tri == 'GLY': key = 'gly'
			elif tri == 'PRO': key = 'pro'
			elif tri in ('ILE', 'VAL'): key = 'valile'
			else: key = 'all'
			mu_g, mu_ypp, sig_g, sig_ypp = cache_o[key]
			phi_nn = phi
			while phi_nn < 0: phi_nn += 360
			psi_nn = psi
			while psi_nn < 0: psi_nn += 360
			fp = (phi_nn - 5.0) / 10.0
			fs = (psi_nn - 5.0) / 10.0
			fp = fp % 36.0
			fs = fs % 36.0
			mu = cache['fadun_spline_eval'](mu_g, mu_ypp, fp, fs)
			sigma = cache['fadun_spline_eval'](sig_g, sig_ypp, fp, fs)
			if sigma < 1e-6: continue
			entropy = -math.log(1.0 / (sigma * math.sqrt(2 * math.pi)))
			# offset = subtract_degree_angles(omega_p, mu)
			offset = ((om_p - mu + 180) % 360) - 180
			logprob = offset * offset / (2 * sigma * sigma)
			raw += normalization + entropy + logprob
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * weight,
			'raw': raw}
	def ProClosePotential(self, pose, cache, ligand=None, **kw):
		'''
		Pro_close - proline ring closure (ProClosureEnergy term).
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		weight = float(self.Parameters['ProClose']['weight'])
		aas = pose.data.get('Amino Acids') or {}
		atoms = pose.data['Atoms']
		coords = np.asarray(pose.data['Coordinates'])
		raw = 0.0
		sd_sq = float(self.Parameters['ProClose']['planar_sd']) ** 2
		PC = self.Parameters.get('ProClose') or {}
		trans_mean = math.radians(float(PC['trans_chi4_mean']))
		trans_sd = math.radians(float(PC['trans_chi4_sd']))
		cis_mean = math.radians(float(PC['cis_chi4_mean']))
		cis_sd = math.radians(float(PC['cis_chi4_sd']))
		def place(p, g, gg, bond, theta, phi):
			'''
			Place a virtual atom at a fixed bond length, angle, and dihedral from three reference atoms
			Arguments:
			----------
				a: np.ndarray - first reference atom position
				b: np.ndarray - second reference atom position
				c: np.ndarray - third reference atom position
				r: float - bond length from c
				theta: float - bond angle b-c-virt in degrees
				phi: float - dihedral a-b-c-virt in degrees
			Returns:
			--------
				np.ndarray: virtual-atom position
			'''
			e_pg = g - p
			e_pg = e_pg / np.linalg.norm(e_pg)
			e_ggg = gg - g
			e_ggg = e_ggg / np.linalg.norm(e_ggg)
			perp = e_ggg - np.dot(e_ggg, e_pg) * e_pg
			perp = perp / np.linalg.norm(perp)
			normal = np.cross(e_pg, perp)
			d = (-math.cos(theta) * e_pg
				+ math.sin(theta) * (math.cos(phi) * perp
					+ math.sin(phi) * normal))
			return p + bond * d
		def dihedral(a, b, c, d):
			'''
			Dihedral angle of four points in degrees
			Arguments:
			----------
				p1: np.ndarray - first point
				p2: np.ndarray - second point
				p3: np.ndarray - third point
				p4: np.ndarray - fourth point
			Returns:
			--------
				float: dihedral angle in degrees
			'''
			b1 = a - b; b2 = c - b; b3 = d - c
			b2_norm = b2 / np.linalg.norm(b2)
			v = b1 - np.dot(b1, b2_norm) * b2_norm
			w = b3 - np.dot(b3, b2_norm) * b2_norm
			x = float(np.dot(v, w))
			y = float(np.dot(np.cross(b2_norm, v), w))
			return math.atan2(y, x)
		n_term_set = set()
		if aas:
			chains = {}
			for ri, info in aas.items():
				ch = info[1] if len(info) > 1 else ''
				chains.setdefault(ch, []).append(int(ri))
			for ch, ris in chains.items():
				ris.sort()
				if ris: n_term_set.add(ris[0])
		ri_sorted = sorted(int(r) for r in aas.keys())
		ri_to_prev = {}
		for i, r in enumerate(ri_sorted):
			if i > 0: ri_to_prev[r] = ri_sorted[i - 1]
		for ri, info in aas.items():
			tri = info[5] if len(info) >= 6 else None
			if tri != 'PRO': continue
			name_to_idx = {atoms[int(a)][0]: int(a)
				for a in info[2] + info[3]}
			needed = ('N', 'CA', 'CG', 'CD')
			if not all(nm in name_to_idx for nm in needed): continue
			n = coords[name_to_idx['N']]
			ca = coords[name_to_idx['CA']]
			cg = coords[name_to_idx['CG']]
			cd = coords[name_to_idx['CD']]
			nv = place(cd, cg, n, float(PC['nv_d']),
				math.radians(float(PC['nv_theta'])), 0.0)
			d2_n_nv = float(np.sum((nv - n) ** 2))
			raw += d2_n_nv / sd_sq
			if int(ri) in n_term_set:
				cav = place(nv, cd, ca, float(PC['cav_d']),
					math.radians(float(PC['cav_theta'])), 0.0)
				d2_ca_cav = float(np.sum((cav - ca) ** 2))
				raw += d2_ca_cav / sd_sq
			prev = ri_to_prev.get(int(ri))
			if prev is None: continue
			prev_info = aas.get(prev)
			if prev_info is None: continue
			prev_atoms = {atoms[int(a)][0]: int(a)
				for a in prev_info[2] + prev_info[3]}
			if 'C' not in prev_atoms or 'O' not in prev_atoms: continue
			c_prev = coords[prev_atoms['C']]
			o_prev = coords[prev_atoms['O']]
			chi4 = dihedral(coords[name_to_idx['CD']], n,
				c_prev, o_prev)
			if chi4 < -math.pi / 2: chi4 += 2 * math.pi
			if chi4 > math.pi / 2:
				diff = chi4 - trans_mean
				raw += diff * diff / (trans_sd * trans_sd)
			else:
				diff = chi4 - cis_mean
				raw += diff * diff / (cis_sd * cis_sd)
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * weight,
			'raw': raw}
	def DslfFa13Potential(self, pose, cache, ligand=None, **kw):
		'''
		Dslf_fa13 - disulfide geometry potential, per the
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		from math import erfc
		weight = float(self.Parameters['DslfFa13']['weight'])
		aas = pose.data.get('Amino Acids') or {}
		atoms = pose.data['Atoms']
		coords = np.asarray(pose.data['Coordinates'])
		raw = 0.0
		shift = 2.0
		mest = math.exp(float(
			self.Parameters['DslfFa13']['mest_log']))
		# von Mises / skew-normal fits, installed by tools.Port('ref15')
		# from Rosetta's FullatomDisulfideParams13 constructor.
		DS = self.Parameters['DslfFa13']
		wt_len = float(DS['wt_len']); wt_ang = float(DS['wt_ang'])
		wt_dihSS = float(DS['wt_dihSS'])
		wt_dihCS = float(DS['wt_dihCS'])
		d_location = DS['d_location']; d_scale = DS['d_scale']
		d_shape = DS['d_shape']
		a_logA = DS['a_logA']; a_kappa = DS['a_kappa']; a_mu = DS['a_mu']
		dss_logA1 = DS['dss_logA1']; dss_kappa1 = DS['dss_kappa1']
		dss_mu1 = DS['dss_mu1']
		dss_logA2 = DS['dss_logA2']; dss_kappa2 = DS['dss_kappa2']
		dss_mu2 = DS['dss_mu2']
		dcs_logA1 = DS['dcs_logA1']; dcs_mu1 = DS['dcs_mu1']
		dcs_kappa1 = DS['dcs_kappa1']
		dcs_logA2 = DS['dcs_logA2']; dcs_mu2 = DS['dcs_mu2']
		dcs_kappa2 = DS['dcs_kappa2']
		dcs_logA3 = DS['dcs_logA3']; dcs_mu3 = DS['dcs_mu3']
		dcs_kappa3 = DS['dcs_kappa3']
		def dihedral(a, b, c, d):
			'''
			Dihedral angle of four points in degrees
			Arguments:
			----------
				p1: np.ndarray - first point
				p2: np.ndarray - second point
				p3: np.ndarray - third point
				p4: np.ndarray - fourth point
			Returns:
			--------
				float: dihedral angle in degrees
			'''
			b1 = a - b; b2 = c - b; b3 = d - c
			n = np.linalg.norm(b2)
			if n < 1e-9: return 0.0
			b2n = b2 / n
			v = b1 - np.dot(b1, b2n) * b2n
			w = b3 - np.dot(b3, b2n) * b2n
			x = float(np.dot(v, w))
			y = float(np.dot(np.cross(b2n, v), w))
			return math.degrees(math.atan2(y, x))
		def angle(a, b, c):
			'''
			Three-point angle in degrees
			Arguments:
			----------
				p1: np.ndarray - first point
				p2: np.ndarray - vertex point
				p3: np.ndarray - third point
			Returns:
			--------
				float: angle p1-p2-p3 in degrees
			'''
			v1 = a - b; v2 = c - b
			c_val = float(np.dot(v1, v2) / max(
				np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))
			return math.degrees(math.acos(max(-1, min(1, c_val))))
		cys = []
		for ri, info in aas.items():
			tri = info[5] if len(info) >= 6 else None
			if tri != 'CYS': continue
			name_to_idx = {atoms[int(a)][0]: int(a)
				for a in info[2] + info[3]}
			if all(nm in name_to_idx
					for nm in ('SG', 'CB', 'CA')):
				cys.append((name_to_idx['SG'],
					name_to_idx['CB'], name_to_idx['CA']))
		PI = math.pi
		for i in range(len(cys)):
			for j in range(i + 1, len(cys)):
				sg1, cb1, ca1 = cys[i]
				sg2, cb2, ca2 = cys[j]
				ssdist = float(np.linalg.norm(
					coords[sg1] - coords[sg2]))
				if ssdist > 3.0: continue
				score = -shift
				z = (ssdist - d_location) / d_scale
				score_d = (z * z / 2.0
					- math.log(erfc(-d_shape * z / math.sqrt(2.0))
						+ mest))
				score += wt_len * score_d
				csang1 = angle(coords[cb1], coords[sg1], coords[sg2])
				csang2 = angle(coords[cb2], coords[sg2], coords[sg1])
				score += wt_ang * (-a_logA - a_kappa
					* math.cos(PI / 180 * (csang1 - a_mu)))
				score += wt_ang * (-a_logA - a_kappa
					* math.cos(PI / 180 * (csang2 - a_mu)))
				ss_dih = dihedral(coords[cb1], coords[sg1],
					coords[sg2], coords[cb2])
				e1 = (math.exp(dss_logA1) * math.exp(dss_kappa1
					* math.cos(PI/180 * (ss_dih - dss_mu1))))
				e2 = (math.exp(dss_logA2) * math.exp(dss_kappa2
					* math.cos(PI/180 * (ss_dih - dss_mu2))))
				score += wt_dihSS * (-math.log(e1 + e2 + mest))
				for ca_, cb_, sg_, sgo_ in (
						(ca1, cb1, sg1, sg2),
						(ca2, cb2, sg2, sg1)):
					ang = dihedral(coords[ca_], coords[cb_],
						coords[sg_], coords[sgo_])
					e1 = (math.exp(dcs_logA1) * math.exp(
						dcs_kappa1 * math.cos(
							PI/180 * (ang - dcs_mu1))))
					e2 = (math.exp(dcs_logA2) * math.exp(
						dcs_kappa2 * math.cos(
							PI/180 * (ang - dcs_mu2))))
					e3 = (math.exp(dcs_logA3) * math.exp(
						dcs_kappa3 * math.cos(
							PI/180 * (ang - dcs_mu3))))
					score += wt_dihCS * (-math.log(
						e1 + e2 + e3 + mest))
				raw += score
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * weight,
			'raw': raw}
	def YhhPlanarityPotential(self, pose, cache, ligand=None, **kw):
		'''
		yhh_planarity - Tyr hydroxyl planarity, 0.5*(cos(pi-2*chi3)+1)
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		weight = float(self.Parameters['YhhPlanarity']['weight'])
		aas = pose.data.get('Amino Acids') or {}
		atoms = pose.data['Atoms']
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		raw = 0.0
		for ri, info in aas.items():
			tri = info[5] if len(info) >= 6 else None
			if tri != 'TYR': continue
			name_to_idx = {atoms[int(a)][0]: int(a)
				for a in info[2] + info[3]}
			needed = ('CE2', 'CZ', 'OH', 'HH')
			if not all(nm in name_to_idx for nm in needed): continue
			a, b, c, d = (coords[name_to_idx[nm]] for nm in needed)
			b1 = b - a; b2 = c - b; b3 = d - c
			n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
			m1 = np.cross(n1, b2 / np.linalg.norm(b2))
			x = float(np.dot(n1, n2))
			y = float(np.dot(m1, n2))
			chi3 = np.arctan2(y, x)
			raw += 0.5 * (np.cos(np.pi - 2 * chi3) + 1.0)
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * weight,
			'raw': raw}
	def RefPotential(self, pose, cache, ligand=None, **kw):
		'''
		Ref - per-amino-acid unfolded reference energy
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		weight = float(self.Parameters['Ref']['weight'])
		refs = self.Parameters.get('METHOD_WEIGHTS_ref', [])
		order = ['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS',
			'LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL',
			'TRP','TYR']
		ref_by_tri = {tri: float(refs[i])
			for i, tri in enumerate(order) if i < len(refs)}
		aas = pose.data.get('Amino Acids') or {}
		raw = 0.0
		for ri, info in aas.items():
			tri = info[5] if len(info) >= 6 else None
			if tri == 'HIS_D': tri = 'HIS'
			raw += ref_by_tri.get(tri, 0.0)
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * weight,
			'raw': raw}
	def DefaultOffsetPotential(self, pose, cache, ligand=None, **kw):
		'''
		Default smoke-test calibration term. Returns per_residue x N
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		weight = float(self.Parameters.get(
			'DefaultOffset', {}).get('weight', 0.0))
		per_res = float(self.Parameters.get(
			'Constants', {}).get('per_residue', 0.0))
		aas = pose.data.get('Amino Acids') or {}
		raw = float(len(aas)) * per_res
		return {'inter_raw': raw, 'intra_raw': 0.0,
			'inter_weighted': raw * weight, 'intra_weighted': 0.0,
			'raw': raw}
	def HBondSrBbPotential(self, pose, cache, ligand=None, **kw):
		'''
		Hbond_sr_bb: short-range bb-bb hbonds
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		raw = cache['fullatomhbond'](pose, cache)['SR_BB']
		w = float(self.Parameters['HBondSrBb']['weight'])
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * w,
			'raw': raw}
	def HBondLrBbPotential(self, pose, cache, ligand=None, **kw):
		'''
		Hbond_lr_bb: long-range bb-bb hbonds
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		raw = cache['fullatomhbond'](pose, cache)['LR_BB']
		w = float(self.Parameters['HBondLrBb']['weight'])
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * w,
			'raw': raw}
	def HBondBbScPotential(self, pose, cache, ligand=None, **kw):
		'''
		Hbond_bb_sc: backbone-sidechain hbonds
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		raw = cache['fullatomhbond'](pose, cache)['BB_SC']
		w = float(self.Parameters['HBondBbSc']['weight'])
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * w,
			'raw': raw}
	def HBondScPotential(self, pose, cache, ligand=None, **kw):
		'''
		Hbond_sc: sidechain-sidechain hbonds
		Arguments:
		----------
			pose:   Pose or Molecule - receptor structure being scored
			cache:  dict - cache returned by ScoreMatch()
			ligand: Molecule or None - optional small-molecule ligand
			**kw:   absorbed; per-term methods take no extra kwargs
		Returns:
		--------
			dict: per-term contribution with keys 'inter_raw', 'intra_raw',
			      'inter_weighted', 'intra_weighted' (plus 'raw' for full-atom
			      terms that decompose intra vs inter)
		'''
		raw = cache['fullatomhbond'](pose, cache)['SC']
		w = float(self.Parameters['HBondSc']['weight'])
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * w,
			'raw': raw}
