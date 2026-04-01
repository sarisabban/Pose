#!/usr/bin/env python3

import re
import json
import numpy as np

np.seterr(all='ignore')

class PoseM():
	''' Data structure that represents a molecule '''
	def __init__(self):
		self.data = {
			'Energy':0,
			'Rg':0,
			'Mass':0,
			'SMILES':None,
			'Atoms':{},
			'Bonds':{},
			'Coordinates':np.array([[0, 0, 0]])}

	# ------------------------------------------------------------------
	# FromSMILES helpers
	# ------------------------------------------------------------------

	def _tok(self, smiles):
		''' Tokenise a SMILES string into a flat list of string tokens '''
		pat = re.compile(
			r'(\[[^\[\]]*\])'             # bracket atoms [NH3+]
			r'|(Cl|Br|[BCNOPSFIbcnops])' # organic subset (Cl/Br first)
			r'|(\%\d{2}|\d)'              # ring closures %12 or 1
			r'|([-=#:\\/])'               # explicit bonds
			r'|([()])'                    # branch open/close
		)
		return [m.group() for m in pat.finditer(smiles)]

	def _parse(self, tokens):
		'''
		Build atom list and bond list from token list.
		Returns (atoms, bonds) where
		  atoms = list of dicts
		  bonds = list of (i, j, order) tuples
		'''
		atoms = []
		bonds = []
		stack = []        # atom indices for branch tracking
		ring_opens = {}   # digit/key -> (atom_idx, bond_order)
		prev_idx  = None
		next_bond = None  # bond order for the next bond to be created

		def parse_bracket(tok):
			# e.g. [C@@H2+], [NH3+], [nH], [2H]
			inner = tok[1:-1]
			# isotope
			m = re.match(r'^(\d+)', inner)
			if m:
				inner = inner[m.end():]
			# element
			em = re.match(r'^([A-Z][a-z]?)', inner)
			if not em:
				em = re.match(r'^([a-z])', inner)
			element = em.group(1).capitalize()
			aromatic = em.group(1).islower()
			inner = inner[em.end():]
			# chirality
			chirality = None
			if inner.startswith('@@'):
				chirality = '@@'; inner = inner[2:]
			elif inner.startswith('@'):
				chirality = '@'; inner = inner[1:]
			# H count
			hcount = 0
			hm = re.match(r'^H(\d*)', inner)
			if hm:
				hcount = int(hm.group(1)) if hm.group(1) else 1
				inner = inner[hm.end():]
			# charge
			charge = 0
			cm = re.match(r'^([+\-])(\d*)', inner)
			if cm:
				sign = 1 if cm.group(1) == '+' else -1
				mag  = int(cm.group(2)) if cm.group(2) else 1
				charge = sign * mag
			return {
				'element':   element,
				'aromatic':  aromatic,
				'chirality': chirality,
				'hcount':    hcount,   # explicit
				'charge':    charge,
				'bracket':   True,
			}

		ORGANIC = {
			'B','C','N','O','P','S','F','I',
			'Cl','Br',
			'b','c','n','o','p','s',
		}

		for tok in tokens:
			# branch open
			if tok == '(':
				stack.append(prev_idx)
				next_bond = None
				continue
			# branch close
			if tok == ')':
				prev_idx  = stack.pop()
				next_bond = None
				continue
			# explicit bond
			if tok in ('-', '=', '#', ':', '/', '\\'):
				order_map = {'-':1, '=':2, '#':3, ':':1.5, '/':1, '\\':1}
				next_bond = order_map[tok]
				continue
			# ring closure digit
			if re.fullmatch(r'\d|\%\d{2}', tok):
				key = tok
				bo  = next_bond if next_bond is not None else 1
				next_bond = None
				if key in ring_opens:
					i, bo0 = ring_opens.pop(key)
					# use whichever bond order is explicit
					final_bo = bo if bo != 1 else bo0
					bonds.append((i, prev_idx, final_bo))
				else:
					ring_opens[key] = (prev_idx, bo)
				continue
			# atom token
			if tok.startswith('['):
				props = parse_bracket(tok)
			elif tok in ORGANIC or tok.capitalize() in ORGANIC:
				element  = tok.capitalize()
				aromatic = tok.islower()
				props = {
					'element':   element,
					'aromatic':  aromatic,
					'chirality': None,
					'hcount':    -1,   # implicit
					'charge':    0,
					'bracket':   False,
				}
			else:
				continue  # unknown token, skip

			idx = len(atoms)
			props['idx'] = idx
			atoms.append(props)

			if prev_idx is not None:
				bo = next_bond
				if bo is None:
					# aromatic-aromatic default 1.5
					if (atoms[prev_idx]['aromatic']
							and props['aromatic']):
						bo = 1.5
					else:
						bo = 1
				bonds.append((prev_idx, idx, bo))

			next_bond = None
			prev_idx  = idx

		return atoms, bonds

	def _add_H(self, atoms, bonds):
		'''
		Add explicit H atoms for implicit valence.
		Returns updated (atoms, bonds).
		'''
		VALENCE = {
			'B':3, 'C':4, 'N':3, 'O':2, 'P':3, 'S':2,
			'F':1, 'Cl':1, 'Br':1, 'I':1,
		}
		# sum bond orders per heavy atom
		bond_sum = {a['idx']: 0.0 for a in atoms}
		for i, j, bo in bonds:
			bond_sum[i] += bo
			bond_sum[j] += bo

		new_atoms = list(atoms)
		new_bonds = list(bonds)

		for a in atoms:
			if a['element'] == 'H':
				continue
			if a['bracket']:
				h = a['hcount']
			else:
				val = VALENCE.get(a['element'], 0)
				h   = max(0, int(val - bond_sum[a['idx']]))

			for _ in range(h):
				hidx = len(new_atoms)
				new_atoms.append({
					'idx':       hidx,
					'element':   'H',
					'aromatic':  False,
					'chirality': None,
					'hcount':    0,
					'charge':    0,
					'bracket':   True,
				})
				new_bonds.append((a['idx'], hidx, 1))

		return new_atoms, new_bonds

	def _hybridize(self, atoms, bonds):
		''' Assign sp / sp2 / sp3 hybridisation to each atom in-place '''
		max_bo  = {}
		has_dbl = {}
		has_tri = {}
		for a in atoms:
			max_bo[a['idx']]  = 0
			has_dbl[a['idx']] = False
			has_tri[a['idx']] = False

		for i, j, bo in bonds:
			for idx in (i, j):
				if bo > max_bo[idx]:
					max_bo[idx] = bo
			if bo == 2:
				has_dbl[i] = has_dbl[j] = True
			if bo == 3:
				has_tri[i] = has_tri[j] = True

		for a in atoms:
			idx = a['idx']
			if has_tri[idx]:
				a['hyb'] = 'sp'
			elif has_dbl[idx] or a['aromatic']:
				a['hyb'] = 'sp2'
			else:
				a['hyb'] = 'sp3'

	def _rings(self, atoms, bonds):
		'''
		Find SSSR via DFS on heavy atoms.
		Returns list of rings, each a list of atom indices.
		'''
		# build heavy-atom adjacency
		heavy = {a['idx'] for a in atoms if a['element'] != 'H'}
		adj   = {i: [] for i in heavy}
		for i, j, _ in bonds:
			if i in heavy and j in heavy:
				adj[i].append(j)
				adj[j].append(i)

		found_rings = []
		visited     = {}  # idx -> parent idx
		on_stack    = set()

		def dfs(node, parent, path):
			visited[node] = parent
			on_stack.add(node)
			path.append(node)
			for nb in adj[node]:
				if nb == parent:
					continue
				if nb in on_stack:
					# found a back edge: extract cycle
					cycle_start = path.index(nb)
					ring = path[cycle_start:]
					found_rings.append(list(ring))
				elif nb not in visited:
					dfs(nb, node, path)
			path.pop()
			on_stack.discard(node)

		for start in heavy:
			if start not in visited:
				dfs(start, -1, [])

		# deduplicate and keep smallest rings
		unique = []
		seen   = set()
		for r in sorted(found_rings, key=len):
			key = frozenset(r)
			if key not in seen:
				seen.add(key)
				unique.append(r)

		return unique

	def _bond_len(self, e1, e2, order):
		''' Look up idealised bond length in Angstroms '''
		BL = {
			('C','C',1):1.54, ('C','C',2):1.34,
			('C','C',3):1.20, ('C','C',1.5):1.40,
			('C','N',1):1.47, ('C','N',2):1.27,
			('C','N',3):1.15, ('C','N',1.5):1.34,
			('C','O',1):1.43, ('C','O',2):1.22,
			('C','O',1.5):1.36,
			('C','S',1):1.82, ('C','S',2):1.60,
			('C','H',1):1.09,
			('N','H',1):1.01,
			('N','N',1):1.45, ('N','N',2):1.25,
			('N','O',1):1.40, ('N','O',2):1.21,
			('O','H',1):0.96,
			('O','O',1):1.48,
			('S','H',1):1.34,
			('S','S',1):2.05,
			('C','F',1):1.35,
			('C','Cl',1):1.77,
			('C','Br',1):1.94,
			('C','I',1):2.14,
			('C','P',1):1.84,
			('P','O',1):1.63, ('P','O',2):1.48,
		}
		key1 = (e1, e2, order)
		key2 = (e2, e1, order)
		return BL.get(key1, BL.get(key2, 1.50))

	def _place(self, C, B, A, length, angle, dihedral):
		'''
		Z-matrix placement: place atom D bonded to C.
		C = parent position
		B = grandparent position
		A = great-grandparent position
		angle    in radians (B-C-D)
		dihedral in radians (A-B-C-D)
		'''
		def norm(v):
			n = np.linalg.norm(v)
			if n < 1e-10:
				return v
			return v / n

		bc = norm(C - B)
		ab = norm(B - A)
		n  = np.cross(ab, bc)
		if np.linalg.norm(n) < 1e-10:
			perp = (np.array([1.,0.,0.])
				if abs(bc[2]) < 0.9
				else np.array([0.,0.,1.]))
			n = np.cross(bc, perp)
		n        = norm(n)
		in_plane = norm(np.cross(n, bc))

		D = (C
			+ length * (
				-np.cos(angle) * bc
				+ np.sin(angle) * (
					np.cos(dihedral) * in_plane
					+ np.sin(dihedral) * n
				)
			))
		return D

	def _coords(self, atoms, bonds, rings):
		'''
		Generate 3D coordinates for all atoms.
		Returns list of np.array([x,y,z]) indexed by atom idx.
		'''
		N     = len(atoms)
		coords = [None] * N
		placed = [False] * N

		# index lookups
		a_by_idx = {a['idx']: a for a in atoms}

		# adjacency for all atoms
		adj = {a['idx']: [] for a in atoms}
		bond_order = {}
		for i, j, bo in bonds:
			adj[i].append(j)
			adj[j].append(i)
			bond_order[(i,j)] = bo
			bond_order[(j,i)] = bo

		def get_bo(i, j):
			return bond_order.get((i,j), 1)

		def bl(i, j):
			return self._bond_len(
				a_by_idx[i]['element'],
				a_by_idx[j]['element'],
				get_bo(i, j))

		def hyb_angle(idx):
			h = a_by_idx[idx]['hyb']
			if h == 'sp':   return np.radians(180.)
			if h == 'sp2':  return np.radians(120.)
			return np.radians(109.5)

		# ---- Phase 1: place ring atoms --------------------------------
		placed_ring_sets = []
		for ring in rings:
			n_ring = len(ring)
			# check if any atom already placed (fused ring)
			shared = [(k, ring[k])
				for k in range(n_ring) if placed[ring[k]]]

			if not shared:
				# first ring: regular polygon in XY plane
				l_avg = np.mean([
					bl(ring[k], ring[(k+1) % n_ring])
					for k in range(n_ring)])
				R = l_avg / (2 * np.sin(np.pi / n_ring))
				for k, aidx in enumerate(ring):
					angle_k = 2 * np.pi * k / n_ring
					coords[aidx] = np.array([
						R * np.cos(angle_k),
						R * np.sin(angle_k),
						0.0])
					placed[aidx] = True
			else:
				# fused ring: find two adjacent already-placed atoms
				# that are in this ring
				edge = None
				for k in range(n_ring):
					a0 = ring[k]
					a1 = ring[(k+1) % n_ring]
					if placed[a0] and placed[a1]:
						edge = (k, a0, a1)
						break

				if edge is None:
					# just one anchor atom; fall through to BFS
					continue

				k0, a0, a1 = edge
				# place remaining atoms of the ring as a regular
				# polygon that shares the a0-a1 edge
				unplaced_ring = [
					ring[(k0+2+i) % n_ring]
					for i in range(n_ring - 2)]

				# direction of shared bond
				p0 = coords[a0]
				p1 = coords[a1]
				bond_vec = p1 - p0
				bond_len_shared = np.linalg.norm(bond_vec)
				bond_dir = bond_vec / bond_len_shared

				# normal to existing ring plane (use z-axis if flat)
				# determine plane normal from existing placed ring
				existing = [
					coords[ring[k]]
					for k in range(n_ring)
					if placed[ring[k]]]
				if len(existing) >= 3:
					v1 = existing[1] - existing[0]
					v2 = existing[2] - existing[0]
					plane_n = np.cross(v1, v2)
					pn_norm = np.linalg.norm(plane_n)
					if pn_norm > 1e-10:
						plane_n /= pn_norm
					else:
						plane_n = np.array([0., 0., 1.])
				else:
					plane_n = np.array([0., 0., 1.])

				# for a regular n_ring polygon sharing one edge:
				# interior angle = (n-2)*pi/n
				# circumradius R from chord = bond_len_shared
				#   chord = 2R sin(pi/n)
				l_avg = np.mean([
					bl(ring[k], ring[(k+1) % n_ring])
					for k in range(n_ring)])
				R = l_avg / (2 * np.sin(np.pi / n_ring))

				# midpoint of shared edge
				mid = (p0 + p1) * 0.5

				# perpendicular to bond_dir in the ring plane,
				# pointing AWAY from existing ring centroid
				existing_cent = np.mean(existing, axis=0)
				perp_in_plane = np.cross(plane_n, bond_dir)
				perp_in_plane /= (
					np.linalg.norm(perp_in_plane) + 1e-12)
				# flip if pointing toward existing centroid
				if np.dot(perp_in_plane, mid - existing_cent) > 0:
					perp_in_plane = -perp_in_plane

				# height of regular polygon from mid-edge to center
				h_poly = np.sqrt(
					max(0, R**2 - (bond_len_shared/2)**2))
				new_cent = mid + h_poly * perp_in_plane

				# place unplaced atoms around new_cent
				# angle for atom a0 in the new polygon
				ang0 = np.arctan2(
					np.dot(p0 - new_cent, perp_in_plane),
					np.dot(p0 - new_cent, bond_dir))
				ang1 = np.arctan2(
					np.dot(p1 - new_cent, perp_in_plane),
					np.dot(p1 - new_cent, bond_dir))
				# angles for remaining atoms continue from a1
				step = 2 * np.pi / n_ring
				for i, aidx in enumerate(unplaced_ring):
					ang = ang1 + step * (i + 1)
					coords[aidx] = (
						new_cent
						+ R * np.cos(ang) * bond_dir
						+ R * np.sin(ang) * perp_in_plane)
					placed[aidx] = True

		# ---- Phase 2: BFS spanning tree (heavy atoms first) -----------
		heavy_idx = [a['idx'] for a in atoms
			if a['element'] != 'H']
		H_idx     = [a['idx'] for a in atoms
			if a['element'] == 'H']

		# choose start atom
		start = None
		for idx in heavy_idx:
			if placed[idx]:
				start = idx
				break
		if start is None and heavy_idx:
			start = heavy_idx[0]
			coords[start] = np.array([0., 0., 0.])
			placed[start] = True

		# BFS parent tracking
		parent = {start: None}
		queue  = [start]
		q_head = 0

		while q_head < len(queue):
			cur = queue[q_head]; q_head += 1
			# collect already-placed and unplaced heavy neighbours
			unplaced_nb = [nb for nb in adj[cur]
				if not placed[nb]
				and a_by_idx[nb]['element'] != 'H']
			placed_nb   = [nb for nb in adj[cur]
				if placed[nb]]

			C = coords[cur]

			# get B (parent of cur)
			par = parent.get(cur)
			if par is not None:
				B = coords[par]
				gpar = parent.get(par)
				A = coords[gpar] if gpar is not None else (
					B + np.array([0., 1., 0.]))
			else:
				# cur is the root
				B = C + np.array([-1., 0., 0.])
				A = B + np.array([0., 1., 0.])

			ang = hyb_angle(cur)

			# dihedral offsets for multiple substituents
			# sp3: 180, 60, -60   sp2: 0, 180   sp: 0
			hyb = a_by_idx[cur]['hyb']
			if hyb == 'sp3':
				base_dihedrals = [
					np.radians(180.),
					np.radians(60.),
					np.radians(-60.),
					np.radians(-180.),
				]
			elif hyb == 'sp2':
				base_dihedrals = [
					np.radians(180.),
					np.radians(0.),
				]
			else:
				base_dihedrals = [np.radians(180.)]

			slot = 0
			for nb in unplaced_nb:
				d = base_dihedrals[slot % len(base_dihedrals)]
				length = bl(cur, nb)
				coords[nb] = self._place(C, B, A, length, ang, d)
				placed[nb] = True
				parent[nb] = cur
				queue.append(nb)
				slot += 1

		# ---- Phase 2b: BFS for remaining disconnected heavy atoms -----
		for idx in heavy_idx:
			if not placed[idx]:
				coords[idx] = np.array([0., 0., 0.])
				placed[idx] = True
				parent[idx] = None
				queue_loc   = [idx]
				qh          = 0
				while qh < len(queue_loc):
					cur = queue_loc[qh]; qh += 1
					unp = [nb for nb in adj[cur]
						if not placed[nb]
						and a_by_idx[nb]['element'] != 'H']
					C   = coords[cur]
					par = parent.get(cur)
					B   = coords[par] if par else C + np.array([-1.,0.,0.])
					gp  = parent.get(par) if par else None
					A   = coords[gp] if gp else B + np.array([0.,1.,0.])
					ang = hyb_angle(cur)
					hyb = a_by_idx[cur]['hyb']
					dihs = ([np.radians(180.),
						np.radians(60.), np.radians(-60.)]
						if hyb == 'sp3'
						else [np.radians(180.), np.radians(0.)])
					for sl, nb in enumerate(unp):
						d = dihs[sl % len(dihs)]
						coords[nb] = self._place(
							C, B, A, bl(cur, nb), ang, d)
						placed[nb] = True
						parent[nb] = cur
						queue_loc.append(nb)

		# ---- Phase 3: place hydrogen atoms ----------------------------
		for hidx in H_idx:
			heavy_parent = adj[hidx]
			if not heavy_parent:
				coords[hidx] = np.array([0., 0., 0.])
				placed[hidx] = True
				continue
			par = heavy_parent[0]
			C   = coords[par]
			# collect already-placed heavy neighbours of par
			# to pick a sensible dihedral
			placed_heavy_nb = [
				nb for nb in adj[par]
				if placed[nb] and nb != hidx
				and a_by_idx[nb]['element'] != 'H']
			placed_H_nb = [
				nb for nb in adj[par]
				if placed[nb] and nb != hidx
				and a_by_idx[nb]['element'] == 'H']

			# count how many H already placed on this parent
			h_done = len(placed_H_nb)
			hyb    = a_by_idx[par]['hyb']

			if placed_heavy_nb:
				B  = coords[placed_heavy_nb[0]]
				gp = (placed_heavy_nb[1]
					if len(placed_heavy_nb) > 1
					else None)
				if gp is None:
					# look one level up
					pp = [nb for nb in adj[placed_heavy_nb[0]]
						if placed[nb]
						and nb != par
						and a_by_idx[nb]['element'] != 'H']
					gp_idx = pp[0] if pp else None
					A = coords[gp_idx] if gp_idx else (
						B + np.array([0., 1., 0.]))
				else:
					A = coords[gp]
			else:
				B = C + np.array([-1., 0., 0.])
				A = B + np.array([0.,  1., 0.])

			if hyb == 'sp3':
				h_dihs = [
					np.radians(180.),
					np.radians(60.),
					np.radians(-60.),
					np.radians(-180.),
				]
			elif hyb == 'sp2':
				h_dihs = [np.radians(180.), np.radians(0.)]
			else:
				h_dihs = [np.radians(180.)]

			d_idx = h_done % len(h_dihs)
			ang   = hyb_angle(par)
			length = self._bond_len(
				a_by_idx[par]['element'], 'H', 1)
			coords[hidx] = self._place(
				C, B, A, length, ang, h_dihs[d_idx])
			placed[hidx] = True

		# fill any remaining None with origin
		for i in range(N):
			if coords[i] is None:
				coords[i] = np.array([0., 0., 0.])

		return coords

	# ------------------------------------------------------------------
	# Public method
	# ------------------------------------------------------------------

	def FromSMILES(self, smiles):
		''' Convert a SMILES string into a 3D molecule '''
		tokens       = self._tok(smiles)
		atoms, bonds = self._parse(tokens)
		atoms, bonds = self._add_H(atoms, bonds)
		self._hybridize(atoms, bonds)
		rings        = self._rings(atoms, bonds)
		coords       = self._coords(atoms, bonds, rings)

		adj = {a['idx']: [] for a in atoms}
		for i, j, _ in bonds:
			adj[i].append(j)
			adj[j].append(i)

		self.data['SMILES']      = smiles
		self.data['Atoms']       = {
			a['idx']: [
				f'{a["element"]}{a["idx"]}',
				a['element']
			]
			for a in atoms
		}
		self.data['Bonds']       = adj
		self.data['Coordinates'] = np.array(coords)
