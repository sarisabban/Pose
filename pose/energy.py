import os
import sys
import json
import math
import copy
import base64
import warnings
import numpy as np
from .pose import DBLoad

def SMIRKSMatch(pose, params):
	'''
	Assign Sage 2.3.0 parameters to a pose by SMIRKS pattern matching
	Arguments:
	----------
		pose:   Pose - molecule, protein, DNA, or RNA pose
		params: dict - SMIRKS-keyed force-field section dict containing
			Constraints/Bonds/Angles/ProperTorsions/ImproperTorsions/vdW/
			LibraryCharges keys (typically ForceField.mol)
	Returns:
	--------
		dict: keyed assignments consumed by ForceField._compile:
			'bonds':         {(i, j):       [length, k]} per pair
			'angles':        {(i, j, k):    [angle, k]} per triplet
				(matched against the broad Sage Angles section)
			'ub':            {(i, j, k):    [s0, k_ub]} per angle triplet
				(matched against the separate, narrow UB section)
			'propers':       {(i, j, k, l): [[period, phase, k, idivf], ...]}
			'impropers':     list of (i, j, k, l, period, phase, k_eff)
			'vdw':           {i: [epsilon, sigma]} (rmin_half pre-converted)
			'polarisation':  {i: alpha} per atom
			'charges':       {i: charge or None} (None = Gasteiger fallback)
	'''
	Z_TABLE = {
		'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
		'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
		'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
		'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,
		'Br':35,'Kr':36,'Rb':37,'Sr':38,'I':53,'Xe':54,'Cs':55,'Ba':56}
	atoms = pose.data['Atoms']
	bonds_dict = pose.data['Bonds']
	bond_orders = pose.data.get('BondOrders', {}) or {}
	formal_charges = getattr(pose, '_formal_charges', {}) or {}
	sorted_ids = sorted(atoms.keys())
	nbr = {i: [] for i in sorted_ids}
	for i in sorted_ids:
		for j in bonds_dict.get(i, []):
			if j in atoms and j != i and j not in nbr[i]:
				nbr[i].append(j)
	edges = set()
	for i in sorted_ids:
		for j in nbr[i]:
			edges.add((min(i, j), max(i, j)))
	edges = sorted(edges)
	edge_set = set(edges)
	bo = {}
	for i in sorted_ids:
		bos = bond_orders.get(i, [])
		js = bonds_dict.get(i, [])
		for k, j in enumerate(js):
			if j not in atoms or j == i: continue
			b = bos[k] if k < len(bos) else 1.0
			bo[(min(i, j), max(i, j))] = float(b)
	# Per-atom: atomic number, connectivity X, H count, formal charge
	Z = {i: Z_TABLE.get(atoms[i][1].capitalize(), 0) for i in sorted_ids}
	X = {i: len(nbr[i]) for i in sorted_ids}
	Hc = {i: sum(1 for j in nbr[i] if atoms[j][1] == 'H')
		for i in sorted_ids}
	fc = {i: int(formal_charges.get(i, 0)) for i in sorted_ids}
	# is_arom_bond initial state from raw bond orders; updated post-Kekulisation
	is_arom_bond = {e: (abs(bo.get(e, 1.0) - 1.5) < 1e-6) for e in edges}
	is_arom_atom = {i: any(is_arom_bond.get((min(i, j), max(i, j)), False)
		for j in nbr[i]) for i in sorted_ids}
	def find_rings():
		'''
		Smallest set of smallest rings via per-edge BFS shortest cycle
		Arguments:
		----------
			No arguments taken (closes over nbr, sorted_ids)
		Returns:
		--------
			list: each ring as a tuple of atom indices (closed cycle)
		'''
		rings_seen = set()
		out = []
		for u, v in edges:
			parent = {u: None}
			q = [u]
			while q:
				nq = []
				for x in q:
					for y in nbr[x]:
						if (min(x, y), max(x, y)) == (u, v): continue
						if y in parent: continue
						parent[y] = x
						if y == v:
							q = []
							break
						nq.append(y)
					if not q: break
				q = nq
			if v not in parent: continue
			path = [v]
			cur = v
			while parent[cur] is not None:
				cur = parent[cur]
				path.append(cur)
			ring = tuple(path)
			# Canonicalise: rotate so smallest atom is first, pick lex-min orientation
			mn = min(ring)
			i0 = ring.index(mn)
			rotated = ring[i0:] + ring[:i0]
			fwd = rotated
			rev = (rotated[0],) + rotated[:0:-1]
			canon = min(fwd, rev)
			if canon in rings_seen: continue
			rings_seen.add(canon)
			out.append(canon)
		return out
	rings = find_rings()
	def hyb_of(rec):
		'''
		Hybridisation tag from an atom record, defaulting to sp3
		Arguments:
		----------
			rec: atom record from self.data['Atoms'] (list)
		Returns:
		--------
			str: hybridisation tag at rec[-1], or 'sp3' when rec is empty
		'''
		return rec[-1] if rec else 'sp3'
	def kekulise():
		'''
		Find a valid Kekulé assignment for all 1.5-order bonds via
		constraint propagation and DFS backtracking. Operates per
		connected-component of candidate bonds — solvable components
		(amides, aromatic rings) get a chemically valid assignment;
		unsolvable ones (carboxylate / guanidinium without explicit
		formal charges) fall back to "non-ring 1.5 -> 1.0" heuristic
		for that component only.
		Arguments:
		----------
			No arguments taken (closes over bo, edges, nbr, atoms, fc)
		Returns:
		--------
			None: bo is mutated in place
		'''
		VAL = {'C':4,'N':3,'O':2,'S':2,'P':5,'Se':2,
			'F':1,'Cl':1,'Br':1,'I':1,'H':1,'B':3}
		all_candidates = sorted(e for e in edges
			if abs(bo.get(e, 1.0) - 1.5) < 1e-6)
		if not all_candidates: return
		all_cand_set = set(all_candidates)
		# Group candidates into connected components (sharing atoms)
		atom_to_cands = {}
		for e in all_candidates:
			atom_to_cands.setdefault(e[0], []).append(e)
			atom_to_cands.setdefault(e[1], []).append(e)
		seen_edges = set()
		components = []
		for start in all_candidates:
			if start in seen_edges: continue
			comp = []; queue = [start]; seen_edges.add(start)
			while queue:
				e = queue.pop()
				comp.append(e)
				for atom in (e[0], e[1]):
					for nbr_e in atom_to_cands.get(atom, []):
						if nbr_e in seen_edges: continue
						seen_edges.add(nbr_e); queue.append(nbr_e)
			components.append(sorted(comp))
		# Mark in-ring candidates so the heuristic fallback knows what to skip
		in_ring_cand = set()
		for r in rings:
			L = len(r)
			for k in range(L):
				a, b = r[k], r[(k + 1) % L]
				e = (min(a, b), max(a, b))
				if e in all_cand_set: in_ring_cand.add(e)
		for comp in components:
			cand_set = set(comp)
			touched = set()
			for (a, b) in comp:
				touched.add(a); touched.add(b)
			budget = {}
			atom_cands = {a: [] for a in touched}
			for ci, e in enumerate(comp):
				atom_cands[e[0]].append(ci)
				atom_cands[e[1]].append(ci)
			ok = True
			for a in touched:
				elem = atoms[a][1]
				if elem not in VAL: ok = False; break
				v = VAL[elem] + fc.get(a, 0)
				for j in nbr[a]:
					e = (min(a, j), max(a, j))
					if e in cand_set: continue
					v -= bo.get(e, 1.0)
				n_cands = len(atom_cands[a])
				v -= n_cands
				budget[a] = int(round(v))
				if budget[a] < 0 or budget[a] > n_cands:
					ok = False; break
			if ok:
				assn = [-1] * len(comp)
				def doubles_seen(a, comp_assn):
					'''
					Count how many candidate edges for atom a are already assigned bond order 2
					Arguments:
					----------
						a: int - atom index
						comp_assn: list - per-edge tentative bond-order assignment
					Returns:
					--------
						int: number of candidate edges currently at order 2
					'''
					return sum(1 for ci in atom_cands[a] if comp_assn[ci] == 2)
				def remaining(a, comp_assn):
					'''
					List candidate edges for atom a that are still unassigned
					Arguments:
					----------
						a: int - atom index
						comp_assn: list - per-edge tentative bond-order assignment
					Returns:
					--------
						list of int: indices of candidate edges still unassigned for atom a
					'''
					return [ci for ci in atom_cands[a] if comp_assn[ci] == -1]
				def propagate(comp_assn):
					'''
					Force-propagate edges whose budget is fully determined; return False on contradiction
					Arguments:
					----------
						comp_assn: list - per-edge tentative bond-order assignment
					Returns:
					--------
						bool: True if propagation completed without contradiction
					'''
					changed = True
					while changed:
						changed = False
						for a in touched:
							rem = remaining(a, comp_assn)
							need = budget[a] - doubles_seen(a, comp_assn)
							if need < 0 or need > len(rem): return False
							if need == 0 and rem:
								for ci in rem: comp_assn[ci] = 1
								changed = True
							elif need == len(rem) and rem:
								for ci in rem: comp_assn[ci] = 2
								changed = True
					return True
				def dfs(comp_assn):
					'''
					Recursive DFS over edge assignments to find a consistent Kekule bond-order solution
					Arguments:
					----------
						comp_assn: list - per-edge tentative bond-order assignment
					Returns:
					--------
						bool: True if a complete assignment was found
					'''
					if not propagate(comp_assn): return False
					una = [ci for ci in range(len(comp)) if comp_assn[ci] == -1]
					if not una: return True
					ci = una[0]
					for v in (2, 1):
						saved = list(comp_assn)
						comp_assn[ci] = v
						if dfs(comp_assn): return True
						for k in range(len(comp_assn)): comp_assn[k] = saved[k]
					return False
				if dfs(assn):
					for ci, e in enumerate(comp):
						bo[e] = 2.0 if assn[ci] == 2 else 1.0
					continue
			# Fallback: non-ring 1.5 -> 1.0; ring 1.5 handled by aromatise_rings
			for e in comp:
				if e not in in_ring_cand:
					bo[e] = 1.0
	def aromatise_rings():
		'''
		Re-mark Kekulé aromatic ring bonds as bo=1.5 for SMIRKS ':' matching
		Arguments:
		----------
			No arguments taken (closes over rings, atoms, bo)
		Returns:
		--------
			None: bo is mutated in place
		'''
		for r in rings:
			L = len(r)
			if L not in (5, 6): continue
			if not all(hyb_of(atoms[a]) == 'sp2' for a in r): continue
			ring_edges = [(min(r[k], r[(k+1) % L]),
				max(r[k], r[(k+1) % L])) for k in range(L)]
			has_pi = any(abs(bo.get(e, 1.0) - 2.0) < 0.1
				or abs(bo.get(e, 1.0) - 1.5) < 0.1 for e in ring_edges)
			if not has_pi: continue
			for e in ring_edges:
				bo[e] = 1.5
				is_arom_bond[e] = True
	kekulise()
	aromatise_rings()
	# Resync after kekulise+aromatise (initial bo state is stale)
	is_arom_bond = {e: (abs(bo.get(e, 1.0) - 1.5) < 1e-6) for e in edges}
	is_arom_atom = {i: any(is_arom_bond.get((min(i, j), max(i, j)), False)
		for j in nbr[i]) for i in sorted_ids}
	# SMARTS r<n> is "smallest ring is size n"; R alone is "in any ring"
	ring_sizes_at = {i: set() for i in sorted_ids}
	for r in rings:
		for a in r: ring_sizes_at[a].add(len(r))
	min_ring_size = {i: (min(ring_sizes_at[i]) if ring_sizes_at[i] else 0)
		for i in sorted_ids}
	in_ring_bond = set()
	for r in rings:
		L = len(r)
		for k in range(L):
			a, b = r[k], r[(k + 1) % L]
			in_ring_bond.add((min(a, b), max(a, b)))
	x_count = {i: sum(1 for j in nbr[i] if ring_sizes_at[j])
		for i in sorted_ids}
	def parse(smirks):
		'''
		Parse a SMIRKS string into an internal pattern graph
		Arguments:
		----------
			smirks: str - the SMIRKS query
		Returns:
		--------
			dict: {'atoms': [...], 'bonds': [...], 'tags': {...}}
		'''
		s = smirks
		pos = [0]
		def peek(off=0):
			'''
			Peek at the character `off` positions ahead of the cursor without consuming it
			Arguments:
			----------
				off: int, default 0 - offset from cursor
			Returns:
			--------
				str: single character at cursor+off, or '' past end of input
			'''
			p = pos[0] + off
			return s[p] if p < len(s) else ''
		def take(c):
			'''
			Consume the expected character at the cursor or raise ValueError
			Arguments:
			----------
				c: str - expected single character
			Returns:
			--------
				No return value
			'''
			if peek() != c: raise ValueError(
				f'Expected {c!r} at {pos[0]} in {s!r}')
			pos[0] += 1
		def read_int():
			'''
			Consume a run of decimal digits at the cursor
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				int or None: integer value if any digits were read, else None
			'''
			start = pos[0]
			while pos[0] < len(s) and s[pos[0]].isdigit(): pos[0] += 1
			return int(s[start:pos[0]]) if pos[0] > start else None
		# atom-expr (until ']' or ':'); precedence low->high: ';' ',' '&' '!'
		def atom_expr():
			# parse low-prec AND chain
			'''
			Parse a full SMIRKS atom expression: AND-chains joined by ';' (low precedence)
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: nested ('and', 'or', 'not', primitive) AST node
			'''
			left = atom_or()
			while peek() == ';':
				pos[0] += 1
				right = atom_or()
				left = ('and', left, right)
			return left
		def atom_or():
			'''
			Parse one OR-chain of atom expressions joined by ','
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: nested AST node for the parsed expression
			'''
			left = atom_and()
			while peek() == ',':
				pos[0] += 1
				right = atom_and()
				left = ('or', left, right)
			return left
		def atom_and():
			'''
			Parse one AND-chain of atom expressions joined by '&' or by adjacency
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: nested AST node for the parsed expression
			'''
			left = atom_neg()
			while peek() not in ('', ',', ';', ']', ':'):
				if peek() == '&': pos[0] += 1
				right = atom_neg()
				left = ('and', left, right)
			return left
		def atom_neg():
			'''
			Parse an optionally-negated atom primitive: '!' prefix toggles negation
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: AST node wrapping the primitive (wrapped in 'not' if prefixed)
			'''
			if peek() == '!':
				pos[0] += 1
				return ('not', atom_neg())
			return atom_prim()
		def atom_prim():
			'''
			Parse a single atom primitive (element symbol, '#n', '@chirality', degree, charge, etc.)
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: AST node for the primitive
			'''
			c = peek()
			if c == '*':
				pos[0] += 1
				return ('wild',)
			if c == 'a':
				pos[0] += 1; return ('arom', True)
			if c == 'A':
				pos[0] += 1; return ('arom', False)
			if c == 'R':
				pos[0] += 1
				n = read_int()
				return ('Rcount', n)
			if c == 'r':
				pos[0] += 1
				n = read_int()
				return ('rsize', n)
			if c == 'X':
				pos[0] += 1; n = read_int()
				return ('X', n if n is not None else 0)
			if c == 'x':
				pos[0] += 1; n = read_int()
				return ('x', n if n is not None else 0)
			if c == 'H':
				pos[0] += 1; n = read_int()
				return ('H', 1 if n is None else n)
			if c == 'h':
				pos[0] += 1; n = read_int()
				return ('h', 1 if n is None else n)
			if c == '+':
				pos[0] += 1; n = read_int()
				return ('fc', 1 if n is None else n)
			if c == '-':
				pos[0] += 1; n = read_int()
				return ('fc', -1 if n is None else -n)
			if c == '#':
				pos[0] += 1; n = read_int()
				return ('Z', n)
			if c == '$':
				pos[0] += 1; take('(')
				# capture balanced parens forming a sub-SMIRKS
				depth = 1; start = pos[0]
				while pos[0] < len(s) and depth:
					ch = s[pos[0]]
					if ch == '(': depth += 1
					elif ch == ')': depth -= 1
					pos[0] += 1
				sub = s[start:pos[0] - 1]
				return ('recurse', sub)
			# Plain element symbol (defensive; not used by Sage 2.3.0 SMIRKS)
			if c.isupper():
				name = c; pos[0] += 1
				if peek().islower(): name += peek(); pos[0] += 1
				z = Z_TABLE.get(name, 0)
				return ('Z', z)
			raise ValueError(f'Unknown primitive {c!r} at pos {pos[0]} in {s!r}')
		# bond-expr: parse a bond expression between two atoms
		def bond_expr():
			# low-prec AND chain (rare in upstream FF)
			'''
			Parse a full SMIRKS bond expression: AND-chains joined by ';'
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: nested AST node for the bond expression
			'''
			left = bond_or()
			while peek() == ';':
				pos[0] += 1
				right = bond_or()
				left = ('and', left, right)
			return left
		def bond_or():
			'''
			Parse one OR-chain of bond expressions joined by ','
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: nested AST node for the parsed bond expression
			'''
			left = bond_and()
			while peek() == ',':
				pos[0] += 1
				right = bond_and()
				left = ('or', left, right)
			return left
		def bond_and():
			'''
			Parse one AND-chain of bond expressions joined by '&' or by adjacency
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: nested AST node for the parsed bond expression
			'''
			left = bond_neg()
			# Explicit '&' AND, plus implicit AND between adjacent bond primitives
			while peek() in ('&', '-', '=', '#', ':', '~', '@', '!'):
				if peek() == '&': pos[0] += 1
				right = bond_neg()
				left = ('and', left, right)
			return left
		def bond_neg():
			'''
			Parse an optionally-negated bond primitive: '!' prefix toggles negation
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: AST node wrapping the primitive (wrapped in 'not' if prefixed)
			'''
			if peek() == '!':
				pos[0] += 1
				return ('not', bond_neg())
			return bond_prim()
		def bond_prim():
			'''
			Parse a single bond primitive (-, =, #, :, @, ~, /, \\, ring digit)
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: AST node for the bond primitive
			'''
			c = peek()
			if c == '-': pos[0] += 1; return ('bo', 1.0)
			if c == '=': pos[0] += 1; return ('bo', 2.0)
			if c == '#': pos[0] += 1; return ('bo', 3.0)
			if c == ':': pos[0] += 1; return ('bo', 1.5)
			if c == '~': pos[0] += 1; return ('any',)
			if c == '@': pos[0] += 1; return ('inring',)
			if c == '/': pos[0] += 1; return ('any',)
			if c == '\\': pos[0] += 1; return ('any',)
			raise ValueError(f'Unknown bond op {c!r} at {pos[0]} in {s!r}')
		# top-level SMIRKS structure
		atom_list = []
		bond_list = []
		tags = {}
		ring_open = {}  # closure_digit -> (atom_idx, bond_expr_or_None)
		def parse_atom():
			# Bare atom forms inside recursion: '*' = wildcard; element symbols
			'''
			Parse a bracketed atom '[...]' or a bare atom symbol, advancing the cursor
			Arguments:
			----------
				No arguments taken
			Returns:
			--------
				tuple: AST node for the atom expression
			'''
			c = peek()
			if c != '[':
				if c == '*':
					pos[0] += 1
					expr = ('wild',)
				elif c.isupper():
					name = c; pos[0] += 1
					if peek().islower() and (name + peek()) in Z_TABLE:
						name += peek(); pos[0] += 1
					expr = ('Z', Z_TABLE.get(name, 0))
				elif c.islower():
					name = c.upper(); pos[0] += 1
					expr = ('and', ('Z', Z_TABLE.get(name, 0)),
						('arom', True))
				else:
					raise ValueError(
						f'Expected atom at {pos[0]} in {s!r}')
				idx = len(atom_list)
				atom_list.append({'expr': expr, 'tag': None})
				return idx
			take('[')
			expr = atom_expr()
			tag = None
			if peek() == ':':
				pos[0] += 1
				tag = read_int()
			take(']')
			idx = len(atom_list)
			atom_list.append({'expr': expr, 'tag': tag})
			if tag is not None: tags[tag] = idx
			return idx
		def is_atom_start(c):
			'''
			Test whether a character can begin an atom token in SMIRKS
			Arguments:
			----------
				c: str - single character
			Returns:
			--------
				bool: True if c starts an atom (bracket, '*', or element letter)
			'''
			return c == '[' or c == '*' or (
				c and (c.isupper() or c.islower()) and c not in 'hRrXx')
		def parse_branch(prev_idx):
			'''
			Parse a parenthesised branch sub-chain attached to atom prev_idx
			Arguments:
			----------
				prev_idx: int - atom index this branch attaches to
			Returns:
			--------
				No return value
			'''
			take('(')
			# optional bond before the next atom in the branch
			if peek() and not is_atom_start(peek()) and peek() != '(':
				be = bond_expr()
			else:
				be = ('bo', 1.0)
			a_idx = parse_atom()
			bond_list.append((prev_idx, a_idx, be))
			parse_chain(a_idx)
			take(')')
		def parse_chain(prev_idx):
			'''
			Parse a chain of atoms-and-bonds at the current cursor, starting from prev_idx
			Arguments:
			----------
				prev_idx: int - atom index this chain extends from (-1 for root)
			Returns:
			--------
				No return value
			'''
			while pos[0] < len(s):
				c = peek()
				if c == ')' or c == '': return
				if c == '(':
					parse_branch(prev_idx); continue
				if is_atom_start(c):
					a_idx = parse_atom()
					bond_list.append((prev_idx, a_idx, ('bo', 1.0)))
					prev_idx = a_idx; continue
				# Bare ring-closure digit after atom (e.g. '[*]1...~1'); default bond '-'
				if c.isdigit() or c == '%':
					if c == '%':
						pos[0] += 1
						digit = int(s[pos[0]:pos[0] + 2])
						pos[0] += 2
					else:
						digit = int(c); pos[0] += 1
					if digit in ring_open:
						a, be_open = ring_open.pop(digit)
						bond_list.append((prev_idx, a, be_open))
					else:
						ring_open[digit] = (prev_idx, ('bo', 1.0))
					continue
				# bond before an atom or ring digit
				be = bond_expr()
				c2 = peek()
				if is_atom_start(c2):
					a_idx = parse_atom()
					bond_list.append((prev_idx, a_idx, be))
					prev_idx = a_idx
				elif c2.isdigit() or c2 == '%':
					if c2 == '%':
						pos[0] += 1
						digit = int(s[pos[0]:pos[0] + 2])
						pos[0] += 2
					else:
						digit = int(c2); pos[0] += 1
					if digit in ring_open:
						a, be_open = ring_open.pop(digit)
						chosen = be if be != ('bo', 1.0) else be_open
						bond_list.append((prev_idx, a, chosen))
					else:
						ring_open[digit] = (prev_idx, be)
				else:
					raise ValueError(
						f'Unexpected after bond at {pos[0]} in {s!r}')
		# bootstrap: leading atom, then chain
		root = parse_atom()
		parse_chain(root)
		if ring_open: raise ValueError(
			f'Unclosed ring digits {list(ring_open)} in {s!r}')
		return {'atoms': atom_list, 'bonds': bond_list, 'tags': tags}
	# ============== expression evaluators =======================
	def eval_atom(expr, i):
		'''
		Evaluate a parsed atom expression against atom index i
		Arguments:
		----------
			expr: tuple - parsed AST node
			i:    int   - candidate atom index in the molecule
		Returns:
		--------
			bool: True iff atom i satisfies the expression
		'''
		k = expr[0]
		if k == 'wild':   return True
		if k == 'Z':      return Z[i] == expr[1]
		if k == 'X':      return X[i] == expr[1]
		if k == 'H':      return Hc[i] == expr[1]
		if k == 'x':      return x_count[i] == expr[1]
		if k == 'h':      return Hc[i] == expr[1]  # implicit H ~ total H
		if k == 'fc':     return fc[i] == expr[1]
		if k == 'arom':   return is_arom_atom[i] == expr[1]
		if k == 'rsize':
			n = expr[1]
			if n is None:  return bool(ring_sizes_at[i])
			# SMARTS 'r<n>' = "smallest ring this atom belongs to is size n"
			return min_ring_size[i] == n
		if k == 'Rcount':
			n = expr[1]
			# count of rings atom i belongs to (approx via SSSR)
			c = sum(1 for r in rings if i in r)
			if n is None: return c > 0
			return c == n
		if k == 'and':    return eval_atom(expr[1], i) and eval_atom(expr[2], i)
		if k == 'or':     return eval_atom(expr[1], i) or  eval_atom(expr[2], i)
		if k == 'not':    return not eval_atom(expr[1], i)
		if k == 'recurse':
			# Recursive sub-pattern rooted at i, anchored on tag :1; (sub, i) memoised
			key = (expr[1], i)
			if key in recurse_cache: return recurse_cache[key]
			recurse_cache[key] = False  # tentative, in case of self-cycle
			sub_pat = parse(expr[1])
			ok = bool(match_anchored(sub_pat, i))
			recurse_cache[key] = ok
			return ok
		raise ValueError(f'Unknown atom-expr node {k!r}')
	def eval_bond(expr, e):
		'''
		Evaluate a parsed bond expression against canonical edge e=(a,b)
		Arguments:
		----------
			expr: tuple - parsed bond AST
			e:    tuple - (a_idx, b_idx) with a_idx < b_idx
		Returns:
		--------
			bool: True iff the bond satisfies the expression
		'''
		k = expr[0]
		if k == 'any':    return True
		if k == 'bo':
			tgt = expr[1]
			b = bo.get(e, 1.0)
			if abs(tgt - 1.5) < 1e-6: return is_arom_bond.get(e, False)
			return abs(b - tgt) < 1e-6 and not is_arom_bond.get(e, False)
		if k == 'inring': return e in in_ring_bond
		if k == 'and':    return eval_bond(expr[1], e) and eval_bond(expr[2], e)
		if k == 'or':     return eval_bond(expr[1], e) or  eval_bond(expr[2], e)
		if k == 'not':    return not eval_bond(expr[1], e)
		raise ValueError(f'Unknown bond-expr node {k!r}')
	# ============== subgraph isomorphism ========================
	recurse_cache = {}
	def match_anchored(pat, anchor_atom):
		'''
		Backtracking match of pat with pat-atom 0 forced to anchor_atom
		Arguments:
		----------
			pat:         dict - parsed pattern (atoms, bonds, tags)
			anchor_atom: int  - molecule atom that pat-atom 0 must map to
		Returns:
		--------
			bool: True if any complete mapping exists
		'''
		# adjacency by pattern atom
		pat_n = len(pat['atoms'])
		pat_adj = {p: [] for p in range(pat_n)}
		for a, b, be in pat['bonds']:
			pat_adj[a].append((b, be))
			pat_adj[b].append((a, be))
		used = set()
		mapping = [-1] * pat_n
		def go(p):
			'''
			Backtracking recursion: try to map pattern atom p to a pose atom
			Arguments:
			----------
				p: int - pattern atom index
			Returns:
			--------
				bool: True when a complete consistent mapping is found
			'''
			if p == pat_n: return True
			# pick candidates
			anchored = (p == 0)
			cands = [anchor_atom] if anchored else None
			# constrain by already-mapped neighbours
			fixed_nbrs = [(q, be) for q, be in pat_adj[p] if mapping[q] >= 0]
			if fixed_nbrs and not anchored:
				q0, be0 = fixed_nbrs[0]
				cands = [m for m in nbr[mapping[q0]] if m not in used]
			if cands is None:
				cands = [a for a in sorted_ids if a not in used]
			for cand in cands:
				if cand in used: continue
				if not eval_atom(pat['atoms'][p]['expr'], cand): continue
				ok = True
				for q, be in fixed_nbrs:
					e = (min(cand, mapping[q]), max(cand, mapping[q]))
					if e not in edge_set: ok = False; break
					if not eval_bond(be, e): ok = False; break
				if not ok: continue
				mapping[p] = cand; used.add(cand)
				if go(p + 1): return True
				used.discard(cand); mapping[p] = -1
			return False
		return go(0)
	def match(pat):
		'''
		Enumerate all matches of pat; return tuples of atom indices in
		ascending tag order (only tagged atoms are returned)
		Arguments:
		----------
			pat: dict - parsed pattern
		Returns:
		--------
			list of tuples: each tuple has length = number of tags
		'''
		pat_n = len(pat['atoms'])
		pat_adj = {p: [] for p in range(pat_n)}
		for a, b, be in pat['bonds']:
			pat_adj[a].append((b, be))
			pat_adj[b].append((a, be))
		tag_order = sorted(pat['tags'].keys())
		results = []
		seen = set()
		used = set()
		mapping = [-1] * pat_n
		def go(p):
			'''
			Enumerative recursion: collect every consistent mapping of pattern atom p
			Arguments:
			----------
				p: int - pattern atom index
			Returns:
			--------
				No return value (appends matches to enclosing `results` list)
			'''
			if p == pat_n:
				key = tuple(mapping[pat['tags'][t]] for t in tag_order)
				if key not in seen:
					seen.add(key); results.append(key)
				return
			fixed_nbrs = [(q, be) for q, be in pat_adj[p] if mapping[q] >= 0]
			if fixed_nbrs:
				q0, be0 = fixed_nbrs[0]
				cands = [m for m in nbr[mapping[q0]] if m not in used]
			else:
				cands = [a for a in sorted_ids if a not in used]
			for cand in cands:
				if cand in used: continue
				if not eval_atom(pat['atoms'][p]['expr'], cand): continue
				ok = True
				for q, be in fixed_nbrs:
					e = (min(cand, mapping[q]), max(cand, mapping[q]))
					if e not in edge_set: ok = False; break
					if not eval_bond(be, e): ok = False; break
				if not ok: continue
				mapping[p] = cand; used.add(cand)
				go(p + 1)
				used.discard(cand); mapping[p] = -1
		go(0)
		return results
	# Assignment dict — last-match-wins per pattern
	out = {'bonds': {}, 'angles': {}, 'ub': {}, 'propers': {},
		'impropers': [], 'vdw': {}, 'vdw14': {}, 'polarisation': {},
		'charges': {i: None for i in sorted_ids},
		'constraints': set(), 'restri': {}}
	parsed = {}
	def get(smirks):
		'''
		Memoised parser: cache parsed SMIRKS expressions across repeated lookups
		Arguments:
		----------
			smirks: str - SMIRKS pattern
		Returns:
		--------
			dict: parsed pattern (atoms, bonds, tags)
		'''
		if smirks not in parsed: parsed[smirks] = parse(smirks)
		return parsed[smirks]
	rmin2sig = 2.0 / (2.0 ** (1.0 / 6.0))
	# ===============================================================
	# Atom-typed force-field layer (AMBER ff19SB / CHARMM36)
	# ---------------------------------------------------------------
	# Atom-typed force fields carry per-residue templates under
	# Constraints['<residue_templates>'] (each atom's name / element /
	# class / charge plus the intra-residue bond list) and key their
	# sections with the <at=c1,...> tag prefix in place of real SMIRKS.
	# Each pose residue is matched to its template by topology (see
	# maptemplate), so the assignment is independent of the pose's
	# atom-naming convention (Build/ReBuild vs an imported PDB). When no
	# templates are present every helper below is inert and the SMIRKS
	# path runs exactly as before, so OpenFF / Default energies stay
	# byte-identical.
	# ===============================================================
	residue_templates = (params.get('Constraints', {})
		.get('<residue_templates>'))
	improper_style = params.get('improper_style', 'smirnoff')
	atom_name   = {i: atoms[i][0] for i in sorted_ids}
	atom_elem   = {i: atoms[i][1] for i in sorted_ids}
	atom_class  = {i: None for i in sorted_ids}
	atom_charge = {i: None for i in sorted_ids}
	atom_reskey = {i: None for i in sorted_ids}
	atom_tname  = dict(atom_name)
	HEAVY_ALIAS = {'CD1': 'CD'}
	def v2v3(nm):
		'''
		Convert a PDB v2 atom name to v3 (leading digits move to end)
		Arguments:
		----------
			nm: str - an atom name, e.g. '1HB'
		Returns:
		--------
			str: the v3 form, e.g. 'HB1' ('HA' is returned unchanged)
		'''
		i = 0
		while i < len(nm) and nm[i].isdigit(): i += 1
		return nm[i:] + nm[:i] if 0 < i < len(nm) else nm
	def assigntypes():
		'''
		Resolve every atom's residue key, atom class and partial charge
		by matching each pose residue to its force-field residue
		template, honouring N/C (protein) and 5'/3' (nucleic acid)
		terminal variants, HIS protonation and disulfide CYX retagging
		Arguments:
		----------
			No arguments taken (closes over pose / residue_templates)
		Returns:
		--------
			None: fills atom_class / atom_charge / atom_reskey /
			atom_tname and out['restri'] in place
		'''
		aas  = pose.data.get('Amino Acids') or {}
		nucs = pose.data.get('Nucleotides') or {}
		def maptemplate(reskeys, ratoms, aliases=None):
			'''
			Match one residue's pose atoms to the first template present
			among reskeys: exact name, then PDB v2<->v3 transform, then
			parent-heavy-atom fallback
			Arguments:
			----------
				reskeys: list of str - candidate variant keys
				ratoms:  list of int - the residue's atom indices
				aliases: dict or None - extra exact-name aliases
			Returns:
			--------
				None: fills atom_class / atom_charge / atom_reskey /
				atom_tname for every matched atom
			'''
			aliases = aliases or {}
			tpl, chosen = None, None
			for k in reskeys:
				t = residue_templates.get(k)
				if t is not None:
					tpl, chosen = t, k
					break
			if tpl is None: return
			tatoms = {a[0]: (a[1], a[2], a[3]) for a in tpl['atoms']}
			tadj = {}
			for x, y in tpl['bonds']:
				tadj.setdefault(x, set()).add(y)
				tadj.setdefault(y, set()).add(x)
			rset = set(ratoms)
			padj = {a: [b for b in nbr[a] if b in rset]
				for a in ratoms}
			hmap = {}
			for a in ratoms:
				if atom_elem[a] == 'H': continue
				nm = atom_name[a]
				if nm in tatoms: hmap[a] = nm
				elif HEAVY_ALIAS.get(nm) in tatoms:
					hmap[a] = HEAVY_ALIAS[nm]
			for a in ratoms:
				nm, el = atom_name[a], atom_elem[a]
				hit, tn = None, None
				if nm in tatoms:
					hit, tn = tatoms[nm], nm
				elif aliases.get(nm) in tatoms:
					tn = aliases[nm]; hit = tatoms[tn]
				elif el != 'H' and HEAVY_ALIAS.get(nm) in tatoms:
					tn = HEAVY_ALIAS[nm]; hit = tatoms[tn]
				elif el == 'H' and v2v3(nm) in tatoms:
					tn = v2v3(nm); hit = tatoms[tn]
				else:
					parents = [hmap[x] for x in padj[a]
						if x in hmap]
					for cn, val in tatoms.items():
						if val[0] != el: continue
						if any(p in tadj.get(cn, ())
								for p in parents):
							hit, tn = val, cn
							break
				if hit is not None:
					atom_class[a]  = hit[1]
					atom_charge[a] = hit[2]
					atom_reskey[a] = chosen
					atom_tname[a]  = tn
		prot = {}
		for ri in sorted(aas):
			prot.setdefault(aas[ri][1], []).append(ri)
		ri_atoms = {ri: list(aas[ri][2]) + list(aas[ri][3])
			for ri in aas}
		sg_of = {a: ri for ri, ats in ri_atoms.items()
			for a in ats if atom_name.get(a) == 'SG'}
		ss_res = set()
		for a in sg_of:
			for b in nbr[a]:
				if b in sg_of and sg_of[b] != sg_of[a]:
					ss_res.add(sg_of[a]); ss_res.add(sg_of[b])
		for chain, ris in prot.items():
			last = len(ris) - 1
			for pos, ri in enumerate(ris):
				tri = str(aas[ri][5]).upper()
				ats = ri_atoms[ri]
				anames = {atom_name.get(a) for a in ats}
				if tri in ('HIS', 'HID', 'HIE', 'HIP',
						'HSD', 'HSE', 'HSP'):
					hd1 = 'HD1' in anames
					he2 = 'HE2' in anames
					if hd1 and he2: cand = ('HIP', 'HSP')
					elif hd1:       cand = ('HID', 'HSD')
					else:           cand = ('HIE', 'HSE')
					tri = next((v for v in cand if any(
						(p + v) in residue_templates
						for p in ('', 'N', 'C'))), cand[0])
				if tri == 'CYS' and ri in ss_res:
					tri = 'CYX'
				out['restri'][ri] = tri
				keys = []
				if pos == 0:    keys.append('N' + tri)
				if pos == last: keys.append('C' + tri)
				keys.append(tri)
				# OpenMM names the first N-terminal proton 'H'; the
				# AMBER NXXX templates call it 'H1'.
				aliases = None
				if pos == 0 and 'H' in anames and 'H1' not in anames:
					aliases = {'H': 'H1'}
				maptemplate(keys, ats, aliases)
		nuc = {}
		for ni in sorted(nucs):
			nuc.setdefault(nucs[ni][1], []).append(ni)
		for chain, nis in nuc.items():
			last = len(nis) - 1
			for pos, ni in enumerate(nis):
				tri = str(nucs[ni][4]).upper()
				ats = list(nucs[ni][2]) + list(nucs[ni][3])
				keys = []
				if pos == 0:    keys.append(tri + '5')
				if pos == last: keys.append(tri + '3')
				keys.append(tri)
				keys.append(tri + 'N')
				maptemplate(keys, ats)
	if residue_templates is not None:
		assigntypes()
		for i in sorted_ids:
			if atom_charge[i] is not None:
				out['charges'][i] = atom_charge[i]
	def tagparse(key):
		'''
		Split a leading <...> tag prefix off a force-field section key
		Arguments:
		----------
			key: str - a force-field section key
		Returns:
		--------
			tuple or None: ('map',) for <residue_templates>;
				('at', [classes]) for <at=...>; ('res', (tri, name)) for
				<res=><atom=>; None when the key carries no tag (a real
				SMIRKS)
		'''
		if not key or key[0] != '<': return None
		if key.startswith('<residue_templates>'): return ('map',)
		if key.startswith('<at='):
			return ('at', key[4:key.index('>')].split(','))
		if key.startswith('<res='):
			tri  = key[5:key.index('>')]
			rest = key[key.index('>') + 1:]
			name = rest[rest.index('=') + 1:rest.index('>')]
			return ('res', (tri, name))
		return None
	cls_of = atom_class.get
	def clsmatch(spec, idxs):
		'''
		Test an atom-class spec (with '*' wildcards) against an atom tuple
		Arguments:
		----------
			spec: list of str - class names; '*' matches any class
			idxs: tuple of int - candidate atom indices, same length
		Returns:
		--------
			bool: True iff every position matches
		'''
		for s, a in zip(spec, idxs):
			if s != '*' and cls_of(a) != s: return False
		return True
	# Topology tuples reused by the atom-typed section loops
	tri_list, quad_list = [], []
	if residue_templates is not None:
		for j in sorted_ids:
			ns = nbr[j]
			for x in range(len(ns)):
				for y in range(x + 1, len(ns)):
					tri_list.append((ns[x], j, ns[y]))
		for (j, k) in edges:
			for i in nbr[j]:
				if i == k: continue
				for l in nbr[k]:
					if l == j or l == i: continue
					quad_list.append((i, j, k, l))
	# ---- Constraints (rigid X-H bonds; SMIRKS force fields only) ----
	for sm, par in params.get('Constraints', {}).items():
		if tagparse(sm) is not None: continue
		try: pat = get(sm)
		except Exception: continue
		for tup in match(pat):
			if len(tup) >= 2:
				a, b = int(tup[0]), int(tup[1])
				out['constraints'].add((min(a, b), max(a, b)))
	# ---- Bonds ------------------------------------------------------
	for sm, par in params.get('Bonds', {}).items():
		tg = tagparse(sm)
		if tg is None:
			try: pat = get(sm)
			except Exception: continue
			for tup in match(pat):
				if len(tup) != 2: continue
				i, j = sorted(tup)
				if (i, j) in edge_set:
					out['bonds'][(i, j)] = [par['r_0'], par['K_b']]
		elif tg[0] == 'at':
			spec = tg[1]
			for (a, b) in edges:
				if clsmatch(spec, (a, b)) or clsmatch(spec, (b, a)):
					out['bonds'][(a, b)] = [par['r_0'], par['K_b']]
	# ---- Angles (tag :2 is the centre atom) -------------------------
	for sm, par in params.get('Angles', {}).items():
		tg = tagparse(sm)
		if tg is None:
			try: pat = get(sm)
			except Exception: continue
			for tup in match(pat):
				if len(tup) != 3: continue
				i, j, k = tup
				if (min(i, j), max(i, j)) in edge_set and \
					(min(j, k), max(j, k)) in edge_set:
					ii, kk = (i, k) if i < k else (k, i)
					out['angles'][(ii, j, kk)] = \
						[par['theta_0'], par['K_theta']]
		elif tg[0] == 'at':
			spec = tg[1]
			val = (par['theta_0'], par['K_theta'])
			for (i, j, k) in tri_list:
				if clsmatch(spec, (i, j, k)) or \
					clsmatch(spec, (k, j, i)):
					ii, kk = (i, k) if i < k else (k, i)
					out['angles'][(ii, j, kk)] = list(val)
	# ---- Urey-Bradley ----------------------------------------------
	for sm, par in params.get('UB', {}).items():
		tg = tagparse(sm)
		if tg is None:
			try: pat = get(sm)
			except Exception: continue
			for tup in match(pat):
				if len(tup) != 3: continue
				i, j, k = tup
				if (min(i, j), max(i, j)) in edge_set and \
					(min(j, k), max(j, k)) in edge_set:
					ii, kk = (i, k) if i < k else (k, i)
					out['ub'][(ii, j, kk)] = [par.get('s_0', 0.0),
						par.get('K_ub', 0.0)]
		elif tg[0] == 'at':
			spec = tg[1]
			val = (par.get('s_0', 0.0), par.get('K_ub', 0.0))
			for (i, j, k) in tri_list:
				if clsmatch(spec, (i, j, k)) or \
					clsmatch(spec, (k, j, i)):
					ii, kk = (i, k) if i < k else (k, i)
					out['ub'][(ii, j, kk)] = list(val)
	# ---- Proper torsions -------------------------------------------
	proper_best = {}
	for sm, par in params.get('ProperTorsions', {}).items():
		tg = tagparse(sm)
		if tg is None:
			try: pat = get(sm)
			except Exception: continue
			for tup in match(pat):
				if len(tup) != 4: continue
				i, j, k, l = tup
				if (min(i, j), max(i, j)) not in edge_set: continue
				if (min(j, k), max(j, k)) not in edge_set: continue
				if (min(k, l), max(k, l)) not in edge_set: continue
				if (i, j, k, l) > (l, k, j, i):
					i, j, k, l = l, k, j, i
				out['propers'][(i, j, k, l)] = [
					[c['n'], c['phi_0'], c['K_phi'],
						c.get('idivf', 1.0)]
					for c in par['components']]
		elif tg[0] == 'at':
			spec  = tg[1]
			score = sum(1 for s in spec if s != '*')
			comps = [[c['n'], c['phi_0'], c['K_phi'],
				c.get('idivf', 1.0)] for c in par['components']]
			for (i, j, k, l) in quad_list:
				if not (clsmatch(spec, (i, j, k, l)) or
					clsmatch(spec, (l, k, j, i))): continue
				canon = (i, j, k, l) if (i, j, k, l) <= (l, k, j, i) \
					else (l, k, j, i)
				prev = proper_best.get(canon)
				if prev is None or score >= prev[0]:
					proper_best[canon] = (score, comps)
	for canon, (score, comps) in proper_best.items():
		out['propers'][canon] = [list(c) for c in comps]
	# ---- Improper torsions -----------------------------------------
	if improper_style in ('amber', 'charmm'):
		# Class-typed impropers: AMBER stores the centre at tag :2 with
		# a Y-shape, CHARMM at tag :1. The match tuple is laid out so
		# ImproperTorsionPotential evaluates the dihedral as OpenMM does
		# (AMBER: outer-outer-CENTRE-outer; CHARMM: CENTRE-t2-t3-t4),
		# at full K with no trefoil expansion.
		imp_best = {}
		cpos = 1 if improper_style == 'amber' else 0
		for sm, par in params.get('ImproperTorsions', {}).items():
			tg = tagparse(sm)
			if tg is None or tg[0] != 'at' or len(tg[1]) != 4:
				continue
			spec  = tg[1]
			cspec = spec[cpos]
			ospec = [spec[p] for p in range(4) if p != cpos]
			score = sum(1 for s in spec if s != '*')
			for c in sorted_ids:
				if cspec != '*' and cls_of(c) != cspec: continue
				ns = nbr[c]
				if len(ns) < 3: continue
				prev = imp_best.get(c)
				if prev is not None and prev[0] > score: continue
				trip = None
				for x in range(len(ns)):
					for y in range(len(ns)):
						if y == x: continue
						for z in range(len(ns)):
							if z == x or z == y: continue
							cand = (ns[x], ns[y], ns[z])
							if clsmatch(ospec, cand):
								trip = cand; break
						if trip: break
					if trip: break
				if trip is None: continue
				o1, o2, o3 = trip
				ent = []
				for cc in par['components']:
					if improper_style == 'amber':
						ent.append((o1, o2, c, o3, cc['n'],
							cc['phi_0'], cc['K_phi']))
					else:
						ent.append((c, o1, o2, o3, cc['n'],
							cc['phi_0'], cc['K_phi']))
				imp_best[c] = (score, ent)
		for score, ent in imp_best.values():
			out['impropers'].extend(ent)
	else:
		# SMIRNOFF trefoil: centre at tag :2, 3 cyclic outer perms, K/3
		imp_by_centre = {}
		for sm, par in params.get('ImproperTorsions', {}).items():
			if tagparse(sm) is not None: continue
			try: pat = get(sm)
			except Exception: continue
			for tup in match(pat):
				if len(tup) != 4: continue
				a1, a2, a3, a4 = tup
				perms = [(a1, a3, a4), (a3, a4, a1), (a4, a1, a3)]
				entries = []
				for o1, o2, o3 in perms:
					for c in par['components']:
						entries.append((a2, o1, o2, o3, c['n'],
							c['phi_0'], c['K_phi'] / 3.0))
				imp_by_centre[a2] = entries
		for entries in imp_by_centre.values():
			out['impropers'].extend(entries)
	# ---- vdW (per-atom eps/sigma; optional separate 1-4 params) -----
	for sm, par in params.get('vdW', {}).items():
		tg = tagparse(sm)
		if tg is not None and tg[0] not in ('at',): continue
		eps = par['epsilon']
		sig = par['sigma'] if 'sigma' in par else par['r'] * rmin2sig
		has14 = 'epsilon14' in par or 'sigma14' in par
		if has14:
			eps14 = par.get('epsilon14', eps)
			if 'sigma14' in par: sig14 = par['sigma14']
			elif 'r14' in par:   sig14 = par['r14'] * rmin2sig
			else:                sig14 = sig
		if tg is None:
			try: pat = get(sm)
			except Exception: continue
			hits = [tup[0] for tup in match(pat) if len(tup) == 1]
		else:
			spec0 = tg[1][0]
			hits = [i for i in sorted_ids
				if spec0 == '*' or cls_of(i) == spec0]
		for i in hits:
			out['vdw'][i] = [eps, sig]
			out['polarisation'][i] = par.get('alpha', 0.0)
			if has14: out['vdw14'][i] = [eps14, sig14]
	# ---- Electrostatic / library charges ---------------------------
	for sm, par in params.get('LibraryCharges', {}).items():
		tg = tagparse(sm)
		if tg is None:
			try: pat = get(sm)
			except Exception: continue
			qs = par.get('q', [])
			for tup in match(pat):
				for k, idx in enumerate(tup):
					if k < len(qs): out['charges'][idx] = float(qs[k])
		elif tg[0] == 'res':
			tri, nm = tg[1]
			qs = par.get('q', [])
			if not qs: continue
			q0 = float(qs[0])
			for i in sorted_ids:
				if atom_reskey.get(i) == tri and \
					atom_tname.get(i) == nm:
					out['charges'][i] = q0
	return out

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
		# Atom-typed force fields (AMBER/CHARMM) carry these flags so
		# SMIRKSMatch knows the improper convention and torsion
		# precedence; absent for SMIRKS force fields (OpenFF/Default).
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
		def addbond(i, j):
			'''
			Add a bond-order-1.0 edge between atoms i and j in the working tables
			Arguments:
			----------
				i: int - atom index
				j: int - atom index
			Returns:
			--------
				No return value
			'''
			bonds.setdefault(i, []).append(j)
			bonds.setdefault(j, []).append(i)
			orders.setdefault(i, []).append(1.0)
			orders.setdefault(j, []).append(1.0)
			bondset.add((min(i, j), max(i, j)))
		added = 0
		orphans = [i for i in ids
			if deg[i] == 0 and atoms[i][1] == 'H']
		for i in orphans:
			d = np.linalg.norm(coords - coords[i], axis=1)
			d[i] = 1e18
			j = int(np.argmin(d))
			if d[j] <= 1.3:
				addbond(i, j); added += 1
		if pose.data.get('Type') == 'Protein':
			sg = [i for i in ids if atoms[i][0] == 'SG']
			for a in range(len(sg)):
				for b in range(a + 1, len(sg)):
					i, j = sg[a], sg[b]
					if (min(i, j), max(i, j)) in bondset: continue
					if np.linalg.norm(coords[i] - coords[j]) <= 2.5:
						addbond(i, j); added += 1
		return added
	def __call__(self, pose, grad=False, box=None):
		'''
		Calculates the total potential energy summed over configured terms
		Arguments:
		----------
			pose: Pose - molecule source protein, DNA, RNA, or Molecule pose
			grad: bool - if True, also return per-atom forces (N, 3) array
			box:  None for no PBC; (3,) for orthorhombic; (3, 3) for triclinic
		Returns:
		--------
			float: potential energy in kJ/mol  (when grad=False)
			(float, ndarray): energy and (N, 3) forces  (when grad=True)
		'''
		if len(pose.data.get('Atoms', {})) == 0:
			return (0.0, np.zeros((0, 3))) if grad else 0.0
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
			assigns = SMIRKSMatch(pose, self.mol)
			atoms_set = set(pose.data['Atoms'].keys())
			bonds_dict = pose.data['Bonds']
			nbr_local = {i: [j for j in bonds_dict.get(i, [])
				if j in atoms_set and j != i] for i in atoms_set}
			gaps = []
			for i in atoms_set:
				for j in nbr_local[i]:
					if i >= j: continue
					if (int(i), int(j)) not in assigns['bonds']:
						gaps.append(f'bond ({i}, {j})')
			matched_angles = {(min(t[0], t[2]), t[1], max(t[0], t[2]))
				for t in assigns['angles']}
			for j in atoms_set:
				ns = nbr_local[j]
				for x in range(len(ns)):
					for y in range(x + 1, len(ns)):
						i, k = ns[x], ns[y]
						tup = (min(i, k), j, max(i, k))
						if tup not in matched_angles:
							gaps.append(f'angle ({i}, {j}, {k})')
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
								gaps.append(f'torsion ({x}, {i}, {j}, {y})')
			# The "every 3-coordinate atom needs an improper" rule
			# only holds for SMIRNOFF force fields; AMBER / CHARMM
			# place impropers on selected centres only.
			if self.Parameters.get('improper_style',
				'smirnoff') == 'smirnoff':
				matched_centres = {tup[0]
					for tup in assigns['impropers']}
				matched_centres |= {tup[2]
					for tup in assigns['impropers']}
				for c in atoms_set:
					if (len(nbr_local[c]) == 3
						and c not in matched_centres):
						gaps.append(f'improper centre {c}')
			for i in atoms_set:
				if assigns['vdw'].get(i) is None:
					gaps.append(f'vdW atom {i}')
			if gaps:
				atoms = pose.data['Atoms']
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
					print(f'[Pose] Note: {msg}', file=sys.stderr)
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
					angle_t0[p] = par[0]; angle_Kt[p] = par[1]
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
			# CHARMM-style separate 1-4 Lennard-Jones parameters:
			# overwrite the mixed sigma/epsilon at 1-4 positions so
			# VDWPotential's sum(lj[far]) + f_lj*sum(lj[14]) yields the
			# 1-4-specific energy (CHARMM uses f_lj = 1.0).
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
				phi_q_list = []; psi_q_list = []; grids = []
				for kk, ri in enumerate(res_order):
					if kk == 0 or kk == len(res_order) - 1:
						continue
					prev_ri = res_order[kk - 1]
					next_ri = res_order[kk + 1]
					chain  = bb_per_res[ri][0]
					if (bb_per_res[prev_ri][0] != chain or
						bb_per_res[next_ri][0] != chain):
						continue
					# Atom-typed force fields key CMAP by the resolved
					# variant tricode (CYX vs CYS, HID/HIE/HIP, ...);
					# SMIRKS force fields key by the 1-letter code.
					restri = assigns.get('restri', {})
					grid = (cmap_section.get(restri.get(ri))
						or cmap_section.get(aas[ri][5])
						or cmap_section.get(bb_per_res[ri][1]))
					if grid is None: continue
					g = np.asarray(grid, dtype=np.float64)
					if g.shape != (24, 24): continue
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
			# Periodic-cubic-spline derivative tables for the
			# OpenMM-exact CMAP bicubic (CMAPPotential alg='openmm').
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
			self._cache = cache
			self._cache_hash = h
		n = self._cache['n']
		E, F = 0.0, np.zeros((n, 3))
		# Sticky CPU FP-exception flags (e.g. from a 1/r or r**12 in one
		# term) are otherwise reported spuriously by a later term's
		# matmul; energies stay finite, so the flags are suppressed here.
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
		def find_rings():
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
		rings = find_rings()
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
		def loadtensor(d):
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
		with np.errstate(over='ignore', under='ignore', divide='ignore',
				invalid='ignore'):
			for layer in nagl['gcn_layers']:
				W_neigh = loadtensor(layer['fc_neigh_w'])
				W_self  = loadtensor(layer['fc_self_w'])
				b_self  = loadtensor(layer['fc_self_b'])
				h_avg        = A_mean @ h
				h_self_proj  = h @ W_self.T + b_self
				h_neigh_proj = h_avg @ W_neigh.T
				h = h_self_proj + h_neigh_proj
				np.maximum(h, 0, out=h)
		W0 = loadtensor(nagl['readout']['linear_0_w'])
		b0 = loadtensor(nagl['readout']['linear_0_b'])
		W1 = loadtensor(nagl['readout']['linear_1_w'])
		b1 = loadtensor(nagl['readout']['linear_1_b'])
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
	def ImproperTorsionPotential(self,pose,cache,alg='harmonic',grad=True,box=None):
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
		forces = -np.sum(M, axis=1) + np.sum(M, axis=0)
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

# Module-level memoization caches for Score helpers (replace what
# used to be `self._<x>_cache` instance attributes; now per-process).
_FADUN_GRID_CACHE = {}
_FADUN_ENT_CACHE = {}
_FADUN_NRCHI_CACHE = {}
_PAAPP_SPLINE_CACHE = {}
_RAMA_ENTROPY = {}
_RAMA_SPLINE_CACHE = {}

def ScoreMatch(pose, params, ligand=None, xs_override=None,
		nrot_override=None):
	'''
	Build the per-pose support cache used by every Score energy term
	Arguments:
	----------
		pose:          Pose or Molecule - receptor / source structure
		params:        dict - the active ['Score Parameters'][NAME] block
		ligand:        Molecule or None - optional small-molecule ligand
		xs_override:   dict or None - validation hook; maps combined index to XS atom type
		nrot_override: int or None - validation hook for ligand n_rot
	Returns:
	--------
		dict: 'hash', 'coords', 'atom_types', 'inter_pairs', 'intra_pairs'
		      plus per-term raw value keys (e.g. 'FaAtrPotential') and
		      callable nested helpers (e.g. 'evalpairs', 'ref15hbond')
	'''
	def patternsearch(pose, params, ligand=None,
			xs_override=None, nrot_override=None):
		'''
		Classify atoms and build pair lists for a Score function
		Arguments:
		----------
			pose:          Pose or Molecule - receptor / source structure
			params:        dict - the active ['Score Parameters'][NAME] block
			ligand:        Molecule or None - optional ligand for docking
			xs_override:   dict or None - validation hook; maps combined
				receptor+ligand atom index to an XS type name, bypassing
				derived typing
			nrot_override: int or None - validation hook; explicit Nrot
		Returns:
		--------
			dict: keys 'xs_types' (int array), 'xs_radii_arr',
			'xs_is_hydrophobic_arr', 'xs_is_donor_arr',
			'xs_is_acceptor_arr', 'coords', 'inter_pairs',
			'intra_ligand_pairs', 'nrot', 'n_r' (receptor atom count)
		'''
		if 'Atom_types' in params and 'Residue_types' in params:
			out = ref15atomcache(pose, params)
			n_atoms = int(len(out['coords']))
			out.setdefault('inter_pairs',
				np.empty((0, 2), dtype=np.int64))
			out.setdefault('intra_ligand_pairs',
				np.empty((0, 2), dtype=np.int64))
			out.setdefault('nrot', 0)
			out.setdefault('xs_types',
				np.full(n_atoms, -1, dtype=np.int64))
			out.setdefault('xs_radii_arr', np.zeros(0))
			out.setdefault('xs_is_hydrophobic_arr',
				np.zeros(0, dtype=bool))
			out.setdefault('xs_is_donor_arr', np.zeros(0, dtype=bool))
			out.setdefault('xs_is_acceptor_arr', np.zeros(0, dtype=bool))
			return out
		if 'XS_atom_types' in params:
			return patternsearchvina(pose, params, ligand,
				xs_override, nrot_override)
		raise Exception(
			'PatternSearch: unsupported params (no recognised typing system)')
	def patternsearchvina(pose, params, ligand, xs_override, nrot_override):
		'''
		XS atom typing + pair lists for the AutoDock Vina score function
		Arguments:
		----------
			pose:          Pose or Molecule - receptor
			params:        dict - the AutoDock Vina param block
			ligand:        Molecule or None - the ligand (None for non-docking)
			xs_override:   dict or None - {combined_index: 'XS_TYPE_NAME', ...}
			nrot_override: int or None - explicit Nrot
		Returns:
		--------
			dict: see PatternSearch
		'''
		r_atoms = pose.data['Atoms']
		r_bonds = pose.data['Bonds']
		r_coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		n_r = len(r_atoms)
		if ligand is not None:
			l_atoms = ligand.data['Atoms']
			l_bonds = ligand.data['Bonds']
			l_coords = np.asarray(ligand.data['Coordinates'], dtype=np.float64)
			n_l = len(l_atoms)
		else:
			l_atoms = {}; l_bonds = {}; l_coords = np.empty((0, 3))
			n_l = 0
		n = n_r + n_l
		coords = np.vstack([r_coords, l_coords]) if n_l else r_coords.copy()
		xs_types_db = params['XS_atom_types']
		xs_names_sorted = sorted(xs_types_db.keys())
		name_to_idx = {nm: i for i, nm in enumerate(xs_names_sorted)}
		xs_radii_arr = np.array(
			[xs_types_db[nm]['radius'] for nm in xs_names_sorted],
			dtype=np.float64)
		xs_is_hphob = np.array(
			[xs_types_db[nm]['hydrophobic'] for nm in xs_names_sorted],
			dtype=bool)
		xs_is_donor = np.array(
			[xs_types_db[nm]['donor'] for nm in xs_names_sorted],
			dtype=bool)
		xs_is_accep = np.array(
			[xs_types_db[nm]['acceptor'] for nm in xs_names_sorted],
			dtype=bool)
		HALOGEN = {'F': 'F_H', 'Cl': 'Cl_H', 'Br': 'Br_H', 'I': 'I_H'}
		METALS = {'Mg', 'Mn', 'Zn', 'Ca', 'Fe', 'Cu', 'Co',
			'Na', 'K', 'Hg', 'Cd', 'Ni'}
		def atomel(gi):
			'''
			Element symbol of the atom at combined index gi (receptor first, then ligand)
			Arguments:
			----------
				gi: int - combined receptor/ligand atom index
			Returns:
			--------
				str: element symbol
			'''
			rec = r_atoms[gi] if gi < n_r else l_atoms[gi - n_r]
			return rec[1]
		def atomnbrs(gi):
			'''
			Combined-index neighbour list of the atom at gi
			Arguments:
			----------
				gi: int - combined receptor/ligand atom index
			Returns:
			--------
				list of int: bonded neighbours in combined indexing
			'''
			if gi < n_r: return list(r_bonds.get(gi, []))
			return [n_r + j for j in l_bonds.get(gi - n_r, [])]
		xs = np.full(n, -1, dtype=np.int64)
		_AA20 = ('ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS',
			'HIS_D','ILE','LEU','LYS','MET','PHE','PRO','SER','THR',
			'TRP','TYR','VAL')
		PROT_XS = {}
		for t in _AA20:
			PROT_XS[(t, 'O')]   = 'O_A'
			PROT_XS[(t, 'OXT')] = 'O_A'
			if t != 'PRO':
				PROT_XS[(t, 'N')] = 'N_D'
		PROT_XS.update({
			('ARG','NE'):'N_D', ('ARG','NH1'):'N_D', ('ARG','NH2'):'N_D',
			('LYS','NZ'):'N_D',
			('TRP','NE1'):'N_D',
			('ASN','ND2'):'N_D', ('ASN','OD1'):'O_A',
			('GLN','NE2'):'N_D', ('GLN','OE1'):'O_A',
			('HIS','ND1'):'N_A', ('HIS','NE2'):'N_D',
			('HIS_D','ND1'):'N_D', ('HIS_D','NE2'):'N_A',
			('SER','OG'):'O_DA',
			('THR','OG1'):'O_DA',
			('TYR','OH'):'O_DA',
			('ASP','OD1'):'O_A', ('ASP','OD2'):'O_A',
			('GLU','OE1'):'O_A', ('GLU','OE2'):'O_A',
		})
		r_atom_to_tri = {}
		aas = pose.data.get('Amino Acids') or {}
		for ri, info in aas.items():
			if not info or len(info) < 6: continue
			tri = info[5]
			for ai in (info[2] if len(info) > 2 else []):
				r_atom_to_tri[int(ai)] = tri
			for ai in (info[3] if len(info) > 3 else []):
				r_atom_to_tri[int(ai)] = tri
		H_coords_r = [coords[i] for i in range(n_r)
			if atomel(i) == 'H']
		H_coords_r = (np.asarray(H_coords_r, dtype=np.float64)
			if H_coords_r else np.zeros((0, 3)))
		def has_polar_h(gi):
			'''
			True iff atom gi has a bonded H (or a receptor H within 1.3 A when gi is receptor)
			Arguments:
			----------
				gi: int - combined receptor/ligand atom index
			Returns:
			--------
				bool: True if at least one polar H is attached
			'''
			for j in atomnbrs(gi):
				if atomel(j) == 'H': return True
			if gi < n_r and len(H_coords_r):
				d = np.linalg.norm(H_coords_r - coords[gi], axis=1)
				if (d < 1.3).any(): return True
			return False
		def protein_xs(gi):
			'''
			Look up the XS atom type override for a receptor protein atom via PROT_XS
			Arguments:
			----------
				gi: int - combined receptor/ligand atom index
			Returns:
			--------
				str or None: XS atom-type code (e.g. N_D, O_DA) or None if no override
			'''
			tri = r_atom_to_tri.get(gi)
			if tri is None: return None
			nm = r_atoms[gi][0] if r_atoms.get(gi) else None
			if nm is None: return None
			return PROT_XS.get((tri, nm))
		if xs_override is not None:
			for gi, nm in xs_override.items():
				if nm in name_to_idx:
					xs[int(gi)] = name_to_idx[nm]
		else:
			for gi in range(n):
				el = atomel(gi)
				if el == 'H': continue
				if el == 'C':
					cp = any(atomel(j) not in ('C', 'H')
						for j in atomnbrs(gi))
					xs[gi] = name_to_idx['C_P' if cp else 'C_H']
				elif el == 'N':
					ov = protein_xs(gi) if gi < n_r else None
					if ov in ('N_D', 'N_A', 'N_DA'):
						xs[gi] = name_to_idx[ov]
					else:
						has_h = has_polar_h(gi)
						xs[gi] = name_to_idx['N_D' if has_h else 'N_A']
				elif el == 'O':
					ov = protein_xs(gi) if gi < n_r else None
					if ov in ('O_A', 'O_D', 'O_DA'):
						xs[gi] = name_to_idx[ov]
					else:
						has_h = has_polar_h(gi)
						xs[gi] = name_to_idx['O_DA' if has_h else 'O_A']
				elif el == 'S':
					xs[gi] = name_to_idx['S_P']
				elif el == 'P':
					xs[gi] = name_to_idx['P_P']
				elif el in HALOGEN:
					xs[gi] = name_to_idx[HALOGEN[el]]
				elif el in METALS:
					xs[gi] = name_to_idx['Met_D']
		cutoff = float(params['Constants'].get('cutoff', 8.0))
		inter_pairs_list = []
		intra_pairs_list = []
		if n_l > 0:
			r_typed = np.array([i for i in range(n_r) if xs[i] >= 0],
				dtype=np.int64)
			l_typed = np.array([n_r + i for i in range(n_l)
				if xs[n_r + i] >= 0], dtype=np.int64)
			if len(r_typed) and len(l_typed):
				diff = coords[r_typed][:, None, :] \
					- coords[l_typed][None, :, :]
				d = np.linalg.norm(diff, axis=2)
				ix, iy = np.where(d < cutoff)
				inter_pairs_list = list(
					zip(r_typed[ix].tolist(), l_typed[iy].tolist()))
			l_adj = {i: set(int(j) for j in l_bonds.get(i, []))
				for i in range(n_l)}
			excluded = {i: bfswithin(l_adj, i, 3) for i in range(n_l)}
			for i in range(n_l):
				if xs[n_r + i] < 0: continue
				for j in range(i + 1, n_l):
					if j in excluded[i]: continue
					if xs[n_r + j] < 0: continue
					dij = np.linalg.norm(
						coords[n_r + i] - coords[n_r + j])
					if dij < cutoff:
						intra_pairs_list.append((n_r + i, n_r + j))
		if nrot_override is not None:
			nrot = float(nrot_override)
		elif ligand is not None:
			nrot = countnumtors(ligand)
		else:
			nrot = 0
		inter_pairs = (np.array(inter_pairs_list, dtype=np.int64)
			if inter_pairs_list else np.empty((0, 2), dtype=np.int64))
		intra_pairs = (np.array(intra_pairs_list, dtype=np.int64)
			if intra_pairs_list else np.empty((0, 2), dtype=np.int64))
		return {
			'xs_types': xs,
			'xs_radii_arr': xs_radii_arr,
			'xs_is_hydrophobic_arr': xs_is_hphob,
			'xs_is_donor_arr': xs_is_donor,
			'xs_is_acceptor_arr': xs_is_accep,
			'coords': coords,
			'inter_pairs': inter_pairs,
			'intra_ligand_pairs': intra_pairs,
			'nrot': nrot,
			'n_r': n_r}
	def ref15atomcache(pose, params):
		'''
		Atom typing + pair lists for the score function
		Arguments:
		----------
			pose:   Pose - protein structure (with hydrogens added)
			params: dict - the param block under Score Parameters
		Returns:
		--------
			dict: per-atom type / LJ / LK / charge arrays + a
			flat pair list with distances and connectivity weights
		'''
		atoms = pose.data['Atoms']
		bonds = pose.data['Bonds']
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		aas = pose.data.get('Amino Acids') or {}
		n = len(atoms)
		atom_types_db = params['Atom_types']
		residue_types_db = params['Residue_types']
		N_TERM_H = {'H1':'H','H2':'H','H3':'H','1H':'H','2H':'H','3H':'H',
			'HN':'H','HT1':'H','HT2':'H','HT3':'H'}
		def lookuptype(tricode, atom_name):
			'''
			Resolve (tricode, atom_name) to a Rosetta atom type and partial charge
			Arguments:
			----------
				tricode: str - 3-letter residue code
				atom_name: str - PDB atom name
			Returns:
			--------
				tuple: (atom_type_str_or_None, partial_charge_float)
			'''
			res = residue_types_db.get(tricode)
			if res is None: return None, 0.0
			aliases = res.get('aliases', {}) or {}
			direct = res['atoms'].get(atom_name)
			if direct is not None:
				return direct['type'], float(direct.get('charge', 0.0))
			al = aliases.get(atom_name)
			if al is not None:
				e = res['atoms'].get(al)
				if e is not None:
					return e['type'], float(e.get('charge', 0.0))
			swap = None
			if atom_name and atom_name[0].isdigit():
				swap = atom_name[1:] + atom_name[0]
			elif atom_name and atom_name[-1].isdigit() and len(atom_name) > 1:
				swap = atom_name[-1] + atom_name[:-1]
			if swap is not None:
				e = res['atoms'].get(swap)
				if e is not None:
					return e['type'], float(e.get('charge', 0.0))
			tgt = N_TERM_H.get(atom_name)
			if tgt is not None:
				e = res['atoms'].get(tgt)
				if e is not None:
					return e['type'], float(e.get('charge', 0.0))
			return None, 0.0
		atom_res = np.full(n, -1, dtype=np.int64)
		for r, info in aas.items():
			for ai in info[2] + info[3]:
				if 0 <= int(ai) < n:
					atom_res[int(ai)] = int(r)
		ros_types = [None] * n
		q_arr = np.zeros(n, dtype=np.float64)
		for ri, info in aas.items():
			tri = info[5] if len(info) >= 6 else None
			if tri is None: continue
			if tri.startswith('D') and len(tri) == 3:
				tri = tri
			if tri == 'HIS':
				res_atom_names = {atoms[int(ai)][0]
					for ai in (info[2] + info[3])
					if 0 <= int(ai) < n}
				has_hd1 = 'HD1' in res_atom_names
				has_he2 = 'HE2' in res_atom_names
				if has_hd1 and not has_he2:
					tri = 'HIS_D'
			for ai in info[2] + info[3]:
				ai = int(ai)
				if not (0 <= ai < n): continue
				nm = atoms[ai][0]
				t, q = lookuptype(tri, nm)
				ros_types[ai] = t
				q_arr[ai] = q
		ljR = np.zeros(n); ljW = np.zeros(n)
		lkdG = np.zeros(n); lkLam = np.ones(n) * 3.5
		lkVol = np.zeros(n)
		is_donor = np.zeros(n, dtype=bool)
		is_accep = np.zeros(n, dtype=bool)
		is_polar_h = np.zeros(n, dtype=bool)
		is_H = np.zeros(n, dtype=bool)
		is_oh_donor = np.zeros(n, dtype=bool)
		has_score = np.zeros(n, dtype=bool)
		for i in range(n):
			t = ros_types[i]
			if t is None or t not in atom_types_db: continue
			info = atom_types_db[t]
			ljR[i]  = info['LJ_RADIUS']
			ljW[i]  = info['LJ_WDEPTH']
			lkdG[i] = info['LK_DGFREE']
			lkLam[i]= info['LK_LAMBDA']
			lkVol[i]= info['LK_VOLUME']
			is_donor[i]   = bool(info.get('donor', False))
			is_accep[i]   = bool(info.get('acceptor', False))
			is_polar_h[i] = bool(info.get('polar_h', False))
			is_H[i]       = info.get('element') in ('H',) or t == 'HOH'
			is_oh_donor[i] = is_donor[i] and (t.startswith('OH')
				or t.startswith('OW') or t == 'Oet3')
			has_score[i] = True
		if aas:
			by_chain = {}
			for ri, info in aas.items():
				ch = info[1] if len(info) > 1 else ''
				by_chain.setdefault(ch, []).append(int(ri))
			n_term_res = set()
			c_term_res = set()
			for ch, ris in by_chain.items():
				ris.sort()
				if ris:
					n_term_res.add(ris[0])
					c_term_res.add(ris[-1])
			NTERM_H_NAMES = {'H', 'H1', 'H2', 'H3', '1H', '2H', '3H',
				'HN', 'HT1', 'HT2', 'HT3'}
			def applyatom(ai, new_type, new_charge):
				'''
				Patch one atom in place: overwrite type, charge, and derived per-atom arrays
				Arguments:
				----------
					ai: int - atom index
					new_type: str - new Rosetta atom type
					new_charge: float - new partial charge
				Returns:
				--------
					No return value
				'''
				ros_types[ai] = new_type
				q_arr[ai] = new_charge
				if new_type in atom_types_db:
					info = atom_types_db[new_type]
					ljR[ai]   = info['LJ_RADIUS']
					ljW[ai]   = info['LJ_WDEPTH']
					lkdG[ai]  = info['LK_DGFREE']
					lkLam[ai] = info['LK_LAMBDA']
					lkVol[ai] = info['LK_VOLUME']
					is_donor[ai] = bool(info.get('donor', False))
					is_accep[ai] = bool(info.get('acceptor', False))
					is_polar_h[ai] = bool(info.get('polar_h', False))
					is_H[ai] = info.get('element') in ('H',) \
						or new_type == 'HOH'
					has_score[ai] = True
			for ri in n_term_res:
				info = aas.get(ri)
				if info is None: continue
				tri = info[5] if len(info) >= 6 else None
				if tri == 'PRO':
					for ai in info[2] + info[3]:
						ai = int(ai)
						nm = atoms[ai][0]
						if nm in ('1H', '2H', 'H1', 'H2',
							'HN', 'HT1', 'HT2'):
							applyatom(ai, 'Hpol', 0.2142)
						elif nm == 'N':
							applyatom(ai, 'Nlys', -0.0285)
					continue
				if tri == 'GLY':
					for ai in info[2] + info[3]:
						ai = int(ai)
						nm = atoms[ai][0]
						if nm == 'N':
							applyatom(ai, 'Nlys', -0.2039)
						elif nm in NTERM_H_NAMES:
							applyatom(ai, 'Hpol', 0.2894)
						elif nm == 'CA':
							q_arr[ai] = 0.1328
						elif nm in ('HA', '1HA', '2HA',
							'HA1', 'HA2', 'HA3'):
							q_arr[ai] = 0.1015
					continue
				for ai in info[2] + info[3]:
					ai = int(ai)
					nm = atoms[ai][0]
					if nm == 'N':
						applyatom(ai, 'Nlys', -0.1987)
					elif nm in NTERM_H_NAMES:
						applyatom(ai, 'Hpol', 0.2946)
					elif nm == 'CA':
						q_arr[ai] = 0.2006
					elif nm == 'HA':
						q_arr[ai] = 0.1144
			NT_H_DIH = {
				'H': 180.0, 'H1': 180.0, '1H': 180.0,
				'HN': 180.0, 'HT1': 180.0,
				'H2': 60.0, '2H': 60.0, 'HT2': 60.0,
				'H3': -60.0, '3H': -60.0, 'HT3': -60.0}
			def idealize_nterm_h(ri):
				'''
				Override H positions for the N-term residue ri's NH3+
				protons in-place on `coords`.
				Arguments:
				----------
					ri: residue index (key into aas)
				Returns:
				--------
					No return; modifies `coords` rows for matched H atoms
				'''
				info = aas.get(ri)
				if info is None: return
				tri = info[5] if len(info) >= 6 else None
				if tri == 'PRO': return
				N_ai = CA_ai = C_ai = None
				h_atoms = []
				for ai in info[2] + info[3]:
					ai = int(ai)
					nm = atoms[ai][0]
					if nm == 'N': N_ai = ai
					elif nm == 'CA': CA_ai = ai
					elif nm == 'C': C_ai = ai
					elif nm in NT_H_DIH: h_atoms.append((ai, nm))
				if N_ai is None or CA_ai is None or C_ai is None: return
				if not h_atoms: return
				N_xyz = coords[N_ai]
				CA_xyz = coords[CA_ai]
				C_xyz = coords[C_ai]
				v_NCA = CA_xyz - N_xyz
				n_NCA = np.linalg.norm(v_NCA)
				if n_NCA < 1e-6: return
				unit_NCA = v_NCA / n_NCA
				v_NC = C_xyz - N_xyz
				v_NC_perp = v_NC - np.dot(v_NC, unit_NCA) * unit_NCA
				n_perp = np.linalg.norm(v_NC_perp)
				if n_perp < 1e-6: return
				unit_py = v_NC_perp / n_perp
				unit_pz = np.cross(unit_NCA, unit_py)
				BOND = 1.0
				ANGLE = math.radians(109.47)
				cos_a = math.cos(ANGLE)
				sin_a = math.sin(ANGLE)
				for ai, nm in h_atoms:
					dih = math.radians(NT_H_DIH[nm])
					H_dir = (cos_a * unit_NCA
						+ sin_a * (math.cos(dih) * unit_py
							+ math.sin(dih) * unit_pz))
					coords[ai] = N_xyz + BOND * H_dir
			for ri in n_term_res:
				idealize_nterm_h(ri)
			for ri in c_term_res:
				info = aas.get(ri)
				if info is None: continue
				c_ai = None; o_ai = None; oxt_ai = None
				for ai in info[2] + info[3]:
					ai = int(ai)
					nm = atoms[ai][0]
					if nm == 'C': c_ai = ai
					elif nm == 'O': o_ai = ai
					elif nm in ('OXT', 'OT1', 'OT2', "O''"): oxt_ai = ai
				if c_ai is not None:
					applyatom(c_ai, 'COO', 0.2158)
				if o_ai is not None:
					applyatom(o_ai, 'OOC', -0.6079)
				if oxt_ai is not None:
					applyatom(oxt_ai, 'OOC', -0.6079)
			sg_idx = [i for i in range(n)
				if atoms[i][1] == 'S' and ros_types[i] == 'SH1']
			X_arr = np.asarray(coords, dtype=np.float64)
			for ii in sg_idx:
				for jj in sg_idx:
					if ii >= jj: continue
					d = float(
						np.linalg.norm(X_arr[ii] - X_arr[jj]))
					if d < 2.5:
						q_arr[ii] = -0.24639810621738434
						q_arr[jj] = -0.24639810621738434
		adj = {int(k): set(int(j) for j in v) for k, v in bonds.items()}
		for i in range(n):
			adj.setdefault(i, set())
		X_arr = np.asarray(coords, dtype=np.float64)
		heavy_mask = np.array([atoms[k][1] != 'H' for k in range(n)])
		for i in range(n):
			if atoms[i][1] != 'H': continue
			dij = np.linalg.norm(X_arr - X_arr[i], axis=1)
			dij[i] = np.inf
			dij = np.where(heavy_mask, dij, np.inf)
			j = int(np.argmin(dij))
			if dij[j] < 1.3:
				adj[i].add(j); adj[j].add(i)
		s_idx = [i for i in range(n) if atoms[i][1] == 'S']
		for ii in s_idx:
			for jj in s_idx:
				if ii >= jj: continue
				d = float(np.linalg.norm(X_arr[ii] - X_arr[jj]))
				if d < 2.5:
					adj[ii].add(jj); adj[jj].add(ii)
		def bfsdists(start, max_depth=4):
			'''
			BFS bond-distance map from `start` up to max_depth bonds
			Arguments:
			----------
				start: int - atom index to start BFS from
				max_depth: int, default 4 - maximum bond distance to expand
			Returns:
			--------
				dict: {atom_index: bond_distance} for atoms at distance 1..max_depth
			'''
			out = {}
			frontier = {start}
			dist = 0
			seen = {start}
			while frontier and dist < max_depth:
				dist += 1
				nxt = set()
				for x in frontier:
					for y in adj.get(x, ()):
						if y in seen: continue
						seen.add(y); nxt.add(y)
						out[y] = dist
				frontier = nxt
			return out
		CP_REP_MAP_BY_AA = {
			'ALA': {'N': 'H', 'C': 'O'},
			'ARG': {'N': 'H', 'C': 'O', 'NE': 'HE'},
			'ASN': {'N': 'H', 'C': 'O', 'CG': 'OD1', 'ND2': 'HD21'},
			'ASP': {'N': 'H', 'C': 'O', 'CG': 'OD1'},
			'CYS': {'N': 'H', 'C': 'O'},
			'GLN': {'N': 'H', 'C': 'O', 'CD': 'OE1', 'NE2': 'HE21'},
			'GLU': {'N': 'H', 'C': 'O', 'CD': 'OE1'},
			'GLY': {'N': 'H', 'C': 'O'},
			'HIS': {'N': 'H', 'C': 'O', 'NE2': 'HE2'},
			'HIS_D': {'N': 'H', 'C': 'O', 'ND1': 'HD1'},
			'ILE': {'N': 'H', 'C': 'O'},
			'LEU': {'N': 'H', 'C': 'O'},
			'LYS': {'N': 'H', 'C': 'O', 'NZ': 'HZ1'},
			'MET': {'N': 'H', 'C': 'O'},
			'PHE': {'N': 'H', 'C': 'O'},
			'PRO': {'C': 'O'},
			'SER': {'N': 'H', 'C': 'O', 'OG': 'HG'},
			'THR': {'N': 'H', 'C': 'O', 'OG1': 'HG1'},
			'TRP': {'N': 'H', 'C': 'O', 'NE1': 'HE1'},
			'TYR': {'N': 'H', 'C': 'O', 'OH': 'HH'},
			'VAL': {'N': 'H', 'C': 'O'},
		}
		rep_atom_idx = np.arange(n, dtype=np.int64)
		if aas:
			for ri, info in aas.items():
				tri = info[5] if len(info) >= 6 else None
				rep_map = CP_REP_MAP_BY_AA.get(tri)
				if rep_map is None: continue
				res_atoms = {}
				for ai in info[2] + info[3]:
					ai = int(ai)
					if 0 <= ai < n:
						res_atoms[atoms[ai][0]] = ai
				is_nterm = ri in n_term_res
				for src_nm, tgt_nm in rep_map.items():
					if is_nterm and src_nm == 'N': continue
					src_ai = res_atoms.get(src_nm)
					tgt_ai = res_atoms.get(tgt_nm)
					if src_ai is not None and tgt_ai is not None:
						rep_atom_idx[src_ai] = tgt_ai
		res_bonded = set()
		res_polymer_bonded = set()
		for ai, neighbors in adj.items():
			ra = atom_res[ai] if 0 <= ai < n else -1
			if ra < 0: continue
			nm_a = atoms[ai][0]
			for bi in neighbors:
				rb = atom_res[bi] if 0 <= bi < n else -1
				if rb < 0 or rb == ra: continue
				nm_b = atoms[bi][0]
				pair = (min(ra, rb), max(ra, rb))
				res_bonded.add(pair)
				if ((nm_a == 'C' and nm_b == 'N')
						or (nm_a == 'N' and nm_b == 'C')):
					res_polymer_bonded.add(pair)
		c0 = float(params['Constants'].get('fa_max_dis', 6.0))
		pairs_i = []; pairs_j = []; pair_d = []
		pair_w = []; pair_same_res = []; pair_path = []
		pair_cp_path = []; pair_is_poly = []
		bfs_cache = {}
		def get_bfs(atom_idx):
			'''
			Memoised wrapper around bfsdists keyed on atom_idx
			Arguments:
			----------
				atom_idx: int - atom index to BFS from
			Returns:
			--------
				dict: {atom_index: bond_distance} (same shape as bfsdists)
			'''
			if atom_idx not in bfs_cache:
				bfs_cache[atom_idx] = bfsdists(int(atom_idx), max_depth=4)
			return bfs_cache[atom_idx]
		typed_idx = np.where(has_score)[0]
		X = coords
		for ii in typed_idx:
			dists_from_ii = get_bfs(int(ii))
			dd = np.linalg.norm(X - X[ii], axis=1)
			mask = (dd < c0) & has_score
			mask[ii] = False
			ri = atom_res[ii]
			for jj in np.where(mask)[0]:
				if jj < ii: continue
				rj = atom_res[jj]
				rpair = (min(ri,rj), max(ri,rj))
				same_or_adj = (ri == rj or rpair in res_bonded)
				is_poly = (ri == rj) or (rpair in res_polymer_bonded)
				if same_or_adj:
					bd = dists_from_ii.get(int(jj), 5)
				else:
					bd = 5
				if is_poly:
					if bd <= 3: w = 0.0
					elif bd == 4: w = 0.2
					else: w = 1.0
				else:
					if bd <= 2: w = 0.0
					elif bd == 3: w = 0.2
					else: w = 1.0
				rep_i = int(rep_atom_idx[ii])
				rep_j = int(rep_atom_idx[jj])
				if not same_or_adj:
					cp_bd = 5
				elif rep_i == rep_j:
					cp_bd = 0
				elif rep_i == int(ii) and rep_j == int(jj):
					cp_bd = bd
				else:
					rep_bfs = get_bfs(rep_i)
					cp_bd = rep_bfs.get(rep_j, 5)
				pairs_i.append(int(ii))
				pairs_j.append(int(jj))
				pair_d.append(float(dd[jj]))
				pair_w.append(w)
				pair_same_res.append(ri == rj and ri >= 0)
				pair_path.append(bd)
				pair_cp_path.append(cp_bd)
		pairs_i = np.array(pairs_i, dtype=np.int64)
		pairs_j = np.array(pairs_j, dtype=np.int64)
		pair_d = np.array(pair_d, dtype=np.float64)
		pair_w = np.array(pair_w, dtype=np.float64)
		pair_same_res = np.array(pair_same_res, dtype=bool)
		pair_path = np.array(pair_path, dtype=np.int64)
		pair_cp_path = np.array(pair_cp_path, dtype=np.int64)
		LKB_WTS = {
			'NH2O': (-0.462, 1.075),
			'Narg': (-0.444, 1.111),
			'Nhis': (-0.254, 0.746),
			'Nlys': (-0.367, 0.633),
			'Ntrp': (-0.231, 0.769),
			'OCbb': (-0.329, 0.671),
			'OH':   (-0.401, 0.599),
			'ONH2': (-0.329, 0.671),
			'OOC':  (-0.306, 0.694)}
		LK_RAMP_W2 = 3.9
		H2O_R = 1.4
		lkb_w_iso = np.zeros(n, dtype=np.float64)
		lkb_w_ball = np.zeros(n, dtype=np.float64)
		lkb_d2_low = np.zeros(n, dtype=np.float64)
		for i in range(n):
			t = ros_types[i]
			if t in LKB_WTS:
				lkb_w_iso[i], lkb_w_ball[i] = LKB_WTS[t]
			if t is not None and t in atom_types_db:
				ljr = atom_types_db[t]['LJ_RADIUS']
				d2h = (H2O_R + ljr) * (H2O_R + ljr)
				lkb_d2_low[i] = max(0.0, d2h - LK_RAMP_W2)
		etb = DBLoad().get('EtablePairParams')
		if etb is not None:
			et_names = list(etb['atom_types'])
			NT = int(etb['n_types'])
			et_pairs = etb['pairs']
			et_name_to_eidx = {t: i for i, t in enumerate(et_names)}
			et_close_start = np.zeros((NT, NT), dtype=np.float64)
			et_close_end   = np.zeros((NT, NT), dtype=np.float64)
			et_close_flat  = np.zeros((NT, NT), dtype=np.float64)
			et_close_poly  = np.zeros((NT, NT, 4), dtype=np.float64)
			et_far_poly    = np.zeros((NT, NT, 4), dtype=np.float64)
			et_lk_coeff    = np.zeros((NT, NT), dtype=np.float64)
			et_lambda_self = np.ones((NT, NT), dtype=np.float64) * 3.5
			et_R_self      = np.zeros((NT, NT), dtype=np.float64)
			et_final_w     = np.ones((NT, NT), dtype=np.float64)
			et_close_flat_comb = np.zeros((NT, NT), dtype=np.float64)
			et_close_poly_comb = np.zeros((NT, NT, 4), dtype=np.float64)
			et_far_poly_comb   = np.zeros((NT, NT, 4), dtype=np.float64)
			et_lj_minimum            = np.zeros((NT, NT), dtype=np.float64)
			et_lj_r12_coeff          = np.zeros((NT, NT), dtype=np.float64)
			et_lj_r6_coeff           = np.zeros((NT, NT), dtype=np.float64)
			et_lj_switch_intercept   = np.zeros((NT, NT), dtype=np.float64)
			et_lj_switch_slope       = np.zeros((NT, NT), dtype=np.float64)
			et_lj_val_at_minimum     = np.zeros((NT, NT), dtype=np.float64)
			et_ljatr_cubic_poly      = np.zeros((NT, NT, 4), dtype=np.float64)
			et_ljatr_cp_xhi          = np.zeros((NT, NT), dtype=np.float64)
			et_ljatr_cp_xlo          = np.zeros((NT, NT), dtype=np.float64)
			et_ljatr_final_weight    = np.ones((NT, NT), dtype=np.float64)
			et_ljrep_linear_ramp_d2  = np.zeros((NT, NT), dtype=np.float64)
			et_ljrep_from_negcrossing = np.zeros((NT, NT), dtype=bool)
			et_hydrogen_interaction  = np.zeros((NT, NT), dtype=bool)
			et_ljrep_xr_xlo   = np.zeros((NT, NT), dtype=np.float64)
			et_ljrep_xr_xhi   = np.zeros((NT, NT), dtype=np.float64)
			et_ljrep_xr_slope = np.zeros((NT, NT), dtype=np.float64)
			et_ljrep_xr_extrap_slope = np.zeros((NT, NT), dtype=np.float64)
			et_ljrep_xr_ylo   = np.zeros((NT, NT), dtype=np.float64)
			et_has         = np.zeros((NT, NT), dtype=bool)
			for is_ in range(NT):
				for io_ in range(NT):
					c = et_pairs[is_ * NT + io_]
					if c is None: continue
					et_close_start[is_, io_] = c['close_start']
					et_close_end[is_, io_]   = c['close_end']
					et_close_flat[is_, io_]  = c['close_flat']
					et_close_poly[is_, io_]  = c['close_poly']
					et_far_poly[is_, io_]    = c['far_poly']
					et_lk_coeff[is_, io_]    = c['lk_coeff']
					et_lambda_self[is_, io_] = c['lambda_self']
					et_R_self[is_, io_]      = c['R_self']
					et_final_w[is_, io_]     = c['final_weight']
					if 'close_flat_comb' in c:
						et_close_flat_comb[is_, io_] = c['close_flat_comb']
						et_close_poly_comb[is_, io_] = c['close_poly_comb']
						et_far_poly_comb[is_, io_]   = c['far_poly_comb']
					if 'lj_minimum' in c:
						et_lj_minimum[is_, io_]          = c['lj_minimum']
						et_lj_r12_coeff[is_, io_]        = c['lj_r12_coeff']
						et_lj_r6_coeff[is_, io_]         = c['lj_r6_coeff']
						et_lj_switch_intercept[is_, io_] = c['lj_switch_intercept']
						et_lj_switch_slope[is_, io_]     = c['lj_switch_slope']
						et_lj_val_at_minimum[is_, io_]   = c['lj_val_at_minimum']
						et_ljatr_cubic_poly[is_, io_]    = c['ljatr_cubic_poly']
						et_ljatr_cp_xhi[is_, io_]        = c['ljatr_cubic_poly_xhi']
						et_ljatr_cp_xlo[is_, io_]        = c['ljatr_cubic_poly_xlo']
						et_ljatr_final_weight[is_, io_]  = c['ljatr_final_weight']
						et_ljrep_linear_ramp_d2[is_, io_] = c['ljrep_linear_ramp_d2_cutoff']
						et_ljrep_from_negcrossing[is_, io_] = c['ljrep_from_negcrossing']
						et_hydrogen_interaction[is_, io_] = c['hydrogen_interaction']
						et_ljrep_xr_xlo[is_, io_]   = c['ljrep_xr_xlo']
						et_ljrep_xr_xhi[is_, io_]   = c['ljrep_xr_xhi']
						et_ljrep_xr_slope[is_, io_] = c['ljrep_xr_slope']
						et_ljrep_xr_extrap_slope[is_, io_] = c['ljrep_xr_extrapolated_slope']
						et_ljrep_xr_ylo[is_, io_]   = c['ljrep_xr_ylo']
					et_has[is_, io_] = True
			at_e_idx = np.full(n, -1, dtype=np.int64)
			for i in range(n):
				t = ros_types[i]
				if t in et_name_to_eidx:
					at_e_idx[i] = et_name_to_eidx[t]
		else:
			NT = 0
			at_e_idx = np.full(n, -1, dtype=np.int64)
			et_close_start = et_close_end = et_close_flat = None
			et_close_poly = et_far_poly = et_lk_coeff = None
			et_lambda_self = et_R_self = et_final_w = et_has = None
		opt_dist = 2.65
		ang_sp2 = math.radians(60.0)   # 180 - 120
		ang_sp3 = math.radians(71.0)   # 180 - 109
		dih_sp2 = (0.0, math.radians(180.0))
		dih_sp3 = (math.radians(120.0), math.radians(240.0))
		water_xyz = []
		water_atom = []
		water_off = np.full(n, -1, dtype=np.int64)
		water_cnt = np.zeros(n, dtype=np.int64)
		def unit(v):
			'''
			Unit vector along v; zero vector when v has zero length
			Arguments:
			----------
				v: np.ndarray - 3-component vector
			Returns:
			--------
				np.ndarray: v / |v|, or v itself if |v| is zero
			'''
			nv = float(np.linalg.norm(v))
			return v / nv if nv > 1e-9 else v
		for i in range(n):
			t = ros_types[i]
			if t not in LKB_WTS: continue
			info = atom_types_db.get(t, {})
			is_d = bool(info.get('donor', False))
			is_a = bool(info.get('acceptor', False))
			is_sp2 = bool(info.get('sp2', False))
			is_sp3 = bool(info.get('sp3', False))
			is_ring = bool(info.get('ring', False))
			i_xyz = X[i]
			i_waters = []
			nbrs = adj.get(i, set())
			heavy_nbrs = [j for j in nbrs if not is_H[j]]
			polar_h_nbrs = [j for j in nbrs if is_polar_h[j]]
			if is_d:
				elem = info.get('element', '')
				NTERM_H_NAMES_SET = {'H1', 'H2', 'H3',
					'1H', '2H', '3H', 'HN', 'HT1', 'HT2', 'HT3'}
				for h in polar_h_nbrs:
					h_nm = atoms[h][0]
					if elem == 'O':
						ideal_bond = 0.96
					elif t == 'NH2O':
						ideal_bond = 1.00
					elif h_nm in NTERM_H_NAMES_SET:
						ideal_bond = 1.00
					else:
						ideal_bond = 1.01
					offset = opt_dist - ideal_bond
					h_xyz = X[h]
					dirvec = unit(h_xyz - i_xyz)
					w = h_xyz + offset * dirvec
					i_waters.append(w)
			if is_a:
				if is_ring and len(heavy_nbrs) >= 2:
					c1, c2 = heavy_nbrs[0], heavy_nbrs[1]
					mid = 0.5 * (X[c1] + X[c2])
					w = i_xyz + opt_dist * unit(i_xyz - mid)
					i_waters.append(w)
				elif is_sp3 and len(heavy_nbrs) >= 1 and \
						len(polar_h_nbrs) >= 1:
					c = heavy_nbrs[0]; h = polar_h_nbrs[0]
					x_hat = unit(i_xyz - X[c])
					v_OH = X[h] - i_xyz
					y_dir = v_OH - np.dot(v_OH, x_hat) * x_hat
					y_hat = unit(y_dir)
					z_hat = np.cross(x_hat, y_hat)
					cos_a = math.cos(ang_sp3)
					sin_a = math.sin(ang_sp3)
					for d in dih_sp3:
						v_off = (cos_a * x_hat
							+ sin_a * (math.cos(d) * y_hat
								+ math.sin(d) * z_hat))
						i_waters.append(i_xyz + opt_dist * v_off)
				elif is_sp2 and len(heavy_nbrs) >= 1:
					c = heavy_nbrs[0]
					c_heavy_nbrs = [k for k in adj.get(c, set())
						if k != i and not is_H[k]]
					if not c_heavy_nbrs: continue
					my_res = atom_res[i]
					same_res_nbrs = sorted(
						k for k in c_heavy_nbrs if atom_res[k] == my_res)
					if same_res_nbrs:
						b2 = same_res_nbrs[0]
					else:
						b2 = sorted(c_heavy_nbrs)[0]
					x_hat = unit(i_xyz - X[c])
					v_b2 = X[b2] - i_xyz
					y_dir = v_b2 - np.dot(v_b2, x_hat) * x_hat
					y_hat = unit(y_dir)
					z_hat = np.cross(x_hat, y_hat)
					cos_a = math.cos(ang_sp2)
					sin_a = math.sin(ang_sp2)
					for d in dih_sp2:
						v_off = (cos_a * x_hat
							+ sin_a * (math.cos(d) * y_hat
								+ math.sin(d) * z_hat))
						i_waters.append(i_xyz + opt_dist * v_off)
			if i_waters:
				water_off[i] = len(water_xyz)
				water_cnt[i] = len(i_waters)
				for w in i_waters:
					water_xyz.append(np.asarray(w, dtype=np.float64))
					water_atom.append(i)
		if water_xyz:
			water_xyz_arr = np.stack(water_xyz, axis=0)
		else:
			water_xyz_arr = np.empty((0, 3), dtype=np.float64)
		water_atom = np.array(water_atom, dtype=np.int64)
		return {
			'ros_types': ros_types,
			'has_score': has_score,
			'charges':   q_arr,
			'lj_R':      ljR,
			'lj_W':      ljW,
			'lk_dG':     lkdG,
			'lk_lambda': lkLam,
			'lk_volume': lkVol,
			'is_donor':  is_donor,
			'is_accep':  is_accep,
			'is_polar_h':is_polar_h,
			'is_H':      is_H,
			'is_oh_donor': is_oh_donor,
			'coords':    X,
			'atom_res':  atom_res,
			'pairs_i':   pairs_i,
			'pairs_j':   pairs_j,
			'pair_d':    pair_d,
			'pair_w':    pair_w,
			'pair_same_res': pair_same_res,
			'pair_path': pair_path,
			'pair_cp_path': pair_cp_path,
			'rep_atom_idx': rep_atom_idx,
			'lkb_w_iso':   lkb_w_iso,
			'lkb_w_ball':  lkb_w_ball,
			'lkb_d2_low':  lkb_d2_low,
			'lkb_water_xyz': water_xyz_arr,
			'lkb_water_off': water_off,
			'lkb_water_cnt': water_cnt,
			'lkb_ramp_w2': LK_RAMP_W2,
			'at_e_idx':      at_e_idx,
			'et_close_start':et_close_start,
			'et_close_end':  et_close_end,
			'et_close_flat': et_close_flat,
			'et_close_poly': et_close_poly,
			'et_far_poly':   et_far_poly,
			'et_lk_coeff':   et_lk_coeff,
			'et_lambda_self':et_lambda_self,
			'et_R_self':     et_R_self,
			'et_final_w':    et_final_w,
			'et_close_flat_comb': et_close_flat_comb,
			'et_close_poly_comb': et_close_poly_comb,
			'et_far_poly_comb':   et_far_poly_comb,
			'et_lj_minimum':          et_lj_minimum,
			'et_lj_r12_coeff':        et_lj_r12_coeff,
			'et_lj_r6_coeff':         et_lj_r6_coeff,
			'et_lj_switch_intercept': et_lj_switch_intercept,
			'et_lj_switch_slope':     et_lj_switch_slope,
			'et_lj_val_at_minimum':   et_lj_val_at_minimum,
			'et_ljatr_cubic_poly':    et_ljatr_cubic_poly,
			'et_ljatr_cp_xhi':        et_ljatr_cp_xhi,
			'et_ljatr_cp_xlo':        et_ljatr_cp_xlo,
			'et_ljatr_final_weight':  et_ljatr_final_weight,
			'et_ljrep_linear_ramp_d2': et_ljrep_linear_ramp_d2,
			'et_ljrep_from_negcrossing': et_ljrep_from_negcrossing,
			'et_hydrogen_interaction': et_hydrogen_interaction,
			'et_ljrep_xr_xlo':   et_ljrep_xr_xlo,
			'et_ljrep_xr_xhi':   et_ljrep_xr_xhi,
			'et_ljrep_xr_slope': et_ljrep_xr_slope,
			'et_ljrep_xr_extrap_slope': et_ljrep_xr_extrap_slope,
			'et_ljrep_xr_ylo':   et_ljrep_xr_ylo,
			'et_has':        et_has,
			'adj':       adj}
	def bfswithin(adj, start, depth):
		'''
		Return the set of atoms within `depth` bonds of `start` (inclusive)
		Arguments:
		----------
			adj:   dict - adjacency map {int: set(int)}
			start: int  - root atom
			depth: int  - bond-depth limit (1-2 is depth 1, 1-4 is depth 3)
		Returns:
		--------
			set of int: atoms reachable within `depth` bonds, including start
		'''
		visited = {start}
		frontier = {start}
		for _ in range(depth):
			nxt = set()
			for x in frontier:
				for y in adj.get(x, ()):
					if y not in visited:
						visited.add(y); nxt.add(y)
			frontier = nxt
			if not frontier: break
		return visited
	def countnrot(ligand):
		'''
		Count non-terminal, non-ring, non-amide single bonds in a ligand
		(Nrot for AutoDock Vina's conf-independent term).
		A bond is rotatable iff:
		  - single bond order
		  - not a ring (= bridge-edge: removing it disconnects its endpoints)
		  - both ends have at least one heavy-atom neighbour besides each
		    other (i.e. not terminal)
		  - not an amide / amidine C-N (where C is sp2-bonded to O or N)
		Arguments:
		----------
			ligand: Molecule - the ligand
		Returns:
		--------
			int: estimated Nrot
		'''
		atoms = ligand.data['Atoms']
		bonds = ligand.data['Bonds']
		orders = ligand.data.get('BondOrders', {})
		def bondorder(a, b):
			'''
			Bond order between atoms i and j in the working bond-order table
			Arguments:
			----------
				i: int - atom index
				j: int - atom index
			Returns:
			--------
				float: bond order, defaulting to 1.0 when no record exists
			'''
			ol = orders.get(a, [])
			nl = bonds[a]
			if len(ol) == len(nl):
				try: return ol[nl.index(b)]
				except ValueError: return 1
			return 1
		def inring(a, b):
			# BFS from a, excluding the direct a-b edge; if we still
			# reach b, the bond is part of a cycle (ring bond).
			'''
			True iff the edge (i, j) lies in a ring of the working topology
			Arguments:
			----------
				i: int - atom index
				j: int - atom index
			Returns:
			--------
				bool: True if the bond is part of any detected ring
			'''
			seen = {a}
			stk = [a]
			while stk:
				x = stk.pop()
				for y in bonds.get(x, []):
					if (x == a and y == b) or (x == b and y == a):
						continue
					if y in seen: continue
					if y == b: return True
					seen.add(y)
					stk.append(y)
			return False
		def isamide(a, b):
			# C-N where the C is acyclic and double-bonded to O (true
			# amide carbonyl). AutoDock's torsion tree leaves these rigid.
			# Aromatic ring C-N bonds are NOT amides — they get caught by
			# the ring/inring filter instead.
			'''
			True iff the bond (i, j) is the C-N of an amide (C also bonded to a =O)
			Arguments:
			----------
				i: int - atom index
				j: int - atom index
			Returns:
			--------
				bool: True for amide C-N bonds
			'''
			def amide(c_idx, n_idx):
				# Amide / amidine: C has a double bond to O or N besides
				# n_idx, and the C-N bond itself is acyclic (so it's not
				# an aromatic ring substituent like pyridine).
				'''
				True iff atom `c` is the carbonyl C of an amide: bonded to N (the other atom) and a =O
				Arguments:
				----------
					c: int - candidate carbonyl-carbon atom index
					n: int - candidate nitrogen atom index
				Returns:
				--------
					bool: True if c is an amide carbonyl carbon
				'''
				if atoms[c_idx][1] != 'C': return False
				if atoms[n_idx][1] != 'N': return False
				# C must not have any cyclic bond (else aromatic carbon)
				for k in bonds.get(c_idx, []):
					if inring(c_idx, k): return False
				for k in bonds.get(c_idx, []):
					if k == n_idx: continue
					if atoms[k][1] not in ('O', 'N'): continue
					if bondorder(c_idx, k) >= 2: return True
				return False
			return amide(a, b) or amide(b, a)
		nrot = 0
		seen = set()
		for i in sorted(bonds):
			for j in bonds[i]:
				if j <= i: continue
				key = (i, j)
				if key in seen: continue
				seen.add(key)
				if atoms[i][1] == 'H' or atoms[j][1] == 'H': continue
				if bondorder(i, j) != 1: continue
				if inring(i, j): continue
				hvi = [k for k in bonds[i]
					if atoms[k][1] != 'H' and k != j]
				hvj = [k for k in bonds[j]
					if atoms[k][1] != 'H' and k != i]
				has_h_i = any(atoms[k][1] == 'H' for k in bonds[i]
					if k != j)
				has_h_j = any(atoms[k][1] == 'H' for k in bonds[j]
					if k != i)
				# Skip if either endpoint is a "no rotation effect"
				# terminal: terminal C (methyl-like) or terminal heavy
				# atom with no H attached (halogen, lone substituent).
				# Terminal O/N/S with H is still rotatable (H direction
				# matters for H-bonding).
				if not hvi and (atoms[i][1] == 'C' or not has_h_i):
					continue
				if not hvj and (atoms[j][1] == 'C' or not has_h_j):
					continue
				if isamide(i, j): continue
				nrot += 1
		return nrot
	def countnumtors(ligand):
		'''
		Compute the Vina conf-independent term's "num_tors" input:
		sum over rotatable bonds of 0.5 from each side, where each side
		contributes 0.5 only if it has > 1 heavy non-H neighbour. So a
		regular rotation between two heavy-substituted carbons contributes
		1.0, while a rotation where one side has only one heavy neighbour
		(like a carboxyl C-OH) contributes 0.5. This is the actual quantity
		used in Vina's affinity denominator `1 + 0.05846 * num_tors`.
		Arguments:
		----------
			ligand: Molecule
		Returns:
		--------
			float: Vina num_tors (typically equal to Nrot or Nrot-k/2)
		'''
		atoms = ligand.data['Atoms']
		bonds = ligand.data['Bonds']
		orders = ligand.data.get('BondOrders', {})
		def bondorder(a, b):
			'''
			Bond order between atoms i and j in the working bond-order table
			Arguments:
			----------
				i: int - atom index
				j: int - atom index
			Returns:
			--------
				float: bond order, defaulting to 1.0 when no record exists
			'''
			ol = orders.get(a, [])
			nl = bonds[a]
			if len(ol) == len(nl):
				try: return ol[nl.index(b)]
				except ValueError: return 1
			return 1
		def inring(a, b):
			'''
			True iff the edge (i, j) lies in a ring of the working topology
			Arguments:
			----------
				i: int - atom index
				j: int - atom index
			Returns:
			--------
				bool: True if the bond is part of any detected ring
			'''
			seen = {a}; stk = [a]
			while stk:
				x = stk.pop()
				for y in bonds.get(x, []):
					if (x == a and y == b) or (x == b and y == a):
						continue
					if y in seen: continue
					if y == b: return True
					seen.add(y); stk.append(y)
			return False
		def isamide(a, b):
			'''
			True iff the bond (i, j) is the C-N of an amide (C also bonded to a =O)
			Arguments:
			----------
				i: int - atom index
				j: int - atom index
			Returns:
			--------
				bool: True for amide C-N bonds
			'''
			def amide(c_idx, n_idx):
				'''
				True iff atom `c` is the carbonyl C of an amide: bonded to N (the other atom) and a =O
				Arguments:
				----------
					c: int - candidate carbonyl-carbon atom index
					n: int - candidate nitrogen atom index
				Returns:
				--------
					bool: True if c is an amide carbonyl carbon
				'''
				if atoms[c_idx][1] != 'C': return False
				if atoms[n_idx][1] != 'N': return False
				for k in bonds.get(c_idx, []):
					if inring(c_idx, k): return False
				for k in bonds.get(c_idx, []):
					if k == n_idx: continue
					if atoms[k][1] not in ('O', 'N'): continue
					if bondorder(c_idx, k) >= 2: return True
				return False
			return amide(a, b) or amide(b, a)
		num_tors = 0.0
		seen = set()
		for i in sorted(bonds):
			for j in bonds[i]:
				if j <= i: continue
				if (i, j) in seen: continue
				seen.add((i, j))
				if atoms[i][1] == 'H' or atoms[j][1] == 'H': continue
				if bondorder(i, j) != 1: continue
				if inring(i, j): continue
				hvi = [k for k in bonds[i]
					if atoms[k][1] != 'H' and k != j]
				hvj = [k for k in bonds[j]
					if atoms[k][1] != 'H' and k != i]
				has_h_i = any(atoms[k][1] == 'H' for k in bonds[i]
					if k != j)
				has_h_j = any(atoms[k][1] == 'H' for k in bonds[j]
					if k != i)
				if not hvi and (atoms[i][1] == 'C' or not has_h_i):
					continue
				if not hvj and (atoms[j][1] == 'C' or not has_h_j):
					continue
				if isamide(i, j): continue
				# 0.5 per side where the side has > 1 heavy neighbour
				if len(hvi) >= 1: num_tors += 0.5
				if len(hvj) >= 1: num_tors += 0.5
		return num_tors
	def ringatoms(bonds):
		'''
		Identify atoms that belong to any ring via DFS back-edge detection
		Arguments:
		----------
			bonds: dict - adjacency map {int: list(int)}
		Returns:
		--------
			set of int: atom indices on any cycle
		'''
		visited = set()
		parent = {}
		ring = set()
		for root in sorted(bonds):
			if root in visited: continue
			stk = [(root, None)]
			while stk:
				node, par = stk.pop()
				if node in visited:
					if par is not None and par != parent.get(node):
						a = par; b = node
						while a is not None and a != b:
							ring.add(a); a = parent.get(a)
						ring.add(b)
					continue
				visited.add(node); parent[node] = par
				for nb in bonds.get(node, []):
					if nb == par: continue
					if nb in visited:
						a = node
						while a is not None and a != nb:
							ring.add(a); a = parent.get(a)
						ring.add(nb)
					else:
						stk.append((nb, node))
		return ring
	def topologyhash(pose, ligand=None):
		'''
		Deterministic hash of the pose and optional ligand
		Arguments:
		----------
			pose:   Pose or Molecule
			ligand: Molecule or None
		Returns:
		--------
			int: hash used for cache invalidation
		'''
		def keyof(obj):
			'''
			Build the canonical (lo_type, hi_type) tuple key for typed-pair lookups
			Arguments:
			----------
				ti: int - first atom type code
				tj: int - second atom type code
			Returns:
			--------
				tuple: (min(ti,tj), max(ti,tj))
			'''
			if obj is None: return None
			a = tuple((int(k), tuple(v))
				for k, v in sorted(obj.data['Atoms'].items()))
			b = tuple((int(k), tuple(sorted(int(j) for j in v)))
				for k, v in sorted(obj.data['Bonds'].items()))
			return (a, b)
		return hash((params.get('_name',''), keyof(pose), keyof(ligand)))
	def evalpairs(cache, kind, pair_fn):
		'''
		Apply a per-pair function and sum its result over inter and
		intra-ligand pair lists, gating on cutoff
		Arguments:
		----------
			cache:   dict - PatternSearch result
			kind:    str  - 'inter' or 'intra' or 'both'
			pair_fn: callable - takes (ai, aj, rij, cache) and returns
				a per-pair contribution array
		Returns:
		--------
			tuple (inter_sum, intra_sum)
		'''
		def go(pairs):
			'''
			Inner recursion driving the typed-pair sum over a precomputed pair list
			Arguments:
			----------
				lo: int - starting pair index
				hi: int - end pair index (exclusive)
			Returns:
			--------
				No return value (mutates the enclosing accumulator arrays)
			'''
			if len(pairs) == 0: return 0.0
			ai = pairs[:, 0]; aj = pairs[:, 1]
			coords = cache['coords']
			rij = np.linalg.norm(coords[ai] - coords[aj], axis=1)
			return float(pair_fn(ai, aj, rij, cache).sum())
		inter_sum = go(cache['inter_pairs'])
		intra_sum = go(cache['intra_ligand_pairs'])
		return inter_sum, intra_sum
	def termresult(inter_raw, intra_raw, weight):
		'''
		Pack one term's contribution into the standard dict shape
		Arguments:
		----------
			inter_raw: float - unweighted intermolecular sum
			intra_raw: float - unweighted intramolecular sum
			weight:    float - kJ-scale weight (stored in DB x4.184)
		Returns:
		--------
			dict: 'inter_raw', 'intra_raw', 'inter_weighted',
			'intra_weighted' (the last two are inter_raw/intra_raw
			x weight, in kJ/mol)
		'''
		return {
			'inter_raw': inter_raw, 'intra_raw': intra_raw,
			'inter_weighted': inter_raw * weight,
			'intra_weighted': intra_raw * weight}
	def gausspair(cache, key):
		'''
		Evaluate one Vina gaussian term and pack the result
		Arguments:
		----------
			cache: dict - PatternSearch result
			key:   str  - 'Gauss1' or 'Gauss2'
		Returns:
		--------
			dict: term result via _termresult
		'''
		p = params[key]
		offset = float(p['offset']); width = float(p['width'])
		cutoff = float(p['cutoff']); weight = float(p['weight'])
		radii = cache['xs_radii_arr']; xs = cache['xs_types']
		def fn(ai, aj, rij, c):
			'''
			Per-pair compute kernel: maps (ai, aj, rij, w) into a scalar contribution
			Arguments:
			----------
				ai: np.ndarray - per-pair first-atom indices
				aj: np.ndarray - per-pair second-atom indices
				rij: np.ndarray - per-pair distance
				c: np.ndarray - per-pair connectivity weight
			Returns:
			--------
				np.ndarray: per-pair scalar contribution to the term
			'''
			ri = radii[xs[ai]]; rj = radii[xs[aj]]
			d = rij - (ri + rj + offset)
			gate = ((xs[ai] >= 0) & (xs[aj] >= 0) & (rij < cutoff))
			return np.where(gate, np.exp(-(d / width) ** 2), 0.0)
		inter_raw, intra_raw = evalpairs(cache, 'both', fn)
		return termresult(inter_raw, intra_raw, weight)
	def slopestep(cache, key, mode):
		'''
		Evaluate a slope_step term (hydrophobic or h-bond)
		Arguments:
		----------
			cache: dict - PatternSearch result
			key:   str  - 'Hydrophobic' or 'HBond'
			mode:  str  - 'hydrophobic' or 'hbond'
		Returns:
		--------
			dict: term result via _termresult
		'''
		p = params[key]
		good = float(p['good']); bad = float(p['bad'])
		cutoff = float(p['cutoff']); weight = float(p['weight'])
		radii = cache['xs_radii_arr']; xs = cache['xs_types']
		hphob = cache['xs_is_hydrophobic_arr']
		donor = cache['xs_is_donor_arr']
		accep = cache['xs_is_acceptor_arr']
		def slopestep(x):
			'''
			Linear slope-step: 1 below `good`, 0 above `bad`, linear ramp in between
			Arguments:
			----------
				cache: dict - ScoreMatch cache
				pkey: str - parameter section key (e.g. Hydrophobic, HBond)
				mask: str - cache flag selecting which atom subtype to sum over
			Returns:
			--------
				dict: termresult with inter/intra raw and weighted contributions
			'''
			if bad < good:
				return np.clip((x - bad) / (good - bad), 0.0, 1.0)
			elif bad > good:
				return np.clip((bad - x) / (bad - good), 0.0, 1.0)
			else:
				return np.where(x <= good, 1.0, 0.0)
		def fn(ai, aj, rij, c):
			'''
			Per-pair slope-step kernel used by the slopestep wrapper
			Arguments:
			----------
				ai: np.ndarray - per-pair first-atom indices
				aj: np.ndarray - per-pair second-atom indices
				rij: np.ndarray - per-pair distance
				c: np.ndarray - per-pair connectivity weight
			Returns:
			--------
				np.ndarray: per-pair slope-step contribution
			'''
			ri = radii[xs[ai]]; rj = radii[xs[aj]]
			d = rij - (ri + rj)
			valid = (xs[ai] >= 0) & (xs[aj] >= 0) & (rij < cutoff)
			if mode == 'hydrophobic':
				gate = hphob[xs[ai]] & hphob[xs[aj]]
			else:
				gate = ((donor[xs[ai]] & accep[xs[aj]])
					| (donor[xs[aj]] & accep[xs[ai]]))
			return np.where(valid & gate, slopestep(d), 0.0)
		inter_raw, intra_raw = evalpairs(cache, 'both', fn)
		return termresult(inter_raw, intra_raw, weight)
	def ref15pairs(cache, same_res=False, cp='cp4',
			use_cp_rep=False):
		'''
		Build the typed-pair index arrays (i, j, distance, weight) for one same-res mode
		Arguments:
		----------
			cache: dict - ScoreMatch cache
			same_res: bool - True for intra-residue, False for inter-residue pairs
		Returns:
		--------
			tuple: (pi, pj, r, w) NumPy arrays for the matching pair subset
		'''
		mask = cache['pair_same_res']
		sel = mask if same_res else ~mask
		path = cache['pair_cp_path'] if use_cp_rep else cache['pair_path']
		if same_res:
			if cp == 'cp3':
				w = np.where(path <= 2, 0.0,
					np.where(path == 3, 0.2, 1.0))
			else:
				w = np.where(path <= 3, 0.0,
					np.where(path == 4, 0.2, 1.0))
		else:
			w = cache['pair_w']
			if use_cp_rep:
				if cp == 'cp3':
					w = np.where(path <= 2, 0.0,
						np.where(path == 3, 0.2, 1.0))
				else:
					w = np.where(path <= 3, 0.0,
						np.where(path == 4, 0.2, 1.0))
		sel = sel & (w > 0.0)
		return (cache['pairs_i'][sel], cache['pairs_j'][sel],
			cache['pair_d'][sel], w[sel])
	def ljpair(cache, pi, pj, r):
		'''
		Per-pair LJ (atr, rep) using the analytic etable-evaluation
		formula and the per-atom-type-pair LJ params:
		  dis2 < ljrep_linear_ramp_d2_cutoff:
		    ljE = lj_switch_slope * dis + lj_switch_intercept
		  ljrep_linear_ramp_d2_cutoff <= dis < ljatr_cubic_poly_xlo:
		    ljE = lj_r12_coeff / dis^12 + lj_r6_coeff / dis^6
		  ljatr_cubic_poly_xlo <= dis < ljatr_cubic_poly_xhi (4.5 -> 6.0):
		    ljE = eval(ljatr_cubic_poly, dis)
		  dis >= ljatr_cubic_poly_xhi (= 6.0): ljE = 0
		Split into atr/rep:
		  ljrep_from_negcrossing (REPLS/HREPS):
		    atrE = ljE if ljE < 0 else 0; repE = ljE if ljE >= 0 else 0
		  else if dis < lj_minimum:
		    atrE = lj_val_at_minimum; repE = ljE - lj_val_at_minimum
		  else:
		    atrE = ljE; repE = 0
		Plus per-pair ExtraQuadraticRepulsion adds to repE if dis < xhi.
		Final: atrE *= ljatr_final_weight.
		Arguments:
		----------
			cache: per-pose cache from _ref15atomcache
			pi, pj: atom-i and atom-j indices (np.int64)
			r: pair distances (np.float64)
		Returns:
		--------
			(atrE, repE) numpy arrays length len(pi)
		'''
		at_e_idx = cache.get('at_e_idx')
		n_pairs = len(pi)
		if at_e_idx is None or n_pairs == 0:
			z = np.zeros(n_pairs, dtype=np.float64)
			return z, z.copy()
		ai = at_e_idx[pi]; aj = at_e_idx[pj]
		valid = (ai >= 0) & (aj >= 0)
		a_lo = np.where(ai <= aj, ai, aj)
		a_hi = np.where(ai <= aj, aj, ai)
		a_lo_s = np.where(valid, a_lo, 0)
		a_hi_s = np.where(valid, a_hi, 0)
		ljrep_ramp_d2  = cache['et_ljrep_linear_ramp_d2'][a_lo_s, a_hi_s]
		lj_switch_int  = cache['et_lj_switch_intercept'][a_lo_s, a_hi_s]
		lj_switch_slo  = cache['et_lj_switch_slope'][a_lo_s, a_hi_s]
		lj_r12         = cache['et_lj_r12_coeff'][a_lo_s, a_hi_s]
		lj_r6          = cache['et_lj_r6_coeff'][a_lo_s, a_hi_s]
		ljatr_xlo      = cache['et_ljatr_cp_xlo'][a_lo_s, a_hi_s]
		ljatr_xhi      = cache['et_ljatr_cp_xhi'][a_lo_s, a_hi_s]
		ljatr_cp       = cache['et_ljatr_cubic_poly'][a_lo_s, a_hi_s]
		ljatr_fw       = cache['et_ljatr_final_weight'][a_lo_s, a_hi_s]
		lj_min         = cache['et_lj_minimum'][a_lo_s, a_hi_s]
		lj_val_at_min  = cache['et_lj_val_at_minimum'][a_lo_s, a_hi_s]
		rep_neg        = cache['et_ljrep_from_negcrossing'][a_lo_s, a_hi_s]
		xr_xlo   = cache['et_ljrep_xr_xlo'][a_lo_s, a_hi_s]
		xr_xhi   = cache['et_ljrep_xr_xhi'][a_lo_s, a_hi_s]
		xr_slope = cache['et_ljrep_xr_slope'][a_lo_s, a_hi_s]
		xr_extrap= cache['et_ljrep_xr_extrap_slope'][a_lo_s, a_hi_s]
		xr_ylo   = cache['et_ljrep_xr_ylo'][a_lo_s, a_hi_s]
		d = r
		d2 = d * d
		inv_d2 = 1.0 / np.maximum(d2, 1e-12)
		inv_d6 = inv_d2 ** 3
		inv_d12 = inv_d6 ** 2
		lj_linramp  = lj_switch_slo * d + lj_switch_int
		lj_generic  = lj_r12 * inv_d12 + lj_r6 * inv_d6
		c0=ljatr_cp[:,0]; c1=ljatr_cp[:,1]; c2=ljatr_cp[:,2]; c3=ljatr_cp[:,3]
		lj_atr_poly = ((c3 * d + c2) * d + c1) * d + c0
		ljE = np.where(d2 < ljrep_ramp_d2, lj_linramp,
			np.where(d < ljatr_xlo, lj_generic,
				np.where(d < ljatr_xhi, lj_atr_poly, 0.0)))
		atrE = np.where(rep_neg,
			np.where(ljE < 0, ljE, 0.0),
			np.where(d < lj_min, lj_val_at_min, ljE))
		repE = np.where(rep_neg,
			np.where(ljE >= 0, ljE, 0.0),
			np.where(d < lj_min, ljE - lj_val_at_min, 0.0))
		atrE = atrE * ljatr_fw
		atrE = np.where(valid, atrE, 0.0)
		repE = np.where(valid, repE, 0.0)
		return atrE, repE
	def ref15ljraw(cache, same_res):
		'''
		Compute raw LJ attractive + repulsive contributions over typed pairs
		Arguments:
		----------
			cache: dict - ScoreMatch cache
			same_res: bool - True for intra-residue pairs
		Returns:
		--------
			tuple: (atr_sum, rep_sum) - raw scalar sums before weighting
		'''
		pi, pj, r, w = ref15pairs(cache, same_res=same_res)
		if len(pi) == 0: return 0.0, 0.0
		atrE, repE = ljpair(cache, pi, pj, r)
		return float(np.sum(w * atrE)), float(np.sum(w * repE))
	def lkisopair(cache, pi, pj, r):
		'''
		Return per-direction analytic fa_sol/lk_iso values (one-sided
		desolvation energies) for atom pairs (pi[k], pj[k]) at distance
		r[k], using the ported per-atom-type-pair etable params:
		  - d < close_start: flat constant close_flat
		  - close_start <= d < close_end: cubic poly close
		  - close_end <= d < 4.5: standard analytic LK exponential
		  - 4.5 <= d < 6.0: cubic poly far Hermite fade
		  - d >= 6.0: 0
		Both directions evaluated. Returns (lki, lkj), each length len(pi).
		Pairs where either atom type is not in the etable table
		(e.g. virtuals or H) return 0 for that pair.
		Arguments:
		----------
			cache: per-pose cache from _ref15atomcache
			pi: atom-i indices (np.int64)
			pj: atom-j indices (np.int64)
			r:  pair distances (np.float64)
		Returns:
		--------
			(lki, lkj): tuple of np.float64 arrays length len(pi)
		'''
		at_e_idx = cache.get('at_e_idx')
		n_pairs = len(pi)
		if at_e_idx is None or n_pairs == 0:
			return (np.zeros(n_pairs, dtype=np.float64),
				np.zeros(n_pairs, dtype=np.float64))
		ai = at_e_idx[pi]; aj = at_e_idx[pj]
		valid = (ai >= 0) & (aj >= 0)
		ai_safe = np.where(valid, ai, 0)
		aj_safe = np.where(valid, aj, 0)
		def _eval(io_first, io_second):
			'''
			Evaluate analytic etable lk_iso for direction (self=first).
			Arguments:
			----------
				io_first:  self-atom etable idx (npair,)
				io_second: other-atom etable idx (npair,)
			Returns:
			--------
				np.float64 array (npair,) of one-sided lk_iso values
			'''
			cs = cache['et_close_start'][io_first, io_second]
			ce = cache['et_close_end'][io_first, io_second]
			cf = cache['et_close_flat'][io_first, io_second]
			cp = cache['et_close_poly'][io_first, io_second]
			fp = cache['et_far_poly'][io_first, io_second]
			lc = cache['et_lk_coeff'][io_first, io_second]
			la = cache['et_lambda_self'][io_first, io_second]
			Rs = cache['et_R_self'][io_first, io_second]
			fw = cache['et_final_w'][io_first, io_second]
			d = r
			d2 = d * d
			exp_arg = (d - Rs) / la
			gauss = np.exp(-exp_arg * exp_arg)
			e_mid = lc * gauss / np.maximum(d2, 1e-12)
			c0 = cp[:, 0]; c1 = cp[:, 1]; c2 = cp[:, 2]; c3 = cp[:, 3]
			e_close = c0 + c1 * d + c2 * d * d + c3 * d * d * d
			f0 = fp[:, 0]; f1 = fp[:, 1]; f2 = fp[:, 2]; f3 = fp[:, 3]
			e_far = f0 + f1 * d + f2 * d * d + f3 * d * d * d
			e = np.where(d < cs, cf, e_mid)
			e = np.where((d >= cs) & (d < ce), e_close, e)
			e = np.where((d >= 4.5) & (d < 6.0), e_far, e)
			e = np.where(d >= 6.0, 0.0, e)
			return e * fw
		lki = _eval(ai_safe, aj_safe)
		lkj = _eval(aj_safe, ai_safe)
		lki = np.where(valid, lki, 0.0)
		lkj = np.where(valid, lkj, 0.0)
		return lki, lkj
	def solpair(cache, pi, pj, r):
		'''
		Combined fa_sol per-pair value matching the
		analytic LK-evaluation algorithm (used by FaSol /
		FaIntraSolXover4). Uses the per-atom-pair COMBINED close/far
		cubic polynomials (`fasol_cubic_poly_close` / `_far`), which
		differ from poly1+poly2 because the combined poly is fit to a
		spline
		whose boundary derivatives come from the discrete-etable
		interpolator (not the analytical formula). The exponential
		region uses lk_coeff1*exp(-x1) + lk_coeff2*exp(-x2) per the
		analytic LK evaluation. Returns per-pair combined fa_sol
		energies (length len(pi)).
		Arguments:
		----------
			cache: per-pose cache
			pi: atom-i indices (np.int64)
			pj: atom-j indices (np.int64)
			r:  pair distances (np.float64)
		Returns:
		--------
			np.float64 array: combined fa_sol per pair
		'''
		at_e_idx = cache.get('at_e_idx')
		n_pairs = len(pi)
		if at_e_idx is None or n_pairs == 0:
			return np.zeros(n_pairs, dtype=np.float64)
		ai = at_e_idx[pi]; aj = at_e_idx[pj]
		valid = (ai >= 0) & (aj >= 0)
		a_lo = np.where(ai <= aj, ai, aj)
		a_hi = np.where(ai <= aj, aj, ai)
		a_lo_safe = np.where(valid, a_lo, 0)
		a_hi_safe = np.where(valid, a_hi, 0)
		cs = cache['et_close_start'][a_lo_safe, a_hi_safe]
		ce = cache['et_close_end'][a_lo_safe, a_hi_safe]
		cf = cache['et_close_flat_comb'][a_lo_safe, a_hi_safe]
		cp = cache['et_close_poly_comb'][a_lo_safe, a_hi_safe]
		fp = cache['et_far_poly_comb'][a_lo_safe, a_hi_safe]
		fw = cache['et_final_w'][a_lo_safe, a_hi_safe]
		lc1 = cache['et_lk_coeff'][a_lo_safe, a_hi_safe]
		lc2 = cache['et_lk_coeff'][a_hi_safe, a_lo_safe]
		R1  = cache['et_R_self'][a_lo_safe, a_hi_safe]
		R2  = cache['et_R_self'][a_hi_safe, a_lo_safe]
		la1 = cache['et_lambda_self'][a_lo_safe, a_hi_safe]
		la2 = cache['et_lambda_self'][a_hi_safe, a_lo_safe]
		d = r
		d2 = d * d
		x1 = ((d - R1) / la1) ** 2
		x2 = ((d - R2) / la2) ** 2
		e_mid = (lc1 * np.exp(-x1) + lc2 * np.exp(-x2)) / np.maximum(
			d2, 1e-12)
		c0 = cp[:, 0]; c1 = cp[:, 1]; c2 = cp[:, 2]; c3 = cp[:, 3]
		e_close = ((c3 * d + c2) * d + c1) * d + c0
		f0 = fp[:, 0]; f1 = fp[:, 1]; f2 = fp[:, 2]; f3 = fp[:, 3]
		e_far = ((f3 * d + f2) * d + f1) * d + f0
		e = np.where(d < cs, cf, e_mid)
		e = np.where((d >= cs) & (d < ce), e_close, e)
		e = np.where((d >= 4.5) & (d < 6.0), e_far, e)
		e = np.where(d >= 6.0, 0.0, e)
		e = e * fw
		e = np.where(valid, e, 0.0)
		return e
	def ref15solraw(cache, same_res):
		'''
		Lazaridis-Karplus solvation raw sum, using the
		per-atom-type-pair etable params (close-poly + far-fade).
		Both per-direction terms summed and weighted by the count-pair
		weight w.
		Arguments:
		----------
			cache: per-pose cache from _ref15atomcache
			same_res: if True compute intra-residue (xover4) subset,
				otherwise inter-residue pairs
		Returns:
		--------
			float: raw sum of (lki + lkj) * w over heavy-heavy pairs
		'''
		pi, pj, r, w = ref15pairs(cache, same_res=same_res)
		if len(pi) == 0: return 0.0
		e = solpair(cache, pi, pj, r)
		return float(np.sum(w * e))
	def ref15stubterm(weight_key):
		'''
		Generic stub: compute Sum over typed pairs of a user-supplied per-pair function
		Arguments:
		----------
			cache: dict - ScoreMatch cache
			same_res: bool - True for intra-residue pairs
			fn: callable - per-pair function (ai, aj, rij, w) -> contribution
		Returns:
		--------
			float: scalar sum across the selected pair subset
		'''
		w = float(params.get(weight_key, {}).get('weight', 0.0))
		return {'inter_raw': 0.0, 'intra_raw': 0.0,
			'inter_weighted': 0.0, 'intra_weighted': 0.0,
			'raw': 0.0, '_pending_full_impl': True, '_weight': w}
	def fadun_rotwell_grid(aa, n_chi, residues_db):
		'''
		Build per-(AA, rotwell_index) 36x36 grids of -log(P) and chi
		means/sigmas for the natural cubic spline interpolation. Cached
		on first use.
		Arguments:
		----------
			aa: amino acid 3-letter code
			n_chi: number of chi angles for this AA
			residues_db: rotamer library residues dict
		Returns:
		--------
			dict: rot_idx -> dict with 'neglogP', 'mu', 'sd', plus
				cached 'ypp_psi' for each quantity (built lazily).
		'''
		key = aa
		if key in _FADUN_GRID_CACHE:
			return _FADUN_GRID_CACHE[key]
		entry = residues_db.get(aa)
		if entry is None:
			_FADUN_GRID_CACHE[key] = {}
			return {}
		rot = entry['rotamers']
		offs = rot['bin_offsets']
		tbl = rot['table']
		MAXE = -math.log(1e-6)
		all_rotwells = set()
		for r2 in tbl:
			all_rotwells.add(r2[0])
		grids = {}
		for rw in all_rotwells:
			grids[rw] = {
				'neglogP': np.full((36, 36), MAXE),
				'mu': [np.zeros((36, 36)) for _ in range(n_chi)],
				'sd': [np.full((36, 36), 1.0)
					for _ in range(n_chi)],
				'has_data': np.zeros((36, 36), dtype=bool)}
		for i_phi in range(36):
			for i_psi in range(36):
				bidx = i_phi * 36 + i_psi
				if bidx + 1 >= len(offs): continue
				rows = tbl[offs[bidx]:offs[bidx+1]]
				for r2 in rows:
					rw = r2[0]
					if rw not in grids: continue
					Pk = r2[1]
					if Pk <= 0.0: continue
					Pk_clip = max(Pk, 1e-6)
					g = grids[rw]
					if g['has_data'][i_phi, i_psi]:
						old_P = math.exp(-g['neglogP'][i_phi, i_psi])
						new_P = old_P + Pk_clip
						g['neglogP'][i_phi, i_psi] = min(MAXE,
							-math.log(new_P))
					else:
						g['neglogP'][i_phi, i_psi] = min(MAXE,
							-math.log(Pk_clip))
						for ci in range(n_chi):
							g['mu'][ci][i_phi, i_psi] = r2[2 + ci]
							g['sd'][ci][i_phi, i_psi] = \
								max(r2[2 + n_chi + ci], 0.5)
						g['has_data'][i_phi, i_psi] = True
		for rw, g in grids.items():
			g['neglogP_ypp_psi'] = np.array([
				periodic_cubic_spline(g['neglogP'][i])
				for i in range(36)])
			g['mu_ypp_psi'] = [np.array([
				periodic_cubic_spline(g['mu'][ci][i])
				for i in range(36)]) for ci in range(n_chi)]
			g['sd_ypp_psi'] = [np.array([
				periodic_cubic_spline(g['sd'][ci][i])
				for i in range(36)]) for ci in range(n_chi)]
		_FADUN_GRID_CACHE[key] = grids
		return grids
	def fadun_entropy_grid(aa, residues_db):
		'''
		Build the FaDun entropy-correction grid for one amino acid by Boltzmann reweighting
		Arguments:
		----------
			aa: str - 3-letter amino acid code
		Returns:
		--------
			np.ndarray: entropy grid indexed by (phi_bin, psi_bin)
		'''
		if aa in _FADUN_ENT_CACHE:
			return _FADUN_ENT_CACHE[aa]
		entry = residues_db.get(aa)
		if entry is None:
			_FADUN_ENT_CACHE[aa] = (np.zeros((36, 36)),
				np.zeros((36, 36)))
			return _FADUN_ENT_CACHE[aa]
		rot = entry['rotamers']
		offs = rot['bin_offsets']
		tbl = rot['table']
		ent = np.zeros((36, 36))
		for i_phi in range(36):
			for i_psi in range(36):
				bidx = i_phi * 36 + i_psi
				if bidx + 1 >= len(offs): continue
				rows = tbl[offs[bidx]:offs[bidx+1]]
				groups = {}
				for r2 in rows:
					if r2[1] > 0.0:
						groups.setdefault(r2[0], 0.0)
						groups[r2[0]] += r2[1]
				e = 0.0
				for Pg in groups.values():
					if Pg > 0.0: e += Pg * math.log(Pg)
				ent[i_phi, i_psi] = e
		ypp_psi = np.array([
			periodic_cubic_spline(ent[i]) for i in range(36)])
		_FADUN_ENT_CACHE[aa] = (ent, ypp_psi)
		return _FADUN_ENT_CACHE[aa]
	def fadun_spline_eval(grid_2d, ypp_psi_grid, fp, fs):
		'''
		Evaluate periodic 2D natural cubic spline at (fp, fs) using
		precomputed ypp_psi (2nd deriv along psi).
		Arguments:
		----------
			grid_2d: 36x36 numpy array of values
			ypp_psi_grid: 36x36 numpy array of 2nd derivs along psi
			fp, fs: fractional phi, psi indices in [0, 36)
		Returns:
		--------
			float: interpolated value
		'''
		n = 36
		i_psi = int(math.floor(fs)) % n
		j_psi = (i_psi + 1) % n
		frac_s = fs - math.floor(fs)
		a = 1.0 - frac_s; b = frac_s
		col_f = (a * grid_2d[:, i_psi] + b * grid_2d[:, j_psi]
			+ ((a**3 - a) * ypp_psi_grid[:, i_psi]
			   + (b**3 - b) * ypp_psi_grid[:, j_psi]) / 6.0)
		ypp_phi = periodic_cubic_spline(col_f)
		return spline_eval_1d(col_f, ypp_phi, fp, n)
	def fadun_nrchi_data(tri):
		'''
		Load and cache the per-AA non-rotameric chi_last density tables
		(Shapovalov backbone-dependent source) for the 8 semi-rotameric AAs. Pre-computes 2nd-derivative grids
		for the periodic-bicubic phi/psi interpolation of -log(P_rot)
		and of each chi_last density column.
		Arguments:
		----------
			tri: 3-letter AA code (ASN, ASP, GLU, GLN, HIS, PHE, TRP, TYR)
		Returns:
		--------
			dict: {rotwell_tuple: {neg_log_P_rot_grid, chi_means,
				chi_sigmas, dens_grid, neg_log_dens, neg_log_dens_ypp,
				chi_last_low, chi_last_step, chi_last_n}}, or {} if
			missing
		'''
		if tri in _FADUN_NRCHI_CACHE:
			return _FADUN_NRCHI_CACHE[tri]
		from .pose import DBLoad
		nrchi_db = DBLoad().get('FaDunNrchiDensities', {}) or {}
		aa_entry = nrchi_db.get(tri)
		if aa_entry is None:
			_FADUN_NRCHI_CACHE[tri] = {}
			return {}
		n_disc_chi = int(aa_entry['n_disc_chi'])
		chi_last_n = int(aa_entry['chi_last_n'])
		chi_last_low = float(aa_entry['chi_last_low'])
		chi_last_step = float(aa_entry['chi_last_step'])
		out = {
			'chi_last_low':  chi_last_low,
			'chi_last_step': chi_last_step,
			'chi_last_n':    chi_last_n,
			'n_disc_chi':    n_disc_chi,
			'per_rot':       {}}
		MAXE = 13.815510557964274  # -log(1e-6)
		for rk_str, rot_dat in aa_entry['per_rot'].items():
			rot_tuple = tuple(int(x) for x in rk_str.split(','))
			P_rot = np.asarray(rot_dat['P_rot'],
				dtype=np.float64).reshape(36, 36)
			neglogP_rot = np.asarray(rot_dat['neglogP_rot'],
				dtype=np.float64).reshape(36, 36)
			cm = np.asarray(rot_dat['chi_means'],
				dtype=np.float64).reshape(n_disc_chi, 36, 36)
			cs = np.asarray(rot_dat['chi_sigmas'],
				dtype=np.float64).reshape(n_disc_chi, 36, 36)
			dens = np.asarray(rot_dat['densities'],
				dtype=np.float64).reshape(36, 36, chi_last_n)
			dens_safe = np.maximum(dens, 1e-6)
			neglogD = -np.log(dens_safe)
			neglogD = np.minimum(neglogD, MAXE)
			ypp_rot = fadun_ypp_psi_grid(neglogP_rot)
			ypp_dens = np.zeros_like(neglogD)
			for k in range(chi_last_n):
				ypp_dens[:, :, k] = fadun_ypp_psi_grid(
					neglogD[:, :, k])
			ypp_Prot = fadun_ypp_psi_grid(P_rot)
			out['per_rot'][rot_tuple] = {
				'P_rot':         P_rot,
				'neglogP_rot':   neglogP_rot,
				'chi_means':     cm,
				'chi_sigmas':    cs,
				'densities':     dens,
				'neglogD':       neglogD,
				'ypp_rot':       ypp_rot,
				'ypp_Prot':      ypp_Prot,
				'ypp_dens':      ypp_dens}
		_FADUN_NRCHI_CACHE[tri] = out
		return out
	def fadun_ypp_psi_grid(grid_2d):
		'''
		Precompute psi-direction 2nd derivatives over a 36x36 periodic
		grid for natural cubic spline use in _fadun_spline_eval.
		Arguments:
		----------
			grid_2d: 36x36 numpy array
		Returns:
		--------
			36x36 numpy array of 2nd derivatives along psi axis
		'''
		out = np.zeros_like(grid_2d)
		for i in range(grid_2d.shape[0]):
			out[i, :] = periodic_cubic_spline(grid_2d[i, :])
		return out
	def fadun_nrchi_eval(tri, rot_tuple, phi, psi, chi_last):
		'''
		Evaluate the non-rotameric chi_last density (-log) for a
		semi-rotameric residue at (phi, psi, chi_last) under a specific
		rotwell. Uses periodic bicubic spline over (phi, psi) and linear
		interp over chi_last. Also returns -log(P_rot(phi,psi)) and the
		rotameric chi means/sigmas at this (phi, psi).
		Arguments:
		----------
			tri: 3-letter AA code
			rot_tuple: tuple of rotameric-chi bin indices
			phi, psi: in degrees
			chi_last: in degrees (assumed already folded into AA's
				canonical range)
		Returns:
		--------
			(neg_log_rot, chi_means_list, chi_sigmas_list, neg_log_dens)
			tuple; returns (None, None, None, None) if the rotwell or
			AA is not in the nrchi table.
		'''
		data = fadun_nrchi_data(tri)
		if not data: return (None, None, None, None)
		rdat = data['per_rot'].get(rot_tuple)
		if rdat is None: return (None, None, None, None)
		n_disc_chi = int(data['n_disc_chi'])
		chi_last_low = float(data['chi_last_low'])
		chi_last_step = float(data['chi_last_step'])
		chi_last_n = int(data['chi_last_n'])
		fp = (phi + 180.0) / 10.0
		fs = (psi + 180.0) / 10.0
		neg_log_rot = fadun_spline_eval(
			rdat['neglogP_rot'], rdat['ypp_rot'], fp, fs)
		ip0 = int(math.floor(fp)); js0 = int(math.floor(fs))
		tp = fp - ip0; ts = fs - js0
		chi_means_v = []
		chi_sigmas_v = []
		for k in range(n_disc_chi):
			mu_grid = rdat['chi_means'][k]
			sd_grid = rdat['chi_sigmas'][k]
			mc = mu_grid[ip0 % 36, js0 % 36]
			def _unwrap(v, ref):
				'''
				Unwrap a periodic 1D array so consecutive samples are within +/-180 degrees
				Arguments:
				----------
					arr: np.ndarray - 1D array of angle samples in degrees
				Returns:
				--------
					np.ndarray: unwrapped copy of arr
				'''
				return ref + ((v - ref + 180.0) % 360.0 - 180.0)
			a = _unwrap(mu_grid[ip0 % 36, js0 % 36], mc)
			b = _unwrap(mu_grid[(ip0 + 1) % 36, js0 % 36], mc)
			c = _unwrap(mu_grid[ip0 % 36, (js0 + 1) % 36], mc)
			d = _unwrap(mu_grid[(ip0 + 1) % 36, (js0 + 1) % 36], mc)
			mu_v = ((1 - tp) * (1 - ts) * a + tp * (1 - ts) * b
				+ (1 - tp) * ts * c + tp * ts * d)
			sd_v = ((1 - tp) * (1 - ts) * sd_grid[ip0 % 36, js0 % 36]
				+ tp * (1 - ts) * sd_grid[(ip0 + 1) % 36, js0 % 36]
				+ (1 - tp) * ts * sd_grid[ip0 % 36, (js0 + 1) % 36]
				+ tp * ts * sd_grid[(ip0 + 1) % 36, (js0 + 1) % 36])
			chi_means_v.append(mu_v)
			chi_sigmas_v.append(max(sd_v, 0.5))
		fc = (chi_last - chi_last_low) / chi_last_step
		fc_mod = fc - chi_last_n * math.floor(fc / chi_last_n)
		v_arr = np.empty(chi_last_n)
		for c in range(chi_last_n):
			v_arr[c] = fadun_spline_eval(
				rdat['neglogD'][:, :, c],
				rdat['ypp_dens'][:, :, c], fp, fs)
		ypp_c = periodic_cubic_spline(v_arr)
		neg_log_dens = spline_eval_1d(
			v_arr, ypp_c, fc_mod, chi_last_n)
		return (neg_log_rot, chi_means_v, chi_sigmas_v, neg_log_dens)
	def periodic_cubic_spline(y):
		'''
		Periodic natural cubic spline 2nd derivatives on uniform grid
		(h=1). Uses FFT-based circulant tridiagonal solve.
		Arguments:
		----------
			y: 1-D numpy array (length n) of values on uniform grid
		Returns:
		--------
			numpy array: 2nd derivatives at each grid point
		'''
		y = np.asarray(y, dtype=float)
		n = len(y)
		b = 6.0 * (np.roll(y, -1) - 2.0 * y + np.roll(y, 1))
		k = np.arange(n)
		A_diag = 4.0 + 2.0 * np.cos(2.0 * np.pi * k / n)
		b_fft = np.fft.fft(b)
		return np.real(np.fft.ifft(b_fft / A_diag))
	def spline_eval_1d(y, ypp, t, n):
		'''
		Evaluate a 1D cubic spline at x using precomputed second-derivative coefficients
		Arguments:
		----------
			xs: np.ndarray - sample x-values (strictly increasing)
			ys: np.ndarray - sample y-values
			y2: np.ndarray - precomputed second derivatives
			x: float - query point
		Returns:
		--------
			float: spline value at x
		'''
		i = int(math.floor(t)) % n
		j = (i + 1) % n
		frac = t - math.floor(t)
		a = 1.0 - frac
		return (a * y[i] + frac * y[j]
			+ ((a*a*a - a) * ypp[i] + (frac*frac*frac - frac) * ypp[j])
			/ 6.0)
	def rama_spline_eval(table, fp, fs):
		'''
		Periodic 2D natural cubic spline on 36x36 grid. Caches the
		ypp_psi (2nd deriv along psi for each phi row). Per-query
		computation of ypp_phi for the column-of-f values.
		Arguments:
		----------
			table: 36x36 list/array of values
			fp: fractional phi index (0..36)
			fs: fractional psi index (0..36)
		Returns:
		--------
			float: spline-interpolated value
		'''
		key = id(table)
		cached = _RAMA_SPLINE_CACHE.get(key)
		if cached is None:
			arr = np.asarray(table, dtype=float)
			ypp_psi = np.zeros_like(arr)
			for i in range(arr.shape[0]):
				ypp_psi[i] = periodic_cubic_spline(arr[i])
			cached = (arr, ypp_psi)
			_RAMA_SPLINE_CACHE[key] = cached
		arr, ypp_psi = cached
		n = arr.shape[0]
		i_psi = int(math.floor(fs)) % n
		j_psi = (i_psi + 1) % n
		frac_s = fs - math.floor(fs)
		a = 1.0 - frac_s; b = frac_s
		col_f = (a * arr[:, i_psi] + b * arr[:, j_psi]
			+ ((a**3 - a) * ypp_psi[:, i_psi]
			   + (b**3 - b) * ypp_psi[:, j_psi]) / 6.0)
		ypp_phi = periodic_cubic_spline(col_f)
		return spline_eval_1d(col_f, ypp_phi, fp, n)
	def hbond_chemtype_maps():
		'''
		Build forward and reverse chemtype lookup maps for HBond polynomial selection
		Arguments:
		----------
			params: dict - the Score-Parameters block carrying HBond_data
		Returns:
		--------
			tuple: (cidx_by_donor, cidx_by_acceptor, poly_table)
		'''
		donor_map = {}; acceptor_map = {}; base_map = {}
		for tri in ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS',
			'HIS_D','ILE','LEU','LYS','MET','PHE','PRO','SER','THR',
			'TRP','TYR','VAL']:
			if tri != 'PRO': donor_map[(tri, 'N')] = 'hbdon_PBA'
		donor_map[('ASN', 'ND2')] = 'hbdon_CXA'
		donor_map[('GLN', 'NE2')] = 'hbdon_CXA'
		donor_map[('HIS', 'NE2')] = 'hbdon_IME'
		donor_map[('HIS_D', 'ND1')] = 'hbdon_IMD'
		donor_map[('TRP', 'NE1')] = 'hbdon_IND'
		donor_map[('LYS', 'NZ')] = 'hbdon_AMO'
		donor_map[('ARG', 'NE')] = 'hbdon_GDE'
		donor_map[('ARG', 'NH1')] = 'hbdon_GDH'
		donor_map[('ARG', 'NH2')] = 'hbdon_GDH'
		donor_map[('TYR', 'OH')] = 'hbdon_AHX'
		donor_map[('SER', 'OG')] = 'hbdon_HXL'
		donor_map[('THR', 'OG1')] = 'hbdon_HXL'
		for tri in ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS',
			'HIS_D','ILE','LEU','LYS','MET','PHE','PRO','SER','THR',
			'TRP','TYR','VAL']:
			acceptor_map[(tri, 'O')] = 'hbacc_PBA'
			base_map[(tri, 'O')] = 'C'
			acceptor_map[(tri, 'OXT')] = 'hbacc_PBA'
			base_map[(tri, 'OXT')] = 'C'
		acceptor_map[('ASN', 'OD1')] = 'hbacc_CXA'; base_map[('ASN','OD1')] = 'CG'
		acceptor_map[('GLN', 'OE1')] = 'hbacc_CXA'; base_map[('GLN','OE1')] = 'CD'
		acceptor_map[('ASP', 'OD1')] = 'hbacc_CXL'; base_map[('ASP','OD1')] = 'CG'
		acceptor_map[('ASP', 'OD2')] = 'hbacc_CXL'; base_map[('ASP','OD2')] = 'CG'
		acceptor_map[('GLU', 'OE1')] = 'hbacc_CXL'; base_map[('GLU','OE1')] = 'CD'
		acceptor_map[('GLU', 'OE2')] = 'hbacc_CXL'; base_map[('GLU','OE2')] = 'CD'
		acceptor_map[('HIS', 'ND1')] = 'hbacc_IME'; base_map[('HIS','ND1')] = 'CG'
		acceptor_map[('HIS_D', 'NE2')] = 'hbacc_IMD'; base_map[('HIS_D','NE2')] = 'CD2'
		acceptor_map[('TYR', 'OH')] = 'hbacc_AHX'; base_map[('TYR','OH')] = 'CZ'
		acceptor_map[('SER', 'OG')] = 'hbacc_HXL'; base_map[('SER','OG')] = 'CB'
		acceptor_map[('THR', 'OG1')] = 'hbacc_HXL'; base_map[('THR','OG1')] = 'CB'
		return donor_map, acceptor_map, base_map
	def hbond_eval_lookup(hb):
		'''
		Look up the polynomial-coefficient row for a (donor_chem, acceptor_chem) pair
		Arguments:
		----------
			don_chem: str - donor chemical type code
			acc_chem: str - acceptor chemical type code
		Returns:
		--------
			np.ndarray or None: 4xN polynomial coefficients or None if no match
		'''
		key = {}
		for e in hb['eval_table']:
			key[(e['don'], e['acc'], e['sep'])] = e
		return key
	def hbond_poly_eval(poly, x):
		'''
		Horner evaluation of a polynomial with clamping to (xmin, xmax)
		Arguments:
		----------
			poly: dict - {'xmin', 'xmax', 'min_val', 'max_val', 'coeffs'} polynomial entry
			x:    float - query value
		Returns:
		--------
			float: polynomial value at x, clamped at the table endpoints
		'''
		if poly is None: return 0.0
		if x <= poly['xmin']: return poly['min_val']
		if x >= poly['xmax']: return poly['max_val']
		c = poly['coeffs']
		if not c: return 0.0
		v = c[0]
		for i in range(1, len(c)):
			v = v * x + c[i]
		return v
	def hbond_fade(fade, x):
		'''
		Sigmoid fade-out factor for HBond energy across the distance shell
		Arguments:
		----------
			r: float - donor-acceptor distance
		Returns:
		--------
			float: fade weight in [0, 1]
		'''
		if fade is None: return 1.0
		kind = fade.get('kind', 'smoothed')
		mn1 = fade['min1']; mn2 = fade['min2']
		mx1 = fade['max1']; mx2 = fade['max2']
		if x <= mn1 or x >= mx2: return 0.0
		if mn2 <= x <= mx1: return 1.0
		if x < mn2:
			t = (x - mn1) / max(mn2 - mn1, 1e-12)
			return t * t * (3.0 - 2.0 * t)
		t = (mx2 - x) / max(mx2 - mx1, 1e-12)
		return t * t * (3.0 - 2.0 * t)
	def ref15hbond(pose, cache, per_hb=None):
		'''
		Compute the hydrogen-bond energy with the four categories partitioned
		Arguments:
		----------
			cache: dict - ScoreMatch cache
		Returns:
		--------
			dict: per-category raw and weighted contributions ('sr_bb', 'lr_bb', 'bb_sc', 'sc')
		'''
		hb = params.get('HBond_data') or {}
		if not hb: return {'SR_BB': 0.0, 'LR_BB': 0.0,
			'BB_SC': 0.0, 'SC': 0.0}
		donor_map, acceptor_map, base_map = hbond_chemtype_maps()
		eval_key = hbond_eval_lookup(hb)
		polys = hb['polynomials']; fades = hb['fade_intervals']
		don_str_tab = hb['donor_strengths']
		acc_str_tab = hb['acceptor_strengths']
		atoms = pose.data['Atoms']
		coords = np.asarray(pose.data['Coordinates'])
		bonds = cache.get('adj') or pose.data['Bonds']
		aas = pose.data.get('Amino Acids') or {}
		atom_to_res = {}
		res_atom = {}
		for ri, info in aas.items():
			tri = info[5] if len(info) >= 6 else None
			if tri == 'HIS': pass
			for ai in info[2] + info[3]:
				ai = int(ai)
				atom_to_res[ai] = (int(ri), tri)
				res_atom.setdefault((int(ri), atoms[ai][0]), ai)
		nb_count = {}
		nb_xyz = {}
		for ri, info in aas.items():
			tri = info[5] if len(info) >= 6 else None
			nb_atom = 'CA' if tri == 'GLY' else 'CB'
			ai = res_atom.get((int(ri), nb_atom))
			if ai is None:
				ai = res_atom.get((int(ri), 'CA'))
			if ai is not None:
				nb_xyz[int(ri)] = coords[ai]
		ri_list = list(nb_xyz.keys())
		nb_arr = np.stack([nb_xyz[r] for r in ri_list], axis=0)
		dd = np.linalg.norm(nb_arr[:, None, :] - nb_arr[None, :, :],
			axis=2)
		within = (dd < 10.0)
		counts = within.sum(axis=1)
		for k, r in enumerate(ri_list):
			nb_count[r] = int(counts[k])
		def burial_w(n):
			'''
			Heavy-neighbour burial weight used by LkBallWtd (count of neighbours within 5 A)
			Arguments:
			----------
				gi: int - atom index
			Returns:
			--------
				float: burial weight
			'''
			if n < 7: return 0.1
			if n > 24: return 0.5
			return (n - 2.75) * (0.5 / 21.25)
		donors = []
		for ai, info in atoms.items():
			if info[1] not in ('N', 'O'): continue
			tri_pair = atom_to_res.get(int(ai))
			if tri_pair is None: continue
			ri, tri = tri_pair
			key = (tri, info[0])
			if key not in donor_map: continue
			for j in bonds.get(int(ai), []):
				jinfo = atoms.get(int(j))
				if jinfo is None: continue
				if jinfo[1] != 'H': continue
				donors.append({'D': int(ai), 'H': int(j),
					'ri': ri, 'tri': tri, 'chem': donor_map[key]})
		acceptors = []
		for ai, info in atoms.items():
			if info[1] not in ('N', 'O'): continue
			tri_pair = atom_to_res.get(int(ai))
			if tri_pair is None: continue
			ri, tri = tri_pair
			key = (tri, info[0])
			if key not in acceptor_map: continue
			b_name = base_map.get(key)
			b_ai = res_atom.get((ri, b_name))
			if b_ai is None: continue
			chem = acceptor_map[key]
			b2_ai = None
			if chem in ('hbacc_HXL', 'hbacc_AHX'):
				for k in bonds.get(int(ai), []):
					k = int(k)
					if atoms.get(k, [None,'X'])[1] == 'H':
						b2_ai = k; break
			else:
				same_res_nbrs = []
				other_nbrs = []
				for k in bonds.get(b_ai, []):
					k = int(k)
					if k == int(ai): continue
					if atoms.get(k, [None,'H'])[1] == 'H': continue
					tri_pair_k = atom_to_res.get(k)
					if tri_pair_k is not None and tri_pair_k[0] == ri:
						same_res_nbrs.append(k)
					else:
						other_nbrs.append(k)
				if same_res_nbrs:
					b2_ai = sorted(same_res_nbrs)[0]
				elif other_nbrs:
					b2_ai = sorted(other_nbrs)[0]
			acceptors.append({'A': int(ai), 'B': b_ai, 'B2': b2_ai,
				'ri': ri, 'tri': tri, 'chem': chem})
		cat_totals = {'SR_BB': 0.0, 'LR_BB': 0.0,
			'BB_SC': 0.0, 'SC': 0.0}
		if not donors or not acceptors: return cat_totals
		all_hbonds = []
		H_idx = np.array([d['H'] for d in donors], dtype=np.int64)
		A_idx = np.array([a['A'] for a in acceptors], dtype=np.int64)
		Hc = coords[H_idx]; Ac = coords[A_idx]
		dHA = np.linalg.norm(
			Hc[:, None, :] - Ac[None, :, :], axis=2)
		within = np.where(dHA < 3.2)
		for ix, iy in zip(*within):
			d = donors[ix]; a = acceptors[iy]
			if d['D'] == a['A']: continue
			if d['ri'] == a['ri']: continue
			diff = a['ri'] - d['ri']
			if abs(diff) > 4 or diff == 0:
				sep = 'seq_sep_other'
			elif diff == -4: sep = 'seq_sep_M4'
			elif diff == -3: sep = 'seq_sep_M3'
			elif diff == -2: sep = 'seq_sep_M2'
			elif abs(diff) == 1: sep = 'seq_sep_PM1'
			elif diff == 2: sep = 'seq_sep_P2'
			elif diff == 3: sep = 'seq_sep_P3'
			elif diff == 4: sep = 'seq_sep_P4'
			else: sep = 'seq_sep_other'
			entry = eval_key.get((d['chem'], a['chem'], sep))
			if entry is None:
				entry = eval_key.get((d['chem'], a['chem'], 'seq_sep_other'))
			if entry is None: continue
			D_xyz = coords[d['D']]; H_xyz = coords[d['H']]
			A_xyz = coords[a['A']]; B_xyz = coords[a['B']]
			AH = float(dHA[ix, iy])
			vDH = D_xyz - H_xyz; vAH = A_xyz - H_xyz
			cosAHD = float(np.dot(vDH, vAH) /
				max(np.linalg.norm(vDH) * np.linalg.norm(vAH), 1e-12))
			vBA = B_xyz - A_xyz; vHA = -vAH
			cosBAH = float(np.dot(vBA, vHA) /
				max(np.linalg.norm(vBA) * np.linalg.norm(vHA), 1e-12))
			poly_d = polys.get(entry['poly_AHdist'])
			poly_bah_short = polys.get(entry['poly_cosBAH_short'])
			poly_bah_long = polys.get(entry['poly_cosBAH_long'])
			poly_ahd_short = polys.get(entry['poly_cosAHD_short'])
			poly_ahd_long = polys.get(entry['poly_cosAHD_long'])
			xH = -cosBAH
			xD = -cosAHD
			AHD_rad = math.acos(max(-1.0, min(1.0, cosAHD)))
			def ahd_arg(poly):
				'''
				Acceptor-hydrogen-donor angle argument used by HBond polynomials
				Arguments:
				----------
					D: np.ndarray - donor position
					H: np.ndarray - hydrogen position
					A: np.ndarray - acceptor position
				Returns:
				--------
					float: cosine of the A-H-D angle
				'''
				if poly is None: return xD
				return AHD_rad if poly.get('xmin', -1) > 0.5 else xD
			Pr = hbond_poly_eval(poly_d, AH)
			PSxH = hbond_poly_eval(poly_bah_short, xH)
			PLxH = hbond_poly_eval(poly_bah_long, xH)
			PSxD = hbond_poly_eval(
				poly_ahd_short, ahd_arg(poly_ahd_short))
			PLxD = hbond_poly_eval(
				poly_ahd_long, ahd_arg(poly_ahd_long))
			FSr = hbond_fade(
				fades.get(entry['fade_AHdist']), AH)
			FLr = 0.0
			fbah_s_name = entry['fade_cosBAH_short']
			fbah_l_name = entry['fade_cosBAH_long']
			fbah_s = fades.get(fbah_s_name)
			fbah_l = fades.get(fbah_l_name)
			FxH = hbond_fade(fbah_l, xH)
			fahd_s = fades.get(entry['fade_cosAHD_short'])
			FxD = hbond_fade(fahd_s, xD)
			e = (Pr * FxD * FxH
				+ FSr * (PSxD * FxH + FxD * PSxH)
				+ FLr * (PLxD * FxH + FxD * PLxH))
			acc_hyb = params.get('HBond_data', {}) \
				.get('acc_hybridization', {}).get(a['chem'])
			s = (don_str_tab.get(d['chem'], 1.0)
				* acc_str_tab.get(a['chem'], 1.0))
			e *= s
			if acc_hyb == 'SP2_HYBRID' and a.get('B2') is not None:
				B2_xyz = coords[a['B2']]
				b1 = B_xyz - B2_xyz
				b2 = A_xyz - B_xyz
				b3 = H_xyz - A_xyz
				n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
				n1n = np.linalg.norm(n1); n2n = np.linalg.norm(n2)
				if n1n > 1e-9 and n2n > 1e-9:
					m1 = n1 / n1n; m2 = n2 / n2n
					cos_chi = float(np.dot(m1, m2))
					sin_chi_sign = float(np.dot(np.cross(m1, m2),
						b2 / max(np.linalg.norm(b2), 1e-12)))
					chi = math.atan2(sin_chi_sign, cos_chi)
					d_p = 0.75; m_p = 1.6; l_p = 0.357
					PI = math.pi
					PI_minus_BAH = math.acos(
						max(-1.0, min(1.0, xH)))
					BAH = PI - PI_minus_BAH
					H_chi = (math.cos(2 * chi) + 1) * 0.5
					if BAH >= PI * 2.0 / 3.0:
						F_p = d_p * 0.5 * math.cos(3 * PI_minus_BAH) \
							+ d_p * 0.5 - 0.5
						G_p = d_p - 0.5
					elif BAH >= PI * (2.0 / 3.0 - l_p):
						outer = math.cos(
							PI - (PI * 2.0 / 3.0 - BAH) / l_p)
						F_p = m_p * 0.5 * outer + m_p * 0.5 - 0.5
						G_p = (m_p - d_p) * 0.5 * outer \
							+ (m_p - d_p) * 0.5 + d_p - 0.5
					else:
						F_p = m_p - 0.5; G_p = m_p - 0.5
					e += s * (H_chi * F_p + (1 - H_chi) * G_p)
			elif acc_hyb == 'SP3_HYBRID' and a['chem'] in (
					'hbacc_HXL', 'hbacc_AHX') and a.get('B2') is not None:
				B2_xyz = coords[a['B2']]
				b1 = H_xyz - A_xyz
				b2 = A_xyz - B_xyz
				b3 = B_xyz - B2_xyz
				n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
				n1n = np.linalg.norm(n1); n2n = np.linalg.norm(n2)
				if n1n > 1e-9 and n2n > 1e-9:
					m1 = n1 / n1n; m2 = n2 / n2n
					cos_chi = float(np.dot(m1, m2))
					sin_chi_sign = float(np.dot(np.cross(m1, m2),
						b2 / max(np.linalg.norm(b2), 1e-12)))
					chi = math.atan2(sin_chi_sign, cos_chi)
					PI = math.pi
					max_penalty = 0.125
					PI_minus_BAH = math.acos(
						max(-1.0, min(1.0, xH)))
					BAH = PI - PI_minus_BAH
					chi_scale = 0.0
					if ((chi > PI/3 and chi < PI/2) or
							(chi < -PI/3 and chi > -PI/2) or
							(chi > 3*PI/2 and chi < 5*PI/3)):
						chi_scale = (-math.cos(6 * chi) + 1) / 2
					elif ((chi > PI/2 and chi < 3*PI/2) or
							(chi < -PI/2 and chi > -3*PI/2)):
						chi_scale = 1.0
					BAH_bonus = -1.0
					if BAH > 2 * PI / 3:
						BAH_bonus = -math.cos(3 * BAH) / 2 - 0.5
					sp3_acc_penalty = (s * max_penalty
						* (1 + BAH_bonus * chi_scale))
					e += sp3_acc_penalty
			input_e = e
			if input_e > 0.1:
				continue
			if input_e > -0.1:
				e = -0.025 + 0.5 * input_e \
					- 2.5 * input_e * input_e
			w = entry['weight']
			don_is_bb = atoms[d['D']][0] == 'N'
			acc_is_bb = atoms[a['A']][0] in ('O', 'OXT', 'OT1', 'OT2')
			all_hbonds.append({
				'ri_d': d['ri'], 'd_atom': atoms[d['H']][0],
				'ri_a': a['ri'], 'a_atom': atoms[a['A']][0],
				'e': e, 'w': w,
				'don_is_bb': don_is_bb,
				'acc_is_bb': acc_is_bb,
				'AH': AH, 'cosBAH': cosBAH, 'cosAHD': cosAHD})
		don_bbg = set()
		acc_bbg = set()
		for h in all_hbonds:
			if h['don_is_bb'] and h['acc_is_bb']:
				don_bbg.add(h['ri_d'])
				acc_bbg.add(h['ri_a'])
		for h in all_hbonds:
			if h['don_is_bb'] and not h['acc_is_bb']:
				if h['ri_d'] in don_bbg: continue
			elif not h['don_is_bb'] and h['acc_is_bb']:
				if h['ri_a'] in acc_bbg: continue
			e = h['e']; w = h['w']
			cat = None
			if w == 'hbw_SR_BB':
				cat_totals['SR_BB'] += e; cat = 'SR_BB'
			elif w == 'hbw_LR_BB':
				cat_totals['LR_BB'] += e; cat = 'LR_BB'
			elif w in ('hbw_SR_BB_SC', 'hbw_LR_BB_SC'):
				cat_totals['BB_SC'] += e; cat = 'BB_SC'
			elif w == 'hbw_SC':
				cat_totals['SC'] += e; cat = 'SC'
			if per_hb is not None and cat is not None:
				per_hb.append((h['ri_d'], h['d_atom'], h['ri_a'],
					h['a_atom'], e, cat, h['AH'], h['cosBAH'],
					h['cosAHD']))
		return cat_totals
	cache = {}
	if 'XS_atom_types' in params:
		cache.update(patternsearchvina(pose, params, ligand,
			xs_override, nrot_override))
	if 'Atom_types' in params:
		cache.update(ref15atomcache(pose, params))
	if not cache:
		raise Exception(
			'ScoreMatch: unsupported params')
	cache['patternsearch'] = patternsearch
	cache['patternsearchvina'] = patternsearchvina
	cache['ref15atomcache'] = ref15atomcache
	cache['bfswithin'] = bfswithin
	cache['countnrot'] = countnrot
	cache['countnumtors'] = countnumtors
	cache['ringatoms'] = ringatoms
	cache['topologyhash'] = topologyhash
	cache['evalpairs'] = evalpairs
	cache['termresult'] = termresult
	cache['gausspair'] = gausspair
	cache['slopestep'] = slopestep
	cache['ref15pairs'] = ref15pairs
	cache['ljpair'] = ljpair
	cache['ref15ljraw'] = ref15ljraw
	cache['lkisopair'] = lkisopair
	cache['solpair'] = solpair
	cache['ref15solraw'] = ref15solraw
	cache['ref15stubterm'] = ref15stubterm
	cache['fadun_rotwell_grid'] = fadun_rotwell_grid
	cache['fadun_entropy_grid'] = fadun_entropy_grid
	cache['fadun_spline_eval'] = fadun_spline_eval
	cache['fadun_nrchi_data'] = fadun_nrchi_data
	cache['fadun_ypp_psi_grid'] = fadun_ypp_psi_grid
	cache['fadun_nrchi_eval'] = fadun_nrchi_eval
	cache['periodic_cubic_spline'] = periodic_cubic_spline
	cache['spline_eval_1d'] = spline_eval_1d
	cache['rama_spline_eval'] = rama_spline_eval
	cache['hbond_chemtype_maps'] = hbond_chemtype_maps
	cache['hbond_eval_lookup'] = hbond_eval_lookup
	cache['hbond_poly_eval'] = hbond_poly_eval
	cache['hbond_fade'] = hbond_fade
	cache['ref15hbond'] = ref15hbond
	return cache

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
				(e.g. 'REF15', 'AutoDock Vina', 'Default');
				case-insensitive
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
		self._cache = ScoreMatch(pose, self.Parameters, ligand,
			xs_override, nrot_override)
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
		Vina gaussian-1 attractive term, exp(-(d/0.5)^2)
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
		Vina gaussian-2 long-range attractive term, exp(-((d-3)/2)^2)
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
		Vina repulsion term, d^2 where d < 0 (atomic overlap)
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
		Vina hydrophobic contact term, slope_step over hydrophobic pairs
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
		Vina non-directional hydrogen-bond term over donor-acceptor pairs
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
		raw, _ = cache['ref15ljraw'](cache, same_res=False)
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
		_, raw = cache['ref15ljraw'](cache, same_res=False)
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
		raw = cache['ref15solraw'](cache, same_res=False)
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
		pi, pj, r, w = cache['ref15pairs'](cache, same_res=True, cp='cp3')
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
		raw = cache['ref15solraw'](cache, same_res=True)
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
		C0 = float(C.get('coulomb_C0', 322.0637))
		D = float(C.get('sigmoidal_D', 80.0))
		D0 = float(C.get('sigmoidal_D0', 6.0))
		S = float(C.get('sigmoidal_S', 0.4))
		d_min = float(C.get('fa_elec_min_dis', 1.45))
		d_max = float(C.get('fa_elec_max_dis', 5.5))
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
		pi, pj, r, w = cache['ref15pairs'](cache, same_res=False, cp='cp4',
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
		pi, pj, r, w = cache['ref15pairs'](cache, same_res=False, cp='cp4')
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
			return cache['ref15stubterm']('FaDun')
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
				phi = pose.GetDihedral(int(ri), 'PHI')
				psi = pose.GetDihedral(int(ri), 'PSI')
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
			return cache['ref15stubterm']('RamaPreProTerm')
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
				phi = pose.GetDihedral(ri, 'PHI')
				psi = pose.GetDihedral(ri, 'PSI')
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
			return cache['ref15stubterm']('PAaPp')
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
				phi = pose.GetDihedral(int(ri), 'PHI')
				psi = pose.GetDihedral(int(ri), 'PSI')
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
		weight = float(self.Parameters['Omega']['weight'])
		omega_tab = self.Parameters.get('Omega_tables') or {}
		if not omega_tab:
			return cache['ref15stubterm']('Omega')
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
				phi = pose.GetDihedral(int(ri), 'PHI')
				psi = pose.GetDihedral(int(ri), 'PSI')
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
				raw += 0.01 * dangle * dangle
				continue
			if math.isnan(phi): phi = 0.0
			if math.isnan(psi): psi = 0.0
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
		sd_sq = 0.01
		trans_mean = math.radians(176.3)
		trans_sd = math.radians(6.0158)
		cis_mean = math.radians(-2.9105)
		cis_sd = math.radians(5.8239)
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
			nv = place(cd, cg, n, 1.4754, math.radians(77.3), 0.0)
			d2_n_nv = float(np.sum((nv - n) ** 2))
			raw += d2_n_nv / sd_sq
			if int(ri) in n_term_set:
				cav = place(nv, cd, ca, 1.383018,
					math.radians(65.869), 0.0)
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
		mest = math.exp(-20.0)
		wt_len = wt_ang = wt_dihSS = wt_dihCS = 0.1
		d_location = 2.01; d_scale = 0.08; d_shape = 6.0
		a_logA = -419.8120; a_kappa = 419.7; a_mu = 104.22
		dss_logA1 = -32.9599; dss_kappa1 = 30.9053; dss_mu1 = -86.0964
		dss_logA2 = -23.3471; dss_kappa2 = 20.9805; dss_mu2 = 92.3915
		dcs_logA1 = -15.8644; dcs_mu1 = -72.2016; dcs_kappa1 = 13.3778
		dcs_logA2 = -16.9017; dcs_mu2 =  78.0303; dcs_kappa2 = 13.6370
		dcs_logA3 = -7.0219;  dcs_mu3 = -172.5505; dcs_kappa3 = 2.9327
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
		raw = cache['ref15hbond'](pose, cache)['SR_BB']
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
		raw = cache['ref15hbond'](pose, cache)['LR_BB']
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
		raw = cache['ref15hbond'](pose, cache)['BB_SC']
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
		raw = cache['ref15hbond'](pose, cache)['SC']
		w = float(self.Parameters['HBondSc']['weight'])
		return {'inter_raw': 0.0, 'intra_raw': raw,
			'inter_weighted': 0.0, 'intra_weighted': raw * w,
			'raw': raw}
