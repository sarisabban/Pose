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
	def hyb_of(rec): return rec[-1] if rec else 'sp3'
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
					return sum(1 for ci in atom_cands[a] if comp_assn[ci] == 2)
				def remaining(a, comp_assn):
					return [ci for ci in atom_cands[a] if comp_assn[ci] == -1]
				def propagate(comp_assn):
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
			p = pos[0] + off
			return s[p] if p < len(s) else ''
		def take(c):
			if peek() != c: raise ValueError(
				f'Expected {c!r} at {pos[0]} in {s!r}')
			pos[0] += 1
		def read_int():
			start = pos[0]
			while pos[0] < len(s) and s[pos[0]].isdigit(): pos[0] += 1
			return int(s[start:pos[0]]) if pos[0] > start else None
		# atom-expr (until ']' or ':'); precedence low->high: ';' ',' '&' '!'
		def atom_expr():
			# parse low-prec AND chain
			left = atom_or()
			while peek() == ';':
				pos[0] += 1
				right = atom_or()
				left = ('and', left, right)
			return left
		def atom_or():
			left = atom_and()
			while peek() == ',':
				pos[0] += 1
				right = atom_and()
				left = ('or', left, right)
			return left
		def atom_and():
			left = atom_neg()
			while peek() not in ('', ',', ';', ']', ':'):
				if peek() == '&': pos[0] += 1
				right = atom_neg()
				left = ('and', left, right)
			return left
		def atom_neg():
			if peek() == '!':
				pos[0] += 1
				return ('not', atom_neg())
			return atom_prim()
		def atom_prim():
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
			left = bond_or()
			while peek() == ';':
				pos[0] += 1
				right = bond_or()
				left = ('and', left, right)
			return left
		def bond_or():
			left = bond_and()
			while peek() == ',':
				pos[0] += 1
				right = bond_and()
				left = ('or', left, right)
			return left
		def bond_and():
			left = bond_neg()
			# Explicit '&' AND, plus implicit AND between adjacent bond primitives
			while peek() in ('&', '-', '=', '#', ':', '~', '@', '!'):
				if peek() == '&': pos[0] += 1
				right = bond_neg()
				left = ('and', left, right)
			return left
		def bond_neg():
			if peek() == '!':
				pos[0] += 1
				return ('not', bond_neg())
			return bond_prim()
		def bond_prim():
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
			return c == '[' or c == '*' or (
				c and (c.isupper() or c.islower()) and c not in 'hRrXx')
		def parse_branch(prev_idx):
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
		'impropers': [], 'vdw': {}, 'polarisation': {},
		'charges': {i: None for i in sorted_ids},
		'constraints': set()}
	parsed = {}
	def get(smirks):
		if smirks not in parsed: parsed[smirks] = parse(smirks)
		return parsed[smirks]
	# Constraints: rigid X-H bonds (SHAKE/RATTLE in MD; zero bond energy)
	for sm, par in params.get('Constraints', {}).items():
		try: pat = get(sm)
		except Exception: continue
		for tup in match(pat):
			if len(tup) >= 2:
				a, b = int(tup[0]), int(tup[1])
				out['constraints'].add((min(a, b), max(a, b)))
	for sm, par in params.get('Bonds', {}).items():
		try: pat = get(sm)
		except Exception: continue
		for tup in match(pat):
			if len(tup) != 2: continue
			i, j = sorted(tup)
			if (i, j) in edge_set:
				out['bonds'][(i, j)] = [par['r_0'], par['K_b']]
	# Angles: tag :2 is the centre atom
	for sm, par in params.get('Angles', {}).items():
		try: pat = get(sm)
		except Exception: continue
		for tup in match(pat):
			if len(tup) != 3: continue
			i, j, k = tup
			if (min(i, j), max(i, j)) in edge_set and \
				(min(j, k), max(j, k)) in edge_set:
				ii, kk = (i, k) if i < k else (k, i)
				out['angles'][(ii, j, kk)] = [par['theta_0'], par['K_theta']]
	# Dedicated UB section: 3-atom narrow SMIRKS (independent of Angles)
	for sm, par in params.get('UB', {}).items():
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
	for sm, par in params.get('ProperTorsions', {}).items():
		try: pat = get(sm)
		except Exception: continue
		for tup in match(pat):
			if len(tup) != 4: continue
			i, j, k, l = tup
			if (min(i, j), max(i, j)) not in edge_set: continue
			if (min(j, k), max(j, k)) not in edge_set: continue
			if (min(k, l), max(k, l)) not in edge_set: continue
			if (i, j, k, l) > (l, k, j, i): i, j, k, l = l, k, j, i
			out['propers'][(i, j, k, l)] = [
				[c['n'], c['phi_0'], c['K_phi'],
					c.get('idivf', 1.0)]
				for c in par['components']]
	# Impropers: tag :2 is centre; trefoil = 3 cyclic outer perms; centre stored at tup[0]
	imp_by_centre = {}
	for sm, par in params.get('ImproperTorsions', {}).items():
		try: pat = get(sm)
		except Exception: continue
		for tup in match(pat):
			if len(tup) != 4: continue
			a1, a2, a3, a4 = tup
			perms = [(a1, a3, a4), (a3, a4, a1), (a4, a1, a3)]
			entries = []
			for o1, o2, o3 in perms:
				for c in par['components']:
					entries.append((a2, o1, o2, o3,
						c['n'], c['phi_0'], c['K_phi'] / 3.0))
			imp_by_centre[a2] = entries
	for entries in imp_by_centre.values():
		out['impropers'].extend(entries)
	# vdW: per-atom (eps, sigma); convert rmin_half->sigma if rmin_half provided
	rmin2sig = 2.0 / (2.0 ** (1.0 / 6.0))
	for sm, par in params.get('vdW', {}).items():
		try: pat = get(sm)
		except Exception: continue
		for tup in match(pat):
			if len(tup) != 1: continue
			i = tup[0]
			eps = par['epsilon']
			if 'sigma' in par: sig = par['sigma']
			else: sig = par['r'] * rmin2sig
			out['vdw'][i] = [eps, sig]
			# Atomic polarisability co-located with the vdW entry;
			# defaults to 0 for legacy DBs that don't carry alpha.
			out['polarisation'][i] = par.get('alpha', 0.0)
	for sm, par in params.get('LibraryCharges', {}).items():
		try: pat = get(sm)
		except Exception: continue
		qs = par.get('q', [])
		for tup in match(pat):
			for k, idx in enumerate(tup):
				if k < len(qs): out['charges'][idx] = float(qs[k])
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
				(e.g. 'Default', 'openFF'); matched case-insensitively
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
			matched_centres = {tup[0] for tup in assigns['impropers']}
			for c in atoms_set:
				if len(nbr_local[c]) == 3 and c not in matched_centres:
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
					code = bb_per_res[ri][1]
					grid = cmap_section.get(code)
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
			self._cache = cache
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
		raise NotImplementedError(
			"Score() reads database keys (weights, ref_state, "
			"lk_solvation, hbond, kbp, rotamer) that were removed in "
			"the Energy Parameters refactor. Pending migration to "
			"OpenFF SMIRKS-driven score terms.")
		if ff is None: ff = ForceField()
		self.ff = ff
		self.box = box
		db = DBLoad()
		P = db['Energy Parameters']
		self.weights     = P['weights']
		self.ref_state   = P['ref_state']
		self.lk          = P['lk_solvation']
		self.hb          = P['hbond']
		self.kbp         = P['kbp']
		self._kbp_table  = np.asarray(self.kbp['table'], dtype=np.float64)
		self._lk_types   = None
		self._kbp_types  = None
		self._lk_dG      = None
		self._lk_lam     = None
		self._lk_V       = None
		self._cache_hash = None
		# Rotamer Library: backbone-dependent rotamer mixture data, CSR-packed
		# (residues -> n_chi / rotamers{columns, table, bin_offsets} / densities).
		# Used by _rotamer_prior to evaluate the multimodal mixture log-likelihood.
		rl = db.get('Rotamer Library', {}) or {}
		self._rotlib       = rl.get('residues', {})
		self._rl_phi_start = float(rl.get('phi_start', -180.0))
		self._rl_phi_step  = float(rl.get('phi_step',   10.0))
		self._rl_phi_n     = int  (rl.get('phi_n',     36))
		self._rl_psi_start = float(rl.get('psi_start', -180.0))
		self._rl_psi_step  = float(rl.get('psi_step',   10.0))
		self._rl_psi_n     = int  (rl.get('psi_n',     36))
		self._rl_warned    = set()
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
		h = self.ff._topologyhash(pose)
		if self.ff._cache is None or self.ff._cache_hash != h:
			self.ff._prepare(pose)
			self._cache_hash = None
		cache = self.ff._cache
		if self._cache_hash != h:
			self._build_typing(pose)
			self._cache_hash = h
		E_lj   = self.ff.VDWPotential(pose, cache=cache, grad=False,
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
					f"missing from database.json"
					f"['Energy Parameters']['lk_solvation']['atom_types']. "
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
					f"missing from database.json"
					f"['Energy Parameters']['kbp']['atom_types']. "
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
			float: sum of ref_state[aa] over every residue, in kJ/mol
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
					f"database.json['Energy Parameters']['ref_state']. "
					f"Add a per-aa reference energy. Known canonical "
					f"codes: ACDEFGHIKLMNPQRSTVWY; non-canonical: "
					f"BJOUXZ.")
			total += d[key]
		return total
	def _rotlib_cell(self, three_letter, phi_deg, psi_deg):
		'''
		Slice the Rotamer Library CSR table for one (residue, phi, psi) cell
		Arguments:
		----------
			three_letter: str - 3-letter residue code (uppercase)
			phi_deg, psi_deg: float - backbone angles in degrees
		Returns:
		--------
			tuple (entry, table_slice) where:
				entry: dict {'n_chi': int} or None if the residue is not in the library
				table_slice: list of rows (each [r..., count, prob, chi..., sig...])
		'''
		entry = self._rotlib.get(three_letter)
		if entry is None: return None, []
		rot = entry['rotamers']
		bin_offsets = rot['bin_offsets']
		i_phi = int(math.floor((phi_deg - self._rl_phi_start)
			/ self._rl_phi_step)) % self._rl_phi_n
		i_psi = int(math.floor((psi_deg - self._rl_psi_start)
			/ self._rl_psi_step)) % self._rl_psi_n
		bidx = i_phi * self._rl_psi_n + i_psi
		start = bin_offsets[bidx]
		end = bin_offsets[bidx + 1]
		return entry, rot['table'][start:end]
	def _rotamer_prior(self, pose):
		'''
		Multimodal rotamer prior: per-residue mixture-of-Gaussians log-likelihood
		evaluated at the current chi tuple, given the residue's backbone cell
		from the Rotamer Library. Each rotamer k contributes
			P_k(phi,psi) * prod_c N(chi_c; mu_kc, sigma_kc)
		and the residue energy is  -kT * log( sum_k that ),
		stably evaluated via logsumexp.
		Per-rotamer sigmas come from the library (NOT a global hyperparameter).
		Arguments:
		----------
			pose: Pose - molecule source pose
		Returns:
		--------
			float: total rotamer-prior energy in kJ/mol summed over residues
		'''
		aas = pose.data.get('Amino Acids')
		if aas is None or not self._rotlib: return 0.0
		kT      = 2.494                  # RT at 300 K, kJ/mol
		LOG_2PI = math.log(2.0 * math.pi)
		SIG_MIN = 0.5                    # degrees, numerical floor
		total   = 0.0
		for r, info in aas.items():
			c    = info[0]
			aa_u = c.upper()
			aa_db     = pose.aminoacids.get(aa_u, {})
			chi_atoms = aa_db.get('Chi Angle Atoms') or []
			if not chi_atoms: continue
			three = aa_db.get('Tricode')
			if not three: continue
			# D-amino acid handling: pose stores lowercase code; library is
			# keyed on the L-form 3-letter. Mirror phi/psi for lookup and
			# negate library mu values when reading rotamers back.
			flip = (c != aa_u)
			phi = pose.GetDihedral(r, 'PHI')
			psi = pose.GetDihedral(r, 'PSI')
			if math.isnan(phi) or math.isnan(psi): continue
			phi_q = -phi if flip else phi
			psi_q = -psi if flip else psi
			entry, rows = self._rotlib_cell(three, phi_q, psi_q)
			if entry is None:
				if three not in self._rl_warned:
					self._rl_warned.add(three)
				continue
			if not rows: continue
			n_chi = int(entry['n_chi'])
			if n_chi == 0: continue
			# Snapshot residue's current chi values once.
			chi_now = np.empty(n_chi, dtype=np.float64)
			bad = False
			for ci in range(n_chi):
				v = pose.GetDihedral(r, 'CHI', chi_type=ci+1)
				if math.isnan(v): bad = True; break
				chi_now[ci] = v
			if bad: continue
			# Column layout: [count, prob, chi1..N, sig1..N]
			prob_i = 1
			chi_i  = 2
			sig_i  = 2 + n_chi
			log_terms = []
			for row in rows:
				P_k = row[prob_i]
				if P_k <= 0.0: continue
				lt = math.log(P_k)
				for ci in range(n_chi):
					mu_kc  = row[chi_i + ci]
					sig_kc = row[sig_i + ci]
					if sig_kc < SIG_MIN: sig_kc = SIG_MIN
					if flip: mu_kc = -mu_kc
					d = ((chi_now[ci] - mu_kc + 180.0) % 360.0) - 180.0
					lt += -0.5*LOG_2PI - math.log(sig_kc) \
						- 0.5 * (d / sig_kc) ** 2
				log_terms.append(lt)
			if not log_terms: continue
			# Stable logsumexp.
			m = max(log_terms)
			lse = m + math.log(sum(math.exp(lt - m) for lt in log_terms))
			total += -kT * lse
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
			float: solvation free energy in kJ/mol
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
			float: H-bond energy contribution in kJ/mol
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
			float: KBP energy contribution in kJ/mol
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
