#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import re
import io
import os
import sys
import json
import math
import time
import shutil
import base64
import pickle
import zipfile
import itertools
import numpy as np
import urllib.request
import xml.etree.ElementTree as ET
from .pose import DBLoad
from .energy import ForceField
from collections import defaultdict, deque

def SMIRKSMatch(pose, params):
	'''
	Assign force-field parameters to a pose by SMIRKS pattern matching,
	or by atom class for the <at=...>-keyed force fields
	Arguments:
	----------
		pose:   Pose - molecule, protein, DNA, or RNA pose
		params: dict - force-field section dict containing Constraints/
			Bonds/Angles/UB/ProperTorsions/ImproperTorsions/vdW/
			LibraryCharges keys (typically ForceField.mol)
	Returns:
	--------
		dict: 'bonds' {(i,j): [r_0, K_b]}; 'angles' and 'ub'
		{(i,j,k): [x_0, K]}; 'propers' {(i,j,k,l): [[n, phi_0, K, idivf]]};
		'impropers' list of (i,j,k,l,n,phi_0,K); 'vdw' and 'vdw14'
		{i: [epsilon, sigma]}; 'polarisation' {i: alpha}; 'charges'
		{i: charge or None}; 'constraints' set of (i,j); 'restri'
		{residue index: resolved tricode}
	'''
	Z_TABLE = {
		'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
		'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
		'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
		'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,
		'Br':35,'Kr':36,'Rb':37,'Sr':38,'I':53,'Xe':54,'Cs':55,'Ba':56}
	VAL = {'C':4,'N':3,'O':2,'S':2,'P':5,'Se':2,
		'F':1,'Cl':1,'Br':1,'I':1,'H':1,'B':3}
	HEAVY_ALIAS = {'CD1': 'CD'}
	def findrings(ctx):
		'''
		Smallest set of smallest rings via per-edge BFS shortest cycle
		Arguments:
		----------
			ctx: dict - molecule tables carrying 'edges' and 'nbr'
		Returns:
		--------
			list: each ring as a tuple of atom indices (closed cycle)
		'''
		nbr = ctx['nbr']
		seen = set()
		out = []
		for u, v in ctx['edges']:
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
			mn = min(ring)
			rot = ring[ring.index(mn):] + ring[:ring.index(mn)]
			canon = min(rot, (rot[0],) + rot[:0:-1])
			if canon in seen: continue
			seen.add(canon)
			out.append(canon)
		return out
	def hybof(rec):
		'''
		Hybridisation tag from an atom record, defaulting to sp3
		Arguments:
		----------
			rec: list - atom record from pose.data['Atoms']
		Returns:
		--------
			str: hybridisation tag at rec[-1], or 'sp3' when rec is empty
		'''
		return rec[-1] if rec else 'sp3'
	def kekulise(ctx, rings):
		'''
		Assign Kekule bond orders to every 1.5-order bond, per connected
		component, by budget propagation plus depth-first search. Components
		with no valid assignment fall back to "non-ring 1.5 -> 1.0"
		Arguments:
		----------
			ctx:   dict - molecule tables; ctx['bo'] is mutated in place
			rings: list - SSSR rings used to protect in-ring bonds
		Returns:
		--------
			None: ctx['bo'] is mutated in place
		'''
		bo = ctx['bo']
		nbr = ctx['nbr']
		atoms = ctx['atoms']
		fc = ctx['fc']
		cands = sorted(e for e in ctx['edges']
			if abs(bo.get(e, 1.0) - 1.5) < 1e-6)
		if not cands: return
		candset = set(cands)
		inring = set()
		for r in rings:
			for k in range(len(r)):
				a, b = r[k], r[(k + 1) % len(r)]
				e = (min(a, b), max(a, b))
				if e in candset: inring.add(e)
		byatom = {}
		for e in cands:
			byatom.setdefault(e[0], []).append(e)
			byatom.setdefault(e[1], []).append(e)
		seen = set()
		comps = []
		for start in cands:
			if start in seen: continue
			comp = []
			queue = [start]
			seen.add(start)
			while queue:
				e = queue.pop()
				comp.append(e)
				for atom in (e[0], e[1]):
					for ne in byatom.get(atom, []):
						if ne in seen: continue
						seen.add(ne)
						queue.append(ne)
			comps.append(sorted(comp))
		for comp in comps:
			cset = set(comp)
			touched = set()
			for (a, b) in comp:
				touched.add(a)
				touched.add(b)
			atcands = {a: [] for a in touched}
			for ci, e in enumerate(comp):
				atcands[e[0]].append(ci)
				atcands[e[1]].append(ci)
			budget = {}
			ok = True
			for a in touched:
				if atoms[a][1] not in VAL: ok = False; break
				v = VAL[atoms[a][1]] + fc.get(a, 0)
				for j in nbr[a]:
					e = (min(a, j), max(a, j))
					if e not in cset: v -= bo.get(e, 1.0)
				v -= len(atcands[a])
				budget[a] = int(round(v))
				if budget[a] < 0 or budget[a] > len(atcands[a]):
					ok = False; break
			done = None
			stack = [[-1] * len(comp)] if ok else []
			while stack:
				cur = stack.pop()
				bad = False
				changed = True
				while changed and not bad:
					changed = False
					for a in touched:
						rem = [c for c in atcands[a] if cur[c] == -1]
						got = sum(1 for c in atcands[a] if cur[c] == 2)
						need = budget[a] - got
						if need < 0 or need > len(rem): bad = True; break
						if not rem: continue
						if need and need != len(rem): continue
						for c in rem: cur[c] = 1 if need == 0 else 2
						changed = True
				if bad: continue
				una = [c for c in range(len(comp)) if cur[c] == -1]
				if not una:
					done = cur
					break
				for v in (1, 2):
					nxt = list(cur)
					nxt[una[0]] = v
					stack.append(nxt)
			if done is not None:
				for ci, e in enumerate(comp):
					bo[e] = 2.0 if done[ci] == 2 else 1.0
				continue
			for e in comp:
				if e not in inring: bo[e] = 1.0
	def aromatiserings(ctx, rings):
		'''
		Re-mark Kekule aromatic 5- and 6-rings as bo=1.5 for SMIRKS ':'
		Arguments:
		----------
			ctx:   dict - molecule tables; ctx['bo'] is mutated in place
			rings: list - SSSR rings as tuples of atom indices
		Returns:
		--------
			None: ctx['bo'] is mutated in place
		'''
		bo = ctx['bo']
		atoms = ctx['atoms']
		for r in rings:
			if len(r) not in (5, 6): continue
			if not all(hybof(atoms[a]) == 'sp2' for a in r): continue
			re = [(min(r[k], r[(k + 1) % len(r)]),
				max(r[k], r[(k + 1) % len(r)])) for k in range(len(r))]
			if not any(abs(bo.get(e, 1.0) - 2.0) < 0.1
				or abs(bo.get(e, 1.0) - 1.5) < 0.1 for e in re): continue
			for e in re: bo[e] = 1.5
	def peek(st, off=0):
		'''
		Peek at the character off positions ahead of the cursor
		Arguments:
		----------
			st:  dict - parser state carrying 's' and 'pos'
			off: int, default 0 - offset from the cursor
		Returns:
		--------
			str: single character at cursor+off, or '' past end of input
		'''
		p = st['pos'][0] + off
		return st['s'][p] if p < len(st['s']) else ''
	def take(st, c):
		'''
		Consume the expected character at the cursor or raise ValueError
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
			c:  str  - expected single character
		Returns:
		--------
			No return value; the cursor advances by one
		'''
		if peek(st) != c: raise ValueError(
			f'Expected {c!r} at {st["pos"][0]} in {st["s"]!r}')
		st['pos'][0] += 1
	def readint(st):
		'''
		Consume a run of decimal digits at the cursor
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			int or None: value if any digits were read, else None
		'''
		s, pos = st['s'], st['pos']
		start = pos[0]
		while pos[0] < len(s) and s[pos[0]].isdigit(): pos[0] += 1
		return int(s[start:pos[0]]) if pos[0] > start else None
	def atomprim(st):
		'''
		Parse one atom primitive: wildcard, aromaticity, ring, degree, H
		count, formal charge, atomic number, recursion, or element symbol
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			tuple: AST node for the primitive
		'''
		s, pos = st['s'], st['pos']
		c = peek(st)
		if c == '*':
			pos[0] += 1; return ('wild',)
		if c == 'a':
			pos[0] += 1; return ('arom', True)
		if c == 'A':
			pos[0] += 1; return ('arom', False)
		if c == 'R':
			pos[0] += 1; return ('Rcount', readint(st))
		if c == 'r':
			pos[0] += 1; return ('rsize', readint(st))
		if c == 'X':
			pos[0] += 1; n = readint(st)
			return ('X', n if n is not None else 0)
		if c == 'x':
			pos[0] += 1; n = readint(st)
			return ('x', n if n is not None else 0)
		if c == 'H':
			pos[0] += 1; n = readint(st)
			return ('H', 1 if n is None else n)
		if c == 'h':
			pos[0] += 1; n = readint(st)
			return ('h', 1 if n is None else n)
		if c == '+':
			pos[0] += 1; n = readint(st)
			return ('fc', 1 if n is None else n)
		if c == '-':
			pos[0] += 1; n = readint(st)
			return ('fc', -1 if n is None else -n)
		if c == '#':
			pos[0] += 1; return ('Z', readint(st))
		if c == '$':
			pos[0] += 1; take(st, '(')
			depth = 1
			start = pos[0]
			while pos[0] < len(s) and depth:
				if s[pos[0]] == '(': depth += 1
				elif s[pos[0]] == ')': depth -= 1
				pos[0] += 1
			return ('recurse', s[start:pos[0] - 1])
		if c.isupper():
			name = c
			pos[0] += 1
			if peek(st).islower(): name += peek(st); pos[0] += 1
			return ('Z', Z_TABLE.get(name, 0))
		raise ValueError(f'Unknown primitive {c!r} at pos {pos[0]} in {s!r}')
	def atomneg(st):
		'''
		Parse an optionally-negated atom primitive; '!' toggles negation
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			tuple: AST node, wrapped in ('not', ...) when prefixed by '!'
		'''
		if peek(st) == '!':
			st['pos'][0] += 1
			return ('not', atomneg(st))
		return atomprim(st)
	def atomand(st):
		'''
		Parse one AND-chain of atom expressions joined by '&' or adjacency
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			tuple: nested AST node for the parsed expression
		'''
		left = atomneg(st)
		while peek(st) not in ('', ',', ';', ']', ':'):
			if peek(st) == '&': st['pos'][0] += 1
			left = ('and', left, atomneg(st))
		return left
	def atomor(st):
		'''
		Parse one OR-chain of atom expressions joined by ','
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			tuple: nested AST node for the parsed expression
		'''
		left = atomand(st)
		while peek(st) == ',':
			st['pos'][0] += 1
			left = ('or', left, atomand(st))
		return left
	def atomexpr(st):
		'''
		Parse a full atom expression: AND-chains joined by ';' (low precedence)
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			tuple: nested AST node for the parsed expression
		'''
		left = atomor(st)
		while peek(st) == ';':
			st['pos'][0] += 1
			left = ('and', left, atomor(st))
		return left
	def bondprim(st):
		'''
		Parse one bond primitive: - = # : ~ @ / or backslash
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			tuple: AST node for the bond primitive
		'''
		ORD = {'-': 1.0, '=': 2.0, '#': 3.0, ':': 1.5}
		c = peek(st)
		if c in ORD:
			st['pos'][0] += 1; return ('bo', ORD[c])
		if c == '@':
			st['pos'][0] += 1; return ('inring',)
		if c in ('~', '/', '\\'):
			st['pos'][0] += 1; return ('any',)
		raise ValueError(
			f'Unknown bond op {c!r} at {st["pos"][0]} in {st["s"]!r}')
	def bondneg(st):
		'''
		Parse an optionally-negated bond primitive; '!' toggles negation
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			tuple: AST node, wrapped in ('not', ...) when prefixed by '!'
		'''
		if peek(st) == '!':
			st['pos'][0] += 1
			return ('not', bondneg(st))
		return bondprim(st)
	def bondand(st):
		'''
		Parse one AND-chain of bond expressions joined by '&' or adjacency
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			tuple: nested AST node for the parsed bond expression
		'''
		left = bondneg(st)
		while peek(st) in ('&', '-', '=', '#', ':', '~', '@', '!'):
			if peek(st) == '&': st['pos'][0] += 1
			left = ('and', left, bondneg(st))
		return left
	def bondor(st):
		'''
		Parse one OR-chain of bond expressions joined by ','
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			tuple: nested AST node for the parsed bond expression
		'''
		left = bondand(st)
		while peek(st) == ',':
			st['pos'][0] += 1
			left = ('or', left, bondand(st))
		return left
	def bondexpr(st):
		'''
		Parse a full bond expression: AND-chains joined by ';'
		Arguments:
		----------
			st: dict - parser state carrying 's' and 'pos'
		Returns:
		--------
			tuple: nested AST node for the bond expression
		'''
		left = bondor(st)
		while peek(st) == ';':
			st['pos'][0] += 1
			left = ('and', left, bondor(st))
		return left
	def parseatom(st):
		'''
		Parse a bracketed atom '[...]' or a bare atom symbol
		Arguments:
		----------
			st: dict - parser state carrying 's', 'pos', 'atoms' and 'tags'
		Returns:
		--------
			int: index of the newly appended atom in st['atoms']
		'''
		pos = st['pos']
		c = peek(st)
		if c != '[':
			if c == '*':
				pos[0] += 1
				expr = ('wild',)
			elif c.isupper():
				name = c
				pos[0] += 1
				if peek(st).islower() and (name + peek(st)) in Z_TABLE:
					name += peek(st); pos[0] += 1
				expr = ('Z', Z_TABLE.get(name, 0))
			elif c.islower():
				pos[0] += 1
				expr = ('and', ('Z', Z_TABLE.get(c.upper(), 0)),
					('arom', True))
			else:
				raise ValueError(f'Expected atom at {pos[0]} in {st["s"]!r}')
			st['atoms'].append({'expr': expr, 'tag': None})
			return len(st['atoms']) - 1
		take(st, '[')
		expr = atomexpr(st)
		tag = None
		if peek(st) == ':':
			pos[0] += 1
			tag = readint(st)
		take(st, ']')
		st['atoms'].append({'expr': expr, 'tag': tag})
		if tag is not None: st['tags'][tag] = len(st['atoms']) - 1
		return len(st['atoms']) - 1
	def parsebranch(st, previdx):
		'''
		Parse a parenthesised branch sub-chain attached to atom previdx
		Arguments:
		----------
			st:      dict - parser state
			previdx: int  - atom index this branch attaches to
		Returns:
		--------
			No return value; st['atoms'] and st['bonds'] are extended
		'''
		take(st, '(')
		c = peek(st)
		start = c == '[' or c == '*' or (
			c and (c.isupper() or c.islower()) and c not in 'hRrXx')
		be = ('bo', 1.0) if (not c or start or c == '(') else bondexpr(st)
		aidx = parseatom(st)
		st['bonds'].append((previdx, aidx, be))
		parsechain(st, aidx)
		take(st, ')')
	def parsechain(st, previdx):
		'''
		Parse a chain of atoms and bonds at the cursor, extending previdx
		Arguments:
		----------
			st:      dict - parser state
			previdx: int  - atom index this chain extends from
		Returns:
		--------
			No return value; st['atoms'] and st['bonds'] are extended
		'''
		s, pos = st['s'], st['pos']
		while pos[0] < len(s):
			c = peek(st)
			if c == ')' or c == '': return
			if c == '(':
				parsebranch(st, previdx); continue
			start = c == '[' or c == '*' or (
				c and (c.isupper() or c.islower()) and c not in 'hRrXx')
			if start:
				aidx = parseatom(st)
				st['bonds'].append((previdx, aidx, ('bo', 1.0)))
				previdx = aidx
				continue
			be = None
			if not (c.isdigit() or c == '%'):
				be = bondexpr(st)
				c = peek(st)
				start = c == '[' or c == '*' or (
					c and (c.isupper() or c.islower()) and c not in 'hRrXx')
				if start:
					aidx = parseatom(st)
					st['bonds'].append((previdx, aidx, be))
					previdx = aidx
					continue
				if not (c.isdigit() or c == '%'): raise ValueError(
					f'Unexpected after bond at {pos[0]} in {s!r}')
			if c == '%':
				pos[0] += 1
				digit = int(s[pos[0]:pos[0] + 2])
				pos[0] += 2
			else:
				digit = int(c); pos[0] += 1
			if digit not in st['ring']:
				st['ring'][digit] = (previdx, be if be else ('bo', 1.0))
				continue
			a, beopen = st['ring'].pop(digit)
			if be is None: st['bonds'].append((previdx, a, beopen))
			else: st['bonds'].append((previdx, a,
				be if be != ('bo', 1.0) else beopen))
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
		st = {'s': smirks, 'pos': [0], 'atoms': [], 'bonds': [],
			'tags': {}, 'ring': {}}
		parsechain(st, parseatom(st))
		if st['ring']: raise ValueError(
			f'Unclosed ring digits {list(st["ring"])} in {smirks!r}')
		return {'atoms': st['atoms'], 'bonds': st['bonds'], 'tags': st['tags']}
	def getpat(smirks, ctx):
		'''
		Memoised parser: cache parsed SMIRKS across repeated lookups
		Arguments:
		----------
			smirks: str  - SMIRKS pattern
			ctx:    dict - molecule tables carrying the 'parsed' cache
		Returns:
		--------
			dict: parsed pattern (atoms, bonds, tags)
		'''
		if smirks not in ctx['parsed']:
			try:
				ctx['parsed'][smirks] = parse(smirks)
			except ValueError as e:
				warnings.warn(f'Unparseable SMIRKS {smirks!r}: {e}')
				raise
		return ctx['parsed'][smirks]
	def evalatom(expr, i, ctx):
		'''
		Evaluate a parsed atom expression against atom index i
		Arguments:
		----------
			expr: tuple - parsed AST node
			i:    int   - candidate atom index in the molecule
			ctx:  dict  - molecule tables
		Returns:
		--------
			bool: True iff atom i satisfies the expression
		'''
		k = expr[0]
		if k == 'wild':   return True
		if k == 'Z':      return ctx['Z'][i] == expr[1]
		if k == 'X':      return ctx['X'][i] == expr[1]
		if k == 'H':      return ctx['Hc'][i] == expr[1]
		if k == 'x':      return ctx['xcount'][i] == expr[1]
		if k == 'h':      return ctx['Hc'][i] == expr[1]
		if k == 'fc':     return ctx['fc'][i] == expr[1]
		if k == 'arom':   return ctx['aroma'][i] == expr[1]
		if k == 'rsize':
			if expr[1] is None: return bool(ctx['ringsz'][i])
			return ctx['minring'][i] == expr[1]
		if k == 'Rcount':
			c = sum(1 for r in ctx['rings'] if i in r)
			if expr[1] is None: return c > 0
			return c == expr[1]
		if k == 'and':
			return evalatom(expr[1], i, ctx) and evalatom(expr[2], i, ctx)
		if k == 'or':
			return evalatom(expr[1], i, ctx) or evalatom(expr[2], i, ctx)
		if k == 'not':    return not evalatom(expr[1], i, ctx)
		if k == 'recurse':
			key = (expr[1], i)
			if key in ctx['rcache']: return ctx['rcache'][key]
			ctx['rcache'][key] = False
			ok = bool(match(parse(expr[1]), ctx, anchor=i))
			ctx['rcache'][key] = ok
			return ok
		raise ValueError(f'Unknown atom-expr node {k!r}')
	def evalbond(expr, e, ctx):
		'''
		Evaluate a parsed bond expression against canonical edge e=(a,b)
		Arguments:
		----------
			expr: tuple - parsed bond AST
			e:    tuple - (a_idx, b_idx) with a_idx < b_idx
			ctx:  dict  - molecule tables
		Returns:
		--------
			bool: True iff the bond satisfies the expression
		'''
		k = expr[0]
		if k == 'any':    return True
		if k == 'bo':
			if abs(expr[1] - 1.5) < 1e-6: return ctx['aromb'].get(e, False)
			return abs(ctx['bo'].get(e, 1.0) - expr[1]) < 1e-6 \
				and not ctx['aromb'].get(e, False)
		if k == 'inring': return e in ctx['inringbond']
		if k == 'and':
			return evalbond(expr[1], e, ctx) and evalbond(expr[2], e, ctx)
		if k == 'or':
			return evalbond(expr[1], e, ctx) or evalbond(expr[2], e, ctx)
		if k == 'not':    return not evalbond(expr[1], e, ctx)
		raise ValueError(f'Unknown bond-expr node {k!r}')
	def walk(p, mst, ctx):
		'''
		Backtracking subgraph isomorphism over pattern atoms in index order.
		With an anchor set, pattern atom 0 is pinned and the search stops at
		the first complete mapping; otherwise every distinct tag tuple is
		collected into mst['res']
		Arguments:
		----------
			p:   int  - pattern atom index being mapped
			mst: dict - match state (pattern, adjacency, mapping, results)
			ctx: dict - molecule tables
		Returns:
		--------
			bool: True once a complete consistent mapping is found
		'''
		pat, mapping, used = mst['pat'], mst['map'], mst['used']
		if p == mst['n']:
			if mst['anchor'] is not None: return True
			key = tuple(mapping[pat['tags'][t]] for t in mst['tags'])
			if key not in mst['seen']:
				mst['seen'].add(key)
				mst['res'].append(key)
			return True
		fixed = [(q, be) for q, be in mst['adj'][p] if mapping[q] >= 0]
		if p == 0 and mst['anchor'] is not None: cands = [mst['anchor']]
		elif fixed: cands = [m for m in ctx['nbr'][mapping[fixed[0][0]]]
			if m not in used]
		else: cands = [a for a in ctx['ids'] if a not in used]
		for cand in cands:
			if cand in used: continue
			if not evalatom(pat['atoms'][p]['expr'], cand, ctx): continue
			ok = True
			for q, be in fixed:
				e = (min(cand, mapping[q]), max(cand, mapping[q]))
				if e not in ctx['edgeset']: ok = False; break
				if not evalbond(be, e, ctx): ok = False; break
			if not ok: continue
			mapping[p] = cand
			used.add(cand)
			if walk(p + 1, mst, ctx) and mst['anchor'] is not None:
				return True
			used.discard(cand)
			mapping[p] = -1
		return False
	def match(pat, ctx, anchor=None):
		'''
		Match a parsed pattern against the molecule
		Arguments:
		----------
			pat:    dict - parsed pattern (atoms, bonds, tags)
			ctx:    dict - molecule tables
			anchor: int or None - when given, pin pattern atom 0 to it
		Returns:
		--------
			bool when anchor is given, else a list of tuples of atom indices
			in ascending tag order (only tagged atoms are returned)
		'''
		n = len(pat['atoms'])
		adj = {p: [] for p in range(n)}
		for a, b, be in pat['bonds']:
			adj[a].append((b, be))
			adj[b].append((a, be))
		mst = {'pat': pat, 'adj': adj, 'n': n, 'map': [-1] * n,
			'used': set(), 'res': [], 'seen': set(),
			'tags': sorted(pat['tags'].keys()), 'anchor': anchor}
		hit = walk(0, mst, ctx)
		return hit if anchor is not None else mst['res']
	def v2v3(nm):
		'''
		Convert a PDB v2 atom name to v3 (leading digits move to the end)
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
	def maptemplate(reskeys, ratoms, aliases, tp, ctx):
		'''
		Match one residue's pose atoms to the first template present among
		reskeys: exact name, alias, PDB v2/v3 transform, then a parent
		heavy-atom topology fallback
		Arguments:
		----------
			reskeys: list of str - candidate residue variant keys
			ratoms:  list of int - the residue's atom indices
			aliases: dict or None - extra exact-name aliases
			tp:      dict - typing tables (class, charge, reskey, tname)
			ctx:     dict - molecule tables
		Returns:
		--------
			None: tp is filled in place for every matched atom
		'''
		aliases = aliases or {}
		tpl, chosen = None, None
		for k in reskeys:
			if ctx['templates'].get(k) is not None:
				tpl, chosen = ctx['templates'][k], k
				break
		if tpl is None: return
		tatoms = {a[0]: (a[1], a[2], a[3]) for a in tpl['atoms']}
		tadj = {}
		for x, y in tpl['bonds']:
			tadj.setdefault(x, set()).add(y)
			tadj.setdefault(y, set()).add(x)
		rset = set(ratoms)
		padj = {a: [b for b in ctx['nbr'][a] if b in rset] for a in ratoms}
		hmap = {}
		for a in ratoms:
			if tp['elem'][a] == 'H': continue
			nm = tp['name'][a]
			if nm in tatoms: hmap[a] = nm
			elif HEAVY_ALIAS.get(nm) in tatoms: hmap[a] = HEAVY_ALIAS[nm]
		for a in ratoms:
			nm, el = tp['name'][a], tp['elem'][a]
			hit, tn = None, None
			if nm in tatoms: hit, tn = tatoms[nm], nm
			elif aliases.get(nm) in tatoms:
				tn = aliases[nm]; hit = tatoms[tn]
			elif el != 'H' and HEAVY_ALIAS.get(nm) in tatoms:
				tn = HEAVY_ALIAS[nm]; hit = tatoms[tn]
			elif el == 'H' and v2v3(nm) in tatoms:
				tn = v2v3(nm); hit = tatoms[tn]
			else:
				parents = [hmap[x] for x in padj[a] if x in hmap]
				for cn, val in tatoms.items():
					if val[0] != el: continue
					if any(p in tadj.get(cn, ()) for p in parents):
						hit, tn = val, cn
						break
			if hit is None: continue
			tp['class'][a] = hit[1]
			tp['charge'][a] = hit[2]
			tp['reskey'][a] = chosen
			tp['tname'][a] = tn
	def assigntypes(pose, tp, ctx, out):
		'''
		Resolve every atom's residue key, atom class and partial charge from
		the force-field residue templates, honouring N/C and 5'/3' terminal
		variants, HIS protonation and disulfide CYX retagging
		Arguments:
		----------
			pose: Pose - the molecule being typed
			tp:   dict - typing tables filled in place
			ctx:  dict - molecule tables
			out:  dict - assignment dict; out['restri'] is filled in place
		Returns:
		--------
			None: tp and out['restri'] are filled in place
		'''
		aas = pose.data.get('Amino Acids') or {}
		nucs = pose.data.get('Nucleotides') or {}
		nbr = ctx['nbr']
		riatoms = {ri: list(aas[ri][2]) + list(aas[ri][3]) for ri in aas}
		sgof = {a: ri for ri, ats in riatoms.items()
			for a in ats if tp['name'].get(a) == 'SG'}
		ssres = set()
		for a in sgof:
			for b in nbr[a]:
				if b in sgof and sgof[b] != sgof[a]:
					ssres.add(sgof[a]); ssres.add(sgof[b])
		resof = {a: ri for ri, ats in riatoms.items() for a in ats}
		prot = {}
		for ri in sorted(aas):
			prot.setdefault(aas[ri][1], []).append(ri)
		for ris in prot.values():
			for ri in ris:
				tri = str(aas[ri][5]).upper()
				ats = riatoms[ri]
				anames = {tp['name'].get(a) for a in ats}
				if tri in ('HIS', 'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP'):
					hd1 = 'HD1' in anames
					if hd1 and 'HE2' in anames: cand = ('HIP', 'HSP')
					elif hd1: cand = ('HID', 'HSD')
					else: cand = ('HIE', 'HSE')
					tri = next((v for v in cand if any((p + v) in
						ctx['templates'] for p in ('', 'N', 'C'))), cand[0])
				if tri == 'CYS' and ri in ssres: tri = 'CYX'
				out['restri'][ri] = tri
				nat = next((a for a in ats if tp['name'].get(a) == 'N'), None)
				cat = next((a for a in ats if tp['name'].get(a) == 'C'), None)
				isn = nat is not None and not any(
					tp['name'].get(b) == 'C' and resof.get(b, ri) != ri
					for b in nbr.get(nat, []))
				isc = cat is not None and not any(
					tp['name'].get(b) == 'N' and resof.get(b, ri) != ri
					for b in nbr.get(cat, []))
				keys = []
				if isn: keys.append('N' + tri)
				if isc: keys.append('C' + tri)
				keys.append(tri)
				aliases = None
				if isn and 'H' in anames and 'H1' not in anames:
					aliases = {'H': 'H1'}
				maptemplate(keys, ats, aliases, tp, ctx)
		nuc = {}
		for ni in sorted(nucs):
			nuc.setdefault(nucs[ni][1], []).append(ni)
		for nis in nuc.values():
			for pos, ni in enumerate(nis):
				tri = str(nucs[ni][4]).upper()
				keys = []
				if pos == 0: keys.append(tri + '5')
				if pos == len(nis) - 1: keys.append(tri + '3')
				keys.append(tri)
				keys.append(tri + 'N')
				maptemplate(keys, list(nucs[ni][2]) + list(nucs[ni][3]),
					None, tp, ctx)
	def tagparse(key):
		'''
		Split a leading <...> tag prefix off a force-field section key
		Arguments:
		----------
			key: str - a force-field section key
		Returns:
		--------
			tuple or None: ('map',) for <residue_templates>; ('at', [classes])
			for <at=...>; ('res', (tri, name)) for <res=><atom=>; None when
			the key carries no tag and is therefore a real SMIRKS
		'''
		if not key or key[0] != '<': return None
		if key.startswith('<residue_templates>'): return ('map',)
		if key.startswith('<at='):
			return ('at', key[4:key.index('>')].split(','))
		if key.startswith('<res='):
			tri = key[5:key.index('>')]
			rest = key[key.index('>') + 1:]
			return ('res', (tri, rest[rest.index('=') + 1:rest.index('>')]))
		return None
	def clsmatch(spec, idxs, cls):
		'''
		Test an atom-class spec (with '*' wildcards) against an atom tuple
		Arguments:
		----------
			spec: list of str  - class names; '*' matches any class
			idxs: tuple of int - candidate atom indices, same length
			cls:  dict - per-atom class lookup
		Returns:
		--------
			bool: True iff every position matches
		'''
		for s, a in zip(spec, idxs):
			if s != '*' and cls.get(a) != s: return False
		return True
	atoms = pose.data['Atoms']
	bonds = pose.data['Bonds']
	orders = pose.data.get('BondOrders', {}) or {}
	charges = getattr(pose, '_formal_charges', {}) or {}
	ids = sorted(atoms.keys())
	nbr = {i: [] for i in ids}
	for i in ids:
		for j in bonds.get(i, []):
			if j in atoms and j != i and j not in nbr[i]: nbr[i].append(j)
	edges = sorted({(min(i, j), max(i, j)) for i in ids for j in nbr[i]})
	bo = {}
	for i in ids:
		bos = orders.get(i, [])
		for k, j in enumerate(bonds.get(i, [])):
			if j not in atoms or j == i: continue
			bo[(min(i, j), max(i, j))] = float(
				bos[k] if k < len(bos) else 1.0)
	ctx = {'atoms': atoms, 'nbr': nbr, 'ids': ids, 'edges': edges,
		'edgeset': set(edges), 'bo': bo, 'parsed': {}, 'rcache': {},
		'Z': {i: Z_TABLE.get(atoms[i][1].capitalize(), 0) for i in ids},
		'X': {i: len(nbr[i]) for i in ids},
		'Hc': {i: sum(1 for j in nbr[i] if atoms[j][1] == 'H')
			for i in ids},
		'fc': {i: int(charges.get(i, 0)) for i in ids},
		'templates': params.get('Constraints', {}).get(
			'<residue_templates>')}
	rings = findrings(ctx)
	ctx['rings'] = rings
	kekulise(ctx, rings)
	aromatiserings(ctx, rings)
	ctx['aromb'] = {e: (abs(bo.get(e, 1.0) - 1.5) < 1e-6) for e in edges}
	ctx['aroma'] = {i: any(ctx['aromb'].get((min(i, j), max(i, j)), False)
		for j in nbr[i]) for i in ids}
	ctx['ringsz'] = {i: set() for i in ids}
	for r in rings:
		for a in r: ctx['ringsz'][a].add(len(r))
	ctx['minring'] = {i: (min(ctx['ringsz'][i]) if ctx['ringsz'][i] else 0)
		for i in ids}
	ctx['inringbond'] = set()
	for r in rings:
		for k in range(len(r)):
			a, b = r[k], r[(k + 1) % len(r)]
			ctx['inringbond'].add((min(a, b), max(a, b)))
	ctx['xcount'] = {i: sum(1 for j in nbr[i]
		if (min(i, j), max(i, j)) in ctx['inringbond']) for i in ids}
	out = {'bonds': {}, 'angles': {}, 'ub': {}, 'propers': {},
		'impropers': [], 'vdw': {}, 'vdw14': {}, 'polarisation': {},
		'charges': {i: None for i in ids},
		'constraints': set(), 'restri': {}}
	tp = {'name': {i: atoms[i][0] for i in ids},
		'elem': {i: atoms[i][1] for i in ids},
		'class': {i: None for i in ids},
		'charge': {i: None for i in ids},
		'reskey': {i: None for i in ids},
		'tname': {i: atoms[i][0] for i in ids}}
	cls = tp['class']
	rmin2sig = 2.0 / (2.0 ** (1.0 / 6.0))
	style = params.get('improper_style', 'smirnoff')
	tris, quads = [], []
	if ctx['templates'] is not None:
		assigntypes(pose, tp, ctx, out)
		for i in ids:
			if tp['charge'][i] is not None:
				out['charges'][i] = tp['charge'][i]
		for j in ids:
			ns = nbr[j]
			for x in range(len(ns)):
				for y in range(x + 1, len(ns)):
					tris.append((ns[x], j, ns[y]))
		for (j, k) in edges:
			for i in nbr[j]:
				for l in nbr[k]:
					if i == k or l == j or l == i: continue
					quads.append((i, j, k, l))
	for sm, par in params.get('Constraints', {}).items():
		if tagparse(sm) is not None: continue
		try: pat = getpat(sm, ctx)
		except Exception: continue
		for tup in match(pat, ctx):
			if len(tup) < 2: continue
			out['constraints'].add((min(tup[0], tup[1]),
				max(tup[0], tup[1])))
	for sm, par in params.get('Bonds', {}).items():
		tg = tagparse(sm)
		val = [par['r_0'], par['K_b']]
		if tg is None:
			try: pat = getpat(sm, ctx)
			except Exception: continue
			for tup in match(pat, ctx):
				if len(tup) != 2: continue
				i, j = sorted(tup)
				if (i, j) in ctx['edgeset']: out['bonds'][(i, j)] = list(val)
		elif tg[0] == 'at':
			for (a, b) in edges:
				if clsmatch(tg[1], (a, b), cls) or \
					clsmatch(tg[1], (b, a), cls):
					out['bonds'][(a, b)] = list(val)
	for sm, par in params.get('Angles', {}).items():
		tg = tagparse(sm)
		val = [par['theta_0'], par['K_theta']]
		if tg is None:
			try: pat = getpat(sm, ctx)
			except Exception: continue
			for tup in match(pat, ctx):
				if len(tup) != 3: continue
				i, j, k = tup
				if (min(i, j), max(i, j)) not in ctx['edgeset']: continue
				if (min(j, k), max(j, k)) not in ctx['edgeset']: continue
				out['angles'][(min(i, k), j, max(i, k))] = list(val)
		elif tg[0] == 'at':
			for (i, j, k) in tris:
				if clsmatch(tg[1], (i, j, k), cls) or \
					clsmatch(tg[1], (k, j, i), cls):
					out['angles'][(min(i, k), j, max(i, k))] = list(val)
	for sm, par in params.get('UB', {}).items():
		tg = tagparse(sm)
		val = [par.get('s_0', 0.0), par.get('K_ub', 0.0)]
		if tg is None:
			try: pat = getpat(sm, ctx)
			except Exception: continue
			for tup in match(pat, ctx):
				if len(tup) != 3: continue
				i, j, k = tup
				if (min(i, j), max(i, j)) not in ctx['edgeset']: continue
				if (min(j, k), max(j, k)) not in ctx['edgeset']: continue
				out['ub'][(min(i, k), j, max(i, k))] = list(val)
		elif tg[0] == 'at':
			for (i, j, k) in tris:
				if clsmatch(tg[1], (i, j, k), cls) or \
					clsmatch(tg[1], (k, j, i), cls):
					out['ub'][(min(i, k), j, max(i, k))] = list(val)
	best = {}
	for sm, par in params.get('ProperTorsions', {}).items():
		tg = tagparse(sm)
		comps = [[c['n'], c['phi_0'], c['K_phi'], c.get('idivf', 1.0)]
			for c in par['components']]
		if tg is None:
			try: pat = getpat(sm, ctx)
			except Exception: continue
			for tup in match(pat, ctx):
				if len(tup) != 4: continue
				i, j, k, l = tup
				if (min(i, j), max(i, j)) not in ctx['edgeset']: continue
				if (min(j, k), max(j, k)) not in ctx['edgeset']: continue
				if (min(k, l), max(k, l)) not in ctx['edgeset']: continue
				if (i, j, k, l) > (l, k, j, i): i, j, k, l = l, k, j, i
				out['propers'][(i, j, k, l)] = [list(c) for c in comps]
		elif tg[0] == 'at':
			score = sum(1 for s in tg[1] if s != '*')
			for (i, j, k, l) in quads:
				if not (clsmatch(tg[1], (i, j, k, l), cls) or
					clsmatch(tg[1], (l, k, j, i), cls)): continue
				canon = min((i, j, k, l), (l, k, j, i))
				prev = best.get(canon)
				if prev is None or score >= prev[0]:
					best[canon] = (score, comps)
	for canon in best:
		out['propers'][canon] = [list(c) for c in best[canon][1]]
	impbest = {}
	cpos = 1 if style == 'amber' else 0
	for sm, par in params.get('ImproperTorsions', {}).items():
		tg = tagparse(sm)
		if style not in ('amber', 'charmm'):
			if tg is not None: continue
			try: pat = getpat(sm, ctx)
			except Exception: continue
			for tup in match(pat, ctx):
				if len(tup) != 4: continue
				a1, a2, a3, a4 = tup
				ent = []
				for o1, o2, o3 in ((a1, a3, a4), (a3, a4, a1),
						(a4, a1, a3)):
					for c in par['components']:
						ent.append((a2, o1, o2, o3, c['n'], c['phi_0'],
							c['K_phi'] / 3.0))
				impbest[a2] = (0, ent)
			continue
		if tg is None or tg[0] != 'at' or len(tg[1]) != 4: continue
		cspec = tg[1][cpos]
		ospec = [tg[1][p] for p in range(4) if p != cpos]
		score = sum(1 for s in tg[1] if s != '*')
		for c in ids:
			if cspec != '*' and cls.get(c) != cspec: continue
			ns = nbr[c]
			if len(ns) < 3: continue
			prev = impbest.get(c)
			if prev is not None and prev[0] > score: continue
			trip = next((t for t in ((ns[x], ns[y], ns[z])
				for x in range(len(ns)) for y in range(len(ns))
				for z in range(len(ns)) if y != x and z != x and z != y)
				if clsmatch(ospec, t, cls)), None)
			if trip is None: continue
			ent = []
			for cc in par['components']:
				if style == 'amber':
					ent.append((trip[0], trip[1], c, trip[2], cc['n'],
						cc['phi_0'], cc['K_phi']))
				else:
					ent.append((c, trip[0], trip[1], trip[2], cc['n'],
						cc['phi_0'], cc['K_phi']))
			impbest[c] = (score, ent)
	for c in impbest:
		out['impropers'].extend(impbest[c][1])
	for sm, par in params.get('vdW', {}).items():
		tg = tagparse(sm)
		if tg is not None and tg[0] != 'at': continue
		eps = par['epsilon']
		sig = par['sigma'] if 'sigma' in par else par['r'] * rmin2sig
		has14 = 'epsilon14' in par or 'sigma14' in par
		sig14 = sig
		if 'sigma14' in par: sig14 = par['sigma14']
		elif 'r14' in par: sig14 = par['r14'] * rmin2sig
		if tg is None:
			try: pat = getpat(sm, ctx)
			except Exception: continue
			hits = [t[0] for t in match(pat, ctx) if len(t) == 1]
		else:
			hits = [i for i in ids
				if tg[1][0] == '*' or cls.get(i) == tg[1][0]]
		for i in hits:
			out['vdw'][i] = [eps, sig]
			out['polarisation'][i] = par.get('alpha', 0.0)
			if has14: out['vdw14'][i] = [par.get('epsilon14', eps), sig14]
	for sm, par in params.get('LibraryCharges', {}).items():
		tg = tagparse(sm)
		qs = par.get('q', [])
		if tg is None:
			try: pat = getpat(sm, ctx)
			except Exception: continue
			for tup in match(pat, ctx):
				for k, idx in enumerate(tup):
					if k < len(qs): out['charges'][idx] = float(qs[k])
		elif tg[0] == 'res' and qs:
			for i in ids:
				if tp['reskey'].get(i) == tg[1][0] and \
					tp['tname'].get(i) == tg[1][1]:
					out['charges'][i] = float(qs[0])
	return out

def ScoreMatch(pose, params, ligand=None, xs_override=None, nrot_override=None):
	'''
	Build the per-pose support cache used by every Score energy term
	Arguments:
	----------
		pose:          Pose or Molecule - receptor / source structure
		params:        dict - the active ['Score Parameters'][NAME] block
		ligand:        Molecule or None - optional small-molecule ligand
		xs_override:   dict or None - validation hook; maps combined index to XS
			atom type
		nrot_override: int or None - validation hook for ligand n_rot
	Returns:
	--------
		dict: 'hash', 'coords', 'atom_types', 'inter_pairs', 'intra_pairs'
			plus per-term raw value keys (e.g. 'FaAtrPotential') and
			callable nested helpers (e.g. 'evalpairs', 'fullatomhbond')
	'''
	_FADUN_GRID_CACHE = {}
	_FADUN_ENT_CACHE = {}
	_RAMA_SPLINE_CACHE = {}
	_PCS_M = {}
	_FROZEN_ADJ = {}
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
			out = fullatomcache(pose, params)
			n_atoms = int(len(out['coords']))
			out.setdefault('inter_pairs', np.empty((0, 2), dtype=np.int64))
			out.setdefault('intra_ligand_pairs',
				np.empty((0, 2), dtype=np.int64))
			out.setdefault('nrot', 0)
			out.setdefault('xs_types', np.full(n_atoms, -1, dtype=np.int64))
			out.setdefault('xs_radii_arr', np.zeros(0))
			out.setdefault('xs_is_hydrophobic_arr', np.zeros(0, dtype=bool))
			out.setdefault('xs_is_donor_arr', np.zeros(0, dtype=bool))
			out.setdefault('xs_is_acceptor_arr', np.zeros(0, dtype=bool))
			return out
		if 'XS_atom_types' in params:
			return patternsearchsmall(pose, params, ligand,
				xs_override, nrot_override)
		raise Exception(
			'PatternSearch: unsupported params (no recognised typing system)')
	def atomel(gi, r_atoms, l_atoms, n_r):
		'''
		Element symbol of the atom at combined index gi
		Arguments:
		----------
			gi: int - combined receptor/ligand atom index
			r_atoms: dict - receptor atom records
			l_atoms: dict - ligand atom records, empty when no ligand
			n_r: int - receptor atom count
		Returns:
		--------
			str: element symbol
		'''
		rec = r_atoms[gi] if gi < n_r else l_atoms[gi - n_r]
		return rec[1]
	def atomnbrs(gi, r_bonds, l_bonds, n_r):
		'''
		Combined-index neighbour list of the atom at gi
		Arguments:
		----------
			gi: int - combined receptor/ligand atom index
			r_bonds: dict - receptor bond graph
			l_bonds: dict - ligand bond graph, empty when no ligand
			n_r: int - receptor atom count
		Returns:
		--------
			list of int: bonded neighbours in combined indexing
		'''
		if gi < n_r: return list(r_bonds.get(gi, []))
		return [n_r + j for j in l_bonds.get(gi - n_r, [])]
	def protein_xs(gi, r_atoms, r_atom_to_tri, PROT_XS):
		'''
		XS atom type override for a receptor protein atom
		Arguments:
		----------
			gi: int - combined receptor/ligand atom index
			r_atoms: dict - receptor atom records
			r_atom_to_tri: dict - receptor atom index to residue tricode
			PROT_XS: dict - tricode and atom name to XS type override
		Returns:
		--------
			str or None: XS atom-type code (e.g. N_D, O_DA) or None if no
				override
		'''
		tri = r_atom_to_tri.get(gi)
		if tri is None: return None
		nm = r_atoms[gi][0] if r_atoms.get(gi) else None
		if nm is None: return None
		return PROT_XS.get((tri, nm))
	def has_polar_h(gi, r_atoms, l_atoms, r_bonds,
			l_bonds, coords, H_coords_r, n_r):
		'''
		True iff atom gi has a bonded H, or a nearby receptor H
		Arguments:
		----------
			gi: int - combined receptor/ligand atom index
			r_atoms: dict - receptor atom records
			l_atoms: dict - ligand atom records, empty when no ligand
			r_bonds: dict - receptor bond graph
			l_bonds: dict - ligand bond graph, empty when no ligand
			coords: np.ndarray - combined (N, 3) coordinates
			H_coords_r: np.ndarray - receptor hydrogen coordinates
			n_r: int - receptor atom count
		Returns:
		--------
			bool: True if at least one polar H is attached
		'''
		for j in atomnbrs(gi, r_bonds, l_bonds, n_r):
			if atomel(j, r_atoms, l_atoms, n_r) == 'H': return True
		if gi < n_r and len(H_coords_r):
			d = np.linalg.norm(H_coords_r - coords[gi], axis=1)
			if (d < 1.3).any(): return True
		return False
	def patternsearchsmall(pose, params, ligand, xs_override, nrot_override):
		'''
		XS atom typing and pair lists for the docking score
		Arguments:
		----------
			pose:          Pose or Molecule - receptor
			params:        dict - the small-molecule docking param block
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
			if atomel(i, r_atoms, l_atoms, n_r) == 'H']
		H_coords_r = (np.asarray(H_coords_r, dtype=np.float64)
			if H_coords_r else np.zeros((0, 3)))
		if xs_override is not None:
			for gi, nm in xs_override.items():
				if nm in name_to_idx:
					xs[int(gi)] = name_to_idx[nm]
		else:
			for gi in range(n):
				el = atomel(gi, r_atoms, l_atoms, n_r)
				if el == 'H': continue
				if el == 'C':
					cp = any(atomel(j, r_atoms, l_atoms, n_r) not in ('C', 'H')
						for j in atomnbrs(gi, r_bonds, l_bonds, n_r))
					xs[gi] = name_to_idx['C_P' if cp else 'C_H']
				elif el == 'N':
					ov = (protein_xs(gi, r_atoms, r_atom_to_tri, PROT_XS)
						if gi < n_r else None)
					if ov in ('N_D', 'N_A', 'N_DA'):
						xs[gi] = name_to_idx[ov]
					else:
						has_h = has_polar_h(gi, r_atoms, l_atoms,
					r_bonds, l_bonds, coords, H_coords_r, n_r)
						xs[gi] = name_to_idx['N_D' if has_h else 'N_A']
				elif el == 'O':
					ov = (protein_xs(gi, r_atoms, r_atom_to_tri, PROT_XS)
						if gi < n_r else None)
					if ov in ('O_A', 'O_D', 'O_DA'):
						xs[gi] = name_to_idx[ov]
					else:
						has_h = has_polar_h(gi, r_atoms, l_atoms,
					r_bonds, l_bonds, coords, H_coords_r, n_r)
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
					dij = np.linalg.norm( coords[n_r + i] - coords[n_r + j])
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
	def bfsdists(start, adj, max_depth=4):
		'''
		BFS bond-distance map from `start` up to max_depth bonds
		Arguments:
		----------
			start: int - atom index to start BFS from
			adj: dict - atom index to set of bonded neighbours
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
	def get_bfs(atom_idx, bfs_cache, adj):
		'''
		Memoised wrapper around bfsdists keyed on atom_idx
		Arguments:
		----------
			atom_idx: int - atom index to BFS from
			bfs_cache: dict - memo of atom index to BFS distance map
			adj: dict - atom index to set of bonded neighbours
		Returns:
		--------
			dict: {atom_index: bond_distance} (same shape as bfsdists)
		'''
		if atom_idx not in bfs_cache:
			bfs_cache[atom_idx] = bfsdists(int(atom_idx), adj, max_depth=4)
		return bfs_cache[atom_idx]
	def lookuptype(tricode, atom_name, atoms, bonds, n,
			atom_types_db, residue_types_db, D_TO_L, N_TERM_H,
			ai=None):
		'''
		Resolve (tricode, atom_name) to a Rosetta atom type and charge
		Arguments:
		----------
			tricode: str - 3-letter residue code (D-codes map to L)
			atom_name: str - built atom name
			ai: int - atom index for the bonded-neighbour fallback
			atoms: dict - atom index to atom record
			bonds: dict - atom index to neighbour list
			n: int - atom count
			atom_types_db: dict - Rosetta atom-type table
			residue_types_db: dict - Rosetta residue-type table
			D_TO_L: dict - D-amino-acid tricode to L tricode
			N_TERM_H: set - N-terminal hydrogen atom names
		Returns:
		--------
			tuple: (atom_type_or_None, partial_charge_float)
		'''
		res = residue_types_db.get(D_TO_L.get(tricode, tricode))
		if res is None: return None, 0.0
		table = res['atoms']
		aliases = res.get('aliases', {}) or {}
		cands = [atom_name, aliases.get(atom_name)]
		if atom_name and atom_name[0].isdigit():
			cands.append(atom_name[1:] + atom_name[0])
		elif atom_name and atom_name[-1].isdigit() and len(atom_name) > 1:
			cands.append(atom_name[-1] + atom_name[:-1])
		cands.append(N_TERM_H.get(atom_name))
		cands.append('1' + atom_name)
		cands.append(atom_name + '1')
		for c in cands:
			e = table.get(c) if c else None
			if e is not None:
				return e['type'], float(e.get('charge', 0.0))
		if ai is None or not (0 <= ai < n): return None, 0.0
		parent = next((atoms[b][0] for b in bonds.get(ai, ())
			if atoms[b][1] != 'H'), None)
		ptbl = parent if parent in table else (
			parent + '1' if parent and parent + '1' in table else None)
		if ptbl is None: return None, 0.0
		tadj = []
		for bd in res.get('bonds', ()):
			if bd[0] == ptbl: tadj.append(bd[1])
			elif bd[1] == ptbl: tadj.append(bd[0])
		elem = atoms[ai][1]
		cand = [t for t in tadj if t in table and atom_types_db.get(
			table[t]['type'], {}).get('element') == elem]
		if not cand: return None, 0.0
		d = atom_name[0] if atom_name and atom_name[0].isdigit() else None
		pick = next((m for m in cand if d and m.startswith(d)), sorted(cand)[0])
		e = table[pick]
		return e['type'], float(e.get('charge', 0.0))
	def applyatom(ai, new_type, new_charge, ros_types, q_arr, ljR, ljW,
			lkdG, lkLam, lkVol, is_donor, is_accep,
			is_polar_h, is_H, has_score, atom_types_db):
		'''
		Patch one atom: overwrite type, charge and derived arrays
		Arguments:
		----------
			ai: int - atom index
			new_type: str - new Rosetta atom type
			new_charge: float - new partial charge
			ros_types: list - per-atom Rosetta type names
			q_arr: np.ndarray - per-atom partial charges
			ljR: np.ndarray - per-atom LJ radius
			ljW: np.ndarray - per-atom LJ well depth
			lkdG: np.ndarray - per-atom LK free energy
			lkLam: np.ndarray - per-atom LK lambda
			lkVol: np.ndarray - per-atom LK volume
			is_donor: np.ndarray - bool, atom is an h-bond donor
			is_accep: np.ndarray - bool, atom is an h-bond acceptor
			is_polar_h: np.ndarray - bool, atom is a polar hydrogen
			is_H: np.ndarray - bool, atom is a hydrogen
			has_score: np.ndarray - bool, atom has scoring parameters
			atom_types_db: dict - Rosetta atom-type table
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
			is_H[ai] = info.get('element') in ('H',) or new_type == 'HOH'
			has_score[ai] = True
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
	def patchtermini(aas, atom_types_db, atoms, coords, has_score, is_H,
			is_accep, is_donor, is_polar_h, ljR, ljW, lkLam, lkVol, lkdG, n,
			q_arr, ros_types):
		'''
		Retype and recharge terminal residues and disulfide cysteines
		Arguments:
		----------
			aas: dict - residue index to amino-acid record
			atom_types_db: dict - Rosetta atom-type table
			atoms: dict - atom index to atom record
			coords: np.ndarray - (N, 3) coordinates
			has_score: np.ndarray - scored flags, edited in place
			is_H: np.ndarray - hydrogen flags, edited in place
			is_accep: np.ndarray - acceptor flags, edited in place
			is_donor: np.ndarray - donor flags, edited in place
			is_polar_h: np.ndarray - polar-H flags, edited in place
			ljR: np.ndarray - per-atom LJ radius, edited in place
			ljW: np.ndarray - per-atom LJ well depth, edited in place
			lkLam: np.ndarray - per-atom LK lambda, edited in place
			lkVol: np.ndarray - per-atom LK volume, edited in place
			lkdG: np.ndarray - per-atom LK free energy, edited in place
			n: int - atom count
			q_arr: np.ndarray - per-atom charges, edited in place
			ros_types: list - per-atom type names, edited in place
		Returns:
		--------
			set: residue indices found to be at an N-terminus
		'''
		by_chain = {}
		for ri, info in aas.items():
			ch = info[1] if len(info) > 1 else ''
			by_chain.setdefault(ch, []).append(int(ri))
		bonds_raw = pose.data.get('Bonds', {}) or {}
		res_of = {}
		for _ri, _info in aas.items():
			for _ai in list(_info[2]) + list(_info[3]):
				res_of[int(_ai)] = int(_ri)
		n_term_res = set()
		c_term_res = set()
		for _ri, _info in aas.items():
			_ri = int(_ri)
			_n = _c = None
			for _ai in list(_info[2]) + list(_info[3]):
				_nm = atoms[int(_ai)][0]
				if _nm == 'N': _n = int(_ai)
				elif _nm == 'C': _c = int(_ai)
			if _n is not None and not any(
					atoms[int(_j)][0] == 'C'
					and res_of.get(int(_j), _ri) != _ri
					for _j in bonds_raw.get(_n, [])):
				n_term_res.add(_ri)
			if _c is not None and not any(
					atoms[int(_j)][0] == 'N'
					and res_of.get(int(_j), _ri) != _ri
					for _j in bonds_raw.get(_c, [])):
				c_term_res.add(_ri)
		NTERM_H_NAMES = {'H', 'H1', 'H2', 'H3', '1H', '2H', '3H',
			'HN', 'HT1', 'HT2', 'HT3'}
		TQ = params.get('TerminalCharges')
		if TQ is None:
			raise ValueError(
				"ScoreMatch: ['TerminalCharges'] is missing from the "
				"score parameters. Run tools.Port('ref15') to install it.")
		for ri in n_term_res:
			info = aas.get(ri)
			if info is None: continue
			tri = info[5] if len(info) >= 6 else None
			if tri == 'PRO':
				for ai in info[2] + info[3]:
					ai = int(ai)
					nm = atoms[ai][0]
					if nm in ('1H', '2H', 'H1', 'H2', 'HN', 'HT1', 'HT2'):
						applyatom(ai, 'Hpol', TQ['PRO']['H'], ros_types, q_arr,
							ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
							is_polar_h, is_H, has_score, atom_types_db)
					elif nm == 'N':
						applyatom(ai, 'Nlys', TQ['PRO']['N'], ros_types, q_arr,
							ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
							is_polar_h, is_H, has_score, atom_types_db)
				continue
			if tri == 'GLY':
				for ai in info[2] + info[3]:
					ai = int(ai)
					nm = atoms[ai][0]
					if nm == 'N':
						applyatom(ai, 'Nlys', TQ['GLY']['N'], ros_types, q_arr,
							ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
							is_polar_h, is_H, has_score, atom_types_db)
					elif nm in NTERM_H_NAMES:
						applyatom(ai, 'Hpol', TQ['GLY']['H'], ros_types, q_arr,
							ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
							is_polar_h, is_H, has_score, atom_types_db)
					elif nm == 'CA':
						q_arr[ai] = TQ['GLY']['CA']
					elif nm in ('HA', '1HA', '2HA', 'HA1', 'HA2', 'HA3'):
						q_arr[ai] = TQ['GLY']['HA']
				continue
			for ai in info[2] + info[3]:
				ai = int(ai)
				nm = atoms[ai][0]
				if nm == 'N':
					applyatom(ai, 'Nlys', TQ['generic']['N'], ros_types, q_arr,
						ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
						is_polar_h, is_H, has_score, atom_types_db)
				elif nm in NTERM_H_NAMES:
					applyatom(ai, 'Hpol', TQ['generic']['H'], ros_types, q_arr,
						ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
						is_polar_h, is_H, has_score, atom_types_db)
				elif nm == 'CA':
					q_arr[ai] = TQ['generic']['CA']
				elif nm == 'HA':
					q_arr[ai] = TQ['generic']['HA']
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
				applyatom(c_ai, 'COO', TQ['cterm']['C'], ros_types, q_arr, ljR,
					ljW, lkdG, lkLam, lkVol, is_donor, is_accep, is_polar_h,
					is_H, has_score, atom_types_db)
			if o_ai is not None:
				applyatom(o_ai, 'OOC', TQ['cterm']['O'], ros_types, q_arr, ljR,
					ljW, lkdG, lkLam, lkVol, is_donor, is_accep, is_polar_h,
					is_H, has_score, atom_types_db)
			if oxt_ai is not None:
				applyatom(oxt_ai, 'OOC', TQ['cterm']['O'], ros_types, q_arr,
					ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
					is_polar_h, is_H, has_score, atom_types_db)
		sg_idx = [i for i in range(n)
			if atoms[i][1] == 'S' and ros_types[i] == 'SH1']
		X_arr = np.asarray(coords, dtype=np.float64)
		for ii in sg_idx:
			for jj in sg_idx:
				if ii >= jj: continue
				d = float( np.linalg.norm(X_arr[ii] - X_arr[jj]))
				if d < 2.5:
					q_arr[ii] = TQ['disulfide_SG']
					q_arr[jj] = TQ['disulfide_SG']
		return n_term_res
	def atomwaters(i, X, water_atom, water_cnt, water_off, water_xyz, LKB_WTS,
			adj, ang_sp2, ang_sp3, atom_res, atom_types_db, dih_sp2, dih_sp3,
			is_H, is_polar_h, opt_dist, ros_types):
		'''
		Append the LkBall virtual waters belonging to one atom
		Arguments:
		----------
			i: int - atom whose virtual waters are built
			X: np.ndarray - (N, 3) coordinates
			water_atom: list - owning atom per water, appended
			water_cnt: np.ndarray - per-atom water count, filled
			water_off: np.ndarray - per-atom water offset, filled
			water_xyz: list - water positions, appended in place
			LKB_WTS: dict - LkBall weights keyed by Rosetta type
			adj: dict - atom index to set of bonded neighbours
			ang_sp2: float - sp2 water angle in radians
			ang_sp3: float - sp3 water angle in radians
			atom_res: np.ndarray - per-atom residue index
			atom_types_db: dict - Rosetta atom-type table
			dih_sp2: list - sp2 water dihedrals in radians
			dih_sp3: list - sp3 water dihedrals in radians
			is_H: np.ndarray - bool, atom is a hydrogen
			is_polar_h: np.ndarray - bool, atom is a polar hydrogen
			opt_dist: float - ideal atom-to-water distance
			ros_types: list - per-atom Rosetta type names
		Returns:
		--------
			No return value, the water_* containers are filled in place
		'''
		t = ros_types[i]
		if t not in LKB_WTS: return
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
			for h in polar_h_nbrs:
				w = i_xyz + opt_dist * unit(X[h] - i_xyz)
				i_waters.append(w)
		if is_a:
			if is_ring and len(heavy_nbrs) >= 2:
				c1, c2 = heavy_nbrs[0], heavy_nbrs[1]
				mid = 0.5 * (X[c1] + X[c2])
				w = i_xyz + opt_dist * unit(i_xyz - mid)
				i_waters.append(w)
			elif is_sp3 and len(heavy_nbrs) >= 1 and len(polar_h_nbrs) >= 1:
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
				if not c_heavy_nbrs: return
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
		if not i_waters: return
		water_off[i] = len(water_xyz)
		water_cnt[i] = len(i_waters)
		for w in i_waters:
			water_xyz.append(np.asarray(w, dtype=np.float64))
			water_atom.append(i)
	def fullatomcache(pose, params):
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
		cp_half = float((params.get('CountPair') or {})['half'])
		atoms = pose.data['Atoms']
		bonds = pose.data['Bonds']
		coords = np.array(pose.data['Coordinates'], dtype=np.float64)
		aas = pose.data.get('Amino Acids') or {}
		n = len(atoms)
		atom_types_db = params['Atom_types']
		residue_types_db = params['Residue_types']
		N_TERM_H = {'H1':'H','H2':'H','H3':'H','1H':'H','2H':'H','3H':'H',
			'HN':'H','HT1':'H','HT2':'H','HT3':'H'}
		D_TO_L = {'DAL':'ALA','DAR':'ARG','DAS':'ASP','DSG':'ASN','DCY':'CYS',
			'DGN':'GLN','DGL':'GLU','DHI':'HIS','DIL':'ILE','DLE':'LEU',
			'DLY':'LYS','MED':'MET','DPN':'PHE','DPR':'PRO','DSN':'SER',
			'DTH':'THR','DTR':'TRP','DTY':'TYR','DVA':'VAL',
			# DSE is D-MSE (selenomethionine), NOT D-serine; that is DSN.
			'DRN':'ORN','DSE':'MSE','DPO':'TPO','DEC':'SEC',
			'DF6':'FT6','DPT':'PTR'}
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
				t, q = lookuptype(tri, nm, atoms, bonds, n,
						atom_types_db, residue_types_db, D_TO_L,
						N_TERM_H, ai=ai)
				ros_types[ai] = t
				q_arr[ai] = q
		ljR = np.zeros(n); ljW = np.zeros(n)
		lkdG = np.zeros(n)
		lkLam = np.ones(n) * float(
			(params.get('LkBall') or {})['lk_lambda_default'])
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
			n_term_res = patchtermini(aas, atom_types_db, atoms, coords,
				has_score, is_H, is_accep, is_donor, is_polar_h, ljR, ljW,
				lkLam, lkVol, lkdG, n, q_arr, ros_types)
		adj = {int(k): set(int(j) for j in v) for k, v in bonds.items()}
		for i in range(n):
			adj.setdefault(i, set())
		_sig = hash((n, tuple(sorted(
			(int(k), tuple(sorted(int(j) for j in v)))
			for k, v in bonds.items()))))
		_frozen = _FROZEN_ADJ.get(_sig)
		if _frozen is not None:
			for i in range(n):
				adj[i] = set(_frozen[i])
		else:
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
			_FROZEN_ADJ[_sig] = {i: set(adj[i]) for i in range(n)}
		rep_atom_idx = np.arange(n, dtype=np.int64)
		for ri, info in aas.items():
			is_nterm = ri in n_term_res
			for ai in info[2] + info[3]:
				ci = int(ai)
				if not (0 <= ci < n): continue
				elem = atoms[ci][1]
				if is_nterm and atoms[ci][0] == 'N': continue
				if elem == 'C':
					rep = min((b for b in adj.get(ci, ()) if
						atoms[b][1] == 'O' and not any(atoms[h][1] == 'H'
						for h in adj.get(b, ()))), default=None)
				elif elem == 'N' or elem == 'O':
					rep = min((b for b in adj.get(ci, ())
						if atoms[b][1] == 'H'), default=None)
				else:
					continue
				if rep is not None: rep_atom_idx[ci] = rep
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
		c0 = float(params['Constants']['fa_max_dis'])
		pairs_i = []; pairs_j = []; pair_d = []
		pair_w = []; pair_same_res = []; pair_path = []
		pair_cp_path = []; pair_is_poly = []
		bfs_cache = {}
		typed_idx = np.where(has_score)[0]
		X = coords
		for ii in typed_idx:
			dists_from_ii = get_bfs(int(ii), bfs_cache, adj)
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
					elif bd == 4: w = cp_half
					else: w = 1.0
				else:
					if bd <= 2: w = 0.0
					elif bd == 3: w = cp_half
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
					rep_bfs = get_bfs(rep_i, bfs_cache, adj)
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
		LKB_WTS = {k: tuple(v) for k, v in
			(params.get('LkBallWtd', {}).get('atom_weights') or {}).items()}
		if not LKB_WTS and any(
				float(params.get(s, {}).get('weight', 0.0) or 0.0) != 0.0
				for s in ('LkBallWtd', 'LkBallIso', 'LkBallBridge')):
			raise ValueError(
				'ScoreMatch: lk_ball is weighted but '
				"['LkBallWtd']['atom_weights'] is missing from the score "
				"parameters. Run tools.Port('ref15') to install it.")
		LKB = params.get('LkBall') or {}
		LK_RAMP_W2 = float(LKB['ramp_w2'])
		H2O_R = float(LKB['h2o_radius'])
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
		etb = params.get('EtablePairParams')
		if etb is None:
			raise ValueError(
				"ScoreMatch: ['EtablePairParams'] is missing from the score "
				"parameters. Run tools.Port('ref15') to install it.")
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
			et_lambda_self = np.ones((NT, NT), dtype=np.float64) * float(
				(params.get('LkBall') or {})['lk_lambda_default'])
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
			for _k in range(NT * NT):
				is_, io_ = divmod(_k, NT)
				c = et_pairs[_k]
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
					et_lj_switch_intercept[is_, io_] = (
						c['lj_switch_intercept'])
					et_lj_switch_slope[is_, io_]     = c['lj_switch_slope']
					et_lj_val_at_minimum[is_, io_] = ( c['lj_val_at_minimum'])
					et_ljatr_cubic_poly[is_, io_]    = c['ljatr_cubic_poly']
					et_ljatr_cp_xhi[is_, io_] = ( c['ljatr_cubic_poly_xhi'])
					et_ljatr_cp_xlo[is_, io_] = ( c['ljatr_cubic_poly_xlo'])
					et_ljatr_final_weight[is_, io_] = ( c['ljatr_final_weight'])
					et_ljrep_linear_ramp_d2[is_, io_] = (
						c['ljrep_linear_ramp_d2_cutoff'])
					et_ljrep_from_negcrossing[is_, io_] = (
						c['ljrep_from_negcrossing'])
					et_hydrogen_interaction[is_, io_] = (
						c['hydrogen_interaction'])
					et_ljrep_xr_xlo[is_, io_]   = c['ljrep_xr_xlo']
					et_ljrep_xr_xhi[is_, io_]   = c['ljrep_xr_xhi']
					et_ljrep_xr_slope[is_, io_] = c['ljrep_xr_slope']
					et_ljrep_xr_extrap_slope[is_, io_] = (
						c['ljrep_xr_extrapolated_slope'])
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
		opt_dist = float(LKB['opt_dist'])
		ang_sp2 = math.radians(60.0)
		ang_sp3 = math.radians(71.0)
		dih_sp2 = (0.0, math.radians(180.0))
		dih_sp3 = (math.radians(120.0), math.radians(240.0))
		def place_waters(X):
			'''
			Build LkBall virtual-water positions for coordinates X from the
			cached per-atom topology (donor/acceptor types, bond graph),
			so waters can be refreshed on a cached re-score
			Arguments:
			----------
				X: ndarray (n, 3) - current coordinates
			Returns:
			--------
				tuple: (water_xyz_arr, water_off, water_cnt) - stacked water
					coordinates, per-atom offset into them, per-atom count
			'''
			water_xyz = []
			water_atom = []
			water_off = np.full(n, -1, dtype=np.int64)
			water_cnt = np.zeros(n, dtype=np.int64)
			for i in range(n):
				atomwaters(i, X, water_atom, water_cnt, water_off, water_xyz,
					LKB_WTS, adj, ang_sp2, ang_sp3, atom_res, atom_types_db,
					dih_sp2, dih_sp3, is_H, is_polar_h, opt_dist, ros_types)
			if water_xyz:
				water_xyz_arr = np.stack(water_xyz, axis=0)
			else:
				water_xyz_arr = np.empty((0, 3), dtype=np.float64)
			water_atom = np.array(water_atom, dtype=np.int64)
			return water_xyz_arr, water_off, water_cnt
		water_xyz_arr, water_off, water_cnt = place_waters(X)
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
			'place_waters': place_waters,
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
	def bondorder(a, b, bonds, orders):
		'''
		Bond order between atoms a and b in the bond-order table
		Arguments:
		----------
			a: int - atom index
			b: int - atom index
			bonds: dict - atom index to neighbour list
			orders: dict - atom index to bond-order list
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
	def inring(a, b, bonds):
		'''
		True iff the edge (a, b) lies in a ring of the topology
		Arguments:
		----------
			a: int - atom index
			b: int - atom index
			bonds: dict - atom index to neighbour list
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
	def amide(c_idx, n_idx, atoms, bonds, orders):
		'''
		True iff atom c_idx is an amide carbonyl C bonded to n_idx
		Arguments:
		----------
			c_idx: int - candidate carbonyl-carbon atom index
			n_idx: int - candidate nitrogen atom index
			atoms: dict - atom index to atom record
			bonds: dict - atom index to neighbour list
			orders: dict - atom index to bond-order list
		Returns:
		--------
			bool: True if c_idx is an amide carbonyl carbon
		'''
		if atoms[c_idx][1] != 'C': return False
		if atoms[n_idx][1] != 'N': return False
		for k in bonds.get(c_idx, []):
			if inring(c_idx, k, bonds): return False
		for k in bonds.get(c_idx, []):
			if k == n_idx: continue
			if atoms[k][1] not in ('O', 'N'): continue
			if bondorder(c_idx, k, bonds, orders) >= 2: return True
		return False
	def isamide(a, b, atoms, bonds, orders):
		'''
		True iff the bond (a, b) is the C-N of an amide
		Arguments:
		----------
			a: int - atom index
			b: int - atom index
			atoms: dict - atom index to atom record
			bonds: dict - atom index to neighbour list
			orders: dict - atom index to bond-order list
		Returns:
		--------
			bool: True for amide C-N bonds
		'''
		return (amide(a, b, atoms, bonds, orders)
			or amide(b, a, atoms, bonds, orders))
	def countnrot(ligand):
		'''
		Count non-terminal, non-ring, non-amide single bonds in a ligand
		(n_rot for the rotational-entropy penalty term).
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
		nrot = 0
		seen = set()
		for i in sorted(bonds):
			for j in bonds[i]:
				if j <= i: continue
				key = (i, j)
				if key in seen: continue
				seen.add(key)
				if atoms[i][1] == 'H' or atoms[j][1] == 'H': continue
				if bondorder(i, j, bonds, orders) != 1: continue
				if inring(i, j, bonds): continue
				hvi = [k for k in bonds[i] if atoms[k][1] != 'H' and k != j]
				hvj = [k for k in bonds[j] if atoms[k][1] != 'H' and k != i]
				has_h_i = any(atoms[k][1] == 'H' for k in bonds[i] if k != j)
				has_h_j = any(atoms[k][1] == 'H' for k in bonds[j] if k != i)
				if not hvi and (atoms[i][1] == 'C' or not has_h_i):
					continue
				if not hvj and (atoms[j][1] == 'C' or not has_h_j):
					continue
				if isamide(i, j, atoms, bonds, orders): continue
				nrot += 1
		return nrot
	def countnumtors(ligand):
		'''
		Compute the rotational-entropy penalty term's "num_tors" input:
		sum over rotatable bonds of 0.5 from each side, where each side
		contributes 0.5 only if it has > 1 heavy non-H neighbour. So a
		regular rotation between two heavy-substituted carbons contributes
		1.0, while a rotation where one side has only one heavy neighbour
		(like a carboxyl C-OH) contributes 0.5. This is the actual quantity
		used in the rotational-entropy penalty denominator `1 + 0.05846 *
			num_tors`.
		Arguments:
		----------
			ligand: Molecule
		Returns:
		--------
			float: rotational-entropy num_tors (typically equal to Nrot or
				Nrot-k/2)
		'''
		atoms = ligand.data['Atoms']
		bonds = ligand.data['Bonds']
		orders = ligand.data.get('BondOrders', {})
		num_tors = 0.0
		seen = set()
		for i in sorted(bonds):
			for j in bonds[i]:
				if j <= i: continue
				if (i, j) in seen: continue
				seen.add((i, j))
				if atoms[i][1] == 'H' or atoms[j][1] == 'H': continue
				if bondorder(i, j, bonds, orders) != 1: continue
				if inring(i, j, bonds): continue
				hvi = [k for k in bonds[i] if atoms[k][1] != 'H' and k != j]
				hvj = [k for k in bonds[j] if atoms[k][1] != 'H' and k != i]
				has_h_i = any(atoms[k][1] == 'H' for k in bonds[i] if k != j)
				has_h_j = any(atoms[k][1] == 'H' for k in bonds[j] if k != i)
				if not hvi and (atoms[i][1] == 'C' or not has_h_i):
					continue
				if not hvj and (atoms[j][1] == 'C' or not has_h_j):
					continue
				if isamide(i, j, atoms, bonds, orders): continue
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
		return hash((params.get('_name',''), keyof(pose), keyof(ligand)))
	def go(pairs, cache, pair_fn):
		'''
		Sum a per-pair function over one precomputed pair list
		Arguments:
		----------
			pairs: np.ndarray - (M, 2) array of atom index pairs
			cache: dict - PatternSearch result
			pair_fn: callable - per-pair contribution function
		Returns:
		--------
			float: the summed contribution, 0.0 for an empty pair list
		'''
		if len(pairs) == 0: return 0.0
		ai = pairs[:, 0]; aj = pairs[:, 1]
		coords = cache['coords']
		rij = np.linalg.norm(coords[ai] - coords[aj], axis=1)
		return float(pair_fn(ai, aj, rij, cache).sum())
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
		inter_sum = go(cache['inter_pairs'], cache, pair_fn)
		intra_sum = go(cache['intra_ligand_pairs'], cache, pair_fn)
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
		Evaluate one small-molecule Gaussian pair term and pack it
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
			Per-pair kernel mapping (ai, aj, rij, c) to a contribution
			Arguments:
			----------
				ai: np.ndarray - per-pair first-atom indices
				aj: np.ndarray - per-pair second-atom indices
				rij: np.ndarray - per-pair distance
				c: dict - the ScoreMatch cache, unused by this kernel
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
	def sloperamp(x, bad, good):
		'''
		Linear ramp: 1 below good, 0 above bad, linear in between
		Arguments:
		----------
			x: np.ndarray - surface-distance values to ramp
			bad: float - distance at which the ramp reaches 0
			good: float - distance at or below which the ramp is 1
		Returns:
		--------
			np.ndarray: ramp value in [0, 1] for each element of x
		'''
		if bad < good:
			return np.clip((x - bad) / (good - bad), 0.0, 1.0)
		elif bad > good:
			return np.clip((bad - x) / (bad - good), 0.0, 1.0)
		else:
			return np.where(x <= good, 1.0, 0.0)
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
		def fn(ai, aj, rij, c):
			'''
			Per-pair slope-step kernel used by the slopestep wrapper
			Arguments:
			----------
				ai: np.ndarray - per-pair first-atom indices
				aj: np.ndarray - per-pair second-atom indices
				rij: np.ndarray - per-pair distance
				c: dict - the ScoreMatch cache, unused by this kernel
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
			return np.where(valid & gate, sloperamp(d, bad, good), 0.0)
		inter_raw, intra_raw = evalpairs(cache, 'both', fn)
		return termresult(inter_raw, intra_raw, weight)
	def fullatompairs(cache, same_res=False, cp='cp4', use_cp_rep=False):
		'''
		Build the typed-pair arrays (i, j, distance, weight)
		Arguments:
		----------
			cache: dict - ScoreMatch cache
			same_res: bool - True for intra-residue, False for inter-residue
				pairs
		Returns:
		--------
			tuple: (pi, pj, r, w) NumPy arrays for the matching pair subset
		'''
		cp_half = float((params.get('CountPair') or {})['half'])
		mask = cache['pair_same_res']
		sel = mask if same_res else ~mask
		path = cache['pair_cp_path'] if use_cp_rep else cache['pair_path']
		if same_res:
			if cp == 'cp3':
				w = np.where(path <= 2, 0.0, np.where(path == 3, cp_half, 1.0))
			else:
				w = np.where(path <= 3, 0.0, np.where(path == 4, cp_half, 1.0))
		else:
			w = cache['pair_w']
			if use_cp_rep:
				if cp == 'cp3':
					w = np.where(path <= 2, 0.0,
						np.where(path == 3, cp_half, 1.0))
				else:
					w = np.where(path <= 3, 0.0,
						np.where(path == 4, cp_half, 1.0))
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
			cache: per-pose cache from _fullatomcache
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
	def fullatomljraw(cache, same_res):
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
		pi, pj, r, w = fullatompairs(cache, same_res=same_res)
		if len(pi) == 0: return 0.0, 0.0
		atrE, repE = ljpair(cache, pi, pj, r)
		return float(np.sum(w * atrE)), float(np.sum(w * repE))
	_lkb = params.get('LkBall') or {}
	lk_max = float(_lkb.get('max_dis', 0.0))
	lk_far_lo = float(_lkb.get('far_lo', 0.0))
	def _eval(io_first, io_second, cache, r):
		'''
		Evaluate analytic etable lk_iso for direction (self=first).
		Arguments:
		----------
			io_first:  self-atom etable idx (npair,)
			io_second: other-atom etable idx (npair,)
			cache: dict - ScoreMatch cache
			r: np.ndarray - per-pair distances
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
		e = np.where((d >= lk_far_lo) & (d < lk_max), e_far, e)
		e = np.where(d >= lk_max, 0.0, e)
		return e * fw
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
			cache: per-pose cache from _fullatomcache
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
		lki = _eval(ai_safe, aj_safe, cache, r)
		lkj = _eval(aj_safe, ai_safe, cache, r)
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
		e_mid = (lc1 * np.exp(-x1) + lc2 * np.exp(-x2)) / np.maximum( d2, 1e-12)
		c0 = cp[:, 0]; c1 = cp[:, 1]; c2 = cp[:, 2]; c3 = cp[:, 3]
		e_close = ((c3 * d + c2) * d + c1) * d + c0
		f0 = fp[:, 0]; f1 = fp[:, 1]; f2 = fp[:, 2]; f3 = fp[:, 3]
		e_far = ((f3 * d + f2) * d + f1) * d + f0
		e = np.where(d < cs, cf, e_mid)
		e = np.where((d >= cs) & (d < ce), e_close, e)
		e = np.where((d >= lk_far_lo) & (d < lk_max), e_far, e)
		e = np.where(d >= lk_max, 0.0, e)
		e = e * fw
		e = np.where(valid, e, 0.0)
		return e
	def fullatomsolraw(cache, same_res):
		'''
		Lazaridis-Karplus solvation raw sum, using the
		per-atom-type-pair etable params (close-poly + far-fade).
		Both per-direction terms summed and weighted by the count-pair
		weight w.
		Arguments:
		----------
			cache: per-pose cache from _fullatomcache
			same_res: if True compute intra-residue (xover4) subset,
				otherwise inter-residue pairs
		Returns:
		--------
			float: raw sum of (lki + lkj) * w over heavy-heavy pairs
		'''
		pi, pj, r, w = fullatompairs(cache, same_res=same_res)
		if len(pi) == 0: return 0.0
		e = solpair(cache, pi, pj, r)
		return float(np.sum(w * e))
	def fullatomstubterm(weight_key):
		'''
		Sum a user-supplied per-pair function over typed pairs
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
		for bidx in range(36 * 36):
			i_phi, i_psi = divmod(bidx, 36)
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
					g['neglogP'][i_phi, i_psi] = min(MAXE, -math.log(new_P))
				else:
					g['neglogP'][i_phi, i_psi] = min(MAXE, -math.log(Pk_clip))
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
		Build the FaDun entropy-correction grid for one AA
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
			_FADUN_ENT_CACHE[aa] = (np.zeros((36, 36)), np.zeros((36, 36)))
			return _FADUN_ENT_CACHE[aa]
		rot = entry['rotamers']
		offs = rot['bin_offsets']
		tbl = rot['table']
		ent = np.zeros((36, 36))
		for bidx in range(36 * 36):
			i_phi, i_psi = divmod(bidx, 36)
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
		ypp_psi = np.array([ periodic_cubic_spline(ent[i]) for i in range(36)])
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
	nrchi_cache = {}
	def fadun_nrchi_data(tri):
		'''
		Load and cache the per-AA non-rotameric chi_last density tables
		(Shapovalov backbone-dependent source) for the 8 semi-rotameric AAs.
			Pre-computes 2nd-derivative grids
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
		if tri in nrchi_cache:
			return nrchi_cache[tri]
		nrchi_db = params.get('FaDunNrchiDensities') or {}
		aa_entry = nrchi_db.get(tri)
		if aa_entry is None:
			nrchi_cache[tri] = {}
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
		MAXE = 13.815510557964274
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
				ypp_dens[:, :, k] = fadun_ypp_psi_grid( neglogD[:, :, k])
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
		nrchi_cache[tri] = out
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
	def _unwrap(v, ref):
		'''
		Unwrap a periodic 1D array to within +/-180 degrees
		Arguments:
		----------
			arr: np.ndarray - 1D array of angle samples in degrees
		Returns:
		--------
			np.ndarray: unwrapped copy of arr
		'''
		return ref + ((v - ref + 180.0) % 360.0 - 180.0)
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
		neg_log_dens = spline_eval_1d( v_arr, ypp_c, fc_mod, chi_last_n)
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
		M = _PCS_M.get(n)
		if M is None:
			F = np.fft.fft(np.eye(n), axis=0)
			k = np.arange(n)
			A_diag = 4.0 + 2.0 * np.cos(2.0 * np.pi * k / n)
			B = np.zeros((n, n))
			for i in range(n):
				B[i, i] = -12.0
				B[i, (i + 1) % n] += 6.0
				B[i, (i - 1) % n] += 6.0
			M = np.real(np.fft.ifft(np.diag(1.0 / A_diag) @ F @ B, axis=0))
			_PCS_M[n] = M
		return M @ y
	def spline_eval_1d(y, ypp, t, n):
		'''
		Evaluate a 1D cubic spline from precomputed 2nd derivs
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
		Build forward and reverse chemtype maps for HBond
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
			'TRP','TYR','VAL',
			'ORN','MSE','TPO','SEC','FT6','PTR']:
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
			'TRP','TYR','VAL',
			'ORN','MSE','TPO','SEC','FT6','PTR']:
			acceptor_map[(tri, 'O')] = 'hbacc_PBA'
			base_map[(tri, 'O')] = 'C'
			acceptor_map[(tri, 'OXT')] = 'hbacc_PBA'
			base_map[(tri, 'OXT')] = 'C'
		acceptor_map[('ASN', 'OD1')] = 'hbacc_CXA'
		base_map[('ASN','OD1')] = 'CG'
		acceptor_map[('GLN', 'OE1')] = 'hbacc_CXA'
		base_map[('GLN','OE1')] = 'CD'
		acceptor_map[('ASP', 'OD1')] = 'hbacc_CXL'
		base_map[('ASP','OD1')] = 'CG'
		acceptor_map[('ASP', 'OD2')] = 'hbacc_CXL'
		base_map[('ASP','OD2')] = 'CG'
		acceptor_map[('GLU', 'OE1')] = 'hbacc_CXL'
		base_map[('GLU','OE1')] = 'CD'
		acceptor_map[('GLU', 'OE2')] = 'hbacc_CXL'
		base_map[('GLU','OE2')] = 'CD'
		acceptor_map[('HIS', 'ND1')] = 'hbacc_IME'
		base_map[('HIS','ND1')] = 'CG'
		acceptor_map[('HIS_D', 'NE2')] = 'hbacc_IMD'
		base_map[('HIS_D','NE2')] = 'CD2'
		acceptor_map[('TYR', 'OH')] = 'hbacc_AHX'; base_map[('TYR','OH')] = 'CZ'
		acceptor_map[('SER', 'OG')] = 'hbacc_HXL'; base_map[('SER','OG')] = 'CB'
		acceptor_map[('THR', 'OG1')] = 'hbacc_HXL'
		base_map[('THR','OG1')] = 'CB'
		# Non-canonical sidechains. Chemical types follow the closest
		# canonical analogue: ORN's neutral amine as the amide amine
		# (its NH2O atom type already is), the phosphate esters as
		# hydroxyls, and FT6's indole NE1 exactly as TRP's.
		donor_map[('ORN', 'NE')] = 'hbdon_CXA'
		donor_map[('FT6', 'NE1')] = 'hbdon_IND'
		donor_map[('TPO', 'O2P')] = 'hbdon_HXL'
		donor_map[('TPO', 'O3P')] = 'hbdon_HXL'
		donor_map[('PTR', 'O2P')] = 'hbdon_HXL'
		donor_map[('PTR', 'O3P')] = 'hbdon_HXL'
		for tri in ('TPO', 'PTR'):
			for o in ('O1P', 'O2P', 'O3P'):
				acceptor_map[(tri, o)] = 'hbacc_HXL'
				base_map[(tri, o)] = 'P'
		acceptor_map[('TPO', 'OG1')] = 'hbacc_HXL'
		base_map[('TPO', 'OG1')] = 'CB'
		acceptor_map[('PTR', 'OH')] = 'hbacc_AHX'
		base_map[('PTR', 'OH')] = 'CZ'
		return donor_map, acceptor_map, base_map
	def hbond_eval_lookup(hb):
		'''
		Look up the polynomial row for a donor/acceptor pair
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
			poly: dict - {'xmin', 'xmax', 'min_val', 'max_val', 'coeffs'}
				polynomial entry
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
	def burial_w(n):
		'''
		Heavy-neighbour burial weight used by LkBallWtd
		Arguments:
		----------
			gi: int - atom index
		Returns:
		--------
			float: burial weight
		'''
		BU = (params.get('HBondSp2') or {})['burial']
		if n < BU['nb_lo']: return BU['w_lo']
		if n > BU['nb_hi']: return BU['w_hi']
		return (n - BU['shift']) * BU['slope']
	def ahd_arg(poly, AHD_rad, xD):
		'''
		Acceptor-hydrogen-donor angle argument used by HBond polynomials
		Arguments:
		----------
			D: np.ndarray - donor position
			H: np.ndarray - hydrogen position
			A: np.ndarray - acceptor position
			AHD_rad: float - the A-H-D angle in radians
			xD: float - minus the cosine of the A-H-D angle
		Returns:
		--------
			float: cosine of the A-H-D angle
		'''
		if poly is None: return xD
		return AHD_rad if poly.get('xmin', -1) > 0.5 else xD
	def hbondpair(ix, iy, HS, acc_str_tab, acceptors, all_hbonds, atoms, coords,
			dHA, don_str_tab, donors, eval_key, fades, polys):
		'''
		Score one candidate donor/acceptor pair and record it if accepted
		Arguments:
		----------
			ix: int - index into the donors list
			iy: int - index into the acceptors list
			HS: dict - h-bond fade and strength settings
			acc_str_tab: dict - acceptor strength table
			acceptors: list - acceptor descriptor dicts
			all_hbonds: list - accepted h-bonds, appended in place
			atoms: dict - atom index to atom record
			coords: np.ndarray - (N, 3) coordinates
			dHA: np.ndarray - donor-H to acceptor distance matrix
			don_str_tab: dict - donor strength table
			donors: list - donor descriptor dicts
			eval_key: dict - donor, acceptor and separation to params
			fades: dict - named fade-function definitions
			polys: dict - named polynomial definitions
		Returns:
		--------
			No return value, an accepted h-bond is appended to all_hbonds
		'''
		d = donors[ix]; a = acceptors[iy]
		if d['D'] == a['A']: return
		if d['ri'] == a['ri']: return
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
		if entry is None: return
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
		Pr = hbond_poly_eval(poly_d, AH)
		PSxH = hbond_poly_eval(poly_bah_short, xH)
		PLxH = hbond_poly_eval(poly_bah_long, xH)
		PSxD = hbond_poly_eval(
			poly_ahd_short, ahd_arg(poly_ahd_short, AHD_rad, xD))
		PLxD = hbond_poly_eval(
			poly_ahd_long, ahd_arg(poly_ahd_long, AHD_rad, xD))
		FSr = hbond_fade( fades.get(entry['fade_AHdist']), AH)
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
		s = (don_str_tab.get(d['chem'], 1.0) * acc_str_tab.get(a['chem'], 1.0))
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
				d_p = float(HS['BAH180_rise'])
				m_p = float(HS['fade_slope'])
				l_p = float(HS['outer_width'])
				PI = math.pi
				PI_minus_BAH = math.acos( max(-1.0, min(1.0, xH)))
				BAH = PI - PI_minus_BAH
				H_chi = (math.cos(2 * chi) + 1) * 0.5
				if BAH >= PI * 2.0 / 3.0:
					F_p = d_p * 0.5 * math.cos(3 * PI_minus_BAH) \
						+ d_p * 0.5 - 0.5
					G_p = d_p - 0.5
				elif BAH >= PI * (2.0 / 3.0 - l_p):
					outer = math.cos( PI - (PI * 2.0 / 3.0 - BAH) / l_p)
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
				max_penalty = float(HS['max_penalty'])
				PI_minus_BAH = math.acos( max(-1.0, min(1.0, xH)))
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
		if input_e > HS['fade_hi']:
			return
		if input_e > HS['fade_lo']:
			e = (HS['fade_c0'] + HS['fade_c1'] * input_e
				+ HS['fade_c2'] * input_e * input_e)
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
	def fullatomhbond(pose, cache, per_hb=None):
		'''
		Compute the hydrogen-bond energy with the four categories partitioned
		Arguments:
		----------
			cache: dict - ScoreMatch cache
		Returns:
		--------
			dict: per-category raw and weighted contributions ('sr_bb', 'lr_bb',
				'bb_sc', 'sc')
		'''
		HS = params.get('HBondSp2') or {}
		hb = params.get('HBond_data') or {}
		if not hb: return {'SR_BB': 0.0, 'LR_BB': 0.0, 'BB_SC': 0.0, 'SC': 0.0}
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
		dd = np.linalg.norm(nb_arr[:, None, :] - nb_arr[None, :, :], axis=2)
		within = (dd < 10.0)
		counts = within.sum(axis=1)
		for k, r in enumerate(ri_list):
			nb_count[r] = int(counts[k])
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
		cat_totals = {'SR_BB': 0.0, 'LR_BB': 0.0, 'BB_SC': 0.0, 'SC': 0.0}
		if not donors or not acceptors: return cat_totals
		all_hbonds = []
		H_idx = np.array([d['H'] for d in donors], dtype=np.int64)
		A_idx = np.array([a['A'] for a in acceptors], dtype=np.int64)
		Hc = coords[H_idx]; Ac = coords[A_idx]
		dHA = np.linalg.norm( Hc[:, None, :] - Ac[None, :, :], axis=2)
		within = np.where(dHA < 3.2)
		for ix, iy in zip(*within):
			hbondpair(ix, iy, HS, acc_str_tab, acceptors, all_hbonds, atoms,
				coords, dHA, don_str_tab, donors, eval_key, fades, polys)
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
		cache.update(patternsearchsmall(pose, params, ligand,
			xs_override, nrot_override))
	if 'Atom_types' in params:
		cache.update(fullatomcache(pose, params))
	if not cache:
		raise Exception( 'ScoreMatch: unsupported params')
	cache['patternsearch'] = patternsearch
	cache['patternsearchsmall'] = patternsearchsmall
	cache['fullatomcache'] = fullatomcache
	cache['bfswithin'] = bfswithin
	cache['countnrot'] = countnrot
	cache['countnumtors'] = countnumtors
	cache['ringatoms'] = ringatoms
	cache['topologyhash'] = topologyhash
	cache['evalpairs'] = evalpairs
	cache['termresult'] = termresult
	cache['gausspair'] = gausspair
	cache['slopestep'] = slopestep
	cache['fullatompairs'] = fullatompairs
	cache['ljpair'] = ljpair
	cache['fullatomljraw'] = fullatomljraw
	cache['lkisopair'] = lkisopair
	cache['solpair'] = solpair
	cache['fullatomsolraw'] = fullatomsolraw
	cache['fullatomstubterm'] = fullatomstubterm
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
	def cached_dihedral(pose, ri, dtype, chi_type=None):
		'''
		Memoised pose.GetDihedral: backbone phi/psi are requested per
		residue by FaDun/Rama/PAaPp/Omega each score, so compute each
		dihedral once per score and share it (memo cleared each Score call)
		Arguments:
		----------
			pose: Pose - structure being scored
			ri: int - residue index
			dtype: str - dihedral name ('PHI', 'PSI', ...)
			chi_type: int or None - chi index when dtype is 'CHI'
		Returns:
		--------
			float: dihedral angle; re-raises GetDihedral's error if undefined
		'''
		memo = cache.setdefault('_dihedral_memo', {})
		key = (int(ri), dtype, chi_type)
		if key in memo:
			v = memo[key]
			if isinstance(v, BaseException): raise v
			return v
		try:
			if chi_type is None:
				r = pose.GetDihedral(int(ri), dtype)
			else:
				r = pose.GetDihedral(int(ri), dtype, chi_type=chi_type)
		except BaseException as ex:
			memo[key] = ex
			raise
		memo[key] = r
		return r
	cache['cdih'] = cached_dihedral
	def fullatomhbond_memo(pose, cache, per_hb=None):
		'''
		Memoised fullatomhbond: the four HBond terms each request the full
		four-category result, so compute it once per score and reuse it
		(the memo is cleared each Score call)
		Arguments:
		----------
			pose: Pose - structure being scored
			cache: dict - the ScoreMatch cache
			per_hb: list or None - per-bond collector; bypasses the memo
		Returns:
		--------
			dict: the four-category HBond contribution dict
		'''
		if per_hb is not None:
			return fullatomhbond(pose, cache, per_hb)
		m = cache.get('_hbond_memo')
		if m is None:
			m = fullatomhbond(pose, cache)
			cache['_hbond_memo'] = m
		return m
	cache['fullatomhbond'] = fullatomhbond_memo
	return cache



























































































































def _validate_rot_entry(rot_entry, expected_tricode):
	'''
	Validate a rotamer JSON against the Dunbrack BBDEP2010 schema.
	The JSON must come from https://github.com/sarisabban/ncaarotamers.
	Arguments:
	----------
		rot_entry         : dict - parsed JSON content
		expected_tricode  : str  - tricode the caller is asking us to
		                    insert; must match rot_entry['tricode']
	Returns:
	--------
		None - raises ValueError on any schema violation with explicit
		context.
	'''
	REQUIRED_TOP = ('tricode', 'n_chi', 'rotamers')
	REQUIRED_ROT = ('columns', 'table', 'bin_offsets', 'top_chi')
	PHI_N, PSI_N = 36, 36
	missing_top = [k for k in REQUIRED_TOP if k not in rot_entry]
	if missing_top:
		raise ValueError(
			f'rotamer JSON missing required keys: {missing_top}')
	tri = rot_entry['tricode']
	if not isinstance(tri, str) or len(tri) != 3:
		raise ValueError(
			f'rotamer JSON tricode must be a 3-letter str, got {tri!r}')
	if tri.upper() != expected_tricode.upper():
		raise ValueError(
			f'rotamer JSON tricode {tri!r} does not match argument '
			f'{expected_tricode!r}')
	n_chi = int(rot_entry['n_chi'])
	if n_chi < 1 or n_chi > 8:
		raise ValueError(f'n_chi out of range (1-8): {n_chi}')
	if ('method' not in rot_entry
			or 'chi_axes' not in rot_entry['method']):
		raise ValueError(
			'rotamer JSON missing method.chi_axes (required as the '
			'source of truth for Amino Acids "Chi Angle Atoms")')
	chi_axes = rot_entry['method']['chi_axes']
	if len(chi_axes) != n_chi:
		raise ValueError(
			f'method.chi_axes has {len(chi_axes)} axes but '
			f'n_chi={n_chi}')
	for k, axis in enumerate(chi_axes):
		if len(axis) != 4:
			raise ValueError(
				f'chi_axes[{k}] has {len(axis)} atoms (need 4): {axis}')
	rot = rot_entry['rotamers']
	missing_rot = [k for k in REQUIRED_ROT if k not in rot]
	if missing_rot:
		raise ValueError(f'rotamers missing keys: {missing_rot}')
	expect_cols = (['count', 'prob']
		+ [f'chi{k+1}' for k in range(n_chi)]
		+ [f'sig{k+1}' for k in range(n_chi)])
	if rot['columns'] != expect_cols:
		raise ValueError(
			f'rotamer columns mismatch.\n'
			f'  got:      {rot["columns"]}\n'
			f'  expected: {expect_cols}')
	bo_off = rot['bin_offsets']
	if len(bo_off) != PHI_N * PSI_N + 1:
		raise ValueError(
			f'bin_offsets length {len(bo_off)} != '
			f'{PHI_N * PSI_N + 1}')
	tc = rot['top_chi']
	if len(tc) != PHI_N:
		raise ValueError(
			f'top_chi outer length {len(tc)} != {PHI_N}')
	if any(len(r) != PSI_N for r in tc):
		raise ValueError(
			f'top_chi inner length != {PSI_N}')

def _clamp_sigmas(rot_entry, floor=0.5):
	'''
	Clamp sigma columns of the rotamer table to >= floor degrees.

	Some BGMM bins emit zero-width sigmas when the underlying data is
	a single delta; the unified-DB schema requires sigmas >= 0.5 deg
	to avoid divide-by-zero in Score._rotamer_prior.

	Arguments:
	----------
		rot_entry : dict
		floor     : float - minimum sigma in degrees (default 0.5)
	Returns:
	--------
		int : count of values clamped (informational only)
	'''
	n_chi = int(rot_entry['n_chi'])
	sig_col0 = 2 + n_chi
	table = rot_entry['rotamers']['table']
	n_clamped = 0
	for row in table:
		for k in range(n_chi):
			v = float(row[sig_col0 + k])
			if v < floor:
				row[sig_col0 + k] = floor
				n_clamped += 1
	return n_clamped

def Parameterise(cif_file, rotamer_json_file, tricode, unicode,
		backup=True):
	'''
	Add a non-canonical amino acid (NCAA) to Pose's unified
	database.json.

	Builds the entry under "Amino Acids"[unicode] from cif_file
	(verified RCSB Chemical Component Dictionary CIF) and inserts the
	matching backbone-dependent rotamer library under
	"Rotamer Library"["residues"][tricode] from rotamer_json_file
	(Dunbrack BBDEP2010-format JSON produced by the
	https://github.com/sarisabban/ncaarotamers). Both insertions
	land in a single atomic write.

	Arguments:
	----------
		cif_file          : str  - Path to RCSB CCD CIF
		rotamer_json_file : str  - Path to Dunbrack-format rotamer JSON
		tricode           : str  - Three-letter residue code, e.g. 'PTR'
		unicode           : str  - Single-letter key for db['Amino Acids']
		backup            : bool - If True (default), copy database.json
		                    to database.json.bak.<YYYYMMDD-HHMMSS>
		                    before modifying. Set False for batch / CI
		                    runs that handle backups externally.
	Behaviour on existing keys:
	---------------------------
		If `unicode` is already a key in db['Amino Acids'], or `tricode`
		is already in db['Rotamer Library']['residues'], a warning is
		logged to stderr identifying the old entry, and the new entries
		overwrite the old.
	Returns:
	--------
		None - database.json is updated in place; DBLoad cache is cleared
		so subsequently constructed Pose / ForceField / Score / Rotamers
		instances see the new residue without restart.
	'''
	# 1. ALA reference frame (N at origin): N, H1-3, CA, HA, CB, 1HB-3HB, C, O, OXT.
	ALA = np.array([
		[ 0.000,  0.000,  0.000], [-0.334, -0.943,  0.000],
		[-0.334,  0.471,  0.816], [-0.334,  0.471, -0.816],
		[ 1.458,  0.000,  0.000], [ 1.822, -0.535,  0.877],
		[ 1.988, -0.773, -1.199], [ 3.078, -0.764, -1.185],
		[ 1.633, -1.802, -1.154], [ 1.633, -0.307, -2.117],
		[ 2.009,  1.420,  0.000], [ 2.058,  2.045,  1.023],
		[ 2.394,  1.914, -1.023]])
	unicode, tricode = unicode.upper(), tricode.upper()
	# 2. Load + validate the rotamer JSON FIRST. Failing fast on a bad
	#    schema means we never half-write a CIF-derived entry into the
	#    DB without a matching rotamer library.
	with open(rotamer_json_file) as fh:
		rot_entry = json.load(fh)
	_validate_rot_entry(rot_entry, tricode)
	n_clamped = _clamp_sigmas(rot_entry)
	if n_clamped:
		print(f'Note: clamped {n_clamped} rotamer sigma values to '
			f'>=0.5 deg floor.')
	chi_axes_from_json = rot_entry['method']['chi_axes']
	# 3. Parse CIF: atom rows have >=18 tokens (coords at [15:18] or fallback [12:15]); bond rows have 7 tokens.
	COORD_RAW, ATOMS_RAW, BONDS = [], [], []
	with open(cif_file) as fh:
		for line in fh:
			t = line.strip().split()
			if not t or t[0] != tricode: continue
			if len(t) == 7 and t[3] in ('SING','DOUB','TRIP','AROM'):
				BONDS.append((t[1], t[2], t[3], t[4]))
			elif len(t) >= 18:
				try:
					try:c = [float(t[i]) for i in (15, 16, 17)]
					except (ValueError, IndexError):
						c = [float(t[i]) for i in (12, 13, 14)]
					COORD_RAW.append(c)
					ATOMS_RAW.append({'id': t[1],
						'elem': t[3].capitalize(),
						'bb': (t[9] == 'Y')})
				except (ValueError, IndexError): pass
	COORD   = np.array(COORD_RAW)
	CIF_IDS = [a['id'] for a in ATOMS_RAW]
	if 'CB' not in CIF_IDS:
		raise ValueError(f'No CB atom found in {cif_file}. '
			'Only standard amino acids (not GLY) are supported.')
	# 4. Validate every chi-axis atom from the JSON exists in the CIF.
	#    Chi axes only ever reference heavy atoms, so a CIF-name lookup
	#    is sufficient (no need to canonicalise H names yet).
	cif_atom_set = set(CIF_IDS)
	for k, axis in enumerate(chi_axes_from_json):
		for an in axis:
			if an not in cif_atom_set:
				raise ValueError(
					f'chi axis {k+1} references atom {an!r} which '
					f'does not exist in {cif_file}. CIF atoms: '
					f'{sorted(cif_atom_set)}')
	bb_set = {a['id'] for a in ATOMS_RAW if a['bb']} or {
		'N','CA','C','O','OXT','H','H1','H2','H3',
		'HA','HA2','HA3','HXT'}
	elem    = {a['id']: a['elem'] for a in ATOMS_RAW}
	cif_ord = {a['id']: i for i, a in enumerate(ATOMS_RAW)}
	# 5. Superimpose onto ALA backbone frame via rigid motion (N, CA, CB, C) with CA as origin.
	try:
		Ni, CAi = CIF_IDS.index('N'),  CIF_IDS.index('CA')
		CBi, Ci = CIF_IDS.index('CB'), CIF_IDS.index('C')
	except ValueError as e:
		raise ValueError(f'Missing backbone atom in {cif_file}: {e}')
	A = np.c_[ALA,   np.ones(len(ALA))]
	B = np.c_[COORD, np.ones(len(COORD))]
	AL = np.array([A[0]-A[4], A[6]-A[4], A[-3]-A[4], A[4]])
	BL = np.array([B[Ni]-B[CAi], B[CBi]-B[CAi], B[Ci]-B[CAi], B[CAi]])
	COORD = (B @ (np.linalg.inv(BL) @ AL))[:, :3]
	# 6. Build undirected bond graph indexed by atom id.
	adj = defaultdict(set)
	for a1, a2, _v, _r in BONDS:
		adj[a1].add(a2); adj[a2].add(a1)
	# 7. BFS sidechain from CB (heavy-atom queue; H neighbours land next to their parent in CIF order).
	ordered = []
	seen    = set(bb_set) | {'CB'}
	q       = deque(['CB'])
	while q:
		atom = q.popleft()
		ordered.append(atom)
		for n in sorted(adj[atom], key=lambda m: cif_ord.get(m, 9999)):
			if n in seen: continue
			seen.add(n)
			(ordered if elem.get(n, '').upper() in ('H', 'D') else q).append(n)
	# 8. Rename atoms: CIF HB2 becomes Pose 1HB (per-base counter on H/D atoms).
	name_map, counter = {}, defaultdict(int)
	for name in ordered:
		m = re.match(r'^([A-Z]+)(\d+)$', name)
		if m and elem.get(name, '').upper() in ('H', 'D'):
			counter[m.group(1)] += 1
			name_map[name] = f'{counter[m.group(1)]}{m.group(1)}'
		else:
			name_map[name] = name
	# 9. Detect fused sidechain: any sidechain atom bonded back to N (e.g. PRO).
	fused_atom = next((sc for sc in ordered if 'N' in adj[sc]), None)
	fused = fused_atom is not None
	# 10. Sidechain bond graph on new indices; -5 sentinel stands in for the backbone N of a fused ring.
	sc_set  = set(ordered)
	new_idx = {n: i for i, n in enumerate(ordered)}
	sc_bonds, sc_orders, bo_lookup = (
		defaultdict(list), defaultdict(list), {})
	for a1, a2, vo, ar in BONDS:
		u = vo.upper()
		if   ar == 'Y':   bo = 1.5
		elif u == 'SING': bo = 1
		elif u == 'DOUB': bo = 2
		elif u == 'TRIP': bo = 3
		elif u == 'AROM': bo = 1.5
		else:
			print(f'Warning: unknown bond order {vo!r} for '
				f'{a1}-{a2} in {tricode}, defaulting to 1')
			bo = 1
		bo_lookup[(a1, a2)] = bo_lookup[(a2, a1)] = bo
	for a1, a2, _v, _r in BONDS:
		if a1 in sc_set and a2 in sc_set:
			i1, i2 = new_idx[a1], new_idx[a2]
			sc_bonds[i1].append(i2);  sc_bonds[i2].append(i1)
			bo = bo_lookup[(a1, a2)]
			sc_orders[i1].append(bo); sc_orders[i2].append(bo)
	if fused:
		fi = new_idx[fused_atom]
		sc_bonds[fi].append(-5);  sc_bonds[-5].append(fi)
		sc_orders[fi].append(1);  sc_orders[-5].append(1)
	# 11. Two-pass aromaticity: C with >=2 O/N neighbours and at least one double bond gets resonance (all C-O/N -> 1.5).
	elem_at_idx = {new_idx[n]: elem[n] for n in ordered}
	for _p in range(2):
		for i in list(sc_bonds.keys()):
			if i < 0 or elem_at_idx.get(i, '') != 'C': continue
			xs = [(k, nb, bo) for k, (nb, bo) in enumerate(
					zip(sc_bonds[i], sc_orders[i]))
				if nb >= 0 and elem_at_idx.get(nb, '') in ('O','N')]
			if len(xs) < 2 or not any(bo >= 2 for _, _, bo in xs): continue
			for k, nb, bo in xs:
				if bo == 1.5: continue
				sc_orders[i][k] = 1.5
				for kk, mnb in enumerate(sc_bonds[nb]):
					if mnb == i:
						sc_orders[nb][kk] = 1.5
						break
	# 12. Final bond dicts with sorted neighbours for determinism; -5 sentinel kept last if fused.
	pos_keys     = sorted(k for k in sc_bonds if k >= 0)
	final_bonds  = {k: sorted(sc_bonds[k]) for k in pos_keys}
	final_orders = {k: [dict(zip(sc_bonds[k], sc_orders[k]))[nb]
		for nb in final_bonds[k]] for k in pos_keys}
	if fused:
		final_bonds[-5]  = sorted(sc_bonds[-5])
		final_orders[-5] = [dict(zip(sc_bonds[-5], sc_orders[-5]))[nb]
			for nb in final_bonds[-5]]
	# 13. Chi axes come from the rotamer JSON's method.chi_axes (the
	#     plan's user-confirmed source of truth). The CIF walker that
	#     this section used to do is removed -- it was fragile for
	#     NCAAs with non-standard atom orderings, and any drift between
	#     the Amino Acids "Chi Angle Atoms" field and the rotamer
	#     library's chi convention silently corrupts the rotamer prior.
	chis = [list(a) for a in chi_axes_from_json]
	# 14. Assemble the new entry in the same field order as existing AAs.
	id_to_i = {cid: i for i, cid in enumerate(CIF_IDS)}
	def _infer_hybridisation(elem, bond_orders):
		'''
		Classify an atom's hybridization from its element and the list of
		bond orders incident on it. Relies on aromatic/resonance bonds
		having been encoded as order 1.5 by the caller (true of Parameterise
		after the two-pass aromaticity rewrite, and of Molecule.Import after
		bond-order inference).
		Arguments:
		----------
			elem:        str, element symbol (case-insensitive)
			bond_orders: iterable of numeric bond orders on this atom
		Returns:
		--------
			str: one of 's', 'sp', 'sp2', 'sp3'
		'''
		if elem and elem.upper() == 'H': return 's'
		bos = list(bond_orders)
		if any(bo == 3 for bo in bos):   return 'sp'
		if any(bo >= 1.5 for bo in bos): return 'sp2'
		return 'sp3'
	entry = {
		'Vectors':         [COORD[id_to_i[n]].tolist() for n in ordered],
		'Tricode':         tricode,
		'Fused':           fused,
		'Sidechain Atoms': [[name_map[n], elem[n], 0, 1.0, 0,
			_infer_hybridisation(elem[n], sc_orders[new_idx[n]])]
			for n in ordered],
		'Chi Angle Atoms': chis,
		'Bonds':           {str(k): v for k, v in final_bonds.items()},
		'BondOrders':      {str(k): v for k, v in final_orders.items()}}
	# 15. Load the existing database.
	db_path = os.path.join(
		os.path.dirname(os.path.abspath(__file__)), 'database.json')
	with open(db_path) as fh: db = json.load(fh)
	# 16. Warn-and-overwrite on key collisions, per user-confirmed plan.
	#     Both single-letter unicode (Amino Acids) and 3-letter tricode
	#     (Rotamer Library.residues) are checked independently.
	if unicode in db.get('Amino Acids', {}):
		old_tri = db['Amino Acids'][unicode].get('Tricode', '?')
		print(f'Warning: db["Amino Acids"]["{unicode}"] already '
			f'exists (was Tricode={old_tri}); overwriting with '
			f'Tricode={tricode}.', file=sys.stderr)
	rl       = db.setdefault('Rotamer Library', {})
	rl_resid = rl.setdefault('residues', {})
	if tricode in rl_resid:
		print(f'Warning: db["Rotamer Library"]["residues"]'
			f'["{tricode}"] already exists; overwriting.',
			file=sys.stderr)
	# 17. Insert both entries. The Rotamer Library form keeps only
	#     n_chi/rotamers/densities (matching merge_into_database.py);
	#     the method/metadata fields are stripped on insertion to keep
	#     database.json compact.
	def dtricode(ltri, db):
		taken = set()
		for ek, e in db.get('Amino Acids', {}).items():
			if ek == unicode: continue
			t = e.get('Tricode')
			if isinstance(t, list): taken.update(t)
			elif isinstance(t, str): taken.add(t)
		cands = ['D'+ltri[1:], 'D'+ltri[0]+ltri[2], 'D'+ltri[0:2]]
		cands += ['D'+chr(65+n//26)+chr(65+n%26) for n in range(676)]
		for c in cands:
			if c not in taken: return c
		raise Exception(f'No free D-tricode for {ltri}')
	entry['Tricode'] = [tricode, dtricode(tricode, db)]
	db.setdefault('Amino Acids', {})[unicode] = entry
	rl_resid[tricode] = {
		'n_chi':     int(rot_entry['n_chi']),
		'rotamers':  rot_entry['rotamers'],
		'densities': rot_entry.get('densities'),
	}
	# 18. Validate Bonds/BondOrders symmetry across the whole DB before
	#     writing anything. Fails loudly on any malformed entry so the
	#     hot paths in Pose (_bondtree, Import) can stay guard-free.
	def _validate_db(db):
		for section in ('Amino Acids', 'Nucleotides'):
			for ekey, e in db.get(section, {}).items():
				if 'Bonds' not in e: continue
				bonds = e['Bonds']
				if 'BondOrders' not in e:
					raise ValueError(
						f'{section}[{ekey!r}]: '
						f'has Bonds but no BondOrders')
				bo = e['BondOrders']
				for k, nbrs in bonds.items():
					if k not in bo:
						raise ValueError(
							f'{section}[{ekey!r}]: '
							f'BondOrders missing key {k!r}')
					if len(bo[k]) != len(nbrs):
						raise ValueError(
							f'{section}[{ekey!r}][{k!r}]: '
							f'Bonds has {len(nbrs)} entries but '
							f'BondOrders has {len(bo[k])}')
	_validate_db(db)
	# 19. Optional timestamped backup before atomic write.
	if backup:
		ts = time.strftime('%Y%m%d-%H%M%S')
		bak_path = db_path + f'.bak.{ts}'
		shutil.copy2(db_path, bak_path)
		print(f'Backup: {bak_path}')
	# 20. Compact atomic write (no whitespace -- matches the rest of
	#     the unified-DB infrastructure).
	tmp_path = db_path + '.tmp'
	try:
		with open(tmp_path, 'w') as fh:
			json.dump(db, fh, separators=(',', ':'))
		os.replace(tmp_path, db_path)
	except BaseException:
		if os.path.exists(tmp_path):
			os.remove(tmp_path)
		raise
	# 21. Invalidate the DBLoad cache so subsequently constructed Pose
	#     / ForceField / Score / Rotamers instances see the new residue
	#     without restart.
	DBLoad.cache_clear()
	print(f'Added {tricode} as "{unicode}" to database.json '
		f'(Amino Acids + Rotamer Library)')

def RMSD(pose1, pose2, alg='align', export=None):
	'''
	Calculate RMSD between two poses (protein or nucleic acid)
	Arguments:
	----------
		pose1  : Pose - First pose (protein or nucleic acid)
		pose2  : Pose - Second pose (must be same Type as pose1)
		alg    : str  - 'align' (default), 'kabsch', 'quaternion', 'simple'
		export : str  - Output filename for aligned PDB pair; None skips export
	Return:
	-------
		float : RMSD value in angstroms, rounded to 5 decimals
	'''
	# 1. Validate algorithm and check both poses are the same molecule type.
	if alg not in ('align', 'kabsch', 'quaternion', 'simple'):
		raise Exception('Unknown algorithm: ' + str(alg))
	t1, t2 = pose1.data['Type'], pose2.data['Type']
	if (t1 == 'Protein') != (t2 == 'Protein'):
		raise Exception(f'Cannot align {t1} with {t2}: '
			'cannot mix protein and nucleic acid')
	# 2. Resolve molecule-specific residue-key and reference-atom name.
	is_pro = (t1 == 'Protein')
	rk     = 'Amino Acids' if is_pro else 'Nucleotides'
	ra     = 'CA' if is_pro else "C1'"
	atoms1, co1, res1 = (pose1.data['Atoms'],
		pose1.data['Coordinates'], pose1.data[rk])
	atoms2, co2, res2 = (pose2.data['Atoms'],
		pose2.data['Coordinates'], pose2.data[rk])
	if alg == 'align':
		# 3. Needleman-Wunsch DP with BLOSUM62 (proteins) or +1/-0.5 (nucleic); gap = -1.
		rk1, rk2 = sorted(res1.keys()), sorted(res2.keys())
		seq1 = ''.join(res1[k][0].upper() for k in rk1)
		seq2 = ''.join(res2[k][0].upper() for k in rk2)
		m, n, gap = len(seq1), len(seq2), -1.0
		dp = np.zeros((m + 1, n + 1))
		dp[:, 0] = np.arange(m + 1) * gap
		dp[0, :] = np.arange(n + 1) * gap
		for i in range(1, m + 1):
			a = seq1[i-1]
			for j in range(1, n + 1):
				b = seq2[j-1]
				s = (_blosum(a, b) if is_pro
					else (1.0 if a == b else -0.5))
				dp[i, j] = max(dp[i-1, j-1] + s,
					dp[i-1, j] + gap, dp[i, j-1] + gap)
		# 4. Traceback the optimal alignment path to recover residue pairs.
		pairs, i, j = [], m, n
		while i > 0 and j > 0:
			a, b = seq1[i-1], seq2[j-1]
			s = (_blosum(a, b) if is_pro
				else (1.0 if a == b else -0.5))
			if   abs(dp[i, j] - (dp[i-1, j-1] + s)) < 1e-9:
				pairs.append((i - 1, j - 1))
				i -= 1; j -= 1
			elif abs(dp[i, j] - (dp[i-1, j]   + gap)) < 1e-9:
				i -= 1
			else:
				j -= 1
		pairs.reverse()
		if len(pairs) < 3:
			raise Exception('Too few aligned residue pairs')
		# 5. Gather reference-atom coordinates for each aligned pair.
		P_aln = np.array([next(co1[ai].copy().astype(float)
			for ai in res1[rk1[ii]][2]
			if atoms1[ai][0] == ra) for ii, _ in pairs])
		Q_aln = np.array([next(co2[ai].copy().astype(float)
			for ai in res2[rk2[jj]][2]
			if atoms2[ai][0] == ra) for _, jj in pairs])
		# 6. Iterative Kabsch with 2.0 A outlier rejection (5 rounds + 1 final fit).
		mask = np.ones(len(pairs), dtype=bool)
		for _ in range(6):
			Pm, Qm   = P_aln[mask], Q_aln[mask]
			t_P, t_Q = Pm.mean(axis=0), Qm.mean(axis=0)
			P, Q     = Pm - t_P, Qm - t_Q
			U, _, Vt = np.linalg.svd(P.T @ Q)
			d = np.sign(np.linalg.det(Vt.T @ U.T))
			R = Vt.T @ np.diag(np.array([1.0, 1.0, d])) @ U.T
			dists = np.sqrt((((P_aln - t_P)
				- (Q_aln - t_Q) @ R) ** 2).sum(axis=1))
			new_mask = dists < 2.0
			if (np.array_equal(new_mask, mask)
					or new_mask.sum() < 3):
				break
			mask = new_mask
	else:
		# 3. Gather all ref-atom coords (skipping residues that lack it), truncate to shorter pose.
		coords1 = [c for c in (next(
			(co1[ai].copy().astype(float)
				for ai in res1[ri][2] if atoms1[ai][0] == ra),
			None) for ri in sorted(res1.keys()))
			if c is not None]
		coords2 = [c for c in (next(
			(co2[ai].copy().astype(float)
				for ai in res2[ri][2] if atoms2[ai][0] == ra),
			None) for ri in sorted(res2.keys()))
			if c is not None]
		if not coords1 or not coords2:
			raise Exception(
				f'No {ra} atoms found in one or both poses')
		n = min(len(coords1), len(coords2))
		P, Q     = np.array(coords1[:n]), np.array(coords2[:n])
		t_P, t_Q = P.mean(axis=0), Q.mean(axis=0)
		P, Q     = P - t_P, Q - t_Q
		# 4. Compute rotation matrix via the selected algorithm (Horn 1987 for quaternion).
		if alg == 'simple':
			R = np.eye(3)
		elif alg == 'kabsch':
			U, _, Vt = np.linalg.svd(P.T @ Q)
			d = np.sign(np.linalg.det(Vt.T @ U.T))
			R = Vt.T @ np.diag(np.array([1.0, 1.0, d])) @ U.T
		else:
			H  = P.T @ Q
			a, b, c = H[0]; d, e, f = H[1]; g, h, k = H[2]
			F = np.array([
				[a+e+k,   f-h,     g-c,     b-d    ],
				[f-h,     a-e-k,   b+d,     c+g    ],
				[g-c,     b+d,    -a+e-k,   f+h    ],
				[b-d,     c+g,     f+h,    -a-e+k  ]])
			q0, q1, q2, q3 = np.linalg.eigh(F)[1][:, -1]
			R = np.array([
				[q0*q0+q1*q1-q2*q2-q3*q3,
					2*(q1*q2-q0*q3),         2*(q1*q3+q0*q2)],
				[2*(q1*q2+q0*q3),
					q0*q0-q1*q1+q2*q2-q3*q3, 2*(q2*q3-q0*q1)],
				[2*(q1*q3-q0*q2),
					2*(q2*q3+q0*q1),         q0*q0-q1*q1-q2*q2+q3*q3]])
	# 7. Compute RMSD from centred coordinate residuals.
	diff = P - Q @ R
	rmsd = np.sqrt(np.mean((diff ** 2).sum(axis=1)))
	# 8. Optionally export aligned pose pair as PDB files.
	if export is not None:
		orig = pose2.data['Coordinates'].copy()
		pose2.data['Coordinates'] = (orig - t_Q) @ R + t_P
		fn, ext = export[:-4], export[-4:]
		pose1.Export(fn + '_1' + ext)
		pose2.Export(fn + '_2' + ext)
		pose2.data['Coordinates'] = orig
	return round(float(rmsd), 5)

# BLOSUM62 scoring matrix — shared by BLAST() and MSA()
_aa  = 'ARNDCQEGHILKMFPSTWYV'
_idx = {c: i for i, c in enumerate(_aa)}
_bm  = [
	[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
	[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
	[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
	[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
	[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
	[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
	[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
	[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
	[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
	[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
	[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
	[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
	[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
	[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
	[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
	[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
	[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-3,-2, 0],
	[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-3,11, 2,-3],
	[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
	[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]]

def _blosum(a, b):
	'''
	BLOSUM62 pairwise amino-acid substitution score
	Arguments:
	----------
		a : str - First amino acid one-letter code (must be uppercase)
		b : str - Second amino acid one-letter code (must be uppercase)
	Return:
	-------
		int : BLOSUM62 score; falls back to +4 for match / -1 for mismatch if unknown
	'''
	# 1. Resolve alphabet indices for both residues.
	ia, ib = _idx.get(a, -1), _idx.get(b, -1)
	# 2. Fallback when either residue is outside the BLOSUM62 alphabet.
	if ia < 0 or ib < 0: return(4 if a == b else -1)
	# 3. Look up the canonical BLOSUM62 score.
	return(_bm[ia][ib])

def BLAST(seq1, seq2):
	'''
	Pairwise protein alignment via Smith-Waterman with BLOSUM62 and Karlin-Altschul E-value
	Arguments:
	----------
		seq1 : str - FASTA sequence of the first (query) protein
		seq2 : str - FASTA sequence of the second (subject) protein
	Return:
	-------
		str   : BLAST-like formatted alignment report
		float : Percent identity over the aligned region
		float : Karlin-Altschul expect value
	'''
	# 1. Uppercase both sequences and capture lengths.
	seq1, seq2 = seq1.upper(), seq2.upper()
	m, n = len(seq1), len(seq2)
	# 2. Smith-Waterman DP with affine gaps (NCBI BLASTP defaults: open=11, extend=1).
	go, ge, INF = 11, 1, float('-inf')
	H  = np.zeros((m+1, n+1))
	E  = np.full((m+1, n+1), INF)
	F  = np.full((m+1, n+1), INF)
	tb = np.zeros((m+1, n+1), dtype=np.int8)
	# 2a. Precompute (m, n) substitution score matrix once, using BLOSUM62
	# for alphabet residues and (+4 self-match / -1 mismatch) fallback for
	# anything outside the 20-letter alphabet.
	BM   = np.array(_bm, dtype=float)
	idx1 = np.array([_idx.get(c, -1) for c in seq1], dtype=np.int64)
	idx2 = np.array([_idx.get(c, -1) for c in seq2], dtype=np.int64)
	valid = (idx1[:, None] >= 0) & (idx2[None, :] >= 0)
	S = np.where(valid,
		BM[np.clip(idx1[:, None], 0, 19),
		   np.clip(idx2[None, :], 0, 19)],
		0.0)
	arr1 = np.array(list(seq1))
	arr2 = np.array(list(seq2))
	eq   = arr1[:, None] == arr2[None, :]
	S    = np.where(valid, S, np.where(eq, 4.0, -1.0))
	best, bi, bj = 0.0, 0, 0
	for i in range(1, m+1):
		# 2b. Vectorised F column update: depends only on H[i-1], F[i-1].
		F[i, 1:] = np.maximum(H[i-1, 1:] - go - ge, F[i-1, 1:] - ge)
		for j in range(1, n+1):
			diag    = H[i-1, j-1] + S[i-1, j-1]
			E[i, j] = max(H[i, j-1] - go - ge, E[i, j-1] - ge)
			h       = max(0.0, diag, E[i, j], F[i, j])
			H[i, j] = h
			if h > best: best, bi, bj = h, i, j
			tb[i, j] = (0 if h == 0 else 1 if h == diag
				else 2 if h == F[i, j] else 3)
	if best == 0:
		raise Exception('No alignment found between the sequences')
	# 3. Traceback from the highest-scoring cell to recover aligned strings.
	aq, as_, i, j = [], [], bi, bj
	while i > 0 and j > 0 and H[i, j] > 0:
		t = int(tb[i, j])
		if t == 1:
			aq.append(seq1[i-1]); as_.append(seq2[j-1])
			i -= 1; j -= 1
		elif t == 2:
			aq.append(seq1[i-1]); as_.append('-'); i -= 1
		else:
			aq.append('-'); as_.append(seq2[j-1]); j -= 1
	aq, as_ = ''.join(reversed(aq)), ''.join(reversed(as_))
	# 4. Compute identity, similarity, gap statistics over the aligned region.
	qs, ss, aln_len = i + 1, j + 1, len(aq)
	n_id  = sum(1 for a, b in zip(aq, as_) if a == b and a != '-')
	n_pos = sum(1 for a, b in zip(aq, as_)
		if a != '-' and b != '-' and _blosum(a, b) > 0)
	n_gap = aq.count('-') + as_.count('-')
	pct     = round(n_id  / aln_len * 100, 2)
	pct_pos = round(n_pos / aln_len * 100, 1)
	pct_gap = round(n_gap / aln_len * 100, 1)
	# 5. Karlin-Altschul E-value (lam=0.270, K=0.041 for BLOSUM62 with gap 11/1).
	lam, K  = 0.270, 0.041
	e_value = K * m * n * math.exp(-lam * best)
	bits    = (lam * best - math.log(K)) / math.log(2)
	# 6. Build the per-column match-symbol line: | identical, + similar, ' ' otherwise.
	mid = ''.join(
		' ' if a == '-' or b == '-'
		else '|' if a == b
		else '+' if _blosum(a, b) > 0
		else ' '
		for a, b in zip(aq, as_))
	# 7. Format the header and stats lines of the BLAST-style report.
	out = [
		f'Query length={m}  Subject length={n}', '',
		(f'Score: {bits:.1f} bits ({int(best)}), '
			f'E-value: {e_value:.3e}'),
		(f'Identities: {n_id}/{aln_len} ({pct}%), '
			f'Positives: {n_pos}/{aln_len} ({pct_pos}%), '
			f'Gaps: {n_gap}/{aln_len} ({pct_gap}%)'), '']
	# 8. Emit 60-column aligned blocks with Query / midline / Sbjct tracks.
	qp, sp, w = qs, ss, 60
	for st in range(0, aln_len, w):
		bq, bm, bs = aq[st:st+w], mid[st:st+w], as_[st:st+w]
		qr, sr = len(bq) - bq.count('-'), len(bs) - bs.count('-')
		out += [
			f'Query  {qp:>6}  {bq}  {qp+qr-1}',
			f'       {"":>6}  {bm}',
			f'Sbjct  {sp:>6}  {bs}  {sp+sr-1}', '']
		qp += qr; sp += sr
	return '\n'.join(out), pct, e_value

def MSA(sequences):
	'''
	Progressive multiple sequence alignment (ClustalW-like) with BLOSUM62 and mean-field DCA
	Arguments:
	----------
		sequences : list[str] - FASTA sequences to align (at least 2 required)
	Return:
	-------
		str        : ClustalW-style formatted alignment text
		list[str]  : Gap-padded aligned sequences in input order
		list[float]: Per-column conservation score = 1 - H/log2(20), range [0, 1]
		list[float]: Per-column Shannon entropy in bits
		np.ndarray : PSSM of shape (L, 20) in AA order 'ARNDCQEGHILKMFPSTWYV'
		np.ndarray : APC-corrected mean-field DCA direct-information matrix (L, L)
	'''
	# 1. Validate input count and normalise sequences to uppercase.
	n = len(sequences)
	if n < 2:
		raise Exception('MSA requires at least 2 sequences')
	seqs   = [s.upper() for s in sequences]
	labels = [f'Seq{i+1}' for i in range(n)]
	go, ge, INF = 11, 1, float('-inf')
	# 2. Pairwise distances via BLAST (1 - pct/100, clipped to 1 on error).
	dist = np.zeros((n, n))
	for i in range(n):
		for j in range(i+1, n):
			try:
				_, pct, _ = BLAST(seqs[i], seqs[j])
				dd = 1.0 - pct / 100.0
			except Exception:
				dd = 1.0
			dist[i, j] = dist[j, i] = dd
	# 3. UPGMA guide tree: repeatedly merge closest active clusters.
	sizes  = {k: 1 for k in range(n)}
	active = list(range(n))
	d      = dist.copy()
	merge_order = []
	for _ in range(n - 1):
		bi, bj, best = -1, -1, float('inf')
		for x in range(len(active)):
			for y in range(x + 1, len(active)):
				ii, jj = active[x], active[y]
				if d[ii, jj] < best:
					best, bi, bj = d[ii, jj], ii, jj
		merge_order.append((bi, bj))
		ni, nj = sizes[bi], sizes[bj]
		for k in active:
			if k == bi or k == bj: continue
			d[bi, k] = d[k, bi] = (
				ni * d[bi, k] + nj * d[bj, k]) / (ni + nj)
		sizes[bi] += sizes[bj]
		active.remove(bj)
	# 4. Progressive profile-to-profile Needleman-Wunsch with affine gaps and BLOSUM62 column scoring.
	BM_aa = np.array(_bm, dtype=float)
	def _profile_freq(profile):
		'''Per-column frequency vector (L, 20) over the 20-letter
		BLOSUM alphabet, normalised by non-gap count. Residues outside
		the alphabet and gaps contribute zero weight.'''
		L = len(profile[0])
		F = np.zeros((L, 20))
		for row in profile:
			for ci, c in enumerate(row):
				k = _idx.get(c, -1)
				if k >= 0: F[ci, k] += 1
		denom = F.sum(axis=1, keepdims=True)
		with np.errstate(divide='ignore', invalid='ignore'):
			return np.divide(F, denom, where=(denom > 0),
				out=np.zeros_like(F))
	profiles = {k: [seqs[k]] for k in range(n)}
	for (ci, cj) in merge_order:
		p1, p2 = profiles[ci], profiles[cj]
		L1, L2 = len(p1[0]), len(p2[0])
		H  = np.zeros((L1+1, L2+1))
		E  = np.full((L1+1, L2+1), INF)
		F  = np.full((L1+1, L2+1), INF)
		tb = np.zeros((L1+1, L2+1), dtype=np.int8)
		for i in range(1, L1+1):
			H[i, 0], tb[i, 0] = -(go + ge * i), 2
		for j in range(1, L2+1):
			H[0, j], tb[0, j] = -(go + ge * j), 3
		# Precompute profile-profile BLOSUM column scores in one shot.
		Fp1 = _profile_freq(p1)
		Fp2 = _profile_freq(p2)
		with np.errstate(all='ignore'):
			CS = Fp1 @ BM_aa @ Fp2.T     # shape (L1, L2)
		for i in range(1, L1+1):
			# Vectorised F column update: depends only on H[i-1], F[i-1].
			F[i, 1:] = np.maximum(H[i-1, 1:] - go - ge, F[i-1, 1:] - ge)
			for j in range(1, L2+1):
				diag    = H[i-1, j-1] + CS[i-1, j-1]
				E[i, j] = max(H[i, j-1] - go - ge, E[i, j-1] - ge)
				h       = max(diag, E[i, j], F[i, j])
				H[i, j] = h
				tb[i, j] = (1 if h == diag
					else 2 if h == F[i, j] else 3)
		np1 = [[] for _ in p1]
		np2 = [[] for _ in p2]
		i, j = L1, L2
		while i > 0 or j > 0:
			if i == 0:
				for k in range(len(p1)): np1[k].append('-')
				for k, r in enumerate(p2): np2[k].append(r[j-1])
				j -= 1
			elif j == 0:
				for k, r in enumerate(p1): np1[k].append(r[i-1])
				for k in range(len(p2)): np2[k].append('-')
				i -= 1
			else:
				t = int(tb[i, j])
				if t == 1:
					for k, r in enumerate(p1): np1[k].append(r[i-1])
					for k, r in enumerate(p2): np2[k].append(r[j-1])
					i -= 1; j -= 1
				elif t == 2:
					for k, r in enumerate(p1): np1[k].append(r[i-1])
					for k in range(len(p2)): np2[k].append('-')
					i -= 1
				else:
					for k in range(len(p1)): np1[k].append('-')
					for k, r in enumerate(p2): np2[k].append(r[j-1])
					j -= 1
		a1 = [''.join(reversed(row)) for row in np1]
		a2 = [''.join(reversed(row)) for row in np2]
		profiles[ci] = a1 + a2
		del profiles[cj]
	final = list(profiles.values())[0]
	L   = len(final[0])
	lw  = max(max(len(lb) for lb in labels), 4)
	# 5. Per-column conservation symbol: * (all identical), : (all similar), . (mean>0), or space.
	con = []
	for ci in range(L):
		col = [final[k][ci] for k in range(n)]
		ng  = [c for c in col if c != '-']
		if not ng:
			con.append(' ')
		elif len(ng) == n and all(c == ng[0] for c in ng):
			con.append('*')
		else:
			pairs = [_blosum(a, b) for x, a in enumerate(ng)
				for b in ng[x+1:]]
			if not pairs:
				con.append('*' if len(ng) == 1 else ' ')
			elif all(s > 0 for s in pairs):
				con.append(':')
			elif sum(pairs) / len(pairs) > 0:
				con.append('.')
			else:
				con.append(' ')
	con = ''.join(con)
	# 6. ClustalW-style output block, 60 columns per block with running residue counts.
	out = [f'Multiple Sequence Alignment ({n} sequences, {L} columns)',
		'']
	pos, w = [0] * n, 60
	for st in range(0, L, w):
		for k, lb in enumerate(labels):
			blk = final[k][st:st+w]
			pos[k] += len(blk) - blk.count('-')
			out.append(f'{lb:<{lw}}  {blk}  {pos[k]}')
		out.append(f'{"":>{lw}}  {con[st:st+w]}')
		out.append('')
	# 7. Encode the MSA as an integer matrix (gap=0, AA=1..20) for downstream stats.
	alphabet = '-' + _aa
	q, B = len(alphabet), n
	a2i  = {c: i for i, c in enumerate(alphabet)}
	M = np.zeros((B, L), dtype=np.int8)
	for bi, s in enumerate(final):
		for ci, ch in enumerate(s):
			M[bi, ci] = a2i.get(ch, 0)
	# 8. Shannon entropy and normalised conservation (1 - H/log2(20)) per column.
	log2_20 = math.log2(20)
	entropy, conservation = [], []
	for ci in range(L):
		nz = M[:, ci][M[:, ci] != 0]
		if len(nz) == 0:
			entropy.append(0.0); conservation.append(0.0); continue
		p = np.bincount(nz, minlength=q)[1:] / len(nz)
		nzp = p[p > 0]
		Hc = float(-np.sum(nzp * np.log2(nzp)))
		entropy.append(round(Hc, 4))
		conservation.append(round(1.0 - Hc / log2_20, 4))
	# 9. Position-specific scoring matrix with Laplace pseudocount against uniform 1/20 background.
	pssm = np.zeros((L, 20), dtype=float)
	for ci in range(L):
		nz = M[:, ci][M[:, ci] != 0]
		counts = np.bincount(nz, minlength=q)[1:]
		pssm[ci] = np.log2((counts + 1.0) / (counts.sum() + 20.0) * 20.0)
	# 10. DCA sequence reweighting by identity clustering (theta=0.2, 80% similarity threshold).
	theta, weights = 0.2, np.ones(B)
	if B > 1:
		simthr = (1.0 - theta) * L
		eq_count = np.zeros(B)
		for a in range(B):
			for b in range(a, B):
				if a == b:
					eq_count[a] += 1; continue
				if int((M[a] == M[b]).sum()) >= simthr:
					eq_count[a] += 1; eq_count[b] += 1
		weights = 1.0 / eq_count
	Beff = float(weights.sum())
	# 11. Single-site and two-site frequencies (Beff-weighted) with lambda=0.5 pseudocount.
	Pi = np.zeros((L, q))
	for bi in range(B):
		for ci in range(L):
			Pi[ci, M[bi, ci]] += weights[bi]
	Pi /= Beff
	lam   = 0.5
	Pi_pc = (1.0 - lam) * Pi + lam / q
	def _pij_pc(i, j):
		'''On-demand q-by-q pair frequency with pseudocount and
		diagonal reset. Replaces the full (L,L,q,q) tensor.'''
		pij = np.zeros((q, q))
		np.add.at(pij, (M[:, i], M[:, j]), weights)
		pij /= Beff
		pij = (1.0 - lam) * pij + lam / (q * q)
		if i == j:
			pij[:] = 0.0
			for a in range(q):
				pij[a, a] = Pi_pc[i, a]
		return pij
	# 12. Covariance matrix with last state dropped as gauge, then invert (pseudo-inverse on failure).
	qm = q - 1
	C = np.zeros((L * qm, L * qm))
	for i in range(L):
		for j in range(L):
			pij = _pij_pc(i, j)
			block = (pij[:qm, :qm]
				- np.outer(Pi_pc[i, :qm], Pi_pc[j, :qm]))
			C[i*qm:(i+1)*qm, j*qm:(j+1)*qm] = block
	try:
		invC = np.linalg.inv(C)
	except np.linalg.LinAlgError:
		invC = np.linalg.pinv(C)
	# 13. Direct-information per residue pair via mean-field fixed-point (tolerance 1e-6, 100-iter cap).
	dca_raw = np.zeros((L, L))
	for i in range(L):
		for j in range(i + 1, L):
			W = np.ones((q, q))
			for a in range(qm):
				for b in range(qm):
					W[a, b] = math.exp(-invC[i*qm + a, j*qm + b])
			mu1, mu2 = np.ones(q) / q, np.ones(q) / q
			pi_i, pi_j = Pi_pc[i], Pi_pc[j]
			for _ in range(100):
				new_mu1 = pi_i / (mu2 @ W.T)
				new_mu2 = pi_j / (mu1 @ W)
				new_mu1 /= new_mu1.sum()
				new_mu2 /= new_mu2.sum()
				if (np.max(np.abs(new_mu1 - mu1)) < 1e-6
						and np.max(np.abs(new_mu2 - mu2)) < 1e-6):
					mu1, mu2 = new_mu1, new_mu2
					break
				mu1, mu2 = new_mu1, new_mu2
			Pdir  = W * np.outer(mu1, mu2)
			Pdir /= Pdir.sum()
			Pfac  = np.outer(pi_i, pi_j)
			mask  = (Pdir > 1e-12) & (Pfac > 1e-12)
			di = float(np.sum(
				Pdir[mask] * np.log(Pdir[mask] / Pfac[mask])))
			dca_raw[i, j] = dca_raw[j, i] = di
	# 14. Apply Average Product Correction (APC) to deflate phylogenetic and compositional bias.
	dca = np.zeros((L, L))
	if L > 1:
		row_mean   = dca_raw.sum(axis=1) / (L - 1)
		total_mean = dca_raw.sum() / (L * (L - 1))
		if total_mean > 0:
			for i in range(L):
				for j in range(L):
					if i == j: continue
					dca[i, j] = dca_raw[i, j] - (
						row_mean[i] * row_mean[j] / total_mean)
		else:
			dca = dca_raw.copy()
		np.fill_diagonal(dca, 0.0)
	return '\n'.join(out), final, conservation, entropy, pssm, dca

def Isoelectric(sequence):
	'''
	Isoelectric point (pI) of a protein via EMBOSS pKa and bisection on [0, 14]
	Arguments:
	----------
		sequence : str - Protein FASTA sequence (one-letter codes)
	Return:
	-------
		float : pH at which the protein has zero net charge, rounded to 2 decimals
	'''
	# 1. Validate input and uppercase the sequence.
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	# 2. Count titratable residues (EMBOSS pKa set: K/R/H positive, D/E/C/Y negative).
	nK, nR, nH = seq.count('K'), seq.count('R'), seq.count('H')
	nD, nE     = seq.count('D'), seq.count('E')
	nC, nY     = seq.count('C'), seq.count('Y')
	# 3. Bisect net charge on [0, 14] using pKa_NT=8.6, pKa_CT=3.6.
	lo, hi = 0.0, 14.0
	for _ in range(100):
		mid = (lo + hi) / 2.0
		pos = 1.0 / (1.0 + 10 ** (mid - 8.6))
		if nK: pos += nK / (1.0 + 10 ** (mid - 10.53))
		if nR: pos += nR / (1.0 + 10 ** (mid - 12.48))
		if nH: pos += nH / (1.0 + 10 ** (mid -  6.00))
		neg = 1.0 / (1.0 + 10 ** (3.6 - mid))
		if nD: neg += nD / (1.0 + 10 ** ( 3.65 - mid))
		if nE: neg += nE / (1.0 + 10 ** ( 4.25 - mid))
		if nC: neg += nC / (1.0 + 10 ** ( 8.33 - mid))
		if nY: neg += nY / (1.0 + 10 ** (10.07 - mid))
		c = pos - neg
		if abs(c) < 1e-4: break
		if c > 0: lo = mid
		else:     hi = mid
	return round(mid, 2)

def Hydrophobicity(sequence, window=9, scale='eisenberg'):
	'''
	Sliding-window hydrophobicity profile (ProtScale-style)
	Arguments:
	----------
		sequence : str - Protein FASTA sequence
		window   : int - Odd window size (default 9)
		scale    : str - Scale name: 'eisenberg', 'kyte-doolittle', 'hopp-woods', or 'engelman'
	Return:
	-------
		list[int]  : 0-indexed centre position of each window
		list[float]: Mean hydrophobicity score in each window, rounded to 3 decimals
	'''
	# 1. Declare the four supported ProtScale hydrophobicity tables.
	_HPHOB_SCALES = {
		'eisenberg': {
			'A': 0.620, 'R':-2.530, 'N':-0.780, 'D':-0.900, 'C': 0.290,
			'Q':-0.850, 'E':-0.740, 'G': 0.480, 'H':-0.400, 'I': 1.380,
			'L': 1.060, 'K':-1.500, 'M': 0.640, 'F': 1.190, 'P': 0.120,
			'S':-0.180, 'T':-0.050, 'W': 0.810, 'Y': 0.260, 'V': 1.080},
		'kyte-doolittle': {
			'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
			'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
			'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
			'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2},
		'hopp-woods': {
			'A':-0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C':-1.0,
			'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H':-0.5, 'I':-1.8,
			'L':-1.8, 'K': 3.0, 'M':-1.3, 'F':-2.5, 'P': 0.0,
			'S': 0.3, 'T':-0.4, 'W':-3.4, 'Y':-2.3, 'V':-1.5},
		'engelman': {
			'A': 1.6, 'R':-12.3,'N':-4.8, 'D':-9.2, 'C': 2.0,
			'Q':-4.1, 'E':-8.2, 'G': 1.0, 'H':-3.0, 'I': 3.1,
			'L': 2.8, 'K':-8.8, 'M': 3.4, 'F': 3.7, 'P':-0.2,
			'S': 0.6, 'T': 1.2, 'W': 1.9, 'Y':-0.7, 'V': 2.6}}
	# 2. Validate window size against sequence length and pick the scale table.
	seq, L = sequence.upper(), len(sequence)
	if window < 1: raise Exception('window must be >= 1')
	if window > L: raise Exception(
		f'window ({window}) larger than sequence ({L})')
	tbl = _HPHOB_SCALES.get(scale.lower())
	if tbl is None: raise Exception(
		f'Unknown scale {scale!r}; choose from '
		f'{list(_HPHOB_SCALES)}')
	# 3. Slide the window, emitting the centre position and mean score per window.
	half, n = (window - 1) // 2, L - window + 1
	return([i + half for i in range(n)],
		[round(sum(tbl.get(seq[i+k], 0.0)
			for k in range(window)) / window, 3) for i in range(n)])

def Aliphatic(sequence):
	'''
	Aliphatic index AI = X(A) + 2.9*X(V) + 3.9*(X(I) + X(L)) from mole percentages (Ikai 1980)
	Arguments:
	----------
		sequence : str - Protein FASTA sequence
	Return:
	-------
		float : Aliphatic index, rounded to 2 decimals
	'''
	# 1. Validate input and uppercase the sequence.
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	# 2. Compute mole percentages of aliphatic residues A, V, I, L.
	xA, xV, xI, xL = (100.0 * seq.count(a) / len(seq) for a in 'AVIL')
	# 3. Weighted sum per Ikai 1980.
	return round(xA + 2.9 * xV + 3.9 * (xI + xL), 2)

def ExtinctCoeff(sequence, reduced=True):
	'''
	Molar extinction coefficient at 280 nm via eps = nW*5500 + nY*1490 + (nC/2)*125 (Pace 1995)
	Arguments:
	----------
		sequence : str  - Protein FASTA sequence
		reduced  : bool - True (default) treats Cys as reduced (no contribution); False as cystines
	Return:
	-------
		int : Molar extinction coefficient in M^-1 cm^-1
	'''
	# 1. Validate input and uppercase the sequence.
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	# 2. Sum W and Y contributions; add C/2 contribution only when oxidised.
	eps = (seq.count('W') * 5500 + seq.count('Y') * 1490
		+ (0 if reduced else (seq.count('C') // 2) * 125))
	return int(round(eps))

# Guruprasad et al. (1990) DIWV dipeptide instability table.
# Rows = first residue, cols = second residue (BLOSUM62 order).
_DIWV_AA = 'ARNDCQEGHILKMFPSTWYV'
_DIWV = {
	'A': {'A':1.0,'R':1.0,'N':1.0,'D':-7.49,'C':44.94,'Q':1.0,
		'E':1.0,'G':1.0,'H':-7.49,'I':1.0,'L':1.0,'K':1.0,
		'M':1.0,'F':1.0,'P':20.26,'S':1.0,'T':1.0,'W':1.0,
		'Y':1.0,'V':1.0},
	'R': {'A':1.0,'R':58.28,'N':13.34,'D':1.0,'C':1.0,'Q':20.26,
		'E':1.0,'G':-7.49,'H':20.26,'I':1.0,'L':1.0,'K':1.0,
		'M':1.0,'F':1.0,'P':20.26,'S':44.94,'T':1.0,'W':58.28,
		'Y':-6.54,'V':1.0},
	'N': {'A':1.0,'R':1.0,'N':1.0,'D':1.0,'C':-1.88,'Q':-6.54,
		'E':1.0,'G':-14.03,'H':1.0,'I':44.94,'L':1.0,'K':24.68,
		'M':1.0,'F':-14.03,'P':-1.88,'S':1.0,'T':-7.49,'W':-9.37,
		'Y':1.0,'V':-1.88},
	'D': {'A':1.0,'R':-6.54,'N':1.0,'D':1.0,'C':1.0,'Q':1.0,
		'E':1.0,'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':-7.49,
		'M':1.0,'F':-6.54,'P':1.0,'S':20.26,'T':-14.03,'W':1.0,
		'Y':1.0,'V':1.0},
	'C': {'A':1.0,'R':1.0,'N':1.0,'D':20.26,'C':1.0,'Q':-6.54,
		'E':1.0,'G':1.0,'H':33.60,'I':1.0,'L':20.26,'K':1.0,
		'M':33.60,'F':1.0,'P':20.26,'S':1.0,'T':33.60,'W':24.68,
		'Y':1.0,'V':-6.54},
	'Q': {'A':1.0,'R':1.0,'N':1.0,'D':20.26,'C':-6.54,'Q':20.26,
		'E':20.26,'G':1.0,'H':1.0,'L':1.0,'I':1.0,'K':1.0,
		'M':1.0,'F':-6.54,'P':20.26,'S':44.94,'T':1.0,'W':1.0,
		'Y':-6.54,'V':-6.54},
	'E': {'A':1.0,'R':1.0,'N':1.0,'D':20.26,'C':44.94,'Q':20.26,
		'E':33.60,'G':1.0,'H':-6.54,'I':20.26,'L':1.0,'K':1.0,
		'M':1.0,'F':1.0,'P':20.26,'S':20.26,'T':1.0,'W':-14.03,
		'Y':1.0,'V':1.0},
	'G': {'A':-7.49,'R':-7.49,'N':-7.49,'D':1.0,'C':1.0,'Q':1.0,
		'E':-6.54,'G':13.34,'H':1.0,'I':-7.49,'L':1.0,'K':-7.49,
		'M':1.0,'F':1.0,'P':1.0,'S':1.0,'T':-7.49,'W':13.34,
		'Y':-7.49,'V':1.0},
	'H': {'A':1.0,'R':1.0,'N':24.68,'D':1.0,'C':1.0,'Q':1.0,
		'E':1.0,'G':-9.37,'H':1.0,'I':44.94,'L':1.0,'K':24.68,
		'M':1.0,'F':-9.37,'P':-1.88,'S':1.0,'T':-6.54,'W':-1.88,
		'Y':44.94,'V':1.0},
	'I': {'A':1.0,'R':1.0,'N':1.0,'D':1.0,'C':1.0,'Q':1.0,
		'E':44.94,'G':1.0,'H':13.34,'I':1.0,'L':20.26,'K':-7.49,
		'M':1.0,'F':1.0,'P':-1.88,'S':1.0,'T':1.0,'W':1.0,
		'Y':1.0,'V':-7.49},
	'L': {'A':1.0,'R':20.26,'N':1.0,'D':1.0,'C':1.0,'Q':33.60,
		'E':1.0,'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':-7.49,
		'M':1.0,'F':1.0,'P':20.26,'S':1.0,'T':1.0,'W':24.68,
		'Y':1.0,'V':1.0},
	'K': {'A':1.0,'R':33.60,'N':1.0,'D':1.0,'C':1.0,'Q':24.68,
		'E':1.0,'G':-7.49,'H':1.0,'I':-7.49,'L':-7.49,'K':1.0,
		'M':33.60,'F':1.0,'P':-6.54,'S':1.0,'T':1.0,'W':1.0,
		'Y':1.0,'V':-7.49},
	'M': {'A':13.34,'R':-6.54,'N':1.0,'D':1.0,'C':1.0,'Q':-6.54,
		'E':1.0,'G':1.0,'H':58.28,'I':1.0,'L':1.0,'K':1.0,
		'M':-1.88,'F':1.0,'P':44.94,'S':44.94,'T':-1.88,'W':1.0,
		'Y':24.68,'V':1.0},
	'F': {'A':1.0,'R':1.0,'N':1.0,'D':13.34,'C':1.0,'Q':1.0,
		'E':1.0,'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':-14.03,
		'M':1.0,'F':1.0,'P':20.26,'S':1.0,'T':1.0,'W':1.0,
		'Y':33.60,'V':1.0},
	'P': {'A':20.26,'R':-6.54,'N':1.0,'D':-6.54,'C':-6.54,'Q':20.26,
		'E':18.38,'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':1.0,
		'M':-6.54,'F':20.26,'P':20.26,'S':20.26,'T':1.0,'W':-1.88,
		'Y':1.0,'V':20.26},
	'S': {'A':1.0,'R':20.26,'N':1.0,'D':1.0,'C':33.60,'Q':20.26,
		'E':20.26,'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':1.0,
		'M':1.0,'F':1.0,'P':44.94,'S':20.26,'T':1.0,'W':1.0,
		'Y':1.0,'V':1.0},
	'T': {'A':1.0,'R':1.0,'N':-14.03,'D':1.0,'C':1.0,'Q':-6.54,
		'E':20.26,'G':-7.49,'H':1.0,'I':1.0,'L':1.0,'K':1.0,
		'M':1.0,'F':13.34,'P':1.0,'S':1.0,'T':1.0,'W':-14.03,
		'Y':1.0,'V':1.0},
	'W': {'A':-14.03,'R':1.0,'N':13.34,'D':1.0,'C':1.0,'Q':1.0,
		'E':1.0,'G':-9.37,'H':24.68,'I':1.0,'L':13.34,'K':1.0,
		'M':24.68,'F':1.0,'P':1.0,'S':1.0,'T':-14.03,'W':1.0,
		'Y':1.0,'V':-7.49},
	'Y': {'A':24.68,'R':-15.91,'N':1.0,'D':24.68,'C':1.0,'Q':1.0,
		'E':-6.54,'G':-7.49,'H':13.34,'I':1.0,'L':1.0,'K':1.0,
		'M':44.94,'F':1.0,'P':13.34,'S':1.0,'T':-7.49,'W':-9.37,
		'Y':13.34,'V':1.0},
	'V': {'A':1.0,'R':1.0,'N':1.0,'D':-14.03,'C':1.0,'Q':1.0,
		'E':1.0,'G':-7.49,'H':1.0,'I':1.0,'L':1.0,'K':-1.88,
		'M':1.0,'F':1.0,'P':20.26,'S':1.0,'T':-7.49,'W':1.0,
		'Y':-6.54,'V':1.0}}

def Instability(sequence):
	'''
	Instability index II = (10/L)*sum DIWV(seq[i], seq[i+1]); <40 suggests stable (Guruprasad 1990)
	Arguments:
	----------
		sequence : str - Protein FASTA sequence
	Return:
	-------
		float : Instability index, rounded to 2 decimals; 0.0 for single-residue input
	'''
	# 1. Validate input, uppercase, and short-circuit on single-residue sequences.
	if not sequence: raise Exception('Empty sequence')
	seq, L = sequence.upper(), len(sequence)
	if L < 2: return 0.0
	# 2. Sum DIWV dipeptide values across the sequence; unknown dipeptides contribute 0.
	total = sum(_DIWV.get(seq[i], {}).get(seq[i+1], 0) for i in range(L - 1))
	# 3. Normalise by length and scale by 10.
	return round(10.0 * total / L, 2)

# Kyte-Doolittle hydropathy (used by GRAVY)
_KD = {
	'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
	'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
	'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
	'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2}

def GRAVY(sequence):
	'''
	Grand average of hydropathy (mean Kyte-Doolittle hydropathy, Kyte & Doolittle 1982)
	Arguments:
	----------
		sequence : str - Protein FASTA sequence
	Return:
	-------
		float : Mean Kyte-Doolittle hydropathy, rounded to 3 decimals
	'''
	# 1. Validate input.
	if not sequence: raise Exception('Empty sequence')
	# 2. Mean KD hydropathy over the uppercased sequence (unknown residues contribute 0).
	return round(sum(_KD.get(a, 0.0)
		for a in sequence.upper()) / len(sequence), 3)

def Split(pose, chain=None, start=None, end=None):
	'''
	Extract a slice of a Pose (by chain or residue range) into a new densely-renumbered Pose
	Arguments:
	----------
		pose  : Pose - Source protein, DNA, or RNA pose
		chain : str  - Chain ID to extract (mutually exclusive with start/end)
		start : int  - First residue index to keep (inclusive, zero-based)
		end   : int  - Last residue index to keep (inclusive, zero-based)
	Return:
	-------
		Pose : New pose with atoms, residues, bonds, and coordinates renumbered from zero
	'''
	# 1. Import Pose locally to avoid a circular import at module load time.
	try:    from .pose import Pose
	except ImportError: from pose import Pose
	# 2. Reject ambiguous arg combos: exactly one of chain= or (start=, end=) must be given.
	if (chain is None) == (start is None and end is None):
		raise Exception("Split requires either chain= OR (start=, end=)")
	# 3. Resolve molecule type and fetch the residue table.
	mol = pose.data.get('Type')
	if mol is None: raise Exception('Source pose is empty')
	is_pro = (mol == 'Protein')
	rk  = 'Amino Acids' if is_pro else 'Nucleotides'
	src = pose.data[rk]
	if not src: raise Exception(f'Source pose has no {rk}')
	all_idx = sorted(src.keys())
	# 4. Select the residue indices to retain, based on chain or range mode.
	if chain is not None:
		keep_res = [i for i in all_idx if src[i][1] == chain]
		if not keep_res:
			raise Exception(f'Chain {chain!r} not in pose')
	else:
		if start is None or end is None:
			raise Exception('Split needs both start and end for range mode')
		if start > end:
			raise Exception(f'start ({start}) > end ({end})')
		keep_res = [i for i in all_idx if start <= i <= end]
		if not keep_res:
			raise Exception(f'Range [{start}, {end}] selects no residues')
	# 5. Collect kept atom indices and build dense remaps for atoms and residues.
	keep_atoms = sorted({ai for ri in keep_res
		for ai in src[ri][2] + src[ri][3]})
	a_remap = {old: new for new, old in enumerate(keep_atoms)}
	r_remap = {old: new for new, old in enumerate(keep_res)}
	src_atoms, src_bonds, src_co = (pose.data['Atoms'],
		pose.data['Bonds'], pose.data['Coordinates'])
	# 6. Build the new pose's data skeleton with remapped atoms, bonds, and coordinates.
	new = Pose()
	new.data = {
		'Type':        mol,  'Energy': 0, 'Rg': 0, 'Mass': 0,
		'Size':        {},   'FASTA':  {}, 'SS': {},
		'Nucleotides': None if is_pro else {},
		'Amino Acids': {} if is_pro else None,
		'Atoms':       {a_remap[o]: list(src_atoms[o])
			for o in keep_atoms},
		'Bonds':       {a_remap[o]: sorted(a_remap[ob]
			for ob in src_bonds.get(o, []) if ob in a_remap)
			for o in keep_atoms},
		'Coordinates': np.array([src_co[o] for o in keep_atoms],
			dtype=float) if keep_atoms else np.zeros((0, 3))}
	# 7. Copy residue rows with atom lists translated to the new indices.
	tgt = new.data[rk]
	for old_ri in keep_res:
		row = list(src[old_ri])
		row[2] = [a_remap[a] for a in row[2] if a in a_remap]
		row[3] = [a_remap[a] for a in row[3] if a in a_remap]
		tgt[r_remap[old_ri]] = row
	# 8. Refresh derived fields (Size, FASTA, SS, Mass, Rg) and return.
	new._update()
	return new

def Concatenate(pose1, pose2, fuse=False):
	'''
	Combine two poses of the same Type by chain-appending or by rebuilding a fused polymer
	Arguments:
	----------
		pose1 : Pose - First pose (protein, DNA, or RNA)
		pose2 : Pose - Second pose; must share Type with pose1
		fuse  : bool - False appends pose2 chains (colliding IDs renamed); True rebuilds as one polymer
	Return:
	-------
		Pose : New combined pose; fuse=True discards original coordinates and idealises geometry
	'''
	# 1. Import Pose locally to avoid a circular import at module load time.
	try:    from .pose import Pose
	except ImportError: from pose import Pose
	# 2. Validate both poses are non-empty and share the same molecule type.
	t1, t2 = pose1.data.get('Type'), pose2.data.get('Type')
	if t1 is None or t2 is None:
		raise Exception('Concatenate: empty pose given')
	if t1 != t2:
		raise Exception(f'Cannot concatenate {t1} with {t2}')
	is_pro = (t1 == 'Protein')
	rk     = 'Amino Acids' if is_pro else 'Nucleotides'
	# 3. Fuse mode: rebuild a single idealised polymer from the concatenated FASTA.
	if fuse:
		f1, f2 = pose1.data['FASTA'], pose2.data['FASTA']
		new = Pose()
		new.Build(''.join(f1[c] for c in sorted(f1))
			+ ''.join(f2[c] for c in sorted(f2)), fmt=t1)
		return new
	# 4. Append mode: initialise the new pose's data skeleton.
	new = Pose()
	new.data = {
		'Type': t1, 'Energy': 0, 'Rg': 0, 'Mass': 0,
		'Size': {}, 'FASTA': {}, 'SS': {},
		'Nucleotides': None if is_pro else {},
		'Amino Acids': {} if is_pro else None,
		'Atoms': {}, 'Bonds': {}, 'Coordinates': np.zeros((0, 3))}
	# 5. Two-pass copy: pose1 first, then compute pose2 chain-collision remap, then pose2.
	coords_all = []
	ai_off, ri_off = 0, 0
	ch_remap = {}
	for step, src_pose in enumerate((pose1, pose2)):
		if step == 1:
			taken = {v[1] for v in new.data[rk].values()}
			for c in sorted({v[1] for v in pose2.data[rk].values()}):
				if c not in taken:
					taken.add(c); continue
				for cand in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
					if cand not in taken:
						taken.add(cand); ch_remap[c] = cand; break
				else:
					raise Exception('Ran out of chain letters')
		src_at, src_bd = src_pose.data['Atoms'], src_pose.data['Bonds']
		src_co, src_aa = src_pose.data['Coordinates'], src_pose.data[rk]
		old_a = sorted(src_at.keys())
		a_map = {oa: ai_off + i for i, oa in enumerate(old_a)}
		for oa in old_a:
			new.data['Atoms'][a_map[oa]] = list(src_at[oa])
			coords_all.append(src_co[oa])
		for oa in old_a:
			new.data['Bonds'][a_map[oa]] = sorted(
				a_map[ob] for ob in src_bd.get(oa, []) if ob in a_map)
		old_r = sorted(src_aa.keys())
		for i, ori in enumerate(old_r):
			row = list(src_aa[ori])
			row[1] = ch_remap.get(row[1], row[1])
			row[2] = [a_map[a] for a in row[2] if a in a_map]
			row[3] = [a_map[a] for a in row[3] if a in a_map]
			new.data[rk][ri_off + i] = row
		ai_off += len(old_a)
		ri_off += len(old_r)
	# 6. Finalise coordinates array and refresh derived fields.
	new.data['Coordinates'] = (np.array(coords_all, dtype=float)
		if coords_all else np.zeros((0, 3)))
	new._update()
	return new

def _has_hairpin(seq, cmap, min_stem=4, min_loop=3):
	'''True if seq has an internal inverted repeat of length >= min_stem
	separated by a loop of >= min_loop nt. Crude proxy for hairpin stability.'''
	L = len(seq)
	for stem in range(min_stem, L // 2 + 1):
		for i in range(L - 2 * stem - min_loop + 1):
			target = seq[i:i+stem][::-1].translate(cmap)
			j = seq.find(target, i + stem + min_loop)
			if j != -1:
				return True
	return False

def _has_3p_selfdimer(seq, cmap, window=5):
	'''True if the 3' last `window` bases of seq pair elsewhere in seq
	(ignoring the trivial self-overlap at the end).'''
	tail_rc = seq[-window:][::-1].translate(cmap)
	hit = seq.find(tail_rc)
	return hit != -1 and hit + window <= len(seq) - 1

def _has_cross_dimer(fwd, rev, cmap, window=5):
	'''True if the 3' tail of either primer pairs with the other primer.'''
	a = fwd[-window:][::-1].translate(cmap)
	b = rev[-window:][::-1].translate(cmap)
	return (a in rev) or (b in fwd)

def PCR(dna_sequence):
	'''
	Design forward and reverse PCR primers for a DNA template via a 5-tier relaxation search
	Arguments:
	----------
		dna_sequence : str - Template DNA sequence (A/C/G/T only, length >= 36 bp)
	Return:
	-------
		str : Forward primer (5' end of template)
		str : Reverse primer (reverse complement of 3' end of template)
		str : Suboptimal-tier warning, or None if the Ideal tier was satisfied
	'''
	# 1. Validate: uppercase, reject illegal bases, require at least 36 bp template.
	seq = dna_sequence.upper()
	for ch in seq:
		if ch not in 'ACGT':
			raise Exception(f'Illegal base {ch!r} in template')
	if len(seq) < 36:
		raise Exception('Template too short for primer design (<36 bp)')
	# 2. Reverse-complement the template via str.translate (replaces _revcomp helper).
	cmap = str.maketrans('ACGTN', 'TGCAN')
	rc = seq[::-1].translate(cmap)
	# 3. SantaLucia 1998 nearest-neighbor thermodynamics (dH kcal/mol, dS cal/mol/K).
	DH = {'AA':-7.9,'TT':-7.9,'AT':-7.2,'TA':-7.2,
		'CA':-8.5,'TG':-8.5,'GT':-8.4,'AC':-8.4,
		'CT':-7.8,'AG':-7.8,'GA':-8.2,'TC':-8.2,
		'CG':-10.6,'GC':-9.8,'GG':-8.0,'CC':-8.0}
	DS = {'AA':-22.2,'TT':-22.2,'AT':-20.4,'TA':-21.3,
		'CA':-22.7,'TG':-22.7,'GT':-22.4,'AC':-22.4,
		'CT':-21.0,'AG':-21.0,'GA':-22.2,'TC':-22.2,
		'CG':-27.2,'GC':-24.4,'GG':-19.9,'CC':-19.9}
	# 4. Relaxation tier table: Ideal -> Good -> Fair -> Poor -> Last resort.
	tiers = [
		{'label':'Ideal',      'len':(18,25),'gc':(40.0,60.0),
			'tm':(55.0,65.0),'clamp':True, 'max_run':4,
			'no_hairpin':True, 'no_dimer':True,
			'no_cross_dimer':True, 'dtm':2.0},
		{'label':'Good',       'len':(18,28),'gc':(35.0,65.0),
			'tm':(50.0,68.0),'clamp':True, 'max_run':5,
			'no_hairpin':True, 'no_dimer':True,
			'no_cross_dimer':True, 'dtm':3.0},
		{'label':'Fair',       'len':(18,30),'gc':(25.0,75.0),
			'tm':(45.0,72.0),'clamp':False,'max_run':5,
			'no_hairpin':False,'no_dimer':True,
			'no_cross_dimer':False,'dtm':5.0},
		{'label':'Poor',       'len':(18,30),'gc':None,'tm':None,
			'clamp':False,'max_run':None,
			'no_hairpin':False,'no_dimer':False,
			'no_cross_dimer':False,'dtm':8.0},
		{'label':'Last resort','len':(18,30),'gc':None,'tm':None,
			'clamp':False,'max_run':None,
			'no_hairpin':False,'no_dimer':False,
			'no_cross_dimer':False,'dtm':float('inf')}]
	# 5. Walk tiers from strict to permissive, building candidate pools and pairing primers.
	max_off = max(0, min(60, len(seq) - 18))
	chosen, chosen_tier = None, None
	for ti, tier in enumerate(tiers):
		# 5a. Build fwd_pool (from seq) and rev_pool (from rc); all helper logic inlined.
		fwd_pool, rev_pool = [], []
		lo, hi = tier['len']
		for source, pool in ((seq, fwd_pool), (rc, rev_pool)):
			for off in range(max_off + 1):
				region = source[off:]
				for L in range(lo, hi + 1):
					if L > len(region): continue
					cand = region[:L]
					if tier['clamp'] and cand[-1] not in 'GC':
						continue
					gc = 100.0 * (cand.count('G') + cand.count('C')) / L
					if tier['gc'] is not None:
						glo, ghi = tier['gc']
						if not (glo <= gc <= ghi): continue
					mr = tier['max_run']
					if mr is not None and any(
							b * mr in cand for b in 'ACGT'):
						continue
					if tier['no_hairpin'] and _has_hairpin(cand, cmap):
						continue
					if tier['no_dimer'] and _has_3p_selfdimer(cand, cmap):
						continue
					# 5b. Tm via SantaLucia 1998 with Owczarzy 2004 salt correction.
					dH = dS = 0.0
					for i in range(L - 1):
						nn = cand[i:i+2]
						dH += DH.get(nn, 0.0)
						dS += DS.get(nn, 0.0)
					if cand[0]  in 'GC': dH += 0.1; dS += -2.8
					else:                dH += 2.3; dS +=  4.1
					if cand[-1] in 'GC': dH += 0.1; dS += -2.8
					else:                dH += 2.3; dS +=  4.1
					dS_salt = dS + 0.368 * (L - 1) * math.log(0.05)
					tm = ((dH * 1000.0) / (dS_salt
						+ 1.987 * math.log(250e-9 / 4.0))) - 273.15
					if tier['tm'] is not None:
						tlo, thi = tier['tm']
						if not (tlo <= tm <= thi): continue
					pool.append((off, cand, tm, gc))
		if not fwd_pool or not rev_pool: continue
		# 5c. Score every fwd/rev pair under the tier's dTm gate; keep the best.
		best, best_score = None, float('inf')
		dtm_max = tier['dtm']
		no_xd = tier['no_cross_dimer']
		for off1, fwd, tmf, gcf in fwd_pool:
			for off2, rev, tmr, gcr in rev_pool:
				dT = abs(tmf - tmr)
				if dT > dtm_max: continue
				if no_xd and _has_cross_dimer(fwd, rev, cmap): continue
				score = (dT * 5.0 + abs(tmf - 60.0) + abs(tmr - 60.0)
					+ abs(gcf - 50.0) * 0.1 + abs(gcr - 50.0) * 0.1
					+ (off1 + off2) * 0.05)
				if score < best_score:
					best_score = score
					best = (fwd, rev, tmf, tmr, gcf, gcr)
		if best is not None:
			chosen, chosen_tier = best, ti
			break
	if chosen is None:
		raise Exception('No primer pair found even at last-resort tier')
	# 6. Build a suboptimal-tier warning message if the Ideal tier failed.
	fwd, rev, tmf, tmr, gcf, gcr = chosen
	msg = None
	if chosen_tier > 0:
		reasons = []
		if not (40.0 <= gcf <= 60.0 and 40.0 <= gcr <= 60.0):
			reasons.append('GC% outside 40-60')
		if not (55.0 <= tmf <= 65.0 and 55.0 <= tmr <= 65.0):
			reasons.append('Tm outside 55-65 \u00b0C')
		if abs(tmf - tmr) > 2.0:
			reasons.append('|\u0394Tm| > 2 \u00b0C')
		if fwd[-1] not in 'GC' or rev[-1] not in 'GC':
			reasons.append('GC clamp missing')
		if _has_cross_dimer(fwd, rev, cmap):
			reasons.append('primer-pair cross-dimer')
		reason = '; '.join(reasons) if reasons else 'gates relaxed'
		msg = (f'Warning: Suboptimal PCR primers '
			f'({tiers[chosen_tier]["label"]} tier) \u2014 {reason}')
	return (fwd, rev, msg)

def Translate(sequence, fmt='protein', organism='ecoli'):
	'''
	Translate between DNA, RNA, and protein with auto-detected source alphabet
	Arguments:
	----------
		sequence : str - Input sequence (alphabet auto-detected: DNA, RNA, or protein)
		fmt      : str - Target alphabet: 'protein' (default), 'dna', or 'rna'
		organism : str - Codon usage for back-translation: 'ecoli' (default) or 'human'
	Return:
	-------
		str : Translated sequence (uppercase, with gaps and spaces stripped)
	'''
	# 1. Validate input and target format.
	if not sequence: raise Exception('Empty sequence')
	tgt = fmt.lower()
	if tgt not in ('protein', 'dna', 'rna'):
		raise Exception(f'Unknown target fmt: {fmt}')
	# 2. Detect source alphabet by character set (gap, *, N excluded from the test).
	chars = set(sequence.upper()) - {'-', '*', 'N'}
	if not chars: src = 'protein'
	elif chars <= set('ACGT'): src = 'dna'
	elif chars <= set('ACGU'): src = 'rna'
	elif chars <= set('ACDEFGHIKLMNPQRSTVWY'): src = 'protein'
	elif chars - set('ACGT') - set('ACGU'): src = 'protein'
	else: src = 'dna'
	# 3. Normalise: uppercase, strip gaps and spaces.
	s = sequence.upper().replace('-', '').replace(' ', '')
	# 4. Identity and DNA<->RNA alphabet swaps.
	if src == tgt: return s
	if src == 'dna' and tgt == 'rna': return s.replace('T', 'U')
	if src == 'rna' and tgt == 'dna': return s.replace('U', 'T')
	# 5. Nucleotide -> protein via the standard genetic code ('*' = stop, unknown codons -> 'X').
	if src in ('dna', 'rna') and tgt == 'protein':
		CODON = {
			'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
			'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
			'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
			'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
			'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
			'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
			'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
			'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
			'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
			'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
			'AAT':'N','AAC':'N','AAA':'K','AAG':'K',
			'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
			'TGT':'C','TGC':'C','TGA':'*','TGG':'W',
			'CGT':'R','CGC':'R','CGA':'R','CGG':'R',
			'AGT':'S','AGC':'S','AGA':'R','AGG':'R',
			'GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
		dna = s.replace('U', 'T')
		dna = dna[:len(dna) - len(dna) % 3]
		return ''.join(CODON.get(dna[i:i+3], 'X')
			for i in range(0, len(dna), 3))
	# 6. Protein -> nucleotide via highest-weight Kazusa codon per amino acid.
	if src == 'protein' and tgt in ('dna', 'rna'):
		BEST = {
			'ecoli': {'F':'TTT','L':'CTG','I':'ATT','M':'ATG','V':'GTG',
				'S':'AGC','P':'CCG','T':'ACC','A':'GCG','Y':'TAT',
				'*':'TAA','H':'CAT','Q':'CAG','N':'AAC','K':'AAA',
				'D':'GAT','E':'GAA','C':'TGC','W':'TGG','R':'CGC','G':'GGC'},
			'human': {'F':'TTC','L':'CTG','I':'ATC','M':'ATG','V':'GTG',
				'S':'AGC','P':'CCC','T':'ACC','A':'GCC','Y':'TAC',
				'*':'TGA','H':'CAC','Q':'CAG','N':'AAC','K':'AAG',
				'D':'GAC','E':'GAG','C':'TGC','W':'TGG','R':'CGG','G':'GGC'}
			}.get(organism.lower())
		if BEST is None:
			raise Exception(
				f"Unknown organism {organism!r}; use 'ecoli' or 'human'")
		out = []
		for aa in s:
			c = BEST.get(aa)
			if c is None:
				raise Exception(f'No codon for residue {aa!r}')
			out.append(c)
		dna = ''.join(out)
		return dna if tgt == 'dna' else dna.replace('T', 'U')
	raise Exception(f'Unsupported translation {src} -> {tgt}')

def PROSITE(sequence, pattern):
	'''
	Search a protein sequence for a PROSITE-style pattern (subset grammar: [..], {..}, x(n,m), < >)
	Arguments:
	----------
		sequence : str - Protein sequence to search
		pattern  : str - PROSITE pattern using literals, [ABC], {ABC}, x, x(n), x(n,m), < anchors >
	Return:
	-------
		list[tuple] : Each hit is (start, end, match) with 1-based inclusive positions
	'''
	# 1. Validate: empty pattern is fatal; empty sequence yields no hits.
	if not pattern: raise Exception('Empty pattern')
	if not sequence: return []
	# 2. Tokenise PROSITE pattern into regex: [..]/{..} -> char classes, x -> '.', < > -> ^ $.
	p = pattern.replace('-', '').replace(' ', '')
	out, i = [], 0
	while i < len(p):
		c = p[i]
		if   c == '<':       out.append('^'); i += 1
		elif c == '>':       out.append('$'); i += 1
		elif c in '[{':
			close = ']' if c == '[' else '}'
			j = p.find(close, i)
			if j == -1:
				raise Exception(f'Unclosed {c} in pattern')
			out.append(('[' if c == '[' else '[^') + p[i+1:j] + ']')
			i = j + 1
		elif c in 'xX':      out.append('.'); i += 1
		elif c.isalpha():    out.append(c.upper()); i += 1
		else:
			raise Exception(
				f'Unexpected character {c!r} at position {i} of pattern')
		if i < len(p) and p[i] == '(':
			j = p.find(')', i)
			if j == -1: raise Exception('Unclosed ( in pattern')
			body = p[i+1:j]
			out.append('{' + ','.join(
				s.strip() for s in body.split(',', 1)) + '}')
			i = j + 1
	# 3. Compile as a zero-width lookahead so overlapping hits are found; scan the sequence.
	rx = re.compile('(?=(' + ''.join(out) + '))', re.IGNORECASE)
	return [(m.start() + 1, m.start() + len(m.group(1)), m.group(1))
		for m in rx.finditer(sequence)]

def HydrogenBondMap(pose):
	'''
	Backbone H-bond donor/acceptor map via the DSSP electrostatic criterion (Kabsch-Sander 1983)
	Arguments:
	----------
		pose : Pose - Protein pose with backbone N, C, O atoms
	Return:
	-------
		np.ndarray : (N_atoms, N_atoms) int8 matrix; 0 = no bond, 1 = donor N, 2 = acceptor O
	'''
	# 1. Validate molecule type and presence of amino-acid residues.
	if pose.data.get('Type') != 'Protein':
		raise Exception('HydrogenBondMap only supports protein poses')
	AAs = pose.data.get('Amino Acids') or {}
	if not AAs:
		raise Exception('Pose has no amino acids')
	# 2. Allocate output matrix and gather residue indices, chain IDs, and tricodes.
	atoms = pose.data['Atoms']
	co    = pose.data['Coordinates']
	N_atoms = max(atoms.keys()) + 1 if atoms else 0
	M = np.zeros((N_atoms, N_atoms), dtype=np.int8)
	res_idx  = sorted(AAs.keys())
	N_res    = len(res_idx)
	chains   = [AAs[r][1] for r in res_idx]
	tricodes = [AAs[r][5].upper() for r in res_idx]
	# 3. Precompute per-residue backbone atom-name -> atom-index lookup.
	ai_of = {r: {atoms[ai][0]: ai for ai in AAs[r][2]} for r in res_idx}
	# 4. Place virtual amide-H: use explicit H/1H when available, else N + unit(C_{i-1}->O_{i-1}).
	H_pos = [None] * N_res
	for k, r in enumerate(res_idx):
		if tricodes[k] == 'PRO': continue
		if k == 0 or chains[k] != chains[k-1]: continue
		idx = ai_of[r]
		ah = idx.get('H', idx.get('1H'))
		if ah is not None:
			H_pos[k] = co[ah]
			continue
		prev = ai_of[res_idx[k-1]]
		if 'N' in idx and 'C' in prev and 'O' in prev:
			cdir = co[prev['C']] - co[prev['O']]
			nm = float(np.linalg.norm(cdir))
			if nm > 0.001:
				H_pos[k] = co[idx['N']] + cdir / nm
	# 5. For every (i, j) pair on the same chain with |i-j|>1, apply DSSP energy threshold E<-2.092.
	for ki in range(N_res):
		if H_pos[ki] is None: continue
		Ni_idx = ai_of[res_idx[ki]].get('N', -1)
		if Ni_idx < 0: continue
		Ni, Hi = co[Ni_idx], H_pos[ki]
		for kj in range(N_res):
			if abs(ki - kj) <= 1 or chains[ki] != chains[kj]: continue
			idxj = ai_of[res_idx[kj]]
			if 'O' not in idxj or 'C' not in idxj: continue
			Cj_idx, Oj_idx = idxj['C'], idxj['O']
			Cj, Oj = co[Cj_idx], co[Oj_idx]
			r_ON = float(np.linalg.norm(Oj - Ni))
			r_CH = float(np.linalg.norm(Cj - Hi))
			r_OH = float(np.linalg.norm(Oj - Hi))
			r_CN = float(np.linalg.norm(Cj - Ni))
			if min(r_ON, r_CH, r_OH, r_CN) < 0.001: continue
			if 0.084*(1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 1389.35458 < -2.092:
				M[Ni_idx, Oj_idx] = 1
				M[Oj_idx, Ni_idx] = 2
	return M

def ContactMap(pose):
	'''
	Residue-residue Euclidean distance map (angstroms) using CA for protein, C1' for DNA/RNA
	Arguments:
	----------
		pose : Pose - Protein or nucleic-acid pose with a non-empty residue table
	Return:
	-------
		np.ndarray : (N_residues, N_residues) pairwise distances, zero on the diagonal
	'''
	# 1. Resolve molecule type, residue table, and reference-atom name.
	mol = pose.data.get('Type')
	if mol is None: raise Exception('Empty pose')
	if   mol == 'Protein':      src, ref = pose.data['Amino Acids'], 'CA'
	elif mol in ('DNA', 'RNA'): src, ref = pose.data['Nucleotides'], "C1'"
	else: raise Exception(f'Unknown molecule type: {mol}')
	if not src: raise Exception('Pose has no residues')
	# 2. Gather reference-atom coordinates for every residue (must exist per residue).
	atoms, co = pose.data['Atoms'], pose.data['Coordinates']
	keys = sorted(src.keys())
	pts  = np.zeros((len(keys), 3))
	for k, ri in enumerate(keys):
		pos = next((co[ai] for ai in src[ri][2]
			if atoms[ai][0] == ref), None)
		if pos is None:
			raise Exception(f'Residue {ri} has no {ref} atom')
		pts[k] = pos
	# 3. Broadcast pairwise difference, take Euclidean norm, zero the diagonal.
	diff = pts[:, None, :] - pts[None, :, :]
	mat  = np.sqrt((diff * diff).sum(-1))
	np.fill_diagonal(mat, 0.0)
	return mat

def _rotlib_lookup(rotlib_root, three_letter, phi_deg, psi_deg):
	'''
	Slice the Rotamer Library CSR table for one (residue, phi, psi) cell
	Arguments:
	----------
		rotlib_root: dict - database['Rotamer Library']
		three_letter: str - 3-letter residue code (uppercase, L-form)
		phi_deg, psi_deg: float - backbone angles in degrees
	Returns:
	--------
		(n_chi, table_slice) where table_slice is a list of rotamer rows;
		(0, []) if the residue has no entry, or rows is empty if the cell
		has no rotamers.
	'''
	residues = rotlib_root.get('residues', {}) if rotlib_root else {}
	entry = residues.get(three_letter)
	if entry is None: return 0, []
	phi_start = float(rotlib_root.get('phi_start', -180.0))
	phi_step  = float(rotlib_root.get('phi_step',   10.0))
	phi_n     = int  (rotlib_root.get('phi_n',     36))
	psi_start = float(rotlib_root.get('psi_start', -180.0))
	psi_step  = float(rotlib_root.get('psi_step',   10.0))
	psi_n     = int  (rotlib_root.get('psi_n',     36))
	rot         = entry['rotamers']
	bin_offsets = rot['bin_offsets']
	i_phi = int(math.floor((phi_deg - phi_start) / phi_step)) % phi_n
	i_psi = int(math.floor((psi_deg - psi_start) / psi_step)) % psi_n
	bidx  = i_phi * psi_n + i_psi
	start = bin_offsets[bidx]
	end   = bin_offsets[bidx + 1]
	return int(entry['n_chi']), rot['table'][start:end]

def Rotamers(index, pose):
	'''
	Single-amino-acid rotamer packer: set every chi of one residue to the
	dominant (most-populated) rotamer from the Rotamer Library at that
	residue's current backbone (phi, psi).

	Algorithm (production):
	  1. Look up the residue's 3-letter code; bail out if it has no chis.
	  2. Read backbone phi, psi; bail out if either is undefined (chain end).
	  3. Snap to the nearest (phi, psi) grid cell in the Rotamer Library.
	  4. Pick the rotamer k* with maximum P_k in that cell.
	  5. Apply mu_k*_chi[c] for c = 1..n_chi via pose.RotateDihedral.

	D-amino acids (lowercase 1-letter codes) are handled via the standard
	chi/backbone mirror trick: lookup with negated phi/psi, negate predicted
	mu values when applying.
	Arguments:
	----------
		index : int - residue index in pose.data['Amino Acids']
		pose  : Pose - protein pose with a non-empty residue table
	Return:
	-------
		None - mutates the pose in place. No-op if the residue has no chis,
		undefined backbone, or no rotamer-library entry for its type.
	'''
	info  = pose.data.get('Amino Acids', {}).get(index)
	if info is None: return
	c     = info[0]
	aa_u  = c.upper()
	aa_db = pose.aminoacids.get(aa_u, {})
	chi_atoms = aa_db.get('Chi Angle Atoms') or []
	if not chi_atoms: return                # Gly, Ala -- no chis
	three = (aa_db.get('Tricode') or [None])[0]
	if not three: return
	phi = pose.GetDihedral(index, 'PHI')
	psi = pose.GetDihedral(index, 'PSI')
	if math.isnan(phi) or math.isnan(psi): return  # chain ends
	flip = (c != aa_u)
	phi_q = -phi if flip else phi
	psi_q = -psi if flip else psi
	rotlib = DBLoad().get('Rotamer Library')
	n_chi, rows = _rotlib_lookup(rotlib, three, phi_q, psi_q)
	if n_chi == 0 or not rows: return
	# Find argmax_k by P_k. Column layout: [count, prob, chi1..N, sig1..N]
	prob_i = 1
	chi_i  = 2
	best   = max(rows, key=lambda row: row[prob_i])
	for ci in range(n_chi):
		mu = best[chi_i + ci]
		if flip: mu = -mu
		pose.RotateDihedral(index, float(mu), 'CHI', ci + 1)

def Minimise(pose, ff=None, max_steps=500, ftol=1.0, dt_fs=0.5,
		dt_max_fs=1.0, step_max=0.2, etol=1e-6, stall_k=10, box=None):
	'''
	Relax pose coordinates with FIRE2 (Guenole et al. 2020). A
	trust-region cap bounds the per-atom displacement; a step that turns
	non-finite or strongly uphill is rejected, dt shrunk and retried;
	and the lowest-|force| frame ever seen is restored before returning,
	so a force-field singularity (e.g. an uncovered atom with no bond)
	can never fling atoms away or corrupt the returned structure
	Arguments:
	----------
		pose:      Pose - molecule source protein, DNA, RNA, or Molecule
		ff:        ForceField - reusable evaluator; created if None
		max_steps: int - maximum number of FIRE2 iterations
		ftol:      float - convergence on max|force| (L_inf) in kJ/mol/A
		dt_fs:     float - initial integrator step in femtoseconds
		dt_max_fs: float - upper bound on the adaptive step in fs
		step_max:  float - trust-region cap on per-atom displacement in A
		etol:      float - energy-stall tolerance in kJ/mol
		stall_k:   int - consecutive stalled steps that trigger early stop
		box:       None for no PBC; (3,) orthorhombic; (3, 3) triclinic
	Returns:
	--------
		tuple: (float, dict) - energy of the best frame in kJ/mol and a
		per-step log ('energies', 'fmax', 'max_step', 'converged',
		'n_steps')
	'''
	if ff is None: ff = ForceField()
	N_MIN, F_INC, F_DEC = 5, 1.1, 0.5
	A_START, F_ALPHA = 0.1, 0.99
	AKMA_FS = 23.91888086
	atoms = pose.data['Atoms']
	m = np.array([pose.masses[atoms[i][1]] for i in sorted(atoms)],
		dtype=np.float64)[:, None]
	v = np.zeros_like(pose.data['Coordinates'], dtype=np.float64)
	dt     = float(dt_fs) / AKMA_FS
	dt_max = float(dt_max_fs) / AKMA_FS
	dt_min = dt * 1e-3
	alpha, n_pos = float(A_START), 0
	energies, fmaxes, max_steps_log = [], [], []
	E, F = ff(pose, grad=True, box=box)
	E = float(E)
	best_fmax   = float(np.max(np.abs(F)))
	best_coords = pose.data['Coordinates'].copy()
	converged, steps_done, stall = False, 0, 0
	for step in range(int(max_steps)):
		fmax = float(np.max(np.abs(F)))
		energies.append(E); fmaxes.append(fmax)
		steps_done = step + 1
		# Remember the lowest-|force| frame; it is restored at the end.
		if np.isfinite(fmax) and fmax < best_fmax:
			best_fmax   = fmax
			best_coords = pose.data['Coordinates'].copy()
		if fmax < ftol or stall >= stall_k:
			converged = True
			break
		# Stop a clear divergence early (a FF singularity); the best
		# frame is restored below, so the returned pose stays intact.
		if (not np.isfinite(fmax)) or (fmax > 1e4
				and fmax > 1e3 * best_fmax):
			break
		# FIRE: mix the velocity toward the force when the power
		# P = F . v is positive (downhill); zero it and shrink dt on
		# an uphill power.
		P = float(np.sum(F * v))
		if P > 0.0:
			f_norm = float(np.linalg.norm(F))
			v_norm = float(np.linalg.norm(v))
			mix = (alpha * v_norm / f_norm) if f_norm > 1e-12 else 0.0
			v = (1.0 - alpha) * v + mix * F
			n_pos += 1
			if n_pos > N_MIN:
				dt = min(dt * F_INC, dt_max)
				alpha *= F_ALPHA
		else:
			v = np.zeros_like(v)
			dt = max(dt * F_DEC, dt_min)
			alpha, n_pos = A_START, 0
		# Semi-implicit Euler step; clamp the per-atom DISPLACEMENT
		# (not the velocity, which carries FIRE's persistent momentum).
		v = v + dt * F / m
		dr = dt * v
		nrm = np.linalg.norm(dr, axis=1, keepdims=True)
		dr = dr * np.minimum(1.0, step_max / np.maximum(nrm, 1e-12))
		max_steps_log.append(float(np.max(np.abs(dr))))
		x_old = pose.data['Coordinates']
		pose.data['Coordinates'] = x_old + dr
		E_new, F_new = ff(pose, grad=True, box=box)
		E_new = float(E_new)
		fmax_new = float(np.max(np.abs(F_new)))
		# Safeguard: undo a step that is non-finite, strongly uphill, or
		# whose force explodes (a downhill run into a FF singularity);
		# zero the velocity and shrink dt so the retry is smaller.
		bad = (not np.isfinite(E_new)
			or not np.isfinite(F_new).all()
			or E_new > E + 1.0 + 0.05 * abs(E)
			or (fmax_new > 1e3 and fmax_new > 100.0 * max(fmax, 1.0)))
		if bad:
			pose.data['Coordinates'] = x_old
			v = np.zeros_like(v)
			dt = max(dt * F_DEC, dt_min)
			alpha, n_pos = A_START, 0
			continue
		stall = stall + 1 if abs(E_new - E) < etol else 0
		E, F = E_new, F_new
	# Restore the lowest-|force| frame and report its energy.
	pose.data['Coordinates'] = best_coords
	E, F = ff(pose, grad=True, box=box)
	log = {
		'energies':  np.asarray(energies,      dtype=np.float64),
		'fmax':      np.asarray(fmaxes,        dtype=np.float64),
		'max_step':  np.asarray(max_steps_log, dtype=np.float64),
		'converged': bool(converged),
		'n_steps':   int(steps_done)}
	return float(E), log

def Anneal(pose, ff=None, n_steps=10000, T_start=2000.0, T_end=10.0,
		sigma_small=5.0, sigma_large=30.0, p_large=0.2, p_shear=0.5,
		target_acc=0.30, adapt_window=100, seed=None, box=None):
	'''
	Simulated annealing with shear+single moves and adaptive small sigma
	Arguments:
	----------
		pose:         Pose - protein pose with Amino Acids dict
		ff:           ForceField - reusable evaluator; created if None
		n_steps:      int - total Metropolis steps in the cooling schedule
		T_start:      float - starting temperature in Kelvin
		T_end:        float - final temperature in Kelvin
		sigma_small:  float - initial small-move std-dev in degrees
		sigma_large:  float - large-move std-dev in degrees (fixed)
		p_large:      float - probability of choosing a large move
		p_shear:      float - probability of choosing a shear move
		target_acc:   float - target acceptance ratio for small moves
		adapt_window: int - small moves between sigma_small updates
		seed:         int or None - RNG seed for reproducibility
		box:          None for no PBC; (3,) ortho; (3, 3) triclinic
	Returns:
	--------
		tuple: (float, dict) - best energy seen and per-step log
	'''
	if ff is None: ff = ForceField()
	if pose.data.get('Amino Acids') is None:
		raise ValueError('Anneal requires a protein pose with Amino Acids')
	GAIN, SIGMA_MIN, SIGMA_MAX = 0.5, 0.5, 60.0
	rng = np.random.default_rng(seed)
	res_ids = np.array(sorted(pose.data['Amino Acids']), dtype=np.int64)
	n_res = len(res_ids)
	kB = 8.31446262e-3
	T_arr = T_start * (T_end / T_start) ** (
		np.arange(n_steps) / max(n_steps - 1, 1))
	res_arr   = res_ids[rng.integers(0, n_res, size=n_steps)]
	kind_arr  = np.where(rng.integers(0, 2, size=n_steps) == 0, 'PHI', 'PSI')
	shear_arr = rng.random(size=n_steps) < p_shear
	large_arr = rng.random(size=n_steps) < p_large
	noise_arr = rng.standard_normal(size=n_steps)
	uni_arr   = rng.random(size=n_steps)
	def try_single(res, kind, delta):
		theta_old = pose.GetDihedral(res, kind)
		if math.isnan(theta_old): return False
		pose.RotateDihedral(res, theta_old + delta, kind)
		return True
	def try_shear(res, delta):
		psi_old = pose.GetDihedral(res, 'PSI')
		phi_next = pose.GetDihedral(res + 1, 'PHI') \
			if (res + 1) in pose.data['Amino Acids'] else float('nan')
		if math.isnan(psi_old) or math.isnan(phi_next): return False
		pose.RotateDihedral(res, psi_old + delta, 'PSI')
		pose.RotateDihedral(res + 1, phi_next - delta, 'PHI')
		return True
	E_curr = float(ff(pose, grad=False, box=box))
	E_best = E_curr
	coords_best = pose.data['Coordinates'].copy()
	energies   = np.empty(n_steps, dtype=np.float64)
	accepted   = np.zeros(n_steps, dtype=bool)
	move_types = np.full(n_steps, 2, dtype=np.int8)  # 0=single,1=shear,2=invalid
	sigma_history = [float(sigma_small)]
	small_count, small_acc, best_step = 0, 0, 0
	for s in range(int(n_steps)):
		sigma = sigma_large if large_arr[s] else sigma_small
		delta = float(noise_arr[s] * sigma)
		res = int(res_arr[s])
		coords_old = pose.data['Coordinates'].copy()
		applied = (try_shear(res, delta) if shear_arr[s]
			else try_single(res, str(kind_arr[s]), delta))
		mtype = 1 if shear_arr[s] else 0
		if not applied and shear_arr[s]:
			applied = try_single(res, str(kind_arr[s]), delta)
			mtype = 0
		if not applied:
			energies[s] = E_curr
			continue
		E_new = float(ff(pose, grad=False, box=box))
		dE = E_new - E_curr
		RT = kB * float(T_arr[s])
		boltz = math.exp(-dE / RT) if (dE > 0.0 and RT > 0.0) else 1.0
		accept = (dE <= 0.0) or (uni_arr[s] < boltz)
		move_types[s] = mtype
		if accept:
			E_curr = E_new
			accepted[s] = True
			if E_curr < E_best:
				E_best = E_curr
				coords_best = pose.data['Coordinates'].copy()
				best_step = s
		else:
			pose.data['Coordinates'] = coords_old
		energies[s] = E_curr
		if not large_arr[s]:
			small_count += 1
			small_acc += int(accept)
			if small_count >= adapt_window:
				rate = small_acc / small_count
				sigma_small *= math.exp(GAIN * (rate - target_acc))
				sigma_small = max(SIGMA_MIN, min(sigma_small, SIGMA_MAX))
				sigma_history.append(float(sigma_small))
				small_count, small_acc = 0, 0
	pose.data['Coordinates'] = coords_best
	log = {
		'energies':      energies,
		'temperatures':  T_arr,
		'accepted':      accepted,
		'move_types':    move_types,
		'sigma_history': np.asarray(sigma_history, dtype=np.float64),
		'best_step':     int(best_step)}
	return float(E_best), log

def Pack(pose, score=None, n_steps=2000, T_start=10.0, T_end=0.1,
		patience=400, seed=None):
	'''
	Sidechain repacking via simulated annealing on the full Rotamer Library
	ensemble at each residue's current backbone (phi, psi).

	Algorithm (production):
	  1. For each residue with chis and a defined (phi, psi), build the static
	     candidate set = list of (mu_chi_tuple, prob) from the rotamer library
	     cell at that residue's backbone.
	  2. Initialise from the pose's current chi configuration and score it.
	  3. SA loop with geometric cooling T = T_start * (T_end/T_start)^(t/N):
	     - pick a random repackable residue
	     - propose one of its rotamers k weighted by prob (so dominant
	       rotamers are explored more often, but rare ones remain reachable)
	     - apply trial chis; rescore
	     - accept if dE <= 0 or random() < exp(-dE / T); else revert
	     - track best-so-far
	  4. Early-exit if no accepted move in `patience` consecutive steps.
	  5. Restore best-found configuration; return its energy.

	D-amino acids: looked up against the L-form table with mirrored phi/psi,
	mu values negated when applied (same convention as Rotamers / _rotamer_prior).

	Arguments:
	----------
		pose:    Pose - protein pose with Amino Acids dict
		score:   Score - scoring function; defaults to Score('Default')
		n_steps: int - max number of SA proposals
		T_start: float - initial temperature (in score units, typically kJ/mol)
		T_end:   float - final temperature
		patience:int - early-exit if no acceptance in this many consecutive steps
		seed:    int or None - RNG seed for reproducibility
	Returns:
	--------
		tuple: (E_best, log) where log contains 'energies', 'temperatures',
		       'accepts', 'best_E', 'steps_run', 'converged', 'n_residues'.
	'''
	if score is None:
		from .energy import Score
		score = Score()
	if pose.data.get('Amino Acids') is None:
		raise ValueError('Pack requires a protein pose with Amino Acids')
	rng = np.random.default_rng(seed)
	rotlib = DBLoad().get('Rotamer Library')
	# Step 1: build candidate sets per repackable residue.
	# Each entry: (mus (K, n_chi), probs (K,) normalised, n_chi)
	candidates = {}
	for r, info in sorted(pose.data['Amino Acids'].items()):
		c = info[0]
		aa_u = c.upper()
		aa_db = pose.aminoacids.get(aa_u, {})
		chi_atoms = aa_db.get('Chi Angle Atoms') or []
		if not chi_atoms: continue
		three = (aa_db.get('Tricode') or [None])[0]
		if not three: continue
		phi = pose.GetDihedral(r, 'PHI')
		psi = pose.GetDihedral(r, 'PSI')
		if math.isnan(phi) or math.isnan(psi): continue
		flip = (c != aa_u)
		phi_q = -phi if flip else phi
		psi_q = -psi if flip else psi
		n_chi, rows = _rotlib_lookup(rotlib, three, phi_q, psi_q)
		if n_chi == 0 or not rows: continue
		# Column layout: [count, prob, chi1..N, sig1..N]
		prob_i = 1
		chi_i  = 2
		K = len(rows)
		mus   = np.empty((K, n_chi), dtype=np.float64)
		probs = np.empty(K,          dtype=np.float64)
		for k, row in enumerate(rows):
			probs[k] = max(float(row[prob_i]), 0.0)
			for ci in range(n_chi):
				m = float(row[chi_i + ci])
				mus[k, ci] = -m if flip else m
		s = probs.sum()
		if s <= 0.0: continue
		probs /= s
		candidates[r] = (mus, probs, n_chi)
	if not candidates:
		E0 = float(score(pose))
		return E0, {
			'energies': np.array([E0]), 'temperatures': np.array([T_start]),
			'accepts': np.array([], dtype=bool), 'best_E': E0, 'steps_run': 0,
			'converged': True, 'n_residues': 0}
	res_ids = list(candidates.keys())
	# Step 2: initial energy + best-state snapshot.
	def _snapshot():
		return {r: tuple(pose.GetDihedral(r, 'CHI', chi_type=ci+1)
			for ci in range(candidates[r][2])) for r in res_ids}
	def _restore(snap):
		for r, chis in snap.items():
			n_chi = candidates[r][2]
			for ci in range(n_chi):
				pose.RotateDihedral(r, float(chis[ci]), 'CHI', ci+1)
	E_curr  = float(score(pose))
	E_best  = E_curr
	best_state = _snapshot()
	# Step 3: SA loop.
	N = max(1, int(n_steps))
	energies     = np.empty(N, dtype=np.float64)
	temperatures = np.empty(N, dtype=np.float64)
	accepts      = np.empty(N, dtype=bool)
	last_accept  = 0
	step         = 0
	for step in range(N):
		T = T_start * (T_end / T_start) ** (step / max(1, N - 1))
		# Pick residue uniformly among repackable.
		r = res_ids[int(rng.integers(0, len(res_ids)))]
		mus, probs, n_chi = candidates[r]
		# Sample rotamer k weighted by prob.
		k = int(rng.choice(len(probs), p=probs))
		# Snapshot current chis for revert.
		snap = tuple(pose.GetDihedral(r, 'CHI', chi_type=ci+1)
			for ci in range(n_chi))
		# Apply trial.
		for ci in range(n_chi):
			pose.RotateDihedral(r, float(mus[k, ci]), 'CHI', ci+1)
		E_trial = float(score(pose))
		dE = E_trial - E_curr
		if dE <= 0.0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
			E_curr = E_trial
			last_accept = step
			accepts[step] = True
			if E_curr < E_best:
				E_best = E_curr
				best_state = _snapshot()
		else:
			# Revert.
			for ci in range(n_chi):
				pose.RotateDihedral(r, float(snap[ci]), 'CHI', ci+1)
			accepts[step] = False
		energies[step]     = E_curr
		temperatures[step] = T
		# Step 4: early-exit on stagnation.
		if step - last_accept >= patience: break
	steps_run = step + 1
	# Step 5: restore best-found state.
	_restore(best_state)
	E_final = float(score(pose))
	# Sanity: best_state may slightly differ from E_best due to caching; trust E_final.
	log = {
		'energies':     energies[:steps_run],
		'temperatures': temperatures[:steps_run],
		'accepts':      accepts[:steps_run],
		'best_E':       float(E_best),
		'steps_run':    int(steps_run),
		'converged':    bool(steps_run < N),
		'n_residues':   len(res_ids)}
	return E_final, log

def MolecularDynamics(pose, ff=None, n_steps=1000, dt_fs=2.0, T=300.0,
		thermostat='nve', friction_ps=1.0, constraints='hbonds',
		shake_tol=1e-8, shake_max=100, seed=None,
		trajectory_every=0, box=None):
	'''
	Velocity-Verlet NVE or BAOAB Langevin NVT MD with SHAKE/RATTLE
	Arguments:
	----------
		pose:             Pose - molecule source pose
		ff:               ForceField - reusable evaluator; created if None
		n_steps:          int - number of integration steps
		dt_fs:            float - integration step in femtoseconds
		T:                float - temperature in Kelvin (initial + bath)
		thermostat:       str - 'nve' or 'langevin'
		friction_ps:      float - Langevin friction in ps^-1
		constraints:      str - 'hbonds' constrains every X-H bond; 'none'
		shake_tol:        float - relative tolerance on |d^2 - r0^2|/r0^2
		shake_max:        int - max iterations for SHAKE/RATTLE projection
		seed:             int or None - RNG seed for reproducibility
		trajectory_every: int - snapshot stride; 0 disables snapshots
		box:              None for no PBC; (3,) ortho; (3, 3) triclinic
	Returns:
	--------
		tuple: (float, dict) - final potential energy and trajectory log
	'''
	if ff is None: ff = ForceField()
	if thermostat not in ('nve', 'langevin'):
		raise ValueError("thermostat must be 'nve' or 'langevin'")
	if constraints not in ('hbonds', 'none'):
		raise ValueError("constraints must be 'hbonds' or 'none'")
	rng = np.random.default_rng(seed)
	atoms = pose.data['Atoms']
	sorted_ids = sorted(atoms)
	m = np.array([pose.masses[atoms[i][1]] for i in sorted_ids],
		dtype=np.float64)
	n = len(m)
	m_col = m[:, None]
	inv_m = 1.0 / m
	inv_m_col = inv_m[:, None]
	AKMA_FS = 23.91888086
	kB = 8.31446262e-3
	dt = float(dt_fs) / AKMA_FS
	gamma = float(friction_ps) * AKMA_FS / 1000.0
	c1 = math.exp(-gamma * dt)
	c2 = np.sqrt((1.0 - c1 * c1) * kB * float(T) / m)[:, None]
	if ff._cache is None or ff._cache_hash != ff._topologyhash(pose):
		ff._prepare(pose)
	cache = ff._cache
	is_h = np.array([atoms[i][1] == 'H' for i in sorted_ids], dtype=bool)
	if constraints == 'hbonds' and len(cache['pairs']):
		cmask = is_h[cache['pairs'][:, 0]] | is_h[cache['pairs'][:, 1]]
		con = cache['pairs'][cmask]
		r0  = cache['bond_r0'][cmask]
	else:
		con = np.empty((0, 2), dtype=np.int64)
		r0  = np.empty((0,),   dtype=np.float64)
	K = len(con)
	i_c, j_c = con[:, 0], con[:, 1]
	r0sq = r0 * r0
	inv_red = inv_m[i_c] + inv_m[j_c] if K else np.empty(0)
	r0sq_max = float(r0sq.max()) if K else 1.0
	def shake(x_new, x_old, vel, dt_eff):
		if K == 0: return
		r_old = x_old[i_c] - x_old[j_c]
		for _ in range(int(shake_max)):
			r = x_new[i_c] - x_new[j_c]
			d2 = np.einsum('ij,ij->i', r, r)
			sigma = d2 - r0sq
			if float(np.max(np.abs(sigma))) < shake_tol * r0sq_max:
				return
			rdot = np.einsum('ij,ij->i', r, r_old)
			lam  = sigma / (2.0 * inv_red * rdot)
			delta = lam[:, None] * r_old
			np.add.at(x_new, i_c, -delta * inv_m_col[i_c])
			np.add.at(x_new, j_c,  delta * inv_m_col[j_c])
			np.add.at(vel,   i_c, -(delta / dt_eff) * inv_m_col[i_c])
			np.add.at(vel,   j_c,  (delta / dt_eff) * inv_m_col[j_c])
	def rattle(x, vel):
		if K == 0: return
		for _ in range(int(shake_max)):
			r = x[i_c] - x[j_c]
			v_rel = vel[i_c] - vel[j_c]
			rv = np.einsum('ij,ij->i', r, v_rel)
			d2 = np.einsum('ij,ij->i', r, r)
			if float(np.max(np.abs(rv))) < shake_tol * r0sq_max:
				return
			mu = rv / (d2 * inv_red)
			delta_v = mu[:, None] * r
			np.add.at(vel, i_c, -delta_v * inv_m_col[i_c])
			np.add.at(vel, j_c,  delta_v * inv_m_col[j_c])
	sigma_v = np.sqrt(kB * float(T) / m)[:, None]
	v = rng.standard_normal(size=(n, 3)) * sigma_v
	v -= ((m_col * v).sum(axis=0) / m.sum())[None, :]
	rattle(pose.data['Coordinates'], v)
	E, F = ff(pose, grad=True, box=box)
	dof = max(3 * n - K - 3, 1)
	energies = np.empty(int(n_steps), dtype=np.float64)
	kinetics = np.empty(int(n_steps), dtype=np.float64)
	temps    = np.empty(int(n_steps), dtype=np.float64)
	frames = []
	use_langevin = (thermostat == 'langevin')
	for step in range(int(n_steps)):
		if use_langevin:
			v += 0.5 * dt * F / m_col
			x_old = pose.data['Coordinates'].copy()
			pose.data['Coordinates'] = x_old + 0.5 * dt * v
			shake(pose.data['Coordinates'], x_old, v, 0.5 * dt)
			v = c1 * v + c2 * rng.standard_normal(size=(n, 3))
			rattle(pose.data['Coordinates'], v)
			x_old = pose.data['Coordinates'].copy()
			pose.data['Coordinates'] = x_old + 0.5 * dt * v
			shake(pose.data['Coordinates'], x_old, v, 0.5 * dt)
			E, F = ff(pose, grad=True, box=box)
			v += 0.5 * dt * F / m_col
			rattle(pose.data['Coordinates'], v)
		else:
			v += 0.5 * dt * F / m_col
			x_old = pose.data['Coordinates'].copy()
			pose.data['Coordinates'] = x_old + dt * v
			shake(pose.data['Coordinates'], x_old, v, dt)
			E, F = ff(pose, grad=True, box=box)
			v += 0.5 * dt * F / m_col
			rattle(pose.data['Coordinates'], v)
		KE = 0.5 * float(np.sum(m_col * v * v))
		energies[step] = float(E)
		kinetics[step] = KE
		temps[step] = 2.0 * KE / (dof * kB)
		if trajectory_every > 0 and (step + 1) % trajectory_every == 0:
			frames.append(pose.data['Coordinates'].copy())
	log = {
		'energies':     energies,
		'kinetic':      kinetics,
		'temperatures': temps,
		'frames':       frames,
		'n_constraints': int(K),
		'dof':           int(dof)}
	return float(E), log

def Port(name='openff'):
	'''
	Port one force field into database.json
	Arguments:
	----------
		name:   str - which force field to port; 'openff', 'ff19SB' or
			'charmm36', matched case-insensitively
	Returns:
	--------
		bool: True on success; raises on download / parse / write failure
	'''
	key     = str(name).upper()
	here    = os.path.dirname(os.path.abspath(__file__))
	db_path = os.path.join(here, 'database.json')
	import gzip
	def download(url):
		'''
		Fetch the text of a pinned GitHub raw URL
		Arguments:
		----------
			url: str - a raw.githubusercontent.com URL on a fixed commit
		Returns:
		--------
			str: the decoded file contents
		'''
		print(f'[port] downloading {url}', file=sys.stderr)
		try:
			with urllib.request.urlopen(url, timeout=120) as resp:
				return resp.read().decode('utf-8')
		except Exception as err:
			raise RuntimeError(f'port: could not download {url}: {err}')
	def cidof(rec, i):
		'''
		Read the namespaced atom identifier at slot i of a bonded record
		Arguments:
		----------
			rec: dict - the XML element's attributes
			i:   int  - 1-based slot index
		Returns:
		--------
			str: the class/type identifier, '' for an unset wildcard slot
		'''
		v = rec.get('class%d' % i)
		if v is None: v = rec.get('type%d' % i)
		return v if v is not None else ''
	def qval(qstr, target):
		'''
		Convert a SMIRNOFF quantity string to a target unit
		Arguments:
		----------
			qstr:   str - a quantity, e.g. '1.5 * angstrom ** 1'
			target: str - the desired unit expression, e.g.
				'kilojoule_per_mole * angstrom ** -2'
		Returns:
		--------
			float: the magnitude of qstr expressed in the target unit
		'''
		units = {
			'angstrom':             (1.0,           {'L': 1}),
			'nanometer':            (10.0,          {'L': 1}),
			'degree':               (1.0,           {'A': 1}),
			'radian':               (180.0/math.pi, {'A': 1}),
			'mole':                 (1.0,           {'N': 1}),
			'kilojoule':            (1.0,           {'E': 1}),
			'kilocalorie':          (4.184,         {'E': 1}),
			'kilojoule_per_mole':   (1.0,           {'E': 1, 'N': -1}),
			'kilocalorie_per_mole': (4.184,         {'E': 1, 'N': -1}),
			'elementary_charge':    (1.0,           {'Q': 1})}
		def reduce(text):
			'''Reduce to (number, {unit: power}, {dimension: power}).'''
			num, powers, dims = 1.0, {}, {}
			for tok in text.strip().replace('**', '^').split('*'):
				tok = tok.strip()
				if not tok: continue
				if '^' in tok:
					nm, _, ex = tok.partition('^')
					nm, ex = nm.strip(), int(ex.strip())
				else:
					nm, ex = tok, 1
				try:
					num *= float(nm) ** ex
					continue
				except ValueError:
					pass
				if nm not in units:
					raise ValueError(
						f'port: unknown unit {nm!r} in {text!r}')
				powers[nm] = powers.get(nm, 0) + ex
				for k, v in units[nm][1].items():
					dims[k] = dims.get(k, 0) + v * ex
			return num, powers, {k: v for k, v in dims.items() if v}
		nq, pq, dq = reduce(qstr)
		nt, pt, dt = reduce(target)
		if dq != dt:
			raise ValueError(
				f'port: cannot convert {qstr!r} to {target!r} '
				f'(dimension mismatch)')
		# Cancel units common to both sides before multiplying, so that a
		# shared factor such as radian ** -2 contributes no rounding.
		value = nq / nt
		for nm in set(pq) | set(pt):
			ex = pq.get(nm, 0) - pt.get(nm, 0)
			if ex: value *= units[nm][0] ** ex
		return value
	def converttorsions(section):
		'''
		Convert a SMIRNOFF torsion section to Pose's component schema
		Arguments:
		----------
			section: xml.etree Element - a <ProperTorsions> or
				<ImproperTorsions> SMIRNOFF section
		Returns:
		--------
			dict: {SMIRKS: {id, components: [{n, phi_0, K_phi, idivf}]}}
		'''
		out = {}
		for p in section:
			a = p.attrib
			comps, i = [], 1
			while ('k%d' % i) in a:
				idivf = a.get('idivf%d' % i)
				comps.append({
					'n':     int(a['periodicity%d' % i]),
					'phi_0': qval(a['phase%d' % i], 'degree'),
					'K_phi': qval(a['k%d' % i], 'kilojoule_per_mole'),
					'idivf': float(idivf) if idivf is not None else 1.0})
				i += 1
			out[a['smirks']] = {'id': a.get('id'), 'components': comps}
		return out
	def charmmtypes(root):
		'''
		Rebuild per-residue templates (atom name / element / class /
		charge plus the intra-residue bond list), with the N/C-terminal
		and disulfide variants, from charmm36.xml Residues + Patches --
		a pure-stdlib replacement for the OpenMM template/patch engine
		Arguments:
		----------
			root: xml.etree Element - the parsed charmm36.xml root
		Returns:
		--------
			dict: {variant: {atoms: [[name, element, class, charge]],
				bonds: [[name, name]]}}
		'''
		at_elem = {t.attrib['name']: t.attrib.get('element', '')
			for t in root.find('AtomTypes')}
		res = {}
		for rr in root.find('Residues'):
			atoms, bonds = {}, []
			for c in rr:
				if c.tag == 'Atom':
					atoms[c.attrib['name']] = [c.attrib['type'],
						float(c.attrib['charge'])]
				elif c.tag == 'Bond':
					bonds.append((c.attrib['atomName1'],
						c.attrib['atomName2']))
			res[rr.attrib['name']] = (atoms, bonds)
		patch = {}
		for pp in root.find('Patches'):
			d = {'change': {}, 'add': {}, 'remove': [],
				'addbond': [], 'rmbond': []}
			for c in pp:
				a = c.attrib
				if   c.tag == 'ChangeAtom':
					d['change'][a['name']] = [a['type'],
						float(a['charge'])]
				elif c.tag == 'AddAtom':
					d['add'][a['name']] = [a['type'],
						float(a['charge'])]
				elif c.tag == 'RemoveAtom':
					d['remove'].append(a['name'])
				elif c.tag == 'AddBond':
					d['addbond'].append((a['atomName1'],
						a['atomName2']))
				elif c.tag == 'RemoveBond':
					d['rmbond'].append((a['atomName1'],
						a['atomName2']))
			patch[pp.attrib['name']] = d
		def patchside(nm):
			'''Strip a 2-residue patch prefix: "1:CB" -> ("1", "CB").'''
			if len(nm) > 2 and nm[1] == ':': return nm[0], nm[2:]
			return None, nm
		def applypatch(base, pname):
			'''Apply one patch (residue-1 side) to a (atoms, bonds) pair.'''
			atoms = {k: list(v) for k, v in base[0].items()}
			bonds = list(base[1])
			d = patch[pname]
			def keep(nm):
				s, real = patchside(nm)
				return real if s in (None, '1') else None
			for nm, v in d['change'].items():
				real = keep(nm)
				if real is not None and real in atoms:
					atoms[real] = list(v)
			for nm, v in d['add'].items():
				real = keep(nm)
				if real is not None: atoms[real] = list(v)
			rem = {keep(nm) for nm in d['remove']} - {None}
			atoms = {k: v for k, v in atoms.items() if k not in rem}
			bonds = [b for b in bonds
				if b[0] not in rem and b[1] not in rem]
			rmb = set()
			for x, y in d['rmbond']:
				rx, ry = keep(x), keep(y)
				if rx is not None and ry is not None:
					rmb.add(frozenset((rx, ry)))
			bonds = [b for b in bonds if frozenset(b) not in rmb]
			for x, y in d['addbond']:
				rx, ry = keep(x), keep(y)
				if rx is not None and ry is not None:
					bonds.append((rx, ry))
			return (atoms, bonds)
		npatch = {'GLY': 'GLYP', 'PRO': 'PROP'}
		protein = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
			'GLY', 'HSD', 'HSE', 'HSP', 'ILE', 'LEU', 'LYS', 'MET',
			'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
		variants = {}
		for rn in protein:
			if rn not in res: continue
			variants[rn]       = res[rn]
			variants['N' + rn] = applypatch(res[rn],
				npatch.get(rn, 'NTER'))
			variants['C' + rn] = applypatch(res[rn], 'CTER')
		if 'CYS' in res:
			cyx = applypatch(res['CYS'], 'DISU')
			variants['CYX']  = cyx
			variants['NCYX'] = applypatch(cyx, 'NTER')
			variants['CCYX'] = applypatch(cyx, 'CTER')
		templates = {}
		for vn, (atoms, bonds) in variants.items():
			templates[vn] = {
				'atoms': [[nm, at_elem.get(cls, ''), cls, chg]
					for nm, (cls, chg) in atoms.items()],
				'bonds': [[a, b] for a, b in bonds]}
		return templates
	with open(db_path) as f: db = json.load(f)
	def openff():
		'''Port OpenFF Sage 2.3.0 into db['Energy Parameters']'''
		ep = db.setdefault('Energy Parameters', {})
		commit = 'edd7724103a558328c358a9e35462334c4a45b6f'
		url = ('https://raw.githubusercontent.com/openforcefield/'
			'openff-forcefields/' + commit
			+ '/openforcefields/offxml/openff-2.3.0.offxml')
		root = ET.fromstring(download(url))
		bonds = {}
		for p in root.find('Bonds'):
			a = p.attrib
			bonds[a['smirks']] = {'id': a.get('id'),
				'r_0': qval(a['length'], 'angstrom'),
				'K_b': qval(a['k'],
					'kilojoule_per_mole * angstrom ** -2')}
		angles = {}
		for p in root.find('Angles'):
			a = p.attrib
			angles[a['smirks']] = {'id': a.get('id'),
				'theta_0': qval(a['angle'], 'degree'),
				'K_theta': qval(a['k'],
					'kilojoule_per_mole * radian ** -2')}
		propers   = converttorsions(root.find('ProperTorsions'))
		impropers = converttorsions(root.find('ImproperTorsions'))
		# ImproperTorsionPotential ignores idivf, and SMIRNOFF impropers
		# carry no such attribute, so drop the placeholder it picks up.
		for par in impropers.values():
			for comp in par['components']: comp.pop('idivf', None)
		vdw = {}
		for p in root.find('vdW'):
			a = p.attrib
			# Store whichever radius the offxml states. SMIRKSMatch reads
			# either 'sigma' or 'r', so converting between them would only
			# discard bits of the published value.
			rec = {'id': a.get('id'),
				'epsilon': qval(a['epsilon'], 'kilojoule_per_mole')}
			if 'sigma' in a: rec['sigma'] = qval(a['sigma'], 'angstrom')
			else: rec['r'] = qval(a['rmin_half'], 'angstrom')
			rec['alpha'] = 0.0
			vdw[a['smirks']] = rec
		charges = {}
		for p in root.find('LibraryCharges'):
			a = p.attrib
			qs, i = [], 1
			while ('charge%d' % i) in a:
				qs.append(qval(a['charge%d' % i], 'elementary_charge'))
				i += 1
			charges[a['smirks']] = {'id': a.get('id'), 'q': qs}
		constraints = {}
		for p in root.find('Constraints'):
			a = p.attrib
			rec = {'id': a.get('id')}
			if 'distance' in a:
				rec['distance'] = qval(a['distance'], 'angstrom')
			constraints[a['smirks']] = rec
		def naglweights(url):
			'''
			Read the NAGL AM1-BCC network weights from a .pt checkpoint
			Arguments:
			----------
				url: str - raw URL of a pinned openff-gnn-am1bcc .pt file
			Returns:
			--------
				dict: 'gcn_layers' list and 'readout' dict, every tensor
				stored as {'shape': [...], 'data': base64 float32}
			'''
			print(f'[port] downloading {url}', file=sys.stderr)
			try:
				with urllib.request.urlopen(url, timeout=300) as resp:
					blob = resp.read()
			except Exception as err:
				raise RuntimeError(f'port: could not download {url}: {err}')
			# A .pt file is a zip of pickled tensors: data.pkl holds the
			# structure and data/<key> the raw storage bytes. Unpickling
			# with stubs for the torch classes and np.frombuffer for the
			# storages recovers every weight without importing torch.
			zf = zipfile.ZipFile(io.BytesIO(blob))
			root = zf.namelist()[0].split('/')[0]
			dtypes = {'FloatStorage': np.float32,
				'DoubleStorage': np.float64, 'HalfStorage': np.float16,
				'LongStorage': np.int64, 'IntStorage': np.int32,
				'ByteStorage': np.uint8, 'BoolStorage': np.bool_}
			class Stub(dict):
				'''Stand-in for any class the checkpoint pickles'''
				def __init__(self, *a, **k): dict.__init__(self)
				def __setstate__(self, state):
					if isinstance(state, dict): self.update(state)
			def rebuild(store, offset, size, stride, *rest):
				'''Reconstruct one tensor from its storage as an array'''
				arr = np.frombuffer(zf.read('%s/data/%s'
					% (root, store[0])), dtype=store[1])
				size = tuple(size)
				n = int(np.prod(size)) if size else arr.size
				return arr[offset:offset + n].reshape(size)
			class Reader(pickle.Unpickler):
				'''Unpickler that yields NumPy arrays, never torch objects'''
				def find_class(self, mod, name):
					if name == '_rebuild_tensor_v2': return rebuild
					if name in dtypes: return dtypes[name]
					try: return super().find_class(mod, name)
					except Exception: return Stub
				def persistent_load(self, pid):
					dt = pid[1] if pid[1] in dtypes.values() else np.float32
					return (pid[2], dt)
			obj = Reader(io.BytesIO(zf.read('%s/data.pkl' % root))).load()
			seen, found = set(), {}
			def collect(node, name):
				'''Walk the unpickled tree and index arrays by parameter name'''
				if id(node) in seen: return
				seen.add(id(node))
				if isinstance(node, dict):
					for k, v in node.items(): collect(v, str(k))
				elif isinstance(node, (list, tuple)):
					for v in node: collect(v, name)
				elif isinstance(node, np.ndarray): found[name] = node
			collect(obj, '')
			pack = lambda a: {'shape': list(a.shape),
				'data': base64.b64encode(np.ascontiguousarray(a,
				dtype=np.float32).tobytes()).decode('ascii')}
			conv = 'convolution_module.gcn_layers.'
			read = 'readout_modules.am1bcc_charges.readout_layers.'
			n_gcn = 1 + max(int(k[len(conv):].split('.')[0])
				for k in found if k.startswith(conv))
			layers = [{
				'fc_neigh_w': pack(found['%s%d.fc_neigh.weight' % (conv, i)]),
				'fc_self_w':  pack(found['%s%d.fc_self.weight' % (conv, i)]),
				'fc_self_b':  pack(found['%s%d.fc_self.bias' % (conv, i)])}
				for i in range(n_gcn)]
			# The checkpoint also carries a table of precomputed AM1-BCC
			# charges that NAGL consults before running the network. Each
			# entry stores an atom-mapped SMILES (which fixes the atom
			# order) and one charge per atom. Without it, molecules in the
			# table get network values where Sage returns tabulated ones.
			tabs = (obj.get('hyperparameters') or {}).get(
				'lookup_tables') or {}
			ents = ((tabs.get('am1bcc_charges') or {}).get('properties')
				or {})
			lookup = []
			for e in ents.values():
				d = e.get('__dict__', e) if isinstance(e, dict) else vars(e)
				smi = d.get('mapped_smiles')
				if not smi: continue
				lookup.append({'smiles': smi,
					'q': [float(x) for x in d['property_value']]})
			return {'gcn_layers': layers, 'lookup': lookup, 'readout': {
				'linear_0_w': pack(found[read + '0.weight']),
				'linear_0_b': pack(found[read + '0.bias']),
				'linear_1_w': pack(found[read + '3.weight']),
				'linear_1_b': pack(found[read + '3.bias'])}}
		nagl_commit = '6a30bde31fc9ba7f9ff218dacd291184e2f70946'
		nagl_url = ('https://raw.githubusercontent.com/openforcefield/'
			'openff-nagl-models/' + nagl_commit + '/openff/nagl_models/'
			'models/am1bcc/openff-gnn-am1bcc-1.0.0.pt')
		prev = ep.get('OpenFF') or ep.get('openFF') or {}
		# Sage covers neither selenium, nor aromatic C:N ring bonds, nor
		# the phosphate improper, and the offxml has no field for the
		# per-type polarisability, so those exist only in the database.
		# Carry them forward, appended last so that they win under the
		# last-match-wins SMIRKS precedence.
		for part, new in (('Bonds', bonds), ('Angles', angles),
				('ProperTorsions', propers),
				('ImproperTorsions', impropers), ('vdW', vdw)):
			for sm, par in (prev.get(part) or {}).items():
				if sm not in new: new[sm] = par
				elif 'alpha' in par: new[sm]['alpha'] = par['alpha']
		block = {
			'Constants': {'epsilon_r': 1.0, 'f_lj': 0.5,
				'f_elec': 5.0 / 6.0},
			'Constraints':      constraints,
			'Bonds':            bonds,
			'Angles':           angles,
			'ProperTorsions':   propers,
			'ImproperTorsions': impropers,
			'vdW':              vdw,
			'Electrostatic':    charges,
			'Terms': [
				['BondPotential',            {'alg': 'harmonic'}],
				['AnglePotential',           {}],
				['ProperTorsionPotential',   {}],
				['ImproperTorsionPotential', {'alg': 'fourier'}],
				['VDWPotential',             {'alg': '12-6'}],
				['ElectrostaticPotential',   {'alg': 'constant'}],
			],
		}
		block['AM1BCC'] = naglweights(nagl_url)
		ep.pop('OpenFF', None)
		ep['OpenFF'] = block
	def ff19sb():
		'''Port AMBER ff19SB into db['Energy Parameters']'''
		ep = db.setdefault('Energy Parameters', {})
		commit = 'f7fa0c27c1f8d943c339d67b3bf22f026d0bd8b5'
		base = ('https://raw.githubusercontent.com/openmm/openmm/'
			+ commit + '/wrappers/python/openmm/app/data/')
		xml_urls = [base + 'amber19/protein.ff19SB.xml',
			base + 'amber14/DNA.OL15.xml',
			base + 'amber14/RNA.OL3.xml']
		bonds, angles, propers, impropers = {}, {}, {}, {}
		vdw, templates, cmap = {}, {}, {}
		for url in xml_urls:
			root = ET.fromstring(download(url))
			type2class, type2elem = {}, {}
			at = root.find('AtomTypes')
			if at is not None:
				for t in at:
					type2class[t.attrib['name']] = \
						t.attrib.get('class', t.attrib['name'])
					type2elem[t.attrib['name']] = \
						t.attrib.get('element', '')
			hbf = root.find('HarmonicBondForce')
			by_class = (hbf is not None and len(hbf) > 0
				and 'class1' in hbf[0].attrib)
			res = root.find('Residues')
			if res is not None:
				for r in res:
					ratoms, rbonds = [], []
					for c in r:
						if c.tag == 'Atom':
							tp = c.attrib['type']
							tid = (type2class.get(tp, tp)
								if by_class else tp)
							ratoms.append([c.attrib['name'],
								type2elem.get(tp, ''), tid,
								float(c.attrib.get('charge', 0.0))])
						elif c.tag == 'Bond':
							rbonds.append([c.attrib['atomName1'],
								c.attrib['atomName2']])
					templates[r.attrib['name']] = {
						'atoms': ratoms, 'bonds': rbonds}
			if hbf is not None:
				for b in hbf:
					c1, c2 = cidof(b, 1), cidof(b, 2)
					bonds[f'<at={c1},{c2}>[*:1]~[*:2]'] = {
						'r_0': float(b.attrib['length']) * 10.0,
						'K_b': float(b.attrib['k']) * 0.01}
			haf = root.find('HarmonicAngleForce')
			if haf is not None:
				for a in haf:
					c1, c2, c3 = (cidof(a, 1), cidof(a, 2), cidof(a, 3))
					angles[f'<at={c1},{c2},{c3}>[*:1]~[*:2]~[*:3]'] = {
						'theta_0': math.degrees(
							float(a.attrib['angle'])),
						'K_theta': float(a.attrib['k'])}
			ptf = root.find('PeriodicTorsionForce')
			if ptf is not None:
				for t in ptf:
					cs = [cidof(t, i) for i in (1, 2, 3, 4)]
					comps, k = [], 1
					while ('k%d' % k) in t.attrib:
						comps.append({
							'n': int(t.attrib['periodicity%d' % k]),
							'phi_0': -math.degrees(
								float(t.attrib['phase%d' % k])),
							'K_phi': float(t.attrib['k%d' % k]),
							'idivf': 1.0})
						k += 1
					if t.tag == 'Improper':
						ro = [cs[1], cs[0], cs[2], cs[3]]
						tag = ','.join('*' if x == '' else x
							for x in ro)
						impropers[f'<at={tag}>'
							'[*:1]~[*:2](~[*:3])~[*:4]'] = {
							'components': comps}
					else:
						tag = ','.join('*' if x == '' else x
							for x in cs)
						propers[f'<at={tag}>[*:1]~[*:2]~[*:3]~[*:4]'] = \
							{'components': comps}
			nbf = root.find('NonbondedForce')
			if nbf is not None:
				for a in nbf:
					if a.tag != 'Atom': continue
					tid = a.attrib.get('class') or a.attrib.get('type')
					vdw[f'<at={tid}>[*:1]'] = {
						'epsilon': float(a.attrib.get('epsilon', 0.0)),
						'sigma': float(a.attrib.get('sigma', 0.0)) * 10.0,
						'alpha': 0.0}
			cmf = root.find('CMAPTorsionForce')
			if cmf is not None:
				maps, ctors = [], []
				for c in cmf:
					if c.tag == 'Map':
						g = [float(x) for x in c.text.split()]
						m = int(round(len(g) ** 0.5))
						maps.append(np.asarray(g,
							dtype=np.float64).reshape(m, m))
					elif c.tag == 'Torsion':
						ctors.append(c.attrib)
				for tr in ctors:
					idx = int(tr.get('map', 0))
					if idx >= len(maps): continue
					parts = (tr.get('type3', '') or '').split('-')
					if len(parts) >= 2 and parts[0] == 'cmap':
						cmap[parts[1]] = maps[idx].tolist()
		block = {
			'Constants': {'epsilon_r': 1.0, 'f_lj': 0.5,
				'f_elec': 0.8333333333333334},
			'improper_style':    'amber',
			'proper_precedence': 'openmm',
			'Constraints':      {'<residue_templates>': templates},
			'Bonds':            bonds,
			'Angles':           angles,
			'UB':               {},
			'ProperTorsions':   propers,
			'ImproperTorsions': impropers,
			'vdW':              vdw,
			'Electrostatic':    {},
			'CMAP':             cmap,
			'Terms': [
				['BondPotential',            {'alg': 'harmonic'}],
				['AnglePotential',           {}],
				['ProperTorsionPotential',   {}],
				['ImproperTorsionPotential', {'alg': 'fourier'}],
				['VDWPotential',             {'alg': '12-6'}],
				['ElectrostaticPotential',   {'alg': 'constant'}],
				['CMAPPotential',            {'alg': 'openmm'}],
			],
		}
		ep.pop('AMBER ff19SB', None)
		ep['ff19SB'] = block
	def charmm36():
		'''Port CHARMM36 into db['Energy Parameters']'''
		ep = db.setdefault('Energy Parameters', {})
		commit = 'f7fa0c27c1f8d943c339d67b3bf22f026d0bd8b5'
		xml_url = ('https://raw.githubusercontent.com/openmm/openmm/'
			+ commit + '/wrappers/python/openmm/app/data/charmm36.xml')
		root = ET.fromstring(download(xml_url))
		bonds = {}
		hbf = root.find('HarmonicBondForce')
		if hbf is not None:
			for b in hbf:
				c1, c2 = cidof(b, 1), cidof(b, 2)
				bonds[f'<at={c1},{c2}>[*:1]~[*:2]'] = {
					'r_0': float(b.attrib['length']) * 10.0,
					'K_b': float(b.attrib['k']) * 0.01}
		angles = {}
		haf = root.find('HarmonicAngleForce')
		if haf is not None:
			for a in haf:
				c1, c2, c3 = cidof(a, 1), cidof(a, 2), cidof(a, 3)
				angles[f'<at={c1},{c2},{c3}>[*:1]~[*:2]~[*:3]'] = {
					'theta_0': math.degrees(float(a.attrib['angle'])),
					'K_theta': float(a.attrib['k'])}
		ub = {}
		ubf = root.find('AmoebaUreyBradleyForce')
		if ubf is not None:
			for u in ubf:
				c1, c2, c3 = cidof(u, 1), cidof(u, 2), cidof(u, 3)
				ub[f'<at={c1},{c2},{c3}>[*:1]~[*:2]~[*:3]'] = {
					's_0':  float(u.attrib['d']) * 10.0,
					'K_ub': float(u.attrib['k']) * 0.01}
		propers = {}
		ptf = root.find('PeriodicTorsionForce')
		if ptf is not None:
			for t in ptf:
				if t.tag != 'Proper': continue
				cs = [cidof(t, i) for i in (1, 2, 3, 4)]
				comps, k = [], 1
				while ('k%d' % k) in t.attrib:
					comps.append({
						'n': int(t.attrib['periodicity%d' % k]),
						'phi_0': -math.degrees(
							float(t.attrib['phase%d' % k])),
						'K_phi': float(t.attrib['k%d' % k]),
						'idivf': 1.0})
					k += 1
				tag = ','.join('*' if x == '' else x for x in cs)
				sm = f'<at={tag}>[*:1]~[*:2]~[*:3]~[*:4]'
				if sm not in propers: propers[sm] = {'components': comps}
		impropers = {}
		ctf = root.find('CustomTorsionForce')
		if ctf is not None:
			for t in ctf:
				if t.tag != 'Improper': continue
				cs = [cidof(t, i) for i in (1, 2, 3, 4)]
				tag = ','.join('*' if x == '' else x for x in cs)
				sm = f'<at={tag}>[*:1](~[*:2])(~[*:3])~[*:4]'
				if sm in impropers: continue
				impropers[sm] = {'components': [{
					'n': 2,
					'phi_0': -math.degrees(
						float(t.attrib.get('theta0', 0.0))),
					'K_phi': float(t.attrib.get('k', 0.0)),
					'idivf': 1.0}]}
		vdw = {}
		ljf = root.find('LennardJonesForce')
		if ljf is not None:
			for a in ljf:
				if a.tag != 'Atom': continue
				tid = a.attrib.get('type') or a.attrib.get('class')
				sig = float(a.attrib.get('sigma', 0.0)) * 10.0
				eps = float(a.attrib.get('epsilon', 0.0))
				s14 = (float(a.attrib['sigma14']) * 10.0
					if 'sigma14' in a.attrib else sig)
				e14 = (float(a.attrib['epsilon14'])
					if 'epsilon14' in a.attrib else eps)
				vdw[f'<at={tid}>[*:1]'] = {'epsilon': eps, 'sigma': sig,
					'epsilon14': e14, 'sigma14': s14, 'alpha': 0.0}
		cmap = {}
		cmf = root.find('CMAPTorsionForce')
		if cmf is not None:
			maps, ctors = [], []
			for c in cmf:
				if c.tag == 'Map':
					g = [float(x) for x in c.text.split()]
					m = int(round(len(g) ** 0.5))
					maps.append(np.asarray(g,
						dtype=np.float64).reshape(m, m))
				elif c.tag == 'Torsion':
					ctors.append(c.attrib)
			standard = list('ARNDCQEHILKMFSTWYV')
			for tr in ctors:
				if (tr.get('type5', '') or '') == 'N': continue
				idx = int(tr.get('map', 0))
				if idx >= len(maps): continue
				t2, t3 = tr.get('type2', ''), tr.get('type3', '')
				if   t3 == 'CT1' and t2 == 'NH1': letters = standard
				elif t3 == 'CT2' and t2 == 'NH1': letters = ['G']
				elif t3 == 'CP1' and t2 == 'N':   letters = ['P']
				else: continue
				grid = maps[idx].tolist()
				for one in letters: cmap[one] = grid
		templates = charmmtypes(root)
		block = {
			'Constants': {'epsilon_r': 1.0, 'f_lj': 1.0,
				'f_elec': 1.0},
			'improper_style':    'charmm',
			'proper_precedence': 'openmm',
			'Constraints':      {'<residue_templates>': templates},
			'Bonds':            bonds,
			'Angles':           angles,
			'UB':               ub,
			'ProperTorsions':   propers,
			'ImproperTorsions': impropers,
			'vdW':              vdw,
			'Electrostatic':    {},
			'CMAP':             cmap,
			'Terms': [
				['BondPotential',            {'alg': 'harmonic'}],
				['AnglePotential',           {}],
				['UBPotential',              {}],
				['ProperTorsionPotential',   {}],
				['ImproperTorsionPotential', {'alg': 'harmonic'}],
				['VDWPotential',             {'alg': '12-6'}],
				['ElectrostaticPotential',   {'alg': 'constant'}],
				['CMAPPotential',            {'alg': 'openmm'}],
			],
		}
		ep['CHARMM36'] = block
	def vina():
		'''Port AutoDock Vina into db['Score Parameters']'''
		sp = db.setdefault('Score Parameters', {})
		KCAL_TO_KJ = 4.184
		BASE = ('https://raw.githubusercontent.com/'
			'ccsb-scripps/AutoDock-Vina/'
			'3c65c0b3e6c2c1d183f6a175ecb65e3c5ba91645/src/lib/')
		FILES = ('potentials.h', 'vina.h',
			'scoring_function.h', 'atom_constants.h')
		def fetch(name):
			'''
			Download one upstream source file as a UTF-8 string
			Arguments:
			----------
				name: str - file name under src/lib/ (e.g. 'vina.h')
			Returns:
			--------
				str: the file contents
			'''
			url = BASE + name
			with urllib.request.urlopen(url, timeout=120) as r:
				return r.read().decode('utf-8')
		src = {n: fetch(n) for n in FILES}
		# 1. Default weights from vina.h set_vina_weights signature.
		m = re.search(
			r'set_vina_weights\s*\(\s*'
			r'double\s+weight_gauss1\s*=\s*(-?[\d.]+)\s*,\s*'
			r'double\s+weight_gauss2\s*=\s*(-?[\d.]+)\s*,\s*'
			r'double\s+weight_repulsion\s*=\s*(-?[\d.]+)\s*,\s*'
			r'double\s+weight_hydrophobic\s*=\s*(-?[\d.]+)\s*,\s*'
			r'double\s+weight_hydrogen\s*=\s*(-?[\d.]+)\s*,\s*'
			r'double\s+weight_glue\s*=\s*(-?[\d.]+)\s*,\s*'
			r'double\s+weight_rot\s*=\s*(-?[\d.]+)\s*\)',
			src['vina.h'])
		if m is None:
			raise Exception('Vina: could not parse weights from vina.h')
		w_gauss1, w_gauss2, w_rep, w_hyd = (float(m.group(i))
			for i in (1, 2, 3, 4))
		w_hbond, w_glue, w_rot = (float(m.group(i)) for i in (5, 6, 7))
		# 2. Term constructor args from scoring_function.h.
		#    The Vina branch is the first one in ScoringFunction's switch.
		sf = src['scoring_function.h']
		def first(pat):
			'''
			Return the float-group tuple of the first regex match
			Arguments:
				pat: str - a regex with capture groups
			Returns:
				tuple of floats
			'''
			mm = re.search(pat, sf)
			if mm is None:
				raise Exception('Vina: missing pattern '+pat)
			return tuple(float(g) for g in mm.groups())
		# vina_gaussian(offset, width, cutoff) -- two of them in order
		gpat = r'new\s+vina_gaussian\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)'
		gs = re.findall(gpat, sf)
		if len(gs) < 2:
			raise Exception('Vina: did not find two vina_gaussian entries')
		g1_off, g1_w, g1_cut = (float(x) for x in gs[0])
		g2_off, g2_w, g2_cut = (float(x) for x in gs[1])
		rep_off, rep_cut = first(
			r'new\s+vina_repulsion\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)')
		hyd_good, hyd_bad, hyd_cut = first(
			r'new\s+vina_hydrophobic\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)')
		hb_good, hb_bad, hb_cut = first(
			r'new\s+vina_non_dir_h_bond\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)')
		# 3. XS atom-type table + predicates from atom_constants.h.
		ac = src['atom_constants.h']
		# XS_TYPE_<NAME> = <index>; collect in declaration order
		xs_decl = re.findall(
			r'^\s*const\s+sz\s+XS_TYPE_([A-Za-z0-9_]+)\s*=\s*(\d+)\s*;',
			ac, re.M)
		if not xs_decl:
			raise Exception('Vina: no XS_TYPE_* found in atom_constants.h')
		xs_idx_to_name = {}
		for nm, i in xs_decl:
			if nm in ('SIZE',): continue
			xs_idx_to_name[int(i)] = nm
		# xs_vdw_radii[] = { v1, // name1\n v2, // name2 ... };
		rad_block = re.search(
			r'const\s+fl\s+xs_vdw_radii\s*\[\s*\]\s*=\s*\{([^}]*)\}\s*;',
			ac, re.S)
		if rad_block is None:
			raise Exception('Vina: xs_vdw_radii block not found')
		rad_vals = []
		for line in rad_block.group(1).split('\n'):
			mm = re.match(r'\s*(-?[\d.]+)\s*,', line)
			if mm: rad_vals.append(float(mm.group(1)))
		# predicates: extract the XS_TYPE_* names listed in each function body
		def grab(name):
			'''
			Extract the XS type names mentioned in one xs_is_* predicate
			Arguments:
				name: str - 'xs_is_hydrophobic', 'xs_is_acceptor', or
					'xs_is_donor'
			Returns:
				set of str: XS type names (without the XS_TYPE_ prefix)
			'''
			body = re.search(
				r'inline\s+bool\s+' + re.escape(name) + r'\s*\([^)]*\)\s*\{([^}]*)\}',
				ac, re.S)
			if body is None:
				raise Exception('Vina: predicate '+name+' not found')
			return set(re.findall(r'XS_TYPE_([A-Za-z0-9_]+)', body.group(1)))
		hphob = grab('xs_is_hydrophobic')
		accept = grab('xs_is_acceptor')
		donor = grab('xs_is_donor')
		# 4. Conf-independent num_tors_div weight (the user-facing weight_rot).
		#    Verified: stored raw weight = 5*weight_rot/0.1 - 1, and the
		#    eval applies weight = 0.1*(raw+1), giving effective denominator
		#    1 + weight_rot*Nrot.
		# 5. Assemble the parameter block. Numeric values in kJ/mol-equiv:
		#    weights are multiplied by 4.184 from the published kcal-scale
		#    so internal storage is uniform with the rest of database.json.
		#    A 'scale' constant (1/4.184) is applied at Score.__call__ exit
		#    to return native kcal/mol.
		xs_types = {}
		for i in sorted(xs_idx_to_name):
			nm = xs_idx_to_name[i]
			if i >= len(rad_vals): continue
			xs_types[nm] = {
				'radius':      rad_vals[i],
				'hydrophobic': nm in hphob,
				'acceptor':    nm in accept,
				'donor':       nm in donor}
		block = {
			'Constants': {
				'scale':  1.0 / KCAL_TO_KJ,
				'cutoff': float(g1_cut),
				'nrot_w': w_rot,
				'glue_w': w_glue * KCAL_TO_KJ},
			'XS_atom_types': xs_types,
			'Gauss1': {
				'offset': g1_off, 'width': g1_w, 'cutoff': g1_cut,
				'weight': w_gauss1 * KCAL_TO_KJ},
			'Gauss2': {
				'offset': g2_off, 'width': g2_w, 'cutoff': g2_cut,
				'weight': w_gauss2 * KCAL_TO_KJ},
			'Repulsion': {
				'offset': rep_off, 'cutoff': rep_cut,
				'weight': w_rep * KCAL_TO_KJ},
			'Hydrophobic': {
				'good': hyd_good, 'bad': hyd_bad, 'cutoff': hyd_cut,
				'weight': w_hyd * KCAL_TO_KJ},
			'HBond': {
				'good': hb_good, 'bad': hb_bad, 'cutoff': hb_cut,
				'weight': w_hbond * KCAL_TO_KJ},
			'Terms': [
				['Gauss1Potential',      {}],
				['Gauss2Potential',      {}],
				['RepulsionPotential',   {}],
				['HydrophobicPotential', {}],
				['HBondPotential',       {}],
				['TorsionalPenalty',     {}]]}
		sp['AutoDock Vina'] = block
	def ref15():
		'''Port Rosetta REF15 into db['Score Parameters'] only. The
		top-level Rotamer Library is never written: it ships with the
		database and is not Rosetta-owned.'''
		sp = db.setdefault('Score Parameters', {})
		ETABLE_ATOM_TYPES = [
			'CNH2', 'COO', 'CH0', 'CH1', 'CH2', 'CH3', 'aroC', 'Ntrp',
			'Nhis', 'NtrR', 'NH2O', 'Nlys', 'Narg', 'Npro', 'OH', 'ONH2',
			'OOC', 'Oaro', 'S', 'SH1', 'Nbb', 'CAbb', 'CObb', 'OCbb',
			'Hpol', 'Hapo', 'Haro', 'HNbb', 'HOH',
			# Required by the non-canonical residues: 'Phos' for the
			# TPO/PTR phosphate, 'F' for FT6. New types go at the END so
			# the indices of the 29 canonical types are unchanged and
			# EtablePairParams stays identical for canonical structures.
			# 'HS' is the CYS thiol hydrogen. It was absent from the
			# original 29, so CYS's HG silently contributed nothing to
			# fa_atr/fa_rep/fa_sol/lk_ball; SEC made that visible.
			'Phos', 'F', 'HS']

		def _parsehbonddata(raw):
			'''
			Parse the bundled Rosetta hbond CSVs into a compact JSON-friendly
			dict for use by Score('REF15')'s HBond* term methods
			Arguments:
			----------
				raw: dict of {filename: str contents}
			Returns:
			--------
				dict: keys 'polynomials', 'fade_intervals', 'eval_table',
					'donor_strengths', 'acceptor_strengths', 'acc_hybridization',
					'seq_sep_names', 'weight_type_names', 'donor_chem_names',
					'acceptor_chem_names'
			'''
			def csvrows(text):
				'''Iterate CSV rows ignoring trailing blank fields'''
				for line in text.splitlines():
					s = line.rstrip()
					if not s or s.startswith('#'): continue
					yield s.split(',')
			# polynomials
			polys = {}
			for r in csvrows(raw.get('HBPoly1D.csv', '')):
				if len(r) < 11: continue
				try:
					pid = int(r[0])
				except ValueError: continue
				name = r[1]
				dim = r[3]
				try:
					xmin = float(r[4]); xmax = float(r[5])
					min_v = float(r[6]); max_v = float(r[7])
					root1 = float(r[8])
					# root2 column is sometimes empty (poly_AHD_1[h..k], etc.)
					root2 = float(r[9]) if r[9] else 0.0
					degree = int(r[10])
				except ValueError: continue
				coeffs = []
				for k in range(11, 11 + degree):
					if k >= len(r) or r[k] == '': break
					try: coeffs.append(float(r[k]))
					except ValueError: break
				polys[name] = {'id': pid, 'dim': dim, 'xmin': xmin,
					'xmax': xmax, 'min_val': min_v, 'max_val': max_v,
					'root1': root1, 'root2': root2, 'degree': degree,
					'coeffs': coeffs}
			# fade intervals
			fades = {}
			for r in csvrows(raw.get('HBFadeIntervals.csv', '')):
				if len(r) < 7: continue
				try:
					fid = int(r[0])
					min1 = float(r[3]); min2 = float(r[4])
					max1 = float(r[5]); max2 = float(r[6])
				except ValueError: continue
				fades[r[1]] = {'id': fid, 'kind': r[2], 'min1': min1,
					'min2': min2, 'max1': max1, 'max2': max2}
			# evaluation table
			eval_table = []
			for r in csvrows(raw.get('HBEval.csv', '')):
				if len(r) < 16: continue
				# Skip the trailing $... id and surplus columns
				entry = {
					'don':           r[0],
					'acc':           r[1],
					'sep':           r[2],
					'fade_AHdist':   r[3],
					'fade_cosBAH_short': r[4],
					'fade_cosBAH_long':  r[5],
					'fade_cosBAH_chi':   r[6],
					'fade_cosAHD_short': r[7],
					'fade_cosAHD_long':  r[8],
					'poly_AHdist':       r[9],
					'poly_cosBAH_short': r[10],
					'poly_cosBAH_long':  r[11],
					'poly_cosBAH_chi':   r[12],
					'poly_cosAHD_short': r[13],
					'poly_cosAHD_long':  r[14],
					'weight':            r[15]}
				eval_table.append(entry)
			# strengths
			don_str = {}
			for r in csvrows(raw.get('DonStrength.csv', '')):
				if len(r) < 2: continue
				try: don_str[r[0]] = float(r[1])
				except ValueError: pass
			acc_str = {}
			for r in csvrows(raw.get('AccStrength.csv', '')):
				if len(r) < 2: continue
				try: acc_str[r[0]] = float(r[1])
				except ValueError: pass
			# acceptor hybridization
			acc_hyb = {}
			for r in csvrows(raw.get('HBAccHybridization.csv', '')):
				if len(r) < 2: continue
				acc_hyb[r[0]] = r[1]
			# Names tables (id -> name lookup for parsing other tables)
			def nametable(text, name_col=1):
				out = []
				for r in csvrows(text):
					if not r: continue
					out.append(r[name_col] if len(r) > name_col else '')
				return out
			return {
				'polynomials':       polys,
				'fade_intervals':    fades,
				'eval_table':        eval_table,
				'donor_strengths':   don_str,
				'acceptor_strengths': acc_str,
				'acc_hybridization': acc_hyb}

		def _parseparams(txt):
			'''
			Parse one Rosetta .params residue-topology file
			Arguments:
			----------
				txt: str - the file contents
			Returns:
			--------
				dict: {'atoms': {pdb_name: {'type':..., 'mm_type':...,
					'charge':...}}, 'bonds': [(a,b,bond_order_or_1), ...],
					'name': str, 'aa': str, 'aliases': {pdb_alias: pdb_name}}
			'''
			atoms = {}
			bonds = []
			aliases = {}
			name = None; aa = None
			for line in txt.splitlines():
				line = line.split('#', 1)[0].rstrip()
				if not line.strip(): continue
				toks = line.split()
				if toks[0] == 'NAME':
					name = toks[1] if len(toks) > 1 else None
				elif toks[0] == 'AA':
					aa = toks[1] if len(toks) > 1 else None
				elif toks[0] == 'ATOM' and len(toks) >= 5:
					pdb_nm = toks[1]
					ros_type = toks[2]
					mm_type = toks[3]
					try: charge = float(toks[4])
					except ValueError: charge = 0.0
					atoms[pdb_nm] = {
						'type': ros_type, 'mm_type': mm_type,
						'charge': charge}
				elif toks[0] == 'ATOM_ALIAS' and len(toks) >= 3:
					tgt = toks[1]
					for alias in toks[2:]:
						aliases[alias] = tgt
				elif toks[0] in ('BOND', 'BOND_TYPE') and len(toks) >= 3:
					a, b = toks[1], toks[2]
					bo = 1
					if toks[0] == 'BOND_TYPE' and len(toks) >= 4:
						try: bo = int(toks[3])
						except ValueError: bo = 1
					bonds.append([a, b, bo])
				elif toks[0] == 'CUT_BOND' and len(toks) >= 3:
					bonds.append([toks[1], toks[2], 1])
			return {'name': name, 'aa': aa, 'atoms': atoms,
				'bonds': bonds, 'aliases': aliases}

		def _splineddy2(x0, y0, dy0, x1, y1, dy1):
			'''
			Second derivatives at the two endpoints of a natural cubic spline
			that passes through (x0, y0) and (x1, y1) with prescribed first
			derivatives dy0 at x0 and dy1 at x1. Derived from Rosetta's
			`spline_second_derivative` (numeric/interpolation/spline/
			spline_functions.cc) specialized to n=2 segments.
			Arguments:
			----------
				x0, y0, dy0: lower endpoint position, value, first deriv
				x1, y1, dy1: upper endpoint position, value, first deriv
			Returns:
			--------
				tuple (y2_lo, y2_hi): second derivatives at the two endpoints
			'''
			h = x1 - x0
			dy = (y1 - y0) / h
			u1 = (3.0 / h) * (dy - dy0)
			un = (3.0 / h) * (dy1 - dy)
			y2_hi = (un - 0.5 * u1) / 0.75
			y2_lo = -0.5 * y2_hi + u1
			return (y2_lo, y2_hi)

		def _cubicfromspline(xlo, xhi, ylo, yhi, y2lo, y2hi):
			'''
			Convert spline (yvals, second-derivs) to cubic polynomial c0..c3
			matching Rosetta's `cubic_polynomial_from_spline`.
			Arguments:
			----------
				xlo, xhi: x-range
				ylo, yhi: function values
				y2lo, y2hi: second derivatives
			Returns:
			--------
				list [c0, c1, c2, c3] of the cubic polynomial evaluated as
				c3*x^3 + c2*x^2 + c1*x + c0
			'''
			a, b = xlo, xhi
			c, d = yhi, ylo
			e, f = y2hi, y2lo
			c0 = (((b*b*b*f - a*a*a*e)/(b-a) + (a*e - b*f)*(b-a)) / 6
				+ (b*d - a*c) / (b-a))
			c1 = ((3*a*a*e/(b-a) - e*(b-a) + f*(b-a) - 3*b*b*f/(b-a)) / 6
				+ (c - d) / (b-a))
			c2 = (3*b*f - 3*a*e) / (6*(b-a))
			c3 = (e - f) / (6*(b-a))
			return [c0, c1, c2, c3]

		def _etableparams(atom_types):
			'''
			Pure-Python re-implementation of Rosetta's EtableParamsOnePair
			initialization (source/src/core/scoring/etable/Etable.cc). For each
			pair of atom types in ETABLE_ATOM_TYPES, computes the 30 analytic
			parameters used by FaAtr / FaRep / FaSol / LkBallWtd / FaIntraRep
			in Pose's energy.py.

			Algorithm:
			1) Build the pair sigma matrix with hbond / water-radius overrides.
			2) precalc_etable_coefficients: closed-form per-pair LJ + LK coeffs.
			3) For each pair, tabulate ljatr/ljrep/fasol1/fasol2/derivatives on
			   721 distance bins (bins_per_A2=20, max_dis=6.0, dis(i) =
			   sqrt((i-1)/20)).
			4) modify_pot_one_pair: OCbb-OCbb extra rep, carbon-carbon
			   close-flat fasol.
			5) zero_hydrogen_and_water_ljatr: zero attr+fasol for H pairs.
			6) smooth_etables_one_pair: spline-smooth ljatr at max_dis, fasol
			   close and far regions; extract Hermite cubic c0..c3.

			Arguments:
			----------
				atom_types: dict from ref15.py atom_properties parsing - per
				atom-name dict with LJ_RADIUS, LJ_WDEPTH, LK_DGFREE, LK_LAMBDA,
				LK_VOLUME plus donor/acceptor/polar_h/h2o flags
			Returns:
			--------
				dict stored at `['Score Parameters']['REF15']`
				['EtablePairParams']:
				{atom_types: [list of 29 names], n_types: 29, pairs: [841 dicts]}
			'''
			import math as _math
			# Rosetta constants (REF15 defaults from EtableOptions.cc)
			BINS_PER_A2     = 20
			MAX_DIS         = 6.0
			MAX_DIS2        = MAX_DIS * MAX_DIS
			W_RADIUS        = 1.0
			LJ_SWITCH_D2S   = 0.6
			LJ_HBOND_DIS    = 3.0  # hardcoded in Etable.cc
			LJ_HBOND_OH_DIS = 2.6
			LJ_HBOND_HDIS   = 1.75
			LK_MIN_DIS2S    = 0.89
			LJ_SLOPE_ICEPT  = 0.0
			NTYPES          = len(ETABLE_ATOM_TYPES)
			ETABLE_DISBINS  = int(MAX_DIS2 * BINS_PER_A2) + 1   # 721
			# Per-type LJ/LK arrays (atom-type-name to scalar)
			LJ_R = [atom_types[n]['LJ_RADIUS'] for n in ETABLE_ATOM_TYPES]
			LJ_W = [atom_types[n]['LJ_WDEPTH'] for n in ETABLE_ATOM_TYPES]
			LK_DG = [atom_types[n]['LK_DGFREE'] for n in ETABLE_ATOM_TYPES]
			LK_L  = [atom_types[n]['LK_LAMBDA'] for n in ETABLE_ATOM_TYPES]
			LK_V  = [atom_types[n]['LK_VOLUME'] for n in ETABLE_ATOM_TYPES]
			IS_ACC  = [atom_types[n]['acceptor']    for n in ETABLE_ATOM_TYPES]
			IS_DON  = [atom_types[n]['donor']       for n in ETABLE_ATOM_TYPES]
			IS_POLH = [atom_types[n]['polar_h']     for n in ETABLE_ATOM_TYPES]
			IS_H2O  = [n == 'HOH' for n in ETABLE_ATOM_TYPES]
			# Hydrogens (used to zero out attractive LJ and fasol)
			HYDRO = set(n for n in ETABLE_ATOM_TYPES
				if atom_types[n].get('element') == 'H')
			# Carbon types for the close-flat fasol linearization. Rosetta
			# uses ONLY (CH1, CH2, CH3, aroC) - not all carbon atom types
			# (Etable.cc:initialize_carbontypes_to_linearize_fasol).
			CARBON = {'CH1', 'CH2', 'CH3', 'aroC'}
			# OCbb pair-special-case
			OCBB_IDX = ETABLE_ATOM_TYPES.index('OCbb')
			# Derived switch constants
			LJ_S2D = 1.0 / LJ_SWITCH_D2S
			LJ_V2W = LJ_S2D**12 - 2.0 * LJ_S2D**6
			LJ_S2W = -12.0 * (LJ_S2D**13 - LJ_S2D**7)
			# Distance bin lookup (dis array)
			DIS = [_math.sqrt((i) / float(BINS_PER_A2))
				for i in range(ETABLE_DISBINS)]
			# Build sigma matrix
			sigma = [[0.0]*NTYPES for _ in range(NTYPES)]
			for i in range(NTYPES):
				ni = ETABLE_ATOM_TYPES[i]
				for j in range(NTYPES):
					nj = ETABLE_ATOM_TYPES[j]
					s = W_RADIUS * (LJ_R[i] + LJ_R[j])
					if s < 1e-9: s = 1e-9
					# hbond radius override
					if ((IS_ACC[i] and IS_DON[j]) or
							(IS_DON[i] and IS_ACC[j])):
						oh_donor = (
							(IS_DON[j] and nj[:2] in ('OH', 'OW')) or
							(IS_DON[i] and ni[:2] in ('OH', 'OW')) or
							(IS_DON[j] and nj == 'Oet3') or
							(IS_DON[i] and ni == 'Oet3'))
						s = LJ_HBOND_OH_DIS if oh_donor else LJ_HBOND_DIS
					elif ((IS_ACC[i] and IS_POLH[j]) or
							(IS_POLH[i] and IS_ACC[j])):
						s = LJ_HBOND_HDIS
					# water radius override (lj_water_dis=3.0, lj_water_hdis=1.95)
					if ((IS_ACC[i] or IS_DON[i]) and IS_H2O[j]) or \
							((IS_ACC[j] or IS_DON[j]) and IS_H2O[i]):
						s = 3.0
					elif (IS_POLH[i] and IS_H2O[j]) or \
							(IS_POLH[j] and IS_H2O[i]):
						s = 1.95
					sigma[i][j] = s
			# precalc_etable_coefficients: closed-form fields
			inv_neg2_pi_sqrt_pi = -0.089793561062583294
			lk_inv_lambda2 = [(1.0 / LK_L[i])**2 for i in range(NTYPES)]
			lk_coeff_tmp = [inv_neg2_pi_sqrt_pi * LK_DG[i] / LK_L[i]
				for i in range(NTYPES)]
			lj_r6  = [[0.0]*NTYPES for _ in range(NTYPES)]
			lj_r12 = [[0.0]*NTYPES for _ in range(NTYPES)]
			lj_si  = [[0.0]*NTYPES for _ in range(NTYPES)]
			lj_ss  = [[0.0]*NTYPES for _ in range(NTYPES)]
			lk_coeff = [[0.0]*NTYPES for _ in range(NTYPES)]
			lk_min_dis2sigma_value = [[0.0]*NTYPES for _ in range(NTYPES)]
			for i in range(NTYPES):
				for j in range(NTYPES):
					s = sigma[i][j]
					s6 = s**6
					s12 = s6 * s6
					wd = _math.sqrt(LJ_W[i] * LJ_W[j])
					lj_r6[i][j]  = -2.0 * wd * s6
					lj_r12[i][j] = wd * s12
					# Rosetta default lj_use_lj_deriv_slope = true (Etable.cc:100)
					lj_ss[i][j] = (wd / s) * LJ_S2W
					lj_si[i][j] = wd * LJ_V2W - lj_ss[i][j] * s * LJ_SWITCH_D2S
					lk_coeff[i][j] = lk_coeff_tmp[i] * LK_V[j]
					# lk_min_dis2sigma_value at the switchover
					thresh_dis = LK_MIN_DIS2S * s
					inv_t2 = 1.0 / (thresh_dis * thresh_dis)
					dis_rad = thresh_dis - LJ_R[i]
					x_thresh = (dis_rad * dis_rad) * lk_inv_lambda2[i]
					lk_min_dis2sigma_value[i][j] = (
						_math.exp(-x_thresh) * lk_coeff[i][j] * inv_t2)
			# Per-pair LJ tabulation + smoothing + cubic fits
			pairs = [None] * (NTYPES * NTYPES)
			for is_ in range(NTYPES):
				ni = ETABLE_ATOM_TYPES[is_]
				for io_ in range(NTYPES):
					nj = ETABLE_ATOM_TYPES[io_]
					# Canonical at1 = lower index of (is_, io_); poly1 refers
					# to at1's desolvation regardless of arg order.
					# But pyrosetta's `analytic_params_for_pair` already
					# canonicalizes; we mirror that by computing (i, j)
					# where i = min, j = max, and then assigning self-vs-other
					# based on the (is_, io_) caller pair.
					i = min(is_, io_); j = max(is_, io_)
					s_ij = sigma[i][j]
					self_is_at1 = (is_ <= io_)
					pair = _etableonepair(
						i, j, is_, io_, ni, nj, self_is_at1,
						s_ij, LJ_R, LJ_W, LK_DG, LK_L, LK_V,
						IS_ACC, IS_DON, IS_POLH, IS_H2O,
						HYDRO, CARBON, OCBB_IDX,
						lj_r6, lj_r12, lj_si, lj_ss, lk_coeff,
						lk_inv_lambda2, lk_min_dis2sigma_value,
						DIS, ETABLE_DISBINS, BINS_PER_A2, MAX_DIS,
						MAX_DIS2, LJ_SWITCH_D2S, LK_MIN_DIS2S)
					pairs[is_ * NTYPES + io_] = pair
			return {'atom_types': list(ETABLE_ATOM_TYPES),
				'n_types': NTYPES, 'pairs': pairs}

		def _etableonepair(at1, at2, self_idx, other_idx, name_self, name_other,
				self_is_at1, sigma_pair, LJ_R, LJ_W, LK_DG, LK_L, LK_V,
				IS_ACC, IS_DON, IS_POLH, IS_H2O, HYDRO, CARBON, OCBB_IDX,
				lj_r6, lj_r12, lj_si, lj_ss, lk_coeff, lk_inv_lambda2,
				lk_min_dis2sigma_value, DIS, ETABLE_DISBINS, BINS_PER_A2,
				MAX_DIS, MAX_DIS2, LJ_SWITCH_D2S, LK_MIN_DIS2S):
			'''
			Compute one pair's EtableParamsOnePair fields. See _etableparams
			docstring for the algorithm overview.
			'''
			import math as _math
			# 1) Tabulate raw ljatr/ljrep/fasol1/fasol2 at all 721 distance bins
			# Bin 0 corresponds to dis=0 (Rosetta's dis(1)); the LJ uses the
			# linear ramp (intercept) and the LK uses the clamp value, so set
			# them explicitly rather than computing 1/dis at dis=0.
			ljatr  = [0.0] * ETABLE_DISBINS
			dljatr = [0.0] * ETABLE_DISBINS
			ljrep  = [0.0] * ETABLE_DISBINS
			dljrep = [0.0] * ETABLE_DISBINS
			fasol1 = [0.0] * ETABLE_DISBINS
			fasol2 = [0.0] * ETABLE_DISBINS
			dfasol1_arr = [0.0] * ETABLE_DISBINS
			# Bin 0: dis=0, dis2sigma=0 < LJ_SWITCH_D2S so linear ramp; LK clamp
			lj0 = lj_si[at1][at2]  # ljE at dis=0 = 0 * slope + intercept
			if lj0 < 0.0:
				ljatr[0] = lj0; dljatr[0] = lj_ss[at1][at2]
			else:
				ljrep[0] = lj0; dljrep[0] = lj_ss[at1][at2]
			fasol1[0] = lk_min_dis2sigma_value[at1][at2]
			fasol2[0] = lk_min_dis2sigma_value[at2][at1]
			for k in range(1, ETABLE_DISBINS):
				dis = DIS[k]
				inv_dis = 1.0 / dis
				inv_dis2 = inv_dis * inv_dis
				dis2sigma = dis / sigma_pair
				if dis2sigma < LJ_SWITCH_D2S:
					d_ljE = lj_ss[at1][at2]
					ljE = dis * d_ljE + lj_si[at1][at2]
				else:
					inv6  = inv_dis2**3
					inv7  = inv6 * inv_dis
					inv12 = inv6 * inv6
					inv13 = inv12 * inv_dis
					ljE = lj_r12[at1][at2] * inv12 + lj_r6[at1][at2] * inv6
					d_ljE = (-12.0 * lj_r12[at1][at2] * inv13
						- 6.0 * lj_r6[at1][at2] * inv7)
				if ljE < 0.0:
					ljatr[k] = ljE
					dljatr[k] = d_ljE
				else:
					ljrep[k] = ljE
					dljrep[k] = d_ljE
				# LK solvation
				if dis2sigma < LK_MIN_DIS2S:
					fasol1[k] = lk_min_dis2sigma_value[at1][at2]
					fasol2[k] = lk_min_dis2sigma_value[at2][at1]
					dfasol1_arr[k] = 0.0
				else:
					dr1 = dis - LJ_R[at1]
					x1 = (dr1 * dr1) * lk_inv_lambda2[at1]
					s1 = _math.exp(-x1) * lk_coeff[at1][at2] * inv_dis2
					fasol1[k] = s1
					dr2 = dis - LJ_R[at2]
					x2 = (dr2 * dr2) * lk_inv_lambda2[at2]
					s2 = _math.exp(-x2) * lk_coeff[at2][at1] * inv_dis2
					fasol2[k] = s2
					ds1 = -2.0 * s1 * (
						(dis - LJ_R[at1]) * lk_inv_lambda2[at1] + inv_dis)
					dfasol1_arr[k] = ds1
			# Disbin 0 (dis2=0): set to min_dis2 values (Rosetta clamps via
			# `if dis2 < min_dis2_ dis2 = min_dis2_`; default min_dis = 0).
			# For min_dis = 0 the formulas blow up; in practice bin 0 is
			# never used at scoring time. Leave at 0.
			# 2) modify_pot_one_pair: REF15 uses unmodifypot=true so the
			#    OCbb-OCbb extra quadratic repulsion is DISABLED. The xr fields
			#    remain at zero (pyrosetta returns zero with REF15 default).
			# Carbon-carbon close-flat fasol mod is still applied.
			ljrep_xr = {
				'xlo': 0.0, 'xhi': 0.0, 'slope': 0.0,
				'extrapolated_slope': 0.0, 'ylo': 0.0}
			# Carbon-carbon close-flat fasol mod (dis < 4.2 -> fasol fixed at
			# value at dis=4.2). Rosetta loops k=1..disbins which corresponds
			# to my k=0..disbins-1; include bin 0 since Rosetta does.
			if (name_self in CARBON) and (name_other in CARBON):
				ibin = int((4.2 * 4.2 / 0.05) + 1.0)
				f1_at_ibin = fasol1[ibin - 1]
				f2_at_ibin = fasol2[ibin - 1]
				for k in range(0, ETABLE_DISBINS):
					d = DIS[k]
					if d < 4.2:
						fasol1[k] = f1_at_ibin
						fasol2[k] = f2_at_ibin
						dfasol1_arr[k] = 0.0
			# REF15 default: fa_hatr=true, so zero_hydrogen_and_water_ljatr is
			# NOT called - hydrogens contribute attractive LJ normally and
			# hydrogen_interaction stays False, final_weight stays 1.0.
			hydrogen_pair = False
			# 3) smooth_etables_one_pair, part 1: find the LJ minimum bin
			min_atr = 0.0; which_min = -1
			for i in range(1, ETABLE_DISBINS):
				if ljatr[i] < min_atr:
					min_atr = ljatr[i]
					which_min = i
			if which_min != -1:
				lj_minimum = sigma_pair
				lj_val_at_minimum = -_math.sqrt(LJ_W[at1] * LJ_W[at2])
			else:
				lj_minimum = MAX_DIS
				lj_val_at_minimum = 0.0
			# Transfer ljatr to ljrep below the minimum
			if min_atr < 0.0 and which_min != -1:
				for i in range(1, which_min + 1):
					ljrep[i]  += ljatr[i] - min_atr
					dljrep[i] += dljatr[i]
					ljatr[i] = ljatr[i] - (ljatr[i] - min_atr)  # = min_atr
					dljatr[i] = 0.0
			# Compute start bin for ljatr cubic fit:
			# start = max(bin_of_dis((max_dis-1.5)*10), minima_bin_index+1)
			# bin_of_dis(i) = int((i/10)^2 * 20 + 1) for i in [1, etable_disbins]
			def _binofdis(ii):
				d = ii / 10.0
				return int(d * d * BINS_PER_A2 + 1)
			bod = _binofdis(int((MAX_DIS - 1.5) * 10.0))  # = 406 for max_dis=6
			start = max(bod, (which_min if which_min != -1 else 0) + 1)
			lbx = DIS[start - 1]  # dis(start) in 1-based = DIS[start-1] in 0-based
			ubx = DIS[ETABLE_DISBINS - 1]
			lby = ljatr[start - 1]
			lbdy = dljatr[start - 1]
			# Compute ljatr_cubic_poly via Hermite fit
			y2lo_atr, y2hi_atr = _splineddy2(lbx, lby, lbdy, ubx, 0.0, 0.0)
			ljatr_cp = _cubicfromspline(lbx, ubx, lby, 0.0, y2lo_atr, y2hi_atr)
			# Apply the spline to ljatr from start to end (so the smoothed
			# ljatr is used for fasol smoothing later — but fasol is
			# independent so this doesn't matter for the rest. We don't need
			# to overwrite ljatr.)
			# 4) Determine fasol close (S1, E1) and far (S2, E2) spline ranges
			S2 = bod  # 406 for max_dis=6
			E2 = ETABLE_DISBINS  # 721
			fasol_far_xlo = DIS[S2 - 1]
			fasol_far_xhi = DIS[E2 - 1]
			# SWTCH: first bin where fasol1 != fasol1(1) or fasol2 != fasol2(1)
			# (i.e., first bin where solvation departs from its constant
			# clamped value).
			SWTCH = ETABLE_DISBINS + 1
			for k in range(1, ETABLE_DISBINS):
				if fasol1[k] != fasol1[0] or fasol2[k] != fasol2[0]:
					SWTCH = k + 1  # match 1-based "bin index"
					break
			if SWTCH > ETABLE_DISBINS:
				# fasol constant everywhere - flat poly
				close_start = fasol_far_xhi
				close_end = fasol_far_xhi + 1.0
				close_flat = 0.0
				close_cp = _cubicfromspline(close_start, close_end,
					0.0, 0.0, 0.0, 0.0)
				far_cp = _cubicfromspline(fasol_far_xlo, fasol_far_xhi,
					0.0, 0.0, 0.0, 0.0)
				close_flat_self = 0.0
				close_cp_self = list(close_cp)
				far_cp_self = list(far_cp)
				final_weight = 1.0
				# Build the cell with all-zero LK-related fields
			else:
				S1 = max(1, SWTCH - 30)
				E1 = min(SWTCH + 20, 406)
				close_start = DIS[S1 - 1]
				close_end = DIS[E1 - 1]
				# Compute the close spline (combined fasol1 + fasol2 at E1).
				# Derivative at E1: forward diff (fasol1+fasol2)(E1+1) -
				# (fasol1+fasol2)(E1) / (DIS[E1] - DIS[E1-1]).
				dE1_d1 = ((fasol1[E1] - fasol1[E1 - 1])
					/ (DIS[E1] - DIS[E1 - 1]))
				dE1_d2 = ((fasol2[E1] - fasol2[E1 - 1])
					/ (DIS[E1] - DIS[E1 - 1]))
				dE1_total = dE1_d1 + dE1_d2
				ylo_c = fasol1[S1 - 1] + fasol2[S1 - 1]
				yhi_c = fasol1[E1 - 1] + fasol2[E1 - 1]
				y2lo_c, y2hi_c = _splineddy2(
					close_start, ylo_c, 0.0, close_end, yhi_c, dE1_total)
				close_flat = ylo_c
				close_cp = _cubicfromspline(close_start, close_end,
					ylo_c, yhi_c, y2lo_c, y2hi_c)
				# Compute per-direction (atom 1 and atom 2) close splines.
				# Boundary deriv at E1 = analytic dlk_solv(dis(E1)) for each.
				dis_E1 = DIS[E1 - 1]
				def _lksolvderiv(at_self, at_other, d):
					inv_d = 1.0 / d
					inv_d2 = inv_d * inv_d
					dr = d - LJ_R[at_self]
					x = (dr * dr) * lk_inv_lambda2[at_self]
					s = _math.exp(-x) * lk_coeff[at_self][at_other] * inv_d2
					ds = -2.0 * s * (
						(d - LJ_R[at_self]) * lk_inv_lambda2[at_self] + inv_d)
					return ds
				dsE1_1 = _lksolvderiv(at1, at2, dis_E1)
				dsE1_2 = _lksolvderiv(at2, at1, dis_E1)
				# Close poly for atom1's desolvation
				y2lo_1c, y2hi_1c = _splineddy2(
					close_start, fasol1[S1 - 1], 0.0,
					close_end, fasol1[E1 - 1], dsE1_1)
				close_flat_1 = fasol1[S1 - 1]
				close_cp_1 = _cubicfromspline(close_start, close_end,
					fasol1[S1 - 1], fasol1[E1 - 1], y2lo_1c, y2hi_1c)
				# Close poly for atom2's desolvation
				y2lo_2c, y2hi_2c = _splineddy2(
					close_start, fasol2[S1 - 1], 0.0,
					close_end, fasol2[E1 - 1], dsE1_2)
				close_flat_2 = fasol2[S1 - 1]
				close_cp_2 = _cubicfromspline(close_start, close_end,
					fasol2[S1 - 1], fasol2[E1 - 1], y2lo_2c, y2hi_2c)
				# Far spline: from S2 to E2. Rosetta uses FORWARD-DIFFERENCE
				# derivatives at S2 (see Etable.cc:987-989). The combined
				# far spline takes dfasol(S2) = dsolv1s2 + dsolv2s2 where
				# dsolv*s2 = (fasol*(S2) - fasol*(S2-1)) / (dis(S2)-dis(S2-1)).
				dis_S2 = DIS[S2 - 1]
				dsS2_1 = ((fasol1[S2 - 1] - fasol1[S2 - 2])
					/ (DIS[S2 - 1] - DIS[S2 - 2]))
				dsS2_2 = ((fasol2[S2 - 1] - fasol2[S2 - 2])
					/ (DIS[S2 - 1] - DIS[S2 - 2]))
				dfasolS2 = dsS2_1 + dsS2_2
				ylo_f = fasol1[S2 - 1] + fasol2[S2 - 1]
				y2lo_f, y2hi_f = _splineddy2(
					fasol_far_xlo, ylo_f, dfasolS2,
					fasol_far_xhi, 0.0, 0.0)
				far_cp = _cubicfromspline(fasol_far_xlo, fasol_far_xhi,
					ylo_f, 0.0, y2lo_f, y2hi_f)
				# Per-direction far splines use analytic LK derivative at S2
				# (Etable.cc:1111: lk_solv_energy_and_deriv(at, at, dis(S2)))
				dsS2_1_a = _lksolvderiv(at1, at2, dis_S2)
				dsS2_2_a = _lksolvderiv(at2, at1, dis_S2)
				y2lo_1f, y2hi_1f = _splineddy2(
					fasol_far_xlo, fasol1[S2 - 1], dsS2_1_a,
					fasol_far_xhi, 0.0, 0.0)
				far_cp_1 = _cubicfromspline(fasol_far_xlo, fasol_far_xhi,
					fasol1[S2 - 1], 0.0, y2lo_1f, y2hi_1f)
				y2lo_2f, y2hi_2f = _splineddy2(
					fasol_far_xlo, fasol2[S2 - 1], dsS2_2_a,
					fasol_far_xhi, 0.0, 0.0)
				far_cp_2 = _cubicfromspline(fasol_far_xlo, fasol_far_xhi,
					fasol2[S2 - 1], 0.0, y2lo_2f, y2hi_2f)
				# Assign per-direction polys: self vs other
				if self_is_at1:
					close_flat_self = close_flat_1
					close_cp_self = close_cp_1
					far_cp_self = far_cp_1
					lkc_self = lk_coeff[at1][at2]
				else:
					close_flat_self = close_flat_2
					close_cp_self = close_cp_2
					far_cp_self = far_cp_2
					lkc_self = lk_coeff[at2][at1]
				final_weight = 1.0
			# Hydrogen-pair / water-pair: zero ljatr_final_weight & fasol_final
			ljatr_final_weight = 0.0 if hydrogen_pair else 1.0
			fasol_final_weight = 0.0 if hydrogen_pair else 1.0
			# Pick lkc/lambda/R for self
			at_self = self_idx
			at_other = other_idx
			lkc_self = lk_coeff[at_self][at_other]
			lam_self = LK_L[at_self]
			R_self = LJ_R[at_self]
			# ljrep_linear_ramp_d2_cutoff
			ljrep_lr_d2 = (LJ_SWITCH_D2S * sigma_pair) ** 2
			# ljrep_from_negcrossing: not implemented (REPLS atom types are
			# excluded from ETABLE_ATOM_TYPES). hydrogen_interaction true if
			# either atom is H.
			cell = {
				'close_start': close_start,
				'close_end':   close_end,
				'close_flat':  close_flat_self,
				'close_poly':  close_cp_self,
				'far_poly':    far_cp_self,
				'lk_coeff':    lkc_self,
				'lambda_self': lam_self,
				'R_self':      R_self,
				'final_weight': fasol_final_weight,
				'close_flat_comb':  close_flat,
				'close_poly_comb':  list(close_cp),
				'far_poly_comb':    list(far_cp),
				# LJ:
				'lj_minimum':         lj_minimum,
				'lj_r12_coeff':       lj_r12[at1][at2],
				'lj_r6_coeff':        lj_r6[at1][at2],
				'lj_switch_intercept': lj_si[at1][at2],
				'lj_switch_slope':    lj_ss[at1][at2],
				'lj_val_at_minimum':  lj_val_at_minimum,
				'lj_min_dis2sigma_value': 0.0,
				'ljatr_cubic_poly':      list(ljatr_cp),
				'ljatr_cubic_poly_xhi':  ubx,
				'ljatr_cubic_poly_xlo':  lbx,
				'ljatr_final_weight':    ljatr_final_weight,
				'ljrep_linear_ramp_d2_cutoff': ljrep_lr_d2,
				'ljrep_from_negcrossing': False,
				'hydrogen_interaction':   hydrogen_pair,
				'ljrep_xr_xlo':    ljrep_xr['xlo'],
				'ljrep_xr_xhi':    ljrep_xr['xhi'],
				'ljrep_xr_slope':  ljrep_xr['slope'],
				'ljrep_xr_extrapolated_slope':
					ljrep_xr['extrapolated_slope'],
				'ljrep_xr_ylo':    ljrep_xr['ylo']}
			return cell

		KCAL_TO_KJ = 4.184
		# Pin to the Rosetta commit that pyrosetta 2026.03 was built from
		# so data is consistent across all REF15 tables (atom_properties,
		# residue topologies, hbond CSVs, rama tables, rotamer libs, etc.).
		# main HEAD has drifted from this commit for HBond/Rama tables.
		REPO = ('https://raw.githubusercontent.com/'
			'RosettaCommons/rosetta/'
			'5e498f1409c68ade56c8ce5842bf79e1b02e8db4/database/')
		def fetch(path):
			'''
			Download one repository file as a UTF-8 string
			Arguments:
			----------
				path: str - path under the database/ root
			Returns:
			--------
				str: file contents
			'''
			with urllib.request.urlopen(REPO + path, timeout=120) as r:
				return r.read().decode('utf-8')
		def fetchgz(path):
			'''
			Download one repository .gz file and return decompressed text
			Arguments:
			----------
				path: str - path under the database/ root
			Returns:
			--------
				str: decompressed file contents
			'''
			with urllib.request.urlopen(REPO + path, timeout=120) as r:
				return gzip.decompress(r.read()).decode('utf-8')
		# 1. atom_properties.txt -> per-Rosetta-type LJ/LK parameters
		props_txt = fetch(
			'chemical/atom_type_sets/fa_standard/atom_properties.txt')
		atom_types = {}
		# Fixed-width columns sometimes have adjacent numbers run together
		# (e.g. "0.161725-20.864641"); extract numbers with regex so a
		# missing space doesn't drop the line.
		for line in props_txt.splitlines():
			s = line.split('#', 1)[0]
			if not s.strip() or s.startswith('NAME'): continue
			toks = s.split()
			if len(toks) < 2: continue
			name = toks[0]
			nums = re.findall(r'[+-]?\d+\.\d+', s)
			if len(nums) < 5: continue
			try:
				lj_r, lj_w, lk_dG, lk_lam, lk_V = (float(x) for x in nums[:5])
			except ValueError: continue
			# Reconstruct flag list by stripping the numeric prefix.
			# Take everything after the LK_VOLUME number on the line.
			idx = 0
			for k in range(5):
				pos = s.find(nums[k], idx)
				if pos < 0: break
				idx = pos + len(nums[k])
			flags = set(t.upper() for t in s[idx:].split()
				if t and not t.startswith('#'))
			atom_types[name] = {
				'element':   toks[1],
				'LJ_RADIUS': lj_r,
				'LJ_WDEPTH': lj_w,
				'LK_DGFREE': lk_dG,
				'LK_LAMBDA': lk_lam,
				'LK_VOLUME': lk_V,
				'acceptor':  'ACCEPTOR' in flags,
				'donor':     'DONOR' in flags,
				'aromatic':  'AROMATIC' in flags,
				'sp2':       'SP2_HYBRID' in flags,
				'sp3':       'SP3_HYBRID' in flags,
				'ring':      'RING_HYBRID' in flags,
				'orbitals':  'ORBITALS' in flags,
				'polar_h':   'POLAR_HYDROGEN' in flags}
		# 1b. lk_ball atom weights -> per-Rosetta-type (iso, ball) pair.
		# Rosetta ships three variants; LK_BallInfo.cc pins the tag
		# "_RATIO23.0_DEFAULT", so that is the file REF15 actually reads.
		# Columns are NAME, ball weight, iso weight; they are stored here
		# swapped to (iso, ball) to match how ScoreMatch unpacks them. The
		# generic "****" row is the zero default and is skipped, since an
		# atom type absent from the table already scores zero.
		lkb_txt = fetch('chemical/atom_type_sets/fa_standard/extras/'
			'lk_ball_wtd_RATIO23.0_DEFAULT.txt')
		lkb_wts = {}
		for line in lkb_txt.splitlines():
			s = line.split('#', 1)[0]
			toks = s.split()
			if len(toks) < 3 or s.lstrip().startswith('NAME'): continue
			if toks[0] == '****': continue
			try:
				ball, iso = float(toks[1]), float(toks[2])
			except ValueError: continue
			lkb_wts[toks[0]] = [iso, ball]
		# 1c. Terminal-variant and disulfide partial charges, the proline
		# ring-closure geometry, and the sp2 hydrogen-bond shape params.
		# These are read from the patch files and option defaults rather
		# than transcribed, so that no Rosetta value lives in Pose's code.
		def patchcases(text):
			'''
			Split a Rosetta patch file into its BEGIN_CASE blocks
			Arguments:
			----------
				text: str - contents of a .txt patch file
			Returns:
			--------
				list: the body of each case, in file order; Rosetta takes
				the first matching case, so the generic one is written last
			'''
			return re.split(r'^BEGIN_CASE', text, flags=re.M)[1:]
		def casecharges(case):
			'''
			Read the added-hydrogen and reassigned charges of one case
			Arguments:
			----------
				case: str - the body of a single BEGIN_CASE block
			Returns:
			--------
				dict: atom name -> charge, using 'H' for the added
				terminal hydrogens and 'HA' for either HA or 1HA
			'''
			out = {}
			m = re.search(r'^ADD_ATOM 1H\s+\S+\s+\S+\s+(-?[\d.]+)',
				case, re.M)
			if m: out['H'] = float(m.group(1))
			for nm in ('N', 'CA', 'HA', '1HA'):
				m = re.search(r'^SET_ATOMIC_CHARGE %s\s+(-?[\d.]+)' % nm,
					case, re.M)
				if m: out['HA' if nm == '1HA' else nm] = float(m.group(1))
			return out
		nterm_txt = fetch('chemical/residue_type_sets/fa_standard/'
			'patches/NtermProteinFull.txt')
		cterm_txt = fetch('chemical/residue_type_sets/fa_standard/'
			'patches/CtermProteinFull.txt')
		cys_txt = fetch('chemical/residue_type_sets/fa_standard/'
			'residue_types/l-caa/CYS.params')
		ncases = patchcases(nterm_txt)
		pro_case = next(c for c in ncases
			if re.search(r'^NAME3 PRO\s*$', c, re.M))
		gly_case = next(c for c in ncases
			if re.search(r'^AA GLY\s*$', c, re.M))
		term_q = {'PRO': casecharges(pro_case),
			'GLY': casecharges(gly_case),
			'generic': casecharges(ncases[-1])}
		ccase = patchcases(cterm_txt)[-1]
		term_q['cterm'] = {nm: float(re.search(
			r'^SET_ATOMIC_CHARGE %s\s+(-?[\d.]+)' % nm, ccase, re.M).group(1))
			for nm in ('C', 'O')}
		term_q['disulfide_SG'] = float(re.search(
			r'^ATOM\s+SG\s+\S+\s+\S+\s+(-?[\d.]+)', cys_txt, re.M).group(1))
		# CAV virtual-atom placement, stated once per proline case
		m = re.search(r'^SET_ICOOR CAV\s+\S+\s+([\d.]+)\s+([\d.]+)',
			pro_case, re.M)
		proclose = {'cav_theta': float(m.group(1)),
			'cav_d': float(m.group(2))}
		# The four proline chi4 tethers are C++ member initialisers, so they
		# come from the source tree rather than from database/.
		pce_url = (REPO.replace('/database/', '/source/src/')
			+ 'core/energy_methods/ProClosureEnergy.cc')
		with urllib.request.urlopen(pce_url, timeout=120) as resp:
			pce_txt = resp.read().decode('utf-8')
		for nm, key in (('trans_chi4_mean_', 'trans_chi4_mean'),
				('trans_chi4_sd_',   'trans_chi4_sd'),
				('cis_chi4_mean_',   'cis_chi4_mean'),
				('cis_chi4_sd_',     'cis_chi4_sd')):
			m = re.search(nm + r'\(\s*(-?[\d.]+)', pce_txt)
			if m is None:
				raise RuntimeError('port: %s not found in %s'
					% (nm, pce_url))
			proclose[key] = float(m.group(1))
		# NV is the virtual nitrogen that closes the proline ring; its
		# placement is an internal coordinate of the residue itself.
		pro_txt = fetch('chemical/residue_type_sets/fa_standard/'
			'residue_types/l-caa/PRO.params')
		m = re.search(r'^ICOOR_INTERNAL\s+NV\s+\S+\s+([\d.]+)\s+([\d.]+)',
			pro_txt, re.M)
		if m is None:
			raise RuntimeError('port: NV ICOOR not found in PRO.params')
		proclose['nv_theta'] = float(m.group(1))
		proclose['nv_d'] = float(m.group(2))
		# dslf_fa13: von Mises / skew-normal fits, set in the
		# FullatomDisulfideParams13 constructor.
		dsl_url = (REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/disulfides/FullatomDisulfidePotential.hh')
		with urllib.request.urlopen(dsl_url, timeout=120) as resp:
			dsl_txt = resp.read().decode('utf-8')
		dslf = {}
		for nm in ('d_location', 'd_scale', 'd_shape',
				'a_logA', 'a_kappa', 'a_mu',
				'dss_logA1', 'dss_kappa1', 'dss_mu1',
				'dss_logA2', 'dss_kappa2', 'dss_mu2',
				'dcs_logA1', 'dcs_mu1', 'dcs_kappa1',
				'dcs_logA2', 'dcs_mu2', 'dcs_kappa2',
				'dcs_logA3', 'dcs_mu3', 'dcs_kappa3'):
			m = re.search(r'(?<![\w])' + nm + r'\s*=\s*(-?[\d.]+)', dsl_txt)
			if m is None:
				raise RuntimeError('port: %s not found in %s'
					% (nm, dsl_url))
			dslf[nm] = float(m.group(1))
		# The sp2 chi/BAH slope is a literal in the hbond kernel, selected
		# by fade_energy; REF15 runs with fading on, which gives 1.6.
		hbg_url = (REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/hbonds/hbonds_geom.cc')
		with urllib.request.urlopen(hbg_url, timeout=120) as resp:
			hbg_txt = resp.read().decode('utf-8')
		m = re.search(r'fade_energy\(\)\s*\?\s*([\d.]+)\s*:\s*([\d.]+)',
			hbg_txt)
		if m is None:
			raise RuntimeError('port: sp2 fade slope not found in %s'
				% hbg_url)
		sp2_slope = float(m.group(1))
		# sp2 hydrogen-bond shape parameters, from the option defaults
		opt_url = (REPO.replace('/database/', '/source/src/')
			+ 'basic/options/options_rosetta.py')
		with urllib.request.urlopen(opt_url, timeout=120) as resp:
			opt_txt = resp.read().decode('utf-8')
		hb_sp2 = {}
		for nm, key in (('hb_sp2_BAH180_rise', 'BAH180_rise'),
				('hb_sp2_outer_width', 'outer_width')):
			m = re.search(r"Option\(\s*'" + nm
				+ r"'.*?default\s*=\s*\"?'?([\d.]+)", opt_txt, re.S)
			if m is None:
				raise RuntimeError('port: %s not found in %s'
					% (nm, opt_url))
			hb_sp2[key] = float(m.group(1))
		hb_sp2['fade_slope'] = sp2_slope
		m2 = re.search(r"pro_close_planar_constraint'.*?default\s*=\s*"
			r"[\"']?([\d.]+)", opt_txt, re.S)
		proclose['planar_sd'] = float(m2.group(1)) if m2 else 0.1
		# sp3 chi penalty magnitude and the energy-fading polynomial, both
		# literals in the hbond kernel; and the burial ramp used when
		# smooth_hb_env_dep is on, which is the REF15 default.
		m = re.search(r'max_penalty\s*=\s*([\d.]+)', hbg_txt)
		if m is None:
			raise RuntimeError('port: max_penalty not found in %s' % hbg_url)
		hb_sp2['max_penalty'] = float(m.group(1))
		m = re.search(r'energy\s*=\s*(-?[\d.]+)\s*\+\s*([\d.]+)\s*\*\s*energy'
			r'\s*-\s*([\d.]+)\s*\*\s*energy\s*\*\s*energy', hbg_txt)
		if m is None:
			raise RuntimeError('port: fade polynomial not found in %s'
				% hbg_url)
		hb_sp2['fade_c0'] = float(m.group(1))
		hb_sp2['fade_c1'] = float(m.group(2))
		hb_sp2['fade_c2'] = -float(m.group(3))
		hbc_url = (REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/hbonds/hbonds.cc')
		with urllib.request.urlopen(hbc_url, timeout=120) as resp:
			hbc_txt = resp.read().decode('utf-8')
		m = re.search(r'burial_weight\(int const nb\)\s*\{(.*?)\n\}',
			hbc_txt, re.S)
		if m is None:
			raise RuntimeError('port: burial_weight not found in %s'
				% hbc_url)
		body = m.group(1)
		lo = re.search(r'nb\s*<\s*(\d+)\s*\)\s*return\s+([\d.]+)', body)
		hi = re.search(r'nb\s*>\s*(\d+)\s*\)\s*return\s+([\d.]+)', body)
		mid = re.search(r'\(nb\s*-\s*([\d.]+)\s*\)\s*\*\s*\(([\d.]+)\s*/\s*'
			r'([\d.]+)\s*\)', body)
		if not (lo and hi and mid):
			raise RuntimeError('port: burial_weight body unparsed')
		hb_burial = {'nb_lo': int(lo.group(1)), 'w_lo': float(lo.group(2)),
			'nb_hi': int(hi.group(1)), 'w_hi': float(hi.group(2)),
			'shift': float(mid.group(1)),
			'slope': float(mid.group(2)) / float(mid.group(3))}
		hb_sp2['burial'] = hb_burial
		# lk_ball water geometry and ramp width
		lki_url = (REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/lkball/LK_BallInfo.cc')
		with urllib.request.urlopen(lki_url, timeout=120) as resp:
			lki_txt = resp.read().decode('utf-8')
		m = re.search(r'optimal_water_distance\(\s*([\d.]+)', lki_txt)
		if m is None:
			raise RuntimeError('port: optimal_water_distance not found')
		lkball = {'opt_dist': float(m.group(1))}
		m = re.search(r"lk_ball_ramp_width_A2'.*?default\s*=\s*[\"']?([\d.]+)",
			opt_txt, re.S)
		lkball['ramp_w2'] = float(m.group(1)) if m else 3.9
		# The LK far-region switch is derived, not independent: Etable takes
		# the bin at (max_dis - 1.5).
		etb_url = (REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/etable/Etable.cc')
		with urllib.request.urlopen(etb_url, timeout=120) as resp:
			etb_txt = resp.read().decode('utf-8')
		m = re.search(r'\(\s*max_dis_\s*-\s*([\d.]+)\s*\)\s*\*\s*10\.0',
			etb_txt)
		far_off = float(m.group(1)) if m else 1.5
		# Store the switch bounds outright: Score adds a Verlet skin to
		# Constants['fa_max_dis'] at construction, so that key must not be
		# used to derive the etable region boundaries.
		lkball['far_offset'] = far_off
		m = re.search(r"Option\(\s*'fa_max_dis'.*?default\s*=\s*"
			r"[\"']?([\d.]+)", opt_txt, re.S)
		if m is None:
			raise RuntimeError('port: fa_max_dis default not found')
		lkball['max_dis'] = float(m.group(1))
		lkball['far_lo'] = lkball['max_dis'] - far_off
		lke_url = (REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/lkball/LK_BallEnergy.cc')
		with urllib.request.urlopen(lke_url, timeout=120) as resp:
			lke_txt = resp.read().decode('utf-8')
		m = re.search(r'h2o_radius\(\s*([\d.]+)', lke_txt)
		if m is None:
			raise RuntimeError('port: h2o_radius not found in %s' % lke_url)
		lkball['h2o_radius'] = float(m.group(1))
		# The disulfide kernel floors its log with a minimum estimate.
		dslc_url = (REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/disulfides/FullatomDisulfidePotential.cc')
		with urllib.request.urlopen(dslc_url, timeout=120) as resp:
			dslc_txt = resp.read().decode('utf-8')
		m = re.search(r'mest_\s*=\s*exp\(\s*(-?[\d.]+)\s*\)', dslc_txt)
		if m is None:
			raise RuntimeError('port: mest_ not found in %s' % dslc_url)
		dslf['mest_log'] = float(m.group(1))
		for nm in ('wt_dihSS', 'wt_dihCS', 'wt_ang', 'wt_len'):
			m = re.search(nm + r'_\(\s*([\d.]+)\s*\)', dslc_txt)
			if m is None:
				raise RuntimeError('port: %s_ not found in %s'
					% (nm, dslc_url))
			dslf[nm] = float(m.group(1))
		# OmegaTether's per-residue weight.
		omg_url = (REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/OmegaTether.cc')
		with urllib.request.urlopen(omg_url, timeout=120) as resp:
			omg_txt = resp.read().decode('utf-8')
		m = re.search(r'Real\s+weight\s*=\s*([\d.]+)\s*;', omg_txt)
		if m is None:
			raise RuntimeError('port: omega weight not found in %s' % omg_url)
		omega_k = float(m.group(1))
		# Countpair path weight: bonded paths of 4 (3 for non-polymers)
		# score at half strength.
		cpf_url = (REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/etable/count_pair/CountPairFunction.cc')
		with urllib.request.urlopen(cpf_url, timeout=120) as resp:
			cpf_txt = resp.read().decode('utf-8')
		m = re.search(r'cp_half\(\s*([\d.]+)', cpf_txt)
		if m is None:
			raise RuntimeError('port: cp_half not found in %s' % cpf_url)
		cp_half = float(m.group(1))
		# Threshold that gates the hbond energy-fading polynomial.
		# Two thresholds gate the fade: energy is zeroed above fade_hi and
		# faded between fade_lo and fade_hi. Anchor on the else-if so the
		# lower bound is not confused with the upper one.
		m = re.search(r'if\s*\(\s*input_energy\s*>\s*([\d.]+)L?\s*\)',
			hbg_txt)
		m2 = re.search(r'else\s+if\s*\(\s*input_energy\s*>\s*'
			r'(-[\d.]+)L?\s*\)', hbg_txt)
		if m is None or m2 is None:
			raise RuntimeError('port: fade thresholds not found in %s'
				% hbg_url)
		hb_sp2['fade_hi'] = float(m.group(1))
		hb_sp2['fade_lo'] = float(m2.group(1))
		# Default LK lambda: the modal LK_LAMBDA of atom_properties.txt.
		lam = [float(x) for x in re.findall(
			r'^\S+\s+\S+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+([\d.]+)',
			props_txt, re.M)]
		if not lam:
			raise RuntimeError('port: no LK_LAMBDA column parsed')
		lk_lambda_default = max(set(lam), key=lam.count)
		# 2. ref2015.wts -> weight list + METHOD_WEIGHTS ref values
		wts_txt = fetch('scoring/weights/ref2015.wts')
		weights = {}
		method_ref = []
		for line in wts_txt.splitlines():
			s = line.strip()
			if not s or s.startswith('#'): continue
			if s.startswith('METHOD_WEIGHTS'):
				toks = s.split()
				if toks[1].lower() == 'ref':
					method_ref = [float(x) for x in toks[2:]]
				continue
			if s in ('INCLUDE_INTRA_RES_PROTEIN', 'NO_HB_ENV_DEP'): continue
			toks = s.split()
			if len(toks) >= 2:
				try: weights[toks[0]] = float(toks[1])
				except ValueError: pass
		# 3. residue topologies for the 20 standard amino acids (l-caa/).
		#    Each gives, per atom: rosetta-type + partial charge, plus bond
		#    list. Terminal variants and HIS_D added; nucleic-acid topology
		#    deferred until Phase 4 of the plan.
		AAS = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
			'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
		residues = {}
		for aa in AAS:
			txt = fetch(
				'chemical/residue_type_sets/fa_standard/residue_types/'
				'l-caa/%s.params' % aa)
			residues[aa] = _parseparams(txt)
		# HIS protonation variant (Rosetta HIS_D)
		try:
			txt = fetch(
				'chemical/residue_type_sets/fa_standard/residue_types/'
				'l-caa/HIS_D.params')
			residues['HIS_D'] = _parseparams(txt)
		except Exception:
			pass

		# 3a. The six non-canonical residues Pose ships. Rosetta supplies
		#     ornithine and 6-fluoro-tryptophan under l-ncaa/ and phosphate
		#     patches for Thr/Tyr; MSE and SEC have no upstream params and are
		#     derived from MET and CYS by the S -> Se substitution that
		#     Rosetta's own MET.params already anticipates (ATOM_ALIAS SD SE).
		#     Pose's templates are the NEUTRAL forms (NH2 amine, diprotonated
		#     phosphate) where Rosetta's are charged, so any group whose
		#     protonation differs is taken from a Rosetta canonical analogue
		#     and the residue renormalised to net zero. No value is invented.
		def ncaaresidues(res, fetch, parse):
			'''
			Build the six non-canonical residue templates Pose ships
			Arguments:
			----------
				res: dict - the canonical templates already parsed
				fetch: callable - Rosetta database file fetcher
				parse: callable - .params text parser
			Returns:
			--------
				dict: Pose tricode to template, same shape as res
			'''
			NC = ('chemical/residue_type_sets/fa_standard/'
				'residue_types/l-ncaa/')
			def rename(e, m, drop=()):
				a = dict((m.get(k, k), dict(v))
					for k, v in e['atoms'].items() if k not in drop)
				b = [[m.get(x, x), m.get(y, y), o]
					for x, y, o in e['bonds']
					if x not in drop and y not in drop]
				al = dict((k, m.get(v, v)) for k, v in e['aliases'].items()
					if k not in drop and v not in drop)
				return {'name': None, 'aa': None, 'atoms': a,
					'bonds': b, 'aliases': al}
			def renorm(a, onto):
				a[onto]['charge'] -= sum(x['charge'] for x in a.values())
			def finish(e, tri, al=None):
				e['name'] = tri
				e['aa'] = tri
				if al: e['aliases'].update(al)
				return e
			out = {}
			# Selenomethionine: methionine with SD renamed SE
			mse = rename(res['MET'], {'SD': 'SE'})
			mse['aliases'].pop('SE', None)
			out['MSE'] = finish(mse, 'MSE')
			# Selenocysteine: cysteine with SG -> SE and HG -> HE. Pose names
			# the two CB hydrogens HB1/HB2, so CYS's PDBv3 aliases (which map
			# HB2 -> 1HB) would mis-resolve and are replaced outright.
			sec = rename(res['CYS'], {'SG': 'SE', 'HG': 'HE'})
			sec['aliases'] = {'HB1': '1HB', 'HB2': '2HB'}
			out['SEC'] = finish(sec, 'SEC')
			# Ornithine, neutral amine. Rosetta's ornithine.params is the
			# NH3+ form with CHARMM-era charges, so the backbone and CH2 chain
			# are re-taken from LYS and the terminal NH2 from ASN's ND2/HD2x,
			# which is Rosetta's own neutral primary amine.
			orn = rename(parse(fetch(NC + 'ornithine.params')), {},
				drop=('3HE',))
			lys = res['LYS']['atoms']
			asn = res['ASN']['atoms']
			for nm, src, ty in (
					('N', 'N', 'Nbb'), ('CA', 'CA', 'CAbb'),
					('C', 'C', 'CObb'), ('O', 'O', 'OCbb'),
					('H', 'H', 'HNbb'), ('HA', 'HA', 'Hapo'),
					('CB', 'CB', 'CH2'), ('CG', 'CG', 'CH2'),
					('CD', 'CD', 'CH2'),
					('1HB', '1HB', 'Hapo'), ('2HB', '2HB', 'Hapo'),
					('1HG', '1HG', 'Hapo'), ('2HG', '2HG', 'Hapo'),
					('1HD', '1HD', 'Hapo'), ('2HD', '2HD', 'Hapo')):
				orn['atoms'][nm]['type'] = ty
				orn['atoms'][nm]['charge'] = lys[src]['charge']
			for nm, src, ty in (('NE', 'ND2', 'NH2O'),
					('1HE', '1HD2', 'Hpol'), ('2HE', '2HD2', 'Hpol')):
				orn['atoms'][nm]['type'] = ty
				orn['atoms'][nm]['charge'] = asn[src]['charge']
			renorm(orn['atoms'], 'NE')
			out['ORN'] = finish(orn, 'ORN')
			# 6-fluoro-L-tryptophan. Charges and types are Rosetta's own; only
			# the atom names are remapped onto Pose's CCD naming, and the map
			# comes from the bond graph: Rosetta CZ1 is the carbon bonded to
			# CE2 (PDB CZ2), CZ2 the one bonded to CE3 (PDB CZ3), CT the one
			# bearing the fluorine (PDB CH2, indole position 6).
			FT = {'CZ1': 'CZ2', 'CZ2': 'CZ3', 'CT': 'CH2', 'FI': 'F01',
				'1HD1': 'HD1', '1HE1': 'HE1', '1HE3': 'HE3',
				'1HZ1': 'HZ2', '1HZ2': 'HZ3'}
			tmp = dict((k, '@%d' % i) for i, k in enumerate(FT))
			ft = parse(fetch(NC + '6-fluoro-tryptophan.params'))
			ft = rename(rename(ft, tmp),
				dict((v, FT[k]) for k, v in tmp.items()))
			renorm(ft['atoms'], 'CH2')
			out['FT6'] = finish(ft, 'FT6')
			# Phosphothreonine and phosphotyrosine. Rosetta's patch gives the
			# dianion (P = Phos +1.50, three OOC at -0.78, hydroxyl proton
			# deleted). Pose keeps two P-OH protons, so O2P/O3P are hydroxyls
			# here: type OH with an Hpol proton, charges from SER's OG/HG.
			def phospho(base, bridge, hs, tri, al):
				e = rename(res[base], {},
					drop=({'THR': 'HG1', 'TYR': 'HH'}[base],))
				ser = res['SER']['atoms']
				e['atoms']['P'] = {'type': 'Phos', 'mm_type': 'X',
					'charge': 1.50}
				e['atoms']['O1P'] = {'type': 'OOC', 'mm_type': 'OC',
					'charge': -0.78}
				e['bonds'] += [[bridge, 'P', 1], ['P', 'O1P', 1]]
				for o, h in zip(('O2P', 'O3P'), hs):
					e['atoms'][o] = {'type': 'OH', 'mm_type': 'OH1',
						'charge': ser['OG']['charge']}
					e['atoms'][h] = {'type': 'Hpol', 'mm_type': 'H',
						'charge': ser['HG']['charge']}
					e['bonds'] += [['P', o, 1], [o, h, 1]]
				renorm(e['atoms'], 'P')
				return finish(e, tri, al)
			out['TPO'] = phospho('THR', 'OG1', ('1HOP', '2HOP'), 'TPO',
				{'1HG': '1HG2', '2HG': '2HG2', '3HG': '3HG2'})
			out['PTR'] = phospho('TYR', 'OH', ('HO2P', 'HO3P'), 'PTR',
				{'1HD': 'HD1', '2HD': 'HD2', '1HE': 'HE1', '2HE': 'HE2'})
			return out
		residues.update(ncaaresidues(residues, fetch, _parseparams))
		# 3b. Hbond polynomial / chem-type / eval / fade tables
		hb_dir = 'scoring/score_functions/hbonds/ref2015_params/'
		hb_files = ('HBPoly1D.csv', 'HBEval.csv', 'HBFadeIntervals.csv',
			'HBDonChemType.csv', 'HBAccChemType.csv',
			'HBAccHybridization.csv', 'HBSeqSep.csv', 'DonStrength.csv',
			'AccStrength.csv', 'HBondWeightType.csv', 'HybridizationType.csv')
		hb_raw = {}
		for fname in hb_files:
			try: hb_raw[fname] = fetch(hb_dir + fname)
			except Exception: hb_raw[fname] = ''
		hb_data = _parsehbonddata(hb_raw)
		# 3c. Rama tables (all + prepro). REF15 uses scoring/score_functions/
		# rama/fd/ (NOT fd_beta_nov2016 — that's for beta_nov16 score fn).
		rama_dir = 'scoring/score_functions/rama/fd/'
		rama_data = {}
		for kind, fname in (('all', 'all.ramaProb'),
				('prepro', 'prepro.ramaProb')):
			try:
				txt = fetch(rama_dir + fname)
				# Parse: each row "AA phi psi prob -log(prob)"
				# Build {aa: [[phi_idx][psi_idx]: -log(prob)]}
				table = {}
				for line in txt.splitlines():
					toks = line.split()
					if len(toks) < 5: continue
					aa = toks[0]
					try:
						phi = int(float(toks[1]))
						psi = int(float(toks[2]))
						nE = float(toks[4])
					except ValueError: continue
					# (phi, psi) on a 36x36 grid, -180..170 step 10
					i_phi = (phi + 180) // 10
					i_psi = (psi + 180) // 10
					if 0 <= i_phi < 36 and 0 <= i_psi < 36:
						t = table.setdefault(aa,
							[[0.0]*36 for _ in range(36)])
						t[i_phi][i_psi] = nE
				rama_data[kind] = table
			except Exception:
				rama_data[kind] = {}
		# 3d. Omega tables (mu/sigma per phi-psi cell, 4 sub-tables)
		omega_dir = 'scoring/score_functions/omega/'
		omega_tables = {}
		for kind in ('all', 'gly', 'pro', 'valile'):
			txt = fetch(omega_dir + 'omega_ppdep.' + kind + '.txt')
			mu = [[0.0] * 36 for _ in range(36)]
			sig = [[0.0] * 36 for _ in range(36)]
			for line in txt.splitlines():
				toks = line.split()
				if len(toks) < 6: continue
				try:
					ip = int(toks[0]); js = int(toks[1])
					mu_v = float(toks[4]); sig_v = float(toks[5])
				except ValueError: continue
				if 0 <= ip < 36 and 0 <= js < 36:
					mu[ip][js] = mu_v
					sig[ip][js] = sig_v
			omega_tables[kind] = {'mu': mu, 'sigma': sig}
		# 3e. P_AA (per-AA marginal frequency, scalar per AA)
		paa_txt = fetch('scoring/score_functions/P_AA_pp/P_AA')
		p_aa = {}
		for line in paa_txt.splitlines():
			toks = line.split()
			if len(toks) >= 2:
				try: p_aa[toks[0]] = float(toks[1])
				except ValueError: pass
		# 3h. FaDunNrchiDensities (8 semi-rotameric AAs).
		# Schema documented in former port_nrchi.py.
		NRCHI_AA = [
			('ASN', 2, 1), ('ASP', 2, 1), ('GLN', 3, 2), ('GLU', 3, 2),
			('HIS', 2, 1), ('PHE', 2, 1), ('TRP', 2, 1), ('TYR', 2, 1)]
		def parsenrchi(txt, n_chi, n_disc_chi):
			'''Parse one bbdep.densities.lib text into the per-rotwell
			36x36 grid layout expected by FaDunPotential.'''
			chi_last_low = None
			chi_last_step = None
			chi_last_n = None
			rows_by_rot = {}
			chi_cols_count = None
			for ln in txt.splitlines():
				s = ln.strip()
				if not s: continue
				if s.startswith('#'):
					if ('chi%d interval' % n_chi) in s:
						b1 = s.find('['); b2 = s.find(']')
						lo, _ = s[b1+1:b2].split(',')
						chi_last_low = float(lo)
					elif ('chi%d step' % n_chi) in s:
						tabs = s.split('\t')
						chi_last_step = float(tabs[-1])
					continue
				parts = s.split('\t')
				if len(parts) < 4:
					parts = s.split()
				phi = float(parts[1]); psi = float(parts[2])
				rot_idx = []
				off = 4
				for k in range(n_disc_chi):
					rot_idx.append(int(parts[off + k]))
				off += n_disc_chi
				P_rot = float(parts[off]); off += 1
				negP_rot = float(parts[off]); off += 1
				chi_means = [float(parts[off + k])
					for k in range(n_disc_chi)]
				off += n_disc_chi
				chi_sigmas = [float(parts[off + k])
					for k in range(n_disc_chi)]
				off += n_disc_chi
				densities = [float(x) for x in parts[off:]]
				if chi_cols_count is None:
					chi_cols_count = len(densities)
					chi_last_n = chi_cols_count
				rows_by_rot.setdefault(tuple(rot_idx), []).append(
					(phi, psi, P_rot, negP_rot, chi_means,
						chi_sigmas, densities))
			MAXE = 13.815510557964274
			per_rot = {}
			for rot_key, rows in rows_by_rot.items():
				P_rot_grid = [0.0] * (36 * 36)
				neglogP_rot_grid = [MAXE] * (36 * 36)
				chi_mean_grid = [[0.0] * (36 * 36)
					for _ in range(n_disc_chi)]
				chi_sigma_grid = [[1.0] * (36 * 36)
					for _ in range(n_disc_chi)]
				dens_grid = [0.0] * (36 * 36 * chi_last_n)
				for phi, psi, P_rot, neg_P, cmeans, csigmas, dens in rows:
					pi = int(round((phi + 180.0) / 10.0)) % 36
					ps = int(round((psi + 180.0) / 10.0)) % 36
					cell = pi * 36 + ps
					P_rot_grid[cell] = P_rot
					neglogP_rot_grid[cell] = neg_P
					for k in range(n_disc_chi):
						chi_mean_grid[k][cell] = cmeans[k]
						chi_sigma_grid[k][cell] = csigmas[k]
					for j, d_val in enumerate(dens):
						dens_grid[cell * chi_last_n + j] = d_val
				rot_key_str = ','.join(str(x) for x in rot_key)
				per_rot[rot_key_str] = {
					'P_rot':       P_rot_grid,
					'neglogP_rot': neglogP_rot_grid,
					'chi_means':   [x for sub in chi_mean_grid for x in sub],
					'chi_sigmas':  [x for sub in chi_sigma_grid for x in sub],
					'densities':   dens_grid}
			return {
				'chi_last_low':  chi_last_low,
				'chi_last_step': chi_last_step,
				'chi_last_n':    chi_last_n,
				'n_chi':         n_chi,
				'n_disc_chi':    n_disc_chi,
				'rotwells':      sorted(per_rot.keys()),
				'phi_step': 10.0, 'psi_step': 10.0,
				'phi_n': 36, 'psi_n': 36,
				'per_rot': per_rot}
		nrchi_db = {}
		for aa3, n_chi, n_disc_chi in NRCHI_AA:
			gz_path = ('rotamer/shapovalov/StpDwn_0-0-0/'
				+ aa3.lower() + '.bbdep.densities.lib.gz')
			txt = fetchgz(gz_path)
			nrchi_db[aa3] = parsenrchi(txt, n_chi, n_disc_chi)
		# 3f. P_AA_pp (Shapovalov a20.prop, 10° kappa=131 propensities)
		# Format per row: "phi\tpsi\tAA\tprop\t-log(prop)" with phi/psi in
		# [-180, 170] step 10° (grid starts at -180).
		prop_txt = fetch('scoring/score_functions/P_AA_pp/shapovalov/'
			'10deg/kappa131/a20.prop')
		p_aa_pp = {}
		for line in prop_txt.splitlines():
			if line.startswith('#'): continue
			toks = line.split()
			if len(toks) < 4: continue
			try:
				phi = float(toks[0]); psi = float(toks[1])
				aa = toks[2]; prop = float(toks[3])
			except ValueError: continue
			ip = int(round((phi + 180.0) / 10.0))
			js = int(round((psi + 180.0) / 10.0))
			if ip >= 36: ip -= 36
			if js >= 36: js -= 36
			if 0 <= ip < 36 and 0 <= js < 36:
				t = p_aa_pp.setdefault(aa,
					[[0.0] * 36 for _ in range(36)])
				t[ip][js] = prop
		# 4. Assemble the parameter block. All numeric weights x4.184 so
		#    internal storage is uniform kJ/mol; a Constants.scale of
		#    1/4.184 is applied at Score.__call__ exit to return REU.
		def w(key, dflt=0.0):
			return weights.get(key, dflt) * KCAL_TO_KJ
		block = {
			'Constants': {
				'scale':           1.0 / KCAL_TO_KJ,
				'fa_max_dis':      6.0,
				'fa_elec_max_dis': 5.5,
				'fa_elec_min_dis': 1.6,
				'fa_atr_short':    4.5,
				'fa_atr_long':     6.0,
				'lj_hbond_OH':     0.6,
				'eps_core':        6.0,
				'eps_solvent':     80.0,
				'coulomb_C0':      322.0637,
				'sigmoidal_D':     80.0,
				'sigmoidal_D0':    6.0,
				'sigmoidal_S':     0.4,
				'connectivity_weight': {'3': 0.0, '4': 0.2, '5+': 1.0}},
			'TerminalCharges': term_q,
			'HBondSp2':        hb_sp2,
			'LkBall':          dict(lkball, lk_lambda_default=lk_lambda_default),
			'CountPair':       {'half': cp_half},
			'Atom_types':    atom_types,
			'Residue_types': residues,
			'HBond_data':    hb_data,
			'Rama_data':     rama_data,
			'Omega_tables':  omega_tables,
			'P_AA':          p_aa,
			'P_AA_pp':       p_aa_pp,
			'P_AA_pp_grid_start': -180.0,
			'METHOD_WEIGHTS_ref': method_ref,
			'FaAtr':              {'weight': w('fa_atr',              1.000)},
			'FaRep':              {'weight': w('fa_rep',              0.550)},
			'FaSol':              {'weight': w('fa_sol',              1.000)},
			'FaIntraRep':         {'weight': w('fa_intra_rep',        0.005)},
			'FaIntraSolXover4':   {'weight': w('fa_intra_sol_xover4', 1.000)},
			'LkBallWtd':          {'weight': w('lk_ball_wtd',         1.000),
				'atom_weights': lkb_wts},
			'FaElec':             {'weight': w('fa_elec',             1.000)},
			'HBondSrBb':          {'weight': w('hbond_sr_bb',         1.000)},
			'HBondLrBb':          {'weight': w('hbond_lr_bb',         1.000)},
			'HBondBbSc':          {'weight': w('hbond_bb_sc',         1.000)},
			'HBondSc':            {'weight': w('hbond_sc',            1.000)},
			'DslfFa13':           dict(dslf,
				weight=w('dslf_fa13',           1.250)),
			'Omega':              {'weight': w('omega',               0.400),
				'tether_k': omega_k,
				# Value used for a terminal phi/psi that has no defining
				# atoms. Rosetta has no constant here: a pose read from a PDB
				# reports 0 for the first residue's phi, while one built from
				# sequence reports 180 because the ideal build sets it. 0.0
				# matches the imported case, which is what gets scored.
				'undefined_torsion': 0.0},
			'FaDun':              {'weight': w('fa_dun',              0.700)},
			'PAaPp':              {'weight': w('p_aa_pp',             0.600)},
			'YhhPlanarity':       {'weight': w('yhh_planarity',       0.625)},
			'Ref':                {'weight': w('ref',                 1.000)},
			'RamaPreProTerm':     {'weight': w('rama_prepro',         0.450)},
			'ProClose':           dict(proclose,
				weight=w('pro_close',           1.250)),
			# 0-weight inactive terms exposed in the schema (user spec)
			'FaIntraAtr':         {'weight': 0.0},
			'LkBallIso':          {'weight': 0.0},
			'LkBallBridge':       {'weight': 0.0},
			'CartBonded':         {'weight': 0.0},
			# Phase 2 implements the physics block only; later phases add the
			# rest of the methods. The Terms list is the same shape as
			# ForceField/Score - it drives __call__'s dispatch.
			'Terms': [
				['FaAtrPotential',                {}],
				['FaRepPotential',                {}],
				['FaSolPotential',                {}],
				['FaIntraRepPotential',           {}],
				['FaIntraSolXover4Potential',     {}],
				['LkBallWtdPotential',            {}],
				['FaElecPotential',               {}],
				['HBondSrBbPotential',            {}],
				['HBondLrBbPotential',            {}],
				['HBondBbScPotential',            {}],
				['HBondScPotential',              {}],
				['FaDunPotential',                {}],
				['RamaPreProTermPotential',       {}],
				['PAaPpPotential',                {}],
				['OmegaPotential',                {}],
				['ProClosePotential',             {}],
				['DslfFa13Potential',             {}],
				['YhhPlanarityPotential',         {}],
				['RefPotential',                  {}]]}
		sp['REF15'] = block
		sp['REF15']['FaDunNrchiDensities'] = nrchi_db
		# EtablePairParams (pure-Python LJ/LK analytic fit). It lives inside
		# the REF15 block rather than at the top level so that every
		# Rosetta-derived value sits under 'Score Parameters' and can be
		# added or removed as one unit.
		sp['REF15']['EtablePairParams'] = _etableparams(atom_types)
	dispatch = {
		'OPENFF':        openff,
		'FF19SB':        ff19sb,
		'CHARMM36':      charmm36,
		'AUTODOCK VINA': vina,
		'REF15':         ref15,
	}
	if key not in dispatch:
		raise ValueError(
			'Port: unknown name=%r (available: %r)'
			% (name, sorted(dispatch)))
	dispatch[key]()
	tmp = db_path + '.tmp'
	try:
		with open(tmp, 'w') as f:
			json.dump(db, f, separators=(',', ':'))
		os.replace(tmp, db_path)
	except BaseException:
		if os.path.exists(tmp): os.remove(tmp)
		raise
	try: DBLoad.cache_clear()
	except Exception: pass
	return True

def Cyclise(pose, mode='head-to-tail',
	res1=None, atom1=None, res2=None, atom2=None, precoil=True):
	'''
	Form an intramolecular bond to build a cyclic peptide (macrocycle)
	Arguments:
	----------
		mode:  'head-to-tail' to amide-bond the N-terminus to the
		       C-terminus (default), or 'sidechain' to bond the two
		       named atoms res1/atom1 and res2/atom2 (e.g. a disulfide)
		res1:  First residue index (sidechain mode)
		atom1: Atom name in res1 (sidechain mode)
		res2:  Second residue index (sidechain mode)
		atom2: Atom name in res2 (sidechain mode)
		precoil: head-to-tail only - coil the backbone and run cyclic
		       coordinate descent so the closing bond forms at ~1.33 A
		       instead of a stretched gap (default True)
	Returns:
	--------
		Modifies the pose in place: head-to-tail drops the extra
		N-terminal hydrogens and the C-terminal OXT, adds the closing
		bond to the graph, re-assigns Gasteiger charges, and records the
		closure in pose.data['Cyclic']. With precoil the closing bond is
		already at ~1.33 A (CCD); refine with tools.Minimise (the
		'Default' force field is recommended - the SMIRNOFF/'OpenFF'
		improper surface is poorly behaved for macrocyclic backbones).
		RotateDihedral/AdjustDistance are undefined on a closed ring and
		must not be used after cyclization. Returns no value.
	'''
	atoms = pose.data['Atoms']
	src = (pose.data['Amino Acids'] or pose.data['Nucleotides'])
	def atomof(res, nm):
		for a in src[res][2] + src[res][3]:
			if atoms[a][0] == nm: return a
		return None
	def reindex(drop):
		keep = [i for i in sorted(atoms) if i not in drop]
		nx = {old: k for k, old in enumerate(keep)}
		coords = np.asarray(pose.data['Coordinates'], dtype=float)
		pose.data['Coordinates'] = coords[keep]
		pose.data['Atoms'] = {nx[i]: atoms[i] for i in keep}
		bonds = pose.data['Bonds']
		bo = pose.data['BondOrders']
		nb = {}
		nbo = {}
		for i in keep:
			lst = []
			ol = []
			for j, o in zip(bonds.get(i, []), bo.get(i, [])):
				if j in nx:
					lst.append(nx[j])
					ol.append(o)
			nb[nx[i]] = lst
			nbo[nx[i]] = ol
		pose.data['Bonds'] = nb
		pose.data['BondOrders'] = nbo
		for ri in src:
			src[ri][2] = [nx[i] for i in src[ri][2] if i in nx]
			src[ri][3] = [nx[i] for i in src[ri][3] if i in nx]
		return nx
	def ccdclose():
		'''
		Close the ring geometry before the bond is formed: coil the
		backbone, then cyclic-coordinate-descent the C-terminal C onto
		its ideal amide position next to the N-terminal N
		Arguments:
		----------
			No arguments taken
		Returns:
		--------
			Rotates the still-linear backbone in place; no return value
		'''
		rr = sorted(src)
		for ri in rr:
			for ang, val in (('PHI', 0.0), ('PSI', 180.0)):
				try:
					if not np.isnan(pose.GetDihedral(ri, ang)):
						pose.RotateDihedral(ri, val, ang)
				except Exception:
					pass
		nC = atomof(rr[-1], 'C')
		n0 = atomof(rr[0], 'N')
		hd = atomof(rr[0], '2H') or atomof(rr[0], '3H')
		if nC is None or n0 is None or hd is None: return
		co = np.asarray(pose.data['Coordinates'], dtype=float)
		d = co[hd] - co[n0]
		nd = np.linalg.norm(d)
		if nd < 1e-9: return
		tgt = co[n0] + 1.33 * d / nd
		dih = []
		for ri in rr:
			for ang in ('PHI', 'PSI'):
				try:
					if not np.isnan(pose.GetDihedral(ri, ang)):
						dih.append((ri, ang))
				except Exception:
					pass
		for _ in range(300):
			co = np.asarray(pose.data['Coordinates'])
			if np.linalg.norm(co[nC] - tgt) < 0.02: break
			for ri, ang in reversed(dih):
				co = np.asarray(pose.data['Coordinates'])
				M = co[nC]
				if ang == 'PHI':
					O = co[atomof(ri, 'N')]
					u = co[atomof(ri, 'CA')] - O
				else:
					O = co[atomof(ri, 'CA')]
					u = co[atomof(ri, 'C')] - O
				nu = np.linalg.norm(u)
				if nu < 1e-9: continue
				u = u / nu
				a = (M - O) - np.dot(M - O, u) * u
				b = (tgt - O) - np.dot(tgt - O, u) * u
				na = np.linalg.norm(a)
				nb = np.linalg.norm(b)
				if na < 1e-6 or nb < 1e-6: continue
				a = a / na
				b = b / nb
				th = math.atan2(float(np.dot(np.cross(a, b), u)),
					float(np.dot(a, b)))
				pose.RotateDihedral(ri,
					pose.GetDihedral(ri, ang) + math.degrees(th), ang)
	if mode == 'head-to-tail':
		if precoil:
			ccdclose()
		ris = sorted(src)
		a_n = atomof(ris[0], 'N')
		a_c = atomof(ris[-1], 'C')
		drop = set()
		for nm in ('2H', '3H', 'H2', 'H3'):
			a = atomof(ris[0], nm)
			if a is not None: drop.add(a)
		for nm in ('OXT', 'OT1', 'OT2', "O''"):
			a = atomof(ris[-1], nm)
			if a is not None: drop.add(a)
		nx = reindex(drop)
		i1, i2, bov = nx[a_c], nx[a_n], 1.5
		rec = [int(ris[-1]), int(ris[0])]
	else:
		i1 = atomof(res1, atom1)
		i2 = atomof(res2, atom2)
		if i1 is None or i2 is None:
			raise Exception('Cyclize: sidechain atoms not found')
		bov = 1.0
		rec = [int(res1), int(res2)]
	pose.data['Bonds'].setdefault(i1, []).append(i2)
	pose.data['BondOrders'].setdefault(i1, []).append(bov)
	pose.data['Bonds'].setdefault(i2, []).append(i1)
	pose.data['BondOrders'].setdefault(i2, []).append(bov)
	pose.data.setdefault('Cyclic', []).append(rec)
	pose.CalcCharge()
