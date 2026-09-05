#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import re
import io
import os
import sys
import gzip
import json
import math
import time
import shutil
import base64
import pickle
import zipfile
import numpy as np
import urllib.request
import xml.etree.ElementTree as ET
from .pose import *
from .energy import ForceField, Score
from collections import defaultdict, deque

def _blosum(a, b):
	'''
	BLOSUM62 pairwise amino acid substitution score (Henikoff &
	Henikoff 1992, PNAS 89:10915-10919; matrix as distributed by NCBI,
	public domain)
	Arguments:
	----------
		a: First amino acid one-letter code, uppercase
		b: Second amino acid one-letter code, uppercase
	Returns:
	--------
		int: BLOSUM62 score for the pair, falling back to 4 for a match
		and -1 for a mismatch when either code is outside the alphabet
	'''
	BLOSUM_AA = 'ARNDCQEGHILKMFPSTWYV'
	BLOSUM_IDX = {c: i for i, c in enumerate(BLOSUM_AA)}
	BLOSUM62 = [
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
	ia, ib = BLOSUM_IDX.get(a, -1), BLOSUM_IDX.get(b, -1)
	if ia < 0 or ib < 0: return(4 if a == b else -1)
	return(BLOSUM62[ia][ib])

def _rotliblookup(rotlib, tri, phi, psi):
	'''
	Slice the Rotamer Library table for one residue and backbone bin
	Arguments:
	----------
		rotlib: The database['Rotamer Library'] dictionary
		tri:    Residue three-letter code, uppercase and L-form
		phi:    Backbone phi angle in degrees
		psi:    Backbone psi angle in degrees
	Returns:
	--------
		int: Number of chi angles the residue has, 0 when it has no entry
		list: Rotamer rows held in that backbone bin, each laid out as
		count, probability, chi 1..n, sigma 1..n; empty when the residue
		has no entry or the bin holds no rotamers
	'''
	entry = (rotlib.get('residues', {}) if rotlib else {}).get(tri)
	if entry is None: return 0, []
	pn, sn = int(rotlib.get('phi_n', 36)), int(rotlib.get('psi_n', 36))
	i = int(math.floor((phi - float(rotlib.get('phi_start', -180.0)))
		/ float(rotlib.get('phi_step', 10.0)))) % pn
	j = int(math.floor((psi - float(rotlib.get('psi_start', -180.0)))
		/ float(rotlib.get('psi_step', 10.0)))) % sn
	rot = entry['rotamers']
	off = rot['bin_offsets']
	b = i * sn + j
	return int(entry['n_chi']), rot['table'][off[b]:off[b + 1]]

def Parameterise(cif_file, rotamer_json_file, unicode, tricode,
		parent='', backup=True):
	'''
	Add a non-canonical amino acid (NCAA) to Pose's database.json
	Arguments:
	----------
		cif_file:          Path to the RCSB Chemical Component
		                   Dictionary CIF of the residue
		rotamer_json_file: Path to the Dunbrack BBDEP2010-format
		                   rotamer library JSON of the residue
		unicode:           Single-letter key for db['Amino Acids']
		tricode:           Three-letter residue code, e.g. PTR
		parent:            Three-letter code of the canonical amino
		                   acid of similar chemistry, '' if none
		backup:            True copies database.json to
		                   database.json.bak.<YYYYMMDD-HHMMSS> first
	Returns:
	--------
		None: database.json gains the residue under 'Amino Acids' and
		'Rotamer Library', existing keys are overwritten with a warning
		on stderr, and the DBLoad cache is cleared so objects built
		afterwards see the residue without a restart
	'''
	def ALAframe():
		'''
		The alanine reference frame with its nitrogen at the origin
		Arguments:
		----------
			No arguments taken
		Returns:
		--------
			np.array: 13x3 coordinates ordered N, H1, H2, H3, CA, HA,
			CB, 1HB, 2HB, 3HB, C, O, OXT
		'''
		return np.array([
			[ 0.000,  0.000,  0.000], [-0.334, -0.943,  0.000],
			[-0.334,  0.471,  0.816], [-0.334,  0.471, -0.816],
			[ 1.458,  0.000,  0.000], [ 1.822, -0.535,  0.877],
			[ 1.988, -0.773, -1.199], [ 3.078, -0.764, -1.185],
			[ 1.633, -1.802, -1.154], [ 1.633, -0.307, -2.117],
			[ 2.009,  1.420,  0.000], [ 2.058,  2.045,  1.023],
			[ 2.394,  1.914, -1.023]])
	def validrot(rot, tri):
		'''
		Check a rotamer JSON against the Dunbrack BBDEP2010 schema
		Arguments:
		----------
			rot: Parsed rotamer JSON content
			tri: Three-letter code the caller is inserting, must match
			     the code the JSON declares
		Returns:
		--------
			None: raises ValueError naming the first schema violation
		'''
		miss = [k for k in ('tricode', 'n_chi', 'rotamers')
			if k not in rot]
		if miss: raise ValueError(
			f'rotamer JSON missing required keys: {miss}')
		jt = rot['tricode']
		if not isinstance(jt, str) or len(jt) != 3: raise ValueError(
			f'rotamer JSON tricode must be a 3-letter str, got {jt!r}')
		if jt.upper() != tri.upper(): raise ValueError(
			f'rotamer JSON tricode {jt!r} does not match argument '
			f'{tri!r}')
		nchi = int(rot['n_chi'])
		if nchi < 1 or nchi > 8: raise ValueError(
			f'n_chi out of range (1-8): {nchi}')
		if 'chi_axes' not in rot.get('method', {}): raise ValueError(
			'rotamer JSON missing method.chi_axes (required as the '
			'source of truth for Amino Acids "Chi Angle Atoms")')
		axes = rot['method']['chi_axes']
		if len(axes) != nchi: raise ValueError(
			f'method.chi_axes has {len(axes)} axes but n_chi={nchi}')
		bad = next((k for k, a in enumerate(axes) if len(a) != 4), None)
		if bad is not None: raise ValueError(
			f'chi_axes[{bad}] has {len(axes[bad])} atoms (need 4): '
			f'{axes[bad]}')
		tab = rot['rotamers']
		miss = [k for k in ('columns', 'table', 'bin_offsets',
			'top_chi') if k not in tab]
		if miss: raise ValueError(f'rotamers missing keys: {miss}')
		cols = (['count', 'prob'] + [f'chi{k+1}' for k in range(nchi)]
			+ [f'sig{k+1}' for k in range(nchi)])
		if tab['columns'] != cols: raise ValueError(
			f'rotamer columns mismatch.\n'
			f'  got:      {tab["columns"]}\n'
			f'  expected: {cols}')
		if len(tab['bin_offsets']) != 1297: raise ValueError(
			f'bin_offsets length {len(tab["bin_offsets"])} != 1297')
		if len(tab['top_chi']) != 36: raise ValueError(
			f'top_chi outer length {len(tab["top_chi"])} != 36')
		if any(len(r) != 36 for r in tab['top_chi']): raise ValueError(
			'top_chi inner length != 36')
	def sigmas(rot, floor=0.5):
		'''
		Raise every rotamer sigma below floor degrees up to floor
		Arguments:
		----------
			rot:   Parsed rotamer JSON content, modified in place
			floor: Minimum sigma in degrees, 0.5 by default
		Returns:
		--------
			int: How many sigma values were raised
		'''
		first = 2 + int(rot['n_chi'])
		last = first + int(rot['n_chi'])
		raised = 0
		for row in rot['rotamers']['table']:
			low = [i for i in range(first, last) if float(row[i]) < floor]
			raised += len(low)
			for i in low: row[i] = floor
		return raised
	def cifparse(cif_file, tri):
		'''
		Read the atom and bond records of one residue out of a CIF
		Arguments:
		----------
			cif_file: Path to the RCSB Chemical Component Dictionary CIF
			tri:      Three-letter code whose records are wanted
		Returns:
		--------
			np.array: One XYZ coordinate row per atom, in CIF order
			list:     One {id, elem, bb} dict per atom, in CIF order
			list:     One (atom1, atom2, value_order, aromatic) tuple
			          per bond
		'''
		coords, atoms, bonds = [], [], []
		with open(cif_file) as fh: lines = fh.readlines()
		for line in lines:
			t = line.strip().split()
			if not t or t[0] != tri: continue
			if len(t) == 7 and t[3] in ('SING','DOUB','TRIP','AROM'):
				bonds.append((t[1], t[2], t[3], t[4]))
			if len(t) < 18: continue
			try: c = [float(t[i]) for i in (15, 16, 17)]
			except (ValueError, IndexError):
				try: c = [float(t[i]) for i in (12, 13, 14)]
				except (ValueError, IndexError): continue
			coords.append(c)
			atoms.append({'id': t[1], 'elem': t[3].capitalize(),
				'bb': (t[9] == 'Y')})
		return np.array(coords), atoms, bonds
	def bondgraph(bonds):
		'''
		Build an undirected bond graph keyed by CIF atom id
		Arguments:
		----------
			bonds: List of (atom1, atom2, value_order, aromatic) tuples
		Returns:
		--------
			defaultdict: Atom id mapped to the set of ids bonded to it
		'''
		adj = defaultdict(set)
		for a1, a2, vo, ar in bonds:
			adj[a1].add(a2)
			adj[a2].add(a1)
		return adj
	def bfs(adj, elem, ciford, bbset):
		'''
		Walk the sidechain outwards from CB in breadth-first order
		Arguments:
		----------
			adj:    Undirected bond graph keyed by CIF atom id
			elem:   CIF atom id mapped to its element symbol
			ciford: CIF atom id mapped to its row number in the CIF
			bbset:  CIF atom ids belonging to the backbone
		Returns:
		--------
			list: Sidechain atom ids, each heavy atom followed by the
			hydrogens bonded to it, as Pose stores them
		'''
		ordered, seen, q = [], set(bbset) | {'CB'}, deque(['CB'])
		while q:
			atom = q.popleft()
			ordered.append(atom)
			for n in sorted(adj[atom], key=lambda m: ciford.get(m, 9999)):
				if n in seen: continue
				seen.add(n)
				h = elem.get(n, '').upper() in ('H', 'D')
				(ordered if h else q).append(n)
		return ordered
	def renamehydrogens(ordered, elem):
		'''
		Rename trailing-digit hydrogens to Pose's leading-digit form
		Arguments:
		----------
			ordered: Sidechain atom ids in Pose order
			elem:    CIF atom id mapped to its element symbol
		Returns:
		--------
			dict: CIF atom id mapped to its Pose name, so CIF HB2 on the
			first beta carbon becomes 1HB and heavy atoms are unchanged
		'''
		namemap, counter = {}, defaultdict(int)
		for name in ordered:
			m = re.match(r'^([A-Z]+)(\d+)$', name)
			h = elem.get(name, '').upper() in ('H', 'D')
			if not m or not h:
				namemap[name] = name
				continue
			counter[m.group(1)] += 1
			namemap[name] = f'{counter[m.group(1)]}{m.group(1)}'
		return namemap
	def fused(ordered, adj):
		'''
		Find the sidechain atom that bonds back to the backbone nitrogen
		Arguments:
		----------
			ordered: Sidechain atom ids in Pose order
			adj:     Undirected bond graph keyed by CIF atom id
		Returns:
		--------
			str: The bonded atom id, as proline's CD is, or None when
			the sidechain is not fused to the backbone
		'''
		return next((sc for sc in ordered if 'N' in adj[sc]), None)
	def renumberbondgraph(bonds, ordered, fusedatom, tri):
		'''
		Re-index the sidechain bonds onto Pose's atom numbering
		Arguments:
		----------
			bonds:     List of (atom1, atom2, value_order, aromatic)
			           tuples
			ordered:   Sidechain atom ids in Pose order
			fusedatom: Atom id bonded to the backbone nitrogen, or None
			tri:       Three-letter code, named in the warning printed
			           for an unrecognised bond order
		Returns:
		--------
			defaultdict: Pose atom index mapped to its bonded indices
			defaultdict: Pose atom index mapped to its bond orders
			dict:        CIF atom id mapped to its Pose atom index
			The backbone nitrogen of a fused sidechain enters both
			graphs as the sentinel index -5
		'''
		ORDERS = {'SING': 1, 'DOUB': 2, 'TRIP': 3, 'AROM': 1.5}
		scb, sco, lookup = defaultdict(list), defaultdict(list), {}
		for a1, a2, vo, ar in bonds:
			bo = 1.5 if ar == 'Y' else ORDERS.get(vo.upper())
			if bo is None:
				print(f'Warning: unknown bond order {vo!r} for '
					f'{a1}-{a2} in {tri}, defaulting to 1')
				bo = 1
			lookup[(a1, a2)] = lookup[(a2, a1)] = bo
		newidx = {n: i for i, n in enumerate(ordered)}
		scset = set(ordered)
		inside = [(a1, a2) for a1, a2, vo, ar in bonds
			if a1 in scset and a2 in scset]
		for a1, a2 in inside:
			i1, i2, bo = newidx[a1], newidx[a2], lookup[(a1, a2)]
			scb[i1].append(i2)
			scb[i2].append(i1)
			sco[i1].append(bo)
			sco[i2].append(bo)
		if fusedatom is None: return scb, sco, newidx
		fi = newidx[fusedatom]
		scb[fi].append(-5)
		scb[-5].append(fi)
		sco[fi].append(1)
		sco[-5].append(1)
		return scb, sco, newidx
	def aromaticity(scb, sco, elemidx):
		'''
		Spread resonance over carboxylate, amide and guanidinium groups
		Arguments:
		----------
			scb:     Pose atom index mapped to its bonded indices
			sco:     Pose atom index mapped to its bond orders, modified
			         in place
			elemidx: Pose atom index mapped to its element symbol
		Returns:
		--------
			None: every bond from a carbon carrying two or more oxygen
			or nitrogen neighbours and at least one double bond becomes
			order 1.5, in two passes so a group reached through another
			group is caught
		'''
		for p in range(2):
			for i in [k for k in scb if k >= 0 and elemidx.get(k) == 'C']:
				xs = [(k, nb, bo) for k, (nb, bo)
					in enumerate(zip(scb[i], sco[i]))
					if nb >= 0 and elemidx.get(nb) in ('O', 'N')]
				if len(xs) < 2: continue
				if not any(bo >= 2 for k, nb, bo in xs): continue
				for k, nb, bo in [x for x in xs if x[2] != 1.5]:
					sco[i][k] = 1.5
					sco[nb][scb[nb].index(i)] = 1.5
	def hybridisation(el, orders):
		'''
		Classify an atom's hybridisation from its element and bonds
		Arguments:
		----------
			el:     Element symbol of the atom, case-insensitive
			orders: Bond orders incident on the atom, where resonance
			        and aromatic bonds carry order 1.5
		Returns:
		--------
			str: One of s, sp, sp2, sp3
		'''
		if el and el.upper() == 'H': return 's'
		if any(bo == 3 for bo in orders): return 'sp'
		if any(bo >= 1.5 for bo in orders): return 'sp2'
		return 'sp3'
	def DAA(ltri, db, uni):
		'''
		Allocate an unused three-letter code for the D-form enantiomer
		Arguments:
		----------
			ltri: Three-letter code of the L-form, e.g. PTR
			db:   The loaded database.json
			uni:  Single-letter key being written, whose own codes stay
			      available so re-running keeps the same D-form
		Returns:
		--------
			str: D plus two letters, tried as the last two letters of
			ltri, then its first and last, then its first two, then
			every two-letter combination, skipping codes already used
		'''
		taken = set()
		for ek, e in db.get('Amino Acids', {}).items():
			t = e.get('Tricode')
			if ek == uni or t is None: continue
			taken.update([t] if isinstance(t, str) else t)
		cands = ['D' + ltri[1:], 'D' + ltri[0] + ltri[2], 'D' + ltri[:2]]
		cands += ['D' + chr(65 + n // 26) + chr(65 + n % 26)
			for n in range(676)]
		free = next((c for c in cands if c not in taken), None)
		if free is None: raise Exception(f'No free D-tricode for {ltri}')
		return free
	def validateDB(db):
		'''
		Check that Bonds and BondOrders agree across the whole database
		Arguments:
		----------
			db: The loaded database.json
		Returns:
		--------
			None: raises ValueError naming the first entry whose
			BondOrders is missing, lacks a key its Bonds carries, or
			lists a different number of orders than that key has bonds
		'''
		for section in ('Amino Acids', 'Nucleotides'):
			for ekey, e in db.get(section, {}).items():
				if 'Bonds' not in e: continue
				if 'BondOrders' not in e: raise ValueError(
					f'{section}[{ekey!r}]: has Bonds but no BondOrders')
				bo, bn = e['BondOrders'], e['Bonds']
				bad = next((k for k, v in bn.items()
					if k not in bo or len(bo[k]) != len(v)), None)
				if bad is None: continue
				if bad not in bo: raise ValueError(
					f'{section}[{ekey!r}]: '
					f'BondOrders missing key {bad!r}')
				raise ValueError(f'{section}[{ekey!r}][{bad!r}]: '
					f'Bonds has {len(bn[bad])} entries but '
					f'BondOrders has {len(bo[bad])}')
	unicode, tricode, parent = (unicode.upper(), tricode.upper(),
		parent.upper())
	with open(rotamer_json_file) as fh: rot = json.load(fh)
	validrot(rot, tricode)
	raised = sigmas(rot)
	if raised: print(f'Note: clamped {raised} rotamer sigma values to '
		f'>=0.5 deg floor.')
	chis = [list(a) for a in rot['method']['chi_axes']]
	COORD, ATOMS, BONDS = cifparse(cif_file, tricode)
	ids = [a['id'] for a in ATOMS]
	if 'CB' not in ids: raise ValueError(f'No CB atom found in '
		f'{cif_file}. Only standard amino acids (not GLY) are '
		f'supported.')
	idset = set(ids)
	bad = next(((k, a) for k, ax in enumerate(chis) for a in ax
		if a not in idset), None)
	if bad: raise ValueError(f'chi axis {bad[0] + 1} references atom '
		f'{bad[1]!r} which does not exist in {cif_file}. CIF atoms: '
		f'{sorted(idset)}')
	bbset = {a['id'] for a in ATOMS if a['bb']} or {'N', 'CA', 'C', 'O',
		'OXT', 'H', 'H1', 'H2', 'H3', 'HA', 'HA2', 'HA3', 'HXT'}
	elem = {a['id']: a['elem'] for a in ATOMS}
	ciford = {a['id']: i for i, a in enumerate(ATOMS)}
	try: idx = [ids.index(n) for n in ('N', 'CA', 'CB', 'C')]
	except ValueError as e:
		raise ValueError(f'Missing backbone atom in {cif_file}: {e}')
	A = np.c_[ALAframe(), np.ones(13)]
	B = np.c_[COORD, np.ones(len(COORD))]
	AL = np.array([A[0] - A[4], A[6] - A[4], A[-3] - A[4], A[4]])
	BL = np.array([B[i] - B[idx[1]] for i in (idx[0], idx[2], idx[3])]
		+ [B[idx[1]]])
	COORD = (B @ (np.linalg.inv(BL) @ AL))[:, :3]
	adj = bondgraph(BONDS)
	ordered = bfs(adj, elem, ciford, bbset)
	namemap = renamehydrogens(ordered, elem)
	fusedatom = fused(ordered, adj)
	scb, sco, newidx = renumberbondgraph(BONDS, ordered, fusedatom,
		tricode)
	aromaticity(scb, sco, {newidx[n]: elem[n] for n in ordered})
	keys = sorted(k for k in scb if k >= 0)
	keys += [] if fusedatom is None else [-5]
	bonds = {k: sorted(scb[k]) for k in keys}
	orders = {k: [dict(zip(scb[k], sco[k]))[nb] for nb in bonds[k]]
		for k in keys}
	cifidx = {c: i for i, c in enumerate(ids)}
	entry = {
		'Vectors':         [COORD[cifidx[n]].tolist() for n in ordered],
		'Tricode':         tricode,
		'Fused':           fusedatom is not None,
		'Sidechain Atoms': [[namemap[n], elem[n], 0, 1.0, 0,
			hybridisation(elem[n], sco[newidx[n]])] for n in ordered],
		'Chi Angle Atoms': chis,
		'Bonds':           {str(k): v for k, v in bonds.items()},
		'BondOrders':      {str(k): v for k, v in orders.items()}}
	if parent: entry['Parent'] = parent
	dbpath = os.path.join(
		os.path.dirname(os.path.abspath(__file__)), 'database.json')
	with open(dbpath) as fh: db = json.load(fh)
	if unicode in db.get('Amino Acids', {}):
		old = db['Amino Acids'][unicode].get('Tricode', '?')
		print(f'Warning: db["Amino Acids"]["{unicode}"] already exists '
			f'(was Tricode={old}); overwriting with '
			f'Tricode={tricode}.', file=sys.stderr)
	resid = db.setdefault('Rotamer Library', {}).setdefault(
		'residues', {})
	if tricode in resid:
		print(f'Warning: db["Rotamer Library"]["residues"]'
			f'["{tricode}"] already exists; overwriting.',
			file=sys.stderr)
	entry['Tricode'] = [tricode, DAA(tricode, db, unicode)]
	db.setdefault('Amino Acids', {})[unicode] = entry
	resid[tricode] = {'n_chi': int(rot['n_chi']),
		'rotamers': rot['rotamers'],
		'densities': rot.get('densities')}
	if rot.get('FaDunNrchiDensities'):
		db.setdefault('Score Parameters', {}).setdefault(
			'REF15', {}).setdefault('FaDunNrchiDensities', {})[
			tricode] = rot['FaDunNrchiDensities']
	validateDB(db)
	if backup:
		bak = f'{dbpath}.bak.{time.strftime("%Y%m%d-%H%M%S")}'
		shutil.copy2(dbpath, bak)
		print(f'Backup: {bak}')
	tmp = dbpath + '.tmp'
	try:
		with open(tmp, 'w') as fh:
			json.dump(db, fh, separators=(',', ':'))
		os.replace(tmp, dbpath)
	except BaseException:
		if os.path.exists(tmp): os.remove(tmp)
		raise
	DBLoad.cache_clear()
	print(f'Added {tricode} as "{unicode}" to database.json '
		f'(Amino Acids + Rotamer Library)')

def Isoelectric(sequence):
	'''
	Isoelectric point (pI) of a protein via Lehninger pKa values
	Arguments:
	----------
		sequence: Protein FASTA sequence using one-letter codes,
		including the six Pose non-canonical letters B J O U X Z
	Returns:
	--------
		float: pH at which the protein carries zero net charge,
		found by bisection on [0, 14] and rounded to 2 decimals
	'''
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	pos = [(1, 9.60), (seq.count('K'), 10.53), (seq.count('R'), 12.48),
		(seq.count('H'), 6.00), (seq.count('B'), 10.76)]
	neg = [(1, 2.34), (seq.count('D'), 3.65), (seq.count('E'), 4.25),
		(seq.count('C'), 8.33), (seq.count('Y'), 10.07),
		(seq.count('U'), 5.20), (seq.count('O'), 2.00),
		(seq.count('O'), 6.30), (seq.count('Z'), 2.00),
		(seq.count('Z'), 5.96)]
	lo, hi = 0.0, 14.0
	for _ in range(100):
		mid = (lo + hi) / 2.0
		c = sum(n / (1.0 + 10 ** (mid - p)) for n, p in pos) \
			- sum(n / (1.0 + 10 ** (p - mid)) for n, p in neg)
		if abs(c) < 1e-4: break
		if c > 0: lo = mid
		else: hi = mid
	return(round(mid, 2))

def Hydrophobicity(sequence, window=9, scale='eisenberg'):
	'''
	Sliding-window hydrophobicity profile (ProtScale-style)
	Arguments:
	----------
		sequence: Protein FASTA sequence using one-letter codes;
			the non-canonical letters B J O U X Z take the value
			of their closest canonical analogue
		window:   Window size, default 9
		scale:    'eisenberg' (Eisenberg et al. 1984, J Mol Biol
			179:125), 'kyte-doolittle' (Kyte & Doolittle 1982, J Mol
			Biol 157:105), 'hopp-woods' (Hopp & Woods 1981, PNAS
			78:3824), or 'engelman' (Engelman et al. 1986, Annu Rev
			Biophys Biophys Chem 15:321)
	Returns:
	--------
		list: 0-indexed centre position of each window
		list: Mean hydrophobicity of each window, rounded to 3 decimals
	'''
	scales = {
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
	ncaa = {'J':'M', 'X':'W', 'B':'K', 'U':'C', 'O':'D', 'Z':'D'}
	seq, L = sequence.upper(), len(sequence)
	if window < 1: raise Exception('window must be >= 1')
	if window > L: raise Exception(
		f'window ({window}) larger than sequence ({L})')
	tbl = scales.get(scale.lower())
	if tbl is None: raise Exception(
		f'Unknown scale {scale!r}; choose from {list(scales)}')
	tbl = {**tbl, **{k: tbl[v] for k, v in ncaa.items()}}
	half, n = (window - 1) // 2, L - window + 1
	return([i + half for i in range(n)],
		[round(sum(tbl.get(seq[i+k], 0.0)
			for k in range(window)) / window, 3) for i in range(n)])

def Aliphatic(sequence):
	'''
	Aliphatic index from mole percentages (Ikai 1980), computed as
	AI = X(A) + 2.9*X(V) + 3.9*(X(I) + X(L))
	Arguments:
	----------
		sequence: Protein FASTA sequence using one-letter codes
	Returns:
	--------
		float: Aliphatic index, rounded to 2 decimals
	'''
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	xA, xV, xI, xL = (100.0 * seq.count(a) / len(seq) for a in 'AVIL')
	return round(xA + 2.9 * xV + 3.9 * (xI + xL), 2)

def ExtinctCoeff(sequence, reduced=True):
	'''
	Molar extinction coefficient at 280 nm (Pace 1995), computed as
	eps = nW*5500 + nY*1490 + (nC/2)*125
	Arguments:
	----------
		sequence: Protein FASTA sequence using one-letter codes;
			FT6 absorbs as Trp and PTR at about 200, while MSE,
			ORN, TPO and reduced SEC do not absorb at all
		reduced:  True (default) treats Cys as reduced and
			non-absorbing; False pairs them into cystines
	Returns:
	--------
		int: Molar extinction coefficient in M^-1 cm^-1
	'''
	if not sequence: raise Exception('Empty sequence')
	seq = sequence.upper()
	return ((seq.count('W') + seq.count('X')) * 5500
		+ seq.count('Y') * 1490 + seq.count('Z') * 200
		+ (0 if reduced else (seq.count('C') // 2) * 125))

def Instability(sequence):
	'''
	Instability index (Guruprasad et al. 1990, Protein Eng 4:155-161), computed
	as II = (10/L) * sum DIWV(seq[i], seq[i+1]), where II < 40
	suggests a stable protein
	Arguments:
	----------
		sequence: Protein FASTA sequence using one-letter codes;
			the non-canonical letters J X B U O Z are read as
			their parent residues M W K C T Y
	Returns:
	--------
		float: Instability index, rounded to 2 decimals; 0.0 when
		the sequence is a single residue
	'''
	DIWV = {
		'AA':1,'AR':1,'AN':1,'AD':-7.49,'AC':44.94,'AQ':1,'AE':1,'AG':1,
		'AH':-7.49,'AI':1,'AL':1,'AK':1,'AM':1,'AF':1,'AP':20.26,'AS':1,
		'AT':1,'AW':1,'AY':1,'AV':1,'RA':1,'RR':58.28,'RN':13.34,'RD':1,
		'RC':1,'RQ':20.26,'RE':1,'RG':-7.49,'RH':20.26,'RI':1,'RL':1,'RK':1,
		'RM':1,'RF':1,'RP':20.26,'RS':44.94,'RT':1,'RW':58.28,'RY':-6.54,
		'RV':1,'NA':1,'NR':1,'NN':1,'ND':1,'NC':-1.88,'NQ':-6.54,'NE':1,
		'NG':-14.03,'NH':1,'NI':44.94,'NL':1,'NK':24.68,'NM':1,'NF':-14.03,
		'NP':-1.88,'NS':1,'NT':-7.49,'NW':-9.37,'NY':1,'NV':-1.88,'DA':1,
		'DR':-6.54,'DN':1,'DD':1,'DC':1,'DQ':1,'DE':1,'DG':1,'DH':1,'DI':1,
		'DL':1,'DK':-7.49,'DM':1,'DF':-6.54,'DP':1,'DS':20.26,'DT':-14.03,
		'DW':1,'DY':1,'DV':1,'CA':1,'CR':1,'CN':1,'CD':20.26,'CC':1,
		'CQ':-6.54,'CE':1,'CG':1,'CH':33.6,'CI':1,'CL':20.26,'CK':1,
		'CM':33.6,'CF':1,'CP':20.26,'CS':1,'CT':33.6,'CW':24.68,'CY':1,
		'CV':-6.54,'QA':1,'QR':1,'QN':1,'QD':20.26,'QC':-6.54,'QQ':20.26,
		'QE':20.26,'QG':1,'QH':1,'QI':1,'QL':1,'QK':1,'QM':1,'QF':-6.54,
		'QP':20.26,'QS':44.94,'QT':1,'QW':1,'QY':-6.54,'QV':-6.54,'EA':1,
		'ER':1,'EN':1,'ED':20.26,'EC':44.94,'EQ':20.26,'EE':33.6,'EG':1,
		'EH':-6.54,'EI':20.26,'EL':1,'EK':1,'EM':1,'EF':1,'EP':20.26,
		'ES':20.26,'ET':1,'EW':-14.03,'EY':1,'EV':1,'GA':-7.49,'GR':-7.49,
		'GN':-7.49,'GD':1,'GC':1,'GQ':1,'GE':-6.54,'GG':13.34,'GH':1,
		'GI':-7.49,'GL':1,'GK':-7.49,'GM':1,'GF':1,'GP':1,'GS':1,'GT':-7.49,
		'GW':13.34,'GY':-7.49,'GV':1,'HA':1,'HR':1,'HN':24.68,'HD':1,'HC':1,
		'HQ':1,'HE':1,'HG':-9.37,'HH':1,'HI':44.94,'HL':1,'HK':24.68,'HM':1,
		'HF':-9.37,'HP':-1.88,'HS':1,'HT':-6.54,'HW':-1.88,'HY':44.94,
		'HV':1,'IA':1,'IR':1,'IN':1,'ID':1,'IC':1,'IQ':1,'IE':44.94,'IG':1,
		'IH':13.34,'II':1,'IL':20.26,'IK':-7.49,'IM':1,'IF':1,'IP':-1.88,
		'IS':1,'IT':1,'IW':1,'IY':1,'IV':-7.49,'LA':1,'LR':20.26,'LN':1,
		'LD':1,'LC':1,'LQ':33.6,'LE':1,'LG':1,'LH':1,'LI':1,'LL':1,
		'LK':-7.49,'LM':1,'LF':1,'LP':20.26,'LS':1,'LT':1,'LW':24.68,'LY':1,
		'LV':1,'KA':1,'KR':33.6,'KN':1,'KD':1,'KC':1,'KQ':24.68,'KE':1,
		'KG':-7.49,'KH':1,'KI':-7.49,'KL':-7.49,'KK':1,'KM':33.6,'KF':1,
		'KP':-6.54,'KS':1,'KT':1,'KW':1,'KY':1,'KV':-7.49,'MA':13.34,
		'MR':-6.54,'MN':1,'MD':1,'MC':1,'MQ':-6.54,'ME':1,'MG':1,'MH':58.28,
		'MI':1,'ML':1,'MK':1,'MM':-1.88,'MF':1,'MP':44.94,'MS':44.94,
		'MT':-1.88,'MW':1,'MY':24.68,'MV':1,'FA':1,'FR':1,'FN':1,'FD':13.34,
		'FC':1,'FQ':1,'FE':1,'FG':1,'FH':1,'FI':1,'FL':1,'FK':-14.03,'FM':1,
		'FF':1,'FP':20.26,'FS':1,'FT':1,'FW':1,'FY':33.6,'FV':1,'PA':20.26,
		'PR':-6.54,'PN':1,'PD':-6.54,'PC':-6.54,'PQ':20.26,'PE':18.38,
		'PG':1,'PH':1,'PI':1,'PL':1,'PK':1,'PM':-6.54,'PF':20.26,'PP':20.26,
		'PS':20.26,'PT':1,'PW':-1.88,'PY':1,'PV':20.26,'SA':1,'SR':20.26,
		'SN':1,'SD':1,'SC':33.6,'SQ':20.26,'SE':20.26,'SG':1,'SH':1,'SI':1,
		'SL':1,'SK':1,'SM':1,'SF':1,'SP':44.94,'SS':20.26,'ST':1,'SW':1,
		'SY':1,'SV':1,'TA':1,'TR':1,'TN':-14.03,'TD':1,'TC':1,'TQ':-6.54,
		'TE':20.26,'TG':-7.49,'TH':1,'TI':1,'TL':1,'TK':1,'TM':1,'TF':13.34,
		'TP':1,'TS':1,'TT':1,'TW':-14.03,'TY':1,'TV':1,'WA':-14.03,'WR':1,
		'WN':13.34,'WD':1,'WC':1,'WQ':1,'WE':1,'WG':-9.37,'WH':24.68,'WI':1,
		'WL':13.34,'WK':1,'WM':24.68,'WF':1,'WP':1,'WS':1,'WT':-14.03,
		'WW':1,'WY':1,'WV':-7.49,'YA':24.68,'YR':-15.91,'YN':1,'YD':24.68,
		'YC':1,'YQ':1,'YE':-6.54,'YG':-7.49,'YH':13.34,'YI':1,'YL':1,'YK':1,
		'YM':44.94,'YF':1,'YP':13.34,'YS':1,'YT':-7.49,'YW':-9.37,
		'YY':13.34,'YV':1,'VA':1,'VR':1,'VN':1,'VD':-14.03,'VC':1,'VQ':1,
		'VE':1,'VG':-7.49,'VH':1,'VI':1,'VL':1,'VK':-1.88,'VM':1,'VF':1,
		'VP':20.26,'VS':1,'VT':-7.49,'VW':1,'VY':-6.54,'VV':1}
	ncaa = str.maketrans('JXBUOZ', 'MWKCTY')
	if not sequence: raise Exception('Empty sequence')
	seq, L = sequence.upper().translate(ncaa), len(sequence)
	if L < 2: return 0.0
	total = sum(DIWV.get(seq[i:i+2], 0) for i in range(L - 1))
	return round(10.0 * total / L, 2)

def GRAVY(sequence):
	'''
	Grand average of hydropathy, the mean Kyte-Doolittle hydropathy
	value over every residue of the sequence (Kyte & Doolittle 1982)
	Arguments:
	----------
		sequence: Protein FASTA sequence using one-letter codes;
			the non-canonical letters J X B U O Z are read as
			their analogues M W K C D D, matching Hydrophobicity
	Returns:
	--------
		float: Mean Kyte-Doolittle hydropathy, rounded to 3 decimals
	'''
	kd = {
		'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
		'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
		'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
		'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2}
	ncaa = str.maketrans('JXBUOZ', 'MWKCDD')
	if not sequence: raise Exception('Empty sequence')
	return round(sum(kd.get(a, 0.0)
		for a in sequence.upper().translate(ncaa))
		/ len(sequence), 3)

def PROSITE(sequence, pattern):
	'''
	Search a protein sequence for a PROSITE-style pattern, using the
	subset grammar of literals, [ABC], {ABC}, x(n,m) and < > anchors
	Arguments:
	----------
		sequence: Protein sequence to search, matched case-insensitively;
			the non-canonical letters J X B U O Z are read as
			their parent residues M W K C T Y
		pattern:  PROSITE pattern built from literals, [ABC], {ABC},
			x, x(n), x(n,m), and the < and > terminal anchors
	Returns:
	--------
		list: Each hit is a (start, end, match) tuple using 1-based
		inclusive positions; overlapping hits are all reported
	'''
	if not pattern: raise Exception('Empty pattern')
	if not sequence: return []
	sub = {'<': '^', '>': '$', 'x': '.', 'X': '.'}
	ncaa = str.maketrans('JXBUOZ', 'MWKCTY')
	p = pattern.replace('-', '').replace(' ', '')
	out, i = [], 0
	while i < len(p):
		c = p[i]
		if c in '[{':
			j = p.find(']' if c == '[' else '}', i)
			if j == -1: raise Exception(f'Unclosed {c} in pattern')
			out.append(('[' if c == '[' else '[^') + p[i+1:j] + ']')
			i = j + 1
		elif c in sub: out.append(sub[c]); i += 1
		elif c.isalpha(): out.append(c.upper()); i += 1
		else: raise Exception(
			f'Unexpected character {c!r} at position {i} of pattern')
		if i < len(p) and p[i] == '(':
			j = p.find(')', i)
			if j == -1: raise Exception('Unclosed ( in pattern')
			out.append('{' + ','.join(
				s.strip() for s in p[i+1:j].split(',', 1)) + '}')
			i = j + 1
	rx = re.compile('(?=(' + ''.join(out) + '))', re.IGNORECASE)
	return [(m.start() + 1, m.start() + len(m.group(1)), m.group(1))
		for m in rx.finditer(sequence.translate(ncaa))]

def Split(pose, chain=None, start=None, end=None):
	'''
	Extract a slice of a pose, by chain or by residue range, into a
	new pose whose atoms and residues are renumbered densely from zero
	Arguments:
	----------
		pose:  Source protein, DNA, or RNA pose
		chain: Chain ID to extract, mutually exclusive with the
			start and end range
		start: First residue index to keep, inclusive and zero-based
		end:   Last residue index to keep, inclusive and zero-based
	Returns:
	--------
		Pose: New pose holding the selected residues, with atoms,
		bonds, and coordinates renumbered from zero
	'''
	if (chain is None) == (start is None and end is None):
		raise Exception("Split requires either chain= OR (start=, end=)")
	mol = pose.data.get('Type')
	if mol is None: raise Exception('Source pose is empty')
	is_pro = (mol == 'Protein')
	rk = 'Amino Acids' if is_pro else 'Nucleotides'
	src = pose.data[rk]
	if not src: raise Exception(f'Source pose has no {rk}')
	all_idx = sorted(src.keys())
	if chain is not None:
		keep_res = [i for i in all_idx if src[i][1] == chain]
		if not keep_res: raise Exception(f'Chain {chain!r} not in pose')
	else:
		if start is None or end is None:
			raise Exception('Split needs both start and end for range mode')
		if start > end: raise Exception(f'start ({start}) > end ({end})')
		keep_res = [i for i in all_idx if start <= i <= end]
		if not keep_res:
			raise Exception(f'Range [{start}, {end}] selects no residues')
	keep_atoms = sorted({ai for ri in keep_res
		for ai in src[ri][2] + src[ri][3]})
	a_remap = {old: new for new, old in enumerate(keep_atoms)}
	sa, sb, sc = (pose.data['Atoms'], pose.data['Bonds'],
		pose.data['Coordinates'])
	new = Pose()
	new.data = {
		'Type':        mol,  'Energy': 0, 'Rg': 0, 'Mass': 0,
		'Size':        {},   'FASTA':  {}, 'SS': {},
		'Nucleotides': None if is_pro else {},
		'Amino Acids': {} if is_pro else None,
		'Atoms':       {a_remap[o]: list(sa[o]) for o in keep_atoms},
		'Bonds':       {a_remap[o]: sorted(a_remap[b]
			for b in sb.get(o, []) if b in a_remap)
			for o in keep_atoms},
		'Coordinates': np.array([sc[o] for o in keep_atoms],
			dtype=float) if keep_atoms else np.zeros((0, 3))}
	tgt = new.data[rk]
	for n, o in enumerate(keep_res):
		row = list(src[o])
		row[2] = [a_remap[a] for a in row[2] if a in a_remap]
		row[3] = [a_remap[a] for a in row[3] if a in a_remap]
		tgt[n] = row
	new._update()
	return new

def Concatenate(pose1, pose2, fuse=False):
	'''
	Combine two poses of the same type, either by appending pose2 as
	extra chains or by rebuilding both as one idealised polymer
	Arguments:
	----------
		pose1: First pose, a protein, DNA, or RNA pose
		pose2: Second pose, which must share its Type with pose1
		fuse:  False appends pose2 as separate chains, renaming any
			whose ID collides; True rebuilds the joined sequence as a
			single polymer, discarding the original coordinates
	Returns:
	--------
		Pose: New combined pose, renumbered from zero
	'''
	t1, t2 = pose1.data.get('Type'), pose2.data.get('Type')
	if t1 is None or t2 is None:
		raise Exception('Concatenate: empty pose given')
	if t1 != t2: raise Exception(f'Cannot concatenate {t1} with {t2}')
	is_pro = (t1 == 'Protein')
	rk = 'Amino Acids' if is_pro else 'Nucleotides'
	if fuse:
		f1, f2 = pose1.data['FASTA'], pose2.data['FASTA']
		new = Pose()
		new.Build(''.join(f1[c] for c in sorted(f1))
			+ ''.join(f2[c] for c in sorted(f2)), fmt=t1)
		return new
	az = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	taken = {v[1] for v in pose1.data[rk].values()}
	remap = {}
	for c in sorted({v[1] for v in pose2.data[rk].values()}):
		use = c
		if c in taken:
			use = next((x for x in az if x not in taken), None)
			if use is None: raise Exception('Ran out of chain letters')
			remap[c] = use
		taken.add(use)
	new = Pose()
	new.data = {
		'Type': t1, 'Energy': 0, 'Rg': 0, 'Mass': 0,
		'Size': {}, 'FASTA': {}, 'SS': {},
		'Nucleotides': None if is_pro else {},
		'Amino Acids': {} if is_pro else None,
		'Atoms': {}, 'Bonds': {}, 'Coordinates': np.zeros((0, 3))}
	coords, ai, ri = [], 0, 0
	for src, cmap in ((pose1, {}), (pose2, remap)):
		sa, sb = src.data['Atoms'], src.data['Bonds']
		sc, sr = src.data['Coordinates'], src.data[rk]
		old_a = sorted(sa.keys())
		a_map = {o: ai + i for i, o in enumerate(old_a)}
		for o in old_a:
			new.data['Atoms'][a_map[o]] = list(sa[o])
			coords.append(sc[o])
		for o in old_a:
			new.data['Bonds'][a_map[o]] = sorted(
				a_map[b] for b in sb.get(o, []) if b in a_map)
		old_r = sorted(sr.keys())
		for i, o in enumerate(old_r):
			row = list(sr[o])
			row[1] = cmap.get(row[1], row[1])
			row[2] = [a_map[a] for a in row[2] if a in a_map]
			row[3] = [a_map[a] for a in row[3] if a in a_map]
			new.data[rk][ri + i] = row
		ai += len(old_a)
		ri += len(old_r)
	new.data['Coordinates'] = (np.array(coords, dtype=float)
		if coords else np.zeros((0, 3)))
	new._update()
	return new

def Translate(sequence, fmt='protein', organism='ecoli',
		src=None):
	'''
	Translate between DNA, RNA, and protein, detecting the source
	alphabet from the characters present in the sequence
	Arguments:
	----------
		sequence: Input sequence, alphabet auto-detected as DNA, RNA,
			or protein
		fmt:      Target alphabet, 'protein' (default), 'dna', or 'rna'
		organism: Codon usage for back-translation, 'ecoli' (default)
			or 'human'
		src:      Source alphabet, None (default) detects it from the
			characters present, or force 'protein', 'dna', or 'rna'
	Returns:
	--------
		str: Translated sequence, uppercased with gaps and spaces
		stripped; stop codons are '*' and unknown codons are 'X'
	'''
	if not sequence: raise Exception('Empty sequence')
	tgt = fmt.lower()
	if tgt not in ('protein', 'dna', 'rna'):
		raise Exception(f'Unknown target fmt: {fmt}')
	if src is not None and src.lower() not in ('protein', 'dna', 'rna'):
		raise Exception(f'Unknown source alphabet: {src}')
	chars = set(sequence.upper()) - {'-', '*', 'N'}
	if src is not None: src = src.lower()
	elif not chars: src = 'protein'
	elif chars <= set('ACGT'): src = 'dna'
	elif chars <= set('ACGU'): src = 'rna'
	elif chars <= set('ACDEFGHIKLMNPQRSTVWY'): src = 'protein'
	elif chars - set('ACGT') - set('ACGU'): src = 'protein'
	else: src = 'dna'
	s = sequence.upper().replace('-', '').replace(' ', '')
	if src == tgt: return s
	if src == 'dna' and tgt == 'rna': return s.replace('T', 'U')
	if src == 'rna' and tgt == 'dna': return s.replace('U', 'T')
	if src in ('dna', 'rna') and tgt == 'protein':
		codon = {
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
		return ''.join(codon.get(dna[i:i+3], 'X')
			for i in range(0, len(dna), 3))
	if src == 'protein' and tgt in ('dna', 'rna'):
		best = {
			'ecoli': {'F':'TTT','L':'CTG','I':'ATT','M':'ATG',
				'V':'GTG','S':'AGC','P':'CCG','T':'ACC',
				'A':'GCG','Y':'TAT','*':'TAA','H':'CAT',
				'Q':'CAG','N':'AAC','K':'AAA','D':'GAT',
				'E':'GAA','C':'TGC','W':'TGG','R':'CGC',
				'G':'GGC','J':'ATG','X':'TGG','O':'ACC','Z':'TAT'},
			'human': {'F':'TTC','L':'CTG','I':'ATC','M':'ATG',
				'V':'GTG','S':'AGC','P':'CCC','T':'ACC',
				'A':'GCC','Y':'TAC','*':'TGA','H':'CAC',
				'Q':'CAG','N':'AAC','K':'AAG','D':'GAC',
				'E':'GAG','C':'TGC','W':'TGG','R':'CGG',
				'G':'GGC','J':'ATG','X':'TGG','O':'ACC','Z':'TAC'}
			}.get(organism.lower())
		if best is None: raise Exception(
			f"Unknown organism {organism!r}; use 'ecoli' or 'human'")
		bad = [a for a in s if a not in best]
		if bad: raise Exception(f'No codon for residue {bad[0]!r}')
		dna = ''.join(best[a] for a in s)
		return dna if tgt == 'dna' else dna.replace('T', 'U')
	raise Exception(f'Unsupported translation {src} -> {tgt}')

def HydrogenBondMap(pose):
	'''
	Backbone hydrogen bond map from the DSSP electrostatic criterion
	(Kabsch & Sander 1983, Biopolymers 22:2577)
	E = 0.084*(1/r_ON + 1/r_CH - 1/r_OH
	- 1/r_CN)*1389.3545756874 kJ/mol, accepted when E < -2.092 kJ/mol
	Arguments:
	----------
		pose: Protein pose carrying backbone N, C, and O atoms
	Returns:
	--------
		np.ndarray: (N_atoms, N_atoms) int8 matrix where 0 is no bond,
		1 marks the donor nitrogen and 2 marks the acceptor oxygen
	'''
	if pose.data.get('Type') != 'Protein':
		raise Exception('HydrogenBondMap only supports protein poses')
	AAs = pose.data.get('Amino Acids') or {}
	if not AAs: raise Exception('Pose has no amino acids')
	atoms, co = pose.data['Atoms'], pose.data['Coordinates']
	M = np.zeros((max(atoms.keys()) + 1 if atoms else 0,) * 2, dtype=np.int8)
	res = sorted(AAs.keys())
	chains = [AAs[r][1] for r in res]
	tri = [AAs[r][5].upper() for r in res]
	ai_of = {r: {atoms[a][0]: a for a in AAs[r][2]} for r in res}
	H = [None] * len(res)
	for k, r in enumerate(res):
		if tri[k] in ('PRO', 'DPR') or k == 0: continue
		if chains[k] != chains[k-1]: continue
		idx, prev = ai_of[r], ai_of[res[k-1]]
		ah = idx.get('H', idx.get('1H'))
		if ah is not None: H[k] = co[ah]; continue
		if 'N' not in idx or 'C' not in prev or 'O' not in prev: continue
		d = co[prev['C']] - co[prev['O']]
		nm = float(np.linalg.norm(d))
		if nm > 0.001: H[k] = co[idx['N']] + d / nm
	Ni = [ai_of[r].get('N', -1) for r in res]
	Ci = [ai_of[r].get('C', -1) for r in res]
	Oi = [ai_of[r].get('O', -1) for r in res]
	for ki in range(len(res)):
		if H[ki] is None or Ni[ki] < 0: continue
		n, h = co[Ni[ki]], H[ki]
		for kj in range(len(res)):
			if abs(ki - kj) <= 1 or chains[ki] != chains[kj]: continue
			if Ci[kj] < 0 or Oi[kj] < 0: continue
			c, o = co[Ci[kj]], co[Oi[kj]]
			d = [float(np.linalg.norm(o - n)), float(np.linalg.norm(c - h)),
				float(np.linalg.norm(o - h)), float(np.linalg.norm(c - n))]
			if min(d) < 0.001: continue
			e = 0.084 * (1/d[0] + 1/d[1] - 1/d[2] - 1/d[3]) * 1389.3545756874
			if e >= -2.092: continue
			M[Ni[ki], Oi[kj]] = 1
			M[Oi[kj], Ni[ki]] = 2
	return M

def ContactMap(pose):
	'''
	Residue-residue Euclidean distance map in angstroms, measured
	between CA atoms for proteins and C1' atoms for DNA and RNA
	Arguments:
	----------
		pose: Protein or nucleic acid pose with a non-empty residue table
	Returns:
	--------
		np.ndarray: (N_residues, N_residues) matrix of pairwise
		distances, ordered by residue index and zero on the diagonal
	'''
	mol = pose.data.get('Type')
	if mol is None: raise Exception('Empty pose')
	if mol == 'Protein': src, ref = pose.data['Amino Acids'], 'CA'
	elif mol in ('DNA', 'RNA'): src, ref = pose.data['Nucleotides'], "C1'"
	else: raise Exception(f'Unknown molecule type: {mol}')
	if not src: raise Exception('Pose has no residues')
	atoms, co = pose.data['Atoms'], pose.data['Coordinates']
	pts = []
	for ri in sorted(src.keys()):
		pos = next((co[a] for a in src[ri][2] if atoms[a][0] == ref), None)
		if pos is None: raise Exception(f'Residue {ri} has no {ref} atom')
		pts.append(pos)
	pts = np.array(pts, dtype=float)
	d = pts[:, None, :] - pts[None, :, :]
	mat = np.sqrt((d * d).sum(-1))
	np.fill_diagonal(mat, 0.0)
	return mat

def PCR(dna_sequence):
	'''
	Design forward and reverse PCR primers for a DNA template using a
	five tier relaxation search, with SantaLucia 1998 nearest-neighbour
	melting temperatures and the SantaLucia 1998 salt correction
	Arguments:
	----------
		dna_sequence: Template DNA sequence, A/C/G/T only, at least 36 bp
	Returns:
	--------
		str: Forward primer, taken from the 5' end of the template
		str: Reverse primer, reverse complement of the 3' end
		str: Warning naming the relaxed tier and the gates it missed,
		or None when the Ideal tier was satisfied
	'''
	seq = dna_sequence.upper()
	bad = [c for c in seq if c not in 'ACGT']
	if bad: raise Exception(f'Illegal base {bad[0]!r} in template')
	if len(seq) < 36:
		raise Exception('Template too short for primer design (<36 bp)')
	cmap = str.maketrans('ACGTN', 'TGCAN')
	rc = seq[::-1].translate(cmap)
	DH = {'AA':-7.9,'TT':-7.9,'AT':-7.2,'TA':-7.2,
		'CA':-8.5,'TG':-8.5,'GT':-8.4,'AC':-8.4,
		'CT':-7.8,'AG':-7.8,'GA':-8.2,'TC':-8.2,
		'CG':-10.6,'GC':-9.8,'GG':-8.0,'CC':-8.0}
	DS = {'AA':-22.2,'TT':-22.2,'AT':-20.4,'TA':-21.3,
		'CA':-22.7,'TG':-22.7,'GT':-22.4,'AC':-22.4,
		'CT':-21.0,'AG':-21.0,'GA':-22.2,'TC':-22.2,
		'CG':-27.2,'GC':-24.4,'GG':-19.9,'CC':-19.9}
	tiers = [
		{'label':'Ideal', 'len':(18,25), 'gc':(40.0,60.0),
			'tm':(55.0,65.0), 'clamp':True, 'max_run':4,
			'hp':True, 'sd':True, 'xd':True, 'dtm':2.0},
		{'label':'Good', 'len':(18,28), 'gc':(35.0,65.0),
			'tm':(50.0,68.0), 'clamp':True, 'max_run':5,
			'hp':True, 'sd':True, 'xd':True, 'dtm':3.0},
		{'label':'Fair', 'len':(18,30), 'gc':(25.0,75.0),
			'tm':(45.0,72.0), 'clamp':False, 'max_run':5,
			'hp':False, 'sd':True, 'xd':False, 'dtm':5.0},
		{'label':'Poor', 'len':(18,30), 'gc':None, 'tm':None,
			'clamp':False, 'max_run':None,
			'hp':False, 'sd':False, 'xd':False, 'dtm':8.0},
		{'label':'Last resort', 'len':(18,30), 'gc':None,
			'tm':None, 'clamp':False, 'max_run':None,
			'hp':False, 'sd':False, 'xd':False,
			'dtm':float('inf')}]
	max_off = max(0, min(60, len(seq) - 18))
	chosen, chosen_tier = None, None
	for ti, tier in enumerate(tiers):
		fwd_pool, rev_pool = [], []
		lo, hi = tier['len']
		for source, pool in ((seq, fwd_pool), (rc, rev_pool)):
			for off in range(max_off + 1):
				for L in range(lo, hi + 1):
					cand = source[off:off+L]
					if len(cand) < L: continue
					if tier['clamp'] and cand[-1] not in 'GC': continue
					gc = 100.0 * (cand.count('G') + cand.count('C')) / L
					if tier['gc'] and not tier['gc'][0] <= gc <= tier['gc'][1]:
						continue
					mr = tier['max_run']
					if mr and any(b * mr in cand for b in 'ACGT'): continue
					if tier['hp'] and any(cand.find(
							cand[i:i+s][::-1].translate(cmap),
							i + s + 3) != -1
							for s in range(4, L // 2 + 1)
							for i in range(L - 2 * s - 2)): continue
					if tier['sd'] and 0 <= cand.find(
							cand[-5:][::-1].translate(cmap)) <= L - 6: continue
					dH = sum(DH.get(cand[i:i+2], 0.0) for i in range(L - 1))
					dS = sum(DS.get(cand[i:i+2], 0.0) for i in range(L - 1))
					dH += 0.1 if cand[0] in 'GC' else 2.3
					dS += -2.8 if cand[0] in 'GC' else 4.1
					dH += 0.1 if cand[-1] in 'GC' else 2.3
					dS += -2.8 if cand[-1] in 'GC' else 4.1
					dS += 0.368 * (L - 1) * math.log(0.05)
					tm = ((dH * 1000.0) / (dS
						+ 1.987 * math.log(250e-9 / 4.0))) - 273.15
					if tier['tm'] and not tier['tm'][0] <= tm <= tier['tm'][1]:
						continue
					pool.append((off, cand, tm, gc))
		if not fwd_pool or not rev_pool: continue
		best, best_score = None, float('inf')
		for off1, fwd, tmf, gcf in fwd_pool:
			for off2, rev, tmr, gcr in rev_pool:
				dT = abs(tmf - tmr)
				if dT > tier['dtm']: continue
				if tier['xd'] and (
						fwd[-5:][::-1].translate(cmap) in rev
						or rev[-5:][::-1].translate(cmap) in fwd): continue
				score = (dT * 5.0 + abs(tmf - 60.0) + abs(tmr - 60.0)
					+ abs(gcf - 50.0) * 0.1 + abs(gcr - 50.0) * 0.1
					+ (off1 + off2) * 0.05)
				if score >= best_score: continue
				best_score = score
				best = (fwd, rev, tmf, tmr, gcf, gcr)
		if best is None: continue
		chosen, chosen_tier = best, ti
		break
	if chosen is None:
		raise Exception('No primer pair found even at last-resort tier')
	fwd, rev, tmf, tmr, gcf, gcr = chosen
	if chosen_tier == 0: return (fwd, rev, None)
	reasons = []
	if not (40.0 <= gcf <= 60.0 and 40.0 <= gcr <= 60.0):
		reasons.append('GC% outside 40-60')
	if not (55.0 <= tmf <= 65.0 and 55.0 <= tmr <= 65.0):
		reasons.append('Tm outside 55-65 \u00b0C')
	if abs(tmf - tmr) > 2.0: reasons.append('|\u0394Tm| > 2 \u00b0C')
	if fwd[-1] not in 'GC' or rev[-1] not in 'GC':
		reasons.append('GC clamp missing')
	if (fwd[-5:][::-1].translate(cmap) in rev
			or rev[-5:][::-1].translate(cmap) in fwd):
		reasons.append('primer-pair cross-dimer')
	return (fwd, rev, f'Warning: Suboptimal PCR primers '
		f'({tiers[chosen_tier]["label"]} tier) \u2014 '
		f'{"; ".join(reasons) if reasons else "gates relaxed"}')

def RMSD(pose1, pose2, alg='align', export=None):
	'''
	Root mean square deviation between two poses, protein or nucleic
	acid, over CA atoms for proteins and C1' atoms for nucleic acids
	Arguments:
	----------
		pose1:  First pose, a protein or a nucleic acid
		pose2:  Second pose, which must share its Type with pose1
		alg:    'align' (default) pairs residues by Needleman-Wunsch then
			fits with outlier-rejecting Kabsch; 'kabsch' fits by SVD;
			'quaternion' fits by the Horn eigenvalue method; 'simple'
			centres both poses and applies no rotation
		export: Filename for the aligned PDB pair, None skips the export
	Returns:
	--------
		float: RMSD in angstroms, rounded to 5 decimals
	'''
	if alg not in ('align', 'kabsch', 'quaternion', 'simple'):
		raise Exception('Unknown algorithm: ' + str(alg))
	t1, t2 = pose1.data['Type'], pose2.data['Type']
	if (t1 == 'Protein') != (t2 == 'Protein'):
		raise Exception(f'Cannot align {t1} with {t2}: '
			'cannot mix protein and nucleic acid')
	is_pro = (t1 == 'Protein')
	rk = 'Amino Acids' if is_pro else 'Nucleotides'
	ra = 'CA' if is_pro else "C1'"
	atoms1, co1, res1 = (pose1.data['Atoms'],
		pose1.data['Coordinates'], pose1.data[rk])
	atoms2, co2, res2 = (pose2.data['Atoms'],
		pose2.data['Coordinates'], pose2.data[rk])
	if alg != 'align':
		c1 = [c for c in (next((co1[ai].copy().astype(float)
			for ai in res1[ri][2] if atoms1[ai][0] == ra), None)
			for ri in sorted(res1.keys())) if c is not None]
		c2 = [c for c in (next((co2[ai].copy().astype(float)
			for ai in res2[ri][2] if atoms2[ai][0] == ra), None)
			for ri in sorted(res2.keys())) if c is not None]
		if not c1 or not c2:
			raise Exception(f'No {ra} atoms found in one or both poses')
		k = min(len(c1), len(c2))
		P, Q = np.array(c1[:k]), np.array(c2[:k])
		t_P, t_Q = P.mean(axis=0), Q.mean(axis=0)
		P, Q = P - t_P, Q - t_Q
	if alg == 'simple': R = np.eye(3)
	elif alg == 'kabsch':
		U, _, Vt = np.linalg.svd(P.T @ Q)
		d = np.sign(np.linalg.det(Vt.T @ U.T))
		R = Vt.T @ np.diag(np.array([1.0, 1.0, d])) @ U.T
	elif alg == 'quaternion':
		H = P.T @ Q
		a, b, c = H[0]; d, e, f = H[1]; g, h, k = H[2]
		F = np.array([
			[a+e+k, f-h,    g-c,    b-d   ],
			[f-h,   a-e-k,  b+d,    c+g   ],
			[g-c,   b+d,   -a+e-k,  f+h   ],
			[b-d,   c+g,    f+h,   -a-e+k ]])
		q0, q1, q2, q3 = np.linalg.eigh(F)[1][:, -1]
		R = np.array([
			[q0*q0+q1*q1-q2*q2-q3*q3,
				2*(q1*q2-q0*q3),         2*(q1*q3+q0*q2)],
			[2*(q1*q2+q0*q3),
				q0*q0-q1*q1+q2*q2-q3*q3, 2*(q2*q3-q0*q1)],
			[2*(q1*q3-q0*q2),
				2*(q2*q3+q0*q1), q0*q0-q1*q1-q2*q2+q3*q3]])
	else:
		rk1, rk2 = sorted(res1.keys()), sorted(res2.keys())
		seq1 = ''.join(res1[k][0].upper() for k in rk1)
		seq2 = ''.join(res2[k][0].upper() for k in rk2)
		m, n, gap = len(seq1), len(seq2), -1.0
		S = np.array([[(_blosum(a, b) if is_pro
			else (1.0 if a == b else -0.5))
			for b in seq2] for a in seq1], dtype=float)
		step = np.arange(n + 1) * gap
		dp = np.zeros((m + 1, n + 1))
		dp[:, 0] = np.arange(m + 1) * gap
		dp[0, :] = step
		for i in range(1, m + 1):
			base = np.maximum(dp[i-1, :-1] + S[i-1], dp[i-1, 1:] + gap)
			run = np.concatenate(([dp[i, 0]], base - step[1:]))
			dp[i] = np.maximum.accumulate(run) + step
		pairs, i, j = [], m, n
		while i > 0 and j > 0:
			if abs(dp[i, j] - (dp[i-1, j-1] + S[i-1, j-1])) < 1e-9:
				pairs.append((i - 1, j - 1)); i -= 1; j -= 1
			elif abs(dp[i, j] - (dp[i-1, j] + gap)) < 1e-9: i -= 1
			else: j -= 1
		pairs.reverse()
		if len(pairs) < 3: raise Exception('Too few aligned residue pairs')
		P_aln = np.array([next(co1[ai].copy().astype(float)
			for ai in res1[rk1[ii]][2]
			if atoms1[ai][0] == ra) for ii, _ in pairs])
		Q_aln = np.array([next(co2[ai].copy().astype(float)
			for ai in res2[rk2[jj]][2]
			if atoms2[ai][0] == ra) for _, jj in pairs])
		mask = np.ones(len(pairs), dtype=bool)
		for _ in range(6):
			Pm, Qm = P_aln[mask], Q_aln[mask]
			t_P, t_Q = Pm.mean(axis=0), Qm.mean(axis=0)
			P, Q = Pm - t_P, Qm - t_Q
			U, _, Vt = np.linalg.svd(P.T @ Q)
			d = np.sign(np.linalg.det(Vt.T @ U.T))
			R = Vt.T @ np.diag(np.array([1.0, 1.0, d])) @ U.T
			dists = np.sqrt((((P_aln - t_P)
				- (Q_aln - t_Q) @ R) ** 2).sum(axis=1))
			new_mask = dists < 2.0
			if np.array_equal(new_mask, mask) or new_mask.sum() < 3: break
			mask = new_mask
	diff = P - Q @ R
	rmsd = np.sqrt(np.mean((diff ** 2).sum(axis=1)))
	if export is not None:
		orig = pose2.data['Coordinates'].copy()
		pose2.data['Coordinates'] = (orig - t_Q) @ R + t_P
		pose1.Export(export[:-4] + '_1' + export[-4:])
		pose2.Export(export[:-4] + '_2' + export[-4:])
		pose2.data['Coordinates'] = orig
	return round(float(rmsd), 5)

def BLAST(seq1, seq2):
	'''
	Pairwise protein alignment by Smith-Waterman with BLOSUM62 and
	affine gaps, scored with a Karlin-Altschul expect value
	Arguments:
	----------
		seq1: FASTA sequence of the first, query, protein
		seq2: FASTA sequence of the second, subject, protein
	Returns:
	--------
		str: BLAST-style alignment report in 60-column blocks
		float: Percent identity over the aligned region
		float: Karlin-Altschul expect value
	'''
	seq1, seq2 = seq1.upper(), seq2.upper()
	m, n = len(seq1), len(seq2)
	go, ge, INF = 11, 1, float('-inf')
	H = np.zeros((m+1, n+1))
	E = np.full((m+1, n+1), INF)
	F = np.full((m+1, n+1), INF)
	tb = np.zeros((m+1, n+1), dtype=np.int8)
	S = np.array([[_blosum(a, b) for b in seq2] for a in seq1],
		dtype=float)
	best, bi, bj = 0.0, 0, 0
	for i in range(1, m+1):
		F[i, 1:] = np.maximum(H[i-1, 1:] - go - ge, F[i-1, 1:] - ge)
		for j in range(1, n+1):
			diag = H[i-1, j-1] + S[i-1, j-1]
			E[i, j] = max(H[i, j-1] - go - ge, E[i, j-1] - ge)
			h = max(0.0, diag, E[i, j], F[i, j])
			H[i, j] = h
			if h > best: best, bi, bj = h, i, j
			tb[i, j] = (0 if h == 0 else 1 if h == diag
				else 2 if h == F[i, j] else 3)
	if best == 0:
		raise Exception('No alignment found between the sequences')
	aq, asb, i, j = [], [], bi, bj
	while i > 0 and j > 0 and H[i, j] > 0:
		t = int(tb[i, j])
		if t == 1:
			aq.append(seq1[i-1]); asb.append(seq2[j-1]); i -= 1; j -= 1
		elif t == 2: aq.append(seq1[i-1]); asb.append('-'); i -= 1
		else: aq.append('-'); asb.append(seq2[j-1]); j -= 1
	aq, asb = ''.join(reversed(aq)), ''.join(reversed(asb))
	qs, ss, ln = i + 1, j + 1, len(aq)
	n_id = sum(1 for a, b in zip(aq, asb) if a == b and a != '-')
	n_pos = sum(1 for a, b in zip(aq, asb)
		if a != '-' and b != '-' and _blosum(a, b) > 0)
	n_gap = aq.count('-') + asb.count('-')
	pct = round(n_id / ln * 100, 2)
	lam, K = 0.267, 0.041
	e_value = K * m * n * math.exp(-lam * best)
	bits = (lam * best - math.log(K)) / math.log(2)
	mid = ''.join(' ' if a == '-' or b == '-'
		else '|' if a == b
		else '+' if _blosum(a, b) > 0
		else ' ' for a, b in zip(aq, asb))
	out = [
		f'Query length={m}  Subject length={n}', '',
		(f'Score: {bits:.1f} bits ({int(best)}), '
			f'E-value: {e_value:.3e}'),
		(f'Identities: {n_id}/{ln} ({pct}%), '
			f'Positives: {n_pos}/{ln} ({round(n_pos / ln * 100, 1)}%), '
			f'Gaps: {n_gap}/{ln} ({round(n_gap / ln * 100, 1)}%)'), '']
	qp, sp = qs, ss
	for st in range(0, ln, 60):
		bq, bm, bs = aq[st:st+60], mid[st:st+60], asb[st:st+60]
		qr, sr = len(bq) - bq.count('-'), len(bs) - bs.count('-')
		out += [
			f'Query  {qp:>6}  {bq}  {qp+qr-1}',
			f'       {"":>6}  {bm}',
			f'Sbjct  {sp:>6}  {bs}  {sp+sr-1}', '']
		qp += qr; sp += sr
	return '\n'.join(out), pct, e_value

def MSA(sequences):
	'''
	Progressive multiple sequence alignment in the ClustalW style, with
	BLOSUM62 profile scoring and a mean-field direct coupling analysis
	(Morcos et al. 2011, PNAS 108:E1293) with average-product correction
	(Dunn et al. 2008, Bioinformatics 24:333)
	Arguments:
	----------
		sequences: FASTA sequences to align, at least two
	Returns:
	--------
		str: ClustalW-style formatted alignment text
		list: Gap-padded aligned sequences in input order
		list: Per-column conservation, 1 - H/log2(20), in [0, 1]
		list: Per-column Shannon entropy in bits
		np.ndarray: (L, 20) PSSM in the order ARNDCQEGHILKMFPSTWYV
		np.ndarray: (L, L) APC-corrected direct-information matrix
	'''
	def proffreq(profile):
		'''
		Per-column residue frequency profile over a fixed alphabet
		Arguments:
		----------
			profile: List of equal-length gapped sequences
		Returns:
		--------
			np.ndarray: (L, 20) frequencies normalised by the
			non-gap count of each column, and zero where a column is all
			gaps or holds only residues outside the alphabet
		'''
		idx = {c: i for i, c in enumerate(AA)}
		F = np.zeros((len(profile[0]), 20))
		for row in profile:
			for ci, c in enumerate(row):
				k = idx.get(c, -1)
				if k >= 0: F[ci, k] += 1
		d = F.sum(axis=1, keepdims=True)
		with np.errstate(divide='ignore', invalid='ignore'):
			return np.divide(F, d, where=(d > 0), out=np.zeros_like(F))
	n = len(sequences)
	if n < 2: raise Exception('MSA requires at least 2 sequences')
	seqs = [s.upper() for s in sequences]
	labels = [f'Seq{i+1}' for i in range(n)]
	go, ge, INF = 11, 1, float('-inf')
	AA = 'ARNDCQEGHILKMFPSTWYV'
	BM = np.array([[_blosum(a, b) for b in AA] for a in AA],
		dtype=float)
	dist = np.zeros((n, n))
	for i in range(n):
		for j in range(i+1, n):
			try: dd = 1.0 - BLAST(seqs[i], seqs[j])[1] / 100.0
			except Exception: dd = 1.0
			dist[i, j] = dist[j, i] = dd
	sizes = {k: 1 for k in range(n)}
	active, d, merge_order = list(range(n)), dist.copy(), []
	for _ in range(n - 1):
		bi, bj, best = -1, -1, float('inf')
		for x in range(len(active)):
			for y in range(x + 1, len(active)):
				ii, jj = active[x], active[y]
				if d[ii, jj] >= best: continue
				best, bi, bj = d[ii, jj], ii, jj
		merge_order.append((bi, bj))
		ni, nj = sizes[bi], sizes[bj]
		for k in active:
			if k == bi or k == bj: continue
			d[bi, k] = d[k, bi] = (
				ni * d[bi, k] + nj * d[bj, k]) / (ni + nj)
		sizes[bi] += sizes[bj]
		active.remove(bj)
	profiles = {k: [seqs[k]] for k in range(n)}
	for ci, cj in merge_order:
		p1, p2 = profiles[ci], profiles[cj]
		L1, L2 = len(p1[0]), len(p2[0])
		H = np.zeros((L1+1, L2+1))
		E = np.full((L1+1, L2+1), INF)
		F = np.full((L1+1, L2+1), INF)
		tb = np.zeros((L1+1, L2+1), dtype=np.int8)
		H[1:, 0] = -(go + ge * np.arange(1, L1+1)); tb[1:, 0] = 2
		H[0, 1:] = -(go + ge * np.arange(1, L2+1)); tb[0, 1:] = 3
		with np.errstate(all='ignore'):
			CS = proffreq(p1) @ BM @ proffreq(p2).T
		for i in range(1, L1+1):
			F[i, 1:] = np.maximum(H[i-1, 1:] - go - ge, F[i-1, 1:] - ge)
			for j in range(1, L2+1):
				diag = H[i-1, j-1] + CS[i-1, j-1]
				E[i, j] = max(H[i, j-1] - go - ge, E[i, j-1] - ge)
				h = max(diag, E[i, j], F[i, j])
				H[i, j] = h
				tb[i, j] = (1 if h == diag
					else 2 if h == F[i, j] else 3)
		np1, np2 = [[] for _ in p1], [[] for _ in p2]
		i, j = L1, L2
		while i > 0 or j > 0:
			t = 3 if i == 0 else 2 if j == 0 else int(tb[i, j])
			for k, r in enumerate(p1):
				np1[k].append(r[i-1] if t != 3 else '-')
			for k, r in enumerate(p2):
				np2[k].append(r[j-1] if t != 2 else '-')
			if t != 3: i -= 1
			if t != 2: j -= 1
		profiles[ci] = ([''.join(reversed(r)) for r in np1]
			+ [''.join(reversed(r)) for r in np2])
		del profiles[cj]
	final = list(profiles.values())[0]
	L = len(final[0])
	lw = max(max(len(lb) for lb in labels), 4)
	con = []
	for ci in range(L):
		ng = [final[k][ci] for k in range(n) if final[k][ci] != '-']
		pairs = [_blosum(a, b) for x, a in enumerate(ng) for b in ng[x+1:]]
		con.append(
			' ' if not ng
			else '*' if len(ng) == n and all(c == ng[0] for c in ng)
			else ('*' if len(ng) == 1 else ' ') if not pairs
			else ':' if all(s > 0 for s in pairs)
			else '.' if sum(pairs) / len(pairs) > 0 else ' ')
	con = ''.join(con)
	out = [f'Multiple Sequence Alignment ({n} sequences, {L} columns)',
		'']
	pos = [0] * n
	for st in range(0, L, 60):
		for k, lb in enumerate(labels):
			blk = final[k][st:st+60]
			pos[k] += len(blk) - blk.count('-')
			out.append(f'{lb:<{lw}}  {blk}  {pos[k]}')
		out.append(f'{"":>{lw}}  {con[st:st+60]}')
		out.append('')
	alphabet = '-' + AA
	q, B = len(alphabet), n
	a2i = {c: i for i, c in enumerate(alphabet)}
	M = np.zeros((B, L), dtype=np.int8)
	for bi, s in enumerate(final):
		for ci, ch in enumerate(s):
			M[bi, ci] = a2i.get(ch, 0)
	log2_20 = math.log2(20)
	entropy, conservation = [], []
	for ci in range(L):
		nz = M[:, ci][M[:, ci] != 0]
		p = np.bincount(nz, minlength=q)[1:] / max(len(nz), 1)
		nzp = p[p > 0]
		Hc = float(-np.sum(nzp * np.log2(nzp))) if len(nz) else 0.0
		entropy.append(round(Hc, 4))
		conservation.append(round(1.0 - Hc / log2_20, 4)
			if len(nz) else 0.0)
	pssm = np.zeros((L, 20), dtype=float)
	for ci in range(L):
		nz = M[:, ci][M[:, ci] != 0]
		counts = np.bincount(nz, minlength=q)[1:]
		pssm[ci] = np.log2((counts + 1.0) / (counts.sum() + 20.0) * 20.0)
	theta, weights = 0.2, np.ones(B)
	simthr = (1.0 - theta) * L
	eq_count = np.zeros(B)
	for a in range(B):
		for b in range(a, B):
			if a == b: eq_count[a] += 1; continue
			if int((M[a] == M[b]).sum()) < simthr: continue
			eq_count[a] += 1; eq_count[b] += 1
	if B > 1: weights = 1.0 / eq_count
	Beff = float(weights.sum())
	Pi = np.zeros((L, q))
	for bi in range(B):
		for ci in range(L):
			Pi[ci, M[bi, ci]] += weights[bi]
	Pi /= Beff
	lam = 0.5
	Pi_pc = (1.0 - lam) * Pi + lam / q
	qm = q - 1
	C = np.zeros((L * qm, L * qm))
	for i in range(L):
		for j in range(L):
			pij = np.zeros((q, q))
			np.add.at(pij, (M[:, i], M[:, j]), weights)
			pij = (1.0 - lam) * (pij / Beff) + lam / (q * q)
			if i == j: pij = np.diag(Pi_pc[i])
			C[i*qm:(i+1)*qm, j*qm:(j+1)*qm] = (pij[:qm, :qm]
				- np.outer(Pi_pc[i, :qm], Pi_pc[j, :qm]))
	try: invC = np.linalg.inv(C)
	except np.linalg.LinAlgError: invC = np.linalg.pinv(C)
	dca_raw = np.zeros((L, L))
	for i in range(L):
		for j in range(i + 1, L):
			W = np.ones((q, q))
			W[:qm, :qm] = np.exp(
				-invC[i*qm:i*qm+qm, j*qm:j*qm+qm])
			mu1, mu2 = np.ones(q) / q, np.ones(q) / q
			pi_i, pi_j = Pi_pc[i], Pi_pc[j]
			for _ in range(100):
				nm1 = pi_i / (mu2 @ W.T)
				nm2 = pi_j / (mu1 @ W)
				nm1 /= nm1.sum(); nm2 /= nm2.sum()
				done = (np.max(np.abs(nm1 - mu1)) < 1e-6
					and np.max(np.abs(nm2 - mu2)) < 1e-6)
				mu1, mu2 = nm1, nm2
				if done: break
			Pdir = W * np.outer(mu1, mu2)
			Pdir /= Pdir.sum()
			Pfac = np.outer(pi_i, pi_j)
			mask = (Pdir > 1e-12) & (Pfac > 1e-12)
			dca_raw[i, j] = dca_raw[j, i] = float(np.sum(
				Pdir[mask] * np.log(Pdir[mask] / Pfac[mask])))
	dca = np.zeros((L, L))
	if L > 1:
		row_mean = dca_raw.sum(axis=1) / (L - 1)
		total_mean = dca_raw.sum() / (L * (L - 1))
		dca = (dca_raw - np.outer(row_mean, row_mean) / total_mean
			if total_mean > 0 else dca_raw.copy())
		np.fill_diagonal(dca, 0.0)
	return '\n'.join(out), final, conservation, entropy, pssm, dca

def Cyclise(pose, mode='head-to-tail', res1=None, atom1=None,
		res2=None, atom2=None, precoil=True, recoil=True):
	'''
	Form an intramolecular bond to build a cyclic peptide (macrocycle)
	Arguments:
	----------
		pose:    Pose to cyclise, modified in place
		mode:    'head-to-tail' amide-bonds the N-terminus to the
			C-terminus (default), 'sidechain' bonds the two named
			atoms res1/atom1 and res2/atom2, such as a disulfide
		res1:    First residue index, sidechain mode only
		atom1:   Atom name within res1, sidechain mode only
		res2:    Second residue index, sidechain mode only
		atom2:   Atom name within res2, sidechain mode only
		precoil: head-to-tail only, coil the backbone and run cyclic
			coordinate descent (Canutescu & Dunbrack 2003, Protein Sci
			12:963) so the closing bond forms near 1.33 A rather than
			across a stretched gap (default True)
	Returns:
	--------
		Modifies the pose in place and returns no value. Head-to-tail
		drops the surplus N-terminal hydrogens and the C-terminal OXT,
		reindexes every atom, adds the closing bond, reassigns Gasteiger
		charges, and records the closure in pose.data['Cyclic'].
		RotateDihedral and AdjustDistance are undefined on a closed ring
		and must not be used afterwards; refine with tools.Minimise, for
		which the 'Default' force field is recommended
	'''
	atoms = pose.data['Atoms']
	src = (pose.data['Amino Acids'] or pose.data['Nucleotides'])
	def atomof(res, nm):
		'''
		Index of the atom called nm within residue res
		Arguments:
		----------
			res: Residue index into the pose residue table
			nm:  Atom name to look for, backbone or sidechain
		Returns:
		--------
			int: Atom index, or None when the residue has no such atom
		'''
		return next((a for a in src[res][2] + src[res][3]
			if atoms[a][0] == nm), None)
	rr = sorted(src)
	if mode == 'head-to-tail' and precoil:
		if recoil:
			for ri in rr:
				for ang, val in (('PHI', 0.0), ('PSI', 180.0)):
					try:
						if not np.isnan(pose.GetDihedral(ri, ang)):
							pose.RotateDihedral(ri, val, ang)
					except Exception: pass
		nC, n0 = atomof(rr[-1], 'C'), atomof(rr[0], 'N')
		hd = atomof(rr[0], '2H') or atomof(rr[0], '3H')
		co = np.asarray(pose.data['Coordinates'], dtype=float)
		d = co[hd] - co[n0] if None not in (nC, n0, hd) else None
		nd = float(np.linalg.norm(d)) if d is not None else 0.0
		dih = []
		for ri in rr if nd >= 1e-9 else []:
			for ang in ('PHI', 'PSI'):
				try:
					if not np.isnan(pose.GetDihedral(ri, ang)):
						dih.append((ri, ang))
				except Exception: pass
		tgt = co[n0] + 1.33 * d / nd if dih else None
		for _ in range(300 if dih else 0):
			co = np.asarray(pose.data['Coordinates'])
			if np.linalg.norm(co[nC] - tgt) < 0.02: break
			for ri, ang in reversed(dih):
				co = np.asarray(pose.data['Coordinates'])
				p, s = ('N', 'CA') if ang == 'PHI' else ('CA', 'C')
				M, O = co[nC], co[atomof(ri, p)]
				u = co[atomof(ri, s)] - O
				nu = np.linalg.norm(u)
				if nu < 1e-9: continue
				u = u / nu
				a = (M - O) - np.dot(M - O, u) * u
				b = (tgt - O) - np.dot(tgt - O, u) * u
				na, nb = np.linalg.norm(a), np.linalg.norm(b)
				if na < 1e-6 or nb < 1e-6: continue
				a, b = a / na, b / nb
				th = math.atan2(float(np.dot(np.cross(a, b), u)),
					float(np.dot(a, b)))
				pose.RotateDihedral(ri,
					pose.GetDihedral(ri, ang) + math.degrees(th), ang)
	if mode != 'head-to-tail':
		i1, i2 = atomof(res1, atom1), atomof(res2, atom2)
		if i1 is None or i2 is None:
			raise Exception('Cyclize: sidechain atoms not found')
		bov, rec = 1.0, [int(res1), int(res2)]
		bonds, bo = pose.data['Bonds'], pose.data['BondOrders']
		if i2 in bonds.get(i1, []): return
		drop = set()
		for s in (i1, i2):
			h = next((j for j in bonds.get(s, []) if atoms[j][1] == 'H'), None)
			if h is not None: drop.add(h)
		keep = [i for i in sorted(atoms) if i not in drop]
		nx = {old: k for k, old in enumerate(keep)}
		pose.data['Coordinates'] = np.asarray(
			pose.data['Coordinates'], dtype=float)[keep]
		pose.data['Atoms'] = {nx[i]: atoms[i] for i in keep}
		pose.data['Bonds'] = {nx[i]: [nx[j] for j in bonds.get(i, [])
			if j in nx] for i in keep}
		pose.data['BondOrders'] = {nx[i]: [o for j, o in zip(
			bonds.get(i, []), bo.get(i, [])) if j in nx] for i in keep}
		for ri in src:
			src[ri][2] = [nx[i] for i in src[ri][2] if i in nx]
			src[ri][3] = [nx[i] for i in src[ri][3] if i in nx]
		i1, i2 = nx[i1], nx[i2]
	else:
		a_n, a_c = atomof(rr[0], 'N'), atomof(rr[-1], 'C')
		drop = {atomof(rr[0], nm) for nm in ('2H', '3H', 'H2', 'H3')}
		co = np.asarray(pose.data['Coordinates'], dtype=float)
		d = float(np.linalg.norm(co[a_c] - co[a_n]))
		if d > 2.0: raise Exception(
			'Cyclise: closure failed, C-N is %.2f A' % d)
		drop |= {atomof(rr[-1], nm)
			for nm in ('OXT', 'OT1', 'OT2', "O''")}
		drop.discard(None)
		keep = [i for i in sorted(atoms) if i not in drop]
		nx = {old: k for k, old in enumerate(keep)}
		bonds, bo = pose.data['Bonds'], pose.data['BondOrders']
		pose.data['Coordinates'] = np.asarray(
			pose.data['Coordinates'], dtype=float)[keep]
		pose.data['Atoms'] = {nx[i]: atoms[i] for i in keep}
		pose.data['Bonds'] = {nx[i]: [nx[j] for j in bonds.get(i, [])
			if j in nx] for i in keep}
		pose.data['BondOrders'] = {nx[i]: [o for j, o in zip(
			bonds.get(i, []), bo.get(i, [])) if j in nx] for i in keep}
		for ri in src:
			src[ri][2] = [nx[i] for i in src[ri][2] if i in nx]
			src[ri][3] = [nx[i] for i in src[ri][3] if i in nx]
		i1, i2, bov = nx[a_c], nx[a_n], 1.5
		rec = [int(rr[-1]), int(rr[0])]
	pose.data['Bonds'].setdefault(i1, []).append(i2)
	pose.data['BondOrders'].setdefault(i1, []).append(bov)
	pose.data['Bonds'].setdefault(i2, []).append(i1)
	pose.data['BondOrders'].setdefault(i2, []).append(bov)
	pose.data.setdefault('Cyclic', []).append(rec)
	pose.CalcCharge()

def Rotamers(index, pose):
	'''
	Set every chi of one residue to the most populated rotamer that the
	Rotamer Library holds for its current backbone phi and psi
	Arguments:
	----------
		index: Residue index into pose.data['Amino Acids']
		pose:  Protein pose with a non-empty residue table
	Returns:
	--------
		Rotates the side chain in place and returns no value. Does
		nothing when the residue has no chi angles, sits at a chain end
		where phi or psi is undefined, or has no library entry.
		D-amino acids are looked up with negated phi and psi and the
		resulting chi values are negated before they are applied
	'''
	info = pose.data.get('Amino Acids', {}).get(index)
	if info is None: return
	c = info[0]
	db = pose.aminoacids.get(c.upper(), {})
	if not (db.get('Chi Angle Atoms') or []): return
	tri = (db.get('Tricode') or [None])[0]
	if not tri: return
	phi, psi = pose.GetDihedral(index, 'PHI'), pose.GetDihedral(index, 'PSI')
	if math.isnan(phi) or math.isnan(psi): return
	flip = c != c.upper()
	n_chi, rows = _rotliblookup(DBLoad().get('Rotamer Library'), tri,
		-phi if flip else phi, -psi if flip else psi)
	if n_chi == 0 or not rows: return
	best = max(rows, key=lambda r: r[1])
	for ci in range(n_chi):
		mu = best[2 + ci]
		pose.RotateDihedral(index, float(-mu if flip else mu),
			'CHI', ci + 1)

def Pack(pose, score=None, n_steps=2000, T_start=10.0, T_end=0.1,
		patience=400, seed=None):
	'''
	Repack side chains by simulated annealing over the rotamer ensemble
	available to each residue at its current backbone phi and psi
	Arguments:
	----------
		pose:     Protein pose carrying an Amino Acids table
		score:    Score function to minimise, Score() when None
		n_steps:  Maximum number of annealing proposals
		T_start:  Initial temperature in score units
		T_end:    Final temperature, reached by geometric cooling
		patience: Stop early after this many consecutive rejections
		seed:     Seed for the random generator, None for unseeded
	Returns:
	--------
		float: Score of the best configuration found, rescored after it
		is restored into the pose
		dict: Log holding 'energies', 'temperatures', 'accepts',
		'best_E', 'steps_run', 'converged' and 'n_residues'
	'''
	if score is None: score = Score()
	if pose.data.get('Amino Acids') is None:
		raise ValueError('Pack requires a protein pose with Amino Acids')
	rng = np.random.default_rng(seed)
	rotlib = DBLoad().get('Rotamer Library')
	candidates = {}
	for r, info in sorted(pose.data['Amino Acids'].items()):
		c = info[0]
		db = pose.aminoacids.get(c.upper(), {})
		tri = (db.get('Tricode') or [None])[0]
		if not (db.get('Chi Angle Atoms') or []) or not tri: continue
		phi = pose.GetDihedral(r, 'PHI')
		psi = pose.GetDihedral(r, 'PSI')
		if math.isnan(phi) or math.isnan(psi): continue
		flip = c != c.upper()
		n_chi, rows = _rotliblookup(rotlib, tri,
			-phi if flip else phi, -psi if flip else psi)
		if n_chi == 0 or not rows: continue
		probs = np.array([max(float(row[1]), 0.0) for row in rows],
			dtype=np.float64)
		if probs.sum() <= 0.0: continue
		mus = np.array([[float(row[2 + ci]) for ci in range(n_chi)]
			for row in rows], dtype=np.float64)
		candidates[r] = (-mus if flip else mus, probs / probs.sum(), n_chi)
	if not candidates:
		E0 = float(score(pose))
		return E0, {'energies': np.array([E0]),
			'temperatures': np.array([T_start]),
			'accepts': np.array([], dtype=bool), 'best_E': E0,
			'steps_run': 0, 'converged': True, 'n_residues': 0}
	res_ids = list(candidates.keys())
	E_curr = float(score(pose))
	E_best = E_curr
	best_state = {q: tuple(pose.GetDihedral(q, 'CHI', chi_type=ci+1)
		for ci in range(candidates[q][2])) for q in res_ids}
	N = max(1, int(n_steps))
	energies = np.empty(N, dtype=np.float64)
	temperatures = np.empty(N, dtype=np.float64)
	accepts = np.empty(N, dtype=bool)
	last_accept = step = 0
	for step in range(N):
		T = T_start * (T_end / T_start) ** (step / max(1, N - 1))
		r = res_ids[int(rng.integers(0, len(res_ids)))]
		mus, probs, n_chi = candidates[r]
		k = int(rng.choice(len(probs), p=probs))
		snap = tuple(pose.GetDihedral(r, 'CHI', chi_type=ci+1)
			for ci in range(n_chi))
		for ci in range(n_chi):
			pose.RotateDihedral(r, float(mus[k, ci]), 'CHI', ci+1)
		E_trial = float(score(pose))
		dE = E_trial - E_curr
		ok = dE <= 0.0 or rng.random() < math.exp(-dE / max(T, 1e-12))
		accepts[step] = ok
		for ci in range(n_chi if not ok else 0):
			pose.RotateDihedral(r, float(snap[ci]), 'CHI', ci+1)
		if ok: E_curr, last_accept = E_trial, step
		if ok and E_curr < E_best:
			E_best = E_curr
			best_state = {q: tuple(
				pose.GetDihedral(q, 'CHI', chi_type=ci+1)
				for ci in range(candidates[q][2])) for q in res_ids}
		energies[step] = E_curr
		temperatures[step] = T
		if step - last_accept >= patience: break
	steps_run = step + 1
	for q, chis in best_state.items():
		for ci in range(candidates[q][2]):
			pose.RotateDihedral(q, float(chis[ci]), 'CHI', ci+1)
	return float(score(pose)), {
		'energies': energies[:steps_run],
		'temperatures': temperatures[:steps_run],
		'accepts': accepts[:steps_run],
		'best_E': float(E_best),
		'steps_run': int(steps_run),
		'converged': bool(steps_run < N),
		'n_residues': len(res_ids)}

def Anneal(pose, ff=None, n_steps=10000, T_start=2000.0, T_end=10.0,
		sigma_small=5.0, sigma_large=30.0, p_large=0.2, p_shear=0.5,
		target_acc=0.30, adapt_window=100, seed=None, box=None):
	'''
	Simulated annealing over backbone torsions, mixing single-torsion and
	shear moves, with the small-move step size adapted to hold a target
	acceptance ratio
	Arguments:
	----------
		pose:         Protein pose carrying an Amino Acids table
		ff:           ForceField to evaluate, created when None
		n_steps:      Total Metropolis steps in the cooling schedule
		T_start:      Starting temperature in Kelvin
		T_end:        Final temperature in Kelvin
		sigma_small:  Initial small-move standard deviation in degrees
		sigma_large:  Large-move standard deviation in degrees, fixed
		p_large:      Probability that a step takes a large move
		p_shear:      Probability that a step attempts a shear move
		target_acc:   Acceptance ratio the small sigma is tuned toward
		adapt_window: Small moves between updates of sigma_small
		seed:         Seed for the random generator, None for unseeded
		box:          None for no PBC, (3,) orthorhombic, (3, 3) triclinic
	Returns:
	--------
		float: Lowest energy seen, whose coordinates are left in the pose
		dict: Log holding 'energies', 'temperatures', 'accepted',
		'move_types' (0 single, 1 shear, 2 no move applied),
		'sigma_history' and 'best_step'
	'''
	if ff is None: ff = ForceField()
	if pose.data.get('Amino Acids') is None:
		raise ValueError('Anneal requires a protein pose with Amino Acids')
	GAIN, SIGMA_MIN, SIGMA_MAX = 0.5, 0.5, 60.0
	NAN, kB = float('nan'), 8.31446262e-3
	rng = np.random.default_rng(seed)
	res_ids = np.array(sorted(pose.data['Amino Acids']), dtype=np.int64)
	T_arr = T_start * (T_end / T_start) ** (
		np.arange(n_steps) / max(n_steps - 1, 1))
	res_arr = res_ids[rng.integers(0, len(res_ids), size=n_steps)]
	kind_arr = np.where(rng.integers(0, 2, size=n_steps) == 0,
		'PHI', 'PSI')
	shear_arr = rng.random(size=n_steps) < p_shear
	large_arr = rng.random(size=n_steps) < p_large
	noise_arr = rng.standard_normal(size=n_steps)
	uni_arr = rng.random(size=n_steps)
	E_curr = float(ff(pose, grad=False, box=box))
	E_best = E_curr
	coords_best = pose.data['Coordinates'].copy()
	energies = np.empty(n_steps, dtype=np.float64)
	accepted = np.zeros(n_steps, dtype=bool)
	move_types = np.full(n_steps, 2, dtype=np.int8)
	sigma_history = [float(sigma_small)]
	small_count, small_acc, best_step = 0, 0, 0
	for s in range(int(n_steps)):
		delta = float(noise_arr[s] * (sigma_large if large_arr[s]
			else sigma_small))
		res, kind = int(res_arr[s]), str(kind_arr[s])
		coords_old = pose.data['Coordinates'].copy()
		psi0 = pose.GetDihedral(res, 'PSI') if shear_arr[s] else NAN
		phi1 = (pose.GetDihedral(res + 1, 'PHI') if shear_arr[s]
			and (res + 1) in pose.data['Amino Acids'] else NAN)
		shear = not (math.isnan(psi0) or math.isnan(phi1))
		if shear:
			pose.RotateDihedral(res, psi0 + delta, 'PSI')
			pose.RotateDihedral(res + 1, phi1 - delta, 'PHI')
		th = NAN if shear else pose.GetDihedral(res, kind)
		if not math.isnan(th):
			pose.RotateDihedral(res, th + delta, kind)
		if not shear and math.isnan(th):
			energies[s] = E_curr
			continue
		move_types[s] = 1 if shear else 0
		E_new = float(ff(pose, grad=False, box=box))
		dE = E_new - E_curr
		RT = kB * float(T_arr[s])
		boltz = math.exp(-dE / RT) if (dE > 0.0 and RT > 0.0) else 1.0
		accept = (dE <= 0.0) or (uni_arr[s] < boltz)
		accepted[s] = accept
		if not accept: pose.data['Coordinates'] = coords_old
		if accept: E_curr = E_new
		if accept and E_curr < E_best:
			E_best, best_step = E_curr, s
			coords_best = pose.data['Coordinates'].copy()
		energies[s] = E_curr
		small_count += 0 if large_arr[s] else 1
		small_acc += 0 if large_arr[s] else int(accept)
		if large_arr[s] or small_count < adapt_window: continue
		sigma_small *= math.exp(
			GAIN * (small_acc / small_count - target_acc))
		sigma_small = max(SIGMA_MIN, min(sigma_small, SIGMA_MAX))
		sigma_history.append(float(sigma_small))
		small_count, small_acc = 0, 0
	pose.data['Coordinates'] = coords_best
	return float(E_best), {
		'energies': energies,
		'temperatures': T_arr,
		'accepted': accepted,
		'move_types': move_types,
		'sigma_history': np.asarray(sigma_history, dtype=np.float64),
		'best_step': int(best_step)}

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
		D_TO_L = {v['Tricode'][1]: v['Tricode'][0]
			for v in DBLoad()['Amino Acids'].values()
			if len(v.get('Tricode') or []) >= 2}
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
				tri = D_TO_L.get(tri, tri)
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
		TT = params.get('TerminalTypes')
		if TQ is None or TT is None:
			raise ValueError(
				"ScoreMatch: ['TerminalCharges'] or ['TerminalTypes'] is "
				"missing from the score parameters. Run "
				"tools.Port('ref15') to install it.")
		D_TO_L = {v['Tricode'][1]: v['Tricode'][0]
			for v in DBLoad()['Amino Acids'].values()
			if len(v.get('Tricode') or []) >= 2}
		for ri in n_term_res:
			info = aas.get(ri)
			if info is None: continue
			tri = info[5] if len(info) >= 6 else None
			tri = D_TO_L.get(tri, tri)
			if tri == 'PRO':
				for ai in info[2] + info[3]:
					ai = int(ai)
					nm = atoms[ai][0]
					if nm in ('1H', '2H', 'H1', 'H2', 'HN', 'HT1', 'HT2'):
						applyatom(ai, TT['PRO']['H'], TQ['PRO']['H'], ros_types, q_arr,
							ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
							is_polar_h, is_H, has_score, atom_types_db)
					elif nm == 'N':
						applyatom(ai, TT['PRO']['N'], TQ['PRO']['N'], ros_types, q_arr,
							ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
							is_polar_h, is_H, has_score, atom_types_db)
				continue
			if tri == 'GLY':
				for ai in info[2] + info[3]:
					ai = int(ai)
					nm = atoms[ai][0]
					if nm == 'N':
						applyatom(ai, TT['GLY']['N'], TQ['GLY']['N'], ros_types, q_arr,
							ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
							is_polar_h, is_H, has_score, atom_types_db)
					elif nm in NTERM_H_NAMES:
						applyatom(ai, TT['GLY']['H'], TQ['GLY']['H'], ros_types, q_arr,
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
					applyatom(ai, TT['generic']['N'], TQ['generic']['N'], ros_types, q_arr,
						ljR, ljW, lkdG, lkLam, lkVol, is_donor, is_accep,
						is_polar_h, is_H, has_score, atom_types_db)
				elif nm in NTERM_H_NAMES:
					applyatom(ai, TT['generic']['H'], TQ['generic']['H'], ros_types, q_arr,
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
				applyatom(c_ai, TT['cterm']['C'], TQ['cterm']['C'], ros_types, q_arr, ljR,
					ljW, lkdG, lkLam, lkVol, is_donor, is_accep, is_polar_h,
					is_H, has_score, atom_types_db)
			if o_ai is not None:
				applyatom(o_ai, TT['cterm']['O'], TQ['cterm']['O'], ros_types, q_arr, ljR,
					ljW, lkdG, lkLam, lkVol, is_donor, is_accep, is_polar_h,
					is_H, has_score, atom_types_db)
			if oxt_ai is not None:
				applyatom(oxt_ai, TT['cterm']['O'], TQ['cterm']['O'], ros_types, q_arr,
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
		D_TO_L = {v['Tricode'][1]: v['Tricode'][0]
			for v in DBLoad()['Amino Acids'].values()
			if len(v.get('Tricode') or []) >= 2}
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
				xover = 4 if is_poly else 3
				if bd < xover: w = 0.0
				elif bd == xover: w = cp_half
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
		ang_sp2 = math.radians(float(LKB['ang_sp2']))
		ang_sp3 = math.radians(float(LKB['ang_sp3']))
		dih_sp2 = tuple(math.radians(float(v)) for v in LKB['dih_sp2'])
		dih_sp3 = tuple(math.radians(float(v)) for v in LKB['dih_sp3'])
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
		used in the rotational-entropy penalty denominator `1 + nrot_w*num_tors`
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
		xover = int(cp[-1])
		if same_res:
			w = np.where(path < xover, 0.0,
				np.where(path == xover, cp_half, 1.0))
		else:
			w = cache['pair_w']
			if use_cp_rep:
				w = np.where(path < xover, 0.0,
					np.where(path == xover, cp_half, 1.0))
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
			No arguments taken
		Returns:
		--------
			dict: donor_map, (tricode, atom name) to donor chemical type
			dict: acceptor_map, the same for acceptor chemical types
			dict: base_map, (tricode, atom name) to the acceptor's base
				atom name
		The hbdon_ and hbacc_ strings are interoperability keys, not
		values. They index the downloaded eval_table, donor_strengths,
		acceptor_strengths and acc_hybridization tables, so they must
		match those keys verbatim. The assignments themselves follow
		standard side-chain chemistry.
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
		acceptor_map[('HIS', 'ND1')] = 'hbacc_IMD'
		base_map[('HIS','ND1')] = 'CG'
		acceptor_map[('HIS_D', 'NE2')] = 'hbacc_IME'
		base_map[('HIS_D','NE2')] = 'CD2'
		acceptor_map[('TYR', 'OH')] = 'hbacc_AHX'; base_map[('TYR','OH')] = 'CZ'
		acceptor_map[('SER', 'OG')] = 'hbacc_HXL'; base_map[('SER','OG')] = 'CB'
		acceptor_map[('THR', 'OG1')] = 'hbacc_HXL'
		base_map[('THR','OG1')] = 'CB'
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
		Score one candidate donor/acceptor pair and record it if accepted,
		using the geometric hydrogen bond potential of O'Meara et al. 2015
		(J. Chem. Theory Comput. 11:609), whose sp2 acceptor term replaces
		the earlier distance-and-angle-only form. The polynomials, fade
		intervals and the sp2/sp3 scalars are all read from the downloaded
		HBond_data tables; only the functional form is expressed here.
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
		if (params.get('HBond_data', {}).get('acc_hybridization', {})
				.get(a['chem']) == 'RING_HYBRID'
				and a.get('B2') is not None):
			B_xyz = 0.5 * (B_xyz + coords[a['B2']])
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
			if str(info[0]).islower():
				tri = (pose.aminoacids.get(str(info[0]).upper(),
					{}).get('Tricode') or [tri])[0]
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
			hybk = (params.get('HBond_data', {})
				.get('acc_hybridization', {}).get(chem))
			if hybk == 'RING_HYBRID':
				for k in bonds.get(int(ai), []):
					k = int(k)
					if k == b_ai: continue
					if atoms.get(k, [None, 'H'])[1] == 'H': continue
					b2_ai = k; break
			elif chem in ('hbacc_HXL', 'hbacc_AHX'):
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

def Minimise(pose, ff=None, max_steps=500, ftol=1.0, dt_fs=0.5,
		dt_max_fs=1.0, step_max=0.2, etol=1e-6, stall_k=10, box=None):
	'''
	Relax pose coordinates with the FIRE2 damped dynamics minimiser
	(Guénolé et al. 2020, Comput Mater Sci 175:109584), guarded so that
	a force field singularity can neither fling atoms apart nor corrupt the
	returned structure
	Arguments:
	----------
		pose:      Protein, DNA, RNA, or Molecule pose to relax in place
		ff:        ForceField to evaluate, created when None
		max_steps: Maximum number of FIRE2 iterations
		ftol:      Convergence threshold on max|force| in kJ/mol/A
		dt_fs:     Initial integrator step in femtoseconds
		dt_max_fs: Upper bound on the adaptive step in femtoseconds
		step_max:  Trust-region cap on per-atom displacement in angstroms
		etol:      Energy-stall tolerance in kJ/mol
		stall_k:   Consecutive stalled steps that stop the run early
		box:       None for no PBC, (3,) orthorhombic, (3, 3) triclinic
	Returns:
	--------
		float: Energy in kJ/mol of the lowest-force frame, which is the
		frame left in the pose
		dict: Log holding 'energies', 'fmax', 'max_step', 'converged'
		and 'n_steps'
	'''
	if ff is None: ff = ForceField()
	N_MIN, F_INC, F_DEC = 5, 1.1, 0.5
	A_START, F_ALPHA, AKMA_FS = 0.1, 0.99, 23.91888086
	atoms = pose.data['Atoms']
	m = np.array([pose.masses[atoms[i][1]] for i in sorted(atoms)],
		dtype=np.float64)[:, None]
	v = np.zeros_like(pose.data['Coordinates'], dtype=np.float64)
	dt = float(dt_fs) / AKMA_FS
	dt_max = float(dt_max_fs) / AKMA_FS
	dt_min = dt * 1e-3
	alpha, n_pos = float(A_START), 0
	energies, fmaxes, max_steps_log = [], [], []
	E, F = ff(pose, grad=True, box=box)
	E = float(E)
	best_fmax = float(np.max(np.abs(F)))
	best_coords = pose.data['Coordinates'].copy()
	converged, steps_done, stall = False, 0, 0
	for step in range(int(max_steps)):
		fmax = float(np.max(np.abs(F)))
		energies.append(E); fmaxes.append(fmax)
		steps_done = step + 1
		if np.isfinite(fmax) and fmax < best_fmax:
			best_fmax = fmax
			best_coords = pose.data['Coordinates'].copy()
		if fmax < ftol or stall >= stall_k:
			converged = True
			break
		if (not np.isfinite(fmax)) or (fmax > 1e4
				and fmax > 1e3 * best_fmax): break
		P = float(np.sum(F * v))
		if P <= 0.0:
			v = np.zeros_like(v)
			dt = max(dt * F_DEC, dt_min)
			alpha, n_pos = A_START, 0
		if P > 0.0:
			fn = float(np.linalg.norm(F))
			mix = (alpha * float(np.linalg.norm(v)) / fn
				if fn > 1e-12 else 0.0)
			v = (1.0 - alpha) * v + mix * F
			n_pos += 1
		if P > 0.0 and n_pos > N_MIN:
			dt = min(dt * F_INC, dt_max)
			alpha *= F_ALPHA
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
	pose.data['Coordinates'] = best_coords
	E, F = ff(pose, grad=True, box=box)
	return float(E), {
		'energies': np.asarray(energies, dtype=np.float64),
		'fmax': np.asarray(fmaxes, dtype=np.float64),
		'max_step': np.asarray(max_steps_log, dtype=np.float64),
		'converged': bool(converged),
		'n_steps': int(steps_done)}

def MolecularDynamics(pose, ff=None, n_steps=1000, dt_fs=2.0, T=300.0,
		thermostat='nve', friction_ps=1.0, constraints='hbonds',
		shake_tol=1e-8, shake_max=100, seed=None,
		trajectory_every=0, box=None):
	'''
	Molecular dynamics by velocity Verlet in the NVE ensemble or BAOAB
	Langevin in NVT, with bond lengths to hydrogen held by SHAKE and RATTLE
	(BAOAB integrator, Leimkuhler & Matthews 2013, Appl Math Res
	Express 2013:34)
	Arguments:
	----------
		pose:             All-atom pose, protein, DNA, RNA, or Molecule
		ff:               ForceField to evaluate, created when None
		n_steps:          Number of integration steps
		dt_fs:            Integration step in femtoseconds
		T:                Temperature in Kelvin, initial and bath
		thermostat:       'nve' or 'langevin'
		friction_ps:      Langevin friction in inverse picoseconds
		constraints:      'hbonds' constrains every X-H bond, or 'none'
		shake_tol:        Relative tolerance on the constraint residual
		shake_max:        Maximum SHAKE or RATTLE iterations per call
		seed:             Seed for the random generator, None for unseeded
		trajectory_every: Snapshot stride, 0 stores no snapshots
		box:              None for no PBC, (3,) ortho, (3, 3) triclinic
	Returns:
	--------
		float: Final potential energy in kJ/mol
		dict: Log holding 'energies', 'kinetic', 'temperatures',
		'frames', 'n_constraints' and 'dof'
	'''
	if ff is None: ff = ForceField()
	if thermostat not in ('nve', 'langevin'):
		raise ValueError("thermostat must be 'nve' or 'langevin'")
	if constraints not in ('hbonds', 'none'):
		raise ValueError("constraints must be 'hbonds' or 'none'")
	atoms = pose.data['Atoms']
	sorted_ids = sorted(atoms)
	is_h = np.array([atoms[i][1] == 'H' for i in sorted_ids], dtype=bool)
	polymer = (pose.data.get('Amino Acids')
		or pose.data.get('Nucleotides'))
	if polymer and not is_h.any():
		raise ValueError(
			'MolecularDynamics needs an all-atom pose, but this one has '
			'no hydrogens. A structure imported from a crystallographic '
			'PDB has none, which leaves the force field charges '
			'unbalanced and the dynamics will diverge. Call '
			'pose.ReBuild() after Import() to add them.')
	rng = np.random.default_rng(seed)
	m = np.array([pose.masses[atoms[i][1]] for i in sorted_ids],
		dtype=np.float64)
	n = len(m)
	m_col = m[:, None]
	inv_m = 1.0 / m
	inv_m_col = inv_m[:, None]
	AKMA_FS, kB = 23.91888086, 8.31446262e-3
	dt = float(dt_fs) / AKMA_FS
	gamma = float(friction_ps) * AKMA_FS / 1000.0
	c1 = math.exp(-gamma * dt)
	c2 = np.sqrt((1.0 - c1 * c1) * kB * float(T) / m)[:, None]
	if ff._cache is None or ff._cache_hash != ff._topologyhash(pose):
		ff._prepare(pose)
	cache = ff._cache
	cmask = (is_h[cache['pairs'][:, 0]] | is_h[cache['pairs'][:, 1]]
		if constraints == 'hbonds' and len(cache['pairs'])
		else np.zeros(len(cache['pairs']), dtype=bool))
	con = cache['pairs'][cmask]
	r0 = cache['bond_r0'][cmask]
	K = len(con)
	i_c, j_c = con[:, 0], con[:, 1]
	r0sq = r0 * r0
	inv_red = inv_m[i_c] + inv_m[j_c] if K else np.empty(0)
	r0sq_max = float(r0sq.max()) if K else 1.0
	def shake(x_new, x_old, vel, dt_eff):
		'''
		Project positions back onto the bond-length constraints
		Arguments:
		----------
			x_new:  Coordinates after the drift, corrected in place
			x_old:  Coordinates before the drift, giving the bond axes
			vel:    Velocities, corrected in place by the same impulse
			dt_eff: Length of the drift the projection is undoing
		Returns:
		--------
			Corrects x_new and vel in place and returns no value
		'''
		r_old = x_old[i_c] - x_old[j_c]
		for _ in range(int(shake_max) if K else 0):
			r = x_new[i_c] - x_new[j_c]
			d2 = np.einsum('ij,ij->i', r, r)
			sigma = d2 - r0sq
			if float(np.max(np.abs(sigma))) < shake_tol * r0sq_max: return
			rdot = np.einsum('ij,ij->i', r, r_old)
			delta = (sigma / (2.0 * inv_red * rdot))[:, None] * r_old
			np.add.at(x_new, i_c, -delta * inv_m_col[i_c])
			np.add.at(x_new, j_c, delta * inv_m_col[j_c])
			np.add.at(vel, i_c, -(delta / dt_eff) * inv_m_col[i_c])
			np.add.at(vel, j_c, (delta / dt_eff) * inv_m_col[j_c])
	def rattle(x, vel):
		'''
		Project velocities so that no constrained bond changes length
		Arguments:
		----------
			x:   Current coordinates, giving the bond axes
			vel: Velocities, corrected in place
		Returns:
		--------
			Corrects vel in place and returns no value
		'''
		for _ in range(int(shake_max) if K else 0):
			r = x[i_c] - x[j_c]
			rv = np.einsum('ij,ij->i', r, vel[i_c] - vel[j_c])
			d2 = np.einsum('ij,ij->i', r, r)
			if float(np.max(np.abs(rv))) < shake_tol * r0sq_max: return
			delta_v = (rv / (d2 * inv_red))[:, None] * r
			np.add.at(vel, i_c, -delta_v * inv_m_col[i_c])
			np.add.at(vel, j_c, delta_v * inv_m_col[j_c])
	v = rng.standard_normal(size=(n, 3)) * np.sqrt(
		kB * float(T) / m)[:, None]
	v -= ((m_col * v).sum(axis=0) / m.sum())[None, :]
	rattle(pose.data['Coordinates'], v)
	E, F = ff(pose, grad=True, box=box)
	dof = max(3 * n - K - 3, 1)
	energies = np.empty(int(n_steps), dtype=np.float64)
	kinetics = np.empty(int(n_steps), dtype=np.float64)
	temps = np.empty(int(n_steps), dtype=np.float64)
	frames = []
	for step in range(int(n_steps)):
		if thermostat == 'langevin':
			v += 0.5 * dt * F / m_col
			x_old = pose.data['Coordinates'].copy()
			pose.data['Coordinates'] = x_old + 0.5 * dt * v
			shake(pose.data['Coordinates'], x_old, v, 0.5 * dt)
			v = c1 * v + c2 * rng.standard_normal(size=(n, 3))
			rattle(pose.data['Coordinates'], v)
			x_old = pose.data['Coordinates'].copy()
			pose.data['Coordinates'] = x_old + 0.5 * dt * v
			shake(pose.data['Coordinates'], x_old, v, 0.5 * dt)
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
	return float(E), {
		'energies': energies,
		'kinetic': kinetics,
		'temperatures': temps,
		'frames': frames,
		'n_constraints': int(K),
		'dof': int(dof)}

def Port(name='openff', accept_rosetta_license=False):
	'''
	Download one force field or score function from its pinned upstream
	commit and write the parsed parameters into database.json
	Arguments:
	----------
		name: Which set to port, matched case-insensitively, one of
			'OpenFF', 'ff19SB', 'CHARMM36', 'AutoDock Vina' or 'REF15'
		accept_rosetta_license: Required True for 'REF15'. Rosetta is
			distributed under the Rosetta Software Non-Commercial
			License Agreement, not an open-source licence. Setting
			this asserts you qualify as a Non-Commercial User
	Returns:
	--------
		bool: True when the set was written, False otherwise, in which
		case the reason has already been printed
	'''
	_KCAL_TO_KJ = 4.184
	_BINS_PER_A2 = 20
	_VINA_BASE = ('https://raw.githubusercontent.com/'
		'ccsb-scripps/AutoDock-Vina/'
		'3c65c0b3e6c2c1d183f6a175ecb65e3c5ba91645/src/lib/')
	_REF15_REPO = ('https://raw.githubusercontent.com/'
		'RosettaCommons/rosetta/'
		'5e498f1409c68ade56c8ce5842bf79e1b02e8db4/database/')
	_NAGL_DTYPES = {'FloatStorage': np.float32,
		'DoubleStorage': np.float64, 'HalfStorage': np.float16,
		'LongStorage': np.int64, 'IntStorage': np.int32,
		'ByteStorage': np.uint8, 'BoolStorage': np.bool_}
	_ETABLE_ATOM_TYPES = [
		'CNH2', 'COO', 'CH0', 'CH1', 'CH2', 'CH3', 'aroC', 'Ntrp',
		'Nhis', 'NtrR', 'NH2O', 'Nlys', 'Narg', 'Npro', 'OH', 'ONH2',
		'OOC', 'Oaro', 'S', 'SH1', 'Nbb', 'CAbb', 'CObb', 'OCbb',
		'Hpol', 'Hapo', 'Haro', 'HNbb', 'HOH',
		'Phos', 'F', 'HS']
	def _download(url):
		'''
		Fetch the text of a pinned GitHub raw URL
		Arguments:
		----------
			url: str - a raw.githubusercontent.com URL on a fixed commit
		Returns:
		--------
			str: the decoded file contents
		'''
		try:
			with urllib.request.urlopen(url, timeout=120) as resp:
				return resp.read().decode('utf-8')
		except Exception as err:
			raise RuntimeError(f'port: could not download {url}: {err}')
	def _cidof(rec, i):
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
	def _reduceunits(text, units):
		'''
		Reduce a unit expression to a number, unit powers, dimensions
		Arguments:
		----------
			text: Unit expression such as 'kilocalorie/mole * angstrom**-2'
			units: Table mapping a unit name to its scale and dimensions
		Returns:
		--------
			tuple: the numeric scale, a {unit: power} map, and a
			{dimension: power} map with the zero powers dropped
		'''
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
	def _qval(qstr, target):
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
		nq, pq, dq = _reduceunits(qstr, units)
		nt, pt, dt = _reduceunits(target, units)
		if dq != dt:
			raise ValueError(
				f'port: cannot convert {qstr!r} to {target!r} '
				f'(dimension mismatch)')
		value = nq / nt
		for nm in set(pq) | set(pt):
			ex = pq.get(nm, 0) - pt.get(nm, 0)
			if ex: value *= units[nm][0] ** ex
		return value
	def _converttorsions(section):
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
					'phi_0': _qval(a['phase%d' % i], 'degree'),
					'K_phi': _qval(a['k%d' % i], 'kilojoule_per_mole'),
					'idivf': float(idivf) if idivf is not None else 1.0})
				i += 1
			out[a['smirks']] = {'id': a.get('id'), 'components': comps}
		return out
	def _patchside(nm):
		'''
		Split a two-residue patch atom name into its side and name
		Arguments:
		----------
			nm: Atom name, optionally prefixed as '1:' or '2:'
		Returns:
		--------
			str: the side, '1', '2', or None when the name carries no prefix
			str: the atom name with any prefix removed
		'''
		if len(nm) > 2 and nm[1] == ':': return nm[0], nm[2:]
		return None, nm
	def _keepside(nm):
		'''
		Keep an atom name only when it belongs to residue one
		Arguments:
		----------
			nm: Atom name from a patch record, possibly side-prefixed
		Returns:
		--------
			str: the bare atom name, or None when it belongs to residue two
		'''
		s, real = _patchside(nm)
		return real if s in (None, '1') else None
	def _applypatch(base, pname, patch):
		'''
		Apply one CHARMM patch to a residue's atoms and bonds
		Arguments:
		----------
			base: The (atoms, bonds) pair of the residue being patched
			pname: Name of the patch to apply, a key of patch
			patch: Table of every parsed patch, keyed by patch name
		Returns:
		--------
			tuple: a new (atoms, bonds) pair with the patch applied, leaving
			the input untouched
		'''
		atoms = {k: list(v) for k, v in base[0].items()}
		bonds = list(base[1])
		d = patch[pname]
		for nm, v in d['change'].items():
			real = _keepside(nm)
			if real is not None and real in atoms:
				atoms[real] = list(v)
		for nm, v in d['add'].items():
			real = _keepside(nm)
			if real is not None: atoms[real] = list(v)
		rem = {_keepside(nm) for nm in d['remove']} - {None}
		atoms = {k: v for k, v in atoms.items() if k not in rem}
		bonds = [b for b in bonds
			if b[0] not in rem and b[1] not in rem]
		rmb = set()
		for x, y in d['rmbond']:
			rx, ry = _keepside(x), _keepside(y)
			if rx is not None and ry is not None:
				rmb.add(frozenset((rx, ry)))
		bonds = [b for b in bonds if frozenset(b) not in rmb]
		for x, y in d['addbond']:
			rx, ry = _keepside(x), _keepside(y)
			if rx is not None and ry is not None:
				bonds.append((rx, ry))
		return (atoms, bonds)
	def _charmmtypes(root):
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
		npatch = {'GLY': 'GLYP', 'PRO': 'PROP'}
		protein = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
			'GLY', 'HSD', 'HSE', 'HSP', 'ILE', 'LEU', 'LYS', 'MET',
			'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
		variants = {}
		for rn in protein:
			if rn not in res: continue
			variants[rn]       = res[rn]
			variants['N' + rn] = _applypatch(res[rn],
				npatch.get(rn, 'NTER'), patch)
			variants['C' + rn] = _applypatch(res[rn], 'CTER', patch)
		if 'CYS' in res:
			cyx = _applypatch(res['CYS'], 'DISU', patch)
			variants['CYX']  = cyx
			variants['NCYX'] = _applypatch(cyx, 'NTER', patch)
			variants['CCYX'] = _applypatch(cyx, 'CTER', patch)
		templates = {}
		for vn, (atoms, bonds) in variants.items():
			templates[vn] = {
				'atoms': [[nm, at_elem.get(cls, ''), cls, chg]
					for nm, (cls, chg) in atoms.items()],
				'bonds': [[a, b] for a, b in bonds]}
		return templates
	class _NaglStub(dict):
		'''
		Stand-in for any class the checkpoint pickles
		Arguments:
		----------
			No arguments taken
		Returns:
		--------
			dict: instances absorb whatever state the pickle carries
		'''
		def __init__(self, *a, **k):
			'''
			Accept any constructor arguments and ignore them
			Arguments:
			----------
				a: Positional arguments from the pickle, unused
				k: Keyword arguments from the pickle, unused
			Returns:
			--------
				Initialises an empty dict and returns no value
			'''
			dict.__init__(self)
		def __setstate__(self, state):
			'''
			Absorb the pickled state as dictionary entries
			Arguments:
			----------
				state: State object the pickle supplies
			Returns:
			--------
				Updates the instance in place and returns no value
			'''
			if isinstance(state, dict): self.update(state)
	def _naglrebuild(root, zf, store, offset, size, stride, *rest):
		'''
		Reconstruct one tensor from its storage as a NumPy array
		Arguments:
		----------
			root: Name of the top-level directory inside the archive
			zf: Open ZipFile holding the checkpoint
			store: Storage record, its file name and dtype
			offset: Element offset of this tensor within the storage
			size: Shape of the tensor
			stride: Stride of the tensor, accepted and unused
			rest: Any further arguments the pickle supplies, unused
		Returns:
		--------
			np.ndarray: the tensor read from the archive and reshaped
		'''
		arr = np.frombuffer(zf.read('%s/data/%s'
			% (root, store[0])), dtype=store[1])
		size = tuple(size)
		n = int(np.prod(size)) if size else arr.size
		return arr[offset:offset + n].reshape(size)
	class _NaglReader(pickle.Unpickler):
		'''
		Unpickler that yields NumPy arrays, never torch objects
		Arguments:
		----------
			No arguments taken
		Returns:
		--------
			pickle.Unpickler: a reader whose load() returns plain containers
		'''
		def __init__(self, fh, root, zf):
			'''
			Record the archive so tensors can be rebuilt from it
			Arguments:
			----------
				fh: File object holding the pickle stream
				root: Name of the top-level directory inside the archive
				zf: Open ZipFile holding the checkpoint
			Returns:
			--------
				Initialises the unpickler and returns no value
			'''
			super().__init__(fh)
			self.root, self.zf = root, zf
		def find_class(self, mod, name):
			'''
			Resolve a pickled global to a safe local substitute
			Arguments:
			----------
				mod: Module name recorded in the pickle
				name: Attribute name recorded in the pickle
			Returns:
			--------
				object: the tensor rebuilder, a NumPy dtype, the real class when
				it is importable, or a stub
			'''
			if name == '_rebuild_tensor_v2':
				return functools.partial(_naglrebuild, self.root, self.zf)
			if name in _NAGL_DTYPES: return _NAGL_DTYPES[name]
			try: return super().find_class(mod, name)
			except Exception: return _NaglStub
		def persistent_load(self, pid):
			'''
			Resolve a persistent id to a storage key and dtype
			Arguments:
			----------
				pid: Persistent id tuple the pickle supplies
			Returns:
			--------
				tuple: the storage file name and the NumPy dtype to read it as
			'''
			dt = pid[1] if pid[1] in _NAGL_DTYPES.values() else np.float32
			return (pid[2], dt)
	def _naglcollect(node, name, found, seen):
		'''
		Walk the unpickled tree and index every array by name
		Arguments:
		----------
			node: Current node of the unpickled structure
			name: Parameter name accumulated from the enclosing keys
			found: Output map from parameter name to array, filled in place
			seen: Set of visited object ids, guarding against cycles
		Returns:
		--------
			Fills found in place and returns no value
		'''
		if id(node) in seen: return
		seen.add(id(node))
		if isinstance(node, dict):
			for k, v in node.items(): _naglcollect(v, str(k), found, seen)
		elif isinstance(node, (list, tuple)):
			for v in node: _naglcollect(v, name, found, seen)
		elif isinstance(node, np.ndarray): found[name] = node
	def _naglweights(url):
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
		try:
			with urllib.request.urlopen(url, timeout=300) as resp:
				blob = resp.read()
		except Exception as err:
			raise RuntimeError(f'port: could not download {url}: {err}')
		zf = zipfile.ZipFile(io.BytesIO(blob))
		root = zf.namelist()[0].split('/')[0]
		obj = _NaglReader(io.BytesIO(
			zf.read('%s/data.pkl' % root)), root, zf).load()
		seen, found = set(), {}
		_naglcollect(obj, '', found, seen)
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
	def _vinafetch(name):
		'''
		Download one upstream source file as a UTF-8 string
		Arguments:
		----------
			name: str - file name under src/lib/ (e.g. 'vina.h')
		Returns:
		--------
			str: the file contents
		'''
		url = _VINA_BASE + name
		with urllib.request.urlopen(url, timeout=120) as r:
			return r.read().decode('utf-8')
	def _vinafirst(pat, sf):
		'''
		Return the float groups of the first match of a pattern
		Arguments:
		----------
			pat: Regular expression carrying one or more capture groups
			sf: Source text to search
		Returns:
		--------
			tuple: every capture group of the first match, as floats
		'''
		mm = re.search(pat, sf)
		if mm is None:
			raise Exception('Vina: missing pattern '+pat)
		return tuple(float(g) for g in mm.groups())
	def _vinagrab(name, ac):
		'''
		Read the XS type names named by one xs_is_ predicate
		Arguments:
		----------
			name: Predicate name, such as 'xs_is_hydrophobic'
			ac: Text of atom_constants.h
		Returns:
		--------
			set: the XS type names, with the XS_TYPE_ prefix removed
		'''
		body = re.search(
			r'inline\s+bool\s+' + re.escape(name)
				+ r'\s*\([^)]*\)\s*\{([^}]*)\}',
			ac, re.S)
		if body is None:
			raise Exception('Vina: predicate '+name+' not found')
		return set(re.findall(r'XS_TYPE_([A-Za-z0-9_]+)', body.group(1)))
	def _csvrows(text):
		'''
		Iterate the rows of a CSV text, ignoring trailing blanks
		Arguments:
		----------
			text: Raw CSV text
		Returns:
		--------
			generator: yields one list of fields per non-empty row
		'''
		for line in text.splitlines():
			s = line.rstrip()
			if not s or s.startswith('#'): continue
			yield s.split(',')
	def _nametable(text, name_col=1):
		'''
		Index the rows of a CSV by the name in one column
		Arguments:
		----------
			text: Raw CSV text whose first column is a row index
			name_col: Zero-based column holding the name, 1 by default
		Returns:
		--------
			dict: map from row index to the name in that column
		'''
		out = []
		for r in _csvrows(text):
			if not r: continue
			out.append(r[name_col] if len(r) > name_col else '')
		return out
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
		polys = {}
		for r in _csvrows(raw.get('HBPoly1D.csv', '')):
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
		fades = {}
		for r in _csvrows(raw.get('HBFadeIntervals.csv', '')):
			if len(r) < 7: continue
			try:
				fid = int(r[0])
				min1 = float(r[3]); min2 = float(r[4])
				max1 = float(r[5]); max2 = float(r[6])
			except ValueError: continue
			fades[r[1]] = {'id': fid, 'kind': r[2], 'min1': min1,
				'min2': min2, 'max1': max1, 'max2': max2}
		eval_table = []
		for r in _csvrows(raw.get('HBEval.csv', '')):
			if len(r) < 16: continue
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
		don_str = {}
		for r in _csvrows(raw.get('DonStrength.csv', '')):
			if len(r) < 2: continue
			try: don_str[r[0]] = float(r[1])
			except ValueError: pass
		acc_str = {}
		for r in _csvrows(raw.get('AccStrength.csv', '')):
			if len(r) < 2: continue
			try: acc_str[r[0]] = float(r[1])
			except ValueError: pass
		acc_hyb = {}
		for r in _csvrows(raw.get('HBAccHybridization.csv', '')):
			if len(r) < 2: continue
			acc_hyb[r[0]] = r[1]
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
		Second derivatives at the two endpoints of the cubic Hermite
		interpolant through (x0, y0) and (x1, y1) with prescribed first
		derivatives dy0 at x0 and dy1 at x1
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
		y2_lo = (2.0 / h) * (3.0 * dy - 2.0 * dy0 - dy1)
		y2_hi = (2.0 / h) * (dy0 - 3.0 * dy + 2.0 * dy1)
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
	def _binofdis(ii):
		'''
		Rosetta distance-bin index for a tenth-angstrom step
		Arguments:
		----------
			ii: Distance step, in tenths of an angstrom
		Returns:
		--------
			int: index of the bin holding that squared distance
		'''
		d = ii / 10.0
		return int(d * d * _BINS_PER_A2 + 1)
	def _lksolvderiv(at_self, at_other, d, LJ_R, lk_inv_lambda2,
			lk_coeff):
		'''
		Distance derivative of one Lazaridis-Karplus solvation term
		Arguments:
		----------
			at_self: Index of the atom being desolvated
			at_other: Index of the atom doing the desolvating
			d: Separation in angstroms
			LJ_R: Lennard-Jones radius per atom type
			lk_inv_lambda2: Inverse squared correlation length per atom type
			lk_coeff: Solvation prefactor for each ordered type pair
		Returns:
		--------
			float: the derivative of the solvation energy with respect to d
		'''
		inv_d = 1.0 / d
		inv_d2 = inv_d * inv_d
		dr = d - LJ_R[at_self]
		x = (dr * dr) * lk_inv_lambda2[at_self]
		s = math.exp(-x) * lk_coeff[at_self][at_other] * inv_d2
		ds = -2.0 * s * (
			(d - LJ_R[at_self]) * lk_inv_lambda2[at_self] + inv_d)
		return ds
	def _etableonepair(at1, at2, self_idx, other_idx, sigma_pair,
			LJ_R, LJ_W, LK_DG, LK_L, LK_V, lj_r6, lj_r12, lj_si, lj_ss,
			lk_coeff, far_lo, max_dis, bpa2, sol_lo, sol_hi, sol_cap,
			carb_dis, lk_d2s, ljrep_d2):
		'''
		Analytic energy-table fields for one atom pair, following the
		published definition of the Rosetta all-atom energy function
		(Alford et al. 2017, J. Chem. Theory Comput. 13:3031, eqs 2-5
		and 11-13) with Lazaridis-Karplus implicit solvation (Lazaridis
		& Karplus 1999, Proteins 35:133)
		Arguments:
		----------
			at1, at2:        Atom type indices of the pair
			self_idx:        Index of the atom being desolvated
			other_idx:       Index of the atom doing the desolvating
			sigma_pair:      Summed atomic radii of the pair
			LJ_R, LJ_W:      Per-type radius and well depth
			LK_DG, LK_L, LK_V: Per-type solvation free energy, lambda
			                 and volume
			lj_r6, lj_r12:   Per-pair r^-6 and r^-12 coefficients
			lj_si, lj_ss:    Per-pair linear-ramp intercept and slope
			lk_coeff:        Per-pair solvation prefactor
			far_lo, max_dis: Start and end of the transition to zero
			sol_c0, sol_c1:  Solvation smoothing window either side of
			                 sigma
			ljrep_d2:        Squared distance below which the repulsive
			                 term is linear
		Returns:
		--------
			dict: the analytic fields for this pair, in the schema
			ScoreMatch reads back out of ['EtablePairParams']['pairs']
		'''
		sig = sigma_pair
		sig_lk = lk_d2s * sig
		r12 = lj_r12[at1][at2]
		r6 = lj_r6[at1][at2]
		def _lj(d):
			d6 = d ** 6
			return r12 / (d6 * d6) + r6 / d6
		def _dlj(d):
			d6 = d ** 6
			return -12.0 * r12 / (d6 * d6 * d) - 6.0 * r6 / (d6 * d)
		ylo = _lj(far_lo)
		y2lo, y2hi = _splineddy2(far_lo, ylo, _dlj(far_lo),
			max_dis, 0.0, 0.0)
		ljatr_cp = _cubicfromspline(far_lo, max_dis, ylo, 0.0,
			y2lo, y2hi)
		CARBS = ('CH1', 'CH2', 'CH3', 'aroC')
		iscarb = (_ETABLE_ATOM_TYPES[at1] in CARBS
			and _ETABLE_ATOM_TYPES[at2] in CARBS)
		ibin = int(carb_dis * carb_dis * bpa2 + 1.0)
		flat_d = math.sqrt((ibin - 1) / bpa2) if iscarb else sig_lk
		swtch = (ibin + 1) if iscarb \
			else int(sig_lk * sig_lk * bpa2 + 1.0) + 1
		sol_lo_d = math.sqrt((max(1, swtch - sol_lo) - 1) / bpa2)
		sol_hi_d = math.sqrt((min(swtch + sol_hi, sol_cap) - 1) / bpa2)
		def _fdslope(fn, x, back):
			y = math.sqrt(x * x - 1.0 / bpa2) if back \
				else math.sqrt(x * x + 1.0 / bpa2)
			return (fn(x) - fn(y)) / (x - y)
		def _solv(a_self, a_other):
			'''
			Solvation curve pieces for one direction of the pair
			Arguments:
			----------
				a_self:  Atom type index being desolvated
				a_other: Atom type index doing the desolvating
			Returns:
			--------
				tuple: the flat close value, the close cubic, the far
				cubic and the pair prefactor
			'''
			lam = LK_L[a_self]
			pre = lk_coeff[a_self][a_other]
			def f(d):
				x = (d - LJ_R[a_self]) / lam
				return pre * math.exp(-x * x) / (d * d)
			def df(d):
				x = (d - LJ_R[a_self]) / lam
				g = math.exp(-x * x)
				return pre * g * (-2.0 * x / (lam * d * d)
					- 2.0 / (d * d * d))
			lo, hi = sol_lo_d, sol_hi_d
			flat = f(flat_d)
			a2, b2 = _splineddy2(lo, flat, 0.0, hi, f(hi), df(hi))
			ccp = _cubicfromspline(lo, hi, flat, f(hi), a2, b2)
			a3, b3 = _splineddy2(far_lo, f(far_lo), df(far_lo),
				max_dis, 0.0, 0.0)
			fcp = _cubicfromspline(far_lo, max_dis, f(far_lo), 0.0,
				a3, b3)
			a2f, b2f = _splineddy2(lo, flat, 0.0, hi, f(hi),
				_fdslope(f, hi, False))
			ccpf = _cubicfromspline(lo, hi, flat, f(hi), a2f, b2f)
			a3f, b3f = _splineddy2(far_lo, f(far_lo),
				_fdslope(f, far_lo, True), max_dis, 0.0, 0.0)
			fcpf = _cubicfromspline(far_lo, max_dis, f(far_lo), 0.0,
				a3f, b3f)
			return flat, ccp, fcp, lam, ccpf, fcpf
		f_s, cp_s, fp_s, lam_s, cpf_s, fpf_s = _solv(self_idx, other_idx)
		f_o, cp_o, fp_o, _, cpf_o, fpf_o = _solv(other_idx, self_idx)
		return {
			'close_start': sol_lo_d,
			'close_end':   sol_hi_d,
			'close_flat':  f_s,
			'close_poly':  list(cp_s),
			'far_poly':    list(fp_s),
			'lk_coeff':    lk_coeff[self_idx][other_idx],
			'lambda_self': lam_s,
			'R_self':      LJ_R[self_idx],
			'final_weight': 1.0,
			'close_flat_comb':  f_s + f_o,
			'close_poly_comb':  [x + y for x, y in zip(cpf_s, cpf_o)],
			'far_poly_comb':    [x + y for x, y in zip(fpf_s, fpf_o)],
			'lj_minimum':         sig,
			'lj_r12_coeff':       r12,
			'lj_r6_coeff':        r6,
			'lj_switch_intercept': lj_si[at1][at2],
			'lj_switch_slope':    lj_ss[at1][at2],
			'lj_val_at_minimum':  -math.sqrt(LJ_W[at1] * LJ_W[at2]),
			'lj_min_dis2sigma_value': 0.0,
			'ljatr_cubic_poly':      list(ljatr_cp),
			'ljatr_cubic_poly_xhi':  max_dis,
			'ljatr_cubic_poly_xlo':  far_lo,
			'ljatr_final_weight':    1.0,
			'ljrep_linear_ramp_d2_cutoff': ljrep_d2,
			'ljrep_from_negcrossing': False,
			'hydrogen_interaction':   False,
			'ljrep_xr_xlo':    0.0,
			'ljrep_xr_xhi':    0.0,
			'ljrep_xr_slope':  0.0,
			'ljrep_xr_extrapolated_slope': 0.0,
			'ljrep_xr_ylo':    0.0}
	def _etableparams(atom_types, etopt):
		'''
		Analytic pair parameters for the van der Waals and solvation
		terms, following the published definition of the Rosetta all-atom
		energy function (Alford et al. 2017, J. Chem. Theory Comput.
		13:3031, eqs 2-5 and 11-13) with Lazaridis-Karplus implicit
		solvation (Lazaridis & Karplus 1999, Proteins 35:133)
		Algorithm:
		1) Pair sigma is the sum of the two atomic radii, epsilon the
		   geometric mean of the two well depths.
		2) The repulsive term is linear below 0.6 sigma, fitted for value
		   and slope continuity, Lennard-Jones between 0.6 sigma and
		   sigma, and zero beyond.
		3) The attractive term is constant at -epsilon below sigma,
		   Lennard-Jones out to far_lo, then a cubic matched in value and
		   slope that reaches zero with zero slope at max_dis.
		4) Solvation is a Gaussian in the distance from sigma, held
		   constant below sigma - c0, joined by a cubic across
		   sigma - c0 to sigma + c1, and taken to zero by a second cubic
		   between far_lo and max_dis.
		Arguments:
		----------
			atom_types: Per atom-name dict parsed from atom_properties,
				carrying LJ_RADIUS, LJ_WDEPTH, LK_DGFREE, LK_LAMBDA and
				LK_VOLUME
			etopt: Cutoffs and smoothing window widths
		Returns:
		--------
			dict: stored at ['Score Parameters']['REF15']
			['EtablePairParams'] as {atom_types, n_types, pairs}
		'''
		MAX_DIS         = etopt['MAX_DIS']
		LJ_SWITCH_D2S   = etopt['LJ_SWITCH_D2S']
		NTYPES          = len(_ETABLE_ATOM_TYPES)
		LJ_R = [atom_types[n]['LJ_RADIUS'] for n in _ETABLE_ATOM_TYPES]
		LJ_W = [atom_types[n]['LJ_WDEPTH'] for n in _ETABLE_ATOM_TYPES]
		LK_DG = [atom_types[n]['LK_DGFREE'] for n in _ETABLE_ATOM_TYPES]
		LK_L  = [atom_types[n]['LK_LAMBDA'] for n in _ETABLE_ATOM_TYPES]
		LK_V  = [atom_types[n]['LK_VOLUME'] for n in _ETABLE_ATOM_TYPES]
		LJ_S2D = 1.0 / LJ_SWITCH_D2S
		LJ_V2W = LJ_S2D**12 - 2.0 * LJ_S2D**6
		LJ_S2W = -12.0 * (LJ_S2D**13 - LJ_S2D**7)
		sigma = [[max(LJ_R[i] + LJ_R[j], 1e-9) for j in range(NTYPES)]
			for i in range(NTYPES)]
		ACC = [bool(atom_types[n].get('acceptor')) for n in _ETABLE_ATOM_TYPES]
		DON = [bool(atom_types[n].get('donor')) for n in _ETABLE_ATOM_TYPES]
		POL = [bool(atom_types[n].get('polar_h')) for n in _ETABLE_ATOM_TYPES]
		OHD = [n[:2] in ('OH', 'OW') or n == 'Oet3'
			for n in _ETABLE_ATOM_TYPES]
		for i in range(NTYPES):
			for j in range(NTYPES):
				if (ACC[i] and DON[j]) or (DON[i] and ACC[j]):
					sigma[i][j] = (etopt['LJ_HB_OH']
						if (DON[i] and OHD[i]) or (DON[j] and OHD[j])
						else etopt['LJ_HB_DIS'])
				elif (ACC[i] and POL[j]) or (POL[i] and ACC[j]):
					sigma[i][j] = etopt['LJ_HB_HDIS']
		inv_neg2_pi_sqrt_pi = -1.0 / (2.0 * math.pi * math.sqrt(math.pi))
		lk_coeff_tmp = [inv_neg2_pi_sqrt_pi * LK_DG[i] / LK_L[i]
			for i in range(NTYPES)]
		lj_r6  = [[0.0]*NTYPES for _ in range(NTYPES)]
		lj_r12 = [[0.0]*NTYPES for _ in range(NTYPES)]
		lj_si  = [[0.0]*NTYPES for _ in range(NTYPES)]
		lj_ss  = [[0.0]*NTYPES for _ in range(NTYPES)]
		lk_coeff = [[0.0]*NTYPES for _ in range(NTYPES)]
		for i in range(NTYPES):
			for j in range(NTYPES):
				s = sigma[i][j]
				s6 = s**6
				s12 = s6 * s6
				wd = math.sqrt(LJ_W[i] * LJ_W[j])
				lj_r6[i][j]  = -2.0 * wd * s6
				lj_r12[i][j] = wd * s12
				lj_ss[i][j] = (wd / s) * LJ_S2W
				lj_si[i][j] = wd * LJ_V2W - lj_ss[i][j] * s * LJ_SWITCH_D2S
				lk_coeff[i][j] = lk_coeff_tmp[i] * LK_V[j]
		pairs = [None] * (NTYPES * NTYPES)
		for is_ in range(NTYPES):
			for io_ in range(NTYPES):
				i = min(is_, io_); j = max(is_, io_)
				s_ij = sigma[i][j]
				pair = _etableonepair(
					i, j, is_, io_, s_ij,
					LJ_R, LJ_W, LK_DG, LK_L, LK_V,
					lj_r6, lj_r12, lj_si, lj_ss, lk_coeff,
					etopt['FAR_LO'], MAX_DIS,
					etopt['BPA2'], etopt['SOL_LO'], etopt['SOL_HI'],
					etopt['SOL_CAP'], etopt['CARB_DIS'],
					etopt['LK_MIN_D2S'],
					(LJ_SWITCH_D2S * s_ij) ** 2)
				pairs[is_ * NTYPES + io_] = pair
		return {'atom_types': list(_ETABLE_ATOM_TYPES),
			'n_types': NTYPES, 'pairs': pairs}
	def _r15fetch(path):
		'''
		Download one repository file as a UTF-8 string
		Arguments:
		----------
			path: str - path under the database/ root
		Returns:
		--------
			str: file contents
		'''
		with urllib.request.urlopen(_REF15_REPO + path, timeout=120) as r:
			return r.read().decode('utf-8')
	def _r15fetchgz(path):
		'''
		Download one repository .gz file and return decompressed text
		Arguments:
		----------
			path: str - path under the database/ root
		Returns:
		--------
			str: decompressed file contents
		'''
		with urllib.request.urlopen(_REF15_REPO + path, timeout=120) as r:
			return gzip.decompress(r.read()).decode('utf-8')
	def _patchcases(text):
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
	def _casecharges(case):
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
	def _casetypes(case):
		'''
		Read the added-hydrogen and reassigned atom types of one case
		Arguments:
		----------
			case: str - the body of a single BEGIN_CASE block
		Returns:
		--------
			dict: atom name to atom type, using 'H' for the added
			terminal hydrogens
		'''
		out = {}
		m = re.search(r'^ADD_ATOM 1H\s+(\S+)', case, re.M)
		if m: out['H'] = m.group(1)
		for nm in ('N', 'CA'):
			m = re.search(r'^SET_ATOM_TYPE %s\s+(\S+)' % nm, case, re.M)
			if m: out[nm] = m.group(1)
		return out
	def _ncaarename(e, m, drop=()):
		'''
		Copy a residue template, renaming and dropping atoms
		Arguments:
		----------
			e: Source residue template
			m: Map from old atom name to new atom name
			drop: Atom names to leave out of the copy
		Returns:
		--------
			dict: a new template, leaving the source untouched
		'''
		a = dict((m.get(k, k), dict(v))
			for k, v in e['atoms'].items() if k not in drop)
		b = [[m.get(x, x), m.get(y, y), o]
			for x, y, o in e['bonds']
			if x not in drop and y not in drop]
		al = dict((k, m.get(v, v)) for k, v in e['aliases'].items()
			if k not in drop and v not in drop)
		return {'name': None, 'aa': None, 'atoms': a,
			'bonds': b, 'aliases': al}
	def _ncaarenorm(a, onto):
		'''
		Shift partial charges so a template sums to its target
		Arguments:
		----------
			a: Atom table of the template, adjusted in place
			onto: Atom name that absorbs the rounding residual
		Returns:
		--------
			Adjusts a in place and returns no value
		'''
		a[onto]['charge'] -= sum(x['charge'] for x in a.values())
	def _ncaafinish(e, tri, al=None):
		'''
		Finish a residue template with its codes and connectivity
		Arguments:
		----------
			e: Template being completed
			tri: Three-letter code to record
			al: Atom name aliases to record, or None for none
		Returns:
		--------
			dict: the completed template
		'''
		e['name'] = tri
		e['aa'] = tri
		if al: e['aliases'].update(al)
		return e
	def _ncaaphospho(base, bridge, hs, tri, al, res, patch):
		'''
		Build a phosphorylated residue template from its parent
		Arguments:
		----------
			base: Three-letter code of the parent residue
			bridge: Atom name of the oxygen carrying the phosphate
			hs: Names of the two phosphate hydroxyl hydrogens
			tri: Three-letter code of the phosphorylated residue
			al: Atom name aliases to record
			res: Table of parsed residue templates to read the parent from
		Returns:
		--------
			dict: the phosphorylated template
		P and O1P are taken from the upstream phosphorylation patch.
		O2P and O3P are built as protonated hydroxyls from SER instead
		of the patch's third free oxygen, so the residue carries a
		neutral phosphate rather than the upstream dianion.
		'''
		e = _ncaarename(res[base], {},
			drop=({'THR': 'HG1', 'TYR': 'HH'}[base],))
		ser = res['SER']['atoms']
		add = {}
		for line in patch.splitlines():
			toks = line.split('#', 1)[0].split()
			if len(toks) < 5 or toks[0] != 'ADD_ATOM': continue
			add[toks[1]] = {'type': toks[2], 'mm_type': toks[3],
				'charge': float(toks[4])}
		for nm in ('P', 'O1P'):
			if nm not in add:
				raise RuntimeError('port: %s patch has no %s' % (tri, nm))
			e['atoms'][nm] = dict(add[nm])
		e['bonds'] += [[bridge, 'P', 1], ['P', 'O1P', 1]]
		for o, h in zip(('O2P', 'O3P'), hs):
			e['atoms'][o] = dict(ser['OG'])
			e['atoms'][h] = dict(ser['HG'])
			e['bonds'] += [['P', o, 1], [o, h, 1]]
		_ncaarenorm(e['atoms'], 'P')
		return _ncaafinish(e, tri, al)
	def _ncaaresidues(res, fetch, parse):
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
		ORN keeps the atom types of the upstream ornithine topology but
		takes its partial charges from LYS and ASN, so the residue is
		consistent with the rest of the score function rather than with
		the CHARMM-style charges the upstream file carries. Its NE is
		retyped to the neutral amine NH2O and 3HE is dropped, modelling
		the free base rather than the protonated ammonium.
		'''
		NC = ('chemical/residue_type_sets/fa_standard/'
			'residue_types/l-ncaa/')
		out = {}
		mse = _ncaarename(res['MET'], {'SD': 'SE'})
		mse['aliases'].pop('SE', None)
		out['MSE'] = _ncaafinish(mse, 'MSE')
		sec = _ncaarename(res['CYS'], {'SG': 'SE', 'HG': 'HE'})
		sec['aliases'] = {'HB1': '1HB', 'HB2': '2HB'}
		out['SEC'] = _ncaafinish(sec, 'SEC')
		orn = _ncaarename(parse(fetch(NC + 'ornithine.params')), {},
			drop=('3HE',))
		lys = res['LYS']['atoms']
		asn = res['ASN']['atoms']
		for nm in ('N', 'CA', 'C', 'O', 'H', 'HA', 'CB', 'CG', 'CD',
				'1HB', '2HB', '1HG', '2HG', '1HD', '2HD'):
			orn['atoms'][nm]['charge'] = lys[nm]['charge']
		orn['atoms']['NE']['type'] = 'NH2O'
		for nm, src in (('NE', 'ND2'), ('1HE', '1HD2'),
				('2HE', '2HD2')):
			orn['atoms'][nm]['charge'] = asn[src]['charge']
		_ncaarenorm(orn['atoms'], 'NE')
		out['ORN'] = _ncaafinish(orn, 'ORN')
		FT = {'CZ1': 'CZ2', 'CZ2': 'CZ3', 'CT': 'CH2', 'FI': 'F01',
			'1HD1': 'HD1', '1HE1': 'HE1', '1HE3': 'HE3',
			'1HZ1': 'HZ2', '1HZ2': 'HZ3'}
		tmp = dict((k, '@%d' % i) for i, k in enumerate(FT))
		ft = parse(fetch(NC + '6-fluoro-tryptophan.params'))
		ft = _ncaarename(_ncaarename(ft, tmp),
			dict((v, FT[k]) for k, v in tmp.items()))
		_ncaarenorm(ft['atoms'], 'CH2')
		out['FT6'] = _ncaafinish(ft, 'FT6')
		PATCH = 'chemical/residue_type_sets/fa_standard/patches/'
		out['TPO'] = _ncaaphospho('THR', 'OG1', ('1HOP', '2HOP'), 'TPO',
			{'1HG': '1HG2', '2HG': '2HG2', '3HG': '3HG2'}, res,
			fetch(PATCH + 'thr_phosphorylated.txt'))
		out['PTR'] = _ncaaphospho('TYR', 'OH', ('HO2P', 'HO3P'), 'PTR',
			{'1HD': 'HD1', '2HD': 'HD2', '1HE': 'HE1', '2HE': 'HE2'},
			res, fetch(PATCH + 'tyr_phosphorylated.txt'))
		return out
	def _parsenrchi(txt, n_chi, n_disc_chi):
		'''
		Parse one non-rotameric chi density library
		Arguments:
		----------
			txt: Text of a bbdep.densities.lib file
			n_chi: Number of chi angles the residue has
			n_disc_chi: Number of those chi angles that are rotameric
		Returns:
		--------
			dict: per backbone bin and rotamer well, the sampled density of
			the terminal non-rotameric chi
		'''
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
	def _wkcal(weights, key):
		'''
		Read one REF15 term weight and convert it to kJ/mol
		Arguments:
		----------
			weights: Parsed ref2015 weight table
			key: Rosetta name of the term
		Returns:
		--------
			float: the weight in kJ/mol
		'''
		if key not in weights:
			raise RuntimeError(
				'port: term %r absent from ref2015.wts' % key)
		return weights[key] * _KCAL_TO_KJ
	def _openff(db):
		'''
		Port OpenFF Sage 2.3.0 into the Energy Parameters section
		Arguments:
		----------
			db: The loaded database, modified in place
		Returns:
		--------
			Writes db['Energy Parameters']['OpenFF'] and returns no value
		'''
		ep = db.setdefault('Energy Parameters', {})
		commit = 'edd7724103a558328c358a9e35462334c4a45b6f'
		url = ('https://raw.githubusercontent.com/openforcefield/'
			'openff-forcefields/' + commit
			+ '/openforcefields/offxml/openff-2.3.0.offxml')
		root = ET.fromstring(_download(url))
		bonds = {}
		for p in root.find('Bonds'):
			a = p.attrib
			bonds[a['smirks']] = {'id': a.get('id'),
				'r_0': _qval(a['length'], 'angstrom'),
				'K_b': _qval(a['k'],
					'kilojoule_per_mole * angstrom ** -2')}
		angles = {}
		for p in root.find('Angles'):
			a = p.attrib
			angles[a['smirks']] = {'id': a.get('id'),
				'theta_0': _qval(a['angle'], 'degree'),
				'K_theta': _qval(a['k'],
					'kilojoule_per_mole * radian ** -2')}
		propers   = _converttorsions(root.find('ProperTorsions'))
		impropers = _converttorsions(root.find('ImproperTorsions'))
		for par in impropers.values():
			for comp in par['components']: comp.pop('idivf', None)
		vdw = {}
		for p in root.find('vdW'):
			a = p.attrib
			rec = {'id': a.get('id'),
				'epsilon': _qval(a['epsilon'], 'kilojoule_per_mole')}
			if 'sigma' in a: rec['sigma'] = _qval(a['sigma'], 'angstrom')
			else: rec['r'] = _qval(a['rmin_half'], 'angstrom')
			rec['alpha'] = 0.0
			vdw[a['smirks']] = rec
		charges = {}
		for p in root.find('LibraryCharges'):
			a = p.attrib
			qs, i = [], 1
			while ('charge%d' % i) in a:
				qs.append(_qval(a['charge%d' % i], 'elementary_charge'))
				i += 1
			charges[a['smirks']] = {'id': a.get('id'), 'q': qs}
		constraints = {}
		for p in root.find('Constraints'):
			a = p.attrib
			rec = {'id': a.get('id')}
			if 'distance' in a:
				rec['distance'] = _qval(a['distance'], 'angstrom')
			constraints[a['smirks']] = rec
		nagl_commit = '6a30bde31fc9ba7f9ff218dacd291184e2f70946'
		nagl_url = ('https://raw.githubusercontent.com/openforcefield/'
			'openff-nagl-models/' + nagl_commit + '/openff/nagl_models/'
			'models/am1bcc/openff-gnn-am1bcc-1.0.0.pt')
		prev = ep.get('OpenFF') or ep.get('openFF') or {}
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
		block['AM1BCC'] = _naglweights(nagl_url)
		ep.pop('OpenFF', None)
		ep['OpenFF'] = block
	def _ff19sb(db):
		'''
		Port AMBER ff19SB into the Energy Parameters section
		Arguments:
		----------
			db: The loaded database, modified in place
		Returns:
		--------
			Writes db['Energy Parameters']['ff19SB'] and returns no value
		'''
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
			root = ET.fromstring(_download(url))
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
					c1, c2 = _cidof(b, 1), _cidof(b, 2)
					bonds[f'<at={c1},{c2}>[*:1]~[*:2]'] = {
						'r_0': float(b.attrib['length']) * 10.0,
						'K_b': float(b.attrib['k']) * 0.01}
			haf = root.find('HarmonicAngleForce')
			if haf is not None:
				for a in haf:
					c1, c2, c3 = (_cidof(a, 1), _cidof(a, 2), _cidof(a, 3))
					angles[f'<at={c1},{c2},{c3}>[*:1]~[*:2]~[*:3]'] = {
						'theta_0': math.degrees(
							float(a.attrib['angle'])),
						'K_theta': float(a.attrib['k'])}
			ptf = root.find('PeriodicTorsionForce')
			if ptf is not None:
				for t in ptf:
					cs = [_cidof(t, i) for i in (1, 2, 3, 4)]
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
	def _charmm36(db):
		'''
		Port CHARMM36 into the Energy Parameters section
		Arguments:
		----------
			db: The loaded database, modified in place
		Returns:
		--------
			Writes db['Energy Parameters']['CHARMM36'] and returns no value
		'''
		ep = db.setdefault('Energy Parameters', {})
		commit = 'f7fa0c27c1f8d943c339d67b3bf22f026d0bd8b5'
		xml_url = ('https://raw.githubusercontent.com/openmm/openmm/'
			+ commit + '/wrappers/python/openmm/app/data/charmm36.xml')
		root = ET.fromstring(_download(xml_url))
		bonds = {}
		hbf = root.find('HarmonicBondForce')
		if hbf is not None:
			for b in hbf:
				c1, c2 = _cidof(b, 1), _cidof(b, 2)
				bonds[f'<at={c1},{c2}>[*:1]~[*:2]'] = {
					'r_0': float(b.attrib['length']) * 10.0,
					'K_b': float(b.attrib['k']) * 0.01}
		angles = {}
		haf = root.find('HarmonicAngleForce')
		if haf is not None:
			for a in haf:
				c1, c2, c3 = _cidof(a, 1), _cidof(a, 2), _cidof(a, 3)
				angles[f'<at={c1},{c2},{c3}>[*:1]~[*:2]~[*:3]'] = {
					'theta_0': math.degrees(float(a.attrib['angle'])),
					'K_theta': float(a.attrib['k'])}
		ub = {}
		ubf = root.find('AmoebaUreyBradleyForce')
		if ubf is not None:
			for u in ubf:
				c1, c2, c3 = _cidof(u, 1), _cidof(u, 2), _cidof(u, 3)
				ub[f'<at={c1},{c2},{c3}>[*:1]~[*:2]~[*:3]'] = {
					's_0':  float(u.attrib['d']) * 10.0,
					'K_ub': float(u.attrib['k']) * 0.01}
		propers = {}
		ptf = root.find('PeriodicTorsionForce')
		if ptf is not None:
			for t in ptf:
				if t.tag != 'Proper': continue
				cs = [_cidof(t, i) for i in (1, 2, 3, 4)]
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
				cs = [_cidof(t, i) for i in (1, 2, 3, 4)]
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
		templates = _charmmtypes(root)
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
	def _vina(db):
		'''
		Port AutoDock Vina into the Score Parameters section
		Arguments:
		----------
			db: The loaded database, modified in place
		Returns:
		--------
			Writes db['Score Parameters']['AutoDock Vina'] and returns no
			value
		'''
		sp = db.setdefault('Score Parameters', {})
		FILES = ('potentials.h', 'vina.h',
			'scoring_function.h', 'atom_constants.h')
		src = {n: _vinafetch(n) for n in FILES}
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
		sf = src['scoring_function.h']
		gpat = (r'new\s+vina_gaussian\(\s*(-?[\d.]+)\s*,'
			r'\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)')
		gs = re.findall(gpat, sf)
		if len(gs) < 2:
			raise Exception('Vina: did not find two vina_gaussian entries')
		g1_off, g1_w, g1_cut = (float(x) for x in gs[0])
		g2_off, g2_w, g2_cut = (float(x) for x in gs[1])
		rep_off, rep_cut = _vinafirst(
			r'new\s+vina_repulsion\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)', sf)
		hyd_good, hyd_bad, hyd_cut = _vinafirst(
			r'new\s+vina_hydrophobic\(\s*(-?[\d.]+)\s*,'
			r'\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)', sf)
		hb_good, hb_bad, hb_cut = _vinafirst(
			r'new\s+vina_non_dir_h_bond\(\s*(-?[\d.]+)\s*,'
			r'\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)', sf)
		ac = src['atom_constants.h']
		xs_decl = re.findall(
			r'^\s*const\s+sz\s+XS_TYPE_([A-Za-z0-9_]+)\s*=\s*(\d+)\s*;',
			ac, re.M)
		if not xs_decl:
			raise Exception('Vina: no XS_TYPE_* found in atom_constants.h')
		xs_idx_to_name = {}
		for nm, i in xs_decl:
			if nm in ('SIZE',): continue
			xs_idx_to_name[int(i)] = nm
		rad_block = re.search(
			r'const\s+fl\s+xs_vdw_radii\s*\[\s*\]\s*=\s*\{([^}]*)\}\s*;',
			ac, re.S)
		if rad_block is None:
			raise Exception('Vina: xs_vdw_radii block not found')
		rad_vals = []
		for line in rad_block.group(1).split('\n'):
			mm = re.match(r'\s*(-?[\d.]+)\s*,', line)
			if mm: rad_vals.append(float(mm.group(1)))
		hphob = _vinagrab('xs_is_hydrophobic', ac)
		accept = _vinagrab('xs_is_acceptor', ac)
		donor = _vinagrab('xs_is_donor', ac)
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
				'scale':  1.0 / _KCAL_TO_KJ,
				'cutoff': float(g1_cut),
				'nrot_w': w_rot,
				'glue_w': w_glue * _KCAL_TO_KJ},
			'XS_atom_types': xs_types,
			'Gauss1': {
				'offset': g1_off, 'width': g1_w, 'cutoff': g1_cut,
				'weight': w_gauss1 * _KCAL_TO_KJ},
			'Gauss2': {
				'offset': g2_off, 'width': g2_w, 'cutoff': g2_cut,
				'weight': w_gauss2 * _KCAL_TO_KJ},
			'Repulsion': {
				'offset': rep_off, 'cutoff': rep_cut,
				'weight': w_rep * _KCAL_TO_KJ},
			'Hydrophobic': {
				'good': hyd_good, 'bad': hyd_bad, 'cutoff': hyd_cut,
				'weight': w_hyd * _KCAL_TO_KJ},
			'HBond': {
				'good': hb_good, 'bad': hb_bad, 'cutoff': hb_cut,
				'weight': w_hbond * _KCAL_TO_KJ},
			'Terms': [
				['Gauss1Potential',      {}],
				['Gauss2Potential',      {}],
				['RepulsionPotential',   {}],
				['HydrophobicPotential', {}],
				['HBondPotential',       {}],
				['TorsionalPenalty',     {}]]}
		sp['AutoDock Vina'] = block
	def _ref15(db):
		'''
		Port Rosetta REF15 into the Score Parameters section
		Arguments:
		----------
			db: The loaded database, modified in place
		Returns:
		--------
			Writes db['Score Parameters']['REF15'] and returns no value
		'''
		if not accept_rosetta_license:
			raise RuntimeError(
				'port: REF15 downloads data from RosettaCommons, which '
				'is distributed under the Rosetta Software '
				'Non-Commercial License Agreement, NOT an open-source '
				'licence. It is free for employees of not-for-profit '
				'research institutions, government laboratories and '
				'universities, and for individuals not acting for or '
				'on behalf of a for-profit entity. Commercial use '
				'requires a separate licence from University of '
				'Washington CoMotion (license@uw.edu). Pose itself is '
				'Apache-2.0 and does not redistribute any Rosetta '
				'data. If you qualify, call '
				"Port('ref15', accept_rosetta_license=True)")
		sp = db.setdefault('Score Parameters', {})
		props_txt = _r15fetch(
			'chemical/atom_type_sets/fa_standard/atom_properties.txt')
		atom_types = {}
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
		lkb_txt = _r15fetch('chemical/atom_type_sets/fa_standard/extras/'
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
		nterm_txt = _r15fetch('chemical/residue_type_sets/fa_standard/'
			'patches/NtermProteinFull.txt')
		cterm_txt = _r15fetch('chemical/residue_type_sets/fa_standard/'
			'patches/CtermProteinFull.txt')
		cys_txt = _r15fetch('chemical/residue_type_sets/fa_standard/'
			'residue_types/l-caa/CYS.params')
		ncases = _patchcases(nterm_txt)
		pro_case = next(c for c in ncases
			if re.search(r'^NAME3 PRO\s*$', c, re.M))
		gly_case = next(c for c in ncases
			if re.search(r'^AA GLY\s*$', c, re.M))
		term_q = {'PRO': _casecharges(pro_case),
			'GLY': _casecharges(gly_case),
			'generic': _casecharges(ncases[-1])}
		ccase = _patchcases(cterm_txt)[-1]
		term_q['cterm'] = {nm: float(re.search(
			r'^SET_ATOMIC_CHARGE %s\s+(-?[\d.]+)' % nm, ccase, re.M).group(1))
			for nm in ('C', 'O')}
		term_t = {'PRO': _casetypes(pro_case),
			'GLY': _casetypes(gly_case),
			'generic': _casetypes(ncases[-1])}
		term_t['cterm'] = {nm: re.search(
			r'^SET_ATOM_TYPE %s\s+(\S+)' % nm, ccase, re.M).group(1)
			for nm in ('C', 'O')}
		term_q['disulfide_SG'] = float(re.search(
			r'^ATOM\s+SG\s+\S+\s+\S+\s+(-?[\d.]+)', cys_txt, re.M).group(1))
		m = re.search(r'^SET_ICOOR CAV\s+\S+\s+([\d.]+)\s+([\d.]+)',
			pro_case, re.M)
		proclose = {'cav_theta': float(m.group(1)),
			'cav_d': float(m.group(2))}
		pce_url = (_REF15_REPO.replace('/database/', '/source/src/')
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
		pro_txt = _r15fetch('chemical/residue_type_sets/fa_standard/'
			'residue_types/l-caa/PRO.params')
		m = re.search(r'^ICOOR_INTERNAL\s+NV\s+\S+\s+([\d.]+)\s+([\d.]+)',
			pro_txt, re.M)
		if m is None:
			raise RuntimeError('port: NV ICOOR not found in PRO.params')
		proclose['nv_theta'] = float(m.group(1))
		proclose['nv_d'] = float(m.group(2))
		dsl_url = (_REF15_REPO.replace('/database/', '/source/src/')
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
		hbg_url = (_REF15_REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/hbonds/hbonds_geom.cc')
		with urllib.request.urlopen(hbg_url, timeout=120) as resp:
			hbg_txt = resp.read().decode('utf-8')
		m = re.search(r'fade_energy\(\)\s*\?\s*([\d.]+)\s*:\s*([\d.]+)',
			hbg_txt)
		if m is None:
			raise RuntimeError('port: sp2 fade slope not found in %s'
				% hbg_url)
		sp2_slope = float(m.group(1))
		opt_url = (_REF15_REPO.replace('/database/', '/source/src/')
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
		if m2 is None:
			raise RuntimeError('port: pro_close_planar_constraint default '
				'not found in %s' % opt_url)
		proclose['planar_sd'] = float(m2.group(1))
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
		hbc_url = (_REF15_REPO.replace('/database/', '/source/src/')
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
		lki_url = (_REF15_REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/lkball/LK_BallInfo.cc')
		with urllib.request.urlopen(lki_url, timeout=120) as resp:
			lki_txt = resp.read().decode('utf-8')
		m = re.search(r'optimal_water_distance\(\s*([\d.]+)', lki_txt)
		if m is None:
			raise RuntimeError('port: optimal_water_distance not found')
		lkball = {'opt_dist': float(m.group(1))}
		m = re.search(r"lk_ball_ramp_width_A2'.*?default\s*=\s*[\"']?([\d.]+)",
			opt_txt, re.S)
		if m is None:
			raise RuntimeError('port: lk_ball_ramp_width_A2 default not '
				'found in %s' % opt_url)
		lkball['ramp_w2'] = float(m.group(1))
		etb_url = (_REF15_REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/etable/Etable.cc')
		with urllib.request.urlopen(etb_url, timeout=120) as resp:
			etb_txt = resp.read().decode('utf-8')
		eto_url = (_REF15_REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/etable/EtableOptions.cc')
		with urllib.request.urlopen(eto_url, timeout=120) as resp:
			eto_txt = resp.read().decode('utf-8')
		etopt = {'LJ_SWITCH_D2S': 0.6,
			'BPA2': float(re.search(
			r'bins_per_A2\s*\(\s*([\d.]+)\s*\)', eto_txt).group(1)),
			'SOL_LO': int(re.search(
			r'SWTCH\s*-\s*(\d+)\s*\)', etb_txt).group(1)),
			'SOL_HI': int(re.search(
			r'SWTCH\s*\+\s*(\d+)\s*,', etb_txt).group(1)),
			'SOL_CAP': int(re.search(
			r'SWTCH\s*\+\s*\d+\s*,\s*(\d+)\s*\)', etb_txt).group(1)),
			'CARB_DIS': float(re.search(
			r'Real const bin = \(\s*([\d.]+)\s*\*', etb_txt).group(1)),
			'LJ_HB_DIS': float(re.search(
			r'lj_hbond_dis_\s*\(\s*([\d.]+)\s*\)', etb_txt).group(1)),
			'LJ_HB_HDIS': float(re.search(
			r"'lj_hbond_hdis'.*?default\s*=\s*'([\d.]+)'", opt_txt).group(1)),
			'LJ_HB_OH': float(re.search(
			r"'lj_hbond_OH_donor_dis'.*?default\s*=\s*'([\d.]+)'",
			opt_txt).group(1)),
			'LK_MIN_D2S': float(re.search(
			r'lk_min_dis2sigma_\s*\(\s*([\d.]+)\s*\)', etb_txt).group(1))}
		m = re.search(r'\(\s*max_dis_\s*-\s*([\d.]+)\s*\)\s*\*\s*10\.0',
			etb_txt)
		if m is None:
			raise RuntimeError('port: Etable far offset not found in %s'
				% etb_url)
		far_off = float(m.group(1))
		lkball['far_offset'] = far_off
		m = re.search(r"Option\(\s*'fa_max_dis'.*?default\s*=\s*"
			r"[\"']?([\d.]+)", opt_txt, re.S)
		if m is None:
			raise RuntimeError('port: fa_max_dis default not found')
		lkball['max_dis'] = float(m.group(1))
		etopt['MAX_DIS'] = lkball['max_dis']
		lkball['far_lo'] = lkball['max_dis'] - far_off
		etopt['FAR_LO'] = lkball['far_lo']
		lke_url = (_REF15_REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/lkball/LK_BallEnergy.cc')
		with urllib.request.urlopen(lke_url, timeout=120) as resp:
			lke_txt = resp.read().decode('utf-8')
		m = re.search(r'h2o_radius\(\s*([\d.]+)', lke_txt)
		if m is None:
			raise RuntimeError('port: h2o_radius not found in %s' % lke_url)
		lkball['h2o_radius'] = float(m.group(1))
		for hyb in ('sp2', 'sp3'):
			rows = re.findall(
				r'params_' + hyb + r'\.push_back\(distance\);'
				r'params_' + hyb + r'\.push_back\(\s*([\d.]+)\s*\);'
				r'params_' + hyb + r'\.push_back\(\s*([\d.]+)\s*\)',
				lki_txt)
			if not rows:
				raise RuntimeError('port: lk_ball %s waters not found in %s'
					% (hyb, lki_url))
			lkball['ang_' + hyb] = 180.0 - float(rows[0][0])
			lkball['dih_' + hyb] = [float(r[1]) for r in rows]
		dslc_url = (_REF15_REPO.replace('/database/', '/source/src/')
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
		omg_url = (_REF15_REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/OmegaTether.cc')
		with urllib.request.urlopen(omg_url, timeout=120) as resp:
			omg_txt = resp.read().decode('utf-8')
		m = re.search(r'Real\s+weight\s*=\s*([\d.]+)\s*;', omg_txt)
		if m is None:
			raise RuntimeError('port: omega weight not found in %s' % omg_url)
		omega_k = float(m.group(1))
		cpf_url = (_REF15_REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/etable/count_pair/CountPairFunction.cc')
		with urllib.request.urlopen(cpf_url, timeout=120) as resp:
			cpf_txt = resp.read().decode('utf-8')
		m = re.search(r'cp_half\(\s*([\d.]+)', cpf_txt)
		if m is None:
			raise RuntimeError('port: cp_half not found in %s' % cpf_url)
		cp_half = float(m.group(1))
		m = re.search(r'if\s*\(\s*input_energy\s*>\s*([\d.]+)L?\s*\)',
			hbg_txt)
		m2 = re.search(r'else\s+if\s*\(\s*input_energy\s*>\s*'
			r'(-[\d.]+)L?\s*\)', hbg_txt)
		if m is None or m2 is None:
			raise RuntimeError('port: fade thresholds not found in %s'
				% hbg_url)
		hb_sp2['fade_hi'] = float(m.group(1))
		hb_sp2['fade_lo'] = float(m2.group(1))
		lam = [float(x) for x in re.findall(
			r'^\S+\s+\S+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+([\d.]+)',
			props_txt, re.M)]
		if not lam:
			raise RuntimeError('port: no LK_LAMBDA column parsed')
		lk_lambda_default = max(set(lam), key=lam.count)
		wts_txt = _r15fetch('scoring/weights/ref2015.wts')
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
		AAS = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
			'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
		residues = {}
		for aa in AAS:
			txt = _r15fetch(
				'chemical/residue_type_sets/fa_standard/residue_types/'
				'l-caa/%s.params' % aa)
			residues[aa] = _parseparams(txt)
		try:
			txt = _r15fetch(
				'chemical/residue_type_sets/fa_standard/residue_types/'
				'l-caa/HIS_D.params')
			residues['HIS_D'] = _parseparams(txt)
		except Exception:
			pass
		residues.update(_ncaaresidues(residues, _r15fetch, _parseparams))
		hb_dir = 'scoring/score_functions/hbonds/ref2015_params/'
		hb_files = ('HBPoly1D.csv', 'HBEval.csv', 'HBFadeIntervals.csv',
			'HBDonChemType.csv', 'HBAccChemType.csv',
			'HBAccHybridization.csv', 'HBSeqSep.csv', 'DonStrength.csv',
			'AccStrength.csv', 'HBondWeightType.csv', 'HybridizationType.csv')
		hb_raw = {}
		for fname in hb_files:
			try: hb_raw[fname] = _r15fetch(hb_dir + fname)
			except Exception: hb_raw[fname] = ''
		hb_data = _parsehbonddata(hb_raw)
		rama_dir = 'scoring/score_functions/rama/fd/'
		rama_data = {}
		for kind, fname in (('all', 'all.ramaProb'),
				('prepro', 'prepro.ramaProb')):
			try:
				txt = _r15fetch(rama_dir + fname)
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
					i_phi = (phi + 180) // 10
					i_psi = (psi + 180) // 10
					if 0 <= i_phi < 36 and 0 <= i_psi < 36:
						t = table.setdefault(aa,
							[[0.0]*36 for _ in range(36)])
						t[i_phi][i_psi] = nE
				rama_data[kind] = table
			except Exception:
				rama_data[kind] = {}
		omega_dir = 'scoring/score_functions/omega/'
		omega_tables = {}
		for kind in ('all', 'gly', 'pro', 'valile'):
			txt = _r15fetch(omega_dir + 'omega_ppdep.' + kind + '.txt')
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
		paa_txt = _r15fetch('scoring/score_functions/P_AA_pp/P_AA')
		p_aa = {}
		for line in paa_txt.splitlines():
			toks = line.split()
			if len(toks) >= 2:
				try: p_aa[toks[0]] = float(toks[1])
				except ValueError: pass
		NRCHI_AA = [
			('ASN', 2, 1), ('ASP', 2, 1), ('GLN', 3, 2), ('GLU', 3, 2),
			('HIS', 2, 1), ('PHE', 2, 1), ('TRP', 2, 1), ('TYR', 2, 1)]
		nrchi_db = {}
		for aa3, n_chi, n_disc_chi in NRCHI_AA:
			gz_path = ('rotamer/shapovalov/StpDwn_0-0-0/'
				+ aa3.lower() + '.bbdep.densities.lib.gz')
			txt = _r15fetchgz(gz_path)
			nrchi_db[aa3] = _parsenrchi(txt, n_chi, n_disc_chi)
		prop_txt = _r15fetch('scoring/score_functions/P_AA_pp/shapovalov/'
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
		cou_url = (_REF15_REPO.replace('/database/', '/source/src/')
			+ 'core/scoring/etable/coulomb/Coulomb.cc')
		with urllib.request.urlopen(cou_url, timeout=120) as resp:
			cou_txt = resp.read().decode('utf-8')
		m = re.search(r'C0_\s*=\s*([\d.]+)', cou_txt)
		if m is None:
			raise RuntimeError('port: C0_ not found in %s' % cou_url)
		coulomb_c0 = float(m.group(1))
		elec = {}
		for nm, key in (('elec_max_dis', 'fa_elec_max_dis'),
				('elec_min_dis', 'fa_elec_min_dis'),
				('elec_sigmoidal_die_D', 'sigmoidal_D'),
				('elec_sigmoidal_die_D0', 'sigmoidal_D0'),
				('elec_sigmoidal_die_S', 'sigmoidal_S')):
			m = re.search(r"Option\(\s*'" + nm
				+ r"'.*?default\s*=\s*[\"']?([\d.]+)", opt_txt, re.S)
			if m is None:
				raise RuntimeError('port: %s not found in %s'
					% (nm, opt_url))
			elec[key] = float(m.group(1))
		block = {
			'Constants': {
				'scale':           1.0 / _KCAL_TO_KJ,
				'fa_max_dis':      lkball['max_dis'],
				'fa_elec_max_dis': elec['fa_elec_max_dis'],
				'fa_elec_min_dis': elec['fa_elec_min_dis'],
				'coulomb_C0':      coulomb_c0,
				'sigmoidal_D':     elec['sigmoidal_D'],
				'sigmoidal_D0':    elec['sigmoidal_D0'],
				'sigmoidal_S':     elec['sigmoidal_S']},
			'TerminalCharges': term_q,
			'TerminalTypes':   term_t,
			'HBondSp2':        hb_sp2,
			'LkBall': dict(lkball,
				lk_lambda_default=lk_lambda_default),
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
			'FaAtr':          {'weight': _wkcal(weights, 'fa_atr')},
			'FaRep':          {'weight': _wkcal(weights, 'fa_rep')},
			'FaSol':          {'weight': _wkcal(weights, 'fa_sol')},
			'FaIntraRep':     {'weight': _wkcal(weights,
				'fa_intra_rep')},
			'FaIntraSolXover4':
				{'weight': _wkcal(weights, 'fa_intra_sol_xover4')},
			'LkBallWtd':      {'weight': _wkcal(weights, 'lk_ball_wtd'),
				'atom_weights': lkb_wts},
			'FaElec':         {'weight': _wkcal(weights, 'fa_elec')},
			'HBondSrBb':      {'weight': _wkcal(weights, 'hbond_sr_bb')},
			'HBondLrBb':      {'weight': _wkcal(weights, 'hbond_lr_bb')},
			'HBondBbSc':      {'weight': _wkcal(weights, 'hbond_bb_sc')},
			'HBondSc':        {'weight': _wkcal(weights, 'hbond_sc')},
			'DslfFa13':       dict(dslf,
				weight=_wkcal(weights, 'dslf_fa13')),
			'Omega':          {'weight': _wkcal(weights, 'omega'),
				'tether_k': omega_k,
				'undefined_torsion': 0.0},
			'FaDun':          {'weight': _wkcal(weights, 'fa_dun')},
			'PAaPp':          {'weight': _wkcal(weights, 'p_aa_pp')},
			'YhhPlanarity':   {'weight': _wkcal(weights,
				'yhh_planarity')},
			'Ref':            {'weight': _wkcal(weights, 'ref')},
			'RamaPreProTerm': {'weight': _wkcal(weights, 'rama_prepro')},
			'ProClose':       dict(proclose,
				weight=_wkcal(weights, 'pro_close')),
			'FaIntraAtr':         {'weight': 0.0},
			'LkBallIso':          {'weight': 0.0},
			'LkBallBridge':       {'weight': 0.0},
			'CartBonded':         {'weight': 0.0},
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
		sp['REF15']['EtablePairParams'] = _etableparams(atom_types, etopt)
	dispatch = {'OPENFF': _openff, 'FF19SB': _ff19sb,
		'CHARMM36': _charmm36, 'AUTODOCK VINA': _vina, 'REF15': _ref15}
	key = str(name).upper()
	if key not in dispatch:
		print(f'[-] Error: unknown name {name!r}, '
			f'choose from {sorted(dispatch)}')
		return False
	db_path = os.path.join(
		os.path.dirname(os.path.abspath(__file__)), 'database.json')
	print(f"[+] Downloading {name}'s data ...")
	try:
		with open(db_path) as f: db = json.load(f)
		dispatch[key](db)
	except Exception as err:
		print(f'[-] Error: {err}')
		return False
	print(f"[+] Porting {name}'s data ...")
	tmp = db_path + '.tmp'
	try:
		with open(tmp, 'w') as f:
			json.dump(db, f, separators=(',', ':'))
		os.replace(tmp, db_path)
	except BaseException as err:
		if os.path.exists(tmp): os.remove(tmp)
		print(f'[-] Error: {err}')
		return False
	try: DBLoad.cache_clear()
	except Exception: pass
	print('[+] Done')
	return True
