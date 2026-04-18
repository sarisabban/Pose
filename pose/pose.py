#!/usr/bin/env python3

import os
import re
import sys
import json
import math
import copy
import datetime
import numpy as np
from collections import defaultdict

class Pose():
	''' A class that builds and manipulates protein, DNA, and RNA '''
	def __init__(self):
		path, modulename = os.path.split(__file__)
		with open(f'{path}/database.json') as f: database = json.load(f)
		self.aminoacids=database['Amino Acids']
		self.nucleotides=database['Nucleotides']
		self.probbatoms = {
			'N', 'H', '1H', '2H', '3H', 'H1', 'H2', 'H3',
			'HT1', 'HT2', 'HT3', 'HN',
			'CA', 'HA', 'HA1', 'HA2', 'HA3',
			'C', 'O', 'OXT', 'OT1', 'OT2'}
		self.nucbbatoms = {
			'P'  , 'OP1', 'OP2', "O5'", "C5'", "H5'", "H5''", "C4'",
			"H4'", "O4'", "C3'", "H3'", "C2'", "H2'", "H2''", "C1'",
			"H1'", "O3'", "O2'", "HO2'"}
		self.masses = {
			'H' :1.008  , 'He':4.003  , 'Li':6.941  , 'Be':9.012  ,
			'B' :10.811 , 'C' :12.011 , 'N' :14.007 ,  'O':15.999 ,
			'F' :18.998 , 'Ne':20.180 , 'Na':22.990 , 'Mg':24.305 ,
			'Al':26.982 , 'Si':28.086 , 'P' :30.974 ,  'S':32.066 ,
			'Cl':35.453 , 'Ar':39.948 , 'K' :39.098 , 'Ca':40.078 ,
			'Sc':44.956 , 'Ti':47.867 , 'V' :50.942 , 'Cr':51.996 ,
			'Mn':54.938 , 'Fe':55.845 , 'Co':58.933 , 'Ni':58.693 ,
			'Cu':63.546 , 'Zn':65.38  , 'Ga':69.723 , 'Ge':72.631 ,
			'As':74.922 , 'Se':78.971 , 'Br':79.904 , 'Kr':84.798 ,
			'Rb':84.468 , 'Sr':87.62  , 'Y' :88.906 , 'Zr':91.224 ,
			'Nb':92.906 , 'Mo':95.95  , 'Tc':98.907 , 'Ru':101.07 ,
			'Rh':102.906, 'Pd':106.42 , 'Ag':107.868, 'Cd':112.414,
			'In':114.818, 'Sn':118.711, 'Sb':121.760, 'Te':126.7  ,
			'I' :126.904, 'Xe':131.294, 'Cs':132.905, 'Ba':137.328,
			'La':138.905, 'Ce':140.116, 'Pr':140.908, 'Nd':144.243,
			'Pm':144.913, 'Sm':150.36 , 'Eu':151.964, 'Gd':157.25 ,
			'Tb':158.925, 'Dy':162.500, 'Ho':164.930, 'Er':167.259,
			'Tm':168.934, 'Yb':173.055, 'Lu':174.967, 'Hf':178.49 ,
			'Ta':180.948, 'W' :183.84 , 'Re':186.207, 'Os':190.23 ,
			'Ir':192.217, 'Pt':195.085, 'Au':196.967, 'Hg':200.592,
			'Tl':204.383, 'Pb':207.2  , 'Bi':208.980, 'Po':208.982,
			'At':209.987, 'Rn':222.081, 'Fr':223.020, 'Ra':226.025,
			'Ac':227.028, 'Th':232.038, 'Pa':231.036, 'U' :238.029,
			'Np':237    , 'Pu':244}
		self.data = {'Type':None, 'Energy':0, 'Rg':0, 'Mass':0, 'Size':{},
		'FASTA':{}, 'SS':{}, 'Nucleotides':None, 'Amino Acids':None,
		'Atoms':{}, 'Bonds':{}, 'BondOrders':{},
		'Coordinates':np.zeros((0, 3))}
	def _rotmat(self, theta, u):
		''' Rotate a matrix around axis u by theta angle '''
		ux, uy, uz = u[0], u[1], u[2]
		S = math.sin(math.radians(theta))
		C = math.cos(math.radians(theta))
		R = np.array([
		[C+ux**2*(1-C)   , ux*uy*(1-C)-uz*S, ux*uz*(1-C)+uy*S],
		[uy*ux*(1-C)+uz*S, C+uy**2*(1-C)   , uy*uz*(1-C)-ux*S],
		[uz*ux*(1-C)-uy*S, uz*uy*(1-C)+ux*S, C+uz**2*(1-C)   ]])
		return R
	def _atomiter(self):
		''' Yield (atom, coord, res_idx) for all atoms '''
		src = (self.data['Amino Acids'] or self.data['Nucleotides'])
		At = self.data['Atoms']
		Co = self.data['Coordinates']
		for ri, info in src.items():
			for ai in info[2] + info[3]:
				yield (ai, At[ai]), Co[ai], ri
	def _bondtreefused(self, BB, SC):
		''' Construct bond graph for when a sidechain is fused to a backbone '''
		BBb = {int(k): v for k, v in
			copy.deepcopy(self.aminoacids[BB]['Bonds']).items()}
		SCb = {int(k): v for k, v in
			copy.deepcopy(self.aminoacids[SC]['Bonds']).items()}
		length = len(SCb)
		BBb.pop(1)
		nBBb = {}
		for i, a in enumerate(BBb.items()):
			v = a[1]
			v = [x if x==0 else x-1 for x in v]
			nBBb[i] = v
		if BB == 'Backbone' or BB == 'Backbone start':
			n = 3
			nBBb[0][0] = length+1
		else:
			n = 1
			nBBb[0][0] = length-1
		BBb = nBBb
		for i in reversed(range(len(BBb))):
			if i > n+1:
				newvals = [x if x<=n else x+length-1 for x in BBb[i]]
				newk = i+length-1
				del BBb[i]
				BBb[newk] = newvals
		BBb[n].append(n+1+length)
		for i, (k, v) in enumerate(SCb.items(), start=n+2):
			if k < 0: break
			k = i
			if BB == 'Backbone' or BB == 'Backbone start': v=[x+n+2 for x in v]
			else: v = [x+n+4 if x<0 else x+n+2 for x in v]
			if i == n+2: v.append(n)
			BBb[k] = sorted(v)
		return BBb
	def _bondtreenotfused(self, BB, SC):
		''' Construct amino acid bond graph by adding sidechain to backbone '''
		SC = SC.upper()
		if self.aminoacids[SC]['Fused']: return self._bondtreefused(BB, SC)
		BBb = {int(k): v for k, v in
			copy.deepcopy(self.aminoacids[BB]['Bonds']).items()}
		SCb = {int(k): v for k, v in
			copy.deepcopy(self.aminoacids[SC]['Bonds']).items()}
		length = len(SCb)
		if BB == 'Backbone' or BB == 'Backbone start': n = 4
		else: n = 2
		for i in reversed(range(len(BBb))):
			if i > n+1:
				newvals = [x if x<=n
					else x+length
					for x in BBb[i]]
				newk = i+length
				del BBb[i]
				BBb[newk] = newvals
		BBb[n].append(n + 2 + length)
		for i, (k, v) in enumerate(SCb.items(), start=n+2):
			k = i
			if length != 1:
				v = [x+n+2 for x in v]
				if i == n+2: v.append(n)
			else: v = [x+n+1 for x in v]
			BBb[k] = sorted(v)
		return BBb
	def _bondtree(self, BB, AA, new_chain=False):
		''' Update the pose bond graph when adding a new amino acid '''
		BBb = self._bondtreenotfused(BB, AA)
		BT = self.data['Bonds']
		BO = self.data.setdefault('BondOrders', {})
		aa_db = self.aminoacids
		aa_u = AA.upper()
		bb_atoms_all = [a[0] for a in aa_db[BB]['Backbone Atoms']]
		sc_atoms_all = [a[0] for a in aa_db[aa_u]['Sidechain Atoms']]
		name_bo = {}
		for k_str, nbrs in aa_db[BB]['Bonds'].items():
			k = int(k_str)
			orders = aa_db[BB]['BondOrders'][k_str]
			if k < 0 or k >= len(bb_atoms_all): continue
			nm_i = bb_atoms_all[k]
			for nb, bo in zip(nbrs, orders):
				if nb < 0 or nb >= len(bb_atoms_all): continue
				name_bo[(nm_i, bb_atoms_all[nb])] = bo
		for k_str, nbrs in aa_db[aa_u]['Bonds'].items():
			k = int(k_str)
			orders = aa_db[aa_u]['BondOrders'][k_str]
			if k < 0 or k >= len(sc_atoms_all): continue
			nm_i = sc_atoms_all[k]
			for nb, bo in zip(nbrs, orders):
				if nb == -5:
					name_bo[(nm_i, 'N')] = bo
					name_bo[('N', nm_i)] = bo
				elif nb >= 0 and nb < len(sc_atoms_all):
					name_bo[(nm_i, sc_atoms_all[nb])] = bo
		length = len(BT)
		idx_to_name = {}
		for i in range(len(BBb)):
			gi = length + i
			if gi in self.data['Atoms']:
				idx_to_name[i] = self.data['Atoms'][gi][0]
		BOb = {}
		for k, nbrs in BBb.items():
			if k < 0:
				BOb[k] = [1] * len(nbrs)
				continue
			nm_i = idx_to_name.get(k)
			orders = []
			for nb in nbrs:
				if nb < 0:
					orders.append(1); continue
				nm_j = idx_to_name.get(nb)
				orders.append(name_bo.get((nm_i, nm_j), 1))
			BOb[k] = orders
		if length == 0:
			self.data['Bonds'] = BBb
			self.data['BondOrders'] = BOb
			return
		i_max = max(BT)
		if not new_chain:
			BT[i_max-1] += [i_max+1]
			BO.setdefault(i_max-1, []).append(1.5)
		for i in range(len(BBb)):
			K = i+length
			V = [x+length for x in BBb[i]]
			OV = list(BOb[i])
			if i == 0 and not new_chain:
				V.append(i_max-1)
				OV.append(1.5)
			BT[K] = V
			BO[K] = OV
		self.data['Bonds'] = BT
		self.data['BondOrders'] = BO
	def _downstreamatoms(self, res1, atom1, res2, atom2):
		''' Atom indices on the atom2 side of the (atom1, atom2) pivot edge.
		    BFS over the pose bond graph, refusing to cross the pivot bond. '''
		Ai = self.GetAtomIdx(res1, atom1)
		Bi = self.GetAtomIdx(res2, atom2)
		bonds = self.data['Bonds']
		result = {Bi}
		stack = [Bi]
		while stack:
			cur = stack.pop()
			for nb in bonds.get(cur, []):
				if nb == Ai: continue
				if nb in result: continue
				result.add(nb)
				stack.append(nb)
		return result
	def _prevres(self, i):
		''' Previous residue on same chain, or None '''
		src = (self.data['Amino Acids'] or self.data['Nucleotides'])
		if i-1 in src and src[i-1][1] == src[i][1]: return i - 1
		return None
	def _nextres(self, i):
		''' Next residue on same chain, or None '''
		src = (self.data['Amino Acids'] or self.data['Nucleotides'])
		if i+1 in src and src[i+1][1] == src[i][1]: return i + 1
		return None
	def _hasatom(self, res, atom):
		''' Check if atom exists in a residue '''
		source = (self.data['Amino Acids'] or self.data['Nucleotides'])
		info = source[res]
		return any(self.data['Atoms'][i][0] == atom for i in info[2] + info[3])
	def _phosphobonds(self, nt_ch):
		''' Add O3'-P phosphodiester bonds '''
		At = self.data['Atoms']
		Bd = self.data['Bonds']
		BO = self.data.setdefault('BondOrders', {})
		nts = self.data['Nucleotides']
		for ids in nt_ch.values():
			for i in range(len(ids) - 1):
				o3 = next((a for a in nts[ids[i]][2] if At[a][0] == "O3'"),None)
				p = next((a for a in nts[ids[i+1]][2] if At[a][0] == 'P'), None)
				if o3 is None or p is None: continue
				Bd.setdefault(o3, [])
				Bd.setdefault(p, [])
				BO.setdefault(o3, [])
				BO.setdefault(p, [])
				if p not in Bd[o3]:
					Bd[o3].append(p)
					BO[o3].append(1)
				if o3 not in Bd[p]:
					Bd[p].append(o3)
					BO[p].append(1)
	def _buildprotein(self, sequence, chain):
		''' Build one protein chain, append to data '''
		is_new = (self.data['Type'] is None or self.data['Amino Acids'] is None)
		if is_new:
			self.data['Type'] = 'Protein'
			self.data['Amino Acids'] = {}
			self.data['Nucleotides'] = None
			self.data['Atoms'] = {}
			self.data['Bonds'] = {}
			self.data['BondOrders'] = {}
			self.data['Coordinates'] = np.zeros((0, 3))
		aa_db = self.aminoacids
		Eadj = (0.400, 1.472, 0)
		Oadj = (0.812, 0.940, 0)
		n = len(sequence)
		aa_start = len(self.data['Amino Acids'])
		atom_start = len(self.data['Atoms'])
		new_coords = []
		X, Y, Z = 0, 0, 0
		for i, aa in enumerate(sequence):
			LD = aa.islower()
			last = i == n - 1
			odd = (i % 2) != 0
			if n == 1 or i == 0:
				bb = 'Backbone' if n == 1 else 'Backbone start'
				idx = 6
				flip = False
			else:
				bb = 'Backbone end' if last else 'Backbone middle'
				adj = Eadj if odd else Oadj
				prev = new_coords[-1][-2]
				X = prev[0] + adj[0]
				Y = prev[1] + adj[1]
				Z = prev[2] + adj[2]
				idx = 4
				flip = odd
			T = np.array([X, Y, Z])
			BB = np.array(aa_db[bb]['Vectors']) + T
			SC = np.array(aa_db[aa.upper()]['Vectors']) + T
			AA_co = np.insert(BB, [idx], SC, axis=0)
			if LD: AA_co = AA_co * [1, 1, -1]
			if flip:
				AA = AA_co
				p = AA[2]
				AA = AA - p
				H1, H2 = AA[3], AA[4]
				u = np.cross(H1, H2)
				nu = np.linalg.norm(u)
				if nu >= 1e-10:
					u = u / nu
					TM = np.array([
						[2*u[0]**2-1, 2*u[0]*u[1], 2*u[0]*u[2]],
						[2*u[0]*u[1], 2*u[1]**2-1, 2*u[1]*u[2]],
						[2*u[0]*u[2], 2*u[1]*u[2], 2*u[2]**2-1]])
					AA_co = np.matmul(AA, TM) + p
			if self.aminoacids[aa.upper()]['Fused']:
				AA_co = np.delete(AA_co, [1], axis=0)
			new_coords.append(AA_co)
		all_co = np.concatenate(new_coords)
		I = atom_start
		for i, aa in enumerate(sequence):
			LD = aa.islower()
			aa_u = aa.upper()
			if n == 1 or i == 0:
				bb = 'Backbone' if n == 1 else 'Backbone start'
				bb_idx = 6
			else:
				bb = 'Backbone end' if i == n-1 else 'Backbone middle'
				bb_idx = 4
			bb_atoms = aa_db[bb]['Backbone Atoms'][:bb_idx]
			if self.aminoacids[aa.upper()]['Fused']:
				bb_atoms = [b for j, b in enumerate(bb_atoms) if j != 1]
			sc_atoms = aa_db[aa_u]['Sidechain Atoms']
			tail = aa_db[bb]['Backbone Atoms'][bb_idx:]
			full = bb_atoms + sc_atoms + tail
			BBi, SCi = [], []
			for v in full:
				self.data['Atoms'][I] = [v[0], v[1], v[2], v[3], v[4]]
				if v[0] in self.probbatoms: BBi.append(I)
				else: SCi.append(I)
				I += 1
			tri = aa_db[aa_u]['Tricode']
			if LD: tri = 'D' + tri[1:]
			ai = aa_start + i
			self.data['Amino Acids'][ai] = [aa, chain, BBi, SCi, 'L', tri, 0]
		self.data['Coordinates'] = \
			np.append(self.data['Coordinates'], all_co, axis=0)
		for i, aa in enumerate(sequence):
			ai = aa_start + i
			if n == 1 or i == 0:
				bb = 'Backbone' if n == 1 else 'Backbone start'
			else:
				bb = 'Backbone end' if i == n-1 else 'Backbone middle'
			new_ch = (i == 0 and aa_start > 0)
			self._bondtree(bb, aa, new_chain=new_ch)
	def _buildnucleotide(self, sequence, fmt, chains=None):
		''' Build DNA or RNA (single or double strand) '''
		sequence = sequence.upper()
		N = len(sequence)
		if fmt == 'DNA':
			comp = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}
			tri = {'A':'DA', 'T':'DT', 'G':'DG', 'C':'DC'}
			rise, twist = 3.375, 36.0
		else:
			comp = {'A':'U', 'U':'A', 'G':'C', 'C':'G'}
			tri = {'A':'A', 'U':'U', 'G':'G', 'C':'C'}
			rise, twist = 2.548, 32.7
		Rflip = np.diag([1.0, -1.0, -1.0])
		is_append = (
			self.data.get('Type') == fmt
			and self.data.get('Nucleotides') is not None)
		if not is_append:
			self.data = {
				'Type': fmt, 'Energy': 0,
				'Rg': 0, 'Mass': 0, 'Size': {},
				'FASTA': {}, 'SS': {},
				'Nucleotides': {},
				'Amino Acids': None,
				'Atoms': {}, 'Bonds': {}, 'BondOrders': {},
				'Coordinates': np.zeros((0, 3))}
			Co = []
			ai, ni = 0, 0
		else:
			Co = [row for row in self.data['Coordinates']]
			ai = max(self.data['Atoms'].keys()) + 1 \
				if self.data['Atoms'] else 0
			ni = max(self.data['Nucleotides'].keys()) + 1 \
				if self.data['Nucleotides'] else 0
		nt_ch = defaultdict(list)
		used = {v[1] for v in self.data['Nucleotides'].values()} \
			if is_append else set()
		if chains is None:
			wanted_a, want_duplex = 'A', True
		elif isinstance(chains, str):
			wanted_a, want_duplex = chains, True
		else:
			wanted_a = chains[0] if chains else 'A'
			want_duplex = len(chains) > 1
		alphabet = list(
			'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
		if wanted_a is not None and wanted_a not in used:
			chain_a = wanted_a
		else:
			chain_a = next((c for c in alphabet if c not in used), None)
			if chain_a is None: raise Exception('Out of chain letters')
		used.add(chain_a)
		if want_duplex:
			chain_b = next((c for c in alphabet if c not in used), None)
			if chain_b is None: raise Exception('Out of chain letters')
		else:
			chain_b = None
		for i, base in enumerate(sequence):
			tricode, chain, k, flip = tri[base], chain_a, i, False
			db = self.nucleotides[tricode]
			vecs = np.array(db['Vectors'])
			bbm = db['Backbone Atoms']
			bsm = db['Base Atoms']
			n_bb = len(bbm)
			deg = -k * twist
			t = math.radians(deg)
			c, s = math.cos(t), math.sin(t)
			R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
			T = np.array([0., 0., -k * rise])
			if flip: tr = (vecs @ Rflip.T) @ R.T + T
			else: tr = vecs @ R.T + T
			ltg = {}
			bbi, bsi = [], []
			for li in range(n_bb):
				am = bbm[li]
				self.data['Atoms'][ai] = [am[0], am[1], am[2], 1.0, 0.0]
				Co.append(tr[li])
				ltg[li] = ai
				bbi.append(ai)
				ai += 1
			for li2, am in enumerate(bsm):
				li = n_bb + li2
				self.data['Atoms'][ai] = [am[0], am[1], am[2], 1.0, 0.0]
				Co.append(tr[li])
				ltg[li] = ai
				bsi.append(ai)
				ai += 1
			bd = self.data['Bonds']
			bo = self.data['BondOrders']
			db_bo = db.get('BondOrders', {})
			for ks, vl in db['Bonds'].items():
				gi = ltg.get(int(ks), -1)
				if gi == -1: continue
				bd.setdefault(gi, [])
				bo.setdefault(gi, [])
				orders = db_bo.get(ks, [1] * len(vl))
				for lj, lo in zip(vl, orders):
					gj = ltg.get(lj, -1)
					if gj != -1 and gj not in bd[gi]:
						bd[gi].append(gj)
						bo[gi].append(lo)
			sym = tricode[-1]
			self.data['Nucleotides'][ni] = [sym, chain, bbi, bsi, tricode]
			nt_ch[chain].append(ni)
			ni += 1
		if want_duplex:
			comp_seq = ''.join(comp[b] for b in reversed(sequence))
			for j, base in enumerate(comp_seq):
				tricode, chain, k, flip = tri[base], chain_b, N-1-j, True
				db = self.nucleotides[tricode]
				vecs = np.array(db['Vectors'])
				bbm = db['Backbone Atoms']
				bsm = db['Base Atoms']
				n_bb = len(bbm)
				deg = -k * twist
				t = math.radians(deg)
				c, s = math.cos(t), math.sin(t)
				R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
				T = np.array([0., 0., -k * rise])
				if flip: tr = (vecs @ Rflip.T) @ R.T + T
				else: tr = vecs @ R.T + T
				ltg = {}
				bbi, bsi = [], []
				for li in range(n_bb):
					am = bbm[li]
					self.data['Atoms'][ai] = [am[0], am[1], am[2], 1.0, 0.0]
					Co.append(tr[li])
					ltg[li] = ai
					bbi.append(ai)
					ai += 1
				for li2, am in enumerate(bsm):
					li = n_bb + li2
					self.data['Atoms'][ai] = [am[0], am[1], am[2], 1.0, 0.0]
					Co.append(tr[li])
					ltg[li] = ai
					bsi.append(ai)
					ai += 1
				bd = self.data['Bonds']
				bo = self.data['BondOrders']
				db_bo = db.get('BondOrders', {})
				for ks, vl in db['Bonds'].items():
					gi = ltg.get(int(ks), -1)
					if gi == -1: continue
					bd.setdefault(gi, [])
					bo.setdefault(gi, [])
					orders = db_bo.get(ks, [1] * len(vl))
					for lj, lo in zip(vl, orders):
						gj = ltg.get(lj, -1)
						if gj != -1 and gj not in bd[gi]:
							bd[gi].append(gj)
							bo[gi].append(lo)
				sym = tricode[-1]
				self.data['Nucleotides'][ni] = [sym, chain, bbi, bsi, tricode]
				nt_ch[chain].append(ni)
				ni += 1
		self.data['Coordinates'] = (
			np.array(Co) if Co else np.zeros((0, 3)))
		self._phosphobonds(nt_ch)
	def _update(self):
		''' Update cached properties after structural changes '''
		self.CalcMass()
		self.CalcSize()
		self.CalcFASTA()
		self.CalcRg()
		if self.data['Type'] == 'Protein': self.CalcDSSP()
		else: self.data['SS'] = {}
	def CalcMass(self):
		''' Calculate mass of peptide in Da'''
		ids = sorted(self.data['Atoms'].keys())
		mass = round(sum(
			self.masses.get(self.data['Atoms'][i][1], 0.0) for i in ids), 3)
		self.data['Mass'] = mass
	def CalcSize(self):
		''' Calculate length of each chain '''
		source = (self.data['Amino Acids'] or self.data['Nucleotides'])
		if not source:
			self.data['Size'] = {}
			return
		size = {}
		for v in source.values(): size[v[1]] = size.get(v[1], 0) + 1
		self.data['Size'] = size
	def CalcFASTA(self):
		''' Return per-chain FASTA dict '''
		source = (self.data['Amino Acids'] or self.data['Nucleotides'])
		if not source:
			self.data['FASTA'] = {}
			return
		fasta = {}
		for v in source.values(): fasta.setdefault(v[1], []).append(v[0])
		self.data['FASTA'] = {k: ''.join(v) for k, v in fasta.items()}
	def CalcRg(self):
		''' Calculate the radius of gyration of a peptide '''
		ids   = sorted(self.data['Atoms'].keys())
		mass  = np.array([
			self.masses.get(self.data['Atoms'][i][1], 0.0) for i in ids])
		tmass = mass.sum()
		if tmass == 0: raise ZeroDivisionError('No atoms in pose')
		coord = self.data['Coordinates'][np.array(ids)] if ids \
			else np.zeros((0, 3))
		xm    = coord * mass[:, np.newaxis]
		rr    = np.sum(coord * xm)
		mm    = np.sum((xm.sum(0) / tmass) ** 2)
		rg    = round(math.sqrt(max(0.0, rr/tmass - mm)), 3)
		self.data['Rg'] = rg
	def CalcCharge(self, iterations=6):
		''' Calculate Gasteiger-Marsili partial charges to all atoms '''
		PARAMS = {
			'C3':(7.98,  9.18,  1.88), 'C2':(8.79,  9.32,  1.51),
			'C1':(10.39, 9.45,  0.73), 'H' :(7.17,  6.24, -0.56),
			'O3':(14.18, 12.92, 1.39), 'O2':(17.07, 13.79, 0.47),
			'N3':(11.54, 10.82, 1.36), 'N2':(12.87, 11.15, 0.85),
			'S' :(10.14, 9.13,  1.38), 'Se':(9.00,  8.00,  1.10),
			'F' :(14.66, 13.85, 2.31), 'Cl':(11.00, 9.69,  1.35),
			'Br':(10.08, 8.47,  1.16), 'I' :(9.90,  7.96,  0.96),
			'P' :(8.90,  8.24,  0.96), 'B' :(5.80,  6.00,  1.56)}
		ids  = sorted(self.data['Atoms'].keys())
		els  = [self.data['Atoms'][i][1].upper() for i in ids]
		id_set = set(ids)
		bonds = {i: [j for j in self.data['Bonds'].get(i, []) if j in id_set]
			for i in ids}
		bond_orders = self.data.get('BondOrders', {})
		sp2 = set()
		for i in ids:
			for bo in bond_orders.get(i, [1.0] * len(bonds[i])):
				if bo >= 1.5:
					sp2.add(i)
					break
		charges = {i: self.data['Atoms'][i][2] for i in ids}
		atype = {}
		for ii, i in enumerate(ids):
			el = els[ii]
			nb = len(bonds[i])
			isp2 = i in sp2
			if   el == 'H':   types = 'H'
			elif el == 'S':   types = 'S'
			elif el == 'SE':  types = 'Se'
			elif el == 'C':
				if isp2:      types = 'C2'
				elif nb <= 2: types = 'C1'
				else:         types = 'C3'
			elif el == 'N':   types = 'N2' if isp2 else 'N3'
			elif el == 'O':   types = 'O2' if isp2 else 'O3'
			else:             types = 'C3'
			atype[i] = PARAMS[types]
		for n in range(iterations):
			damp = 1.0 / (2 ** (n + 1))
			chi = {i: a + q*(b + c*q) for i in ids
				for a, b, c in [atype[i]] for q in [charges[i]]}
			delta = {i: 0.0 for i in ids}
			for i in ids:
				for j in bonds[i]:
					if j <= i: continue
					if chi[j] >= chi[i]: donor, acceptor = i, j
					else: donor, acceptor = j, i
					a, b, c = atype[donor]
					ip = a + b + c
					if ip == 0: continue
					dq = damp * (chi[acceptor] - chi[donor]) / ip
					delta[donor]    += dq
					delta[acceptor] -= dq
			for i in ids: charges[i] += delta[i]
		for i in ids: self.data['Atoms'][i][2] = round(charges[i], 4)
	def CalcDSSP(self):
		''' Assign secondary structures to each amino acid '''
		if self.data['Amino Acids'] is None:
			raise Exception('No protein loaded. ' 'Call Import() first')
		N = len(self.data['Amino Acids'])
		if N == 0:
			self.data['SS'] = {}
			return
		AAs      = self.data['Amino Acids']
		chains   = [AAs[i][1]         for i in range(N)]
		tricodes = [AAs[i][5].upper() for i in range(N)]
		At = self.data['Atoms']
		Co = self.data['Coordinates']
		N_xyz  = np.full((N, 3), np.nan)
		CA_xyz = np.full((N, 3), np.nan)
		C_xyz  = np.full((N, 3), np.nan)
		O_xyz  = np.full((N, 3), np.nan)
		H_atom_xyz = np.full((N, 3), np.nan)
		one_H_xyz  = np.full((N, 3), np.nan)
		has_atom = [set() for _ in range(N)]
		for i in range(N):
			info = AAs[i]
			for ai in info[2] + info[3]:
				nm = At[ai][0]
				has_atom[i].add(nm)
				if   nm == 'N':  N_xyz[i]  = Co[ai]
				elif nm == 'CA': CA_xyz[i] = Co[ai]
				elif nm == 'C':  C_xyz[i]  = Co[ai]
				elif nm == 'O':  O_xyz[i]  = Co[ai]
				elif nm == 'H':  H_atom_xyz[i] = Co[ai]
				elif nm == '1H': one_H_xyz[i] = Co[ai]
		H_pos = [None] * N
		for i in range(N):
			if tricodes[i] == 'PRO': continue
			if i == 0 or chains[i] != chains[i-1]: continue
			if 'H' in has_atom[i]:
				H_pos[i] = H_atom_xyz[i]
			elif '1H' in has_atom[i]:
				H_pos[i] = one_H_xyz[i]
			elif ('N' in has_atom[i]
			and 'C' in has_atom[i-1]
			and 'O' in has_atom[i-1]):
				Ni = N_xyz[i]
				Cp = C_xyz[i-1]
				Op = O_xyz[i-1]
				co = Cp - Op
				nm = np.linalg.norm(co)
				if nm > 0.001: H_pos[i] = Ni + (co / nm)
		has_H_mask = np.array([h is not None for h in H_pos])
		H_arr = np.full((N, 3), np.nan)
		for i in range(N):
			if H_pos[i] is not None: H_arr[i] = H_pos[i]
		chain_arr = np.array(chains)
		same_chain = chain_arr[:, None] == chain_arr[None, :]
		idx = np.arange(N)
		adj_mask = np.abs(idx[:, None] - idx[None, :]) <= 1
		has_O = ~np.isnan(O_xyz[:, 0])
		has_C = ~np.isnan(C_xyz[:, 0])
		valid = (has_H_mask[:, None]
			& has_O[None, :]
			& has_C[None, :]
			& same_chain
			& ~adj_mask)
		Ni_b = N_xyz[:, None, :]
		Hi_b = H_arr[:, None, :]
		Oj_b = O_xyz[None, :, :]
		Cj_b = C_xyz[None, :, :]
		with np.errstate(invalid='ignore'):
			r_ON = np.linalg.norm(Oj_b - Ni_b, axis=2)
			r_CH = np.linalg.norm(Cj_b - Hi_b, axis=2)
			r_OH = np.linalg.norm(Oj_b - Hi_b, axis=2)
			r_CN = np.linalg.norm(Cj_b - Ni_b, axis=2)
		with np.errstate(divide='ignore', invalid='ignore'):
			E_mat = 0.084 * 332 * (
				1.0/r_ON + 1.0/r_CH - 1.0/r_OH - 1.0/r_CN)
		near_zero = (
			(r_ON < 0.001) | (r_CH < 0.001)
			| (r_OH < 0.001) | (r_CN < 0.001))
		E_mat = np.where(valid & ~near_zero & np.isfinite(E_mat), E_mat, 0.0)
		hbond_pairs = np.argwhere(E_mat < -0.5)
		hbond = {(int(i), int(j)) for i, j in hbond_pairs}
		turn3 = [i+3 < N and (i+3, i) in hbond for i in range(N)]
		turn4 = [i+4 < N and (i+4, i) in hbond for i in range(N)]
		turn5 = [i+5 < N and (i+5, i) in hbond for i in range(N)]
		ss = ['L'] * N
		for i in range(N):
			for turn, span in [(turn3,4),(turn4,5),(turn5,6)]:
				if turn[i]:
					for k in range(i, min(i+span, N)):
						if ss[k] == 'L': ss[k] = 'T'
		for i in range(N - 1):
			if turn5[i] and turn5[i + 1]:
				for k in range(i + 1, min(i + 6, N)):
					if ss[k] not in ('H', 'E', 'B'): ss[k] = 'I'
		for i in range(N - 1):
			if turn4[i] and turn4[i + 1]:
				for k in range(i + 1, min(i + 5, N)):
					if ss[k] != 'I': ss[k] = 'H'
		for i in range(N - 1):
			if turn3[i] and turn3[i + 1]:
				for k in range(i + 1, min(i + 4, N)):
					if ss[k] not in ('H', 'E', 'B'): ss[k] = 'G'
		bridges = set()
		for i in range(1, N - 1):
			for k in range(i + 2, N - 1):
				if chains[i] != chains[k]: continue
				ap = (
					((i, k) in hbond
					and (k, i) in hbond)
					or (i > 0 and k < N - 1
					and (i - 1, k + 1) in hbond
					and (k - 1, i + 1) in hbond))
				pp = (
					(i > 0 and k < N - 1
					and (i - 1, k) in hbond
					and (k, i + 1) in hbond)
					or (k > 0 and i < N - 1
					and (k - 1, i) in hbond
					and (i, k + 1) in hbond))
				if ap or pp: bridges.add((i, k))
		br_at = defaultdict(set)
		for i, k in bridges:
			br_at[i].add(k)
			br_at[k].add(i)
		for r in br_at:
			if ss[r] not in ('H', 'G', 'I'): ss[r] = 'B'
		changed = True
		while changed:
			changed = False
			for i in range(N - 1):
				if (ss[i] not in ('B', 'E')
				or ss[i+1] not in ('B', 'E')): continue
				for k1 in br_at.get(i, []):
					for k2 in br_at.get(i+1, []):
						if abs(k1 - k2) != 1: continue
						for r in (i, i+1, k1, k2):
							if ss[r] in ('H', 'G', 'I'): continue
							if ss[r] != 'E':
								ss[r] = 'E'
								changed = True
		for i in range(2, N - 2):
			if ss[i] != 'L': continue
			if (chains[i-2] != chains[i]
			or chains[i+2] != chains[i]): continue
			m2 = CA_xyz[i - 2]
			ci = CA_xyz[i]
			p2 = CA_xyz[i + 2]
			v1 = ci - m2
			v2 = p2 - ci
			n1 = np.linalg.norm(v1)
			n2 = np.linalg.norm(v2)
			if n1 < 0.001 or n2 < 0.001: continue
			cos_k = np.dot(v1, v2) / (n1 * n2)
			cos_k = max(-1.0, min(1.0, cos_k))
			kappa = math.acos(cos_k) * 180.0 / math.pi
			if kappa >= 70.0: ss[i] = 'S'
		PHI_LO, PHI_HI = -104.0, -46.0
		PSI_LO, PSI_HI =  116.0, 174.0
		ppii = [False] * N
		for i in range(N):
			if ss[i] != 'L': continue
			phi = self.GetDihedral(i, 'PHI')
			psi = self.GetDihedral(i, 'PSI')
			ppii[i] = PHI_LO <= phi <= PHI_HI and PSI_LO <= psi <= PSI_HI
		for i in range(N - 2):
			if ppii[i] and ppii[i + 1] and ppii[i + 2]:
				for k in range(i, i + 3):
					if ss[k] == 'L': ss[k] = 'P'
		for i in range(N): AAs[i][4] = ss[i]
		if self.data['Amino Acids'] is None: return {}
		ss = {}
		for v in self.data['Amino Acids'].values():
			ss.setdefault(v[1], []).append(v[4])
		ss = {k: ''.join(v) for k, v in ss.items()}
		self.data['SS'] = ss
	def CalcSASA(self, n_points=960, probe_radius=1.4):
		''' Calculate SASA per residue '''
		if self.data['Amino Acids'] is None:
			raise Exception('No protein loaded. Call Import() first')
		VDW = {
			'H':  1.20, 'C':  1.70, 'N':  1.55, 'O':  1.52,
			'S':  1.80, 'SE': 1.90, 'P':  1.80, 'F':  1.47,
			'CL': 1.75, 'BR': 1.85, 'I':  1.98, 'FE': 1.80,
			'ZN': 1.39, 'MG': 1.73, 'CA': 1.97, 'MN': 1.73,
			'CU': 1.40, 'NI': 1.63}
		DEFAULT_VDW = 2.00
		atoms  = self.data['Atoms']
		coords = self.data['Coordinates']
		ids    = sorted(atoms.keys())
		id_map = {ai: ii for ii, ai in enumerate(ids)}
		n      = len(ids)
		c      = coords[np.array(ids)]
		radii  = np.array([
		VDW.get(atoms[i][1].upper(),DEFAULT_VDW)+probe_radius for i in ids])
		golden = (1 + np.sqrt(5)) / 2
		pts    = np.arange(n_points)
		theta  = np.arccos(1 - 2 * (pts + 0.5) / n_points)
		phi    = 2 * np.pi * pts / golden
		st     = np.sin(theta)
		sph    = np.column_stack([st*np.cos(phi),st*np.sin(phi),np.cos(theta)])
		dm     = np.sqrt(((c[:, None, :] - c[None, :, :])**2).sum(2))
		atom_sasa = np.zeros(n)
		for ii in range(n):
			ri       = radii[ii]
			test_pts = c[ii] + ri * sph
			nbr_mask = (dm[ii] < ri + radii) & (dm[ii] > 0)
			nbrs     = np.where(nbr_mask)[0]
			if len(nbrs) == 0:
				atom_sasa[ii] = 4 * np.pi * ri**2
				continue
			diff_pts = (test_pts[:, None, :] - c[nbrs][None, :, :])
			dist_sq  = (diff_pts**2).sum(2)
			r_sq     = radii[nbrs]**2
			buried   = (dist_sq < r_sq[None, :]).any(axis=1)
			n_exp    = int((~buried).sum())
			atom_sasa[ii] = (n_exp / n_points * 4 * np.pi * ri**2)
		for aa_idx, aa_info in self.data['Amino Acids'].items():
			all_atoms = aa_info[2] + aa_info[3]
			sasa = sum(atom_sasa[id_map[a]] for a in all_atoms if a in id_map)
			self.data['Amino Acids'][aa_idx][6] = round(float(sasa), 5)
	def GetDistance(self, res1, atom1, res2, atom2):
		''' Measure distance between any two atoms '''
		A = self.GetAtomCoord(res1, atom1)
		B = self.GetAtomCoord(res2, atom2)
		return np.linalg.norm(B - A)
	def GetAngle(self, AA1, atom1, AA2, atom2, AA3, atom3):
		''' Measure the angle between three atoms '''
		atom1 = self.GetAtomCoord(AA1, atom1)
		atom2 = self.GetAtomCoord(AA2, atom2)
		atom3 = self.GetAtomCoord(AA3, atom3)
		A = atom2 - atom1
		B = atom2 - atom3
		denom = np.linalg.norm(A) * np.linalg.norm(B)
		if denom < 1e-10: return 0.0
		cos_theta = np.dot(A, B) / denom
		cos_theta = max(-1.0, min(1.0, cos_theta))
		return math.degrees(math.acos(cos_theta))
	def GetAtomBonds(self, index1, index2):
		''' Get the atom pair that participate in a bond from their index '''
		bonds = self.data['Bonds'][index1]
		if index2 not in bonds:
			raise Exception('Requested two atoms are not bonded')
		A = self.data['Atoms']
		return [A[index1][0], A[index1][1], A[index2][0], A[index2][1]]
	def GetIdentity(self, index, item, charge=False):
		''' Identify an atom, atom charge, or amino acid given its index '''
		if item.upper() == 'ATOM':
			Atom = self.data['Atoms'][index]
			if charge: return Atom[2]
			else:      return Atom[0]
		elif item.upper() in ('RESIDUE', 'AMINO ACID'):
			if self.data['Amino Acids'] is None:
				raise Exception('No protein loaded')
			return self.data['Amino Acids'][index][0]
		elif item.upper() in ('NUCLEOTIDE', 'DNA', 'RNA'):
			if self.data['Nucleotides'] is None:
				raise Exception('No nucleotide loaded')
			return self.data['Nucleotides'][index][0]
		else: raise Exception('Incorrect item')
	def GetAtomCoord(self, res, atom):
		''' Get specific atom coordinates '''
		source = (self.data['Amino Acids'] or self.data['Nucleotides'])
		info = source[res]
		for i in info[2] + info[3]:
			if self.data['Atoms'][i][0] == atom:
				return self.data['Coordinates'][i]
		raise Exception(f'Atom {atom} not found in residue {res}')
	def GetAtomIdx(self, res, atom):
		''' Get atom coordinate index by name '''
		source = (self.data['Amino Acids'] or self.data['Nucleotides'])
		info = source[res]
		for i in info[2] + info[3]:
			if self.data['Atoms'][i][0] == atom: return i
		raise Exception(f'Atom {atom} not found in residue {res}')
	def GetAtomList(self, PDB=False):
		''' Return list of all the atoms '''
		idx = 0 if PDB else 1
		ids = sorted(self.data['Atoms'].keys())
		return [self.data['Atoms'][i][idx] for i in ids]
	def GetInfo(self):
		''' Print all basic info about a peptide '''
		print(f"Energy:\t\t\t{self.data['Energy']}")
		print(f"Mass:\t\t\t{self.data['Mass']:,} Da")
		print(f"Rg:\t\t\t{self.data['Rg']} Å")
		for i in self.data['FASTA'].items():
			print(f'Sequence:\t\tChain: {i[0]}\tFASTA: {i[1]}')
		for i in self.data['SS'].items():
			print(f'Secondary Structure:\tChain: {i[0]}\tDSSP: {i[1]}')
		for i in self.data['Size'].items():
			print(f'Size:\t\t\tChain: {i[0]}\tLength: {i[1]}')
	def GetDihedral(self, res, angle_type, chi_type=None):
		''' Get dihedral angles (phi/psi/omega/chi or alpha-zeta) '''
		at = angle_type.upper()
		mol = self.data['Type']
		prv = self._prevres(res)
		nxt = self._nextres(res)
		gc = self.GetAtomCoord
		if mol == 'Protein':
			if at == 'PHI':
				if prv is None: return float('nan')
				r1 = gc(prv, 'C')
				r2 = gc(res, 'N')
				r3 = gc(res, 'CA')
				r4 = gc(res, 'C')
			elif at == 'PSI':
				if nxt is None: return float('nan')
				r1 = gc(res, 'N')
				r2 = gc(res, 'CA')
				r3 = gc(res, 'C')
				r4 = gc(nxt, 'N')
			elif at == 'OMEGA':
				if nxt is None: return float('nan')
				r1 = gc(res, 'CA')
				r2 = gc(res, 'C')
				r3 = gc(nxt, 'N')
				r4 = gc(nxt, 'CA')
			elif at == 'CHI':
				assert chi_type is not None, 'Protein CHI needs chi_type'
				sym = self.data['Amino Acids'][res][0].upper()
				ca = self.aminoacids[sym]['Chi Angle Atoms'][chi_type-1]
				r1 = gc(res, ca[0])
				r2 = gc(res, ca[1])
				r3 = gc(res, ca[2])
				r4 = gc(res, ca[3])
			else:
				raise Exception(
					f'Unknown protein angle: '
					f'{angle_type}')
		elif mol in ('DNA', 'RNA'):
			if at == 'ALPHA':
				if prv is None: return float('nan')
				r1 = gc(prv, "O3'")
				r2 = gc(res, 'P')
				r3 = gc(res, "O5'")
				r4 = gc(res, "C5'")
			elif at == 'BETA':
				r1 = gc(res, 'P')
				r2 = gc(res, "O5'")
				r3 = gc(res, "C5'")
				r4 = gc(res, "C4'")
			elif at == 'GAMMA':
				r1 = gc(res, "O5'")
				r2 = gc(res, "C5'")
				r3 = gc(res, "C4'")
				r4 = gc(res, "C3'")
			elif at == 'DELTA':
				r1 = gc(res, "C5'")
				r2 = gc(res, "C4'")
				r3 = gc(res, "C3'")
				r4 = gc(res, "O3'")
			elif at == 'EPSILON':
				if nxt is None: return float('nan')
				r1 = gc(res, "C4'")
				r2 = gc(res, "C3'")
				r3 = gc(res, "O3'")
				r4 = gc(nxt, 'P')
			elif at == 'ZETA':
				if nxt is None: return float('nan')
				r1 = gc(res, "C3'")
				r2 = gc(res, "O3'")
				r3 = gc(nxt, 'P')
				r4 = gc(nxt, "O5'")
			elif at == 'CHI':
				tri = self.data['Nucleotides'][res][4]
				ca = self.nucleotides[tri]['Chi Angle Atoms']
				r1 = gc(res, ca[0])
				r2 = gc(res, ca[1])
				r3 = gc(res, ca[2])
				r4 = gc(res, ca[3])
			else:
				raise Exception(
					f'Unknown nucleotide angle'
					f': {angle_type}')
		else:
			raise Exception(
				'No structure loaded. '
				'Call Import() first')
		u1 = r2 - r1
		u2 = r3 - r2
		u3 = r4 - r3
		mag = np.linalg.norm(u2)
		c12 = np.cross(u1, u2)
		c23 = np.cross(u2, u3)
		a = np.dot(u2, np.cross(c12, c23))
		b = mag * np.dot(c12, c23)
		return math.atan2(a, b) * 180 / math.pi
	def Import(self, filename, chain=None, model=1):
		''' Import Protein/DNA/RNA from .pdb/.cif '''
		if isinstance(chain, str): chain = [chain]
		if chain is not None and not isinstance(chain, list):
			raise Exception(
				f'chain must be None or a list, got {type(chain).__name__}')
		ext = os.path.splitext(filename)[1].lower()
		if ext == '.pdb':
			with open(filename) as f: raw = f.readlines()
			rows, has_mdl, in_tgt = [], False, False
			found = []
			for line in raw:
				line = line.rstrip()
				if not line: continue
				rec = line[:6].strip()
				if rec == 'MODEL':
					has_mdl = True
					try: mnum = int(line.split()[1])
					except (IndexError, ValueError): mnum = len(found) + 1
					found.append(mnum)
					in_tgt = (mnum == model)
					continue
				if rec == 'ENDMDL':
					if in_tgt: break
					in_tgt = False
					continue
				if rec != 'ATOM': continue
				ch = line[21]
				if chain and ch not in chain: continue
				if has_mdl and not in_tgt: continue
				a = line[12:16].strip()
				alt = line[16:17] if len(line) > 16 else ' '
				e = line[76:78].strip()
				if not e:
					s = a.lstrip('0123456789')
					e = s[0] if s else a[0]
				ic = line[26:27] if len(line) > 26 else ' '
				occ = float(line[54:60]) if len(line) >= 60 else 1.0
				bfc = float(line[60:66]) if len(line) >= 66 else 0.0
				rows.append((
					a, line[17:20].strip(),
					ch, int(line[22:26]),
					float(line[30:38]),
					float(line[38:46]),
					float(line[46:54]),
					occ, bfc, e,
					ic, alt))
			if has_mdl and model not in found:
				raise Exception(
					f'Model {model} not found in '
					f'{filename}. '
					f'Available models: {found}')
		elif ext == '.cif':
			with open(filename) as f: lines = f.readlines()
			C, ci = {}, 0
			for idx, line in enumerate(lines):
				if line.strip() != 'loop_': continue
				nxt = lines[idx+1].strip() if idx+1 < len(lines) else ''
				if not nxt.startswith('_atom_site.'): continue
				i = idx + 1
				while i < len(lines) and lines[i].strip()\
				.startswith('_atom_site.'):
					cn = lines[i].strip().split('.')[1]
					C[cn] = ci
					ci += 1
					i += 1
				start = i
			if not C: raise Exception(f'No _atom_site loop in {filename}')
			i_mdl = C.get('pdbx_PDB_model_num')
			i_alt = C.get('label_alt_id')
			i_ic  = C.get('pdbx_PDB_ins_code')
			rows, found = [], []
			raw_rows = lines[start:]
			logical = []
			ri = 0
			while ri < len(raw_rows):
				ln_ = raw_rows[ri].rstrip('\n')
				s_ = ln_.strip()
				if s_.startswith(';') and logical:
					block = [s_[1:]]
					ri += 1
					while ri < len(raw_rows) \
					and raw_rows[ri].strip() != ';':
						block.append(raw_rows[ri].rstrip('\n'))
						ri += 1
					ri += 1
					joined = ' '.join(x.strip() for x in block)
					logical[-1] = (logical[-1]
						+ " '" + joined.replace("'", '') + "'")
				else:
					logical.append(ln_)
					ri += 1
			for line in logical:
				ln = line.strip()
				if not ln or ln == '#': break
				f = []
				i, n = 0, len(ln)
				while i < n:
					if ln[i] in (' ', '\t'):
						i += 1
						continue
					if ln[i] in ("'", '"'):
						q = ln[i]
						j = ln.find(q, i + 1)
						if j == -1: j = n - 1
						f.append(ln[i+1:j])
						j += 1
						i = j
					else:
						j = i
						while j < n and ln[j] not in (' ', '\t'): j += 1
						f.append(ln[i:j])
						i = j
				if f[C['group_PDB']] != 'ATOM': continue
				ch = f[C['auth_asym_id']]
				if chain and ch not in chain: continue
				if i_mdl is not None:
					mnum = int(f[i_mdl])
					if mnum not in found: found.append(mnum)
					if mnum != model: continue
				alt = f[i_alt] if i_alt is not None else '.'
				ic  = f[i_ic] if i_ic is not None else ' '
				if ic in ('.', '?'): ic = ' '
				raw_seq = f[C['auth_seq_id']]
				try: seqid = int(raw_seq)
				except ValueError: continue
				rows.append((
					f[C['auth_atom_id']].strip('"'),
					f[C['auth_comp_id']], ch,
					seqid,
					float(f[C['Cartn_x']]),
					float(f[C['Cartn_y']]),
					float(f[C['Cartn_z']]),
					float(f[C['occupancy']]),
					float(f[C['B_iso_or_equiv']]),
					f[C['type_symbol']],
					ic, alt))
			if i_mdl is not None and found and model not in found:
				raise Exception(
					f'Model {model} not found in '
					f'{filename}. Available: {found}')
		else:
			raise Exception(
				f'{filename}: unsupported format (use .pdb or .cif)')
		if not rows:
			raise Exception(
				f'No ATOM records for chains {chain} in {filename}')
		occ_sum = {}
		for row in rows:
			key3 = (row[2], row[3], row[10])
			alt  = row[11]
			if alt in ('', '.', ' '): continue
			if key3 not in occ_sum: occ_sum[key3] = {}
			occ_sum[key3][alt] = occ_sum[key3].get(alt, 0.0) + row[7]
		winner = {}
		for key3, d in occ_sum.items():
			winner[key3] = max(
				d.items(),
				key=lambda kv: (kv[1], -ord(kv[0][0])))[0]
		filtered = []
		for row in rows:
			key3 = (row[2], row[3], row[10])
			alt  = row[11]
			if alt in ('', '.', ' '):
				filtered.append(row)
			elif alt == winner.get(key3):
				filtered.append(row)
		rows = filtered
		best = {}
		for idx, row in enumerate(rows):
			key = (row[2], row[3], row[10], row[0])
			if key not in best or row[7] > rows[best[key]][7]: best[key] = idx
		rows = [rows[i] for i in sorted(best.values())]
		residues = defaultdict(list)
		for row in rows: residues[(row[2], row[3], row[10])].append(row)
		resnames = {a[0][1] for a in residues.values()}
		aa = {v['Tricode'] for v in self.aminoacids.values() if 'Tricode' in v}
		if resnames & aa: mol = 'Protein'
		elif 'U' in resnames: mol = 'RNA'
		elif resnames & {'DT', 'DA', 'DG', 'DC', 'T'}: mol = 'DNA'
		elif resnames & {'A', 'G', 'C'}: mol = 'RNA'
		aa_tri = \
			{v['Tricode'] for v in self.aminoacids.values() if 'Tricode' in v}
		has_aa = bool(resnames & aa_tri)
		nuc_names = {r for r in resnames
			if r in self.nucleotides
			or r in ('DT', 'T', 'U', 'A', 'G', 'C', 'DA', 'DG', 'DC')}
		has_nuc = bool(nuc_names)
		if has_aa and has_nuc:
			raise Exception(
				f'{filename}: mixed protein/nucleic-acid '
				f'structures are not supported. Import each '
				f'molecule type into a separate Pose '
				f'(use the chain parameter to filter).')
		self.data = { 'Type': mol, 'Energy': 0, 'Rg': 0, 'Mass': 0, 'Size': {},
			'FASTA': {}, 'SS': {},
			'Nucleotides': {} if mol != 'Protein' else None,
			'Amino Acids': {} if mol == 'Protein' else None,
			'Atoms': {}, 'Bonds': {}, 'BondOrders': {},
			'Coordinates': np.zeros((0, 3))}
		co = chain if chain is not None else sorted({k[0] for k in residues})
		sk = sorted(residues,
			key=lambda k: (
				co.index(k[0]) if k[0] in co else 99, k[1], k[2]))
		if mol == 'Protein':
			At, Am, Co = {}, {}, []
			count, ai = 0, 0
			for ch, rn, ic in sk:
				seen = set()
				uniq = [r for r in residues[(ch, rn, ic)]
					if r[0] not in seen
					and not seen.add(r[0])]
				tri = uniq[0][1]
				sym = next((k for k, v in
					self.aminoacids.items()
					if v['Tricode'] == tri), None)
				if sym is None: continue
				BB, SC = [], []
				for r in uniq:
					Co.append([r[4], r[5], r[6]])
					At[count] = [r[0], r[9], 0.0, r[7], r[8]]
					(BB if r[0] in self.probbatoms else SC).append(count)
					count += 1
				Am[ai] = [sym, ch, BB, SC, 'L', tri, 0]
				ai += 1
			Co = np.array(Co)
			if not Am: raise Exception(f'No recognized residues in {filename}')
			err = []
			req = {'N', 'CA', 'C'}
			for k in sorted(Am):
				miss = req - {At[a][0] for a in Am[k][2]}
				if miss:
					err.append(
						f'  residue {k} ({Am[k][5]})'
						f': missing {sorted(miss)}')
			keys = sorted(Am)
			warn = []
			for ki, kj in zip(keys[:-1], keys[1:]):
				if Am[ki][1] != Am[kj][1]: continue
				ci = next((a for a in Am[ki][2] if At[a][0] == 'C'), None)
				ni = next((a for a in Am[kj][2] if At[a][0] == 'N'), None)
				if ci is None or ni is None: continue
				d = np.linalg.norm(Co[ci] - Co[ni])
				if d >= 2.5:
					err.append(
						f'  {ki} ({Am[ki][5]})'
						f' \u2192 {kj} ({Am[kj][5]})'
						f': C\u2013N={d:.2f}\u00c5')
				elif d > 1.48:
					warn.append(
						f'  {ki} ({Am[ki][5]})'
						f' \u2192 {kj} ({Am[kj][5]})'
						f': C\u2013N={d:.2f}\u00c5 (strained)')
			if err:
				raise Exception(
					f'Broken chain in {filename}'
					f' chains {chain}:\n'
					+ '\n'.join(err))
			if warn:
				print(
					f'Warning: {filename} has strained peptide'
					f' bonds in chains {chain}:\n'
					+ '\n'.join(warn))
			self.data['Coordinates'] = Co
			self.data['Amino Acids'] = Am
			self.data['Atoms'] = At
			cpos = defaultdict(list)
			for i in range(len(Am)): cpos[Am[i][1]].append(i)
			Bd = {}
			BO = {}
			prev_res_c = {}
			for i in range(len(Am)):
				ch = Am[i][1]
				cr = cpos[ch]
				pos = cr.index(i)
				if pos == 0: bb = 'Backbone start'
				elif pos == len(cr)-1: bb = 'Backbone end'
				else: bb = 'Backbone middle'
				sym = Am[i][0]
				aa_db_entry = self.aminoacids[sym]
				bb_entry = self.aminoacids[bb]
				res_atoms = Am[i][2] + Am[i][3]
				name_to_idx = {}
				for gi in res_atoms:
					name_to_idx[At[gi][0]] = gi
				bb_names = [a[0] for a in bb_entry['Backbone Atoms']]
				for k_str, nbrs in bb_entry['Bonds'].items():
					k = int(k_str)
					orders = bb_entry['BondOrders'][k_str]
					if k < 0 or k >= len(bb_names): continue
					gi = name_to_idx.get(bb_names[k])
					if gi is None: continue
					Bd.setdefault(gi, [])
					BO.setdefault(gi, [])
					for nb, bo in zip(nbrs, orders):
						if nb < 0 or nb >= len(bb_names): continue
						gj = name_to_idx.get(bb_names[nb])
						if gj is None: continue
						if gj not in Bd[gi]:
							Bd[gi].append(gj)
							BO[gi].append(bo)
				sc_names = [a[0] for a in aa_db_entry['Sidechain Atoms']]
				if sc_names:
					ca_idx = name_to_idx.get('CA')
					first_sc_idx = name_to_idx.get(sc_names[0])
					if ca_idx is not None and first_sc_idx is not None:
						Bd.setdefault(ca_idx, [])
						BO.setdefault(ca_idx, [])
						Bd.setdefault(first_sc_idx, [])
						BO.setdefault(first_sc_idx, [])
						if first_sc_idx not in Bd[ca_idx]:
							Bd[ca_idx].append(first_sc_idx)
							BO[ca_idx].append(1)
						if ca_idx not in Bd[first_sc_idx]:
							Bd[first_sc_idx].append(ca_idx)
							BO[first_sc_idx].append(1)
				for k_str, nbrs in aa_db_entry['Bonds'].items():
					k = int(k_str)
					orders = aa_db_entry['BondOrders'][k_str]
					if k < 0 or k >= len(sc_names): continue
					gi = name_to_idx.get(sc_names[k])
					if gi is None: continue
					Bd.setdefault(gi, [])
					BO.setdefault(gi, [])
					for nb, bo in zip(nbrs, orders):
						if nb == -5:
							gj = name_to_idx.get('N')
							if gj is None: continue
							if gj not in Bd[gi]:
								Bd[gi].append(gj)
								BO[gi].append(bo)
							Bd.setdefault(gj, [])
							BO.setdefault(gj, [])
							if gi not in Bd[gj]:
								Bd[gj].append(gi)
								BO[gj].append(bo)
							continue
						if nb < 0 or nb >= len(sc_names): continue
						gj = name_to_idx.get(sc_names[nb])
						if gj is None: continue
						if gj not in Bd[gi]:
							Bd[gi].append(gj)
							BO[gi].append(bo)
				if pos > 0 and ch in prev_res_c:
					pc = prev_res_c[ch]
					n_idx = name_to_idx.get('N')
					if n_idx is not None and pc is not None:
						Bd.setdefault(pc, [])
						BO.setdefault(pc, [])
						if n_idx not in Bd[pc]:
							Bd[pc].append(n_idx)
							BO[pc].append(1.5)
						Bd.setdefault(n_idx, [])
						BO.setdefault(n_idx, [])
						if pc not in Bd[n_idx]:
							Bd[n_idx].append(pc)
							BO[n_idx].append(1.5)
				prev_res_c[ch] = name_to_idx.get('C')
			self.data['Bonds'] = Bd
			self.data['BondOrders'] = BO
			self.CalcCharge()
			self.CalcDSSP()
			self.CalcSASA()
		else:
			Co, ai, ni = [], 0, 0
			nt_ch = defaultdict(list)
			for ch, rn, ic in sk:
				resn = residues[(ch, rn, ic)][0][1]
				if mol == 'DNA':
					if resn in ('A', 'G', 'C', 'T'): tri = 'D' + resn
					else:                            tri = resn
				else:
					tri = resn
				if tri not in self.nucleotides: continue
				nr = {r[0]: r for r in residues[(ch, rn, ic)]}
				db = self.nucleotides[tri]
				bbm = db['Backbone Atoms']
				bsm = db['Base Atoms']
				n_bb = len(bbm)
				D = self.data
				bbi, bsi, ltg = [], [], {}
				for li, am in enumerate(bbm + bsm):
					row = nr.get(am[0])
					if row is None:
						ltg[li] = -1
						continue
					D['Atoms'][ai] = [am[0], am[1], am[2], row[7], row[8]]
					Co.append([row[4], row[5], row[6]])
					ltg[li] = ai
					(bsi if li >= n_bb else bbi).append(ai)
					ai += 1
				dbn = {a[0] for a in bbm} | {a[0] for a in bsm}
				for an, row in nr.items():
					if an in dbn: continue
					D['Atoms'][ai] = [an, row[9] or an[0], 0.0, row[7], row[8]]
					Co.append([row[4], row[5], row[6]])
					(bbi if an in self.nucbbatoms else bsi).append(ai)
					ai += 1
				bd = D['Bonds']
				bo_d = D.setdefault('BondOrders', {})
				db_bo = db.get('BondOrders', {})
				for ks, vl in db['Bonds'].items():
					gi = ltg.get(int(ks), -1)
					if gi == -1: continue
					bd.setdefault(gi, [])
					bo_d.setdefault(gi, [])
					orders = db_bo.get(ks, [1] * len(vl))
					for lj, lo in zip(vl, orders):
						gj = ltg.get(lj, -1)
						if gj != -1 and gj not in bd[gi]:
							bd[gi].append(gj)
							bo_d[gi].append(lo)
				self.data['Nucleotides'][ni] = [tri[-1], ch, bbi, bsi, tri]
				nt_ch[ch].append(ni)
				ni += 1
			self.data['Coordinates'] = np.array(Co) if Co else np.zeros((0, 3))
			self._phosphobonds(nt_ch)
			self.CalcCharge()
		self._update()
	def Export(self, filename):
		''' Export structure to a .pdb or .cif file '''
		ext = os.path.splitext(filename)[1].lower()
		if ext not in ('.pdb', '.cif'):
			raise Exception(
				f'{filename}: unsupported format (use .pdb or .cif)')
		src = (self.data['Amino Acids'] or self.data['Nucleotides'])
		if not src:
			raise Exception(
				'No structure loaded. '
				'Call Import() first')
		is_pro = self.data['Type'] == 'Protein'
		ti = 5 if is_pro else 4
		if ext == '.pdb':
			with open(filename, 'w') as f:
				dt = datetime.date.today()
				f.write('HEADER' + ' '*44
					+ dt.strftime('%d-%b-%Y')
					+ ' '*3 + 'XXXX'
					+ ' '*11 + '\n')
				f.write('EXPDTA    '
					'THEORETICAL MODEL'
					+ ' '*52 + '\n')
				f.write('REMARK 220 REMARK: '
					'MODEL GENERATED BY '
					'SARI SABBAN'
					+ ' '*30 + '\n')
				prev_ch = None
				for atom, coord, ri in self._atomiter():
					info = src[ri]
					ch = info[1]
					if prev_ch is not None and ch != prev_ch: f.write('TER\n')
					pdbentry = \
						'{:<6}'     .format('ATOM')          + \
						'{:>5} '    .format(atom[0]+1)       + \
						'{:<4}'     .format(atom[1][0])      + \
						'{:<1}'     .format('')              + \
						'{:>3}'     .format(info[ti])        + \
						'{:>2}'     .format(ch)              + \
						'{:>4}'     .format(ri+1)            + \
						'{:>1}   '  .format('')              + \
						'{:>8.3f}'  .format(coord[0])        + \
						'{:>8.3f}'  .format(coord[1])        + \
						'{:>8.3f} ' .format(coord[2])        + \
						'{:>5.2f} ' .format(atom[1][3])      + \
						('{:>5.2f} '.format(atom[1][4]))[:6] + \
						'{:>10}'    .format('')              + \
						'{:<2}'     .format(atom[1][1])      + \
						'{:<2}'     .format(atom[1][2])      + \
						'\n'
					f.write(pdbentry)
					prev_ch = ch
				f.write('TER\n')
		else:
			name = os.path.splitext(os.path.basename(filename))[0]
			cols = [
				'_atom_site.group_PDB',
				'_atom_site.id',
				'_atom_site.type_symbol',
				'_atom_site.label_atom_id',
				'_atom_site.label_alt_id',
				'_atom_site.label_comp_id',
				'_atom_site.label_asym_id',
				'_atom_site.label_entity_id',
				'_atom_site.label_seq_id',
				'_atom_site.pdbx_PDB_ins_code',
				'_atom_site.Cartn_x',
				'_atom_site.Cartn_y',
				'_atom_site.Cartn_z',
				'_atom_site.occupancy',
				'_atom_site.B_iso_or_equiv',
				'_atom_site.pdbx_formal_charge',
				'_atom_site.auth_seq_id',
				'_atom_site.auth_comp_id',
				'_atom_site.auth_asym_id',
				'_atom_site.auth_atom_id',
				'_atom_site.pdbx_PDB_model_num']
			with open(filename, 'w') as f:
				f.write(f'data_{name}\n#\n')
				f.write(f'_entry.id   {name}\n#\n')
				f.write("_exptl.method   "
					"'THEORETICAL MODEL'\n#\n")
				f.write("_audit_author.name"
					"          'SARI SABBAN'\n")
				f.write("_audit_author."
					"pdbx_ordinal  1\n#\n")
				f.write('loop_\n')
				for c in cols:
					f.write(c + '\n')
				for atom, coord, ri in self._atomiter():
					info = src[ri]
					alt = '.'
					ch = '?' if atom[1][2] == 0.0 else '{:.3f}' \
					.format(atom[1][2])
					cifentry = ' '.join([
						'ATOM', str(atom[0]+1), atom[1][1], atom[1][0], alt,
						info[ti], info[1],
						'1', str(ri+1), '?',
						'{:.3f}'.format(coord[0]),
						'{:.3f}'.format(coord[1]),
						'{:.3f}'.format(coord[2]),
						'{:.2f}'.format(atom[1][3]),
						'{:.2f}'.format(atom[1][4]),
						ch, str(ri+1), info[ti], info[1], atom[1][0], '1'])+'\n'
					f.write(cifentry)
				f.write('#\n')
	def MovePose(self, theta=None, u=None, l=None, ori=None):
		''' Rotate and/or translate the full pose rigidly '''
		coords = self.data['Coordinates'].copy()
		if len(coords) == 0:
			raise Exception('No atoms to move')
		rot_args   = (theta is not None, u is not None)
		trans_args = (l is not None, ori is not None)
		if any(rot_args) and not all(rot_args):
			raise Exception(
				'MovePose rotation requires BOTH theta and u')
		if any(trans_args) and not all(trans_args):
			raise Exception(
				'MovePose translation requires BOTH l and ori')
		if not any(rot_args) and not any(trans_args):
			raise Exception(
				'MovePose called with no arguments (nothing to do)')
		if theta is not None and u is not None:
			u = np.array(u, dtype=float)
			mag = np.linalg.norm(u)
			if mag < 1e-10:
				raise Exception('Rotation axis u cannot be a zero vector')
			u = u / mag
			pivot  = coords.mean(axis=0)
			R      = self._rotmat(theta, u)
			coords = np.matmul(coords - pivot, R) + pivot
		if l is not None and ori is not None:
			ori      = np.array(ori, dtype=float)
			centroid = coords.mean(axis=0)
			d        = ori - centroid
			mag      = np.linalg.norm(d)
			if mag < 1e-10:
				raise Exception(
					'ori coincides with pose centroid, '
					'translation direction is undefined')
			coords = coords + (d / mag) * l
		self.data['Coordinates'] = coords
	def AdjustDistance(self, res1, atom1, res2, atom2, length):
		''' Change distance between two atoms '''
		Ai = self.GetAtomIdx(res1, atom1)
		Bi = self.GetAtomIdx(res2, atom2)
		coords = self.data['Coordinates']
		v = coords[Bi] - coords[Ai]
		mag = np.linalg.norm(v)
		if mag < 1e-10: return
		shift = v * (length / mag) - v
		for idx in self._downstreamatoms(
		res1, atom1, res2, atom2): coords[idx] += shift
		self.data['Coordinates'] = coords
	def AdjustAngle(self, res1, atom1, res2, atom2, res3, atom3, theta):
		''' Change angle between three atoms, res2/atom2 is the pivot '''
		A = (self.GetAtomCoord(res3, atom3)- self.GetAtomCoord(res1, atom1))
		B = (self.GetAtomCoord(res3, atom3)- self.GetAtomCoord(res2, atom2))
		u = np.cross(B, A)
		lu = np.linalg.norm(u)
		if lu < 1e-10: return
		u = u / lu
		ori = self.GetAtomCoord(res2, atom2).copy()
		RM = self._rotmat(theta, u)
		coords = self.data['Coordinates']
		for idx in self._downstreamatoms(
		res2, atom2, res3, atom3):
			v = coords[idx] - ori
			coords[idx] = np.matmul(v, RM) + ori
		self.data['Coordinates'] = coords
	def RotateDihedral(self, res, theta, angle_type, chi_type=None):
		''' Set a dihedral angle to theta degrees '''
		at = angle_type.upper()
		mol = self.data['Type']
		nxt = self._nextres(res)
		if mol == 'Protein':
			if at == 'PHI': pivots = res, 'N', res, 'CA'
			elif at == 'PSI': pivots = res, 'CA', res, 'C'
			elif at == 'OMEGA':
				if nxt is None: return
				pivots = res, 'C', nxt, 'N'
			elif at == 'CHI':
				assert chi_type is not None, 'Protein CHI needs chi_type'
				sym = self.data['Amino Acids'][res][0].upper()
				ca = self.aminoacids[sym]['Chi Angle Atoms'][chi_type-1]
				pivots = res, ca[1], res, ca[2]
			else:
				raise Exception(
					f'Unknown protein angle: {angle_type}')
		elif mol in ('DNA', 'RNA'):
			if   at == 'ALPHA':     pivots = res, 'P', res, "O5'"
			elif at == 'BETA':    pivots = res, "O5'", res, "C5'"
			elif at == 'GAMMA':   pivots = res, "C5'", res, "C4'"
			elif at == 'DELTA':   pivots = res, "C4'", res, "C3'"
			elif at == 'EPSILON': pivots = res, "C3'", res, "O3'"
			elif at == 'ZETA':
				if nxt is None: return
				pivots = res, "O3'", nxt, 'P'
			elif at == 'CHI':
				tri = self.data['Nucleotides'][res][4]
				ca = self.nucleotides[tri]['Chi Angle Atoms']
				pivots = res, ca[1], res, ca[2]
			else:
				raise Exception(
					f'Unknown nucleotide angle'
					f': {angle_type}')
		else:
			raise Exception('No structure loaded. Call Import() first')
		if pivots is None: return
		ra, aa, rb, ab = pivots
		current = self.GetDihedral(res, angle_type, chi_type)
		piv_a = self.GetAtomIdx(ra, aa)
		piv_b = self.GetAtomIdx(rb, ab)
		coords = self.data['Coordinates']
		ori = coords[piv_b].copy()
		u = coords[piv_a] - coords[piv_b]
		mag = np.linalg.norm(u)
		if mag < 1e-10: return
		u = u / mag
		RM_zero = self._rotmat(-current, u)
		RM_new = self._rotmat(theta, u)
		for idx in self._downstreamatoms(
		ra, aa, rb, ab):
			v = coords[idx] - ori
			v = np.matmul(v, RM_zero)
			v = np.matmul(v, RM_new)
			coords[idx] = v + ori
		self.data['Coordinates'] = coords
	def Build(self, sequence, chain='A', fmt='Protein'):
		''' Build protein/DNA/RNA from sequence '''
		fmt = {'protein': 'Protein',
			'dna': 'DNA', 'rna': 'RNA'}.get(fmt.lower(), fmt)
		if fmt == 'Protein': self._buildprotein(sequence, chain)
		elif fmt in ('DNA', 'RNA'):
			self._buildnucleotide(sequence, fmt, chains=chain)
		else: raise Exception(f'Unknown format: {fmt}')
		self.CalcCharge()
		if fmt == 'Protein': self.CalcSASA()
		self._update()
	def ReBuild(self, sequence=None, mirror=False, _mutated=None):
		''' Rebuild a structure from internal coords '''
		mol = self.data['Type']
		if mol == 'Protein':
			AAs = self.data['Amino Acids']
			N = len(AAs)
			fasta = self.data['FASTA']
			if sequence is None: sequence = fasta
			if isinstance(sequence, str) and len(fasta) > 1:
				raise Exception(
					'Multi-chain protein requires '
					'sequence as dict, e.g. '
					'{"A":"SEQ1", "B":"SEQ2"}')
			if isinstance(sequence, dict):
				full_seq = dict(fasta)
				full_seq.update(sequence)
				sequence = full_seq
			if isinstance(sequence, dict):
				for ch in sequence:
					if ch in fasta and len(sequence[ch]) != len(fasta[ch]):
						raise Exception(
							f'Chain {ch}: sequence '
							f'length '
							f'{len(sequence[ch])} '
							f'does not match '
							f'original '
							f'{len(fasta[ch])}')
			elif isinstance(sequence, str):
				ch0 = sorted(fasta.keys())[0]
				if len(sequence) != len(fasta[ch0]):
					raise Exception(
						f'Sequence length '
						f'{len(sequence)} does not '
						f'match original '
						f'{len(fasta[ch0])}')
			orig_coords = {}
			for i in range(N):
				info = AAs[i]
				for ai in info[2] + info[3]:
					aname = self.data['Atoms'][ai][0]
					orig_coords[(i,aname)] = self.data['Coordinates'][ai].copy()
			self.data = {
				'Type': None, 'Energy': 0,
				'Rg': 0, 'Mass': 0, 'Size': {},
				'FASTA': {}, 'SS': {},
				'Nucleotides': None,
				'Amino Acids': None,
				'Atoms': {}, 'Bonds': {}, 'BondOrders': {},
				'Coordinates': np.zeros((0, 3))}
			if isinstance(sequence, dict):
				for ch, seq in sorted(sequence.items()):
					self._buildprotein(seq, ch)
			else:
				ch0 = sorted(fasta.keys())[0]
				self._buildprotein(sequence, ch0)
			AAs2 = self.data['Amino Acids']
			coords = self.data['Coordinates']
			for i in range(len(AAs2)):
				has_orig = (
					(i, 'N') in orig_coords
					and (i, 'CA') in orig_coords
					and (i, 'C') in orig_coords)
				if not has_orig: continue
				info = AAs2[i]
				n_i = next(a for a in info[2] if self.data['Atoms'][a][0]=='N')
				ca_i= next(a for a in info[2] if self.data['Atoms'][a][0]=='CA')
				c_i = next(a for a in info[2] if self.data['Atoms'][a][0]=='C')
				n = coords[n_i].copy()
				ca = coords[ca_i].copy()
				c = coords[c_i].copy()
				e1 = ca - n
				e1 = e1 / np.linalg.norm(e1)
				v = c - n
				e2 = v - np.dot(v, e1) * e1
				nm = np.linalg.norm(e2)
				if nm < 1e-10: return None, None
				e2 = e2 / nm
				F_new, ori_new = np.array([e1, e2, np.cross(e1, e2)]), n
				if F_new is None: continue
				on = orig_coords[(i, 'N')]
				oca = orig_coords[(i, 'CA')]
				oc = orig_coords[(i, 'C')]
				e1 = oca - on
				e1 = e1 / np.linalg.norm(e1)
				v = oc - on
				e2 = v - np.dot(v, e1) * e1
				nm = np.linalg.norm(e2)
				if nm < 1e-10: continue
				e2 = e2 / nm
				F_orig = np.array([e1, e2, np.cross(e1, e2)])
				F_orig_inv = F_orig.T
				for ai in AAs2[i][2] + AAs2[i][3]:
					aname = self.data['Atoms'][ai][0]
					if (i, aname) in orig_coords:
						coords[ai] = orig_coords[(i, aname)]
					else:
						local = F_new @ (coords[ai] - ori_new)
						coords[ai] = on + F_orig_inv @ local
			self.data['Coordinates'] = coords
			self.CalcCharge()
			self.CalcDSSP()
			self.CalcSASA()
			if mirror:
				self.data['Coordinates'] *= [1, 1, -1]
				N2 = len(self.data['Amino Acids'])
				for i in range(N2):
					aa = self.data['Amino Acids'][i]
					sym = aa[0]
					if sym.isupper():
						aa[0] = sym.lower()
						tri_L = self.aminoacids[sym]['Tricode']
						aa[5] = 'D' + tri_L[1:]
					else:
						aa[0] = sym.upper()
						aa[5] = self.aminoacids[sym.upper()]['Tricode']
		elif mol in ('DNA', 'RNA'):
			fmt = self.data['Type']
			nts = self.data['Nucleotides']
			N = len(nts)
			fasta = self.data['FASTA']
			first_ch = sorted(fasta.keys())[0]
			seq_a = fasta[first_ch] if sequence is None else sequence
			orig_len = len(fasta[first_ch])
			if len(seq_a) != orig_len:
				raise Exception(
					f'Sequence length '
					f'{len(seq_a)} does not match '
					f'original {orig_len}')
			old_syms = [nts[i][0] for i in range(N)]
			ALPHAs, BETAs, GAMMAs = {}, {}, {}
			DELTAs, EPSILONs, ZETAs = {}, {}, {}
			CHIs = {}
			aPO5C5, aO5C5C4 = {}, {}
			aC5C4C3, aC4C3O3 = {}, {}
			aC3O3P, aO3PO5 = {}, {}
			bPO5, bO5C5, bC5C4 = {}, {}, {}
			bC4C3, bC3O3, bO3P = {}, {}, {}
			for i in range(N):
				prv = self._prevres(i)
				nxt = self._nextres(i)
				has_P = self._hasatom(i, 'P')
				if prv is not None and self._hasatom(prv, "O3'") and has_P:
					ALPHAs[i] = self.GetDihedral(i, 'ALPHA')
				if has_P:
					BETAs[i] = self.GetDihedral(i, 'BETA')
				GAMMAs[i] = self.GetDihedral(i, 'GAMMA')
				DELTAs[i] = self.GetDihedral(i, 'DELTA')
				if nxt is not None and self._hasatom(nxt, 'P'):
					EPSILONs[i] = self.GetDihedral(i, 'EPSILON')
					ZETAs[i] = self.GetDihedral(i, 'ZETA')
				CHIs[i] = self.GetDihedral(i, 'CHI')
				if has_P:
					aPO5C5[i] = self.GetAngle(i, 'P', i, "O5'", i, "C5'")
					bPO5[i] = self.GetDistance(i, 'P', i, "O5'")
				aO5C5C4[i] = self.GetAngle(i, "O5'", i, "C5'", i, "C4'")
				bO5C5[i] = self.GetDistance(i, "O5'", i, "C5'")
				aC5C4C3[i] = self.GetAngle(i, "C5'", i, "C4'", i, "C3'")
				bC5C4[i] = self.GetDistance(i, "C5'", i, "C4'")
				aC4C3O3[i] = self.GetAngle(i, "C4'", i, "C3'", i, "O3'")
				bC4C3[i] = self.GetDistance(i, "C4'", i, "C3'")
				bC3O3[i] = self.GetDistance(i, "C3'", i, "O3'")
				if nxt is not None:
					aC3O3P[i] = self.GetAngle(i, "C3'", i, "O3'", nxt, 'P')
					aO3PO5[i] = self.GetAngle(i, "O3'", nxt, 'P', nxt, "O5'")
					bO3P[i] = self.GetDistance(i, "O3'", nxt, 'P')
			ref_C1=[self.GetAtomCoord(i,"C1'").copy() if self._hasatom(i,"C1'")
				else None for i in range(N)]
			ring_local = {}
			bb_skip = {"P", "O5'", "C5'", "C4'", "C3'", "O3'"}
			for i in range(N):
				if not (self._hasatom(i, "C4'")
				and self._hasatom(i, "C3'")
				and self._hasatom(i, "C5'")): continue
				c4 = self.GetAtomCoord(i, "C4'").copy()
				c3 = self.GetAtomCoord(i, "C3'").copy()
				c5 = self.GetAtomCoord(i, "C5'").copy()
				e1 = c3 - c4
				e1 = e1 / np.linalg.norm(e1)
				v = c5 - c4
				e2 = v - np.dot(v, e1) * e1
				n = np.linalg.norm(e2)
				if n < 1e-10: return None
				e2 = e2 / n
				F = np.array([e1, e2, np.cross(e1, e2)])
				if F is None: continue
				info = nts[i]
				saved = {}
				for ai in info[2] + info[3]:
					aname = self.data['Atoms'][ai][0]
					if aname in bb_skip: continue
					pos = self.data['Coordinates'][ai].copy()
					saved[aname] = F @ (pos - c4)
				ring_local[i] = saved
			orig_chains = sorted(set(v[1] for v in nts.values()))
			self._buildnucleotide(seq_a, fmt, chains=orig_chains)
			self.CalcCharge()
			nts2 = self.data['Nucleotides']
			if _mutated is None:
				new_syms = [nts2[i][0] for i in range(len(nts2))]
				_mutated = {i for i in range(min(N, len(nts2)))
					if old_syms[i] != new_syms[i]}
			build_base_local = {}
			for mi in _mutated:
				if mi >= len(nts2): continue
				info = nts2[mi]
				At = self.data['Atoms']
				ai_o4 = next((a for a in info[2] if At[a][0] == "O4'"), None)
				ai_c1 = next((a for a in info[2] if At[a][0] == "C1'"), None)
				ai_c2 = next((a for a in info[2] if At[a][0] == "C2'"), None)
				if None in (ai_o4, ai_c1, ai_c2): return None, None
				co = self.data['Coordinates']
				c1 = co[ai_c1].copy()
				o4 = co[ai_o4].copy()
				c2 = co[ai_c2].copy()
				e1 = o4 - c1
				nm = np.linalg.norm(e1)
				if nm < 1e-10: return None, None
				e1 = e1 / nm
				v = c2 - c1
				e2 = v - np.dot(v, e1) * e1
				nm2 = np.linalg.norm(e2)
				if nm2 < 1e-10: return None, None
				e2 = e2 / nm2
				Fb, c1b = np.array([e1, e2, np.cross(e1, e2)]), c1
				if Fb is None: continue
				saved = {}
				for ai in nts2[mi][3]:
					aname = self.data['Atoms'][ai][0]
					saved[aname] = Fb @ (self.data['Coordinates'][ai] - c1b)
				build_base_local[mi] = saved
			for i in range(N):
				if i in ALPHAs: self.RotateDihedral(i, ALPHAs[i], 'ALPHA')
				if i in BETAs: self.RotateDihedral(i, BETAs[i], 'BETA')
				self.RotateDihedral(i, GAMMAs[i], 'GAMMA')
				self.RotateDihedral(i, DELTAs[i], 'DELTA')
				self.RotateDihedral(i, CHIs[i], 'CHI')
				if i in bPO5: self.AdjustDistance(i, 'P', i, "O5'", bPO5[i])
				if i in bO5C5: self.AdjustDistance(i, "O5'", i, "C5'", bO5C5[i])
				if i in bC5C4: self.AdjustDistance(i, "C5'", i, "C4'", bC5C4[i])
				if i in bC4C3: self.AdjustDistance(i, "C4'", i, "C3'", bC4C3[i])
				if i in bC3O3: self.AdjustDistance(i, "C3'", i, "O3'", bC3O3[i])
				if i in aPO5C5:
					cur = self.GetAngle(i, 'P', i, "O5'", i, "C5'")
					self.AdjustAngle(i, 'P', i, "O5'", i, "C5'", cur-aPO5C5[i])
				if i in aO5C5C4:
					cur = self.GetAngle(i, "O5'", i, "C5'", i, "C4'")
					self.AdjustAngle(i, "O5'", i, "C5'", i,"C4'",cur-aO5C5C4[i])
				if i in aC5C4C3:
					cur = self.GetAngle(i, "C5'", i, "C4'", i, "C3'")
					self.AdjustAngle(i, "C5'", i, "C4'", i,"C3'",cur-aC5C4C3[i])
				if i in aC4C3O3:
					cur = self.GetAngle(i, "C4'", i, "C3'", i, "O3'")
					self.AdjustAngle(i, "C4'", i, "C3'", i,"O3'",cur-aC4C3O3[i])
				nxt = self._nextres(i)
				if nxt is not None:
					if i in EPSILONs:
						self.RotateDihedral(i, EPSILONs[i], 'EPSILON')
					if i in ZETAs:
						self.RotateDihedral(i, ZETAs[i], 'ZETA')
					if i in bO3P:
						self.AdjustDistance(i, "O3'", nxt, 'P', bO3P[i])
					if i in aC3O3P:
						cur = self.GetAngle(i, "C3'", i, "O3'", nxt, 'P')
						self.AdjustAngle(
							i, "C3'", i, "O3'", nxt, 'P', cur - aC3O3P[i])
					if i in aO3PO5:
						cur = self.GetAngle(i, "O3'", nxt, 'P', nxt, "O5'")
						self.AdjustAngle(
							i, "O3'", nxt, 'P', nxt, "O5'", cur - aO3PO5[i])
			coords = self.data['Coordinates']
			for i in range(N):
				if i not in ring_local: continue
				if not (self._hasatom(i, "C4'")
				and self._hasatom(i, "C3'")
				and self._hasatom(i, "C5'")): continue
				c4 = self.GetAtomCoord(i, "C4'").copy()
				c3 = self.GetAtomCoord(i, "C3'").copy()
				c5 = self.GetAtomCoord(i, "C5'").copy()
				e1 = c3 - c4
				e1 = e1 / np.linalg.norm(e1)
				v = c5 - c4
				e2 = v - np.dot(v, e1) * e1
				n = np.linalg.norm(e2)
				if n < 1e-10: return None
				e2 = e2 / n
				F = np.array([e1, e2, np.cross(e1, e2)])
				if F is None: continue
				Finv = F.T
				saved = ring_local[i]
				if i in _mutated: restore = nts2[i][2]
				else: restore = nts2[i][2] + nts2[i][3]
				for ai in restore:
					aname = self.data['Atoms'][ai][0]
					if aname in saved: coords[ai] = c4 + Finv @ saved[aname]
			self.data['Coordinates'] = coords
			coords = self.data['Coordinates']
			for ch in set(v[1] for v in nts2.values()):
				ref_pts, new_pts, aids = [], [], []
				for i in range(N):
					if nts2[i][1] != ch: continue
					if ref_C1[i] is None: continue
					ref_pts.append(ref_C1[i])
					new_pts.append(self.GetAtomCoord(i, "C1'").copy())
					for ai in nts2[i][2] + nts2[i][3]:
						if ai not in aids: aids.append(ai)
				if len(ref_pts) < 3: continue
				P = np.array(ref_pts, dtype=float)
				Q = np.array(new_pts, dtype=float)
				Pc = P.mean(0)
				Qc = Q.mean(0)
				H = (P - Pc).T @ (Q - Qc)
				U, S, Vt = np.linalg.svd(H)
				d = np.sign(np.linalg.det(Vt.T @ U.T))
				R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
				for ai in aids: coords[ai] = (coords[ai] - Qc) @ R + Pc
			self.data['Coordinates'] = coords
			coords = self.data['Coordinates']
			for mi in _mutated:
				if mi not in build_base_local: continue
				info = nts2[mi]
				At = self.data['Atoms']
				ai_o4 = next((a for a in info[2] if At[a][0] == "O4'"), None)
				ai_c1 = next((a for a in info[2] if At[a][0] == "C1'"), None)
				ai_c2 = next((a for a in info[2] if At[a][0] == "C2'"), None)
				if None in (ai_o4, ai_c1, ai_c2): return None, None
				co = self.data['Coordinates']
				c1 = co[ai_c1].copy()
				o4 = co[ai_o4].copy()
				c2 = co[ai_c2].copy()
				e1 = o4 - c1
				nm = np.linalg.norm(e1)
				if nm < 1e-10: return None, None
				e1 = e1 / nm
				v = c2 - c1
				e2 = v - np.dot(v, e1) * e1
				nm2 = np.linalg.norm(e2)
				if nm2 < 1e-10: return None, None
				e2 = e2 / nm2
				Fr, c1r = np.array([e1, e2, np.cross(e1, e2)]), c1
				if Fr is None: continue
				Fri = Fr.T
				bsaved = build_base_local[mi]
				for ai in nts2[mi][3]:
					aname = self.data['Atoms'][ai][0]
					if aname in bsaved: coords[ai] = c1r + Fri @ bsaved[aname]
			self.data['Coordinates'] = coords
			for mi in _mutated:
				if mi >= len(nts2): continue
				if mi not in CHIs: continue
				cur = self.GetDihedral(mi, 'CHI')
				target = CHIs[mi]
				tri = nts2[mi][4]
				ca = self.nucleotides[tri]['Chi Angle Atoms']
				piv_a = self.GetAtomIdx(mi, ca[1])
				piv_b = self.GetAtomIdx(mi, ca[2])
				ori = coords[piv_b].copy()
				u = coords[piv_a] - coords[piv_b]
				mag = np.linalg.norm(u)
				if mag < 1e-10: continue
				u = u / mag
				RM0 = self._rotmat(-cur, u)
				RM1 = self._rotmat(target, u)
				for ai in nts2[mi][3]:
					v = coords[ai] - ori
					v = np.matmul(v, RM0)
					v = np.matmul(v, RM1)
					coords[ai] = v + ori
			self.data['Coordinates'] = coords
		else:
			raise Exception(
				'No structure loaded. Call Import() first')
		self._update()
	def Mutate(self, index, residue):
		''' Mutate a residue or nucleotide '''
		mol = self.data['Type']
		if mol == 'Protein':
			ru = residue.upper()
			valid_aa = {k for k in self.aminoacids
				if 'Tricode' in self.aminoacids[k]
				and not k.startswith('Backbone')}
			if ru not in valid_aa:
				raise Exception(
					f"Unknown amino acid '{residue}'. "
					f"Supported: {''.join(sorted(valid_aa))}")
			AAs = self.data['Amino Acids']
			if index not in AAs:
				raise Exception(f'Residue {index} not found')
			ch = AAs[index][1]
			fasta = self.data['FASTA']
			chain_idxs = [i for i in sorted(AAs) if AAs[i][1] == ch]
			pos = chain_idxs.index(index)
			seq = list(fasta[ch])
			seq[pos] = residue
			new_fasta = dict(fasta)
			new_fasta[ch] = ''.join(seq)
			self.ReBuild(sequence=new_fasta)
		elif mol in ('DNA', 'RNA'):
			valid = ({'A', 'T', 'G', 'C'} if mol == 'DNA'
				else {'A', 'U', 'G', 'C'})
			if residue.upper() not in valid:
				raise Exception(
					f"Unknown {mol} base '{residue}'. "
					f"Supported: {sorted(valid)}")
			nts = self.data['Nucleotides']
			if index not in nts:
				raise Exception(f'Nucleotide {index} not found')
			ch = nts[index][1]
			fasta = self.data['FASTA']
			chain_idxs = [i for i in sorted(nts) if nts[i][1] == ch]
			pos = chain_idxs.index(index)
			first_ch = sorted(fasta.keys())[0]
			seq = list(fasta[first_ch])
			N = len(seq)
			if ch == first_ch: seq[pos] = residue.upper()
			else:
				if self.data['Type'] == 'DNA':
					comp = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}
				else:
					comp = {'A':'U', 'U':'A', 'G':'C', 'C':'G'}
				bp = N - 1 - pos
				seq[bp] = comp.get(residue.upper(), residue.upper())
			chains = sorted(set(v[1] for v in nts.values()))
			mutated = {index}
			if len(chains) > 1: mutated.add(2 * N - 1 - index)
			self.ReBuild(sequence=''.join(seq), _mutated=mutated)
		else:
			raise Exception(
				'No structure loaded. Call Import() first')

class Molecule():
	def __init__(self):
		self.masses = {
			'H':1.008, 'He':4.003, 'Li':6.941, 'Be':9.012,
			'B':10.811, 'C':12.011, 'N':14.007, 'O':15.999,
			'F':18.998, 'Ne':20.180, 'Na':22.990, 'Mg':24.305,
			'Al':26.982, 'Si':28.086, 'P':30.974, 'S':32.066,
			'Cl':35.453, 'Ar':39.948, 'K':39.098, 'Ca':40.078,
			'Sc':44.956, 'Ti':47.867, 'V':50.942, 'Cr':51.996,
			'Mn':54.938, 'Fe':55.845, 'Co':58.933, 'Ni':58.693,
			'Cu':63.546, 'Zn':65.38, 'Ga':69.723, 'Ge':72.631,
			'As':74.922, 'Se':78.971, 'Br':79.904, 'Kr':84.798,
			'Rb':84.468, 'Sr':87.62, 'Y':88.906, 'Zr':91.224,
			'Nb':92.906, 'Mo':95.95, 'Tc':98.907, 'Ru':101.07,
			'Rh':102.906, 'Pd':106.42, 'Ag':107.868,
			'Cd':112.414, 'In':114.818, 'Sn':118.711,
			'Sb':121.760, 'Te':126.7, 'I':126.904,
			'Xe':131.294, 'Cs':132.905, 'Ba':137.328,
			'La':138.905, 'Ce':140.116, 'Pr':140.908,
			'Nd':144.243, 'Pm':144.913, 'Sm':150.36,
			'Eu':151.964, 'Gd':157.25, 'Tb':158.925,
			'Dy':162.500, 'Ho':164.930, 'Er':167.259,
			'Tm':168.934, 'Yb':173.055, 'Lu':174.967,
			'Hf':178.49, 'Ta':180.948, 'W':183.84,
			'Re':186.207, 'Os':190.23, 'Ir':192.217,
			'Pt':195.085, 'Au':196.967, 'Hg':200.592,
			'Tl':204.383, 'Pb':207.2, 'Bi':208.980,
			'Po':208.982, 'At':209.987, 'Rn':222.081,
			'Fr':223.020, 'Ra':226.025, 'Ac':227.028,
			'Th':232.038, 'Pa':231.036, 'U':238.029,
			'Np':237, 'Pu':244}
		self.elements = {k for k in self.masses if len(k) == 2}
		self.data = {
			'Type':'Molecule', 'Energy':0, 'Rg':0, 'Mass':0,
			'SMILES':None, 'SMARTS':None, 'Formula':None,
			'Atoms':{}, 'Bonds':{},
			'Coordinates':np.zeros((0, 3))}
		self._bond_orders = {}
		self._formal_charges = {}
	def _invalidate(self):
		self.data['Rg'] = None
		self.data['Energy'] = None
	def _check_idx(self, *idxs):
		n = len(self.data['Coordinates'])
		for i in idxs:
			if i < 0 or i >= n:
				raise IndexError(f'Atom index {i} out of range [0, {n})')
	def _rotmat(self, theta, u):
		mg = np.linalg.norm(u)
		if mg > 1e-10: u = u / mg
		ux, uy, uz = u
		S = math.sin(math.radians(theta))
		C = math.cos(math.radians(theta))
		return np.array([
			[C+ux**2*(1-C), ux*uy*(1-C)-uz*S, ux*uz*(1-C)+uy*S],
			[uy*ux*(1-C)+uz*S, C+uy**2*(1-C), uy*uz*(1-C)-ux*S],
			[uz*ux*(1-C)-uy*S, uz*uy*(1-C)+ux*S, C+uz**2*(1-C)]])
	def _downstream(self, idx1, idx2):
		if idx1 == idx2: raise ValueError('_downstream: idx1 == idx2')
		visited = {idx2}; queue = [idx2]; qi = 0
		while qi < len(queue):
			cur = queue[qi]; qi += 1
			for nb in self.data['Bonds'].get(cur, []):
				if nb == idx1 or nb in visited: continue
				visited.add(nb); queue.append(nb)
		return visited
	def Import(self, filename):
		if isinstance(filename, str) and '\n' in filename:
			lines = filename.splitlines(); ext = 'string'
		else:
			ext = os.path.splitext(filename)[1].lower()
			with open(filename) as f: lines = f.readlines()
		atoms = {}; coords = []; bonds = {}; bords = {}; idx = 0
		if ext == '.pdb':
			for line in lines:
				rec = line[:6].strip()
				if rec == 'ENDMDL': break
				if rec in ('ATOM', 'HETATM'):
					name = line[12:16].strip()
					x = float(line[30:38])
					y = float(line[38:46])
					z = float(line[46:54])
					el = line[76:78].strip() if len(line) > 76 else ''
					if not el:
						raw = line[12:16]
						if len(raw) >= 2 and raw[0].isalpha():
							cand = raw[0].upper() + raw[1].lower()
							if cand in self.elements: el = cand
							else: el = raw[0].upper()
						else:
							el = re.sub(r'[\d\s]', '', raw).strip()
							el = el[0].upper() if el else 'X'
					atoms[idx] = [name, el, 0.0]
					coords.append([x, y, z])
					bonds[idx] = []; idx += 1
				elif rec == 'CONECT':
					vals = [int(line[k:k+5])
						for k in range(6, len(line.rstrip()), 5)
						if line[k:k+5].strip()]
					if vals:
						a1 = vals[0] - 1
						for v in vals[1:]:
							a2 = v - 1
							if a1 < 0 or a2 < 0 or a1 >= idx or a2 >= idx:
								continue
							if a2 not in bonds.get(a1, []):
								bonds.setdefault(a1, []).append(a2)
							if a1 not in bonds.get(a2, []):
								bonds.setdefault(a2, []).append(a1)
							k = (min(a1, a2), max(a1, a2))
							bords[k] = bords.get(k, 0) + 1
					normed = {}
					for (a, b), bo in bords.items():
						if bo > 3: bo = 3
						normed[(a, b)] = bo; normed[(b, a)] = bo
					bords = normed
		elif ext == '.cif':
			loops = []; cur_cols = []; cur_rows = []
			in_loop = False; in_semi = False; semi_buf = []
			for line in lines:
				s = line.strip()
				if in_semi:
					if s == ';':
						in_semi = False
						if in_loop and cur_rows and cur_cols:
							cur_rows[-1].append('\n'.join(semi_buf))
						semi_buf = []
					else: semi_buf.append(line.rstrip())
					continue
				if s.startswith(';'):
					in_semi = True; semi_buf = [s[1:]]; continue
				if s.startswith('loop_'):
					if in_loop and cur_cols:
						loops.append((cur_cols, cur_rows))
					cur_cols = []; cur_rows = []; in_loop = True; continue
				if in_loop and s.startswith('_'):
					cur_cols.append(s); continue
				if in_loop and cur_cols:
					if not s or s.startswith('#') or s.startswith('loop_'):
						loops.append((cur_cols, cur_rows))
						cur_cols = []; cur_rows = []
						in_loop = s.startswith('loop_'); continue
					tokens = []; i = 0; n = len(s)
					while i < n:
						if s[i] in (' ', '\t'): i += 1; continue
						if s[i] in ("'", '"'):
							q = s[i]; j = i + 1
							while j < n and s[j] != q: j += 1
							tokens.append(s[i+1:j]); i = j + 1
						else:
							j = i
							while j < n and s[j] not in (' ', '\t'): j += 1
							tokens.append(s[i:j]); i = j
					cur_rows.append(tokens)
			if in_loop and cur_cols: loops.append((cur_cols, cur_rows))
			APFX = ('_atom_site.', '_atom_site_', '_chem_comp_atom.')
			acols = arows = None
			for cols, rows in loops:
				if any(c.startswith(p) for c in cols for p in APFX):
					acols = cols; arows = rows; break
			if acols and arows:
				cm = {}
				for i, c in enumerate(acols):
					for p in APFX: c = c.replace(p, '')
					cm[c.lower()] = i
				xi = cm.get('cartn_x', cm.get('model_cartn_x', cm.get('x')))
				yi = cm.get('cartn_y', cm.get('model_cartn_y', cm.get('y')))
				zi = cm.get('cartn_z', cm.get('model_cartn_z', cm.get('z')))
				ei = cm.get('type_symbol', cm.get('symbol'))
				ni = cm.get('atom_id', cm.get('label_atom_id',
					cm.get('label', cm.get('id'))))
				if xi is None or yi is None or zi is None:
					raise ValueError('CIF: coordinate columns not found')
				name_map = {}
				for ri, r in enumerate(arows):
					ncol = max(xi, yi, zi,
						ei if ei is not None else 0,
						ni if ni is not None else 0) + 1
					if len(r) < ncol:
						raise ValueError(
							f'CIF atom row {ri}: expected {ncol}'
							f' fields, got {len(r)}')
					el = r[ei] if ei is not None else 'C'
					nm = r[ni] if ni is not None else f'{el}{idx}'
					x, y, z = float(r[xi]), float(r[yi]), float(r[zi])
					atoms[idx] = [nm, el, 0.0]
					coords.append([x, y, z])
					bonds[idx] = []; name_map[nm] = idx; idx += 1
			BPFX = ('_chem_comp_bond.',)
			for cols, rows in loops:
				if not any(c.startswith(p) for c in cols for p in BPFX):
					continue
				bm = {}
				for i, c in enumerate(cols):
					for p in BPFX: c = c.replace(p, '')
					bm[c.lower()] = i
				ai1 = bm.get('atom_id_1'); ai2 = bm.get('atom_id_2')
				voi = bm.get('value_order')
				if ai1 is None or ai2 is None: continue
				vom = {'SING':1, 'DOUB':2, 'TRIP':3, 'AROM':1.5}
				for r in rows:
					n1, n2 = r[ai1], r[ai2]
					if n1 not in name_map or n2 not in name_map: continue
					a1, a2 = name_map[n1], name_map[n2]
					bo = vom.get(r[voi], 1) if voi is not None else 1
					if a2 not in bonds[a1]: bonds[a1].append(a2)
					if a1 not in bonds[a2]: bonds[a2].append(a1)
					bords[(a1, a2)] = bo; bords[(a2, a1)] = bo
				break
		elif ext in ('.sdf', '.mol'):
			if len(lines) < 4:
				raise ValueError('SDF/MOL: file has fewer than 4 lines')
			hdr = lines[3]
			if 'V3000' in hdr:
				section = None
				for li, line in enumerate(lines):
					s = line.strip()
					if 'BEGIN ATOM' in s: section = 'ATOM'; continue
					elif 'END ATOM' in s: section = None; continue
					elif 'BEGIN BOND' in s: section = 'BOND'; continue
					elif 'END BOND' in s: section = None; continue
					if not s.startswith('M  V30'): continue
					p = s[7:].split()
					if section == 'ATOM':
						if len(p) < 5:
							raise ValueError(
								f'V3000 atom line {li}: expected'
								f' >=5 fields, got {len(p)}')
						el = p[1]
						x, y, z = float(p[2]), float(p[3]), float(p[4])
						atoms[idx] = [f'{el}{idx}', el, 0.0]
						coords.append([x, y, z])
						bonds[idx] = []; idx += 1
					elif section == 'BOND':
						if len(p) < 4:
							raise ValueError(
								f'V3000 bond line {li}: expected'
								f' >=4 fields, got {len(p)}')
						bt = int(p[1])
						a1, a2 = int(p[2]) - 1, int(p[3]) - 1
						if a2 not in bonds.get(a1, []):
							bonds.setdefault(a1, []).append(a2)
						if a1 not in bonds.get(a2, []):
							bonds.setdefault(a2, []).append(a1)
						bo = {1:1, 2:2, 3:3, 4:1.5}.get(bt, 1)
						bords[(a1, a2)] = bo; bords[(a2, a1)] = bo
			else:
				na, nb = int(hdr[0:3]), int(hdr[3:6])
				if len(lines) < 4 + na + nb:
					raise ValueError(
						f'SDF/MOL: file has {len(lines)} lines but'
						f' header declares {na} atoms + {nb} bonds')
				for i in range(na):
					ln = lines[4 + i]
					if len(ln) < 34:
						raise ValueError(
							f'SDF atom line {4+i}: too short'
							f' ({len(ln)} chars)')
					x = float(ln[0:10]); y = float(ln[10:20])
					z = float(ln[20:30]); el = ln[31:34].strip()
					atoms[idx] = [f'{el}{idx}', el, 0.0]
					coords.append([x, y, z]); bonds[idx] = []; idx += 1
				bt_map = {1:1, 2:2, 3:3, 4:1.5}
				for i in range(nb):
					ln = lines[4 + na + i]
					if len(ln) < 9:
						raise ValueError(
							f'SDF bond line {4+na+i}: too short'
							f' ({len(ln)} chars)')
					a1 = int(ln[0:3])-1; a2 = int(ln[3:6])-1
					bt = int(ln[6:9])
					if a2 not in bonds[a1]: bonds[a1].append(a2)
					if a1 not in bonds[a2]: bonds[a2].append(a1)
					bo = bt_map.get(bt, 1)
					bords[(a1, a2)] = bo; bords[(a2, a1)] = bo
		elif ext == '.mol2':
			section = None
			for line in lines:
				s = line.strip()
				if s.startswith('@<TRIPOS>'): section = s; continue
				if not s: continue
				if section == '@<TRIPOS>ATOM':
					p = s.split()
					if len(p) < 6:
						raise ValueError(
							f'MOL2 ATOM: expected >=6 fields, got {len(p)}')
					nm = p[1]
					x, y, z = float(p[2]), float(p[3]), float(p[4])
					el = p[5].split('.')[0]
					if len(el) > 1: el = el[0].upper() + el[1:].lower()
					ch = float(p[8]) if len(p) > 8 else 0.0
					atoms[idx] = [nm, el, ch]
					coords.append([x, y, z]); bonds[idx] = []; idx += 1
				elif section == '@<TRIPOS>BOND':
					p = s.split()
					if len(p) < 4:
						raise ValueError(
							f'MOL2 BOND: expected >=4 fields, got {len(p)}')
					a1, a2, bt = int(p[1])-1, int(p[2])-1, p[3]
					if a2 not in bonds.get(a1, []):
						bonds.setdefault(a1, []).append(a2)
					if a1 not in bonds.get(a2, []):
						bonds.setdefault(a2, []).append(a1)
					bo = {'1':1, '2':2, '3':3, 'ar':1.5, 'am':1}.get(bt, 1)
					bords[(a1, a2)] = bo; bords[(a2, a1)] = bo
		elif ext == 'string':
			hdr = lines[3] if len(lines) > 3 else ''
			if 'V2000' in hdr:
				na, nb = int(hdr[0:3]), int(hdr[3:6])
				for i in range(na):
					ln = lines[4 + i]
					x = float(ln[0:10]); y = float(ln[10:20])
					z = float(ln[20:30]); el = ln[31:34].strip()
					atoms[idx] = [f'{el}{idx}', el, 0.0]
					coords.append([x, y, z]); bonds[idx] = []; idx += 1
				bt_map = {1:1, 2:2, 3:3, 4:1.5}
				for i in range(nb):
					ln = lines[4 + na + i]
					a1 = int(ln[0:3])-1; a2 = int(ln[3:6])-1
					bt = int(ln[6:9])
					bonds[a1].append(a2); bonds[a2].append(a1)
					bo = bt_map.get(bt, 1)
					bords[(a1, a2)] = bo; bords[(a2, a1)] = bo
		else: raise Exception(f'Unsupported format: {ext}')
		has_bonds = any(bonds[i] for i in bonds)
		if not has_bonds and ext in ('.pdb', '.cif'):
			c = np.array(coords); n = len(c)
			if n > 0:
				els = [atoms[i][1] for i in range(n)]
				MAX_D = 2.2
				diff = c[:, None, :] - c[None, :, :]
				d2 = np.einsum('ijk,ijk->ij', diff, diff)
				mask = np.triu(d2 < MAX_D * MAX_D, k=1)
				ii, jj = np.where(mask)
				pairs = list(zip(ii.tolist(), jj.tolist()))
				for i, j in pairs:
					ei, ej = els[i], els[j]
					th = 1.9
					if ei == 'H' or ej == 'H': th = 1.3
					elif ei in ('S', 'Se') or ej in ('S', 'Se'): th = 2.1
					d = np.linalg.norm(c[i] - c[j])
					if not (0 < d < th): continue
					bonds[i].append(j); bonds[j].append(i)
					bo = 1; pr = tuple(sorted([ei, ej]))
					if pr == ('C', 'C'):
						if d < 1.27: bo = 3
						elif d < 1.42: bo = 2
					elif pr == ('C', 'N'):
						if d < 1.20: bo = 3
						elif d < 1.35: bo = 2
					elif pr == ('C', 'O'):
						if d < 1.30: bo = 2
					elif pr == ('N', 'N'):
						if d < 1.30: bo = 2
					elif pr == ('N', 'O'):
						if d < 1.30: bo = 2
					bords[(i, j)] = bo; bords[(j, i)] = bo
		self.data['Atoms'] = atoms
		self.data['Bonds'] = bonds
		self.data['Coordinates'] = (
			np.array(coords) if coords else np.zeros((0, 3)))
		self._bond_orders = bords
		self._formal_charges = {}
		self.CalcCharge(); self.CalcMass(); self.CalcRg()
		counts = {}
		for v in self.data['Atoms'].values():
			counts[v[1]] = counts.get(v[1], 0) + 1
		parts = []
		for e in ('C', 'H'):
			if e in counts:
				n = counts.pop(e)
				parts.append(e + (str(n) if n > 1 else ''))
		for e in sorted(counts):
			n = counts[e]
			parts.append(e + (str(n) if n > 1 else ''))
		self.data['Formula'] = ''.join(parts)
		self.CalcSMILES()
		self.CalcSMARTS()
	def Export(self, filename):
		ext = os.path.splitext(filename)[1].lower()
		A = self.data['Atoms']
		C = self.data['Coordinates']
		B = self.data['Bonds']
		bp = {(i, j) for i in sorted(B) for j in B[i] if i < j}
		with open(filename, 'w') as f:
			if ext == '.pdb':
				if len(A) > 99999:
					raise ValueError(
						f'PDB format: {len(A)} atoms'
						f' exceeds 99999 serial limit')
				for i in sorted(A):
					a = A[i]; c = C[i]; nm = a[0]
					if len(nm) < 4: nm = ' ' + nm
					f.write(
						f'HETATM{i+1:>5} {nm:<4} LIG A   1    '
						f'{c[0]:>8.4f}{c[1]:>8.4f}{c[2]:>8.4f}'
						f'  1.00  0.00          {a[1]:>2}\n')
				for i in sorted(B):
					row = [i+1] + [j+1 for j in B[i]]
					for k in range(0, len(row), 5):
						chunk = row[k:k+5]
						line = f'CONECT{chunk[0]:>5}'
						for v in chunk[1:]: line += f'{v:>5}'
						f.write(line + '\n')
				f.write('END\n')
			elif ext == '.cif':
				f.write('data_molecule\nloop_\n')
				for h in ('_atom_site.id', '_atom_site.type_symbol',
					'_atom_site.label_atom_id', '_atom_site.Cartn_x',
					'_atom_site.Cartn_y', '_atom_site.Cartn_z'):
					f.write(h + '\n')
				for i in sorted(A):
					a = A[i]; c = C[i]
					f.write(f'{i+1} {a[1]} {a[0]} '
						f'{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n')
				if bp:
					bm = {1:'SING', 2:'DOUB', 3:'TRIP', 1.5:'AROM'}
					f.write('loop_\n')
					for h in ('_chem_comp_bond.atom_id_1',
						'_chem_comp_bond.atom_id_2',
						'_chem_comp_bond.value_order'):
						f.write(h + '\n')
					for i, j in sorted(bp):
						bo = self._bond_orders.get((i, j), 1)
						f.write(f'{A[i][0]} {A[j][0]} {bm.get(bo, "SING")}\n')
			elif ext in ('.sdf', '.mol'):
				na, nb = len(A), len(bp)
				if na > 999 or nb > 999:
					raise ValueError(
						f'SDF V2000 limit: {na} atoms,'
						f' {nb} bonds (max 999 each)')
				nm = self.data['SMILES'] or ''
				f.write(f'{nm}\n     Molecule\n\n')
				f.write(f'{na:>3}{nb:>3}  0  0  0  0  0'
					'  0  0  0999 V2000\n')
				for i in sorted(A):
					a = A[i]; c = C[i]
					f.write(f'{c[0]:>10.4f}{c[1]:>10.4f}{c[2]:>10.4f}'
						f' {a[1]:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n')
				bm = {1:1, 2:2, 3:3, 1.5:4}
				for i, j in sorted(bp):
					bo = self._bond_orders.get((i, j), 1)
					f.write(f'{i+1:>3}{j+1:>3}{bm.get(bo, 1):>3}'
						'  0  0  0  0\n')
				f.write('M  END\n$$$$\n')
			elif ext == '.mol2':
				na, nb = len(A), len(bp)
				mbo = {i: 0 for i in A}
				for (a1, a2), bo in self._bond_orders.items():
					if a1 in mbo and bo > mbo[a1]: mbo[a1] = bo
					if a2 in mbo and bo > mbo[a2]: mbo[a2] = bo
				f.write('@<TRIPOS>MOLECULE\n')
				nm = self.data['SMILES'] or 'MOL'
				f.write(f'{nm}\n{na} {nb}\nSMALL\n\n@<TRIPOS>ATOM\n')
				for i in sorted(A):
					a = A[i]; c = C[i]; el = a[1]; mb = mbo.get(i, 0)
					if el in ('C', 'N', 'O', 'S'):
						if mb >= 3: st = el + '.1'
						elif mb == 1.5: st = el + '.ar'
						elif mb >= 2: st = el + '.2'
						else: st = el + '.3'
					else: st = el
					f.write(f'{i+1:>4} {a[0]:<4} {c[0]:>10.4f}'
						f' {c[1]:>10.4f} {c[2]:>10.4f}'
						f' {st:<6} 1 LIG {a[2]:.4f}\n')
				f.write('@<TRIPOS>BOND\n')
				bm = {1:'1', 2:'2', 3:'3', 1.5:'ar'}
				for bi, (i, j) in enumerate(sorted(bp), 1):
					bo = self._bond_orders.get((i, j), 1)
					f.write(f'{bi:>4} {i+1:>4} {j+1:>4}'
						f' {bm.get(bo, "1")}\n')
			else:
				raise Exception(f'Unsupported format: {ext}')
	def CalcSMILES(self):
		A = self.data['Atoms']; B = self.data['Bonds']
		heavy = sorted(i for i, v in A.items() if v[1] != 'H')
		if not heavy: self.data['SMILES'] = ''; return ''
		hs = set(heavy)
		adj = {i: [j for j in B.get(i, []) if j in hs] for i in heavy}
		hcount = {i: sum(1 for j in B.get(i, [])
			if A[j][1] == 'H') for i in heavy}
		visited = set(); parent = {}
		children = {i: [] for i in heavy}; roots = []
		for s in heavy:
			if s in visited: continue
			roots.append(s); visited.add(s); stk = [s]
			while stk:
				node = stk[-1]; pushed = False
				for nb in adj[node]:
					if nb not in visited:
						visited.add(nb); parent[nb] = node
						children[node].append(nb)
						stk.append(nb); pushed = True; break
				if not pushed: stk.pop()
		te = {frozenset((n, p)) for n, p in parent.items()}
		back = [(i, j) for i in heavy for j in adj[i]
			if i < j and frozenset((i, j)) not in te]
		ring_at = {}
		for d, (a, b) in enumerate(back, 1):
			bo = self._bond_orders.get((a, b),
				self._bond_orders.get((b, a), 1))
			ring_at.setdefault(a, []).append((d, bo))
			ring_at.setdefault(b, []).append((d, bo))
		seen_d = set()
		parts = []
		for root in roots:
			out = []
			stack = [('V', root)]
			while stack:
				op, arg = stack.pop()
				if op == 'S':
					out.append(arg)
					continue
				node = arg
				el = A[node][1]; nh = hcount[node]
				q = self._formal_charges.get(node, 0)
				tok = '[' + el
				if nh: tok += 'H' + (str(nh) if nh > 1 else '')
				if q > 0: tok += '+' + (str(q) if q > 1 else '')
				elif q < 0: tok += '-' + (str(-q) if -q > 1 else '')
				tok += ']'
				for d, bo in ring_at.get(node, []):
					if d not in seen_d:
						if bo == 2: tok += '='
						elif bo == 3: tok += '#'
						elif bo == 1.5: tok += ':'
					seen_d.add(d)
					tok += str(d) if d < 10 else f'%{d:02d}'
				out.append(tok)
				ch_list = children[node]
				for i in range(len(ch_list) - 1, -1, -1):
					c = ch_list[i]
					bo = self._bond_orders.get((node, c),
						self._bond_orders.get((c, node), 1))
					bsym = ''
					if bo == 2: bsym = '='
					elif bo == 3: bsym = '#'
					elif bo == 1.5: bsym = ':'
					if i < len(ch_list) - 1:
						stack.append(('S', ')'))
						stack.append(('V', c))
						stack.append(('S', bsym))
						stack.append(('S', '('))
					else:
						stack.append(('V', c))
						stack.append(('S', bsym))
			parts.append(''.join(out))
		result = '.'.join(parts)
		self.data['SMILES'] = result; return result
	def CalcSMARTS(self):
		ANUM = {k: i+1 for i, k in enumerate(self.masses)}
		A = self.data['Atoms']; B = self.data['Bonds']
		heavy = sorted(i for i, v in A.items() if v[1] != 'H')
		if not heavy: self.data['SMARTS'] = ''; return ''
		hs = set(heavy)
		adj = {i: [j for j in B.get(i, []) if j in hs] for i in heavy}
		visited = set(); parent = {}
		children = {i: [] for i in heavy}; roots = []
		for s in heavy:
			if s in visited: continue
			roots.append(s); visited.add(s); stk = [s]
			while stk:
				node = stk[-1]; pushed = False
				for nb in adj[node]:
					if nb not in visited:
						visited.add(nb); parent[nb] = node
						children[node].append(nb)
						stk.append(nb); pushed = True; break
				if not pushed: stk.pop()
		te = {frozenset((n, p)) for n, p in parent.items()}
		back = [(i, j) for i in heavy for j in adj[i]
			if i < j and frozenset((i, j)) not in te]
		ring_at = {}
		for d, (a, b) in enumerate(back, 1):
			bo = self._bond_orders.get((a, b),
				self._bond_orders.get((b, a), 1))
			ring_at.setdefault(a, []).append((d, bo))
			ring_at.setdefault(b, []).append((d, bo))
		seen_d = set()
		parts = []
		for root in roots:
			out = []
			stack = [('V', root)]
			while stack:
				op, arg = stack.pop()
				if op == 'S':
					out.append(arg)
					continue
				node = arg
				el = A[node][1]; an = ANUM.get(el, 0)
				tok = f'[#{an}]' if an else f'[{el}]'
				for d, bo in ring_at.get(node, []):
					if d not in seen_d:
						if bo == 1: tok += '-'
						elif bo == 2: tok += '='
						elif bo == 3: tok += '#'
						elif bo == 1.5: tok += ':'
					seen_d.add(d)
					tok += str(d) if d < 10 else f'%{d:02d}'
				out.append(tok)
				ch_list = children[node]
				for i in range(len(ch_list) - 1, -1, -1):
					c = ch_list[i]
					bo = self._bond_orders.get((node, c),
						self._bond_orders.get((c, node), 1))
					bsym = '-'
					if bo == 2: bsym = '='
					elif bo == 3: bsym = '#'
					elif bo == 1.5: bsym = ':'
					if i < len(ch_list) - 1:
						stack.append(('S', ')'))
						stack.append(('V', c))
						stack.append(('S', bsym))
						stack.append(('S', '('))
					else:
						stack.append(('V', c))
						stack.append(('S', bsym))
			parts.append(''.join(out))
		result = '.'.join(parts)
		self.data['SMARTS'] = result; return result
	def CalcRg(self):
		A = self.data['Atoms']
		if not A: self.data['Rg'] = 0.0; return
		mass = np.array([self.masses.get(A[i][1], 0.0) for i in sorted(A)])
		tm = mass.sum()
		if tm == 0: self.data['Rg'] = 0.0; return
		co = self.data['Coordinates']
		if len(mass) != co.shape[0]:
			raise ValueError(
				f'CalcRg: {len(mass)} atoms but {co.shape[0]} coordinates')
		xm = co * mass[:, np.newaxis]
		rr = np.sum(co * xm)
		mm = np.sum((xm.sum(0) / tm) ** 2)
		self.data['Rg'] = round(math.sqrt(max(0.0, rr / tm - mm)), 3)
	def CalcCharge(self, iterations=6):
		PARAMS = {
			'C3':(7.98,  9.18,  1.88), 'C2':(8.79,  9.32,  1.51),
			'C1':(10.39, 9.45,  0.73), 'H' :(7.17,  6.24, -0.56),
			'O3':(14.18, 12.92, 1.39), 'O2':(17.07, 13.79, 0.47),
			'N3':(11.54, 10.82, 1.36), 'N2':(12.87, 11.15, 0.85),
			'S' :(10.14, 9.13,  1.38), 'Se':(9.00,  8.00,  1.10),
			'F' :(14.66, 13.85, 2.31), 'Cl':(11.00, 9.69,  1.35),
			'Br':(10.08, 8.47,  1.16), 'I' :(9.90,  7.96,  0.96),
			'P' :(8.90,  8.24,  0.96), 'B' :(5.80,  6.00,  1.56)}
		A = self.data['Atoms']; B = self.data['Bonds']
		ids = sorted(A.keys())
		if not ids: return
		mbo = {i: 0 for i in ids}
		for (a, b), bo in self._bond_orders.items():
			if a in mbo and bo > mbo[a]: mbo[a] = bo
			if b in mbo and bo > mbo[b]: mbo[b] = bo
		atype = {}
		for i in ids:
			el = A[i][1].upper()
			if el == 'H': atype[i] = PARAMS['H']
			elif el in ('S', 'SE'):
				atype[i] = PARAMS.get(el.capitalize(), PARAMS['S'])
			elif el == 'C':
				if mbo[i] >= 3: atype[i] = PARAMS['C1']
				elif mbo[i] >= 1.5: atype[i] = PARAMS['C2']
				else: atype[i] = PARAMS['C3']
			elif el == 'N':
				atype[i] = PARAMS['N2'] if mbo[i] >= 1.5 else PARAMS['N3']
			elif el == 'O':
				atype[i] = PARAMS['O2'] if mbo[i] >= 1.5 else PARAMS['O3']
			elif el in ('F', 'CL', 'BR', 'I'):
				atype[i] = PARAMS.get(el.capitalize(), PARAMS['Cl'])
			elif el == 'P': atype[i] = PARAMS['P']
			elif el == 'B': atype[i] = PARAMS['B']
			else: atype[i] = None
		charges = {i: 0.0 for i in ids}
		for n in range(iterations):
			damp = 1.0 / (2 ** (n + 1))
			chi = {}
			for i in ids:
				if atype[i] is None: chi[i] = 0.0; continue
				a, b, c = atype[i]; qv = charges[i]
				chi[i] = a + qv * (b + c * qv)
			delta = {i: 0.0 for i in ids}
			for i in ids:
				for j in B.get(i, []):
					if j <= i: continue
					if atype[i] is None or atype[j] is None: continue
					if chi[j] >= chi[i]: donor, acc = i, j
					else: donor, acc = j, i
					a, b, c = atype[donor]; ip = a + b + c
					if ip == 0: continue
					dq = damp * (chi[acc] - chi[donor]) / ip
					delta[donor] += dq; delta[acc] -= dq
			for i in ids: charges[i] += delta[i]
		for i in ids: A[i][2] = round(charges[i], 4)
	def CalcMass(self):
		A = self.data['Atoms']
		self.data['Mass'] = round(
			sum(self.masses.get(v[1], 0.0) for v in A.values()), 3)
	def GetDistance(self, idx1, idx2):
		self._check_idx(idx1, idx2)
		return np.linalg.norm(
			self.data['Coordinates'][idx2] - self.data['Coordinates'][idx1])
	def GetAngle(self, idx1, idx2, idx3):
		self._check_idx(idx1, idx2, idx3)
		C = self.data['Coordinates']
		a, b = C[idx2] - C[idx1], C[idx2] - C[idx3]
		d = np.linalg.norm(a) * np.linalg.norm(b)
		if d < 1e-10: return 0.0
		return math.degrees(math.acos(max(-1.0, min(1.0, np.dot(a, b) / d))))
	def GetDihedral(self, idx1, idx2, idx3, idx4):
		self._check_idx(idx1, idx2, idx3, idx4)
		C = self.data['Coordinates']
		u1, u2, u3 = C[idx2]-C[idx1], C[idx3]-C[idx2], C[idx4]-C[idx3]
		c12, c23 = np.cross(u1, u2), np.cross(u2, u3)
		return math.atan2(
			np.dot(u2, np.cross(c12, c23)),
			np.linalg.norm(u2) * np.dot(c12, c23)) * 180 / math.pi
	def GetAtomCoord(self, idx):
		self._check_idx(idx)
		return self.data['Coordinates'][idx]
	def GetAtomList(self):
		return [v[1] for v in self.data['Atoms'].values()]
	def GetAtomBonds(self, idx):
		A = self.data['Atoms']
		return [A[j][0] for j in self.data['Bonds'].get(idx, [])]
	def AdjustDistance(self, idx1, idx2, length):
		self._check_idx(idx1, idx2)
		C = self.data['Coordinates']
		v = C[idx2] - C[idx1]; mg = np.linalg.norm(v)
		if mg < 1e-10: return
		shift = v * (length / mg) - v
		for i in self._downstream(idx1, idx2): C[i] += shift
		self._invalidate()
	def AdjustAngle(self, idx1, idx2, idx3, theta):
		self._check_idx(idx1, idx2, idx3)
		C = self.data['Coordinates']
		u = np.cross(C[idx3] - C[idx2], C[idx3] - C[idx1])
		lu = np.linalg.norm(u)
		if lu < 1e-10: return
		ori = C[idx2].copy(); RM = self._rotmat(theta, u / lu)
		for i in self._downstream(idx2, idx3):
			C[i] = np.matmul(C[i] - ori, RM) + ori
		self._invalidate()
	def RotateDihedral(self, idx1, idx2, idx3, idx4, theta):
		self._check_idx(idx1, idx2, idx3, idx4)
		C = self.data['Coordinates']
		current = self.GetDihedral(idx1, idx2, idx3, idx4)
		ori = C[idx3].copy()
		u = C[idx2] - C[idx3]; mg = np.linalg.norm(u)
		if mg < 1e-10: return
		u = u / mg
		RM_zero = self._rotmat(-current, u)
		RM_new = self._rotmat(theta, u)
		for i in self._downstream(idx2, idx3):
			v = np.matmul(C[i] - ori, RM_zero)
			C[i] = np.matmul(v, RM_new) + ori
		self._invalidate()
	def MovePose(self, theta=None, u=None, l=None, ori=None):
		C = self.data['Coordinates'].copy()
		if len(C) == 0: return
		rot_args   = (theta is not None, u is not None)
		trans_args = (l is not None, ori is not None)
		if any(rot_args) and not all(rot_args):
			raise Exception(
				'MovePose rotation requires BOTH theta and u')
		if any(trans_args) and not all(trans_args):
			raise Exception(
				'MovePose translation requires BOTH l and ori')
		if not any(rot_args) and not any(trans_args):
			raise Exception(
				'MovePose called with no arguments (nothing to do)')
		if theta is not None and u is not None:
			u = np.array(u, dtype=float); mg = np.linalg.norm(u)
			if mg > 1e-10:
				pivot = C.mean(axis=0)
				R = self._rotmat(theta, u / mg)
				C = np.matmul(C - pivot, R) + pivot
		if l is not None and ori is not None:
			ori = np.array(ori, dtype=float)
			d = ori - C.mean(axis=0); mg = np.linalg.norm(d)
			if mg > 1e-10: C = C + (d / mg) * l
		self.data['Coordinates'] = C
		self._invalidate()
	def GetInfo(self):
		d = self.data
		print(f"Energy:  {d['Energy']}")
		print(f"Mass:    {d['Mass']} Da")
		print(f"Rg:      {d['Rg']} A")
		print(f"Formula: {d['Formula']}")
		print(f"SMILES:  {d['SMILES']}")
		print(f"SMARTS:  {d['SMARTS']}")
		A = d['Atoms']; B = d['Bonds']
		heavy = [i for i, v in A.items() if v[1] != 'H']
		if len(heavy) < 2 or len(heavy) >= 200: return
		Cr = d['Coordinates']; hs = set(heavy)
		hc = Cr[heavy]; hc0 = hc - hc.mean(axis=0)
		_, _, Vt = np.linalg.svd(hc0, full_matrices=False)
		proj = hc0 @ Vt[:2].T
		hi = {v: k for k, v in enumerate(heavy)}
		bls = [np.linalg.norm(proj[hi[i]] - proj[hi[j]])
			for i in heavy for j in B.get(i, []) if j in hs and j > i]
		mbl = np.median(bls) if bls else 1.0
		if mbl < 0.01: mbl = 1.0
		sc = 14.0 / mbl
		spx = {i: proj[k, 0] * sc for k, i in enumerate(heavy)}
		spy = {i: proj[k, 1] * sc for k, i in enumerate(heavy)}
		xmn, ymx = min(spx.values()), max(spy.values())
		for i in heavy:
			spx[i] = int(round(spx[i] - xmn)) + 4
			spy[i] = int(round(ymx - spy[i])) + 4
		PW = max(spx.values()) + 5; PH = max(spy.values()) + 5
		canvas = [[False] * PW for _ in range(PH)]
		drawn = set()
		for i in heavy:
			for j in B.get(i, []):
				if j not in hs or j <= i or (i, j) in drawn: continue
				drawn.add((i, j))
				x0, y0 = spx[i], spy[i]; x1, y1 = spx[j], spy[j]
				bo = self._bond_orders.get((i, j),
					self._bond_orders.get((j, i), 1))
				bx, by = x1 - x0, y1 - y0
				mg = math.sqrt(bx * bx + by * by)
				if mg > 0: pdx, pdy = -by / mg, bx / mg
				else: pdx, pdy = 0.0, 1.0
				offs = [(0, 0)]
				if bo >= 2: offs.append((pdx * 2, pdy * 2))
				if bo >= 3: offs.append((-pdx * 2, -pdy * 2))
				for ox, oy in offs:
					ax0, ay0 = int(round(x0+ox)), int(round(y0+oy))
					ax1, ay1 = int(round(x1+ox)), int(round(y1+oy))
					ddx, ddy = abs(ax1 - ax0), abs(ay1 - ay0)
					ssx = (ax1 > ax0) - (ax1 < ax0)
					ssy = (ay1 > ay0) - (ay1 < ay0)
					if not ddx and not ddy:
						continue
					err = ddx - ddy; cx, cy = ax0, ay0
					while True:
						if 0 <= cx < PW and 0 <= cy < PH:
							canvas[cy][cx] = True
						if cx == ax1 and cy == ay1: break
						e2 = 2 * err
						if e2 > -ddy: err -= ddy; cx += ssx
						if e2 < ddx: err += ddx; cy += ssy
		DOTS = {(0,0):0x01, (1,0):0x08, (0,1):0x02, (1,1):0x10,
			(0,2):0x04, (1,2):0x20, (0,3):0x40, (1,3):0x80}
		CW = (PW + 1) // 2; CH = (PH + 3) // 4
		grid = [[' '] * CW for _ in range(CH)]
		colr = [['' ] * CW for _ in range(CH)]
		for cy in range(CH):
			for cx in range(CW):
				code = 0
				for (dx, dy), bit in DOTS.items():
					px, py = cx * 2 + dx, cy * 4 + dy
					if px < PW and py < PH and canvas[py][px]: code |= bit
				if code: grid[cy][cx] = chr(0x2800 + code)
		GRY = '\033[90m'; RST = '\033[0m'
		for cy in range(CH):
			for cx in range(CW):
				if grid[cy][cx] != ' ': colr[cy][cx] = GRY
		ACOL = {'C':'\033[32m', 'N':'\033[34m', 'O':'\033[31m',
			'P':'\033[38;5;208m', 'S':'\033[33m', 'H':'\033[37m'}
		DCOL = '\033[35m'
		for i in heavy:
			cx, cy = spx[i] // 2, spy[i] // 4
			if cy < 0 or cy >= CH: continue
			c = ACOL.get(A[i][1], DCOL)
			for ci, ch in enumerate(str(i)):
				px = cx + ci
				if 0 <= px < CW:
					grid[cy][px] = ch; colr[cy][px] = c
		for r in range(CH):
			out = []
			for c in range(CW):
				cc, ch = colr[r][c], grid[r][c]
				out.append(cc + ch + RST if cc else ch)
			line = ''.join(out).rstrip()
			if line: print(line)
