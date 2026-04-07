#!/usr/bin/env python3

import os
import json
import math
import copy
import datetime
import numpy as np
from collections import defaultdict

np.seterr(all='ignore')

class Pose():
	''' A class that builds and manipulates protein, DNA, and RNA '''
	def __init__(self):
		path, modulename = os.path.split(__file__)
		with open(f'{path}/database.json') as f: database = json.load(f)
		self.aminoacids=database['Amino Acids']
		self.nucleotides=database['Nucleotides']
		self.probbatoms = {'N', '1H', '2H', '3H', 'CA', 'HA', 'C', 'O', 'OXT'}
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
		self.data = {'Type':None, 'Energy':0, 'Rg':0, 'Mass':0, 'Size':0,
		'FASTA':{}, 'SS':{}, 'Nucleotides':None, 'Amino Acids':None,
		'Atoms':{}, 'Bonds':{}, 'Coordinates':np.array([[0, 0, 0]])}
	def _isfused(self, SC):
		''' Check if an amino acid's sidechain is fused to the backbone '''
		return self.aminoacids[SC.upper()]['Fused']
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
			BBb = nBBb
		else:
			n = 1
			nBBb[0][0] = length-1
			BBb = nBBb
		for i in reversed(range(len(BBb))):
			if i > n+1:
				oldvals = BBb[i]
				newvals = [x if x<=n else x+length-1 for x in BBb[i]]
				oldk = i
				newk = i+length-1
				del BBb[i]
				BBb[newk] = newvals
		BBb[n].append(n+1+length)
		for i, (k, v) in enumerate(SCb.items(), start=n+2):
			if k < 0: break
			k = i
			if BB == 'Backbone' or BB == 'Backbone start':
				v = [x+n+2 for x in v]
			else:
				v = [x+n+4 if x<0 else x+n+2 for x in v]
			if i == n+2: v.append(n)
			BBb[k] = sorted(v)
		return BBb
	def _bondtreenotfused(self, BB, SC):
		''' Construct amino acid bond graph by adding sidechain to backbone '''
		SC = SC.upper()
		if self._isfused(SC):
			BBb = self._bondtreefused(BB, SC)
			return BBb
		BBb = {int(k): v for k, v in
			copy.deepcopy(self.aminoacids[BB]['Bonds']).items()}
		SCb = {int(k): v for k, v in
			copy.deepcopy(self.aminoacids[SC]['Bonds']).items()}
		length = len(SCb)
		if BB == 'Backbone' or BB == 'Backbone start':
			n = 4
		else:
			n = 2
		for i in reversed(range(len(BBb))):
			if i > n+1:
				oldvals = BBb[i]
				newvals = [x if x<=n else x+length for x in BBb[i]]
				oldk = i
				newk = i+length
				del BBb[i]
				BBb[newk] = newvals
		BBb[n].append(n + 2 + length)
		for i, (k, v) in enumerate(SCb.items(), start=n+2):
			k = i
			if length != 1:
				v = [x+n+2 for x in v]
				if i == n+2: v.append(n)
			else:
				v = [x+n+1 for x in v]
			BBb[k] = sorted(v)
		return BBb
	def _bondtree(self, BB, AA, new_chain=False):
		''' Update the pose bond graph when adding a new amino acid '''
		BBb = self._bondtreenotfused(BB, AA)
		BT = self.data['Bonds']
		length = len(BT)
		if length == 0:
			self.data['Bonds'] = BBb
			return
		i_max = max(BT)
		if not new_chain:
			BT[i_max-1] += [i_max+1]
		for i in range(len(BBb)):
			K = i+length
			v = BBb[i]
			V = [x+length for x in v]
			if i == 0 and not new_chain:
				V.append(i_max-1)
			BT[K] = V
		self.data['Bonds'] = BT







	def _update(self):
		''' Update cached properties after structural changes '''
		self.data['Mass']  = self.GetMass()
		self.data['FASTA'] = self.GetFASTA()
		self.data['Size']  = self.GetSize()
		self.data['Rg']    = self.GetRg()
		if self.data['Type'] == 'Protein':
			self.data['SS'] = self.GetSS()
		else:
			self.data['SS'] = {}





	def GetMass(self):
		''' Calculate mass of peptide in Da'''
		mass = sum(self.masses[x] for x in self.GetAtomList())
		return round(mass, 3)
	def GetSize(self):
		''' Calculate length of molecule (first chain) '''
		source = (self.data['Amino Acids']
			or self.data['Nucleotides'])
		if not source: return 0
		first_chain = source[0][1]
		return sum(
			1 for v in source.values()
			if v[1] == first_chain)
	def GetFASTA(self):
		''' Return per-chain FASTA dict '''
		source = (self.data['Amino Acids']
			or self.data['Nucleotides'])
		if not source: return {}
		fasta = {}
		for v in source.values():
			fasta.setdefault(v[1], []).append(v[0])
		return {k: ''.join(v) for k, v in fasta.items()}
	def GetSS(self):
		''' Return per-chain secondary structure dict '''
		if self.data['Amino Acids'] is None: return {}
		ss = {}
		for v in self.data['Amino Acids'].values():
			ss.setdefault(v[1], []).append(v[4])
		return {k: ''.join(v) for k, v in ss.items()}
	def GetRg(self):
		''' Calculate the radius of gyration of a peptide '''
		mass  = np.array([self.masses[e] for e in self.GetAtomList()])
		tmass = mass.sum()
		if tmass == 0: raise ZeroDivisionError('No atoms in pose')
		coord = self.data['Coordinates']
		xm    = coord * mass[:, np.newaxis]
		rr    = np.sum(coord * xm)
		mm    = np.sum((xm.sum(0) / tmass) ** 2)
		return round(math.sqrt(rr / tmass - mm), 3)


	def GetAtomList(self, PDB=False):
		''' Return list of all the atoms '''
		idx = 0 if PDB else 1
		return [x[idx] for x in self.data['Atoms'].values()]
	def GetAtomCoord(self, res, atom):
		''' Get specific atom coordinates '''
		source = (self.data['Amino Acids']
			or self.data['Nucleotides'])
		info = source[res]
		for i in info[2] + info[3]:
			if self.data['Atoms'][i][0] == atom:
				return self.data['Coordinates'][i]
		raise Exception(
			f'Atom {atom} not found in residue {res}')
	def Angle(self, AA, angle_type, chi_type=None):
		''' Measure phi/psi/omega/chi dihedral angle '''
		at = angle_type.upper()
		if at == 'PHI':
			if AA == 0: return 0.0
			AAs = self.data['Amino Acids']
			if AAs[AA][1] != AAs[AA-1][1]:
				return 0.0
			r1 = self.GetAtomCoord(AA-1, 'C')
			r2 = self.GetAtomCoord(AA, 'N')
			r3 = self.GetAtomCoord(AA, 'CA')
			r4 = self.GetAtomCoord(AA, 'C')
		elif at == 'PSI':
			r1 = self.GetAtomCoord(AA, 'N')
			r2 = self.GetAtomCoord(AA, 'CA')
			r3 = self.GetAtomCoord(AA, 'C')
			AAs = self.data['Amino Acids']
			if (AA+1 not in AAs
			or AAs[AA][1] != AAs[AA+1][1]):
				return 0.0
			r4 = self.GetAtomCoord(AA+1, 'N')
		elif at == 'OMEGA':
			r1 = self.GetAtomCoord(AA, 'CA')
			r2 = self.GetAtomCoord(AA, 'C')
			AAs = self.data['Amino Acids']
			if (AA+1 not in AAs
			or AAs[AA][1] != AAs[AA+1][1]):
				return 180.0
			r3 = self.GetAtomCoord(AA+1, 'N')
			r4 = self.GetAtomCoord(AA+1, 'CA')
		elif at == 'CHI':
			aa_sym = self.data[
				'Amino Acids'][AA][0].upper()
			atoms = self.aminoacids[aa_sym][
				'Chi Angle Atoms'][chi_type-1]
			r1 = self.GetAtomCoord(AA, atoms[0])
			r2 = self.GetAtomCoord(AA, atoms[1])
			r3 = self.GetAtomCoord(AA, atoms[2])
			r4 = self.GetAtomCoord(AA, atoms[3])
		u1 = r2 - r1
		u2 = r3 - r2
		u3 = r4 - r3
		mag_u2 = np.linalg.norm(u2)
		u1u2 = np.cross(u1, u2)
		u2u3 = np.cross(u2, u3)
		u1u2Cu2u3 = np.cross(u1u2, u2u3)
		u1u2Du2u3 = np.dot(u1u2, u2u3)
		a = np.dot(u2, u1u2Cu2u3)
		b = mag_u2 * u1u2Du2u3
		return math.atan2(a, b) * 180 / math.pi


	def GetCharge(self, iterations=6):
		''' Calculate Gasteiger-Marsili partial charges to all atoms '''
		PARAMS = {
			'C3': (7.98,  9.18,  1.88),
			'C2': (8.79,  9.32,  1.51),
			'C1': (10.39, 9.45,  0.73),
			'H':  (7.17,  6.24, -0.56),
			'O3': (14.18, 12.92, 1.39),
			'O2': (17.07, 13.79, 0.47),
			'N3': (11.54, 10.82, 1.36),
			'N2': (12.87, 11.15, 0.85),
			'S':  (10.14,  9.13, 1.38),
			'Se': (9.00,   8.00, 1.10)}
		ids  = sorted(self.data['Atoms'].keys())
		crds = self.data['Coordinates']
		els  = [self.data['Atoms'][i][1].upper() for i in ids]
		c = crds[np.array(ids)]
		dm = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(2))
		is_H = np.array([e == 'H'         for e in els])
		is_S = np.array([e in ('S', 'SE') for e in els])
		heavy_thresh = np.full((len(ids), len(ids)), 1.9)
		h_mask = is_H[:, None] | is_H[None, :]
		s_mask = is_S[:, None] | is_S[None, :]
		heavy_thresh[h_mask] = 1.3
		heavy_thresh[s_mask] = 2.1
		bond_mask = (dm < heavy_thresh) & (dm > 0.0)
		bonds = {i: [] for i in ids}
		for ii, jj in np.argwhere(bond_mask):
			bonds[ids[ii]].append(ids[jj])
		heavy = ~is_H
		short = (dm < 1.42) & (dm > 0)
		sp2_mask = short & heavy[:, None] & heavy[None, :]
		sp2 = {ids[ii] for ii in np.where(sp2_mask.any(axis=1))[0]}
		def _types(ii, i):
			''' Classifies each atom into a Gasteiger atom type '''
			el = els[ii]
			if el == 'H':  return 'H'
			if el == 'S':  return 'S'
			if el == 'SE': return 'Se'
			nb    = len(bonds[i])
			isp2  = i in sp2
			if el == 'C':
				if isp2:    return 'C2'
				if nb <= 2: return 'C1'
				return 'C3'
			if el == 'N':
				return 'N2' if isp2 else 'N3'
			if el == 'O':
				return 'O2' if isp2 else 'O3'
			return 'C3'
		charges = {i: self.data['Atoms'][i][2] for i in ids}
		atype   = {i: PARAMS[_types(ii, i)] for ii, i in enumerate(ids)}
		for n in range(iterations):
			damp = 1.0 / (2 ** (n + 1))
			chi = {i: a + q*(b + c*q) for i in ids
				for a, b, c in [atype[i]] for q in [charges[i]]}
			delta = {i: 0.0 for i in ids}
			for i in ids:
				for j in bonds[i]:
					if j <= i:
						continue
					if chi[j] >= chi[i]:
						donor, acceptor = i, j
					else:
						donor, acceptor = j, i
					a, b, c = atype[donor]
					ip = a + b + c
					if ip == 0:
						continue
					dq = damp * (
						chi[acceptor] - chi[donor]) / ip
					delta[donor]    += dq
					delta[acceptor] -= dq
			for i in ids:
				charges[i] += delta[i]
		for i in ids:
			self.data['Atoms'][i][2] = round(charges[i], 4)







	def GetDSSP(self):
		''' Assign secondary structures to each amino acid '''
		N = len(self.data['Amino Acids'])
		AAs      = self.data['Amino Acids']
		chains   = [AAs[i][1]         for i in range(N)]
		tricodes = [AAs[i][5].upper() for i in range(N)]
		H_pos = [None] * N
		for i in range(N):
			if 'PRO' in tricodes[i]:
				continue
			if i == 0 or chains[i] != chains[i - 1]:
				continue
			for hname in ('1H', 'H'):
				try:
					H_pos[i] = self.GetAtomCoord(i, hname)
					break
				except Exception:
					pass
			else:
				try:
					Ni = self.GetAtomCoord(i, 'N')
					Cp = self.GetAtomCoord(i - 1, 'C')
					Op = self.GetAtomCoord(i - 1, 'O')
					co = Cp - Op
					nm = np.linalg.norm(co)
					if nm > 0.001:
						H_pos[i] = Ni + (co / nm)
				except Exception:
					pass
		hbond = set()
		for i in range(N):
			if H_pos[i] is None:
				continue
			try:
				Ni = self.GetAtomCoord(i, 'N')
			except Exception:
				continue
			Hi = H_pos[i]
			for j in range(N):
				if abs(i - j) <= 1:
					continue
				if chains[i] != chains[j]:
					continue
				try:
					Cj = self.GetAtomCoord(j, 'C')
					Oj = self.GetAtomCoord(j, 'O')
				except Exception:
					continue
				r_ON = np.linalg.norm(Oj - Ni)
				r_CH = np.linalg.norm(Cj - Hi)
				r_OH = np.linalg.norm(Oj - Hi)
				r_CN = np.linalg.norm(Cj - Ni)
				if min(r_ON, r_CH, r_OH, r_CN) < 0.001:
					continue
				E = 0.084 * (
					1/r_ON + 1/r_CH
					- 1/r_OH - 1/r_CN) * 332
				if E < -0.5:
					hbond.add((i, j))
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
					if ss[k] not in ('H', 'E', 'B'):
						ss[k] = 'I'
		for i in range(N - 1):
			if turn4[i] and turn4[i + 1]:
				for k in range(i + 1, min(i + 5, N)):
					if ss[k] != 'I':
						ss[k] = 'H'
		for i in range(N - 1):
			if turn3[i] and turn3[i + 1]:
				for k in range(i + 1, min(i + 4, N)):
					if ss[k] not in ('H', 'E', 'B'):
						ss[k] = 'G'
		bridges = set()
		for i in range(1, N - 1):
			for k in range(i + 2, N - 1):
				if chains[i] != chains[k]:
					continue
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
				if ap or pp:
					bridges.add((i, k))
		br_at = defaultdict(set)
		for i, k in bridges:
			br_at[i].add(k)
			br_at[k].add(i)
		for r in br_at:
			if ss[r] not in ('H', 'G', 'I'):
				ss[r] = 'B'
		changed = True
		while changed:
			changed = False
			for i in range(N - 1):
				if (ss[i] not in ('B', 'E')
				or ss[i+1] not in ('B', 'E')):
					continue
				for k1 in br_at.get(i, []):
					for k2 in br_at.get(i+1, []):
						if abs(k1 - k2) != 1:
							continue
						for r in (i, i+1, k1, k2):
							if ss[r] in (
							'H', 'G', 'I'):
								continue
							if ss[r] != 'E':
								ss[r] = 'E'
								changed = True
		for i in range(2, N - 2):
			if ss[i] != 'L':
				continue
			try:
				m2 = self.GetAtomCoord(i - 2, 'CA')
				ci = self.GetAtomCoord(i,     'CA')
				p2 = self.GetAtomCoord(i + 2, 'CA')
			except Exception:
				continue
			v1   = ci - m2
			v2   = p2 - ci
			n1   = np.linalg.norm(v1)
			n2   = np.linalg.norm(v2)
			if n1 < 0.001 or n2 < 0.001:
				continue
			cos_k = np.dot(v1, v2) / (n1 * n2)
			cos_k = max(-1.0, min(1.0, cos_k))
			kappa = math.acos(cos_k) * 180.0 / math.pi
			if kappa >= 70.0:
				ss[i] = 'S'
		PHI_LO, PHI_HI = -104.0, -46.0
		PSI_LO, PSI_HI =  116.0, 174.0
		ppii = [False] * N
		for i in range(N):
			if ss[i] != 'L':
				continue
			try:
				phi = self.Angle(i, 'PHI')
				psi = self.Angle(i, 'PSI')
			except Exception:
				continue
			ppii[i] = PHI_LO <= phi <= PHI_HI and PSI_LO <= psi <= PSI_HI
		for i in range(N - 2):
			if ppii[i] and ppii[i + 1] and ppii[i + 2]:
				for k in range(i, i + 3):
					if ss[k] == 'L':
						ss[k] = 'P'
		for i in range(N):
			AAs[i][4] = ss[i]








	def GetSASA(self, n_points=100, probe_radius=1.4):
		''' Calculate Solvent Accessible Surface Area per residue '''
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
		radii = np.array([
			VDW.get(atoms[i][1].upper(),DEFAULT_VDW)+probe_radius for i in ids])
		golden = (1 + np.sqrt(5)) / 2
		pts    = np.arange(n_points)
		theta  = np.arccos(1 - 2 * (pts + 0.5) / n_points)
		phi    = 2 * np.pi * pts / golden
		st  = np.sin(theta)
		sph = np.column_stack([st*np.cos(phi), st*np.sin(phi), np.cos(theta)])
		dm  = np.sqrt(((c[:, None, :] - c[None, :, :])**2).sum(2))
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





	def Import(self, filename, chain=None, model=1):
		''' Import Protein/DNA/RNA structures from a .pdb or .cif file '''
		ext = filename[-3:].upper()
		chains_to_load = chain
		rows = []
		if ext == 'PDB':
			has_models = False
			in_target = False
			found_models = []
			with open(filename) as f:
				for line in f:
					line = line.rstrip()
					if not line: continue
					rec = line[:6].strip()
					if rec == 'MODEL':
						has_models = True
						try:
							mnum = int(line.split()[1])
						except (IndexError, ValueError):
							mnum = len(found_models) + 1
						found_models.append(mnum)
						in_target = (mnum == model)
					elif rec == 'ENDMDL':
						if in_target:
							break
						in_target = False
					elif rec == 'ATOM':
						ch = line[21]
						if (chains_to_load is not None
						and ch not in chains_to_load):
							continue
						if has_models and not in_target:
							continue
						aname = line[12:16].strip()
						resn = line[17:20].strip()
						resnum = int(
							line[22:26].strip())
						x = float(line[30:38])
						y = float(line[38:46])
						z = float(line[46:54])
						occ = float(line[54:60])
						bfac = float(line[60:66])
						elem = line[76:78].strip()
						if not elem:
							elem = aname.lstrip(
								'0123456789')[0]
						rows.append((
							aname, resn, ch, resnum,
							x, y, z, occ, bfac, elem))
			if has_models and model not in found_models:
				raise Exception(
					f'Model {model} not found in '
					f'{filename}. '
					f'Available models: {found_models}')
		elif ext == 'CIF':
			with open(filename) as f: lines = f.readlines()
			col = {}
			col_idx = 0
			data_start = 0
			for idx, line in enumerate(lines):
				if line.strip() == 'loop_':
					nxt = (lines[idx+1].strip()
						if idx+1 < len(lines) else '')
					if nxt.startswith('_atom_site.'):
						i = idx + 1
						while (i < len(lines)
						and lines[i].strip()
						.startswith('_atom_site.')):
							cn = lines[i].strip()\
								.split('.')[1]
							col[cn] = col_idx
							col_idx += 1
							i += 1
						data_start = i
						break
			i_grp = col['group_PDB']
			i_atom = col['auth_atom_id']
			i_resn = col['auth_comp_id']
			i_chain = col['auth_asym_id']
			i_seqid = col['auth_seq_id']
			i_x = col['Cartn_x']
			i_y = col['Cartn_y']
			i_z = col['Cartn_z']
			i_occ = col['occupancy']
			i_bfac = col['B_iso_or_equiv']
			i_type = col['type_symbol']
			i_model = col.get(
				'pdbx_PDB_model_num')
			i_alt = col.get('label_alt_id')
			found_models = []
			for line in lines[data_start:]:
				ln = line.strip()
				if not ln or ln == '#':
					break
				flds = ln.split()
				if flds[i_grp] != 'ATOM':
					continue
				if (chains_to_load is not None
				and flds[i_chain] not in chains_to_load):
					continue
				if i_model is not None:
					mnum = int(flds[i_model])
					if mnum not in found_models:
						found_models.append(mnum)
					if mnum != model:
						continue
				alt = (flds[i_alt]
					if i_alt is not None else '.')
				if alt not in ('.', 'A', '1', ''):
					continue
				elem = flds[i_type]
				aname = flds[i_atom].strip('"')
				rows.append((
					aname, flds[i_resn],
					flds[i_chain],
					int(flds[i_seqid]),
					float(flds[i_x]),
					float(flds[i_y]),
					float(flds[i_z]),
					float(flds[i_occ]),
					float(flds[i_bfac]),
					elem))
			if (i_model is not None
			and found_models
			and model not in found_models):
				raise Exception(
					f'Model {model} not found in '
					f'{filename}. Available models: '
					f'{found_models}')
		if not rows:
			raise Exception(
				f'No ATOM records for chains '
				f'{chains_to_load} in {filename}')
		best = {}
		for idx, row in enumerate(rows):
			key = (row[2], row[3], row[0])
			if (key not in best
			or row[7] > rows[best[key]][7]):
				best[key] = idx
		rows = [rows[i] for i in sorted(best.values())]
		residues = defaultdict(list)
		for row in rows:
			residues[(row[2], row[3])].append(row)
		all_resnames = {
			ar[0][1] for ar in residues.values()}
		aa_tricodes = {
			v['Tricode'] for v in
			self.aminoacids.values()
			if 'Tricode' in v}
		if all_resnames & aa_tricodes:
			mol_type = 'Protein'
		elif any(rn in ('DT', 'T')
		for rn in all_resnames):
			mol_type = 'DNA'
		elif any(rn == 'U'
		for rn in all_resnames):
			mol_type = 'RNA'
		else:
			nuc_types = set()
			for rn in all_resnames:
				if rn in self.nucleotides:
					nuc_types.add(
						self.nucleotides[rn]['Type'])
			if 'RNA' in nuc_types:
				mol_type = 'RNA'
			else:
				mol_type = 'DNA'
		self.data = {
			'Type': mol_type,
			'Energy': 0, 'Rg': 0,
			'Mass': 0, 'Size': 0,
			'FASTA': {}, 'SS': {},
			'Nucleotides': (
				{} if mol_type != 'Protein'
				else None),
			'Amino Acids': (
				{} if mol_type == 'Protein'
				else None),
			'Atoms': {}, 'Bonds': {},
			'Coordinates': np.zeros((0, 3))}
		if chains_to_load is not None:
			chain_order = chains_to_load
		else:
			chain_order = sorted(set(
				k[0] for k in residues.keys()))
		sorted_keys = sorted(
			residues.keys(),
			key=lambda k: (
				chain_order.index(k[0])
				if k[0] in chain_order
				else 99, k[1]))
		if mol_type == 'Protein':
			count = 0
			Atoms = {}
			Aminos = {}
			Coordinates = []
			aa_idx = 0
			for (ch, resnum) in sorted_keys:
				atom_rows = residues[(ch, resnum)]
				seen = set()
				unique = []
				for r in atom_rows:
					if r[0] not in seen:
						seen.add(r[0])
						unique.append(r)
				BB, SC = [], []
				tricode = unique[0][1]
				for r in unique:
					Coordinates.append(
						[r[4], r[5], r[6]])
					Atoms[count] = [
						r[0], r[9], 0.0,
						r[7], r[8]]
					if r[0] in self.probbatoms:
						BB.append(count)
					else:
						SC.append(count)
					count += 1
				sym = next(
					(k for k, v in
					self.aminoacids.items()
					if v['Tricode'] == tricode),
					None)
				if sym is None:
					continue
				Aminos[aa_idx] = [
					sym, ch, BB, SC,
					'L', tricode, 0]
				aa_idx += 1
			Coordinates = np.array(Coordinates)
			errors = []
			required_bb = {'N', 'CA', 'C'}
			for k in sorted(Aminos.keys()):
				present = {
					Atoms[ai][0]
					for ai in Aminos[k][2]}
				missing = required_bb - present
				if missing:
					tri = Aminos[k][5]
					errors.append(
						f'  residue {k} ({tri}): '
						f'missing atom(s) '
						f'{sorted(missing)}')
			keys = sorted(Aminos.keys())
			for ki, kj in zip(keys[:-1], keys[1:]):
				if Aminos[ki][1] != Aminos[kj][1]:
					continue
				C_idx = next(
					(ai for ai in Aminos[ki][2]
					if Atoms[ai][0] == 'C'), None)
				N_idx = next(
					(ai for ai in Aminos[kj][2]
					if Atoms[ai][0] == 'N'), None)
				if C_idx is None or N_idx is None:
					continue
				dist = np.linalg.norm(
					Coordinates[C_idx]
					- Coordinates[N_idx])
				if dist > 2.0:
					tri_i = Aminos[ki][5]
					tri_j = Aminos[kj][5]
					errors.append(
						f'  residue {ki} '
						f'({tri_i}) \u2192 '
						f'{kj} ({tri_j}): '
						f'C\u2013N = {dist:.2f} \u00c5')
			if errors:
				raise Exception(
					f'Broken chain in {filename} '
					f'chains '
					f'{chains_to_load}:\n'
					+ '\n'.join(errors))
			self.data['Coordinates'] = Coordinates
			self.data['Amino Acids'] = Aminos
			self.data['Atoms'] = Atoms
			prev_chain = None
			for i in range(len(Aminos)):
				aa = Aminos[i][0]
				ch = Aminos[i][1]
				new_ch = (
					prev_chain is not None
					and ch != prev_chain)
				chain_residues = [
					k for k in range(len(Aminos))
					if Aminos[k][1] == ch]
				pos = chain_residues.index(i)
				chain_len = len(chain_residues)
				if pos == 0:
					bb = 'Backbone start'
				elif pos == chain_len - 1:
					bb = 'Backbone end'
				else:
					bb = 'Backbone middle'
				self._bondtree(bb, aa,
					new_chain=new_ch)
				prev_chain = ch
			self.GetCharge()
			self.GetDSSP()
			self.GetSASA()
		else:
			coords_list = []
			atom_idx = 0
			nt_idx = 0
			nt_range_by_chain = defaultdict(list)
			for (ch, resnum) in sorted_keys:
				atom_rows = residues[(ch, resnum)]
				resn_raw = atom_rows[0][1]
				if (mol_type == 'DNA'
				and resn_raw in ('A', 'G', 'C')):
					tricode = 'D' + resn_raw
				elif resn_raw == 'T':
					tricode = 'DT'
				else:
					tricode = resn_raw
				if tricode not in self.nucleotides:
					continue
				db = self.nucleotides[tricode]
				bb_meta = db['Backbone Atoms']
				bas_meta = db['Base Atoms']
				bonds_db = db['Bonds']
				n_bb = len(bb_meta)
				name_row = {r[0]: r for r in atom_rows}
				bb_indices = []
				bas_indices = []
				local_to_global = {}
				for li, am in enumerate(bb_meta):
					row = name_row.get(am[0])
					if row is None:
						local_to_global[li] = -1
						continue
					self.data['Atoms'][atom_idx] = [
						am[0], am[1], am[2],
						row[7], row[8]]
					coords_list.append(
						[row[4], row[5], row[6]])
					local_to_global[li] = atom_idx
					bb_indices.append(atom_idx)
					atom_idx += 1
				for li2, am in enumerate(bas_meta):
					li = n_bb + li2
					row = name_row.get(am[0])
					if row is None:
						local_to_global[li] = -1
						continue
					self.data['Atoms'][atom_idx] = [
						am[0], am[1], am[2],
						row[7], row[8]]
					coords_list.append(
						[row[4], row[5], row[6]])
					local_to_global[li] = atom_idx
					bas_indices.append(atom_idx)
					atom_idx += 1
				db_names = (
					{am[0] for am in bb_meta}
					| {am[0] for am in bas_meta})
				for aname, row in name_row.items():
					if aname in db_names:
						continue
					el = (row[9] if row[9]
						else aname[0])
					self.data['Atoms'][atom_idx] = [
						aname, el, 0.0,
						row[7], row[8]]
					coords_list.append(
						[row[4], row[5], row[6]])
					if aname in self.nucbbatoms:
						bb_indices.append(atom_idx)
					else:
						bas_indices.append(atom_idx)
					atom_idx += 1
				bonds = self.data['Bonds']
				for k_str, v_list in bonds_db.items():
					li = int(k_str)
					gi = local_to_global.get(
						li, -1)
					if gi == -1:
						continue
					bonds.setdefault(gi, [])
					for lj in v_list:
						gj = local_to_global.get(
							lj, -1)
						if (gj != -1
						and gj not in bonds[gi]):
							bonds[gi].append(gj)
				sym = tricode[-1]
				self.data['Nucleotides'][nt_idx] = [
					sym, ch, bb_indices,
					bas_indices, tricode]
				nt_range_by_chain[ch].append(nt_idx)
				nt_idx += 1
			self.data['Coordinates'] = (
				np.array(coords_list)
				if coords_list
				else np.zeros((0, 3)))
			atoms = self.data['Atoms']
			bonds = self.data['Bonds']
			nts = self.data['Nucleotides']
			for chain_nt_ids in \
			nt_range_by_chain.values():
				for i in range(
				len(chain_nt_ids) - 1):
					nt_i = nts[chain_nt_ids[i]]
					nt_j = nts[chain_nt_ids[i+1]]
					o3 = next(
						(a for a in nt_i[2]
						if atoms[a][0] == "O3'"),
						None)
					p = next(
						(a for a in nt_j[2]
						if atoms[a][0] == 'P'),
						None)
					if o3 is None or p is None:
						continue
					bonds.setdefault(o3, [])
					bonds.setdefault(p, [])
					if p not in bonds[o3]:
						bonds[o3].append(p)
					if o3 not in bonds[p]:
						bonds[p].append(o3)
			self.GetCharge()
		self._update()
