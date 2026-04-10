#!/usr/bin/env python3

import os
import re
import sys
import math
import numpy as np

np.seterr(all='ignore')



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
		self.elements = {
			'He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si',
			'Cl', 'Ar', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe',
			'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
			'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc',
			'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
			'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
			'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
			'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Os', 'Ir',
			'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
			'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu'}
		self.data = {
			'Type':'Molecule', 'Energy':0,
			'Rg':0, 'Mass':0,
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
				raise IndexError(
					f'Atom index {i} out of'
					f' range [0, {n})')
	def _rotmat(self, theta, u):
		mg = np.linalg.norm(u)
		if mg > 1e-10: u = u / mg
		ux, uy, uz = u[0], u[1], u[2]
		S = math.sin(math.radians(theta))
		C = math.cos(math.radians(theta))
		return np.array([
			[C+ux**2*(1-C),
				ux*uy*(1-C)-uz*S,
				ux*uz*(1-C)+uy*S],
			[uy*ux*(1-C)+uz*S,
				C+uy**2*(1-C),
				uy*uz*(1-C)-ux*S],
			[uz*ux*(1-C)-uy*S,
				uz*uy*(1-C)+ux*S,
				C+uz**2*(1-C)]])
	def _downstream(self, idx1, idx2):
		if idx1 == idx2:
			raise ValueError(
				'_downstream: idx1 == idx2')
		visited = {idx2}
		queue = [idx2]
		qi = 0
		while qi < len(queue):
			cur = queue[qi]; qi += 1
			for nb in self.data['Bonds'].get(cur, []):
				if nb == idx1 or nb in visited: continue
				visited.add(nb)
				queue.append(nb)
		return visited
	def _place(self, C, B, A, length, angle, dih):
		bc = C - B
		nm = np.linalg.norm(bc)
		if nm > 1e-10: bc = bc / nm
		ab = B - A
		nm = np.linalg.norm(ab)
		if nm > 1e-10: ab = ab / nm
		nv = np.cross(ab, bc)
		if np.linalg.norm(nv) < 1e-10:
			p = (np.array([1., 0., 0.])
				if abs(bc[2]) < 0.9
				else np.array([0., 0., 1.]))
			nv = np.cross(bc, p)
		nv = nv / (np.linalg.norm(nv) + 1e-12)
		inp = np.cross(nv, bc)
		inp = inp / (np.linalg.norm(inp) + 1e-12)
		return C + length * (
			-np.cos(angle) * bc
			+ np.sin(angle) * (
				np.cos(dih) * inp
				+ np.sin(dih) * nv))
	def _cif_tokenize(self, s):
		tokens = []; i = 0; n = len(s)
		while i < n:
			if s[i] in (' ', '\t'):
				i += 1; continue
			if s[i] in ("'", '"'):
				q = s[i]; j = i + 1
				while j < n and s[j] != q: j += 1
				tokens.append(s[i+1:j])
				i = j + 1
			else:
				j = i
				while j < n and s[j] not in (
					' ', '\t'): j += 1
				tokens.append(s[i:j]); i = j
		return tokens
	def _bond_len(self, e1, e2, order):
		BL = {
			('C','C',1):1.54, ('C','C',2):1.34,
			('C','C',3):1.20, ('C','C',1.5):1.40,
			('C','N',1):1.47, ('C','N',2):1.27,
			('C','N',3):1.15, ('C','N',1.5):1.34,
			('C','O',1):1.43, ('C','O',2):1.22,
			('C','O',1.5):1.36,
			('C','S',1):1.82, ('C','S',2):1.60,
			('C','H',1):1.09, ('N','H',1):1.01,
			('N','N',1):1.45, ('N','N',2):1.25,
			('N','O',1):1.40, ('N','O',2):1.21,
			('O','H',1):0.96, ('O','O',1):1.48,
			('S','H',1):1.34, ('S','S',1):2.05,
			('C','F',1):1.35, ('C','Cl',1):1.77,
			('C','Br',1):1.94, ('C','I',1):2.14,
			('C','P',1):1.84, ('P','H',1):1.44,
			('P','O',1):1.63, ('P','O',2):1.48,
			('B','H',1):1.19, ('B','O',1):1.36,
			('B','C',1):1.56, ('B','N',1):1.42}
		return BL.get(
			(e1, e2, order),
			BL.get((e2, e1, order), 1.50))
	def _formula(self):
		counts = {}
		for v in self.data['Atoms'].values():
			e = v[1]
			counts[e] = counts.get(e, 0) + 1
		parts = []
		for e in ('C', 'H'):
			if e in counts:
				n = counts.pop(e)
				parts.append(
					e + (str(n) if n > 1 else ''))
		for e in sorted(counts):
			n = counts[e]
			parts.append(
				e + (str(n) if n > 1 else ''))
		self.data['Formula'] = ''.join(parts)
	def Import(self, filename):
		ext = os.path.splitext(filename)[1].lower()
		with open(filename) as f: lines = f.readlines()
		atoms = {}; coords = []; bonds = {}
		bords = {}; idx = 0
		if ext == '.pdb':
			for line in lines:
				rec = line[:6].strip()
				if rec == 'ENDMDL': break
				if rec in ('ATOM', 'HETATM'):
					name = line[12:16].strip()
					x = float(line[30:38])
					y = float(line[38:46])
					z = float(line[46:54])
					el = (line[76:78].strip()
						if len(line) > 76 else '')
					if not el:
						raw = line[12:16]
						if (len(raw) >= 2
							and raw[0].isalpha()):
							cand = (
								raw[0].upper()
								+ raw[1].lower())
							if cand in self.elements:
								el = cand
							else:
								el = raw[0].upper()
						else:
							el = re.sub(
								r'[\d\s]', '',
								raw).strip()
							if el:
								el = el[0].upper()
							else: el = 'X'
					atoms[idx] = [name, el, 0.0]
					coords.append([x, y, z])
					bonds[idx] = []
					idx += 1
				elif rec == 'CONECT':
					vals = [
						int(line[k:k + 5])
						for k in range(
							6, len(line.rstrip()), 5)
						if line[k:k + 5].strip()]
					if vals:
						a1 = vals[0] - 1
						for v in vals[1:]:
							a2 = v - 1
							if (a1 < 0 or a2 < 0
								or a1 >= idx
								or a2 >= idx):
								continue
							if a2 not in bonds.get(
								a1, []):
								bonds.setdefault(
									a1, []).append(a2)
							if a1 not in bonds.get(
								a2, []):
								bonds.setdefault(
									a2, []).append(a1)
							k = (min(a1, a2),
								max(a1, a2))
							bords[k] = bords.get(
								k, 0) + 1
				normed = {}
				for (a, b), bo in bords.items():
					if bo > 3: bo = 3
					normed[(a, b)] = bo
					normed[(b, a)] = bo
				bords = normed
		elif ext == '.cif':
			loops = []
			cur_cols = []; cur_rows = []
			in_loop = False; in_semi = False
			semi_buf = []
			for line in lines:
				s = line.strip()
				if in_semi:
					if s == ';':
						in_semi = False
						if (in_loop and cur_rows
							and cur_cols):
							cur_rows[-1].append(
								'\n'.join(semi_buf))
						semi_buf = []
					else:
						semi_buf.append(
							line.rstrip())
					continue
				if s.startswith(';'):
					in_semi = True
					semi_buf = [s[1:]]
					continue
				if s.startswith('loop_'):
					if in_loop and cur_cols:
						loops.append(
							(cur_cols, cur_rows))
					cur_cols = []; cur_rows = []
					in_loop = True; continue
				if in_loop and s.startswith('_'):
					cur_cols.append(s); continue
				if in_loop and cur_cols:
					if (not s or s.startswith('#')
						or s.startswith('loop_')):
						loops.append(
							(cur_cols, cur_rows))
						cur_cols = []; cur_rows = []
						in_loop = s.startswith(
							'loop_')
						continue
					cur_rows.append(self._cif_tokenize(s))
			if in_loop and cur_cols:
				loops.append((cur_cols, cur_rows))
			APFX = ('_atom_site.', '_atom_site_', '_chem_comp_atom.')
			acols = arows = None
			for cols, rows in loops:
				if any(c.startswith(p)
					for c in cols for p in APFX):
					acols = cols; arows = rows; break
			if acols and arows:
				cm = {}
				for i, c in enumerate(acols):
					for p in APFX: c = c.replace(p, '')
					cm[c.lower()] = i
				xi = cm.get('cartn_x',
					cm.get('model_cartn_x',
					cm.get('x')))
				yi = cm.get('cartn_y',
					cm.get('model_cartn_y',
					cm.get('y')))
				zi = cm.get('cartn_z',
					cm.get('model_cartn_z',
					cm.get('z')))
				ei = cm.get('type_symbol',
					cm.get('symbol'))
				ni = cm.get('atom_id',
					cm.get('label_atom_id',
					cm.get('label', cm.get('id'))))
				if xi is None or yi is None \
					or zi is None:
					raise ValueError(
						'CIF: coordinate columns'
						' not found')
				name_map = {}
				for ri, r in enumerate(arows):
					ncol = max(
						xi, yi, zi,
						ei if ei is not None else 0,
						ni if ni is not None else 0
					) + 1
					if len(r) < ncol:
						raise ValueError(
							f'CIF atom row {ri}:'
							f' expected {ncol}'
							f' fields, got'
							f' {len(r)}')
					el = r[ei] if ei is not None \
						else 'C'
					nm = r[ni] if ni is not None \
						else f'{el}{idx}'
					x = float(r[xi])
					y = float(r[yi])
					z = float(r[zi])
					atoms[idx] = [nm, el, 0.0]
					coords.append([x, y, z])
					bonds[idx] = []
					name_map[nm] = idx; idx += 1
			BPFX = ('_chem_comp_bond.',)
			for cols, rows in loops:
				if not any(c.startswith(p)
					for c in cols for p in BPFX):
					continue
				bm = {}
				for i, c in enumerate(cols):
					for p in BPFX:
						c = c.replace(p, '')
					bm[c.lower()] = i
				ai1 = bm.get('atom_id_1')
				ai2 = bm.get('atom_id_2')
				voi = bm.get('value_order')
				if ai1 is None or ai2 is None:
					continue
				vom = {'SING':1, 'DOUB':2,
					'TRIP':3, 'AROM':1.5}
				for r in rows:
					n1, n2 = r[ai1], r[ai2]
					if (n1 not in name_map
						or n2 not in name_map):
						continue
					a1, a2 = name_map[n1], name_map[n2]
					bo = (vom.get(r[voi], 1)
						if voi is not None else 1)
					if a2 not in bonds[a1]:
						bonds[a1].append(a2)
					if a1 not in bonds[a2]:
						bonds[a2].append(a1)
					bords[(a1, a2)] = bo
					bords[(a2, a1)] = bo
				break
		elif ext in ('.sdf', '.mol'):
			if len(lines) < 4:
				raise ValueError(
					'SDF/MOL: file has fewer'
					' than 4 lines')
			hdr = lines[3]
			if 'V3000' in hdr:
				section = None
				for li, line in enumerate(lines):
					s = line.strip()
					if 'BEGIN ATOM' in s:
						section = 'ATOM'; continue
					elif 'END ATOM' in s:
						section = None; continue
					elif 'BEGIN BOND' in s:
						section = 'BOND'; continue
					elif 'END BOND' in s:
						section = None; continue
					if not s.startswith('M  V30'):
						continue
					p = s[7:].split()
					if section == 'ATOM':
						if len(p) < 5:
							raise ValueError(
								f'V3000 atom line'
								f' {li}: expected'
								f' >=5 fields,'
								f' got {len(p)}')
						el = p[1]
						x = float(p[2])
						y = float(p[3])
						z = float(p[4])
						atoms[idx] = [
							f'{el}{idx}',
							el, 0.0]
						coords.append([x, y, z])
						bonds[idx] = []
						idx += 1
					elif section == 'BOND':
						if len(p) < 4:
							raise ValueError(
								f'V3000 bond line'
								f' {li}: expected'
								f' >=4 fields,'
								f' got {len(p)}')
						bt = int(p[1])
						a1 = int(p[2]) - 1
						a2 = int(p[3]) - 1
						if a2 not in bonds.get(
							a1, []):
							bonds.setdefault(
								a1,
								[]).append(a2)
						if a1 not in bonds.get(
							a2, []):
							bonds.setdefault(
								a2,
								[]).append(a1)
						bt_map = {
							1:1, 2:2, 3:3, 4:1.5}
						bo = bt_map.get(bt, 1)
						bords[(a1, a2)] = bo
						bords[(a2, a1)] = bo
			else:
				na = int(hdr[0:3])
				nb = int(hdr[3:6])
				if len(lines) < 4 + na + nb:
					raise ValueError(
						f'SDF/MOL: file has'
						f' {len(lines)} lines but'
						f' header declares'
						f' {na} atoms + {nb}'
						f' bonds')
				for i in range(na):
					ln = lines[4 + i]
					if len(ln) < 34:
						raise ValueError(
							f'SDF atom line'
							f' {4+i}: too short'
							f' ({len(ln)} chars)')
					x = float(ln[0:10])
					y = float(ln[10:20])
					z = float(ln[20:30])
					el = ln[31:34].strip()
					atoms[idx] = [
						f'{el}{idx}', el, 0.0]
					coords.append([x, y, z])
					bonds[idx] = []; idx += 1
				bt_map = {1:1, 2:2, 3:3, 4:1.5}
				for i in range(nb):
					ln = lines[4 + na + i]
					if len(ln) < 9:
						raise ValueError(
							f'SDF bond line'
							f' {4+na+i}:'
							f' too short'
							f' ({len(ln)}'
							f' chars)')
					a1 = int(ln[0:3]) - 1
					a2 = int(ln[3:6]) - 1
					bt = int(ln[6:9])
					if a2 not in bonds[a1]:
						bonds[a1].append(a2)
					if a1 not in bonds[a2]:
						bonds[a2].append(a1)
					bo = bt_map.get(bt, 1)
					bords[(a1, a2)] = bo
					bords[(a2, a1)] = bo
		elif ext == '.mol2':
			section = None
			for line in lines:
				s = line.strip()
				if s.startswith('@<TRIPOS>'):
					section = s; continue
				if not s: continue
				if section == '@<TRIPOS>ATOM':
					p = s.split()
					if len(p) < 6:
						raise ValueError(
							f'MOL2 ATOM: expected'
							f' >=6 fields, got'
							f' {len(p)}')
					nm = p[1]
					x, y, z = (
						float(p[2]),
						float(p[3]),
						float(p[4]))
					el = p[5].split('.')[0]
					if len(el) > 1:
						el = el[0].upper() \
							+ el[1:].lower()
					ch = (float(p[8])
						if len(p) > 8 else 0.0)
					atoms[idx] = [nm, el, ch]
					coords.append([x, y, z])
					bonds[idx] = []; idx += 1
				elif section == '@<TRIPOS>BOND':
					p = s.split()
					if len(p) < 4:
						raise ValueError(
							f'MOL2 BOND: expected'
							f' >=4 fields, got'
							f' {len(p)}')
					a1 = int(p[1]) - 1
					a2 = int(p[2]) - 1
					bt = p[3]
					if a2 not in bonds.get(a1, []):
						bonds.setdefault(
							a1, []).append(a2)
					if a1 not in bonds.get(a2, []):
						bonds.setdefault(
							a2, []).append(a1)
					bm = {'1':1, '2':2, '3':3,
						'ar':1.5, 'am':1}
					bo = bm.get(bt, 1)
					bords[(a1, a2)] = bo
					bords[(a2, a1)] = bo
		else:
			raise Exception(
				f'Unsupported format: {ext}')
		has_bonds = any(bonds[i] for i in bonds)
		if not has_bonds and ext in ('.pdb', '.cif'):
			c = np.array(coords); n = len(c)
			if n > 0:
				els = [atoms[i][1]
					for i in range(n)]
				MAX_D = 2.2
				try:
					from scipy.spatial import (
						cKDTree)
					tree = cKDTree(c)
					pairs = tree.query_pairs(MAX_D)
				except ImportError:
					pairs = set()
					for i in range(n):
						for j in range(i+1, n):
							d = np.linalg.norm(
								c[i] - c[j])
							if d < MAX_D:
								pairs.add((i, j))
				for i, j in pairs:
					ei, ej = els[i], els[j]
					th = 1.9
					if ei == 'H' or ej == 'H':
						th = 1.3
					elif (ei in ('S', 'Se')
						or ej in ('S', 'Se')):
						th = 2.1
					d = np.linalg.norm(
						c[i] - c[j])
					if not (0 < d < th): continue
					bonds[i].append(j)
					bonds[j].append(i)
					bo = 1
					pr = tuple(sorted([ei, ej]))
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
					bords[(i, j)] = bo
					bords[(j, i)] = bo
		self.data['Atoms'] = atoms
		self.data['Bonds'] = bonds
		self.data['Coordinates'] = (
			np.array(coords) if coords
			else np.zeros((0, 3)))
		self._bond_orders = bords
		self._formal_charges = {}
		self.CalcCharge()
		self.CalcMass()
		self.CalcRg()
		self._formula()
		self.CalcSMILES()
		self.CalcSMARTS()
	def Export(self, filename):
		ext = os.path.splitext(filename)[1].lower()
		A = self.data['Atoms']
		C = self.data['Coordinates']
		B = self.data['Bonds']
		with open(filename, 'w') as f:
			if ext == '.pdb':
				if len(A) > 99999:
					raise ValueError(
						f'PDB format: {len(A)}'
						f' atoms exceeds 99999'
						f' serial limit')
				for i in sorted(A):
					a = A[i]; c = C[i]
					nm = a[0]
					if len(nm) < 4: nm = ' ' + nm
					f.write(
						f'HETATM{i+1:>5} '
						f'{nm:<4} LIG A'
						f'   1    '
						f'{c[0]:>8.4f}'
						f'{c[1]:>8.4f}'
						f'{c[2]:>8.4f}'
						f'  1.00  0.00'
						f'          '
						f'{a[1]:>2}\n')
				for i in sorted(B):
					row = [i + 1] + [j+1 for j in B[i]]
					for k in range(0, len(row), 5):
						chunk = row[k:k + 5]
						line = f'CONECT{chunk[0]:>5}'
						for v in chunk[1:]:
							line += f'{v:>5}'
						f.write(line + '\n')
				f.write('END\n')
			elif ext == '.cif':
				f.write('data_molecule\nloop_\n')
				for h in (
					'_atom_site.id',
					'_atom_site.type_symbol',
					'_atom_site.label_atom_id',
					'_atom_site.Cartn_x',
					'_atom_site.Cartn_y',
					'_atom_site.Cartn_z'):
					f.write(h + '\n')
				for i in sorted(A):
					a = A[i]; c = C[i]
					f.write(
						f'{i+1} {a[1]} {a[0]} '
						f'{c[0]:.4f} '
						f'{c[1]:.4f} '
						f'{c[2]:.4f}\n')
				bp = set()
				for i in sorted(B):
					for j in B[i]:
						if i < j: bp.add((i, j))
				if bp:
					bm = {1:'SING', 2:'DOUB',
						3:'TRIP', 1.5:'AROM'}
					f.write('loop_\n')
					for h in (
						'_chem_comp_bond.atom_id_1',
						'_chem_comp_bond.atom_id_2',
						'_chem_comp_bond.'
						'value_order'):
						f.write(h + '\n')
					for i, j in sorted(bp):
						bo = self._bond_orders.get(
							(i, j), 1)
						f.write(
							f'{A[i][0]} {A[j][0]}'
							f' {bm.get(bo,"SING")}\n')
			elif ext in ('.sdf', '.mol'):
				bp = set()
				for i in sorted(B):
					for j in B[i]:
						if i < j: bp.add((i, j))
				na = len(A); nb = len(bp)
				if na > 999 or nb > 999:
					raise ValueError(
						f'SDF V2000 limit:'
						f' {na} atoms,'
						f' {nb} bonds'
						f' (max 999 each)')
				nm = self.data['SMILES'] or ''
				f.write(f'{nm}\n')
				f.write('     Molecule\n\n')
				f.write(
					f'{na:>3}{nb:>3}'
					'  0  0  0  0  0'
					'  0  0  0999 V2000\n')
				for i in sorted(A):
					a = A[i]; c = C[i]
					f.write(
						f'{c[0]:>10.4f}'
						f'{c[1]:>10.4f}'
						f'{c[2]:>10.4f}'
						f' {a[1]:<3}'
						' 0  0  0  0  0'
						'  0  0  0  0  0'
						'  0  0\n')
				bm = {1:1, 2:2, 3:3, 1.5:4}
				for i, j in sorted(bp):
					bo = self._bond_orders.get(
						(i, j), 1)
					f.write(
						f'{i+1:>3}{j+1:>3}'
						f'{bm.get(bo, 1):>3}'
						'  0  0  0  0\n')
				f.write('M  END\n$$$$\n')
			elif ext == '.mol2':
				bp = set()
				for i in sorted(B):
					for j in B[i]:
						if i < j: bp.add((i, j))
				na = len(A); nb = len(bp)
				mbo = {i: 0 for i in A}
				for (a1, a2), bo in \
					self._bond_orders.items():
					if a1 in mbo and bo > mbo[a1]:
						mbo[a1] = bo
					if a2 in mbo and bo > mbo[a2]:
						mbo[a2] = bo
				f.write('@<TRIPOS>MOLECULE\n')
				nm = self.data['SMILES'] or 'MOL'
				f.write(f'{nm}\n')
				f.write(f'{na} {nb}\nSMALL\n\n')
				f.write('@<TRIPOS>ATOM\n')
				for i in sorted(A):
					a = A[i]; c = C[i]
					el = a[1]
					mb = mbo.get(i, 0)
					if el in ('C', 'N', 'O', 'S'):
						if mb >= 3:
							st = el + '.1'
						elif mb == 1.5:
							st = el + '.ar'
						elif mb >= 2:
							st = el + '.2'
						else:
							st = el + '.3'
					else: st = el
					f.write(
						f'{i+1:>4} {a[0]:<4}'
						f' {c[0]:>10.4f}'
						f' {c[1]:>10.4f}'
						f' {c[2]:>10.4f}'
						f' {st:<6}'
						f' 1 LIG'
						f' {a[2]:.4f}\n')
				f.write('@<TRIPOS>BOND\n')
				bm = {1:'1', 2:'2', 3:'3', 1.5:'ar'}
				for bi, (i, j) in enumerate(
					sorted(bp), 1):
					bo = self._bond_orders.get(
						(i, j), 1)
					f.write(
						f'{bi:>4} {i+1:>4}'
						f' {j+1:>4}'
						f' {bm.get(bo, "1")}\n')
			else:
				raise Exception(
					f'Unsupported format: {ext}')
	def CalcSMILES(self):
		A = self.data['Atoms']
		B = self.data['Bonds']
		heavy = sorted(
			i for i, v in A.items() if v[1] != 'H')
		if not heavy:
			self.data['SMILES'] = ''; return ''
		hs = set(heavy)
		adj = {i: [j for j in B.get(i, [])
			if j in hs] for i in heavy}
		hcount = {i: sum(1 for j in B.get(i, [])
			if A[j][1] == 'H') for i in heavy}
		visited = set()
		parent = {}
		children = {i: [] for i in heavy}
		roots = []
		for s in heavy:
			if s in visited: continue
			roots.append(s); visited.add(s)
			stk = [s]
			while stk:
				node = stk[-1]; pushed = False
				for nb in adj[node]:
					if nb not in visited:
						visited.add(nb)
						parent[nb] = node
						children[node].append(nb)
						stk.append(nb)
						pushed = True; break
				if not pushed: stk.pop()
		te = set()
		for n, p in parent.items():
			te.add(frozenset((n, p)))
		back = []
		for i in heavy:
			for j in adj[i]:
				if i < j and frozenset((i, j)) \
					not in te:
					back.append((i, j))
		ring_at = {}
		for d, (a, b) in enumerate(back, 1):
			bo = self._bond_orders.get(
				(a, b),
				self._bond_orders.get((b, a), 1))
			ring_at.setdefault(
				a, []).append((d, bo))
			ring_at.setdefault(
				b, []).append((d, bo))
		seen_d = set()
		def tok(i):
			el = A[i][1]; nh = hcount[i]
			q = self._formal_charges.get(i, 0)
			s = '[' + el
			if nh:
				s += 'H' + (str(nh)
					if nh > 1 else '')
			if q > 0:
				s += '+' + (str(q)
					if q > 1 else '')
			elif q < 0:
				s += '-' + (str(-q)
					if -q > 1 else '')
			return s + ']'
		def gen(node):
			s = tok(node)
			for d, bo in ring_at.get(node, []):
				if d not in seen_d:
					if bo == 2: s += '='
					elif bo == 3: s += '#'
					elif bo == 1.5: s += ':'
				seen_d.add(d)
				s += (str(d) if d < 10
					else f'%{d:02d}')
			ch = children[node]
			for i, c in enumerate(ch):
				bo = self._bond_orders.get(
					(node, c),
					self._bond_orders.get(
						(c, node), 1))
				bsym = ''
				if bo == 2: bsym = '='
				elif bo == 3: bsym = '#'
				elif bo == 1.5: bsym = ':'
				if i < len(ch) - 1:
					s += '(' + bsym + gen(c) + ')'
				else:
					s += bsym + gen(c)
			return s
		lim = sys.getrecursionlimit()
		need = len(heavy) + 100
		if need > lim:
			sys.setrecursionlimit(need)
		result = '.'.join(gen(r) for r in roots)
		sys.setrecursionlimit(lim)
		self.data['SMILES'] = result
		return result
	def CalcSMARTS(self):
		ANUM = {
			'H':1, 'He':2, 'Li':3, 'Be':4,
			'B':5, 'C':6, 'N':7, 'O':8,
			'F':9, 'Ne':10, 'Na':11, 'Mg':12,
			'Al':13, 'Si':14, 'P':15, 'S':16,
			'Cl':17, 'Ar':18, 'K':19, 'Ca':20,
			'Sc':21, 'Ti':22, 'V':23, 'Cr':24,
			'Mn':25, 'Fe':26, 'Co':27, 'Ni':28,
			'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32,
			'As':33, 'Se':34, 'Br':35, 'Kr':36,
			'Rb':37, 'Sr':38, 'Y':39, 'Zr':40,
			'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44,
			'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48,
			'In':49, 'Sn':50, 'Sb':51, 'Te':52,
			'I':53, 'Xe':54, 'Cs':55, 'Ba':56,
			'La':57, 'Ce':58, 'Pr':59, 'Nd':60,
			'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64,
			'Tb':65, 'Dy':66, 'Ho':67, 'Er':68,
			'Tm':69, 'Yb':70, 'Lu':71, 'Hf':72,
			'Ta':73, 'W':74, 'Re':75, 'Os':76,
			'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
			'Tl':81, 'Pb':82, 'Bi':83, 'Po':84,
			'At':85, 'Rn':86, 'Fr':87, 'Ra':88,
			'Ac':89, 'Th':90, 'Pa':91, 'U':92,
			'Np':93, 'Pu':94}
		A = self.data['Atoms']
		B = self.data['Bonds']
		heavy = sorted(
			i for i, v in A.items() if v[1] != 'H')
		if not heavy:
			self.data['SMARTS'] = ''; return ''
		hs = set(heavy)
		adj = {i: [j for j in B.get(i, [])
			if j in hs] for i in heavy}
		visited = set()
		parent = {}
		children = {i: [] for i in heavy}
		roots = []
		for s in heavy:
			if s in visited: continue
			roots.append(s); visited.add(s)
			stk = [s]
			while stk:
				node = stk[-1]; pushed = False
				for nb in adj[node]:
					if nb not in visited:
						visited.add(nb)
						parent[nb] = node
						children[node].append(nb)
						stk.append(nb)
						pushed = True; break
				if not pushed: stk.pop()
		te = set()
		for n, p in parent.items():
			te.add(frozenset((n, p)))
		back = []
		for i in heavy:
			for j in adj[i]:
				if i < j and frozenset((i, j)) \
					not in te:
					back.append((i, j))
		ring_at = {}
		for d, (a, b) in enumerate(back, 1):
			bo = self._bond_orders.get(
				(a, b),
				self._bond_orders.get((b, a), 1))
			ring_at.setdefault(
				a, []).append((d, bo))
			ring_at.setdefault(
				b, []).append((d, bo))
		seen_d = set()
		def gen(node):
			el = A[node][1]
			an = ANUM.get(el, 0)
			s = f'[#{an}]' if an else f'[{el}]'
			for d, bo in ring_at.get(node, []):
				if d not in seen_d:
					if bo == 1: s += '-'
					elif bo == 2: s += '='
					elif bo == 3: s += '#'
					elif bo == 1.5: s += ':'
				seen_d.add(d)
				s += (str(d) if d < 10
					else f'%{d:02d}')
			ch = children[node]
			for i, c in enumerate(ch):
				bo = self._bond_orders.get(
					(node, c),
					self._bond_orders.get(
						(c, node), 1))
				bsym = '-'
				if bo == 2: bsym = '='
				elif bo == 3: bsym = '#'
				elif bo == 1.5: bsym = ':'
				if i < len(ch) - 1:
					s += '(' + bsym + gen(c) + ')'
				else:
					s += bsym + gen(c)
			return s
		lim = sys.getrecursionlimit()
		need = len(heavy) + 100
		if need > lim:
			sys.setrecursionlimit(need)
		result = '.'.join(gen(r) for r in roots)
		sys.setrecursionlimit(lim)
		self.data['SMARTS'] = result
		return result
	def CalcRg(self):
		A = self.data['Atoms']
		if not A: self.data['Rg'] = 0.0; return
		mass = np.array([
			self.masses.get(A[i][1], 0.0)
			for i in sorted(A)])
		tm = mass.sum()
		if tm == 0: self.data['Rg'] = 0.0; return
		co = self.data['Coordinates']
		if len(mass) != co.shape[0]:
			raise ValueError(
				f'CalcRg: {len(mass)} atoms'
				f' but {co.shape[0]}'
				f' coordinates')
		xm = co * mass[:, np.newaxis]
		rr = np.sum(co * xm)
		mm = np.sum((xm.sum(0) / tm) ** 2)
		self.data['Rg'] = round(
			math.sqrt(max(0.0, rr / tm - mm)), 3)
	def CalcCharge(self, iterations=6):
		PARAMS = {
			'C3': (7.98, 9.18, 1.88),
			'C2': (8.79, 9.32, 1.51),
			'C1': (10.39, 9.45, 0.73),
			'H': (7.17, 6.24, -0.56),
			'O3': (14.18, 12.92, 1.39),
			'O2': (17.07, 13.79, 0.47),
			'N3': (11.54, 10.82, 1.36),
			'N2': (12.87, 11.15, 0.85),
			'S': (10.14, 9.13, 1.38),
			'Se': (9.00, 8.00, 1.10),
			'F': (14.66, 13.85, 2.31),
			'Cl': (11.00, 9.69, 1.35),
			'Br': (10.08, 8.47, 1.16),
			'I': (9.90, 7.96, 0.96),
			'P': (8.90, 8.24, 0.96),
			'B': (5.80, 6.00, 1.56)}
		A = self.data['Atoms']
		B = self.data['Bonds']
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
				atype[i] = PARAMS.get(
					el.capitalize(), PARAMS['S'])
			elif el == 'C':
				if mbo[i] >= 3:
					atype[i] = PARAMS['C1']
				elif mbo[i] >= 1.5:
					atype[i] = PARAMS['C2']
				else: atype[i] = PARAMS['C3']
			elif el == 'N':
				atype[i] = (PARAMS['N2']
					if mbo[i] >= 1.5
					else PARAMS['N3'])
			elif el == 'O':
				atype[i] = (PARAMS['O2']
					if mbo[i] >= 1.5
					else PARAMS['O3'])
			elif el in ('F', 'CL', 'BR', 'I'):
				atype[i] = PARAMS.get(
					el.capitalize(),
					PARAMS['Cl'])
			elif el == 'P':
				atype[i] = PARAMS['P']
			elif el == 'B':
				atype[i] = PARAMS['B']
			else: atype[i] = None
		charges = {i: 0.0 for i in ids}
		for n in range(iterations):
			damp = 1.0 / (2 ** (n + 1))
			chi = {}
			for i in ids:
				if atype[i] is None:
					chi[i] = 0.0; continue
				a, b, c = atype[i]
				qv = charges[i]
				chi[i] = a + qv * (b + c * qv)
			delta = {i: 0.0 for i in ids}
			for i in ids:
				for j in B.get(i, []):
					if j <= i: continue
					if (atype[i] is None
						or atype[j] is None):
						continue
					if chi[j] >= chi[i]:
						donor, acc = i, j
					else: donor, acc = j, i
					a, b, c = atype[donor]
					ip = a + b + c
					if ip == 0: continue
					dq = damp * (chi[acc]
						- chi[donor]) / ip
					delta[donor] += dq
					delta[acc] -= dq
			for i in ids: charges[i] += delta[i]
		for i in ids: A[i][2] = round(charges[i], 4)
	def CalcMass(self):
		A = self.data['Atoms']
		self.data['Mass'] = round(sum(
			self.masses.get(v[1], 0.0)
			for v in A.values()), 3)
	def GetDistance(self, idx1, idx2):
		self._check_idx(idx1, idx2)
		C = self.data['Coordinates']
		return np.linalg.norm(C[idx2] - C[idx1])
	def GetAngle(self, idx1, idx2, idx3):
		self._check_idx(idx1, idx2, idx3)
		C = self.data['Coordinates']
		a = C[idx2] - C[idx1]
		b = C[idx2] - C[idx3]
		d = np.linalg.norm(a) * np.linalg.norm(b)
		if d < 1e-10: return 0.0
		ct = max(-1.0, min(1.0, np.dot(a, b) / d))
		return math.degrees(math.acos(ct))
	def GetDihedral(self, idx1, idx2, idx3, idx4):
		self._check_idx(idx1, idx2, idx3, idx4)
		C = self.data['Coordinates']
		u1 = C[idx2] - C[idx1]
		u2 = C[idx3] - C[idx2]
		u3 = C[idx4] - C[idx3]
		mg = np.linalg.norm(u2)
		c12 = np.cross(u1, u2)
		c23 = np.cross(u2, u3)
		a = np.dot(u2, np.cross(c12, c23))
		b = mg * np.dot(c12, c23)
		return math.atan2(a, b) * 180 / math.pi
	def GetAtomCoord(self, idx):
		self._check_idx(idx)
		return self.data['Coordinates'][idx]
	def GetAtomList(self):
		return [v[1]
			for v in self.data['Atoms'].values()]
	def GetAtomBonds(self, idx):
		A = self.data['Atoms']
		return [A[j][0]
			for j in self.data['Bonds'].get(idx, [])]
	def AdjustDistance(self, idx1, idx2, length):
		self._check_idx(idx1, idx2)
		C = self.data['Coordinates']
		v = C[idx2] - C[idx1]
		mg = np.linalg.norm(v)
		if mg < 1e-10: return
		shift = v * (length / mg) - v
		for i in self._downstream(idx1, idx2):
			C[i] += shift
		self._invalidate()
	def AdjustAngle(self, idx1, idx2, idx3, theta):
		self._check_idx(idx1, idx2, idx3)
		C = self.data['Coordinates']
		a = C[idx3] - C[idx1]
		b = C[idx3] - C[idx2]
		u = np.cross(b, a)
		lu = np.linalg.norm(u)
		if lu < 1e-10: return
		u = u / lu
		ori = C[idx2].copy()
		RM = self._rotmat(theta, u)
		for i in self._downstream(idx2, idx3):
			C[i] = np.matmul(C[i] - ori, RM) + ori
		self._invalidate()
	def RotateDihedral(
		self, idx1, idx2, idx3, idx4, theta):
		self._check_idx(idx1, idx2, idx3, idx4)
		C = self.data['Coordinates']
		current = self.GetDihedral(
			idx1, idx2, idx3, idx4)
		ori = C[idx3].copy()
		u = C[idx2] - C[idx3]
		mg = np.linalg.norm(u)
		if mg < 1e-10: return
		u = u / mg
		RM_zero = self._rotmat(-current, u)
		RM_new = self._rotmat(theta, u)
		for i in self._downstream(idx2, idx3):
			v = C[i] - ori
			v = np.matmul(v, RM_zero)
			C[i] = np.matmul(v, RM_new) + ori
		self._invalidate()
	def MovePose(
		self, theta=None, u=None, l=None, ori=None):
		C = self.data['Coordinates'].copy()
		if len(C) == 0: return
		if theta is not None and u is not None:
			u = np.array(u, dtype=float)
			mg = np.linalg.norm(u)
			if mg > 1e-10:
				u = u / mg
				pivot = C.mean(axis=0)
				R = self._rotmat(theta, u)
				C = np.matmul(C - pivot, R) + pivot
		if l is not None and ori is not None:
			ori = np.array(ori, dtype=float)
			cent = C.mean(axis=0)
			d = ori - cent
			mg = np.linalg.norm(d)
			if mg > 1e-10:
				C = C + (d / mg) * l
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
		heavy = [i for i, v in A.items()
			if v[1] != 'H']
		if len(heavy) < 2 or len(heavy) >= 200:
			return
		Cr = d['Coordinates']; hs = set(heavy)
		hc = Cr[heavy]
		hc0 = hc - hc.mean(axis=0)
		_, _, Vt = np.linalg.svd(
			hc0, full_matrices=False)
		proj = hc0 @ Vt[:2].T
		hi = {v: k for k, v in enumerate(heavy)}
		bls = []
		for i in heavy:
			for j in B.get(i, []):
				if j in hs and j > i:
					bls.append(np.linalg.norm(
						proj[hi[i]] - proj[hi[j]]))
		mbl = np.median(bls) if bls else 1.0
		if mbl < 0.01: mbl = 1.0
		sc = 14.0 / mbl
		spx = {}; spy = {}
		for k, i in enumerate(heavy):
			spx[i] = proj[k, 0] * sc
			spy[i] = proj[k, 1] * sc
		xv = list(spx.values())
		yv = list(spy.values())
		xmn, ymx = min(xv), max(yv)
		for i in heavy:
			spx[i] = int(round(spx[i] - xmn)) + 4
			spy[i] = int(round(ymx - spy[i])) + 4
		PW = max(spx.values()) + 5
		PH = max(spy.values()) + 5
		canvas = [[False]*PW for _ in range(PH)]
		drawn = set()
		for i in heavy:
			for j in B.get(i, []):
				if j not in hs or j <= i: continue
				if (i, j) in drawn: continue
				drawn.add((i, j))
				x0, y0 = spx[i], spy[i]
				x1, y1 = spx[j], spy[j]
				bo = self._bond_orders.get(
					(i, j),
					self._bond_orders.get(
						(j, i), 1))
				bx = x1 - x0; by = y1 - y0
				mg = math.sqrt(bx*bx + by*by)
				if mg > 0:
					pdx = -by / mg; pdy = bx / mg
				else: pdx, pdy = 0.0, 1.0
				offs = [(0, 0)]
				if bo >= 2:
					offs.append((pdx*2, pdy*2))
				if bo >= 3:
					offs.append((-pdx*2, -pdy*2))
				for ox, oy in offs:
					ax0 = int(round(x0 + ox))
					ay0 = int(round(y0 + oy))
					ax1 = int(round(x1 + ox))
					ay1 = int(round(y1 + oy))
					ddx = abs(ax1 - ax0)
					ddy = abs(ay1 - ay0)
					ssx = (1 if ax0 < ax1
						else (-1 if ax0 > ax1
						else 0))
					ssy = (1 if ay0 < ay1
						else (-1 if ay0 > ay1
						else 0))
					if not ddx and not ddy: continue
					err = ddx - ddy
					cx, cy = ax0, ay0
					while True:
						if (0 <= cx < PW
							and 0 <= cy < PH):
							canvas[cy][cx] = True
						if cx == ax1 and cy == ay1:
							break
						e2 = 2 * err
						if e2 > -ddy:
							err -= ddy; cx += ssx
						if e2 < ddx:
							err += ddx; cy += ssy
		DOTS = {(0,0):0x01, (1,0):0x08,
			(0,1):0x02, (1,1):0x10,
			(0,2):0x04, (1,2):0x20,
			(0,3):0x40, (1,3):0x80}
		CW = (PW + 1) // 2; CH = (PH + 3) // 4
		grid = [[' ']*CW for _ in range(CH)]
		colr = [['']*CW for _ in range(CH)]
		for cy in range(CH):
			for cx in range(CW):
				code = 0
				for (dx, dy), bit in DOTS.items():
					px = cx * 2 + dx
					py = cy * 4 + dy
					if (px < PW and py < PH
						and canvas[py][px]):
						code |= bit
				if code:
					grid[cy][cx] = chr(0x2800+code)
		GRY = '\033[90m'; RST = '\033[0m'
		for cy in range(CH):
			for cx in range(CW):
				if grid[cy][cx] != ' ':
					colr[cy][cx] = GRY
		ACOL = {'C':'\033[32m', 'N':'\033[34m',
			'O':'\033[31m', 'P':'\033[38;5;208m',
			'S':'\033[33m', 'H':'\033[37m'}
		DCOL = '\033[35m'
		for i in heavy:
			cx = spx[i] // 2; cy = spy[i] // 4
			el = A[i][1]
			c = ACOL.get(el, DCOL)
			lbl = str(i)
			if 0 <= cy < CH:
				for ci, ch in enumerate(lbl):
					px = cx + ci
					if 0 <= px < CW:
						grid[cy][px] = ch
						colr[cy][px] = c
		for r in range(CH):
			out = []
			for c in range(CW):
				cc = colr[r][c]; ch = grid[r][c]
				if cc: out.append(cc + ch + RST)
				else: out.append(ch)
			line = ''.join(out).rstrip()
			if line: print(line)
