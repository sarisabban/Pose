#!/usr/bin/env python3

import os
import math
import json
import datetime
import numpy as np
from collections import defaultdict

np.seterr(all='ignore')


class PoseN():
	'''Data structure that represents a nucleic acid'''
	def __init__(self):
		path, _ = os.path.split(__file__)
		with open(f'{path}/NucleotidesDB.json') as f:
			self.NucleotidesDB = json.load(f)
		self.BB_ATOMS = {
			'P', 'OP1', 'OP2',
			"O5'", "C5'", "H5'", "H5''",
			"C4'", "H4'", "O4'",
			"C3'", "H3'", "C2'", "H2'", "H2''",
			"C1'", "H1'", "O3'",
			"O2'", "HO2'"}
		self.Masses = {
			'H':1.008,    'He':4.003,   'Li':6.941,   'Be':9.012,
			'B':10.811,   'C':12.011,   'N':14.007,   'O':15.999,
			'F':18.998,   'Ne':20.180,  'Na':22.990,  'Mg':24.305,
			'Al':26.982,  'Si':28.086,  'P':30.974,   'S':32.066,
			'Cl':35.453,  'Ar':39.948,  'K':39.098,   'Ca':40.078,
			'Sc':44.956,  'Ti':47.867,  'V':50.942,   'Cr':51.996,
			'Mn':54.938,  'Fe':55.845,  'Co':58.933,  'Ni':58.693,
			'Cu':63.546,  'Zn':65.38,   'Ga':69.723,  'Ge':72.631,
			'As':74.922,  'Se':78.971,  'Br':79.904,  'Kr':84.798,
			'Rb':84.468,  'Sr':87.62,   'Y':88.906,   'Zr':91.224,
			'Nb':92.906,  'Mo':95.95,   'Tc':98.907,  'Ru':101.07,
			'Rh':102.906, 'Pd':106.42,  'Ag':107.868, 'Cd':112.414,
			'In':114.818, 'Sn':118.711, 'Sb':121.760, 'Te':126.7,
			'I':126.904,  'Xe':131.294, 'Cs':132.905, 'Ba':137.328,
			'La':138.905, 'Ce':140.116, 'Pr':140.908, 'Nd':144.243,
			'Pm':144.913, 'Sm':150.36,  'Eu':151.964, 'Gd':157.25,
			'Tb':158.925, 'Dy':162.500, 'Ho':164.930, 'Er':167.259,
			'Tm':168.934, 'Yb':173.055, 'Lu':174.967, 'Hf':178.49,
			'Ta':180.948, 'W':183.84,   'Re':186.207, 'Os':190.23,
			'Ir':192.217, 'Pt':195.085, 'Au':196.967, 'Hg':200.592,
			'Tl':204.383, 'Pb':207.2,   'Bi':208.980, 'Po':208.982,
			'At':209.987, 'Rn':222.081, 'Fr':223.020, 'Ra':226.025,
			'Ac':227.028, 'Th':232.038, 'Pa':231.036, 'U':238.029,
			'Np':237,     'Pu':244}
		self.data = {
			'Energy':0, 'Rg':0, 'Mass':0, 'Size':0,
			'FASTA':None, 'Type':None,
			'Nucleotides':{}, 'Atoms':{}, 'Bonds':{},
			'Coordinates':np.zeros((0, 3))}

	# ── Formatting helpers ────────────────────────────────────────────────────

	def PDB_entry(self, atom, n, a, l, r, c, s, i, x, y, z, o, t, q, e):
		'''Construct a PDB atom data row'''
		ATOM = '{:<6}'.format(atom)
		N    = '{:>5} '.format(n)
		A    = '{:<4}'.format(a)
		L    = '{:<1}'.format(l)
		R    = '{:>3}'.format(r)
		C    = '{:>2}'.format(c)
		S    = '{:>4}'.format(s)
		I    = '{:>1}   '.format(i)
		X    = '{:>8.3f}'.format(x)
		Y    = '{:>8.3f}'.format(y)
		Z    = '{:>8.3f} '.format(z)
		O    = '{:>5.2f} '.format(o)
		T    = ('{:>5.2f} '.format(t))[:6]
		Q    = '{:>10}'.format('')
		E    = '{:<2} \n'.format(e)
		return ATOM+N+A+L+R+C+S+I+X+Y+Z+O+T+Q+E

	def CIF_entry(
			self, atom, n, a, l, r, c, label_seq,
			x, y, z, o, t, q, e):
		'''Construct a mmCIF _atom_site data row'''
		alt    = '.' if l == '' else l
		charge = '?' if q == 0.0 else '{:.3f}'.format(q)
		fields = [
			atom, str(n), e, a, alt, r, c, '1',
			str(label_seq), '?',
			'{:.3f}'.format(x), '{:.3f}'.format(y),
			'{:.3f}'.format(z),
			'{:.2f}'.format(o), '{:.2f}'.format(t),
			charge, str(label_seq), r, c, a, '1']
		return ' '.join(fields) + '\n'

	# ── Math utilities ────────────────────────────────────────────────────────

	def Rotation_Matrix(self, theta, u):
		'''Rodrigues rotation matrix: rotate theta degrees around axis u'''
		ux, uy, uz = u[0], u[1], u[2]
		S = math.sin(math.radians(theta))
		C = math.cos(math.radians(theta))
		return np.array([
			[C+ux**2*(1-C),    ux*uy*(1-C)-uz*S, ux*uz*(1-C)+uy*S],
			[uy*ux*(1-C)+uz*S, C+uy**2*(1-C),    uy*uz*(1-C)-ux*S],
			[uz*ux*(1-C)-uy*S, uz*uy*(1-C)+ux*S, C+uz**2*(1-C)  ]])

	def _downstream_atoms(self, pivot_a_idx, pivot_b_idx):
		'''BFS from pivot_b without crossing pivot_a; returns atom idx set'''
		visited = {pivot_a_idx}
		queue   = [pivot_b_idx]
		result  = set()
		bonds   = self.data['Bonds']
		while queue:
			curr = queue.pop(0)
			if curr in visited:
				continue
			visited.add(curr)
			result.add(curr)
			for nb in bonds.get(curr, []):
				if nb not in visited:
					queue.append(nb)
		return result

	def _get_atom_idx(self, nt, atom):
		'''Global atom index for named atom in nucleotide nt'''
		info = self.data['Nucleotides'][nt]
		for i in info[2] + info[3]:
			if self.data['Atoms'][i][0] == atom:
				return i
		raise Exception(f'Atom {atom} not found in nucleotide {nt}')

	def _prev_nt(self, nt):
		'''Index of previous nucleotide on same chain, or None'''
		nts   = self.data['Nucleotides']
		chain = nts[nt][1]
		if nt-1 in nts and nts[nt-1][1] == chain:
			return nt - 1
		return None

	def _next_nt(self, nt):
		'''Index of next nucleotide on same chain, or None'''
		nts   = self.data['Nucleotides']
		chain = nts[nt][1]
		if nt+1 in nts and nts[nt+1][1] == chain:
			return nt + 1
		return None

	# ── Atom access ───────────────────────────────────────────────────────────

	def AtomList(self, PDB=False):
		'''List of all atom names (PDB=True) or elements (PDB=False)'''
		idx = 0 if PDB else 1
		return [x[idx] for x in self.data['Atoms'].values()]

	def GetAtom(self, nt, atom):
		'''XYZ coordinates for named atom in nucleotide nt'''
		info    = self.data['Nucleotides'][nt]
		indices = info[2] if atom in self.BB_ATOMS else info[3]
		for i in indices:
			if self.data['Atoms'][i][0] == atom:
				return self.data['Coordinates'][i]
		raise Exception(f'Nucleotide {nt} does not have atom {atom}')

	def GetBondAtoms(self, index1, index2):
		'''Names and elements of two bonded atoms by global index'''
		if index2 not in self.data['Bonds'].get(index1, []):
			raise Exception('Requested atoms are not bonded')
		return [
			self.data['Atoms'][index1][0],
			self.data['Atoms'][index1][1],
			self.data['Atoms'][index2][0],
			self.data['Atoms'][index2][1]]

	def Identify(self, atom_index):
		'''Classify atom as Phosphate, Sugar, Base, or Hydrogen'''
		name = self.data['Atoms'][atom_index][0]
		elem = self.data['Atoms'][atom_index][1].upper()
		if elem == 'H':
			return 'Hydrogen'
		if name in {'P', 'OP1', 'OP2'}:
			return 'Phosphate'
		if name in self.BB_ATOMS:
			return 'Sugar'
		return 'Base'

	# ── Scalar properties ─────────────────────────────────────────────────────

	def Distance(self, nt1, atom1, nt2, atom2):
		'''Distance in Angstrom between two named atoms'''
		A = self.GetAtom(nt1, atom1)
		B = self.GetAtom(nt2, atom2)
		return float(np.linalg.norm(B - A))

	def Atom3Angle(self, nt1, atom1, nt2, atom2, nt3, atom3):
		'''Bond angle (degrees) at atom2'''
		a1    = self.GetAtom(nt1, atom1)
		a2    = self.GetAtom(nt2, atom2)
		a3    = self.GetAtom(nt3, atom3)
		A     = a2 - a1
		B     = a2 - a3
		cos_t = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
		return math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))

	def Mass(self):
		'''Total mass in Da'''
		return round(sum(self.Masses[e] for e in self.AtomList()), 3)

	def Rg(self):
		'''Radius of gyration in Angstrom'''
		mass  = np.array([self.Masses[e] for e in self.AtomList()])
		tmass = mass.sum()
		if tmass == 0:
			raise ZeroDivisionError('No atoms in pose')
		coord = self.data['Coordinates']
		xm    = coord * mass[:, np.newaxis]
		rr    = np.sum(coord * xm)
		mm    = np.sum((xm.sum(0) / tmass) ** 2)
		return round(math.sqrt(rr / tmass - mm), 3)

	def FASTA(self):
		'''5\'->3\' one-letter sequence of strand A'''
		nts = self.data['Nucleotides']
		return ''.join(v[0] for v in nts.values() if v[1] == 'A')

	def Size(self):
		'''Number of base pairs (length of strand A)'''
		nts = self.data['Nucleotides']
		return sum(1 for v in nts.values() if v[1] == 'A')

	def update_data(self):
		'''Refresh cached scalar properties'''
		self.data['Mass']  = self.Mass()
		self.data['FASTA'] = self.FASTA()
		self.data['Size']  = self.Size()
		self.data['Rg']    = self.Rg()

	def Info(self):
		'''Print a summary of the nucleic acid pose'''
		print('Type:\t\t{}'.format(self.data['Type']))
		print('Sequence:\t{}'.format(self.data['FASTA']))
		print('Mass:\t\t{} Da'.format(self.data['Mass']))
		print('Size:\t\t{} bp'.format(self.data['Size']))
		print('Rg:\t\t{} Å'.format(self.data['Rg']))
		print('Energy:\t\t{}'.format(self.data['Energy']))

	# ── I/O ───────────────────────────────────────────────────────────────────

	def atom_residue_iter(self):
		'''Yield (atom_item, coordinate, nt_index) for every atom'''
		nts    = self.data['Nucleotides']
		atoms  = self.data['Atoms']
		coords = self.data['Coordinates']
		for nt_idx, nt_info in nts.items():
			for ai in nt_info[2] + nt_info[3]:
				yield (ai, atoms[ai]), coords[ai], nt_idx

	def Export(self, filename):
		'''Export pose to a .pdb or .cif file'''
		ext = filename[-3:].upper()
		nts = self.data['Nucleotides']
		if ext == 'PDB':
			with open(filename, 'w') as f:
				DATE = datetime.date.today().strftime('%d-%b-%Y')
				f.write('HEADER' + ' '*44 + DATE
				        + ' '*3 + 'XXXX' + ' '*11 + '\n')
				f.write('EXPDTA    THEORETICAL MODEL'
				        + ' '*52 + '\n')
				f.write('REMARK 220 REMARK: MODEL GENERATED BY '
				        'SARI SABBAN' + ' '*30 + '\n')
				for atom, coord, nt_idx in self.atom_residue_iter():
					nt = nts[nt_idx]
					f.write(self.PDB_entry(
						'ATOM', atom[0]+1, atom[1][0], '',
						nt[4], nt[1], nt_idx+1, '',
						coord[0], coord[1], coord[2],
						atom[1][3], atom[1][4],
						atom[1][2], atom[1][1]))
				f.write('TER')
		elif ext == 'CIF':
			with open(filename, 'w') as f:
				f.write('data_POSE\n#\n')
				f.write('loop_\n')
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
				for col in cols:
					f.write(col + '\n')
				for atom, coord, nt_idx in self.atom_residue_iter():
					nt = nts[nt_idx]
					f.write(self.CIF_entry(
						'ATOM', atom[0]+1, atom[1][0], '',
						nt[4], nt[1], nt_idx+1,
						coord[0], coord[1], coord[2],
						atom[1][3], atom[1][4],
						atom[1][2], atom[1][1]))
				f.write('#\n')

	# ── Build ─────────────────────────────────────────────────────────────────

	def Build(self, sequence, fmt='DNA'):
		'''Build a canonical double-stranded B-DNA or A-RNA helix'''
		sequence = sequence.upper()
		N = len(sequence)
		if fmt == 'DNA':
			comp_map    = {'A':'T','T':'A','G':'C','C':'G'}
			tri_map     = {'A':'DA','T':'DT','G':'DG','C':'DC'}
			rise, twist = 3.4,  36.0
		else:
			comp_map    = {'A':'U','U':'A','G':'C','C':'G'}
			tri_map     = {'A':'A','U':'U','G':'G','C':'C'}
			rise, twist = 2.81, 32.7

		def Rz(deg):
			t = math.radians(deg)
			c, s = math.cos(t), math.sin(t)
			return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

		Rflip = np.diag([-1.0, 1.0, -1.0])

		self.data = {
			'Energy':0, 'Rg':0, 'Mass':0, 'Size':0,
			'FASTA':None, 'Type':fmt,
			'Nucleotides':{}, 'Atoms':{}, 'Bonds':{},
			'Coordinates':np.zeros((0,3))}

		coords_list = []
		atom_idx    = 0
		nt_idx      = 0

		def add_nt(tricode, chain, k, flip, is_5prime):
			nonlocal atom_idx, nt_idx
			db       = self.NucleotidesDB[tricode]
			vecs     = np.array(db['Vectors'])
			bb_meta  = db['Backbone Atoms']
			bas_meta = db['Base Atoms']
			bonds_db = db['Bonds']
			n_bb     = len(bb_meta)
			skip     = 3 if is_5prime else 0   # skip P,OP1,OP2

			R = Rz(k * twist)
			T = np.array([0.0, 0.0, k * rise])
			if flip:
				transformed = (vecs @ Rflip.T) @ R.T + T
			else:
				transformed = vecs @ R.T + T

			local_to_global = {}
			bb_indices  = []
			bas_indices = []

			for li in range(n_bb):
				if li < skip:
					local_to_global[li] = -1
					continue
				am = bb_meta[li]
				self.data['Atoms'][atom_idx] = [
					am[0], am[1], am[2], 1.0, 0.0]
				coords_list.append(transformed[li])
				local_to_global[li] = atom_idx
				bb_indices.append(atom_idx)
				atom_idx += 1

			for li2, am in enumerate(bas_meta):
				li = n_bb + li2
				self.data['Atoms'][atom_idx] = [
					am[0], am[1], am[2], 1.0, 0.0]
				coords_list.append(transformed[li])
				local_to_global[li] = atom_idx
				bas_indices.append(atom_idx)
				atom_idx += 1

			bonds = self.data['Bonds']
			for k_str, v_list in bonds_db.items():
				li = int(k_str)
				gi = local_to_global.get(li, -1)
				if gi == -1:
					continue
				bonds.setdefault(gi, [])
				for lj in v_list:
					gj = local_to_global.get(lj, -1)
					if gj != -1 and gj not in bonds[gi]:
						bonds[gi].append(gj)

			sym = tricode[-1]
			self.data['Nucleotides'][nt_idx] = [
				sym, chain, bb_indices, bas_indices, tricode]
			nt_idx += 1

		for i, base in enumerate(sequence):
			add_nt(tri_map[base], 'A', i, False, i == 0)

		comp_seq = ''.join(comp_map[b] for b in reversed(sequence))
		for j, base in enumerate(comp_seq):
			add_nt(tri_map[base], 'B', N-1-j, True, j == 0)

		self.data['Coordinates'] = np.array(coords_list)

		nts   = self.data['Nucleotides']
		atoms = self.data['Atoms']
		bonds = self.data['Bonds']
		for strand_ids in [list(range(N)), list(range(N, 2*N))]:
			for i in range(len(strand_ids)-1):
				nt_i = nts[strand_ids[i]]
				nt_j = nts[strand_ids[i+1]]
				o3 = next(
					(a for a in nt_i[2]
					 if atoms[a][0] == "O3'"), None)
				p  = next(
					(a for a in nt_j[2]
					 if atoms[a][0] == 'P'), None)
				if o3 is None or p is None:
					continue
				bonds.setdefault(o3, [])
				bonds.setdefault(p,  [])
				if p  not in bonds[o3]: bonds[o3].append(p)
				if o3 not in bonds[p]:  bonds[p].append(o3)

		self.update_data()

	# ── Import ────────────────────────────────────────────────────────────────

	def Import(
			self, filename, chainA='A', chainB='B', model=1):
		'''Import nucleic acid from a .pdb or .cif file'''
		ext            = filename[-3:].upper()
		chains_to_load = [c for c in [chainA, chainB] if c]
		rows           = []

		if ext == 'PDB':
			has_models = False; in_target = False; found_models = []
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
						if in_target: break
						in_target = False
					elif rec in ('ATOM', 'HETATM'):
						ch = line[21]
						if ch not in chains_to_load: continue
						if has_models and not in_target: continue
						aname  = line[12:16].strip()
						resn   = line[17:20].strip()
						resnum = int(line[22:26].strip())
						x = float(line[30:38])
						y = float(line[38:46])
						z = float(line[46:54])
						occ  = float(line[54:60])
						bfac = float(line[60:66])
						elem = line[76:78].strip()
						if not elem:
							elem = aname.lstrip('0123456789')[0]
						rows.append(
							(aname,resn,ch,resnum,x,y,z,occ,bfac,elem))

		elif ext == 'CIF':
			with open(filename) as f: lines = f.readlines()
			col = {}; col_idx = 0; data_start = 0
			for idx, line in enumerate(lines):
				if line.strip() == 'loop_':
					nxt = (lines[idx+1].strip()
					       if idx+1 < len(lines) else '')
					if nxt.startswith('_atom_site.'):
						i = idx + 1
						while (i < len(lines) and
						lines[i].strip().startswith('_atom_site.')):
							cn = lines[i].strip().split('.')[1]
							col[cn] = col_idx; col_idx += 1; i += 1
						data_start = i; break
			i_grp   = col['group_PDB']
			i_atom  = col['auth_atom_id']
			i_resn  = col['auth_comp_id']
			i_chain = col['auth_asym_id']
			i_seqid = col['auth_seq_id']
			i_x     = col['Cartn_x']
			i_y     = col['Cartn_y']
			i_z     = col['Cartn_z']
			i_occ   = col['occupancy']
			i_bfac  = col['B_iso_or_equiv']
			i_type  = col['type_symbol']
			i_model = col.get('pdbx_PDB_model_num')
			i_alt   = col.get('label_alt_id')
			found_models = []
			for line in lines[data_start:]:
				ln = line.strip()
				if not ln or ln == '#': break
				flds = ln.split()
				if flds[i_grp] != 'ATOM': continue
				if flds[i_chain] not in chains_to_load: continue
				if i_model is not None:
					mnum = int(flds[i_model])
					if mnum not in found_models:
						found_models.append(mnum)
					if mnum != model: continue
				alt = flds[i_alt] if i_alt else '.'
				if alt not in ('.', 'A', '1', ''): continue
				rows.append((
					flds[i_atom], flds[i_resn],
					flds[i_chain], int(flds[i_seqid]),
					float(flds[i_x]), float(flds[i_y]),
					float(flds[i_z]),
					float(flds[i_occ]), float(flds[i_bfac]),
					flds[i_type]))

		if not rows:
			raise Exception(
				f'No ATOM records for chains '
				f'{chains_to_load} in {filename}')

		# De-duplicate alternates: keep highest occupancy
		best = {}
		for idx, row in enumerate(rows):
			key = (row[2], row[3], row[0])
			if key not in best or row[7] > rows[best[key]][7]:
				best[key] = idx
		rows = [rows[i] for i in sorted(best.values())]

		# Group by (chain, resnum)
		residues = defaultdict(list)
		for row in rows:
			residues[(row[2], row[3])].append(row)

		# Detect DNA vs RNA
		all_resnames = {atom_rows[0][1]
		                for atom_rows in residues.values()}
		if any(rn in ('DT','T') for rn in all_resnames):
			fmt = 'DNA'
		elif any(rn == 'U' for rn in all_resnames):
			fmt = 'RNA'
		else:
			fmt = 'DNA'   # default

		self.data = {
			'Energy':0, 'Rg':0, 'Mass':0, 'Size':0,
			'FASTA':None, 'Type':fmt,
			'Nucleotides':{}, 'Atoms':{}, 'Bonds':{},
			'Coordinates':np.zeros((0,3))}

		coords_list = []
		atom_idx    = 0
		nt_idx      = 0
		nt_range_by_chain = defaultdict(list)

		chain_order = chains_to_load
		sorted_keys = sorted(
			residues.keys(),
			key=lambda k: (
				chain_order.index(k[0])
				if k[0] in chain_order else 99, k[1]))

		for (chain, resnum) in sorted_keys:
			atom_rows   = residues[(chain, resnum)]
			resn_raw    = atom_rows[0][1]
			# Normalise residue name to NucleotidesDB key
			if fmt == 'DNA' and resn_raw in ('A','G','C'):
				tricode = 'D' + resn_raw
			elif resn_raw == 'T':
				tricode = 'DT'
			else:
				tricode = resn_raw
			if tricode not in self.NucleotidesDB:
				continue

			db       = self.NucleotidesDB[tricode]
			bb_meta  = db['Backbone Atoms']
			bas_meta = db['Base Atoms']
			bonds_db = db['Bonds']
			n_bb     = len(bb_meta)

			name_row = {r[0]: r for r in atom_rows}
			bb_indices  = []
			bas_indices = []
			local_to_global = {}

			for li, am in enumerate(bb_meta):
				row = name_row.get(am[0])
				if row is None:
					local_to_global[li] = -1; continue
				self.data['Atoms'][atom_idx] = [
					am[0], am[1], am[2], row[7], row[8]]
				coords_list.append([row[4], row[5], row[6]])
				local_to_global[li] = atom_idx
				bb_indices.append(atom_idx)
				atom_idx += 1

			for li2, am in enumerate(bas_meta):
				li  = n_bb + li2
				row = name_row.get(am[0])
				if row is None:
					local_to_global[li] = -1; continue
				self.data['Atoms'][atom_idx] = [
					am[0], am[1], am[2], row[7], row[8]]
				coords_list.append([row[4], row[5], row[6]])
				local_to_global[li] = atom_idx
				bas_indices.append(atom_idx)
				atom_idx += 1

			# Any extra atoms in PDB not in DB template
			db_names = ({am[0] for am in bb_meta}
			            | {am[0] for am in bas_meta})
			for aname, row in name_row.items():
				if aname in db_names: continue
				el = row[9] if row[9] else aname[0]
				self.data['Atoms'][atom_idx] = [
					aname, el, 0.0, row[7], row[8]]
				coords_list.append([row[4], row[5], row[6]])
				if aname in self.BB_ATOMS:
					bb_indices.append(atom_idx)
				else:
					bas_indices.append(atom_idx)
				atom_idx += 1

			bonds = self.data['Bonds']
			for k_str, v_list in bonds_db.items():
				li = int(k_str)
				gi = local_to_global.get(li, -1)
				if gi == -1: continue
				bonds.setdefault(gi, [])
				for lj in v_list:
					gj = local_to_global.get(lj, -1)
					if gj != -1 and gj not in bonds[gi]:
						bonds[gi].append(gj)

			sym = tricode[-1]
			self.data['Nucleotides'][nt_idx] = [
				sym, chain, bb_indices, bas_indices, tricode]
			nt_range_by_chain[chain].append(nt_idx)
			nt_idx += 1

		self.data['Coordinates'] = (
			np.array(coords_list) if coords_list
			else np.zeros((0,3)))

		atoms = self.data['Atoms']
		bonds = self.data['Bonds']
		nts   = self.data['Nucleotides']
		for chain_nt_ids in nt_range_by_chain.values():
			for i in range(len(chain_nt_ids)-1):
				nt_i = nts[chain_nt_ids[i]]
				nt_j = nts[chain_nt_ids[i+1]]
				o3 = next(
					(a for a in nt_i[2]
					 if atoms[a][0] == "O3'"), None)
				p  = next(
					(a for a in nt_j[2]
					 if atoms[a][0] == 'P'), None)
				if o3 is None or p is None: continue
				bonds.setdefault(o3, [])
				bonds.setdefault(p,  [])
				if p  not in bonds[o3]: bonds[o3].append(p)
				if o3 not in bonds[p]:  bonds[p].append(o3)

		self.update_data()

	# ── Angle measurement ─────────────────────────────────────────────────────

	def Angle(self, nt, angle_type):
		'''Measure a backbone (alpha/beta/gamma/delta/epsilon/zeta) or
		chi dihedral angle in degrees'''
		nts = self.data['Nucleotides']
		at  = angle_type.lower()
		prv = self._prev_nt(nt)
		nxt = self._next_nt(nt)
		try:
			if at == 'alpha':
				if prv is None: return 0.0
				r1 = self.GetAtom(prv, "O3'")
				r2 = self.GetAtom(nt,  'P')
				r3 = self.GetAtom(nt,  "O5'")
				r4 = self.GetAtom(nt,  "C5'")
			elif at == 'beta':
				r1 = self.GetAtom(nt, 'P')
				r2 = self.GetAtom(nt, "O5'")
				r3 = self.GetAtom(nt, "C5'")
				r4 = self.GetAtom(nt, "C4'")
			elif at == 'gamma':
				r1 = self.GetAtom(nt, "O5'")
				r2 = self.GetAtom(nt, "C5'")
				r3 = self.GetAtom(nt, "C4'")
				r4 = self.GetAtom(nt, "C3'")
			elif at == 'delta':
				r1 = self.GetAtom(nt, "C5'")
				r2 = self.GetAtom(nt, "C4'")
				r3 = self.GetAtom(nt, "C3'")
				r4 = self.GetAtom(nt, "O3'")
			elif at == 'epsilon':
				if nxt is None: return 0.0
				r1 = self.GetAtom(nt,  "C4'")
				r2 = self.GetAtom(nt,  "C3'")
				r3 = self.GetAtom(nt,  "O3'")
				r4 = self.GetAtom(nxt, 'P')
			elif at == 'zeta':
				if nxt is None: return 0.0
				r1 = self.GetAtom(nt,  "C3'")
				r2 = self.GetAtom(nt,  "O3'")
				r3 = self.GetAtom(nxt, 'P')
				r4 = self.GetAtom(nxt, "O5'")
			elif at == 'chi':
				tricode = nts[nt][4]
				catoms  = (
					self.NucleotidesDB[tricode]['Chi Angle Atoms'])
				r1 = self.GetAtom(nt, catoms[0])
				r2 = self.GetAtom(nt, catoms[1])
				r3 = self.GetAtom(nt, catoms[2])
				r4 = self.GetAtom(nt, catoms[3])
			else:
				raise Exception(f'Unknown angle type: {angle_type}')
		except Exception:
			return 0.0
		u1 = r2 - r1; u2 = r3 - r2; u3 = r4 - r3
		mag_u2    = np.linalg.norm(u2)
		u1u2      = np.cross(u1, u2)
		u2u3      = np.cross(u2, u3)
		u1u2Cu2u3 = np.cross(u1u2, u2u3)
		a = np.dot(u2, u1u2Cu2u3)
		b = mag_u2 * np.dot(u1u2, u2u3)
		return math.atan2(a, b) * 180 / math.pi

	# ── Gasteiger partial charges ─────────────────────────────────────────────

	def Gasteiger(self, iterations=6):
		'''Assign Gasteiger-Marsili partial charges to all atoms'''
		PARAMS = {
			'C3': (7.98,  9.18,  1.88),
			'C2': (8.79,  9.32,  1.51),
			'C1': (10.39, 9.45,  0.73),
			'H':  (7.17,  6.24, -0.56),
			'O3': (14.18, 12.92, 1.39),
			'O2': (17.07, 13.79, 0.47),
			'N3': (11.54, 10.82, 1.36),
			'N2': (12.87, 11.15, 0.85),
			'P':  (10.14,  9.13, 1.38),
			'S':  (10.14,  9.13, 1.38),
			'Se': (9.00,   8.00, 1.10)}
		ids  = sorted(self.data['Atoms'].keys())
		crds = self.data['Coordinates']
		els  = [self.data['Atoms'][i][1].upper() for i in ids]
		c    = crds[np.array(ids)]
		dm   = np.sqrt(((c[:,None,:]-c[None,:,:])**2).sum(2))
		is_H = np.array([e == 'H'            for e in els])
		is_S = np.array([e in ('S','SE','P') for e in els])
		heavy_thresh = np.full((len(ids), len(ids)), 1.9)
		h_mask = is_H[:,None] | is_H[None,:]
		s_mask = is_S[:,None] | is_S[None,:]
		heavy_thresh[h_mask] = 1.3
		heavy_thresh[s_mask] = 2.1
		bond_mask = (dm < heavy_thresh) & (dm > 0.0)
		bonds = {i: [] for i in ids}
		for ii, jj in np.argwhere(bond_mask):
			bonds[ids[ii]].append(ids[jj])
		heavy    = ~is_H
		short    = (dm < 1.42) & (dm > 0)
		sp2_mask = short & heavy[:,None] & heavy[None,:]
		sp2  = {ids[ii] for ii in np.where(sp2_mask.any(axis=1))[0]}
		def types(ii, i):
			el = els[ii]
			if el == 'H':   return 'H'
			if el == 'S':   return 'S'
			if el == 'SE':  return 'Se'
			if el == 'P':   return 'P'
			nb   = len(bonds[i])
			isp2 = i in sp2
			if el == 'C':
				if isp2:    return 'C2'
				if nb <= 2: return 'C1'
				return 'C3'
			if el == 'N':   return 'N2' if isp2 else 'N3'
			if el == 'O':   return 'O2' if isp2 else 'O3'
			return 'C3'
		charges = {i: self.data['Atoms'][i][2] for i in ids}
		atype   = {i: PARAMS[types(ii,i)] for ii,i in enumerate(ids)}
		for n in range(iterations):
			damp  = 1.0 / (2 ** (n+1))
			chi   = {i: a + q*(b+c*q)
			         for i in ids
			         for a,b,c in [atype[i]]
			         for q     in [charges[i]]}
			delta = {i: 0.0 for i in ids}
			for i in ids:
				for j in bonds[i]:
					if j <= i: continue
					if chi[j] >= chi[i]:
						donor, acceptor = i, j
					else:
						donor, acceptor = j, i
					a,b,c = atype[donor]
					ip = a + b + c
					if ip == 0: continue
					dq = damp * (chi[acceptor]-chi[donor]) / ip
					delta[donor]    += dq
					delta[acceptor] -= dq
			for i in ids: charges[i] += delta[i]
		for i in ids:
			self.data['Atoms'][i][2] = round(charges[i], 4)

	# ── Structural manipulation ───────────────────────────────────────────────

	def Rotation3Angle(
			self, nt1, atom1, nt2, atom2, nt3, atom3, theta):
		'''Rotate structure after atom2 to adjust nt1-nt2-nt3 angle'''
		atom2i = self._get_atom_idx(nt2, atom2)
		A      = (self.GetAtom(nt3, atom3)
		          - self.GetAtom(nt1, atom1))
		B      = (self.GetAtom(nt3, atom3)
		          - self.GetAtom(nt2, atom2))
		u  = np.cross(B, A)
		lu = np.linalg.norm(u)
		if lu < 1e-10:
			return
		u      = u / lu
		ori    = self.GetAtom(nt2, atom2).copy()
		before = self.data['Coordinates'][:atom2i]
		after  = self.data['Coordinates'][atom2i:] - ori
		RM     = self.Rotation_Matrix(theta, u)
		after  = np.matmul(after, RM) + ori
		self.data['Coordinates'] = np.append(before, after, axis=0)

	def Adjust(self, nt1, atom1, nt2, atom2, length):
		'''Translate atom2 and its downstream atoms to set bond length'''
		Ai     = self._get_atom_idx(nt1, atom1)
		Bi     = self._get_atom_idx(nt2, atom2)
		coords = self.data['Coordinates']
		v      = coords[Bi] - coords[Ai]
		mag    = np.linalg.norm(v)
		if mag < 1e-10:
			return
		shift = v * (length / mag) - v
		for idx in self._downstream_atoms(Ai, Bi):
			coords[idx] += shift
		self.data['Coordinates'] = coords

	def Rotate(self, nt, theta, angle_type):
		'''Set a backbone or chi torsion to theta degrees (absolute)'''
		at  = angle_type.lower()
		nts = self.data['Nucleotides']
		prv = self._prev_nt(nt)
		nxt = self._next_nt(nt)
		try:
			if at == 'alpha':
				if prv is None: return
				piv_a = self._get_atom_idx(prv, "O3'")
				piv_b = self._get_atom_idx(nt,  'P')
			elif at == 'beta':
				piv_a = self._get_atom_idx(nt, 'P')
				piv_b = self._get_atom_idx(nt, "O5'")
			elif at == 'gamma':
				piv_a = self._get_atom_idx(nt, "O5'")
				piv_b = self._get_atom_idx(nt, "C5'")
			elif at == 'delta':
				piv_a = self._get_atom_idx(nt, "C5'")
				piv_b = self._get_atom_idx(nt, "C4'")
			elif at == 'epsilon':
				if nxt is None: return
				piv_a = self._get_atom_idx(nt, "C4'")
				piv_b = self._get_atom_idx(nt, "C3'")
			elif at == 'zeta':
				if nxt is None: return
				piv_a = self._get_atom_idx(nt,  "C3'")
				piv_b = self._get_atom_idx(nt,  "O3'")
			elif at == 'chi':
				tricode = nts[nt][4]
				catoms  = (
					self.NucleotidesDB[tricode]['Chi Angle Atoms'])
				piv_a = self._get_atom_idx(nt, catoms[1])
				piv_b = self._get_atom_idx(nt, catoms[2])
			else:
				raise Exception(f'Unknown angle type: {angle_type}')
		except Exception:
			return
		current    = self.Angle(nt, angle_type)
		downstream = self._downstream_atoms(piv_a, piv_b)
		coords     = self.data['Coordinates']
		ori        = coords[piv_b].copy()
		u          = coords[piv_a] - coords[piv_b]
		u          = u / np.linalg.norm(u)
		RM_zero    = self.Rotation_Matrix(-current, u)
		RM_new     = self.Rotation_Matrix(theta, u)
		for idx in downstream:
			v           = coords[idx] - ori
			v           = np.matmul(v, RM_zero)
			v           = np.matmul(v, RM_new)
			coords[idx] = v + ori
		self.data['Coordinates'] = coords

	def RotatePose(self, theta=None, u=None, l=None, ori=None):
		'''Rigidly rotate and/or translate the full pose'''
		coords = self.data['Coordinates'].copy()
		if theta is not None and u is not None:
			u   = np.array(u, dtype=float)
			mag = np.linalg.norm(u)
			if mag < 1e-10:
				raise Exception(
					'Rotation axis u cannot be a zero vector')
			u      = u / mag
			pivot  = coords.mean(axis=0)
			R      = self.Rotation_Matrix(theta, u)
			coords = np.matmul(coords - pivot, R) + pivot
		if l is not None and ori is not None:
			ori      = np.array(ori, dtype=float)
			centroid = coords.mean(axis=0)
			d        = ori - centroid
			mag      = np.linalg.norm(d)
			if mag < 1e-10:
				raise Exception(
					'ori coincides with pose centroid')
			coords = coords + (d / mag) * l
		self.data['Coordinates'] = coords
