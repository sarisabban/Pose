#!/usr/bin/env python3

import os
import math
import json
import copy
import datetime
import numpy as np
from collections import defaultdict

class Pose():
	''' Data structure that represents a protein '''
	def __init__(self):
#		path = __file__.split('/')[:-1]
#		path = '/'.join(path)
		path, modulename = os.path.split(__file__)
		with open(f'{path}/AminoAcids.json') as f: AminoAcids = json.load(f)
		Masses = {
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
		data ={
			'Energy':0,
			'Rg':0,
			'Mass':0,
			'Size':0,
			'FASTA':None,
			'Amino Acids':{},
			'Atoms':{},
			'Bonds':{},
			'Coordinates':np.array([[0, 0, 0]])}
		self.AminoAcids = AminoAcids
		self.Masses = Masses
		self.data = data
	def PDB_entry(self, atom, n, a, l, r, c, s, i, x, y, z, o, t, q, e):
		''' Construct a PDB atom entry '''
		ATOM = '{:<6}'.format(atom)
		N = '{:>5}  '.format(n)
		A = '{:<4}'.format(a)
		L = '{:>0}'.format(l)
		R = '{:>3}'.format(r)
		C = '{:>2}'.format(c)
		S = '{:>4}'.format(s)
		I = '{:>1}   '.format(i)
		X = '{:>8.3f}'.format(x)
		Y = '{:>8.3f}'.format(y)
		Z = '{:>8.3f} '.format(z)
		O = '{:>5.2f} '.format(o)
		T = '{:>5.2f} '.format(t)
		Q = '{:>9.3f} '.format(q)
		E = '{:<2} \n'.format(e)
		entry = ATOM + N + A + L + R + C + S + I + X + Y + Z + O + T + Q + E
		return(entry)
	def Export(self, filename):
		''' Export pose to a .pdb file '''
		with open(filename, 'w') as f:
			DATE = datetime.date.today().strftime("%d-%b-%Y")
			H1 = 'HEADER'+' '*44+DATE+' '*3+'XXXX'+' '*11+'\n'
			H2 = 'EXPDTA'+' '*4+'THEORETICAL MODEL'+' '*52+'\n'
			H3 = 'REMARK 220 REMARK: MODEL GENERATED BY SARI SABBAN'+' '*30+'\n'
			f.write(H1)
			f.write(H2)
			f.write(H3)
			atoms = self.data['Atoms'].items()
			aminoacids = self.data['Amino Acids']
			coordinates = self.data['Coordinates']
			AAindex = 0
			BBlen = len(self.data['Amino Acids'][AAindex][2])
			SClen = len(self.data['Amino Acids'][AAindex][3])
			length = BBlen + SClen - 1
			for atom, coordinate in zip(atoms, coordinates):
				A = 'ATOM'
				n = atom[0] + 1
				a = atom[1][0]
				l = ''
				r = self.data['Amino Acids'][AAindex][5]
				c = self.data['Amino Acids'][AAindex][1]
				s = AAindex + 1
				i = ''
				x = coordinate[0]
				y = coordinate[1]
				z = coordinate[2]
				o = 1.0
				t = atom[1][3]
				q = atom[1][2]
				e = atom[1][1]
				line = self.PDB_entry(A,n,a,l,r,c,s,i,x,y,z,o,t,q,e)
				f.write(line)
				if length != 0:
					length -= 1
				elif length == 0:
					AAindex += 1
					try:
						BBlen = len(self.data['Amino Acids'][AAindex][2])
						SClen = len(self.data['Amino Acids'][AAindex][3])
						length = BBlen + SClen - 1
					except:
						continue
			TER = 'TER'
			f.write(TER)
	def GetAtom(self, AA, atom):
		''' Get specific atom coordinates '''
		info = self.data['Amino Acids'][AA]
		backbone = ['N', '1H', '2H', '3H', 'CA', 'HA', 'C', 'O', 'OXT']
		if atom in backbone:
			indexes = info[2]
		else:
			indexes = info[3]
		for i in indexes:
			A = self.data['Atoms'][i][0]
			if A == atom:
				coordinates = self.data['Coordinates'][i]
				return(coordinates)
		raise Exception(f'Amino acid does not have the {atom} atom')
	def Insert(self, AA, X, Y, Z):
		''' Inser a backbone or sidechain given its initial coordinates '''
		if len(AA) == 1: AA = AA.upper()
		atoms = np.array(self.AminoAcids[AA]['Vectors']) + np.array([X, Y, Z])
		return(atoms)
	def Flip(self, AA):
		''' Flip an amino acid 180 degrees on CA's H1 H2 axis'''
		p = AA[2]
		AA = AA - p
		H1 = AA[3]
		H2 = AA[4]
		u = np.cross(H1, H2)
		lu = np.linalg.norm(u)
		u = u / lu
		TM = np.array([
		[2*u[0]**2-1, 2*u[0]*u[1], 2*u[0]*u[2]],
		[2*u[0]*u[1], 2*u[1]**2-1, 2*u[2]*u[2]],
		[2*u[0]*u[2], 2*u[2]*u[2], 2*u[2]**2-1]])
		AA = np.matmul(AA, TM)
		AA = AA + p
		return(AA)
	def LD(self, AA):
		''' Convert an amino acid between L and D chirality '''
		AA = AA * [1, 1, -1]
		return(AA)
	def Amino(self, backbone_type, X, Y, Z, aa, index, flip=False, LD=False):
		''' Construct an amino acid and add its coordinates to the data '''
		BB = self.Insert(backbone_type, X, Y, Z)
		SC = self.Insert(aa, X, Y, Z)
		AA = np.insert(BB, index, SC, axis=0)
		if LD: AA = self.LD(AA)
		if flip: AA = self.Flip(AA)
		if aa == 'P': AA = np.delete(AA, [1], axis=0)
		self.data['Coordinates'] = \
		np.append(self.data['Coordinates'], AA, axis=0)
		if backbone_type == 'Backbone' or backbone_type == 'Backbone start':
			self.data['Coordinates'] = \
			np.delete(self.data['Coordinates'], [0], axis=0)
	def Atoms(self, AA, chain, backbone_type, BB_index, AA_index, I, LD=False):
		''' Construct, and add to pose, atom and amino acid identities '''
		BB = backbone_type[:BB_index]
		if AA == 'P': BB.pop(1)
		BB = BB + self.AminoAcids[AA.upper()]['Sidechain Atoms']
		BB = BB + backbone_type[BB_index:]
		BBi = []
		SCi = []
		for atomi, v in enumerate(BB, I):
			self.data['Atoms'][atomi] = v
			if v[0] in ['N', '1H', '2H', '3H', 'CA', 'HA', 'C', 'O', 'OXT']:
				BBi.append(atomi)
			else:
				SCi.append(atomi)
		tri = self.AminoAcids[AA.upper()]['Tricode']
		if LD: tri = 'D' + tri[1:]
		self.data['Amino Acids'][AA_index] = [AA, chain, BBi, SCi, 'L', tri]
	def BondTree_PRO(self, BB, SC):
		''' Construct proline bond graph by adding sidechain to backbone '''
		BBb = copy.deepcopy(self.AminoAcids[BB]['Bonds'])
		SCb = copy.deepcopy(self.AminoAcids[SC]['Bonds'])
		for key in list(BBb.keys()): BBb[int(key)] = BBb.pop(key)
		for key in list(SCb.keys()): SCb[int(key)] = SCb.pop(key)
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
		for i, (k, v) in enumerate(zip(SCb.keys(), SCb.values()), start=n+2):
			if k < 0: break
			k = i
			if BB == 'Backbone' or BB == 'Backbone start':
				v = [x+n+2 for x in v]
			else:
				v = [x+n+4 if x<0 else x+n+2 for x in v]
			if i == n+2: v.append(n)
			BBb[k] = v
			BBb[k] = sorted(BBb[k])
		return(BBb)
	def BondTree_AA(self, BB, SC):
		''' Construct amino acid bond graph by adding sidechain to backbone '''
		SC = SC.upper()
		if SC == 'P':
			BBb = self.BondTree_PRO(BB, SC)
			return(BBb)
		BBb = copy.deepcopy(self.AminoAcids[BB]['Bonds'])
		SCb = copy.deepcopy(self.AminoAcids[SC]['Bonds'])
		for key in list(BBb.keys()): BBb[int(key)] = BBb.pop(key)
		for key in list(SCb.keys()): SCb[int(key)] = SCb.pop(key)
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
		for i, (k, v) in enumerate(zip(SCb.keys(), SCb.values()), start=n+2):
			k = i
			if length != 1: v = [x+n+2 for x in v]
			if length == 1: v = [x+n+1 for x in v]
			if i == n+2 and length != 1: v.append(n)
			BBb[k] = v
			BBb[k] = sorted(BBb[k])
		return(BBb)
	def BondTree(self, BB, AA):
		''' Update the pose bond graph when adding a new amino acid '''
		BBb = self.BondTree_AA(BB, AA)
		BT = self.data['Bonds']
		length = len(BT)
		if length == 0:
			self.data['Bonds'] = BBb
			return
		i_max = max([k for k in BT.keys()])
		BT[i_max-1] += [i_max+1]
		for i in range(len(BBb)):
			K = i+length
			v = BBb[i]
			V = [x+length for x in v]
			if i == 0: V.append(i_max-1)
			BT[K] = V
		self.data['Bonds'] = BT
	def FASTA(self):
		''' Return FASTA sequence of peptide as a string '''
		AAs = self.data['Amino Acids']
		AAs = [x[0] for x in AAs.values()]
		FASTA = ''.join(AAs)
		return(FASTA)
	def SecondaryStructures(self):
		''' Return secondary strucutre of each amino acid of the peptide '''
		SS = [x[4] for x in self.data['Amino Acids'].values()]
		return(SS)
	def Distance(self, AA1, atom1, AA2, atom2):
		''' Measure distance between any two atoms '''
		A = self.GetAtom(AA1, atom1)
		B = self.GetAtom(AA2, atom2)
		mag = np.linalg.norm(B - A)
		return(mag)
	def Size(self):
		''' Calculate length of peptide '''
		AAs = self.data['Amino Acids']
		AAs = [x[0] for x in AAs.values()]
		length = len(AAs)
		return(length)
	def AtomList(self, PDB=False):
		''' Return list of all the atoms '''
		As = self.data['Atoms']
		if PDB:
			As = [x[0] for x in As.values()]
		else:
			As = [x[1] for x in As.values()]
		return(As)
	def Identify(self, index, item, q=False):
		''' Identify an atom, atom charge, or amino acid given its index '''
		if item.upper() == 'ATOM':
			Atom = self.data['Atoms'][index]
			if q: return(Atom[-1])
			else: return(Atom[0])
		elif item.upper() == 'RESIDUE' or item.upper() == 'AMINO ACID':
			AminoAcid = self.data['Amino Acids'][index][0]
			return(AminoAcid)
		else:
			raise Exception('Incorrect item')
	def Mass(self):
		''' Calculate mass of peptide in Da'''
		atoms = self.AtomList()
		masses = [self.Masses[x] for x in atoms]
		mass = sum(masses)
		mass = round(mass, 3)
		return(mass)
	def Rg(self):
		''' Calculate the radius of gyration of a peptide '''
		coord = self.data['Coordinates'].tolist()
		atoms = self.AtomList()
		mass = [self.Masses[x] for x in atoms]
		xm = [(m*i, m*j, m*k) for (i, j, k), m in zip(coord, mass)]
		tmass = sum(mass)
		rr = sum(mi*i + mj*j + mk*k for (i,j,k), (mi,mj,mk) in zip(coord, xm))
		mm = sum((sum(i) / tmass)**2 for i in zip(*xm))
		rg = math.sqrt(rr / tmass-mm)
		return(round(rg, 3))
	def Atom3Angle(self, AA1, atom1, AA2, atom2, AA3, atom3):
		''' Measure the angle between three atoms '''
		atom1 = self.GetAtom(AA1, atom1)
		atom2 = self.GetAtom(AA2, atom2)
		atom3 = self.GetAtom(AA3, atom3)
		A = atom2 - atom1
		B = atom2 - atom3
		magA = math.sqrt(A[0]**2 + A[1]**2 + A[2]**2)
		magB = math.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
		cos_theta = np.dot(A, B) / (magA * magB)
		theta = math.acos(cos_theta)
		theta = theta * 180 / math.pi
		return(theta)
	def GetBondAtoms(self, index1, index2):
		''' Get the atom pair that participate in a bond from their index '''
		bonds = self.data['Bonds'][index1]
		if index2 not in bonds:
			error = 'Requested two atoms are not bonded'
			raise Exception(error)
		atomA = self.data['Atoms'][index1][0]
		elementA = self.data['Atoms'][index1][1]
		atomB = self.data['Atoms'][index2][0]
		elementB = self.data['Atoms'][index2][1]
		return([atomA, elementA, atomB, elementB])
	def Info(self):
		''' Print all basic info about a peptide '''
		print('Sequence:\t{}'.format(self.data['FASTA']))
		print('SS:\t\t{}'.format(''.join(self.SecondaryStructures())))
		print('Mass:\t\t{} Da'.format(self.data['Mass']))
		print('Size:\t\t{} residues'.format(self.data['Size']))
		print('Rg:\t\t{} Å'.format(self.data['Rg']))
		print('Energy:\t\t{}'.format(self.data['Energy']))
	def Build(self, sequence, chain='A'):
		''' Build a polypeptide primary structure from sequence '''
		X, Y, Z = 0, 0, 0
		Ex_adjust, Ey_adjust, Ez_adjust = 0.400, 1.472, 0
		Ox_adjust, Oy_adjust, Oz_adjust = 0.812, 0.940, 0
		for i, aa in enumerate(list(sequence)):
			if len(sequence) == 1:
				I = len(self.data['Coordinates']) - 1
				if aa.isupper():   LD = False
				elif aa.islower(): LD = True
				self.Amino('Backbone', X, Y, Z, aa, [6], LD=LD)
				AAs = self.AminoAcids['Backbone']['Backbone Atoms']
				self.Atoms(aa, chain, AAs, 6, i, I, LD= LD)
				self.BondTree('Backbone', aa)
			elif i == 0:
				I = len(self.data['Coordinates']) - 1
				if aa.isupper():   LD = False
				elif aa.islower(): LD = True
				self.Amino('Backbone start', X, Y, Z, aa, [6], LD=LD)
				AAs = self.AminoAcids['Backbone start']['Backbone Atoms']
				self.Atoms(aa, chain, AAs, 6, i, I, LD=LD)
				self.BondTree('Backbone start', aa)
			elif i == len(sequence)-1:
				if (i % 2) != 0:
					X = self.data['Coordinates'][-2][0] + Ex_adjust
					Y = self.data['Coordinates'][-2][1] + Ey_adjust
					Z = self.data['Coordinates'][-2][2] + Ez_adjust
					I = len(self.data['Coordinates']) - 0
					if aa.isupper():   LD = False
					elif aa.islower(): LD = True
					self.Amino('Backbone end', X, Y, Z, aa, [4],
					flip=True, LD=LD)
					AAs = self.AminoAcids['Backbone end']['Backbone Atoms']
					self.Atoms(aa, chain, AAs, 4, i, I, LD=LD)
					self.BondTree('Backbone end', aa)
				elif (i % 2) == 0:
					X = self.data['Coordinates'][-2][0] + Ox_adjust
					Y = self.data['Coordinates'][-2][1] + Oy_adjust
					Z = self.data['Coordinates'][-2][2] + Oz_adjust
					I = len(self.data['Coordinates']) - 0
					if aa.isupper():   LD = False
					elif aa.islower(): LD = True
					self.Amino('Backbone end', X, Y, Z, aa, [4], LD=LD)
					AAs = self.AminoAcids['Backbone end']['Backbone Atoms']
					self.Atoms(aa, chain, AAs, 4, i, I, LD=LD)
					self.BondTree('Backbone end', aa)
			else:
				if (i % 2) != 0:
					X = self.data['Coordinates'][-2][0] + Ex_adjust
					Y = self.data['Coordinates'][-2][1] + Ey_adjust
					Z = self.data['Coordinates'][-2][2] + Ez_adjust
					I = len(self.data['Coordinates']) - 0
					if aa.isupper():   LD = False
					elif aa.islower(): LD = True
					self.Amino('Backbone middle', X, Y, Z, aa, [4],
					flip=True, LD=LD)
					AAs = self.AminoAcids['Backbone middle']['Backbone Atoms']
					self.Atoms(aa, chain, AAs, 4, i, I, LD=LD)
					self.BondTree('Backbone middle', aa)
				elif (i % 2) == 0:
					X = self.data['Coordinates'][-2][0] + Ox_adjust
					Y = self.data['Coordinates'][-2][1] + Oy_adjust
					Z = self.data['Coordinates'][-2][2] + Oz_adjust
					I = len(self.data['Coordinates']) - 0
					if aa.isupper():   LD = False
					elif aa.islower(): LD = True
					self.Amino('Backbone middle', X, Y, Z, aa, [4], LD=LD)
					AAs = self.AminoAcids['Backbone middle']['Backbone Atoms']
					self.Atoms(aa, chain, AAs, 4, i, I, LD=LD)
					self.BondTree('Backbone middle', aa)
		self.data['Mass'] = self.Mass()
		self.data['FASTA'] = self.FASTA()
		self.data['Size'] = self.Size()
		self.data['Rg'] = self.Rg()
	def Adjust(self, AA1, atom1, AA2, atom2, length):
		''' Change the distance between any two atoms '''
		BB_ATOMS = ['N', 'CA', 'C', 'O']
		sidechain = False
		backbone = True
		if (atom1 not in BB_ATOMS or atom2 not in BB_ATOMS):
			sidechain = True
			backbone = False
		if sidechain:
			index = self.data['Amino Acids'][AA1][2] + \
					self.data['Amino Acids'][AA1][3]
			Ai = None
			Bi = None
			for idx in index:
				atoms = self.data['Atoms'][idx][0]
				if atom1 == atoms:
					Ai = idx
				elif atom2 == atoms:
					Bi = idx
			coordinates = self.data['Coordinates']
			vectors = {}
			for k, v in zip(self.data['Bonds'].keys(), \
			self.data['Bonds'].values()):
				vbonds = [coordinates[x] - coordinates[k] for x in v]
				vectors[k] = vbonds
			Bidx = None
			current_bonds = self.data['Bonds'][Ai]
			for i, b in enumerate(current_bonds):
				if Bi == b:
					Bidx = i
			v = vectors[Ai][Bidx]
			mag = np.linalg.norm(v)
			v = v*(length/mag)
			vectors[Ai][Bidx] = v
			temp = {}
			Bond_Coord = {0: self.data['Coordinates'][0]}
			for i in range(len(vectors)):
				vs = [Bond_Coord[i] + x for x in vectors[i]]
				temp[i] = vs
				bonds = self.data['Bonds'][i]
				for b, v  in zip(bonds, vs):
					Bond_Coord[b] = v
			coordinates = [0 for x in range(len(temp))]
			for k, vs in zip(temp.keys(), temp.values()):
				bonds = self.data['Bonds'][k]
				for b, v in zip(bonds, vs):
					coordinates[b] = v
			coordinates = np.array(coordinates)
			self.data['Coordinates'] = coordinates
		elif backbone:
			A = self.GetAtom(AA1, atom1)
			B = self.GetAtom(AA2, atom2)
			mag = math.sqrt(np.sum((B-A)**2))
			mut = length/mag
			aa1 = self.data['Amino Acids'][AA1][0]
			aa1 = atom1 in [x[0] for x \
			in self.AminoAcids[aa1.upper()]['Sidechain Atoms']]
			aa2 = self.data['Amino Acids'][AA2][0]
			aa2 = atom2 in [x[0] for x \
			in self.AminoAcids[aa2.upper()]['Sidechain Atoms']]
			Aelements = (self.data['Coordinates'] == A)
			Awhole_row = Aelements.all(axis=1)
			Aindex = np.argwhere(Awhole_row)[0][0]
			Belements = (self.data['Coordinates'] == B)
			Bwhole_row = Belements.all(axis=1)
			Bindex = np.argwhere(Bwhole_row)[0][0]
			if aa2 or aa1:
				error = 'Distance adjustments allowed only for backbone'
				raise Exception(error)
			else:
				before = self.data['Coordinates'][:Bindex]
				after = self.data['Coordinates'][Bindex:]
				nB = (B-A) * mut
				after = after - (B-A) + nB
				new = np.concatenate((before, after))
				self.data['Coordinates'] = new
	def Angle(self, AA, angle_type, chi_type=None):
		''' Measure angle at bond '''
		AminoAcid = self.data['Amino Acids'][AA][0].upper()
		if angle_type.upper() == 'PHI':
			if AA == 0: return(0.0)
			else: r1 = self.GetAtom(AA-1, 'C')
			r2 = self.GetAtom(AA, 'N')
			r3 = self.GetAtom(AA, 'CA')
			r4 = self.GetAtom(AA, 'C')
		if angle_type.upper() == 'PSI':
			r1 = self.GetAtom(AA, 'N')
			r2 = self.GetAtom(AA, 'CA')
			r3 = self.GetAtom(AA, 'C')
			try:
				r4 = self.GetAtom(AA+1, 'N')
			except:
				return(0.0)
		if angle_type.upper() == 'OMEGA':
			r1 = self.GetAtom(AA, 'CA')
			r2 = self.GetAtom(AA, 'C')
			try:
				r3 = self.GetAtom(AA+1, 'N')
				r4 = self.GetAtom(AA+1, 'CA')
			except: return(180.0)
		if angle_type.upper() == 'CHI':
			assert type(chi_type) is int, 'Incorrect Chi angle type'
			number_of_chis = len(self.AminoAcids[AminoAcid]['Chi Angle Atoms'])
			if not (number_of_chis >= chi_type):
				error = '{} amino acid at position {} has no Chi {} angle' \
				.format(AminoAcid, AA, chi_type)
				raise Exception(error)
			atoms = self.AminoAcids[AminoAcid]['Chi Angle Atoms'][chi_type - 1]
			r1 = self.GetAtom(AA, atoms[0])
			r2 = self.GetAtom(AA, atoms[1])
			r3 = self.GetAtom(AA, atoms[2])
			r4 = self.GetAtom(AA, atoms[3])
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
		theta = math.atan2(a, b) * 180 / math.pi
		return(theta)
	def Rotation_Matrix(self, theta, u):
		''' Rotate a matrix around axis u by theta angle '''
		ux, uy, uz = u[0], u[1], u[2]
		S = math.sin(math.radians(theta))
		C = math.cos(math.radians(theta))
		R = np.array([
		[C+ux**2*(1-C)   , ux*uy*(1-C)-uz*S, ux*uz*(1-C)+uy*S],
		[uy*ux*(1-C)+uz*S, C+uy**2*(1-C)   , uy*uz*(1-C)-ux*S],
		[uz*ux*(1-C)-uy*S, uz*uy*(1-C)+ux*S, C+uz**2*(1-C)   ]])
		return(R)
	def Rotate(self, AA, theta, angle_type, chi_type=None):
		''' Rotate around a bond '''
		AminoAcid = self.data['Amino Acids'][AA][0].upper()
		if angle_type.upper() == 'PHI':
			ori = self.GetAtom(AA, 'CA')
			n = 1
			if AA == 0: n = 3
			index = self.data['Amino Acids'][AA][2][2]
			before = self.data['Coordinates'][:index + n]
			after = self.data['Coordinates'][index + n:]
			after = after - ori
			A = self.GetAtom(AA, 'CA')
			B = self.GetAtom(AA, 'N')
			u = B - A
			lu = np.linalg.norm(u)
			u = u / lu
			current = self.Angle(AA, 'phi')
			zeroing = 0 - current
			RM = self.Rotation_Matrix(zeroing, u)
			after = np.matmul(after, RM)
			RM = self.Rotation_Matrix(theta, u)
			after = np.matmul(after, RM)
			after = after + ori
			combine = np.append(before, after, axis=0)
			self.data['Coordinates'] = combine
		if angle_type.upper() == 'PSI':
			ori = self.GetAtom(AA, 'C')
			n = 1
			if AA == 0: n = 3
			index = self.data['Amino Acids'][AA][2][4]
			before = self.data['Coordinates'][:index + n]
			after = self.data['Coordinates'][index + n:]
			after = after - ori
			A = self.GetAtom(AA, 'C')
			B = self.GetAtom(AA, 'CA')
			u = B - A
			lu = np.linalg.norm(u)
			u = u / lu
			current = self.Angle(AA, 'psi')
			zeroing = 0 - current
			RM = self.Rotation_Matrix(zeroing, u)
			after = np.matmul(after, RM)
			RM = self.Rotation_Matrix(theta, u)
			after = np.matmul(after, RM)
			after = after + ori
			combine = np.append(before, after, axis=0)
			self.data['Coordinates'] = combine
		if angle_type.upper() == 'OMEGA':
			ori = self.GetAtom(AA + 1, 'N')
			n = 0
			index = self.data['Amino Acids'][AA + 1][2][0]
			before = self.data['Coordinates'][:index + n]
			after = self.data['Coordinates'][index + n:]
			after = after - ori
			A = self.GetAtom(AA + 1, 'N')
			B = self.GetAtom(AA, 'C')
			u = B - A
			lu = np.linalg.norm(u)
			u = u / lu
			current = self.Angle(AA, 'omega')
			zeroing = 0 - current
			RM = self.Rotation_Matrix(zeroing, u)
			after = np.matmul(after, RM)
			RM = self.Rotation_Matrix(theta, u)
			after = np.matmul(after, RM)
			after = after + ori
			combine = np.append(before, after, axis=0)
			self.data['Coordinates'] = combine
		if angle_type.upper() == 'CHI':
			assert type(chi_type) is int, 'Incorrect Chi angle type'
			number_of_chis = len(self.AminoAcids[AminoAcid]['Chi Angle Atoms'])
			if not (number_of_chis >= chi_type):
				error = '{} amino acid at position {} has no Chi {} angle'\
				.format(AminoAcid, AA, chi_type)
				raise Exception(error)
			atoms = self.AminoAcids[AminoAcid]['Chi Angle Atoms'][chi_type-1]
			A = self.GetAtom(AA, atoms[1])
			B = self.GetAtom(AA, atoms[2])
			ori = B
			start = self.AminoAcids[AminoAcid]['Chi Angle Atoms'][chi_type-1][2]
			L = self.AminoAcids[AminoAcid]['Sidechain Atoms']
			i = [x[0] for x in L].index(start)
			Sindex = self.data['Amino Acids'][AA][3][i]
			Eindex = self.data['Amino Acids'][AA][3][-1]
			before = self.data['Coordinates'][:Sindex]
			side   = self.data['Coordinates'][Sindex:Eindex + 1]
			after  = self.data['Coordinates'][Eindex + 1 :]
			side = side - ori
			u = A - B
			lu = np.linalg.norm(u)
			u = u / lu
			current = self.Angle(AA, 'chi', chi_type)
			zeroing = 0 - current
			RM = self.Rotation_Matrix(zeroing, u)
			side = np.matmul(side, RM)
			RM = self.Rotation_Matrix(theta, u)
			side = np.matmul(side, RM)
			side = side + ori
			combine = np.append(before, side, axis=0)
			combine = np.append(combine, after, axis=0)
			self.data['Coordinates'] = combine
	def Rotation3Angle(self, AA1, atom1, AA2, atom2, AA3, atom3, theta):
		''' Change angle between three atoms '''
		atoms = self.data['Atoms']
		BB = self.data['Amino Acids'][AA2][2]
		atom2i = None
		for i in BB:
			if atoms[i][0] == atom2: atom2i = i
		if atom2i == None:
			raise Exception('Chosen atom not in backbone')
		before = self.data['Coordinates'][:atom2i]
		after  = self.data['Coordinates'][atom2i:]
		A = self.GetAtom(AA3, atom3) - self.GetAtom(AA1, atom1)
		B = self.GetAtom(AA3, atom3) - self.GetAtom(AA2, atom2)
		u = np.cross(B, A)
		lu = np.linalg.norm(u)
		u = u / lu
		ori = self.GetAtom(AA2, atom2)
		after = after - ori
		RM = self.Rotation_Matrix(theta, u)
		after = np.matmul(after, RM)
		after = after + ori
		combine = np.append(before, after, axis=0)
		self.data['Coordinates'] = combine
	def RigidMotion(self, AA, A, B, BB='Backbone middle'):
		''' Superimpose amino B into A '''
		n, e = 0, 0
		if BB == 'Backbone start': n = 2
		if BB == 'Backbone end': e = 1
		A1s = np.ones(len(A))
		B1s = np.ones(len(B))
		A = np.c_[A, A1s]
		B = np.c_[B, B1s]
		if AA == 'G':
			Aa, Ao, Ab, Ac = A[0], A[1], A[3], A[2]
			Ba, Bo, Bb, Bc = B[0], B[2+n], B[-1], B[-2-e]
			AL = np.array([Aa - Ao, Ab - Ao, Ac - Ao, Ao])
			BL = np.array([Ba - Bo, Bb - Bo, Bc - Bo, Bo])
			BL[1][2] = 1
			BL_= np.linalg.inv(BL)
		elif AA == 'P':
			Aa, Ao, Ab, Ac = A[0], A[1], A[4], A[2]
			Ba, Bo, Bb, Bc = B[0], B[1+n], B[3], B[-2-e]
			AL = np.array([Aa - Ao, Ab - Ao, Ac - Ao, Ao])
			BL = np.array([Ba - Bo, Bb - Bo, Bc - Bo, Bo])
			BL[1][2] = 1
			BL_= np.linalg.inv(BL)
		else:
			Aa, Ao, Ab, Ac = A[0], A[1], A[4], A[2]
			Ba, Bo, Bb, Bc = B[0], B[2+n], B[4+n], B[-2-e]
			AL = np.array([Aa - Ao, Ab - Ao, Ac - Ao, Ao])
			BL = np.array([Ba - Bo, Bb - Bo, Bc - Bo, Bo])
			BL_= np.linalg.inv(BL)
		M = np.matmul(BL_, AL)
		B = [np.matmul(i, M)[:3] for i in B]
		B = np.array(B)
		Aoxy = A[3][:3]
		B[-1] = Aoxy
		return(B)
	def Mutate(self, index, AA):
		''' Mutate an amino acid to a different amino acid '''
		sequence_old = self.FASTA()
		sequence = sequence_old[:index] + AA + sequence_old[index+1:]
		self.ReBuild(sequence)
	def Import(self, filename, chain='A'):
		''' Import a structure from a .pdb file '''
		ATOM, N, A, L, R, C, S, I, X, Y, Z, O, T, Q, E = \
		[], [], [], [], [], [], [], [], [], [], [], [], [], [], []
		with open(filename) as f:
			for line in f:
				line = line.strip()
				if line.split()[0] == 'ATOM' and line.split()[4] == chain:
					ATOM.append(line[:4].strip())
					N.append(int(line[6:11].strip()))
					A.append(line[12:16].strip())
					L.append(line[16].strip())
					R.append(line[17:20].strip())
					C.append(line[21].strip())
					S.append(int(line[22:26].strip()))
					I.append(line[26].strip())
					X.append(float(line[30:38].strip()))
					Y.append(float(line[38:46].strip()))
					Z.append(float(line[46:54].strip()))
					O.append(float(line[54:60].strip()))
					T.append(float(line[60:66].strip()))
					q = line[70:76].strip()
					if q != '':
						q = float(q)
						Q.append(q)
					else:
						q = 0.0
					Q.append(q)
					E.append(line[76:78].strip())
		N = [x-N[0] for x in N]
		S = [x-S[0] for x in S]
		ALL = [[a, r, c, s, x, y, z, o, t, q, e] \
		for a, r, c, s, x, y, z, o, t, q, e in \
		zip(A, R, C, S, X, Y, Z, O, T, Q, E)]
		Structure = defaultdict(list)
		for atom in ALL: Structure[atom[3]].append(atom)
		for repeat in range(2):
			for k, v in zip(Structure.keys(), Structure.values()):
				atom = None
				for i, entry in enumerate(v):
					if atom == entry[0]:
						Structure[k].pop(i)
					atom = entry[0]
		count = 0
		Atoms = {}
		Aminos = {}
		Coordinates = []
		backbone = ['N', 'CA', 'C', 'O', 'OXT']
		for k, v in zip(Structure.keys(), Structure.values()):
			BB = []
			SC = []
			for info in v:
				atom = info[0]
				amino = info[1]
				chain = info[2]
				residue = info[3]
				x = info[4]
				y = info[5]
				z = info[6]
				o = info[7]
				t = info[8]
				c = info[9]
				e = info[10]
				Coordinates.append([x, y, z])
				Atoms[count] = [atom, e, c, t]
				if atom in backbone:
					BB.append(count)
				else:
					SC.append(count)
				count += 1
			tricode = amino
			amino = \
			[k for k, v in self.AminoAcids.items() if v['Tricode'] == amino][0]
			Aminos[k] = [amino, chain, BB, SC, 'L', tricode]
		Coordinates = np.array(Coordinates)
		self.data['Coordinates'] = Coordinates
		self.data['Amino Acids'] = Aminos
		self.data['Atoms'] = Atoms
		sequence = self.FASTA()
		for i, aa in enumerate(list(sequence)):
			if i == 0:
				self.BondTree('Backbone start', aa)
			elif i == len(sequence)-1:
				self.BondTree('Backbone end', aa)
			else:
				self.BondTree('Backbone middle', aa)
		self.data['Mass'] = self.Mass()
		self.data['FASTA'] = self.FASTA()
		self.data['Size'] = self.Size()
		self.data['Rg'] = self.Rg()
	def ReBuild(self, sequence=None, D_AA=False):
		''' Fold a polypeptide using angles and bonds '''
		if sequence == None:
			sequence = self.data['FASTA']
		PHIs = []
		PSIs = []
		OMGs = []
		NCaC = []
		CaCN = []
		CNCa = [0]
		CHIs = {}
		bNCA = []
		bCAC = []
		bCN1 = []
		for i in range(len(sequence)):
			PHIs.append(self.Angle(i, 'PHI'))
			PSIs.append(self.Angle(i, 'PSI'))
			OMGs.append(self.Angle(i, 'OMEGA'))
			NCaC.append(self.Atom3Angle(i, 'N', i, 'CA', i, 'C'))
			if i != 0:
				CNCa.append(self.Atom3Angle(i-1, 'C', i, 'N', i, 'CA'))
			if i != len(sequence) -1:
				CaCN.append(self.Atom3Angle(i, 'CA', i, 'C', i+1, 'N'))
			chi = []
			for number in range(1, 21):
				try: chi.append(self.Angle(i, 'CHI', number))
				except: pass
			CHIs[i] = chi
			bNCA.append(self.Distance(i, 'N', i, 'CA'))
			bCAC.append(self.Distance(i, 'CA', i, 'C'))
			try: bCN1.append(self.Distance(i, 'C', i+1, 'N'))
			except: pass
		data ={
			'Energy':0,
			'Rg':0,
			'Mass':0,
			'Size':0,
			'FASTA':None,
			'Amino Acids':{},
			'Atoms':{},
			'Bonds':{},
			'Coordinates':np.array([[0, 0, 0]])}
		self.data = copy.deepcopy(data)
		self.Build(sequence)
		for i, (p, s, o, n, a, c, b1, b2, b3) in enumerate(zip(
		PHIs, PSIs, OMGs, NCaC, CaCN, CNCa, bNCA, bCAC, bCN1)):
			self.Rotate(i, p, 'PHI')
			self.Rotate(i, s, 'PSI')
			self.Adjust(i, 'N', i, 'CA', b1)
			self.Adjust(i, 'CA', i, 'C', b2)
			N = self.Atom3Angle(i, 'N', i, 'CA', i, 'C')
			self.Rotation3Angle(i, 'N', i, 'CA', i, 'C', N-n)
			if i != 0:
				C = self.Atom3Angle(i-1, 'C', i, 'N', i, 'CA')
				self.Rotation3Angle(i-1, 'C', i, 'N', i, 'CA', C-c)
			if i != len(sequence) -1:
				A = self.Atom3Angle(i, 'CA', i, 'C', i+1, 'N')
				self.Rotation3Angle(i, 'CA', i, 'C', i+1, 'N', A-a)
				self.Rotate(i, o, 'OMEGA')
				self.Adjust(i, 'C', i+1, 'N', b3)
		for i in range(len(sequence)):
			try:
				chi = CHIs[i]
				if chi == []: continue
				for ii, c in enumerate(chi):
					if sequence[i] == 'P': continue
					self.Rotate(i, c, 'CHI', ii+1)
			except: continue
		self.data['Mass'] = self.Mass()
		self.data['FASTA'] = self.FASTA()
		self.data['Size'] = self.Size()
		self.data['Rg'] = self.Rg()
		if D_AA:
			self.data['Coordinates'] = self.data['Coordinates'] * [1, 1, -1]
			for i in range(len(sequence)):
				Daa = self.data['Amino Acids'][i][0].lower()
				tri = 'D' + self.data['Amino Acids'][i][-1][1:]
				self.data['Amino Acids'][i][0] = Daa
				self.data['Amino Acids'][i][-1] = tri
