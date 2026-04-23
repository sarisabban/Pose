import numpy as np

def harmonic_bond_potential(r, r0, Kb):
	'''
	Calculates the Harmonic Bond Stretching potential energy between two atoms
	Arguments:
	----------
		r:  Current bond length between two atoms
		r0: The bond length between the two atoms at equilibrium
		Kb: The force constant
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	return Kb * (r - r0)**2

def morse_potential(r, r0, De, a):
	'''
	Calculates the Morse Bond potential energy between two atoms
	Arguments:
	----------
		r:  Current bond length between two atoms
		r0: The bond length between the two atoms at equilibrium
		De: Well depth (strength)
		a:  Well width (smoothness)
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	return De * (1 - np.exp(-a * (r - r0)))**2

def harmonic_angle_potential(theta, theta0, K_theta):
	'''
	Calculates the Harmonic Angle Bending potential energy between two atoms
	Arguments:
	----------
		theta:   Current bond angle between two atoms (rad)
		theta0:  The bond angle between two atoms at equilibrium (rad)
		K_theta: Force constant
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	return K_theta * (theta - theta0)**2

def lennard_jones(r, sigma, epsilon, variant='12-6'):
	'''
	Calculates the Lennard-Jones potential energy between two atoms
	Arguments:
	----------
		r:       Current bond length between two atoms
		sigma:   Distance where energy is zero (or energy minima in 9-6 variant)
		epsilon: Energy well depth (strength)
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	if variant == '12-6':
		ratio_12 = (sigma / r)**12
		ratio_6  = (sigma / r)**6
		return 4 * epsilon * (ratio_12 - ratio_6)
	elif variant == '9-6':
		ratio_9 = 2 * (sigma / r)**9
		ratio_6 = 3 * (sigma / r)**6
		return epsilon * (ratio_9 - ratio_6)
	else:
		raise Exception('Incorrect variant type')

def electrostatic_potential(r, q1, q2, epsilon_r=1.0):
	'''
	Calculates the Electrostatic potential energy between two atoms
	Arguments:
	----------
		r:         Distance between two atom centers
		q1:        Partial charge of atom 1
		q2:        Partial charge of atom 2
		epsilon_r: Dielectric constant
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	return (332.06 * q1 * q2) / (epsilon_r * r)

def dihedral_potential(phi, phi0, k_phi, n):
	'''
	Calculates the Dihedral potential energy between two atoms
	Arguments:
	----------
		phi:   Current dihedral angle
		phi0:  Phi at equillibrium
		k_phi: Force constant
		n:     Multiplicity
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	return k_phi * (1 + np.cos(n * phi - phi0))

def improper_dihedral_potential(psi, psi0, k_imp):
	'''
	Calculates the Improper Dihedral potential energy between two atoms
	Arguments:
	----------
		psi:   Current dihedral angle
		psi0:  Psi at equillibrium
		k_imp: Force constant
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	return k_imp * (psi - psi0)**2

def scaling_potential(r14, sigma, epsilon, q1, q2, f_lj=0.5, f_elec=0.833):
	'''
	Calculates the Scaling potential energy between two distant atoms
	Arguments:
	----------
		r14:     Distance between atom 1 and atom 4 (3 atoms between them)
		sigma:   Distance where potential energy is zero
		epsilon: Energy well
		q1:      Charge of atom 1
		q2:      Charge of atom 4
		f_lj:    Scaling of the Lenndard-Jones potential
		e_elec:  Scaling of the Electrostatic potential
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	e_lj = lennard_jones(r14, sigma, epsilon)
	e_elec = coulomb_potential(r14, q1, q2)
	return (e_lj * f_lj) + (e_elec * f_elec)

def polarisation_potential(alpha, E):
	'''
	Calculates the Polarisation potential energy between two atoms
	Arguments:
	----------
		alpha: Polarisability
		E:     Electric field
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	return 0.5 * alpha * E**2

def urey_bradley_potential(r13, s0, k_ub):
	'''
	Calculates the Urey-Bradley potential energy between two atoms
	Arguments:
	----------
		r13:   Distance between atom 1 and atom 3 (2 atoms between them)
		s0:    Equillibrium distamce
		k_ub:  Force constant
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	return k_ub * (r13 - s0)**2








def Lennard_Jones(r, sigma, epsilon, mask):
	ratio_12 = (sigma / r)**12
	ratio_6  = (sigma / r)**6
	lj = 4.0 * epsilon * (ratio_12 - ratio_6)
	return float(np.sum(lj[mask]))






def Energy(pose, alg='lennard_jones'):
	''' Main energy function where you choose wich force field to use '''
	if alg.upper() == 'LENNARD_JONES':
		_LJ = {
			'H':  (2.886, 0.044), 'C':  (3.851, 0.105),
			'N':  (3.660, 0.069), 'O':  (3.500, 0.060),
			'F':  (3.364, 0.050), 'P':  (4.147, 0.305),
			'S':  (4.035, 0.274), 'Cl': (3.947, 0.227),
			'Br': (4.189, 0.251), 'I':  (4.500, 0.339),
			'Se': (4.205, 0.291)}
		atoms = pose.data['Atoms']
		n = len(atoms)
		elems = [atoms[i][1] for i in range(n)]
		sig = np.array([_LJ.get(e,_LJ['C'])[0] for e in elems],dtype=np.float64)
		eps = np.array([_LJ.get(e,_LJ['C'])[1] for e in elems],dtype=np.float64)
		sigma = 0.5 * (sig[:, None] + sig[None, :])
		epsilon = np.sqrt(eps[:, None] * eps[None, :])
		nbrs = [set() for _ in range(n)]
		for k, vs in pose.data['Bonds'].items():
			i = int(k)
			for j in vs:
				nbrs[i].add(int(j))
				nbrs[int(j)].add(i)
		excl = np.eye(n, dtype=bool)
		for i in range(n):
			for j in nbrs[i]:
				excl[i, j] = True
				for kk in nbrs[j]:
					excl[i, kk] = True
		mask = (~excl) & np.triu(np.ones_like(excl), k=1).astype(bool)
		coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
		r = coords[:, None, :] - coords[None, :, :]
		r = np.linalg.norm(r, axis=-1)
		np.fill_diagonal(r, 1.0)
		return Lennard_Jones(r, sigma, epsilon, mask)
	else:
		raise Exception('Algorithm no supported')
