import numpy as np

def harmonic_bond(r, r0, Kb):
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
	Calculates Morse Bond potential energy between two atoms
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

def harmonic_angle(theta, theta0, K_theta):
	'''
	Calculates Harmonic Angle Bending potential energy between two atoms
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



def dihedral_potential(phi, phi0, k_phi, n):
    """
    Calculates Torsional (Dihedral) Potential energy.
    phi:   current dihedral angle (radians)
    phi0:  phase offset/target angle (radians)
    k_phi: force constant (barrier height)
    n:     multiplicity (number of peaks in 360 degrees)
    """
    return k_phi * (1 + np.cos(n * phi - phi0))


def improper_dihedral_potential(psi, psi0, k_imp):
    """
    Calculates Improper Dihedral energy.
    psi: current out-of-plane angle, psi0: target angle, k_imp: force constant
    """
    return k_imp * (psi - psi0)**2


def electrostatic_potential(r, q1, q2, epsilon_r=1.0):
    """
    Calculates Coulombic Potential energy.
    r: distance between atoms
    q1, q2: partial charges of the atoms
    epsilon_r: dielectric constant (default 1.0 for vacuum/explicit solvent)
    """
    # Coulomb constant (k_e) in kcal*A/(mol*e^2) is approx 332.06
    ke = 332.06 
    return (ke * q1 * q2) / (epsilon_r * r)




def lennard_jones(r, sigma, epsilon, variant='12-6'):
	'''
	Calculates Lennard-Jones potential energy between two atoms
	Arguments:
	----------
		r:       Current bond length between two atoms
		sigma:   Distance where energy is zero (or energy minima in 9-6 variant)
		epsilon: Well depth (strength)
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



import numpy as np

def scaled_14_interaction(r, sigma, epsilon, q1, q2, f_lj=0.5, f_elec=0.833):
    """
    Applies scaling factors to non-bonded interactions between 1-4 atom pairs.
    f_lj: scaling for Lennard-Jones (default 0.5 for AMBER)
    f_elec: scaling for Electrostatics (default 0.833 for AMBER)
    """
    # Reuse your existing LJ and Coulomb functions
    e_lj = lennard_jones(r, sigma, epsilon)
    e_elec = coulomb_potential(r, q1, q2)
    
    return (e_lj * f_lj) + (e_elec * f_elec)

def urey_bradley(r13, s0, k_ub):
    """
    Calculates Urey-Bradley energy (Harmonic 1-3 interaction).
    r13: distance between the 1st and 3rd atoms in an angle
    s0: equilibrium distance, k_ub: force constant
    """
    return k_ub * (r13 - s0)**2

def drude_self_energy(d, k_drude):
    """
    Calculates the harmonic 'self-energy' of a Drude oscillator.
    d: distance between the Drude particle and its parent atom core
    k_drude: the spring constant (polarizability constant)
    """
    return 0.5 * k_drude * d**2










def Lennard_Jones(pose, terms):
	'''
	Total Lennard-Jones 12-6 potential energy
	Arguments:
	----------
		pose:  Pose()
		terms: (sigma [Angstrom], epsilon [kcal/mol], mask [bool])
	Returns:
	--------
		float: potential energy in kcal/mol
	'''
	sigma, epsilon, mask = terms
	coords = np.asarray(pose.data['Coordinates'], dtype=np.float64)
	diff = coords[:, None, :] - coords[None, :, :]
	r2 = np.einsum('ijk,ijk->ij', diff, diff)
	np.fill_diagonal(r2, 1.0)
	sr2 = (sigma * sigma) / r2
	sr6 = sr2 * sr2 * sr2
	sr12 = sr6 * sr6
	lj = 4.0 * epsilon * (sr12 - sr6)
	return float(np.sum(lj[mask]))




def Energy(pose, alg='lennard_jones', terms=None):
	''' Main energy function where you choose wich force field to use '''
	if alg.upper() == 'LENNARD_JONES':
		_LJ = {
			'H':  (2.886, 0.044), 'C':  (3.851, 0.105),
			'N':  (3.660, 0.069), 'O':  (3.500, 0.060),
			'F':  (3.364, 0.050), 'P':  (4.147, 0.305),
			'S':  (4.035, 0.274), 'Cl': (3.947, 0.227),
			'Br': (4.189, 0.251), 'I':  (4.500, 0.339),
			'Se': (4.205, 0.291)}
		if terms is None:
			atoms = pose.data['Atoms']
			n = len(atoms)
			elems = [atoms[i][1] for i in range(n)]
			sig = np.array([_LJ.get(e, _LJ['C'])[0] for e in elems],
				dtype=np.float64)
			eps = np.array([_LJ.get(e, _LJ['C'])[1] for e in elems],
				dtype=np.float64)
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
			terms = (sigma, epsilon, mask)
		return Lennard_Jones(pose, terms)
	else:
		raise Exception('Algorithm no supported')
