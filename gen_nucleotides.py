#!/usr/bin/env python3
"""
Generate NucleotidesDB.json.
Strategy:
  1. Build deoxyribose/ribose ring with Z-matrix (place()).
  2. Attach backbone atoms using B-DNA / A-RNA torsion angles.
  3. Embed planar base in 3D using precomputed 2D ring geometry + chi angle.
  4. Transform all atoms to canonical helix frame (Z = helix axis).
  5. Write JSON.

References:
  Arnott & Hukins 1972; Saenger 1984 Tables 6.2 / 7.2.
"""
import numpy as np
import json, math, copy

# Arnott fiber-diffraction backbone torsion angles (IUPAC convention, degrees)
# Arnott & Hukins 1972; Saenger 1984 Tables 6.2 / 7.2
# NOTE: place() convention: dihed(A,B,C,D) = -phi_d, so pass NEGATED IUPAC values.
# IUPAC B-DNA: alpha=-47, beta=175, gamma=48, delta=143, eps=-168, zeta=-95
# IUPAC A-RNA: alpha=-68, beta=177, gamma=54, delta=83,  eps=-153, zeta=-71
ARNOTT_DNA = dict(alpha=47, beta=91, gamma=28, delta=-135, eps=206, zeta=76)
ARNOTT_RNA = dict(alpha=71, beta=-180, gamma=-51, delta=-92,  eps=148, zeta=85)

# ── Utility ──────────────────────────────────────────────────────────────────

def place(r, theta_d, phi_d, A, B, C):
    """Place D: |D-C|=r, angle(B-C-D)=theta_d, dihedral(A-B-C-D)=phi_d."""
    A,B,C = map(lambda v: np.array(v,float), [A,B,C])
    th, ph = math.radians(theta_d), math.radians(phi_d)
    BC = C - B;  bc = BC / np.linalg.norm(BC)
    AB = A - B
    n  = AB - np.dot(AB, bc)*bc
    nl = np.linalg.norm(n)
    if nl < 1e-8:
        tmp = np.array([1,0,0]) if abs(bc[0])<0.9 else np.array([0,1,0])
        n = np.cross(bc, tmp)
    n = n / np.linalg.norm(n)
    m = np.cross(bc, n)
    return C + r*(-math.cos(th)*bc
                  + math.sin(th)*math.cos(ph)*n
                  + math.sin(th)*math.sin(ph)*m)

def dihed(A,B,C,D):
    b1=np.asarray(B,float)-A; b2=np.asarray(C,float)-B; b3=np.asarray(D,float)-C
    n1=np.cross(b1,b2); n2=np.cross(b2,b3)
    m1=np.cross(n1, b2/np.linalg.norm(b2))
    return math.degrees(math.atan2(np.dot(m1,n2), np.dot(n1,n2)))

def dist(A,B): return float(np.linalg.norm(np.asarray(A,float)-B))

def rot_axis(axis, angle_deg):
    """Rodrigues rotation matrix: rotate angle_deg around unit axis."""
    u  = np.asarray(axis, float); u /= np.linalg.norm(u)
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    ux,uy,uz = u
    return np.array([
        [c+ux*ux*(1-c),    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
        [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c),    uy*uz*(1-c)-ux*s],
        [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)   ]])

# ── Sugar ring ────────────────────────────────────────────────────────────────

def build_sugar(form='DNA'):
    """
    Build the (deoxy)ribose ring.
    B-DNA: C2'-endo (Altona & Sundaralingam 1972)
    A-RNA: C3'-endo
    Returns dict atom_name -> np.array([x,y,z]).
    """
    if form == 'DNA':  # C2'-endo
        nu0,nu1,nu2,nu3,nu4 = -25.3, 38.7, -38.2, 24.2, 2.3
    else:              # C3'-endo
        nu0,nu1,nu2,nu3,nu4 =   7.2,-25.3,  37.0,-36.7,18.8

    # Bonds
    bO4C1=1.440; bC1C2=1.530; bC2C3=1.520; bC3C4=1.520; bC4O4=1.440
    # Angles
    aC4O4C1=110.3; aO4C1C2=104.9; aC1C2C3=102.4; aC2C3C4=102.0; aC3C4O4=103.2

    # Place ring: O4' at origin, C4' along +X, C3' in XY plane for bootstrapping
    O4 = np.array([0.0, 0.0, 0.0])
    C4 = np.array([bC4O4, 0.0, 0.0])
    # Bootstrap C3' in XY plane so we can use nu4 to place C1'
    ang_O4C4C3 = 180.0 - aC3C4O4   # supplement for XY placement
    C3_boot = np.array([
        bC4O4 + bC3C4*math.cos(math.radians(180.0 - aC3C4O4)),
       -bC3C4*math.sin(math.radians(180.0 - aC3C4O4)),
        0.0])
    # C1' via nu4 = C3'-C4'-O4'-C1'
    C1 = place(bO4C1, aC4O4C1, nu4, C3_boot, C4, O4)
    # C2' via nu0 = C4'-O4'-C1'-C2'
    C2 = place(bC1C2, aO4C1C2, nu0, C4, O4, C1)
    # C3' placed FROM C4' to guarantee ring closure (C4'-C3'=1.520 exactly).
    # Torsion: Angle_pose(C1',O4',C4',C3') = nu4  (same as nu4 definition
    # C3'-C4'-O4'-C1' by reversal).  Bond angle C3'-C4'-O4' = aC3C4O4.
    C3 = place(bC3C4, aC3C4O4, nu4, C1, O4, C4)
    return {'O4p':O4, 'C4p':C4, 'C1p':C1, 'C2p':C2, 'C3p':C3}

# ── Backbone ──────────────────────────────────────────────────────────────────

def add_backbone(atoms, form='DNA',
                 alpha=-47, beta=175, gamma=48, delta=143, eps=-168, zeta=-95):
    """
    Add backbone atoms to the sugar dict (in-place).
    All angle parameters use IUPAC torsion angle convention (Angle_pose).
    Arnott B-DNA defaults: alpha=-47,beta=175,gamma=48,delta=143,eps=-168,zeta=-95
    Arnott A-RNA defaults: alpha=-68,beta=177,gamma=54,delta=83, eps=-153,zeta=-71
    """
    C1=atoms['C1p']; C2=atoms['C2p']; C3=atoms['C3p']
    C4=atoms['C4p']; O4=atoms['O4p']

    # C5' bonded to C4' (C=C4'); angle C3'-C4'-C5' = 114.0
    C5 = place(1.510, 114.0,  120.0, O4,  C3, C4)

    # O5' bonded to C5'; gamma = Angle_pose(C3',C4',C5',O5') = gamma
    O5 = place(1.440, 111.0,  gamma,  C3,  C4, C5)

    # P bonded to O5'; IUPAC beta = Angle_pose(P,O5',C5',C4')
    # By reversal: Angle_pose(P,O5',C5',C4') = Angle_pose(C4',C5',O5',P)
    # place(A,B,C) sets Angle_pose(C4,C5,O5,P)=phi → IUPAC beta = phi
    P  = place(1.593, 119.7,  beta,   C4,  C5, O5)

    # O3' bonded to C3' (C=C3'); delta = Angle_pose(C5',C4',C3',O3') = +delta
    O3 = place(1.430, 110.0,  delta,  C5,  C4, C3)

    # Phosphate oxygens OP1, OP2 bonded to P (C=P)
    OP1 = place(1.480, 108.0,  60.0, C5, O5, P)
    OP2 = place(1.480, 108.0, -60.0, C5, O5, P)

    # Hydrogens: each bonded to its carbon (C = that carbon)
    H5p  = place(1.090, 109.5,  120.0, C4, O5, C5)   # H5'  at C5'
    H5pp = place(1.090, 109.5, -120.0, C4, O5, C5)   # H5'' at C5'
    H4p  = place(1.090, 109.5,  120.0, O5, C3, C4)   # H4'  at C4'
    H3p  = place(1.090, 109.5,  120.0, C4, C2, C3)   # H3'  at C3'
    H2p  = place(1.090, 109.5,  120.0, C3, C1, C2)   # H2'  at C2'
    H1p  = place(1.090, 109.5,  120.0, C2, O4, C1)   # H1'  at C1'

    atoms.update({'P':P,'OP1':OP1,'OP2':OP2,"O5p":O5,'C5p':C5,
                  "H5p":H5p,"H5pp":H5pp,"H4p":H4p,"H3p":H3p,
                  "H2p":H2p,"H1p":H1p,"O3p":O3})

    if form == 'DNA':
        H2pp = place(1.090, 109.5, -120.0, C3, C1, C2)  # H2'' at C2'
        atoms["H2pp"] = H2pp
    else:
        # RNA: O2' and HO2' instead of H2''
        O2  = place(1.420, 110.0, -120.0, C3,  C1, C2)  # O2' at C2'
        HO2 = place(0.960, 109.5,  180.0, C3,  C2, O2)  # HO2' at O2'
        atoms["O2p"]  = O2
        atoms["HO2p"] = HO2

# ── Base embedding via 2D ring geometry ──────────────────────────────────────

# Pre-computed 2D ring coords (in Angstrom).
# Convention: glycosidic N (N9 for purines, N1 for pyrimidines) at origin;
# chi-related atom (C4 for purines, C2 for pyrimidines) along +X.
# All atoms in XY plane (ring plane).

ADENINE_2D = {  # N9 at (0,0), C4 along +X  (from Saenger 1984, Table 6.2)
    'N9' :( 0.000,  0.000),
    'C8' :(-0.401,  1.310),
    'H8' :(-1.374,  1.634),
    'N7' :( 0.597,  2.166),
    'C5' :( 1.671,  1.583),
    'C4' :( 1.318,  0.403),
    'N3' :( 2.203, -0.561),
    'C2' :( 3.394,  0.020),
    'H2' :( 4.369,  0.278),
    'N1' :( 3.898,  1.186),
    'C6' :( 2.939,  2.086),
    'N6' :( 3.107,  3.374),
    'H61':( 2.295,  4.001),
    'H62':( 4.039,  3.779),
}

GUANINE_2D = {  # same frame as adenine
    'N9' :( 0.000,  0.000),
    'C8' :(-0.401,  1.310),
    'H8' :(-1.374,  1.634),
    'N7' :( 0.597,  2.166),
    'C5' :( 1.671,  1.583),
    'C4' :( 1.318,  0.403),
    'N3' :( 2.203, -0.561),
    'C2' :( 3.394,  0.020),
    'N2' :( 3.983, -1.150),
    'H21':( 3.444, -2.000),
    'H22':( 4.970, -1.100),
    'N1' :( 3.898,  1.186),
    'H1' :( 4.830,  1.600),
    'C6' :( 2.939,  2.086),
    'O6' :( 3.192,  3.297),
}

THYMINE_2D = {  # N1 at (0,0), C2 along +X
    'N1' :( 0.000,  0.000),
    'C2' :( 1.371,  0.000),
    'O2' :( 2.012,  1.087),
    'N3' :( 2.077, -1.153),
    'H3' :( 1.795, -2.127),
    'C4' :( 3.451, -1.024),
    'O4t':( 4.169, -1.967),
    'C5' :( 3.880,  0.401),
    'C7' :( 5.250,  0.637),
    'H71':( 5.602,  1.659),
    'H72':( 5.857, -0.144),
    'H73':( 5.418, -0.003),
    'C6' :( 2.841,  1.252),
    'H6' :( 2.925,  2.326),
}

CYTOSINE_2D = {  # N1 at (0,0), C2 along +X
    'N1' :( 0.000,  0.000),
    'C2' :( 1.395,  0.000),
    'O2' :( 2.043,  1.100),
    'N3' :( 2.051, -1.148),
    'C4' :( 3.419, -0.999),
    'N4' :( 4.183, -1.979),
    'H41':( 3.757, -2.894),
    'H42':( 5.182, -1.828),
    'C5' :( 3.763,  0.386),
    'H5' :( 4.794,  0.645),
    'C6' :( 2.697,  1.238),
    'H6' :( 2.745,  2.314),
}

URACIL_2D = {   # N1 at (0,0), C2 along +X  (same as thymine minus methyl)
    'N1' :( 0.000,  0.000),
    'C2' :( 1.383,  0.000),
    'O2' :( 2.027,  1.083),
    'N3' :( 2.039, -1.142),
    'H3' :( 1.754, -2.114),
    'C4' :( 3.413, -1.011),
    'O4t':( 4.133, -1.960),
    'C5' :( 3.758,  0.380),
    'H5' :( 4.789,  0.630),
    'C6' :( 2.691,  1.232),
    'H6' :( 2.741,  2.306),
}

def embed_base(atoms, base_2d, glc_N, chi_atom, chi_deg, purine=True):
    """
    Embed a planar base ring (given in 2D) into 3D.

    glc_N    : atom key of glycosidic N in atoms dict (N9 or N1)
    chi_atom : key of the chi-defining atom in base_2d (C4 for purines,
               C2 for pyrimidines)
    chi_deg  : target chi dihedral (O4'-C1'-glcN-chi_atom)

    The 2D ring is:
      - glycosidic N at (0,0)
      - chi_atom along +X
    place() gives Angle_pose(A,B,C,D) = phi_d (pose.py convention).
    Pass phi_d = desired Angle_pose torsion directly.
    """
    C1 = atoms['C1p']; O4 = atoms['O4p']; C2s = atoms['C2p']
    bl_glc = 1.462 if purine else 1.480  # C1'-N9 or C1'-N1
    ang_glc = 108.2                       # O4'-C1'-N angle

    # Place glycosidic N with fixed phi_N; any value works because the
    # chi_atom placement controls chi independently.
    N3d = place(bl_glc, ang_glc, 180.0, C2s, O4, C1)

    # Place chi_atom: chi = Angle_pose(O4', C1', N, chi_atom) = chi_deg
    d2d = math.hypot(*base_2d[chi_atom])          # N--chi_atom distance
    ang_N_chi = 126.7 if purine else 118.0         # C1'-N-chi_atom angle
    CA3d = place(d2d, ang_N_chi,  chi_deg, O4, C1, N3d)

    # ── Build 3D frame from N and chi_atom ───────────────────────────────
    X = CA3d - N3d;  X = X / np.linalg.norm(X)   # ring local +X (N→chi_atom)
    u = N3d - C1;    u = u / np.linalg.norm(u)    # C1'→N direction
    Z = np.cross(X, u)
    zl = np.linalg.norm(Z)
    if zl < 1e-8:
        Z = np.array([0,0,1.0])
    else:
        Z = Z / zl
    Y = np.cross(Z, X)

    # ── Map 2D ring atoms into 3D ─────────────────────────────────────────
    base_3d = {}
    for name, (x2, y2) in base_2d.items():
        base_3d[name] = N3d + x2*X + y2*Y
    # Override glc_N with the precisely placed value (2D origin = N3d)
    base_3d[glc_N] = N3d

    atoms.update(base_3d)

# ── Helix-frame transformation ────────────────────────────────────────────────

def extend_chain(prev_atoms, form='DNA',
                 alpha=-47, beta=175, gamma=48,
                 delta=None, eps=-168, zeta=-95):
    """
    Place the next nucleotide's backbone+sugar atoms from prev_atoms
    (needs keys C4p, C3p, O3p).  Returns dict of atom_key -> np.array.
    Does NOT embed bases; caller adds base atoms separately.
    Uses Angle_pose convention (same as place() with positive target).
    """
    if delta is None:
        delta = 143 if form == 'DNA' else 83
    C4_p = prev_atoms['C4p']
    C3_p = prev_atoms['C3p']
    O3_p = prev_atoms['O3p']

    P   = place(1.593, 119.0, eps,    C4_p, C3_p, O3_p)  # C3'-O3'-P=119°
    OP1 = place(1.480, 108.0,  60.0,  C3_p, O3_p, P)    # O3'-P-OP1=108°
    OP2 = place(1.480, 108.0, -60.0,  C3_p, O3_p, P)    # O3'-P-OP2=108°
    O5  = place(1.593, 104.0, zeta,   C3_p, O3_p, P)    # P-O5'=1.593Å, O3'-P-O5'=104°
    C5  = place(1.440, 120.0, alpha,  O3_p, P,    O5)   # O5'-C5'=1.440Å, P-O5'-C5'=120°
    C4  = place(1.524, 114.0, beta,   P,    O5,   C5)
    C3  = place(1.520, 113.0, gamma,  O5,   C5,   C4)

    if form == 'DNA':
        nu0, nu3, nu4 = -25.3, 24.2, 2.3
    else:
        nu0, nu3, nu4 = 7.2, -36.7, 18.8

    # O4': place from C4', angle C3'-C4'-O4'=103.2°.
    # Torsion Angle_pose(C5,C3,C4,O4) from reference build_sugar
    O4  = place(1.440, 103.2, -120.0, C5, C3, C4)
    # C1': nu4 = C3'-C4'-O4'-C1' = Angle_pose(C3,C4,O4,C1)
    C1  = place(1.440, 110.3,  nu4,   C3, C4, O4)
    # C2': place FROM C3' to guarantee C2'-C3'=1.520 (ring closure).
    # nu3 = C2'-C3'-C4'-O4' = Angle_pose(C2,C3,C4,O4) = Angle_pose(O4,C4,C3,C2) by reversal
    C2  = place(1.520, 102.0,  nu3,   O4, C4, C3)
    O3  = place(1.430, 110.0,  delta, C5, C4, C3)

    H5p  = place(1.090, 109.5,  120.0, C4, O5, C5)
    H5pp = place(1.090, 109.5, -120.0, C4, O5, C5)
    H4p  = place(1.090, 109.5,  120.0, O5, C3, C4)
    H3p  = place(1.090, 109.5,  120.0, C4, C2, C3)
    H2p  = place(1.090, 109.5,  120.0, C3, C1, C2)
    H1p  = place(1.090, 109.5,  120.0, C2, O4, C1)

    a = {'P':P,'OP1':OP1,'OP2':OP2,
         'O5p':O5,'C5p':C5,'H5p':H5p,'H5pp':H5pp,
         'C4p':C4,'H4p':H4p,'O4p':O4,
         'C3p':C3,'H3p':H3p,
         'C2p':C2,'H2p':H2p,
         'C1p':C1,'H1p':H1p,'O3p':O3}

    if form == 'DNA':
        a['H2pp'] = place(1.090, 109.5, -120.0, C3, C1, C2)
    else:
        O2  = place(1.420, 110.0, -120.0, C3, C1, C2)
        HO2 = place(0.960, 109.5,  180.0, C3, C2, O2)
        a['O2p'] = O2; a['HO2p'] = HO2
    return a


def compute_helix_frame(form='DNA'):
    """
    Build two consecutive nucleotides via Z-matrix, fit the helical step
    transform (Kabsch SVD), extract the helix axis and centre, then return
    R_total (3x3) and c (3,) such that

        y_helix = R_total @ (x_local - c)

    maps local coords to a helix frame where the step transform is
    exactly  Rz(twist) @ y + (0,0,rise).

    An extra Rz(theta0) is applied so C1' has x_helix ≈ +5.25 Å, which
    makes Rflip = diag(-1,1,-1) give the correct ~10.5 Å C1'-C1' distance
    for the antiparallel complementary strand.
    """
    ang   = ARNOTT_DNA if form == 'DNA' else ARNOTT_RNA
    chi   = 115 if form == 'DNA' else 160
    twist = 36.0  if form == 'DNA' else 32.7

    base = 'DA' if form == 'DNA' else 'A'
    nt0  = build_nucleotide(base, form, chi=chi, **ang)
    nt1  = extend_chain(nt0, form, **ang)

    # Corresponding backbone heavy atoms for Kabsch SVD
    keys = ['P','O5p','C5p','C4p','O4p','C3p','C2p','C1p','O3p']
    src  = np.array([nt0[k] for k in keys])
    tgt  = np.array([nt1[k] for k in keys])

    sc, tc = src.mean(0), tgt.mean(0)
    H = (src - sc).T @ (tgt - tc)
    U, _, Vt = np.linalg.svd(H)
    R_step = Vt.T @ U.T
    if np.linalg.det(R_step) < 0:
        Vt[-1] *= -1
        R_step = Vt.T @ U.T
    t_step = tc - R_step @ sc

    # Helix axis direction: eigenvector of R_step with eigenvalue 1
    eigvals, eigvecs = np.linalg.eig(R_step)
    real_idx = np.argmin(np.abs(np.imag(eigvals)))
    u = np.real(eigvecs[:, real_idx]).copy()
    u /= np.linalg.norm(u)
    if np.dot(u, t_step) < 0:
        u = -u   # point in direction of chain growth

    # Helix axis centre (least-squares; component along u is arbitrary)
    t_perp = t_step - np.dot(t_step, u) * u
    A_mat  = np.eye(3) - R_step
    c, _, _, _ = np.linalg.lstsq(A_mat, t_perp, rcond=None)
    c -= np.dot(c, u) * u   # project to plane perpendicular to u

    # Rotation to align u → Z
    Z = np.array([0., 0., 1.])
    v = np.cross(u, Z)
    s = np.linalg.norm(v)
    if s < 1e-8:
        R_align = np.eye(3) if np.dot(u, Z) > 0 else -np.eye(3)
    else:
        v /= s
        cos_a, sin_a = float(np.dot(u, Z)), float(s)
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R_align = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

    # Extra Rz(theta0) so C1'_x ≈ +5.25 Å  (makes Rflip give C1'-C1'≈10.5Å)
    C1p_local = nt0['C1p']
    C1p_h = R_align @ (C1p_local - c)
    r_C1  = math.hypot(C1p_h[0], C1p_h[1])
    target_x = 5.25
    if r_C1 > target_x:
        target_angle  = math.acos(target_x / r_C1)
        current_angle = math.atan2(C1p_h[1], C1p_h[0])
        theta0 = target_angle - current_angle
        # pick sign so that y_C1 > 0 after rotation
        cos0, sin0 = math.cos(theta0), math.sin(theta0)
        y_new = -C1p_h[0]*sin0 + C1p_h[1]*cos0
        if y_new < 0:
            theta0 = -target_angle - current_angle
            cos0, sin0 = math.cos(theta0), math.sin(theta0)
        Rz0 = np.array([[cos0, -sin0, 0.],
                        [sin0,  cos0, 0.],
                        [0.,    0.,   1.]])
        R_total = Rz0 @ R_align
    else:
        R_total = R_align

    return R_total, c

# ── Full nucleotide builder ───────────────────────────────────────────────────

def build_nucleotide(base_name, form, chi=-115.0,
                     alpha=-47, beta=175, gamma=48,
                     delta=None, eps=-168, zeta=-95):
    if delta is None:
        delta = 143 if form == 'DNA' else 83
    atoms = build_sugar(form)
    add_backbone(atoms, form, alpha=alpha, beta=beta, gamma=gamma,
                 delta=delta, eps=eps, zeta=zeta)
    b = base_name.upper()
    if b in ('A','DA'):
        embed_base(atoms, ADENINE_2D, 'N9', 'C4', chi, purine=True)
    elif b in ('G','DG'):
        embed_base(atoms, GUANINE_2D, 'N9', 'C4', chi, purine=True)
    elif b in ('T','DT'):
        embed_base(atoms, THYMINE_2D, 'N1', 'C2', chi, purine=False)
    elif b in ('C','DC'):
        embed_base(atoms, CYTOSINE_2D, 'N1', 'C2', chi, purine=False)
    elif b in ('U',):
        embed_base(atoms, URACIL_2D, 'N1', 'C2', chi, purine=False)
    return atoms

# ── Canonical atom order and JSON assembly ────────────────────────────────────

# Backbone order (determines index in Vectors list)
BB_ORDER_DNA = ['P','OP1','OP2',"O5p",'C5p',"H5p","H5pp",
                'C4p',"H4p",'O4p','C3p',"H3p",
                'C2p',"H2p","H2pp",'C1p',"H1p","O3p"]
BB_ORDER_RNA = ['P','OP1','OP2',"O5p",'C5p',"H5p","H5pp",
                'C4p',"H4p",'O4p','C3p',"H3p",
                'C2p',"H2p",'O2p','HO2p','C1p',"H1p","O3p"]

BB_ATOMS_DNA = [
    ['P',   'P',0,0],['OP1','O',0,0],['OP2','O',0,0],
    ["O5'", 'O',0,0],["C5'",'C',0,0],["H5'", 'H',0,0],["H5''",'H',0,0],
    ["C4'", 'C',0,0],["H4'", 'H',0,0],["O4'",'O',0,0],
    ["C3'", 'C',0,0],["H3'", 'H',0,0],
    ["C2'", 'C',0,0],["H2'", 'H',0,0],["H2''",'H',0,0],
    ["C1'", 'C',0,0],["H1'", 'H',0,0],["O3'",'O',0,0]]

BB_ATOMS_RNA = [
    ['P',   'P',0,0],['OP1','O',0,0],['OP2','O',0,0],
    ["O5'", 'O',0,0],["C5'",'C',0,0],["H5'", 'H',0,0],["H5''",'H',0,0],
    ["C4'", 'C',0,0],["H4'", 'H',0,0],["O4'",'O',0,0],
    ["C3'", 'C',0,0],["H3'", 'H',0,0],
    ["C2'", 'C',0,0],["H2'", 'H',0,0],["O2'", 'O',0,0],["HO2'",'H',0,0],
    ["C1'", 'C',0,0],["H1'", 'H',0,0],["O3'",'O',0,0]]

BASE_ATOMS = {
    'DA': [["N9",'N',0,0],["C8",'C',0,0],["H8",'H',0,0],
           ["N7",'N',0,0],["C5",'C',0,0],["C6",'C',0,0],
           ["N6",'N',0,0],["H61",'H',0,0],["H62",'H',0,0],
           ["N1",'N',0,0],["C2",'C',0,0],["H2",'H',0,0],
           ["N3",'N',0,0],["C4",'C',0,0]],
    'DG': [["N9",'N',0,0],["C8",'C',0,0],["H8",'H',0,0],
           ["N7",'N',0,0],["C5",'C',0,0],["C6",'C',0,0],
           ["O6",'O',0,0],["N1",'N',0,0],["H1",'H',0,0],
           ["C2",'C',0,0],["N2",'N',0,0],["H21",'H',0,0],
           ["H22",'H',0,0],["N3",'N',0,0],["C4",'C',0,0]],
    'DT': [["N1",'N',0,0],["C2",'C',0,0],["O2",'O',0,0],
           ["N3",'N',0,0],["H3",'H',0,0],["C4",'C',0,0],
           ["O4",'O',0,0],["C5",'C',0,0],["C7",'C',0,0],
           ["H71",'H',0,0],["H72",'H',0,0],["H73",'H',0,0],
           ["C6",'C',0,0],["H6",'H',0,0]],
    'DC': [["N1",'N',0,0],["C2",'C',0,0],["O2",'O',0,0],
           ["N3",'N',0,0],["C4",'C',0,0],["N4",'N',0,0],
           ["H41",'H',0,0],["H42",'H',0,0],
           ["C5",'C',0,0],["H5",'H',0,0],["C6",'C',0,0],["H6",'H',0,0]],
    'A' : [["N9",'N',0,0],["C8",'C',0,0],["H8",'H',0,0],
           ["N7",'N',0,0],["C5",'C',0,0],["C6",'C',0,0],
           ["N6",'N',0,0],["H61",'H',0,0],["H62",'H',0,0],
           ["N1",'N',0,0],["C2",'C',0,0],["H2",'H',0,0],
           ["N3",'N',0,0],["C4",'C',0,0]],
    'G' : [["N9",'N',0,0],["C8",'C',0,0],["H8",'H',0,0],
           ["N7",'N',0,0],["C5",'C',0,0],["C6",'C',0,0],
           ["O6",'O',0,0],["N1",'N',0,0],["H1",'H',0,0],
           ["C2",'C',0,0],["N2",'N',0,0],["H21",'H',0,0],
           ["H22",'H',0,0],["N3",'N',0,0],["C4",'C',0,0]],
    'U' : [["N1",'N',0,0],["C2",'C',0,0],["O2",'O',0,0],
           ["N3",'N',0,0],["H3",'H',0,0],["C4",'C',0,0],
           ["O4",'O',0,0],["C5",'C',0,0],["H5",'H',0,0],
           ["C6",'C',0,0],["H6",'H',0,0]],
    'C' : [["N1",'N',0,0],["C2",'C',0,0],["O2",'O',0,0],
           ["N3",'N',0,0],["C4",'C',0,0],["N4",'N',0,0],
           ["H41",'H',0,0],["H42",'H',0,0],
           ["C5",'C',0,0],["H5",'H',0,0],["C6",'C',0,0],["H6",'H',0,0]],
}

# 2D key → PDB name mapping (O4t = thymine O4, not sugar O4')
KEY_TO_PDB = {'O4t':"O4", "O4p":"O4'", "O5p":"O5'", "C5p":"C5'",
              "H5p":"H5'","H5pp":"H5''","C4p":"C4'","H4p":"H4'",
              "C3p":"C3'","H3p":"H3'","C2p":"C2'","H2p":"H2'",
              "H2pp":"H2''","C1p":"C1'","H1p":"H1'","O3p":"O3'",
              "O2p":"O2'","HO2p":"HO2'"}

def key_to_pdb(k):
    return KEY_TO_PDB.get(k, k)

# Bonds for each nucleotide (0-indexed; backbone first then base)
# Format: {str(atom_idx): [bonded_atom_indices]}

def make_bonds_dna_purine():
    # Indices: BB_ORDER_DNA = 18 atoms (0-17), BASE purine = 14 atoms (18-31)
    # P=0,OP1=1,OP2=2,O5'=3,C5'=4,H5'=5,H5''=6,
    # C4'=7,H4'=8,O4'=9,C3'=10,H3'=11,
    # C2'=12,H2'=13,H2''=14,C1'=15,H1'=16,O3'=17
    # N9=18,C8=19,H8=20,N7=21,C5=22,C6=23,N6=24,H61=25,H62=26,
    # N1=27,C2=28,H2=29,N3=30,C4=31
    b = {
        0:[1,2,3,17],      # P-OP1,OP2,O5',O3'(prev via inter)
        1:[0],2:[0],       # OP1,OP2
        3:[0,4],           # O5'-P,C5'
        4:[3,5,6,7],       # C5'-O5',H5',H5'',C4'
        5:[4],6:[4],       # H5',H5''
        7:[4,8,9,10],      # C4'-C5',H4',O4',C3'
        8:[7],             # H4'
        9:[7,15],          # O4'-C4',C1'
        10:[7,11,12,17],   # C3'-C4',H3',C2',O3'
        11:[10],           # H3'
        12:[10,13,14,15],  # C2'-C3',H2',H2'',C1'
        13:[12],14:[12],   # H2',H2''
        15:[9,12,16,18],   # C1'-O4',C2',H1',N9
        16:[15],           # H1'
        17:[10],           # O3'-C3' (inter-nt bond added by Build)
        # Base: adenine
        18:[15,19,31],     # N9-C1',C8,C4
        19:[18,20,21],     # C8-N9,H8,N7
        20:[19],           # H8
        21:[19,22],        # N7-C8,C5
        22:[21,23,31],     # C5-N7,C6,C4
        23:[22,24,27],     # C6-C5,N6,N1
        24:[23,25,26],     # N6-C6,H61,H62
        25:[24],26:[24],   # H61,H62
        27:[23,28],        # N1-C6,C2
        28:[27,29,30],     # C2-N1,H2,N3
        29:[28],           # H2
        30:[28,31],        # N3-C2,C4
        31:[18,22,30],     # C4-N9,C5,N3
    }
    return {str(k):v for k,v in b.items()}

def make_bonds_dna_guanine():
    # BB 0-17 same; G base 14+1=15 atoms (18-32)
    # N9=18,C8=19,H8=20,N7=21,C5=22,C6=23,O6=24,N1=25,H1=26,
    # C2=27,N2=28,H21=29,H22=30,N3=31,C4=32
    b = {
        0:[1,2,3],1:[0],2:[0],
        3:[0,4],4:[3,5,6,7],5:[4],6:[4],
        7:[4,8,9,10],8:[7],9:[7,15],
        10:[7,11,12,17],11:[10],
        12:[10,13,14,15],13:[12],14:[12],
        15:[9,12,16,18],16:[15],17:[10],
        18:[15,19,32],     # N9
        19:[18,20,21],20:[19],
        21:[19,22],
        22:[21,23,32],     # C5
        23:[22,24,25],     # C6-C5,O6,N1
        24:[23],           # O6
        25:[23,26,27],     # N1-C6,H1,C2
        26:[25],           # H1
        27:[25,28,31],     # C2-N1,N2,N3
        28:[27,29,30],29:[28],30:[28],
        31:[27,32],        # N3-C2,C4
        32:[18,22,31],     # C4
    }
    return {str(k):v for k,v in b.items()}

def make_bonds_dna_thymine():
    # BB 0-17; T base 14 atoms (18-31)
    # N1=18,C2=19,O2=20,N3=21,H3=22,C4=23,O4=24,
    # C5=25,C7=26,H71=27,H72=28,H73=29,C6=30,H6=31
    b = {
        0:[1,2,3],1:[0],2:[0],
        3:[0,4],4:[3,5,6,7],5:[4],6:[4],
        7:[4,8,9,10],8:[7],9:[7,15],
        10:[7,11,12,17],11:[10],
        12:[10,13,14,15],13:[12],14:[12],
        15:[9,12,16,18],16:[15],17:[10],
        18:[15,19,30],     # N1-C1',C2,C6
        19:[18,20,21],     # C2-N1,O2,N3
        20:[19],           # O2
        21:[19,22,23],     # N3-C2,H3,C4
        22:[21],           # H3
        23:[21,24,25],     # C4-N3,O4,C5
        24:[23],           # O4
        25:[23,26,30],     # C5-C4,C7,C6
        26:[25,27,28,29],  # C7-C5,H71,H72,H73
        27:[26],28:[26],29:[26],
        30:[18,25,31],     # C6-N1,C5,H6
        31:[30],           # H6
    }
    return {str(k):v for k,v in b.items()}

def make_bonds_dna_cytosine():
    # BB 0-17; C base 12 atoms (18-29)
    # N1=18,C2=19,O2=20,N3=21,C4=22,N4=23,H41=24,H42=25,
    # C5=26,H5=27,C6=28,H6=29
    b = {
        0:[1,2,3],1:[0],2:[0],
        3:[0,4],4:[3,5,6,7],5:[4],6:[4],
        7:[4,8,9,10],8:[7],9:[7,15],
        10:[7,11,12,17],11:[10],
        12:[10,13,14,15],13:[12],14:[12],
        15:[9,12,16,18],16:[15],17:[10],
        18:[15,19,28],     # N1
        19:[18,20,21],20:[19],
        21:[19,22],        # N3
        22:[21,23,26],     # C4
        23:[22,24,25],24:[23],25:[23],
        26:[22,27,28],27:[26],
        28:[18,26,29],29:[28],
    }
    return {str(k):v for k,v in b.items()}

def make_bonds_rna_purine(base='A'):
    # RNA BB: 19 atoms (0-18), base purine 14 or 15 atoms
    # P=0,OP1=1,OP2=2,O5'=3,C5'=4,H5'=5,H5''=6,
    # C4'=7,H4'=8,O4'=9,C3'=10,H3'=11,
    # C2'=12,H2'=13,O2'=14,HO2'=15,C1'=16,H1'=17,O3'=18
    offset = 19
    if base == 'A':
        # Adenine: N9=19,...,C4=32  (same as DNA adenine but +1 for all base indices)
        b = {
            0:[1,2,3],1:[0],2:[0],
            3:[0,4],4:[3,5,6,7],5:[4],6:[4],
            7:[4,8,9,10],8:[7],9:[7,16],
            10:[7,11,12,18],11:[10],
            12:[10,13,14,16],13:[12],
            14:[12,15],15:[14],  # O2',HO2'
            16:[9,12,17,19],17:[16],18:[10],
            # Adenine base
            19:[16,20,32],20:[19,21,22],21:[20],
            22:[20,23],23:[22,24,32],
            24:[23,25,28],25:[24,26,27],26:[25],27:[25],
            28:[24,29],29:[28,30],30:[29],
            31:[29,32],32:[19,23,31],
        }
    else:  # G
        b = {
            0:[1,2,3],1:[0],2:[0],
            3:[0,4],4:[3,5,6,7],5:[4],6:[4],
            7:[4,8,9,10],8:[7],9:[7,16],
            10:[7,11,12,18],11:[10],
            12:[10,13,14,16],13:[12],
            14:[12,15],15:[14],
            16:[9,12,17,19],17:[16],18:[10],
            19:[16,20,33],20:[19,21,22],21:[20],
            22:[20,23],23:[22,24,33],
            24:[23,25,26],25:[24],
            26:[24,27,28],27:[26],
            28:[26,29,32],29:[28,30,31],30:[29],31:[29],
            32:[28,33],33:[19,23,32],
        }
    return {str(k):v for k,v in b.items()}

def make_bonds_rna_pyrimidine(base='C'):
    # RNA BB 0-18; pyrimidine base starts at 19
    if base == 'C':
        # C: 12 atoms (19-30)
        b = {
            0:[1,2,3],1:[0],2:[0],
            3:[0,4],4:[3,5,6,7],5:[4],6:[4],
            7:[4,8,9,10],8:[7],9:[7,16],
            10:[7,11,12,18],11:[10],
            12:[10,13,14,16],13:[12],
            14:[12,15],15:[14],
            16:[9,12,17,19],17:[16],18:[10],
            19:[16,20,29],20:[19,21,22],21:[20],
            22:[20,23],23:[22,24,27],24:[23,25,26],25:[24],26:[24],
            27:[23,28],28:[27,29,30],29:[19,27,30],30:[28], # wait, check
            # N1=19,C2=20,O2=21,N3=22,C4=23,N4=24,H41=25,H42=26,C5=27,H5=28,C6=29,H6=30
            # Redo:
        }
        b = {
            0:[1,2,3],1:[0],2:[0],
            3:[0,4],4:[3,5,6,7],5:[4],6:[4],
            7:[4,8,9,10],8:[7],9:[7,16],
            10:[7,11,12,18],11:[10],
            12:[10,13,14,16],13:[12],
            14:[12,15],15:[14],
            16:[9,12,17,19],17:[16],18:[10],
            19:[16,20,29],   # N1
            20:[19,21,22],   # C2
            21:[20],         # O2
            22:[20,23],      # N3
            23:[22,24,27],   # C4
            24:[23,25,26],   # N4
            25:[24],26:[24], # H41,H42
            27:[23,28,29],   # C5
            28:[27],         # H5
            29:[19,27,30],   # C6
            30:[29],         # H6
        }
    else:  # U: 11 atoms (19-29)
        b = {
            0:[1,2,3],1:[0],2:[0],
            3:[0,4],4:[3,5,6,7],5:[4],6:[4],
            7:[4,8,9,10],8:[7],9:[7,16],
            10:[7,11,12,18],11:[10],
            12:[10,13,14,16],13:[12],
            14:[12,15],15:[14],
            16:[9,12,17,19],17:[16],18:[10],
            19:[16,20,28],   # N1
            20:[19,21,22],   # C2
            21:[20],         # O2
            22:[20,23,24],   # N3
            23:[22],         # H3
            24:[22,25,27],   # C4
            25:[24],         # O4
            26:[24,27,28],   # C5 (wait, idx)
            # N1=19,C2=20,O2=21,N3=22,H3=23,C4=24,O4=25,C5=26,H5=27,C6=28,H6=29
        }
        b = {
            0:[1,2,3],1:[0],2:[0],
            3:[0,4],4:[3,5,6,7],5:[4],6:[4],
            7:[4,8,9,10],8:[7],9:[7,16],
            10:[7,11,12,18],11:[10],
            12:[10,13,14,16],13:[12],
            14:[12,15],15:[14],
            16:[9,12,17,19],17:[16],18:[10],
            19:[16,20,28],   # N1
            20:[19,21,22],   # C2
            21:[20],         # O2
            22:[20,23,24],   # N3
            23:[22],         # H3
            24:[22,25,26],   # C4
            25:[24],         # O4
            26:[24,27,28],   # C5
            27:[26],         # H5
            28:[19,26,29],   # C6
            29:[28],         # H6
        }
    return {str(k):v for k,v in b.items()}

# ── Assemble one entry ────────────────────────────────────────────────────────

def assemble(code, form, chi, bb_order, bb_atoms, base_atoms_meta, bonds_fn,
             base_keys_order, R_helix, c_helix):
    """
    Build one NucleotidesDB entry.
    base_keys_order: list of internal key names for base atoms in order.
    R_helix, c_helix: helix-frame transform  y = R_helix @ (x - c_helix).
    """
    ang   = ARNOTT_DNA if form == 'DNA' else ARNOTT_RNA
    atoms = build_nucleotide(code, form, chi=chi, **ang)

    def to_helix(v):
        return R_helix @ (np.asarray(v, float) - c_helix)

    # Collect vectors in canonical order (helix frame)
    vectors = []
    for key in bb_order:
        v = atoms.get(key)
        if v is None:
            print(f"  MISSING backbone atom: {key} in {code}")
            v = np.array([0.0,0.0,0.0])
        vectors.append([round(float(x),4) for x in to_helix(v)])

    for key in base_keys_order:
        v = atoms.get(key)
        if v is None:
            print(f"  MISSING base atom: {key} in {code}")
            v = np.array([0.0,0.0,0.0])
        vectors.append([round(float(x),4) for x in to_helix(v)])

    # Chi angle atom
    purine = code in ('DA','DG','A','G')
    chi_atom = "C4" if purine else "C2"
    glc_N   = "N9" if purine else "N1"

    entry = {
        "Tricode":    code,
        "Type":       form,
        "Vectors":    vectors,
        "Backbone Atoms": bb_atoms,
        "Base Atoms": base_atoms_meta,
        "Chi Angle Atoms": ["O4'","C1'",glc_N, chi_atom],
        "Bonds":      bonds_fn(),
    }
    return entry

# ── Base atom key orders ──────────────────────────────────────────────────────

ADENINE_KEYS = ['N9','C8','H8','N7','C5','C6','N6','H61','H62','N1','C2','H2','N3','C4']
GUANINE_KEYS = ['N9','C8','H8','N7','C5','C6','O6','N1','H1','C2','N2','H21','H22','N3','C4']
THYMINE_KEYS = ['N1','C2','O2','N3','H3','C4','O4t','C5','C7','H71','H72','H73','C6','H6']
CYTOSINE_KEYS= ['N1','C2','O2','N3','C4','N4','H41','H42','C5','H5','C6','H6']
URACIL_KEYS  = ['N1','C2','O2','N3','H3','C4','O4t','C5','H5','C6','H6']

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    db = {}

    print("Computing DNA helix frame...")
    R_dna, c_dna = compute_helix_frame('DNA')
    print("Computing RNA helix frame...")
    R_rna, c_rna = compute_helix_frame('RNA')

    configs = [
        # (code, form, chi, bb_order, bb_atoms, base_meta, bonds_fn, base_keys,
        #  R_helix, c_helix)
        ('DA','DNA', 115, BB_ORDER_DNA, BB_ATOMS_DNA, BASE_ATOMS['DA'],
         make_bonds_dna_purine,  ADENINE_KEYS,  R_dna, c_dna),
        ('DG','DNA', 115, BB_ORDER_DNA, BB_ATOMS_DNA, BASE_ATOMS['DG'],
         make_bonds_dna_guanine, GUANINE_KEYS,  R_dna, c_dna),
        ('DT','DNA', 115, BB_ORDER_DNA, BB_ATOMS_DNA, BASE_ATOMS['DT'],
         make_bonds_dna_thymine, THYMINE_KEYS,  R_dna, c_dna),
        ('DC','DNA', 115, BB_ORDER_DNA, BB_ATOMS_DNA, BASE_ATOMS['DC'],
         make_bonds_dna_cytosine,CYTOSINE_KEYS, R_dna, c_dna),
        ('A', 'RNA', 160, BB_ORDER_RNA, BB_ATOMS_RNA, BASE_ATOMS['A'],
         lambda: make_bonds_rna_purine('A'),  ADENINE_KEYS,  R_rna, c_rna),
        ('G', 'RNA', 160, BB_ORDER_RNA, BB_ATOMS_RNA, BASE_ATOMS['G'],
         lambda: make_bonds_rna_purine('G'),  GUANINE_KEYS,  R_rna, c_rna),
        ('C', 'RNA', 160, BB_ORDER_RNA, BB_ATOMS_RNA, BASE_ATOMS['C'],
         lambda: make_bonds_rna_pyrimidine('C'), CYTOSINE_KEYS, R_rna, c_rna),
        ('U', 'RNA', 160, BB_ORDER_RNA, BB_ATOMS_RNA, BASE_ATOMS['U'],
         lambda: make_bonds_rna_pyrimidine('U'), URACIL_KEYS,  R_rna, c_rna),
    ]

    for cfg in configs:
        code = cfg[0]
        print(f"Building {code}...")
        try:
            entry = assemble(*cfg)
            db[code] = entry
            n_vec = len(entry['Vectors'])
            n_bb  = len(entry['Backbone Atoms'])
            n_base= len(entry['Base Atoms'])
            print(f"  vectors={n_vec}, BB={n_bb}, base={n_base}, "
                  f"total expected={n_bb+n_base}")
        except Exception as ex:
            import traceback; traceback.print_exc()

    out = '/Users/slurm/Desktop/Pose/pose/NucleotidesDB.json'
    with open(out, 'w') as f:
        json.dump(db, f, indent=2)
    print(f"\nWrote {out}")

    # Quick sanity check on DA
    if 'DA' in db:
        da = db['DA']
        vecs = da['Vectors']
        # BB_ORDER_DNA: P=0,OP1=1,OP2=2,O5'=3,C5'=4,H5'=5,H5''=6
        #               C4'=7,H4'=8,O4'=9,C3'=10,H3'=11
        #               C2'=12,H2'=13,H2''=14,C1'=15,H1'=16,O3'=17
        # Base (ADENINE_KEYS): N9=18,C8=19,H8=20,N7=21,C5=22,C6=23,
        #                      N6=24,H61=25,H62=26,N1=27,C2=28,H2=29,
        #                      N3=30,C4=31
        P   = np.array(vecs[0]); O5p = np.array(vecs[3])
        C5p = np.array(vecs[4]); C4p = np.array(vecs[7])
        C3p = np.array(vecs[10]); O3p = np.array(vecs[17])
        C2p = np.array(vecs[12]); C1p = np.array(vecs[15])
        O4p = np.array(vecs[9]);  N9  = np.array(vecs[18])
        C4  = np.array(vecs[31])
        print(f"\nDA sanity: P-O5'   = {dist(P,O5p):.3f} Å (should be ~1.593)")
        print(f"DA sanity: C1'-O4' = {dist(C1p,O4p):.3f} Å (should be ~1.440)")
        print(f"DA sanity: C1'-N9  = {dist(C1p,N9):.3f} Å (should be ~1.462)")
        print(f"DA sanity: C5'-C4' = {dist(C5p,C4p):.3f} Å (should be ~1.510)")
        print(f"DA sanity: C3'-O3' = {dist(C3p,O3p):.3f} Å (should be ~1.430)")
        delta_m = dihed(C5p,C4p,C3p,O3p)
        chi_m   = dihed(O4p,C1p,N9,C4)
        # dihed() convention: dihed=-phi when place(phi) was used.
        # IUPAC = -dihed(). Empirical target delta~135 (IUPAC~+135).
        print(f"DA sanity: delta  = {delta_m:.1f}°"
              f" (IUPAC={-delta_m:.0f}°, empirical ~135)")
        print(f"DA sanity: chi    = {chi_m:.1f}°"
              f" (IUPAC={-chi_m:.0f}°, anti=-115)")
        # Measure C1' position
        r_C1 = math.hypot(C1p[0], C1p[1])
        print(f"DA sanity: C1' r_xy = {r_C1:.2f} Å (should be ~8.07)")

    # ── Helix step diagnostic ──────────────────────────────────────────────
    print("\n--- B-DNA helix step diagnostic ---")
    nt0 = build_nucleotide('DA', 'DNA', chi=115, **ARNOTT_DNA)
    nt1 = extend_chain(nt0, 'DNA', **ARNOTT_DNA)
    keys_bb = ['P','O5p','C5p','C4p','O4p','C3p','C2p','C1p','O3p']
    src = np.array([nt0[k] for k in keys_bb])
    tgt = np.array([nt1[k] for k in keys_bb])
    sc, tc = src.mean(0), tgt.mean(0)
    H = (src - sc).T @ (tgt - tc)
    U2, _, Vt2 = np.linalg.svd(H)
    R_s = Vt2.T @ U2.T
    if np.linalg.det(R_s) < 0:
        Vt2[-1] *= -1; R_s = Vt2.T @ U2.T
    t_s = tc - R_s @ sc
    eigvals2, eigvecs2 = np.linalg.eig(R_s)
    real_i = np.argmin(np.abs(np.imag(eigvals2)))
    u2 = np.real(eigvecs2[:, real_i]); u2 /= np.linalg.norm(u2)
    if np.dot(u2, t_s) < 0: u2 = -u2
    rise_m = abs(float(np.dot(t_s, u2)))
    rot_axis_v = u2 / np.linalg.norm(u2)
    trace_val  = np.trace(R_s)
    twist_m = math.degrees(math.acos(max(-1.0,
                           min(1.0, (trace_val - 1.0 - np.dot(rot_axis_v,
                           rot_axis_v)) / 2.0
                           ))))
    # simpler: angle from eigenvalue
    cos_t = (np.trace(R_s) - 1.0) / 2.0
    twist_m2 = math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))
    C1_r = math.hypot(nt0['C1p'][0], nt0['C1p'][1]) if True else 0
    print(f"Rise per step:  {rise_m:.3f} Å (expected 3.38)")
    print(f"Twist per step: {twist_m2:.1f}° (expected 36.0)")
    print(f"C1' r (local frame): {C1_r:.2f} Å")
    # Also measure in helix frame
    if 'DA' in db:
        vecs2 = db['DA']['Vectors']
        C1h = np.array(vecs2[15])
        r_helix = math.hypot(C1h[0], C1h[1])
        print(f"C1' r_xy (helix frame): {r_helix:.2f} Å (expected ~8.07)")
    print("\n--- A-RNA helix step diagnostic ---")
    rnt0 = build_nucleotide('A', 'RNA', chi=160, **ARNOTT_RNA)
    rnt1 = extend_chain(rnt0, 'RNA', **ARNOTT_RNA)
    rsrc = np.array([rnt0[k] for k in keys_bb])
    rtgt = np.array([rnt1[k] for k in keys_bb])
    rsc, rtc = rsrc.mean(0), rtgt.mean(0)
    rH = (rsrc - rsc).T @ (rtgt - rtc)
    rU, _, rVt = np.linalg.svd(rH)
    rR = rVt.T @ rU.T
    if np.linalg.det(rR) < 0:
        rVt[-1] *= -1; rR = rVt.T @ rU.T
    rt = rtc - rR @ rsc
    rcos = (np.trace(rR) - 1.0) / 2.0
    rtwist = math.degrees(math.acos(max(-1.0, min(1.0, rcos))))
    reigv, reigvec = np.linalg.eig(rR)
    ri = np.argmin(np.abs(np.imag(reigv)))
    ru = np.real(reigvec[:, ri]); ru /= np.linalg.norm(ru)
    if np.dot(ru, rt) < 0: ru = -ru
    rrise = abs(float(np.dot(rt, ru)))
    print(f"Rise per step:  {rrise:.3f} Å (expected 2.81)")
    print(f"Twist per step: {rtwist:.1f}° (expected 32.7)")
