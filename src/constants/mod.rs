pub mod vsap;
pub mod c2s;
mod cartesian_gto;

use std::collections::HashMap;

use lazy_static::lazy_static;

pub use crate::constants::vsap::*;
pub use crate::constants::c2s::*;
pub use crate::constants::cartesian_gto::*;

//struct Matrix3x3 {
//    size
//
//}

pub const SPECIES_NAME: [&str; 72] = ["H", "He",
        "Li","Be","B", "C",
        "N", "O", "F", "Ne",
        "Na","Mg","Al","Si",
        "P", "S", "Cl","Ar",
        "K", "Ca","Ga","Ge",
        "As","Se","Br","Kr",
        "Sc","Ti","V", "Cr","Mn",
        "Fe","Co","Ni","Cu","Zn",
        "Rb","Sr","In","Sn",
        "Sb","Te","I", "Xe",
        "Y", "Zr","Nb","Mo","Tc",
        "Ru","Rh","Pd","Ag","Cd",
        "Cs","Ba","Tl","Pb",
        "Bi","Po","At","Rn",
        "La","Hf","Ta","W", "Re",
        "Os","Ir","Pt","Au","Hg"
        ];

pub const MASS_CHARGE: [(f64,f64);72] = [(1.00794,1.0), (4.002602,2.0),
   (6.9410, 3.0),  (9.0122, 4.0), (10.8110, 5.0), (12.0107, 6.0), 
  (14.0067, 7.0), (15.9994, 8.0), (18.9984, 9.0), (20.1797,10.0),
  (22.9897,11.0), (24.3050,12.0), (26.9815,13.0), (28.0855,14.0), 
  (30.9738,15.0), (32.0650,16.0), (35.4530,17.0), (39.9480,18.0),
  (39.0983,19.0), (40.0780,20.0), (69.7230,31.0), (72.6400,32.0), 
  (74.9216,33.0), (78.9600,34.0), (79.9040,35.0), (83.7980,36.0),
  (44.9559,21.0), (47.8670,22.0), (50.9415,23.0), (51.9961,24.0), (54.9380,25.0),
  (55.8450,26.0), (58.9332,27.0), (58.6934,28.0), (63.5460,29.0), (65.3800,30.0),
  (85.4678,37.0), (87.6200,38.0),(114.8180,49.0),(118.7100,50.0),
 (121.7600,51.0),(127.6000,52.0),(126.9045,53.0),(131.2930,54.0),
  (88.9058,39.0), (91.2240,40.0), (92.9064,41.0), (95.9600,42.0), (98.0000,43.0),
 (101.0700,44.0),(102.9055,45.0),(106.4200,46.0),(107.8682,47.0),(112.4110,48.0),
 (132.9054,55.0),(137.3270,56.0),(204.3833,81.0),(207.2000,82.0),
 (208.9804,83.0),(209.0000,84.0),(210.0000,85.0),(222.0000,86.0),
 (138.9055,57.0),(178.4900,72.0),(180.9479,73.0),(183.8400,74.0),(186.2070,75.0),
 (190.2300,76.0),(192.2170,77.0),(195.0840,78.0),(196.9666,79.0),(200.5900,80.0)
   ];

pub const ELEM1ST: [&str;2]  = ["H", "He"];
pub const ELEM2ND: [&str;8]  = ["Li","Be","B", "C", "N", "O", "F", "Ne"];
pub const ELEM3RD: [&str;8]  = ["Na","Mg","Al","Si","P", "S", "Cl","Ar"];
pub const ELEM4TH: [&str;18] = ["K", "Ca","Ga","Ge","As","Se","Br","Kr",
                                "Sc","Ti","V", "Cr","Mn","Fe","Co","Ni","Cu","Zn"];
pub const ELEM5TH: [&str;18] = ["Rb","Sr","In","Sn","Sb","Te","I", "Xe",
                                "Y", "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd"];
pub const ELEM6TH: [&str;18] = ["Cs","Ba","Tl","Pb","Bi","Po","At","Rn", 
                                "La","Hf","Ta","W", "Re","Os","Ir","Pt","Au","Hg"];
pub const ELEMTMS: [&str;30] = ["Sc","Ti","V", "Cr","Mn","Fe","Co","Ni","Cu","Zn",
                                "Y", "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
                                "La","Hf","Ta","W", "Re","Os","Ir","Pt","Au","Hg"];


lazy_static!{
    pub static ref SPECIES_INFO: HashMap<&'static str, &'static (f64,f64)> = {
        let mut m = HashMap::new();
        SPECIES_NAME.iter().zip(MASS_CHARGE.iter()).for_each(|(name,info)| {
            m.insert(*name,info);
        });
        m
    };
}

// use for the inverse (sqrt inverse) of the auxiliary coulomb matrix
pub const AUXBAS_THRESHOLD: f64 = 1.0e-10;
pub const INVERSE_THRESHOLD: f64 = 1.0e-10;
pub const SQRT_THRESHOLD: f64 = 1.0e-10;

pub const CM: f64 = 8065.541;
pub const ANG:f64 = 0.5291772083;
pub const EV: f64 = 27.2113845;
pub const FQ: f64 = 1822.888;
pub const E:  f64 = std::f64::consts::E;
pub const PI: f64 = std::f64::consts::PI;

pub const E5: f64 = 1.0e5;
pub const E6: f64 = 1.0e6;
pub const E7: f64 = 1.0e7;
pub const E8: f64 = 1.0e8;
pub const E9: f64 = 1.0e9;


pub struct DMatrix<const L: usize> {
    size:[usize;2],
    indicing:[usize;2],
    data: [f64; L]
}
pub const ATOM_CONFIGURATION: [[usize; 4]; 119] = [
    [ 0, 0, 0, 0],     //  0  GHOST
    [ 1, 0, 0, 0],     //  1  H
    [ 2, 0, 0, 0],     //  2  He
    [ 3, 0, 0, 0],     //  3  Li
    [ 4, 0, 0, 0],     //  4  Be
    [ 4, 1, 0, 0],     //  5  B
    [ 4, 2, 0, 0],     //  6  C
    [ 4, 3, 0, 0],     //  7  N
    [ 4, 4, 0, 0],     //  8  O
    [ 4, 5, 0, 0],     //  9  F
    [ 4, 6, 0, 0],     // 10  Ne
    [ 5, 6, 0, 0],     // 11  Na
    [ 6, 6, 0, 0],     // 12  Mg
    [ 6, 7, 0, 0],     // 13  Al
    [ 6, 8, 0, 0],     // 14  Si
    [ 6, 9, 0, 0],     // 15  P
    [ 6,10, 0, 0],     // 16  S
    [ 6,11, 0, 0],     // 17  Cl
    [ 6,12, 0, 0],     // 18  Ar
    [ 7,12, 0, 0],     // 19  K
    [ 8,12, 0, 0],     // 20  Ca
    [ 8,12, 1, 0],     // 21  Sc
    [ 8,12, 2, 0],     // 22  Ti
    [ 8,12, 3, 0],     // 23  V
    [ 7,12, 5, 0],     // 24  Cr
    [ 8,12, 5, 0],     // 25  Mn
    [ 8,12, 6, 0],     // 26  Fe
    [ 8,12, 7, 0],     // 27  Co
    [ 8,12, 8, 0],     // 28  Ni
    [ 7,12,10, 0],     // 29  Cu
    [ 8,12,10, 0],     // 30  Zn
    [ 8,13,10, 0],     // 31  Ga
    [ 8,14,10, 0],     // 32  Ge
    [ 8,15,10, 0],     // 33  As
    [ 8,16,10, 0],     // 34  Se
    [ 8,17,10, 0],     // 35  Br
    [ 8,18,10, 0],     // 36  Kr
    [ 9,18,10, 0],     // 37  Rb
    [10,18,10, 0],     // 38  Sr
    [10,18,11, 0],     // 39  Y
    [10,18,12, 0],     // 40  Zr
    [ 9,18,14, 0],     // 41  Nb
    [ 9,18,15, 0],     // 42  Mo
    [10,18,15, 0],     // 43  Tc
    [ 9,18,17, 0],     // 44  Ru
    [ 9,18,18, 0],     // 45  Rh
    [ 8,18,20, 0],     // 46  Pd
    [ 9,18,20, 0],     // 47  Ag
    [10,18,20, 0],     // 48  Cd
    [10,19,20, 0],     // 49  In
    [10,20,20, 0],     // 50  Sn
    [10,21,20, 0],     // 51  Sb
    [10,22,20, 0],     // 52  Te
    [10,23,20, 0],     // 53  I
    [10,24,20, 0],     // 54  Xe
    [11,24,20, 0],     // 55  Cs
    [12,24,20, 0],     // 56  Ba
    [12,24,21, 0],     // 57  La
    [12,24,21, 1],     // 58  Ce
    [12,24,20, 3],     // 59  Pr
    [12,24,20, 4],     // 60  Nd
    [12,24,20, 5],     // 61  Pm
    [12,24,20, 6],     // 62  Sm
    [12,24,20, 7],     // 63  Eu
    [12,24,21, 7],     // 64  Gd
    [12,24,21, 8],     // 65  Tb
    [12,24,20,10],     // 66  Dy
    [12,24,20,11],     // 67  Ho
    [12,24,20,12],     // 68  Er
    [12,24,20,13],     // 69  Tm
    [12,24,20,14],     // 70  Yb
    [12,24,21,14],     // 71  Lu
    [12,24,22,14],     // 72  Hf
    [12,24,23,14],     // 73  Ta
    [12,24,24,14],     // 74  W
    [12,24,25,14],     // 75  Re
    [12,24,26,14],     // 76  Os
    [12,24,27,14],     // 77  Ir
    [11,24,29,14],     // 78  Pt
    [11,24,30,14],     // 79  Au
    [12,24,30,14],     // 80  Hg
    [12,25,30,14],     // 81  Tl
    [12,26,30,14],     // 82  Pb
    [12,27,30,14],     // 83  Bi
    [12,28,30,14],     // 84  Po
    [12,29,30,14],     // 85  At
    [12,30,30,14],     // 86  Rn
    [13,30,30,14],     // 87  Fr
    [14,30,30,14],     // 88  Ra
    [14,30,31,14],     // 89  Ac
    [14,30,32,14],     // 90  Th
    [14,30,31,16],     // 91  Pa
    [14,30,31,17],     // 92  U
    [14,30,31,18],     // 93  Np
    [14,30,30,20],     // 94  Pu
    [14,30,30,21],     // 95  Am
    [14,30,31,21],     // 96  Cm
    [14,30,31,22],     // 97  Bk
    [14,30,30,24],     // 98  Cf
    [14,30,30,25],     // 99  Es
    [14,30,30,26],     //100  Fm
    [14,30,30,27],     //101  Md
    [14,30,30,28],     //102  No
    [14,30,31,28],     //103  Lr
    [14,30,32,28],     //104  Rf
    [14,30,33,28],     //105  Db
    [14,30,34,28],     //106  Sg
    [14,30,35,28],     //107  Bh
    [14,30,36,28],     //108  Hs
    [14,30,37,28],     //109  Mt
    [14,30,38,28],     //110  Ds
    [14,30,39,28],     //111  Rg
    [14,30,40,28],     //112  Cn
    [14,31,40,28],     //113  Nh
    [14,32,40,28],     //114  Fl
    [14,33,40,28],     //115  Mc
    [14,34,40,28],     //116  Lv
    [14,35,40,28],     //117  Ts
    [14,36,40,28],     //118  Og
];

// =========== libcint ===================================
// for the bas index - libcint
pub const BAS_ATM: usize = 0;
pub const BAS_ANG: usize = 1;
pub const BAS_PRM: usize = 2;
pub const BAS_CTR: usize = 3;
pub const BAS_SLOTS: usize = 6;

// for the atm index - libcint 
pub const ATM_NUC: usize = 0;
pub const ATM_ENV: usize = 1;
pub const ATM_NUC_MOD_OF: usize = 2;
pub const ATM_FRAC_CHARGE_OF: usize = 3;
pub const ATM_SLOTS: usize = 6;

// for ECP - libcint
pub const ECP_LMAX: i32 = 5;
pub const NUC_ECP:  i32 = 4;

// for exp cutoff -libcint
pub const PTR_EXPCUTOFF: i32 = 0;
// for dipole - libcint
pub const PTR_COMMON_ORG: i32 = 1;
// for Gauge origin
pub const PTR_RINV_ORIG: i32 = 4;

pub const NUC_MOD_OF: i32 = 2;

pub const NUC_STAD_CHARGE: i32 = 1;
pub const NUC_GAUS_CHARGE: i32 = 2;
pub const NUC_FRAC_CHARGE: i32 = 3;
// =========== libcint ===================================


//// for the atm index - libcint
//pub const ATM_CHARGE_OF: usize = 0;
//pub const ATM_PRT_COORD: usize = 1;
//pub const ATM_NUC_MOD_OF: usize = 2;
//pub const ATM_PRT_ZETA: usize = 3;

pub const ENV_PRT_START: usize = 20;


// SAD and ECP configuration
pub const S_SHELL: [f64; 1] = [2.0];
pub const P_SHELL: [f64; 3] = [2.0, 2.0, 2.0];
pub const D_SHELL: [f64; 5] = [2.0, 2.0, 2.0, 2.0, 2.0];
pub const F_SHELL: [f64; 7] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
pub const XE_SHELL: [f64; 27] =
//   1s   2s   2p   2p   2p   3s   3p   3p   3p   4s   3d   3d   3d   3d   3d   4p   4p   4p   5s   4d   4d   4d   4d   4d   5p   5p   5p  
    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
pub const KR_SHELL: [f64; 18] =
//   1s   2s   2p   2p   2p   3s   3p   3p   3p   4s   3d   3d   3d   3d   3d   4p   4p   4p  
    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

//                                    1s   2s   2p   3s   3p   4s   3d    4p   5s   4d    5p   4f    5d    6s   6p 
pub const NELE_IN_SHELLS: [f64; 15] = [2.0, 2.0, 6.0, 2.0, 6.0, 2.0, 10.0, 6.0, 2.0, 10.0, 6.0, 14.0, 10.0, 2.0, 6.0];

//
pub const LIGHT_SPEED: f64 = 137.03599967994;   // http://physics.nist.gov/cgi-bin/cuu/Value?alph
// BOHR = .529 177 210 92(17) e-10m  // http://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
pub const BOHR: f64 = 0.52917721092;  // Angstroms
pub const BOHR_SI: f64 = BOHR * 1e-10;

pub const G_ELECTRON: f64 = 2.00231930436182;  // http://physics.nist.gov/cgi-bin/cuu/Value?gem
pub const E_MASS: f64 = 9.10938356e-31;         // kg https://physics.nist.gov/cgi-bin/cuu/Value?me
pub const AVOGADRO: f64 = 6.022140857e23;       // https://physics.nist.gov/cgi-bin/cuu/Value?na
pub const PLANCK: f64 = 6.626070040e-34;        // J*s http://physics.nist.gov/cgi-bin/cuu/Value?h
pub const E_CHARGE: f64 = 1.6021766208e-19;
pub const DEBYE:f64 = 3.335641e-30;            // C*m = 1e-18/LIGHT_SPEED_SI https://cccbdb.nist.gov/debye.asp
pub const AU2DEBYE:f64 = E_CHARGE * BOHR*1e-10 / DEBYE; // 2.541746


pub const MPI_CHUNK:usize = 134217728; // around 1 GB