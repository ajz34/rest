use tensors::{MatrixFull, BasicMatrix};
use crate::constants::{F_SHELL, KR_SHELL, NELE_IN_SHELLS, SPECIES_INFO, S_SHELL, XE_SHELL};
use crate::molecule_io::Molecule;
use crate::ctrl_io::InputKeywords;
use crate::geom_io::{GeomCell, formated_element_name};
use crate::scf_io::scf;
use crate::utilities;
use std::collections::HashMap;


pub fn initial_guess_from_sad(mol: &Molecule) -> Vec<MatrixFull<f64>> {
    //let mut elem_name: Vec<String> = vec![];
    //let mut dms_alpha: Vec<MatrixFull<f64>> = vec![];
    //let mut dms_beta: Vec<MatrixFull<f64>> = vec![];
    let mut atom_dms: HashMap<String, Vec<MatrixFull<f64>>> = HashMap::new();

    mol.geom.elem.iter().for_each(|ielem| {
        if let None = &mut atom_dms.get(&ielem.clone()) {


            if mol.ctrl.print_level>0 {println!("Generate SAD of {}", &ielem)};

            //elem_name.push(ielem.to_string());

            let mut atom_ctrl = InputKeywords::init_ctrl();
            atom_ctrl.xc = String::from("hf");
            atom_ctrl.basis_path = mol.ctrl.basis_path.clone();
            atom_ctrl.basis_type = mol.ctrl.basis_type.clone();
            atom_ctrl.auxbas_path = mol.ctrl.auxbas_path.clone();
            atom_ctrl.auxbas_type = mol.ctrl.auxbas_type.clone();
            atom_ctrl.eri_type = String::from("ri_v");
            atom_ctrl.num_threads = Some(mol.ctrl.num_threads.unwrap());
            atom_ctrl.mixer = "diis".to_string();
            atom_ctrl.initial_guess = "vsap".to_string();
            atom_ctrl.print_level = 1;
            atom_ctrl.atom_sad = true;
            atom_ctrl.charge = 0.0_f64;
            let (spin, spin_channel, spin_polarization) = ctrl_setting_atom_sad(ielem);
            atom_ctrl.spin = spin;
            atom_ctrl.spin_channel = spin_channel;
            atom_ctrl.spin_polarization = spin_polarization;
            //atom_ctrl.spin = 1.0;
            //atom_ctrl.spin_channel = 1;
            //atom_ctrl.spin_polarization = false;
            let mut atom_geom = GeomCell::init_geom();
            atom_geom.name = ielem.to_string();
            atom_geom.position = MatrixFull::from_vec([3,1], vec![0.000,0.000,0.000]).unwrap();
            atom_geom.elem = vec![ielem.to_string()];

            let mut atom_mol = Molecule::build_native(atom_ctrl,atom_geom).unwrap();

            let mut atom_scf = scf(atom_mol).unwrap();

            //println!("debug: elem prepared in this loop: {}, size: {:?}", ielem, atom_scf.density_matrix[0].size());


            let mut dms: Vec<MatrixFull<f64>> = vec![];

            if atom_scf.mol.spin_channel == 1 {
                let dm = atom_scf.density_matrix[0].clone()*0.5;
                dms.push(dm.clone());
                dms.push(dm);
            } else {
                dms.push(atom_scf.density_matrix[0].clone());
                dms.push(atom_scf.density_matrix[1].clone());
            }

            atom_dms.insert(ielem.clone(),dms);
        }
    });

    // reset the omp_num_threads to be the correct one
    utilities::omp_set_num_threads_wrapper(mol.ctrl.num_threads.unwrap());

    let (dms_alpha, dms_beta) = block_diag_specific(&atom_dms, &mol.geom.elem);

    if mol.spin_channel == 1 {
        vec![dms_alpha+dms_beta, MatrixFull::empty()]
    } else if mol.spin_channel == 2 {
        vec![dms_alpha, dms_beta]
    } else {
        vec![]
    }

    //if mol.spin_channel == 1{
    //    vec![block_diag(&dms_alpha)+block_diag(&dms_beta), MatrixFull::empty()]
    //} else {
    //    vec![block_diag(&dms_alpha), block_diag(&dms_beta)]
    //}


}

pub fn block_diag_specific(atom_dms: &HashMap<String,Vec<MatrixFull<f64>>>,elem: &Vec<String>) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let mut atom_size = 0;
    elem.iter().for_each(|ielem| {
        if let Some(dm) = &atom_dms.get(ielem) {
            atom_size += &dm[0].size[0];
        } else {
            panic!("Error: Unknown elemement ({}), for which the density matrix is not yet prepared", ielem);
        }
    });
    let mut dms_alpha = MatrixFull::new([atom_size;2], 0.0);
    let mut dms_beta = MatrixFull::new([atom_size;2], 0.0);
    //let dms_alpha = dms.get_mut(0).unwrap();
    //let dms_beta = dms.get_mut(1).unwrap();
    let mut ao_index = 0;
    elem.iter().for_each(|ielem| {
        if let Some(dm_atom_vec) = &atom_dms.get(ielem) {
            let dm_atom_alpha = dm_atom_vec.get(0).unwrap();
            let dm_atom_beta  = dm_atom_vec.get(1).unwrap();
            let loc_length = dm_atom_alpha.size[0];

            dms_alpha.copy_from_matr(ao_index..ao_index+loc_length, ao_index..ao_index+loc_length,
                dm_atom_alpha, 0..loc_length, 0..loc_length);
            dms_beta.copy_from_matr(ao_index..ao_index+loc_length, ao_index..ao_index+loc_length,
                dm_atom_beta, 0..loc_length, 0..loc_length);
            ao_index += loc_length;
        }
    });

    (dms_alpha, dms_beta)
}

pub fn block_diag(dms: &Vec<MatrixFull<f64>>) -> MatrixFull<f64>{
    //scipy.linalg.block_diag()
    let mut atom_size = 0;
    dms.iter().for_each(|dm_atom|{
        atom_size += dm_atom.size[0];
    });
    let mut dm = MatrixFull::new([atom_size;2], 0.0);
    let mut ao_index = 0;
    dms.iter().for_each(|dm_atom|{
        dm_atom.iter_columns_full().zip(dm.iter_columns_mut(ao_index..ao_index+dm_atom.size[0]))
            .for_each(|(x,y)|{
                for i in (0..dm_atom.size[0]){
                    y[ao_index+i] = x[i];
                }

            });
        ao_index += dm_atom.size[0];
    });
    dm
}

pub fn ctrl_setting_atom_sad(elem: &String) -> (f64,usize,bool) {
    match &formated_element_name(elem)[..] {
        "H" | "Li" | "Na" | "K"  | "Rb" | "Cs" | "Fr" => (2.0, 2, true),
        "B" | "Al" | "Ga" | "In" | "Tl" | "Nh" => (1.0, 1, false),
        "C" | "Si" | "Ge" | "Sn" | "Pb" | "Fl" => (1.0, 1, false),
        "N" | "P"  | "As" | "Sb" | "Bi" | "Mc" => (1.0, 1, false),
        "O" | "S"  | "Se" | "Te" | "Po" | "Lv" => (1.0, 1, false),
        "F" | "Cl" | "Br" | "I"  | "At" | "Ts" => (1.0, 1, false),
        _ => (1.0,1,false)
    }
}

pub fn cut_ecp_occ(occ: Vec<f64>, num_ecp: usize) -> Vec<f64> {
    occ[num_ecp/2..].iter().map(|x| *x).collect::<Vec<f64>>()
}

pub fn generate_occupation(elem: &String,num_basis: usize, num_ecp: usize) -> ([Vec<f64>;2],[usize;2],[usize;2]) {
    generate_occupation_v01(elem,num_basis, num_ecp)
}

pub fn generate_occupation_v02(elem: &String,num_basis: usize, num_ecp: usize) -> ([Vec<f64>;2],[usize;2],[usize;2]) {
    let frozen_orb = num_ecp/2;
    let num_elec = SPECIES_INFO.get(&elem.as_str()).unwrap().1;
    let mut occ_o: Vec<f64> = vec![];
    let mut rest_elem = num_elec;
    NELE_IN_SHELLS.iter().for_each(|x| {
        let num_orbs = (x/2.0) as usize;
        if rest_elem >= *x {
            occ_o.extend(vec![2.0; num_orbs]);
            rest_elem -= x;
        } else if rest_elem < *x && rest_elem >= 1.0 {
            let num_elec_per_orb = rest_elem / (num_orbs as f64);
            occ_o.extend(vec![num_elec_per_orb; num_orbs]);
            rest_elem = -1.0;
        }
    });
    let num_occ_o = occ_o.len();
    let mut occ_a = cut_ecp_occ(occ_o,num_ecp);
    let num_occ_a = occ_a.len();
    occ_a.extend(vec![0.0;frozen_orb+num_basis-num_occ_o]);
    println!("{:?}", &occ_a);
    ([occ_a, vec![0.0;num_occ_a]], [num_occ_a-1, 0], [num_occ_a, 0])
}

pub fn generate_occupation_v01(elem: &String,num_basis: usize, num_ecp: usize) -> ([Vec<f64>;2],[usize;2],[usize;2]) {
    let frozen_orb = num_ecp/2;
    match &formated_element_name(elem)[..] {
        "H" => {
            let mut occ_a = cut_ecp_occ(vec![1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-1]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[0-frozen_orb,0],[1-frozen_orb,0])
        },
        "Li" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-2]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[1-frozen_orb,0],[2-frozen_orb,0])
        },
        "Na" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-6]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[5-frozen_orb,0],[6-frozen_orb,0])
        },
        "K" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-10]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[9-frozen_orb,0],[10-frozen_orb,0])
        },

        "Be" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-2]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[1-frozen_orb,0],[2-frozen_orb,0])
        },
        "Mg" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-6]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[5-frozen_orb,0],[6-frozen_orb,0])
        },
        "Ca" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-10]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[9-frozen_orb,0],[10-frozen_orb,0])
        },
        "Sr" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-19]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[18-frozen_orb,0],[19-frozen_orb,0])
        },

        "Sc" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.2, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Ti" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.4, 0.4, 0.4, 0.4, 0.4, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "V" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.6, 0.6, 0.6, 0.6, 0.6, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Cr" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Mn" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Fe" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.2, 1.2, 1.2, 1.2, 1.2, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Co" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.4, 1.4, 1.4, 1.4, 1.4, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Ni" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.6, 1.6, 1.6, 1.6, 1.6, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Cu" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },
        "Zn" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-15]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[14-frozen_orb,0],[15-frozen_orb,0])
        },

        "Y" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.2, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Zr" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.4, 0.4, 0.4, 0.4, 0.4, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Nb" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Mo" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Tc" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Ru" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.4, 1.4, 1.4, 1.4, 1.4, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Rh" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.6, 1.6, 1.6, 1.6, 1.6, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Pd" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Ag" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },
        "Cd" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-24]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[23-frozen_orb,0],[24-frozen_orb,0])
        },

        "B" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 0.333333333, 0.33333333, 0.333333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "Al" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.333333333, 0.33333333, 0.333333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "Ga" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.333333333, 0.33333333, 0.333333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        "In" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.333333333, 0.33333333, 0.333333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-27]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[26-frozen_orb,0],[27-frozen_orb,0])
        },
        "C" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 0.666666667, 0.666666667, 0.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "Si" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.666666667, 0.666666667, 0.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "Ge" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.666666667, 0.666666667, 0.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        "Sn" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.666666667, 0.666666667, 0.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-27]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[26-frozen_orb,0],[27-frozen_orb,0])
        },
        "N" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "P" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "As" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        "Sb" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-27]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[26-frozen_orb,0],[27-frozen_orb,0])
        },
        "O" => {
            let mut occ_a = cut_ecp_occ(vec![2.0,2.0,1.33333333,1.33333333,1.33333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            //occ_b.extend(vec![0.0;frozen_orb+num_basis-5]);
            //([occ_a,occ_b],[4,2],[5,3])
            ([occ_a,vec![]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "S" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.33333333,1.33333333,1.33333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "Se" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.33333333,1.33333333,1.33333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        "Te" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.33333333,1.33333333,1.33333333],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-27]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[26-frozen_orb,0],[27-frozen_orb,0])
        },
        "F" => {
            let mut occ_a = cut_ecp_occ(vec![2.0,2.0,1.666666667,1.666666667,1.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            //occ_b.extend(vec![0.0;frozen_orb+num_basis-5]);
            //([occ_a,occ_b],[4,2],[5,3])
            ([occ_a,vec![]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "Cl" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.666666667,1.666666667,1.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "Br" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.666666667,1.666666667,1.666666667],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        "I" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.666666667,1.666666667,1.666666667],num_ecp);
            //println!("{}, {}, {:?}", num_basis, frozen_orb, &occ_a);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-27]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[26-frozen_orb,0],[27-frozen_orb,0])
        },
        "He" => {
            let mut occ_a = cut_ecp_occ(vec![2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-1]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[0-frozen_orb,0],[1-frozen_orb,0])
        },
        "Ne" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-5]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[4-frozen_orb,0],[5-frozen_orb,0])
        },
        "Ar" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-9]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[8-frozen_orb,0],[9-frozen_orb,0])
        },
        "Kr" => {
            let mut occ_a = cut_ecp_occ(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],num_ecp);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-18]);
            ([occ_a,vec![0.0;frozen_orb+num_basis]],[17-frozen_orb,0],[18-frozen_orb,0])
        },
        // for 5d elements
        "Os" => {
            let mut occ_o: Vec<f64> = vec![];
            occ_o.extend_from_slice(&XE_SHELL);
            occ_o.extend_from_slice(&F_SHELL);
            occ_o.extend_from_slice(&[1.6;5]);
            let num_occ_o = occ_o.len();
            println!("{:?}", &occ_o);
            let mut occ_a = cut_ecp_occ(occ_o,num_ecp);
            let num_occ_a = occ_a.len();
            println!("{:?}", &occ_a);
            occ_a.extend(vec![0.0;frozen_orb+num_basis-num_occ_o]);
            ([occ_a, vec![0.0;num_occ_a]], [num_occ_a-1, 0], [num_occ_a, 0])
        },
        
        _ => ([vec![],vec![]],[0,0],[0,0])
    }
}