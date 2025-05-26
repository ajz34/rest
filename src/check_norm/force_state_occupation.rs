use std::collections::HashMap;

use num_complex::ComplexFloat;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tensors::{matrix_blas_lapack::_dgemv, BasicMatrix, MatrixFull, MatrixUpper};

use crate::{molecule_io::Molecule, scf_io::{SCFType, SCF}};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ForceStateOccupation {
    //          file_index, global_start_index, num_basis, num_state, spin_channel
    ref_index: String,
    prev_state: usize,
    prev_spin: usize,
    target_spin: usize,
    force_occ: f64,
    curr_state: usize,
    force_check_min: usize,
    force_check_max: usize,
    ovlp: f64,
}

impl ForceStateOccupation {
    pub fn init(ref_index: String,
                prev_state: usize, 
                prev_spin: usize,
                target_spin: usize,
                force_occ: f64, 
                force_check_min: usize,
                force_check_max: usize) -> ForceStateOccupation {
        ForceStateOccupation {
            // only the file_index is specified with this initialization
            ref_index,
            prev_state,
            prev_spin,
            target_spin,
            force_occ,
            force_check_min,
            force_check_max,
            curr_state: 0,
            ovlp: 0.0,
        }
    }

    //pub fn set_ref_index(&mut self, global_start: usize, num_basis: usize, num_state: usize, spin_channel: usize) {
    //    self.ref_index.1[0] = global_start;
    //    self.ref_index.1[1] = num_basis;
    //    self.ref_index.1[2] = num_state;
    //    self.ref_index.1[3] = spin_channel;

    //}

    pub fn get_force_occ(&self) -> f64 {
        self.force_occ
    }
    pub fn get_occ_spin(&self) -> usize {
        self.prev_spin
    }

    pub fn get_check_window(&self) -> [usize;2] {
        [self.force_check_min,self.force_check_max]
    }

    pub fn formated_output(&self) -> String {
        let mut output = String::new();
        output = format!(" Prev. State: {}; Prev. Spin: {}; Target Spin: {}; Force Occ. {:16.8}\n", self.prev_state, self.prev_spin, self.target_spin, self.force_occ);
        output = format!("{} State Window: ({}, {})\n", output, self.force_check_min, self.force_check_max);
        output = format!("{} Curr. State: {} with OVLP of {:16.8}", output, self.curr_state, self.ovlp);

        output
    }

    pub fn formated_output_check(&self) -> String {
        let mut output = String::new();
        output = format!(" Prev. State: {}; Prev. Spin: {}; Target Spin: {}; Force Occ. {:16.8}\n", self.prev_state, self.prev_spin, self.target_spin, self.force_occ);
        output = format!("{} State Window: ({}, {})\n", output, self.force_check_min, self.force_check_max);

        output
    }
}

pub fn adapt_occupation_with_force_projection(
    occupation: &mut [Vec<f64>;2], 
    homo: &mut [usize;2],
    lumo: &mut [usize;2],
    force_occ: &mut Vec<ForceStateOccupation>,
    scftype: &SCFType, 
    eigenvectors: &[MatrixFull<f64>;2],
    ovlp: &MatrixUpper<f64>,
    ref_eigenvectors: &HashMap<String,([MatrixFull<f64>;2],[usize;4])>
) {

    determine_force_projection_v02(force_occ, eigenvectors, ovlp, ref_eigenvectors);

    force_occ.iter().for_each(|x| println!("{}", x.formated_output()));

    match scftype {
        SCFType::RHF => {
            let mut occ_s = occupation.get_mut(0).unwrap();
            let num_elec = occ_s.iter().sum::<f64>();
            let mut force_curr = vec![];

            let mut net_force_elec = force_occ.iter()
                .filter(|x| x.prev_spin==0)
                .map(|x| {
                    let original_occ = occ_s[x.curr_state];
                    occ_s[x.curr_state] = x.force_occ;
                    force_curr.push(x.curr_state);
                    original_occ - x.force_occ
                }).sum::<f64>();

            if net_force_elec > 1.0e-6 {
                occ_s.iter_mut().enumerate().for_each(|(i,x)| {
                    if *x<2.0 && net_force_elec > 1.0e-6 {
                        if ! force_curr.iter().fold(false, |check, j| check || i==*j) {
                            let tmp_xa = *x + net_force_elec;
                            let tmp_xb = 2.0 - *x;
                            *x = if tmp_xa <=2.0 {
                                net_force_elec = 0.0;
                                tmp_xa
                            } else {
                                net_force_elec -= tmp_xb;
                                2.0
                            }
                        };
                    }
                });
            } else if net_force_elec < -1.0e-6 {
                occ_s.iter_mut().enumerate().rev().for_each(|(i, x)| {
                    if *x>0.0 && net_force_elec < -1.0e-6 {
                        if ! force_curr.iter().fold(false, |check, j| check || i==*j) {
                            let tmp_xa = *x + net_force_elec;
                            let tmp_xb = *x;
                            *x = if tmp_xa >= 0.0 {
                                net_force_elec = 0.0;
                                tmp_xa
                            } else {
                                net_force_elec += tmp_xb;
                                0.0
                            }
                        }
                    }

                });
            }
            let mut i_homo = homo.get_mut(0).unwrap();
            *i_homo = occ_s.iter().enumerate()
                .filter(|(i,occ)| **occ >=1.0e-6)
                .map(|(i,occ)| i).max().unwrap();
            let mut i_lumo = lumo.get_mut(0).unwrap();
            //*i_lumo = *i_homo + 1;
            *i_lumo = occ_s.iter().enumerate()
                .filter(|(i,occ)| (**occ-2.0).abs() >=1.0e-6 )
                .map(|(i,occ)| i).min().unwrap();
        },
        _ => {
            for i_spin in (0..2) {
                let mut occ_s = occupation.get_mut(i_spin).unwrap();
                let num_elec = occ_s.iter().sum::<f64>();
                let mut force_curr = vec![];

                let mut net_force_elec = force_occ.iter()
                    .filter(|x| x.target_spin==i_spin)
                    .map(|x| {
                        let original_occ = occ_s[x.curr_state];
                        occ_s[x.curr_state] = x.force_occ;
                        force_curr.push(x.curr_state);
                        original_occ - x.force_occ
                    }).sum::<f64>();

                if net_force_elec > 1.0e-6 {
                    occ_s.iter_mut().enumerate().for_each(|(i,x)| {
                        if *x<1.0 && net_force_elec > 1.0e-6 {
                            if ! force_curr.iter().fold(false, |check, j| check || i==*j) {
                                let tmp_xa = *x + net_force_elec;
                                let tmp_xb = 1.0 - *x;
                                *x = if tmp_xa <=1.0 {
                                    net_force_elec = 0.0;
                                    tmp_xa
                                } else {
                                    net_force_elec -= tmp_xb;
                                    1.0
                                }
                            };
                        }
                    });
                } else if net_force_elec < -1.0e-6 {
                    occ_s.iter_mut().enumerate().rev().for_each(|(i, x)| {
                        if *x>0.0 && net_force_elec < -1.0e-6 {
                            if ! force_curr.iter().fold(false, |check, j| check || i==*j) {
                                let tmp_xa = *x + net_force_elec;
                                let tmp_xb = *x;
                                *x = if tmp_xa >= 0.0 {
                                    net_force_elec = 0.0;
                                    tmp_xa
                                } else {
                                    net_force_elec += tmp_xb;
                                    0.0
                                }
                            }
                        }

                    });
                }

                let mut i_homo = homo.get_mut(i_spin).unwrap();
                *i_homo = occ_s.iter().enumerate()
                    .filter(|(i,occ)| **occ >=1.0e-6)
                    .map(|(i,occ)| i).max().unwrap();
                let mut i_lumo = lumo.get_mut(i_spin).unwrap();
                //*i_lumo = *i_homo + 1;
                *i_lumo = occ_s.iter().enumerate()
                    .filter(|(i,occ)| (**occ-1.0).abs() >=1.0e-6 )
                    .map(|(i,occ)| i).min().unwrap();
            }
        },
    }


}

pub fn determine_force_projection(
    force_occ: &mut Vec<ForceStateOccupation>,
    eigenvectors: &[MatrixFull<f64>;2],
    ovlp: &MatrixUpper<f64>,
    ref_eigenvectors: &[MatrixFull<f64>;2]) {

        let ovlp_full = ovlp.to_matrixfull().unwrap();

        force_occ.iter_mut().enumerate().for_each(|(i_force_occ,i_obj)| {
            let p_spin = i_obj.prev_spin;
            let p_state = i_obj.prev_state;
            let c_start = i_obj.force_check_min;

            i_obj.ovlp = 0.0;
            i_obj.curr_state = 0;

            let prev_eigenvector = &ref_eigenvectors[p_spin][(..,p_state)];
            let r_states = (i_obj.force_check_min..i_obj.force_check_max);
            eigenvectors[p_spin].iter_columns(r_states).enumerate().for_each(|(il,curr_eigenvector)| {
                let cur_state = il + c_start;
                let mut tmp_vec = vec![0.0;curr_eigenvector.len()];
                _dgemv(&ovlp_full,curr_eigenvector,&mut tmp_vec,'n',1.0,0.0,1,1);

                let temp_value  = tmp_vec.par_iter().zip(prev_eigenvector.par_iter())
                    .fold(|| 0.0,|acc, (a,b)| acc + a*b).sum::<f64>();

                if temp_value.abs()>i_obj.ovlp {
                    i_obj.curr_state = cur_state;
                    i_obj.ovlp = temp_value.abs();
                }
            });
            if i_obj.ovlp < 1.0e-1 {
                println!("WARNNING: MOM overlap is less than 1.0e-1, indicating that the orbital window should be enlarge");
            }
        });

}

pub fn determine_force_projection_v02(
    force_occ: &mut Vec<ForceStateOccupation>,
    eigenvectors: &[MatrixFull<f64>;2],
    ovlp: &MatrixUpper<f64>,
    ref_eigenvectors_map: &HashMap<String,([MatrixFull<f64>;2],[usize;4])>
) {

        let ovlp_full = ovlp.to_matrixfull().unwrap();



        force_occ.iter_mut().enumerate().for_each(|(i_force_occ,i_obj)| {

            //println!("{}", i_obj.ref_index);
            //println!("{:?}", ref_eigenvectors_map);
            let (ref_eigenvectors, 
                [global_index, num_basis, num_state, spin_channel]) 
                = ref_eigenvectors_map.get(&i_obj.ref_index).unwrap();

            let p_spin = i_obj.prev_spin;
            let t_spin = i_obj.target_spin;
            let p_state = i_obj.prev_state;
            let c_start = i_obj.force_check_min;

            i_obj.ovlp = 0.0;
            i_obj.curr_state = 0;

            //let prev_eigenvector = &ref_eigenvectors[p_spin][(..,p_state)];
            let mut prev_eigenvector = vec![0.0;eigenvectors[0].size()[0]];
            prev_eigenvector[*global_index..*global_index+num_basis].iter_mut().zip(ref_eigenvectors[p_spin][(..,p_state)].iter()).for_each(|(to, from)| {
                *to = *from
            });

            let r_states = (i_obj.force_check_min..i_obj.force_check_max);
            eigenvectors[t_spin].iter_columns(r_states).enumerate().for_each(|(il,curr_eigenvector)| {
                let cur_state = il + c_start;
                let mut tmp_vec = vec![0.0;curr_eigenvector.len()];
                _dgemv(&ovlp_full,curr_eigenvector,&mut tmp_vec,'n',1.0,0.0,1,1);

                let temp_value  = tmp_vec.par_iter().zip(prev_eigenvector.par_iter())
                    .fold(|| 0.0,|acc, (a,b)| acc + a*b).sum::<f64>();

                if temp_value.abs()>i_obj.ovlp {
                    i_obj.curr_state = cur_state;
                    i_obj.ovlp = temp_value.abs();
                }
            });
            if i_obj.ovlp < 1.0e-1 {
                println!("WARNNING: MOM overlap is less than 1.0e-1, indicating that the orbital window should be enlarge");
            }
        });

}

#[test]
fn test() {
    let dd = MatrixFull::from_vec([5,5], vec![1.0;25]).unwrap();

    dd.iter_columns((2..5)).enumerate().rev().for_each(|(x,y)| {
        println!("{:?}, {:?}",x, y);
    });
}