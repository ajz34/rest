use std::{sync::mpsc::channel, num};
use libc::ITIMER_VIRTUAL;
use libm::erfc;
use mpi::collective::SystemOperation;
use num_traits::abs;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use tensors::{MatrixFull,matrix_blas_lapack::_dgemm, TensorOpt, BasicMatrix};

use crate::{mpi_io::{mpi_broadcast, mpi_broadcast_matrixfull, mpi_broadcast_vector, mpi_reduce, MPIData,MPIOperator}, utilities};

pub fn close_shell_sbge2_detailed_rayon(scf_data: &crate::scf_io::SCF) -> anyhow::Result<([f64;3],[MatrixFull<(f64,f64)>;3])> {
//pub fn close_shell_sbge2_rayon(scf_data: &crate::scf_io::SCF) -> anyhow::Result<[f64;3]> {

    let enhanced_factor = 1.0;
    let screening_factor = 1.0;
    let shifted_factor = 0.0;
    //let mut tmp_record = TimeRecords::new();

    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    let num_auxbas = scf_data.mol.num_auxbas;
    let num_basis = scf_data.mol.num_basis;
    let num_state = scf_data.mol.num_state;
    let homo = scf_data.homo.get(0).unwrap().clone();
    let lumo = scf_data.lumo.get(0).unwrap().clone();
    let start_mo: usize = scf_data.mol.start_mo;
    //let num_occu = homo + 1;
    //let num_occu = lumo;
    let num_occu = if scf_data.mol.num_elec[0] <= 1.0e-6 {0} else {homo + 1};

    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;
    let mut e_bge2_ss = 0.0_f64;
    let mut e_bge2_os = 0.0_f64;
    let mut eij_00 = MatrixFull::new([num_occu,num_occu], (0.0_f64,0.0_f64));
    let mut eij_01 = MatrixFull::new([num_occu,num_occu], (0.0_f64,0.0_f64));
    let mut eij_11 = MatrixFull::new([num_occu,num_occu], (0.0_f64,0.0_f64));

    if let Some(ri3mo_vec) = &scf_data.ri3mo {
        let (rimo, vir_range, occ_range) = &ri3mo_vec[0];

        let mut e_ij = MatrixFull::new([num_occu,num_occu], (0.0_f64,0.0_f64,0.0_f64,0.0_f64,0_usize));

        let lumo_min = vir_range.start;

        let eigenvector = scf_data.eigenvectors.get(0).unwrap();
        let eigenvalues = scf_data.eigenvalues.get(0).unwrap();
        let occupation = scf_data.occupation.get(0).unwrap();

        //tmp_record.new_item("dgemm", "prepare four-center integrals from RI-MO");
        //tmp_record.new_item("get2d", "get the ERI values");
        let mut elec_pair: Vec<[usize;2]> = vec![];
        for i_state in start_mo..num_occu {
            for j_state in i_state..num_occu {
                elec_pair.push([i_state,j_state])
            }
        };
        let (sender, receiver) = channel();
        elec_pair.par_iter().for_each_with(sender,|s,i_pair| {
            let mut e_mp2_term_ss = 0.0_f64;
            let mut e_mp2_term_os = 0.0_f64;

            let mut num_eij_iter = 0_usize;
            let mut e_eij_term_ss = 0.0_f64;
            let mut e_eij_term_os = 0.0_f64;

            let i_state = i_pair[0];
            let j_state = i_pair[1];
            let i_state_eigen = eigenvalues[i_state];
            let j_state_eigen = eigenvalues[j_state];
            let i_state_occ = occupation[i_state]/2.0;
            let j_state_occ = occupation[j_state]/2.0;
            let ij_state_eigen = i_state_eigen + j_state_eigen;

            if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {

                // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                // the indices in rimo are shifted
                let i_loc_state = i_state-occ_range.start;
                let j_loc_state = j_state-occ_range.start;
                let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                _dgemm(
                    &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                    &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                    &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                    1.0,0.0);
                let mut denominator = MatrixFull::new([vir_range.len(),vir_range.len()],[0.0_f64;2]);

                for i_virt in lumo..num_state {
                    let i_virt_eigen = eigenvalues[i_virt];
                    let i_virt_occ = occupation[i_virt]/2.0;
                    if (1.0 - i_virt_occ).abs() > 1.0e-6 {
                        for j_virt in lumo..num_state {

                            let j_virt_eigen = eigenvalues[j_virt];
                            let j_virt_occ = occupation[j_virt]/2.0;

                            if (1.0-j_virt_occ).abs() > 1.0e-6 {
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                                let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                if double_gap.abs()<=1.0E-6 {
                                    println!("Warning: too close to degeneracy");
                                    double_gap = 1.0E-6;
                                };

                                // consider fractional occupation
                                //double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));
                                let occ_state = (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                // the indices in rimo are shifted
                                let i_loc_virt = i_virt-vir_range.start;
                                let j_loc_virt = j_virt-vir_range.start;

                                let e_mp2_a = eri_virt[[i_loc_virt,j_loc_virt]];
                                let e_mp2_b = eri_virt[[j_loc_virt,i_loc_virt]];
                                e_mp2_term_ss += (e_mp2_a - e_mp2_b) * e_mp2_a / double_gap * occ_state;
                                e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap * occ_state; 

                                denominator[[i_loc_virt,j_loc_virt]] = [double_gap, occ_state];
                            }

                        }
                    }
                }

                //println!("{debug: {:?}, {:?}", e_mp2_ss, e_mp2_os);
                (e_eij_term_ss, e_eij_term_os,num_eij_iter) = iterator_close_shell_eij_serial(&eri_virt, &denominator, 
                    e_mp2_term_ss, e_mp2_term_os, 
                    enhanced_factor, screening_factor, shifted_factor, 
                    lumo, lumo_min, num_state);

                if i_state != j_state {
                    e_mp2_term_ss *= 2.0;
                    e_mp2_term_os *= 2.0;
                    e_eij_term_ss *= 2.0;
                    e_eij_term_os *= 2.0;
                };
            }

            s.send((e_mp2_term_os, e_mp2_term_ss, e_eij_term_os, e_eij_term_ss, num_eij_iter, i_state, j_state)).unwrap()

        });
        receiver.into_iter().for_each(|(e_mp2_term_os,e_mp2_term_ss, e_eij_term_os, e_eij_term_ss, num_eij_iter, i_state, j_state)| {
            e_mp2_os -= e_mp2_term_os;
            e_mp2_ss -= e_mp2_term_ss;
            e_bge2_os -= e_eij_term_os;
            e_bge2_ss -= e_eij_term_ss;
            e_ij[(i_state,j_state)] = (-e_mp2_term_os,-e_mp2_term_ss, -e_eij_term_os, -e_eij_term_ss,num_eij_iter)
        });
        //if scf_data.mol.ctrl.print_level>0 {
        //    for i_state in start_mo..num_occu {
        //        for j_state in i_state..num_occu {
        //            let (e_mp2_term_os, e_mp2_term_ss, e_eij_term_os, e_eij_term_ss, num_eij_iter) = e_ij[(i_state, j_state)];
        //            println!("Print the electron-pair energy of ({:3},{:3}): PT2=({:16.8},{:16.8}), sBGE2=({:16.8},{:16.8})", i_state,j_state, e_mp2_term_os, e_mp2_term_ss, e_eij_term_os, e_eij_term_ss);
        //        }
        //    };
        //}
        //let num_occ = scf_data.homo[0]+1;
        //let num_occ = scf_data.lumo[0];
        let num_occ = if scf_data.mol.num_elec[0] <= 1.0e-6 {0} else {scf_data.homo[0] + 1};
        if scf_data.mol.ctrl.print_level>0 {
            println!("For (alpha, alpha)");
            println!("Print the correlation energies for each electron-pair:");
        };
        for i_state in start_mo..num_occ {
            for j_state in i_state+1..num_occ {
                let (e_mp2_term_os,e_mp2_term_ss, e_eij_term_os, e_eij_term_ss,num_eij_iter) = e_ij[(i_state, j_state)];
                if scf_data.mol.ctrl.print_level>1 {
                    println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term_ss/2.0, e_eij_term_ss/2.0);
                }
                eij_00[(i_state,j_state)] = (e_mp2_term_ss/2.0,e_eij_term_ss/2.0);
            }
        };
        if scf_data.mol.ctrl.print_level>1 {println!("For (beta, beta)")};
        for i_state in start_mo..num_occ {
            for j_state in i_state+1..num_occ {
                //let (e_mp2_term, e_eij_term) = eij_11[(i_state, j_state)];
                //println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term, e_eij_term);
                let (e_mp2_term_os,e_mp2_term_ss, e_eij_term_os, e_eij_term_ss,num_eij_iter) = e_ij[(i_state, j_state)];
                if scf_data.mol.ctrl.print_level>1 {
                    println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term_ss/2.0, e_eij_term_ss/2.0);
                }
                eij_11[(i_state,j_state)] = (e_mp2_term_ss/2.0,e_eij_term_ss/2.0);
            }
        }
        if scf_data.mol.ctrl.print_level>1 {println!("For (alpha, beta)");}
        for i_state in start_mo..num_occ {
            for j_state in start_mo..num_occ {
                //let (e_mp2_term, e_eij_term) = eij_01[(i_state, j_state)];
                if i_state<j_state {
                    let (e_mp2_term_os,e_mp2_term_ss, e_eij_term_os, e_eij_term_ss,num_eij_iter) = e_ij[(i_state, j_state)];
                    if scf_data.mol.ctrl.print_level>1 {
                        println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term_os/2.0, e_eij_term_os/2.0);
                    } 
                    eij_01[(i_state,j_state)] = (e_mp2_term_os/2.0,e_eij_term_os/2.0);
                } else if i_state==j_state {
                    let (e_mp2_term_os,e_mp2_term_ss, e_eij_term_os, e_eij_term_ss,num_eij_iter) = e_ij[(i_state, j_state)];
                    if scf_data.mol.ctrl.print_level>1 {
                        println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term_os, e_eij_term_os);
                    }
                    eij_01[(i_state,j_state)] = (e_mp2_term_os,e_eij_term_os);
                } else if i_state>j_state {
                    let (e_mp2_term_os,e_mp2_term_ss, e_eij_term_os, e_eij_term_ss,num_eij_iter) = e_ij[(j_state, i_state)];
                    if scf_data.mol.ctrl.print_level>1 {
                        println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term_os/2.0, e_eij_term_os/2.0);
                    }
                    eij_01[(i_state,j_state)] = (e_mp2_term_os/2.0,e_eij_term_os/2.0);
                }
            }
        }
    } else {
        panic!("RI3MO should be initialized before the PT2 calculations")
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    //println!("debug pt2 and sbge2: {:?}, {:?}", e_mp2_os+e_mp2_ss, e_bge2_ss+ e_bge2_os);

    //tmp_record.report_all();
    Ok(([e_bge2_ss+e_bge2_os,e_bge2_os,e_bge2_ss],[eij_00,eij_01,eij_11]))

}

pub fn close_shell_sbge2_rayon(scf_data: &crate::scf_io::SCF) -> anyhow::Result<[f64;3]> {
    let ([e_bge2_tot,e_bge2_os,e_bge2_ss],_) = 
        close_shell_sbge2_detailed_rayon(scf_data).unwrap();
    Ok([e_bge2_tot,e_bge2_os,e_bge2_ss])
}

pub fn iterator_close_shell_eij_serial(
    eri_virt: &MatrixFull<f64>, 
    denominator: &MatrixFull<[f64;2]>, 
    init_eij_ss: f64,
    init_eij_os: f64,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
    lumo: usize,
    lumo_min: usize, num_state: usize) -> (f64,f64,usize)
{
    let threshold_eij = 1.0E-8;
    let max_iteration = 100_usize;
    let mut delta_eij = [1.0,1.0];
    let mut num_iter = 0_usize;

    let mut eij_ss = 0.0;
    let mut eij_os = 0.0;
    let mut previous_eij_ss = init_eij_ss;
    let mut previous_eij_os = init_eij_os;

    //println!("Electron-pair coupling iterations start for sBGE2:");
    //println!("Input Eij[ss]: {:16.8}, Eij[os]: {:16.8}", init_eij_ss,init_eij_os);

    while ! (abs(delta_eij[1])<threshold_eij || num_iter >= max_iteration) {
        (eij_ss, eij_os) = close_shell_eij_serial(
            eri_virt, 
            denominator, 
            previous_eij_ss, 
            previous_eij_os, 
            enhanced_factor, 
            screening_factor, 
            shifted_factor, 
            lumo, 
            lumo_min, 
            num_state);
        delta_eij[0] = delta_eij[1];
        delta_eij[1] = (eij_ss-previous_eij_ss)+(eij_os-previous_eij_os);
        if delta_eij[0]*delta_eij[1]<0.0 {
            previous_eij_os = (eij_os + previous_eij_os)*0.5;
            previous_eij_ss = (eij_ss + previous_eij_ss)*0.5;
        } else {
            previous_eij_ss = eij_ss;
            previous_eij_os = eij_os;
        }
        //println!("Eij[ss]: {:16.8}, Eij[os]: {:16.8} on {:3}th step", eij_ss,eij_os, num_iter);
        num_iter += 1;
    }

    (eij_ss,eij_os,num_iter)

}

pub fn close_shell_eij_serial(
    eri_virt: &MatrixFull<f64>, 
    denominator: &MatrixFull<[f64;2]>, 
    previous_eij_ss: f64,
    previous_eij_os: f64,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
    lumo: usize,
    lumo_min: usize, num_state: usize) -> (f64,f64)
{
    let mut tmp_energy = 0.0f64;
    let mut eij_ss = 0.0f64;
    let mut eij_os = 0.0f64;
    if screening_factor.abs().lt(&1.0E-6) {
        for i_virt in lumo..num_state {
            for j_virt in lumo..num_state {
                let i_loc_virt = i_virt - lumo_min;
                let j_loc_virt = j_virt - lumo_min;
                let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                let [dij,occ_ij] = denominator[[i_loc_virt,j_loc_virt]];
                
                if occ_ij.abs() > 1.0e-6 {
                    let tmp_energy_term_ss = (vij-vji)*vij/(dij + enhanced_factor*previous_eij_ss)*occ_ij;
                    let tmp_energy_term_os = vij.powf(2.0f64)/(dij + enhanced_factor*previous_eij_os)*occ_ij;

                    //tmp_energy += tmp_energy_term_os + tmp_energy_term_ss;
                    eij_ss += tmp_energy_term_ss;
                    eij_os += tmp_energy_term_os;
                }
            }
        }
    } else {
        for i_virt in lumo..num_state {
            for j_virt in lumo..num_state {
                let i_loc_virt = i_virt - lumo_min;
                let j_loc_virt = j_virt - lumo_min;
                let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                let [dij,occ_ij] = denominator[[i_loc_virt,j_loc_virt]];

                if occ_ij.abs() > 1.0e-6 {
                    let screening_enhanced = enhanced_factor*erfc(screening_factor*dij);
                    let tmp_energy_term_ss = (vij-vji)*vij/(dij + screening_enhanced*(previous_eij_ss+shifted_factor))*occ_ij;
                    let tmp_energy_term_os = vij.powf(2.0f64)/(dij + screening_enhanced*(previous_eij_os+shifted_factor))*occ_ij;
                    //tmp_energy += tmp_energy_term_os + tmp_energy_term_ss;
                    eij_ss += tmp_energy_term_ss;
                    eij_os += tmp_energy_term_os;
                }

            }
        }
    }

    (eij_ss, eij_os)
} 

pub fn open_shell_sbge2_detailed_rayon(scf_data: &crate::scf_io::SCF) -> anyhow::Result<([f64;3], [MatrixFull<(f64,f64)>;3])> {
    let enhanced_factor = 1.0;
    let screening_factor = 1.0;
    let shifted_factor = 0.0;

    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;
    let mut e_bge2_ss = 0.0_f64;
    let mut e_bge2_os = 0.0_f64;
    let num_occu_alpha = if scf_data.mol.num_elec[1] <= 1.0e-6 {0} else {scf_data.homo[0] + 1};
    let num_occu_beta = if scf_data.mol.num_elec[2] <= 1.0e-6 {0} else {scf_data.homo[1] + 1};

    //let num_occu_max = scf_data.homo[0].max(scf_data.homo[1])+1;
    let num_occu_max = num_occu_alpha.max(num_occu_beta);

    let mut eij_00 = MatrixFull::new([num_occu_max,num_occu_max], (0.0_f64,0.0_f64));
    let mut eij_01 = MatrixFull::new([num_occu_max,num_occu_max], (0.0_f64,0.0_f64));
    let mut eij_11 = MatrixFull::new([num_occu_max,num_occu_max], (0.0_f64,0.0_f64));

    if let Some(ri3mo_vec) = &scf_data.ri3mo {

        let start_mo: usize = scf_data.mol.start_mo;
        let num_basis = scf_data.mol.num_basis;
        let num_state = scf_data.mol.num_state;
        let num_auxbas = scf_data.mol.num_auxbas;
        let spin_channel = scf_data.mol.spin_channel;
        let i_spin_pair: [(usize,usize);3] = [(0,0),(0,1),(1,1)];


        for (i_spin_1,i_spin_2) in i_spin_pair {
            if i_spin_1 == i_spin_2 {

                let i_spin = i_spin_1;
                let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
                let eigenvalues = scf_data.eigenvalues.get(i_spin).unwrap();
                let occupation = scf_data.occupation.get(i_spin).unwrap();

                let homo = scf_data.homo[i_spin].clone();
                let lumo = scf_data.lumo[i_spin].clone();
                //let num_occu = homo + 1;
                let num_occu = lumo;

                let (rimo, vir_range, occ_range) = &ri3mo_vec[i_spin];
                let lumo_min = vir_range.start;

                //let mut rimo = riao.ao2mo(eigenvector).unwrap();
                let mut elec_pair: Vec<[usize;2]> = vec![];
                for i_state in start_mo..num_occu {
                    for j_state in i_state+1..num_occu {
                        elec_pair.push([i_state,j_state])
                    }
                };
                let (sender, receiver) = channel();
                elec_pair.par_iter().for_each_with(sender,|s,i_pair| {

                    let mut e_mp2_term_ss = 0.0_f64;
                    let mut num_eij_iter = 0_usize;
                    let mut e_eij_term_ss = 0.0_f64;
                    let mut e_eij_term_os = 0.0_f64;

                    let i_state = i_pair[0];
                    let j_state = i_pair[1];
                    let i_state_eigen = eigenvalues[i_state];
                    let j_state_eigen = eigenvalues[j_state];
                    let ij_state_eigen = i_state_eigen + j_state_eigen;
                    let i_state_occ = occupation[i_state];
                    let j_state_occ = occupation[j_state];

                    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {

                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                        // the indices in rimo are shifted
                        let i_loc_state = i_state-occ_range.start;
                        let j_loc_state = j_state-occ_range.start;
                        let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                        let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                        let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                        _dgemm(
                            &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                            &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                            &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                            1.0,0.0);
                        let mut denominator = MatrixFull::new([vir_range.len(),vir_range.len()],[0.0_f64;2]);
                        for i_virt in lumo..num_state {
                            let i_virt_eigen = eigenvalues[i_virt];
                            let i_virt_occ = occupation[i_virt];
                            if (1.0-i_virt_occ).abs() > 1.0e-6 {
                                for j_virt in i_virt+1..num_state {
                                    let j_virt_eigen = eigenvalues[j_virt];
                                    let j_virt_occ = occupation[j_virt];
                                    let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
                                    if (1.0-j_virt_occ).abs() > 1.0e-6 {
                                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                        if double_gap.abs()<=10E-6 {
                                            println!("Warning: too close to degeneracy");
                                            double_gap = 1.0E-6;
                                        }

                                        let occ_state = (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));


                                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                        // the indices in rimo are shifted
                                        let i_loc_virt = i_virt-vir_range.start;
                                        let j_loc_virt = j_virt-vir_range.start;
                                        let e_mp2_a = eri_virt[[i_loc_virt,j_loc_virt]];
                                        let e_mp2_b = eri_virt[[j_loc_virt,i_loc_virt]];
                                        e_mp2_term_ss += (e_mp2_a - e_mp2_b).powf(2.0) / double_gap * occ_state;

                                        denominator[[i_loc_virt,j_loc_virt]] = [double_gap, occ_state];
                                    }
                                }
                            }
                        }
                        //println!("debug, lumo: {}, lumo_min: {}", lumo, lumo_min);
                        //denominator.formated_output(5, "full");

                        (e_eij_term_ss, e_eij_term_os,num_eij_iter) = iterator_open_shell_eij_serial(&eri_virt, &denominator, 
                            e_mp2_term_ss, 0.0, 
                            enhanced_factor, screening_factor, shifted_factor, 
                            lumo,lumo,true,lumo_min, num_state);

                        //println!("debug: i,j=({},{}), mp2_ss={:16.8}, eij_ss={:16.8}", i_state, j_state, e_mp2_term_ss, e_eij_term_ss);
                    }

                    s.send((e_mp2_term_ss,e_eij_term_ss,num_eij_iter,i_state,j_state)).unwrap()
                });

                receiver.into_iter().for_each(|(e_mp2_term_ss, e_eij_term_ss, num_eij_iter,i_state,j_state)| {
                    e_mp2_ss -= e_mp2_term_ss;
                    e_bge2_ss -= e_eij_term_ss;
                    //e_ij[(i_state,j_state)] = (0.0, e_mp2_term_ss,0.0,e_eij_term_ss,num_eij_iter);
                    //e_ij[(i_state,j_state)].1 += e_mp2_term_ss;
                    //e_ij[(i_state,j_state)].3 += e_eij_term_ss;
                    //e_ij[(i_state,j_state)].4 += num_eij_iter;
                    let mut eij = if i_spin==0 {
                        &mut eij_00[(i_state,j_state)]
                    } else {
                        &mut eij_11[(i_state,j_state)]
                    };
                    //println!("debug ({},{}), before: {:?}", i_state, j_state, eij);
                    eij.0 -= e_mp2_term_ss;
                    eij.1 -= e_eij_term_ss;
                    //println!("debug after: {:?}", eij);
                });


            } else {
                let eigenvector_1 = scf_data.eigenvectors.get(i_spin_1).unwrap();
                let eigenvalues_1 = scf_data.eigenvalues.get(i_spin_1).unwrap();
                let occupation_1 = scf_data.occupation.get(i_spin_1).unwrap();
                let homo_1 = scf_data.homo.get(i_spin_1).unwrap().clone();
                let lumo_1 = scf_data.lumo.get(i_spin_1).unwrap().clone();
                //let num_occu_1 = homo_1 + 1;
                let num_occu_1 = lumo_1;
                let (rimo_1, vir_range, occ_range) = &ri3mo_vec[i_spin_1];
                let lumo_min = vir_range.start;

                let eigenvector_2 = scf_data.eigenvectors.get(i_spin_2).unwrap();
                let eigenvalues_2 = scf_data.eigenvalues.get(i_spin_2).unwrap();
                let occupation_2 = scf_data.occupation.get(i_spin_2).unwrap();
                let homo_2 = scf_data.homo.get(i_spin_2).unwrap().clone();
                let lumo_2 = scf_data.lumo.get(i_spin_2).unwrap().clone();
                //let num_occu_2 = homo_2 + 1;
                let num_occu_2 = lumo_2;
                let (rimo_2, _, _) = &ri3mo_vec[i_spin_2];

                // prepare the elec_pair for the rayon parallelization
                let mut elec_pair: Vec<[usize;2]> = vec![];
                for i_state in start_mo..num_occu_1 {
                    for j_state in start_mo..num_occu_2 {
                        elec_pair.push([i_state,j_state])
                    }
                };
                let (sender, receiver) = channel();
                elec_pair.par_iter().for_each_with(sender,|s,i_pair| {
                    let mut e_mp2_term_os = 0.0_f64;
                    let mut num_eij_iter = 0_usize;
                    let mut e_eij_term_ss = 0.0_f64;
                    let mut e_eij_term_os = 0.0_f64;

                    let i_state = i_pair[0];
                    let j_state = i_pair[1];
                    let i_state_eigen = eigenvalues_1.get(i_state).unwrap();
                    let j_state_eigen = eigenvalues_2.get(j_state).unwrap();
                    let i_state_occ = occupation_1[i_state];
                    let j_state_occ = occupation_2[j_state];
                    
                    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                        let ij_state_eigen = i_state_eigen + j_state_eigen;
                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                        // the indices in rimo are shifted
                        let i_loc_state = i_state-occ_range.start;
                        let j_loc_state = j_state-occ_range.start;
                        let ri_i = rimo_1.get_reducing_matrix(i_loc_state).unwrap();
                        let ri_j = rimo_2.get_reducing_matrix(j_loc_state).unwrap();
                        let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                        _dgemm(
                            &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                            &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                            &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                            1.0,0.0);
                        let mut denominator = MatrixFull::new([vir_range.len(),vir_range.len()],[0.0_f64;2]);
                        for i_virt in lumo_1..num_state {
                            let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                            let i_virt_occ = occupation_1[i_virt];
                            if (1.0 - i_virt_occ).abs() > 1.0e-6 {
                                for j_virt in lumo_2..num_state {
                                    let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                                    let j_virt_occ = occupation_2[j_virt];

                                    if (1.0-j_virt_occ).abs() > 1.0e-6 {
                                        let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
    
                                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                        if double_gap.abs()<=10E-6 {
                                            println!("Warning: too close to degeneracy");
                                            double_gap = 1.0E-6;
                                        };
    
                                        let occ_state = (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));
    
    
                                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                        // the indices in rimo are shifted
                                        let i_loc_virt = i_virt-vir_range.start;
                                        let j_loc_virt = j_virt-vir_range.start;
                                        let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                        e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap * occ_state;
                                        denominator[[i_loc_virt,j_loc_virt]] = [double_gap, occ_state];
                                    }
    
                                }
                            }
                        }
                        (e_eij_term_ss, e_eij_term_os,num_eij_iter) = iterator_open_shell_eij_serial(&eri_virt, &denominator, 
                            0.0, e_mp2_term_os, 
                            enhanced_factor, screening_factor, shifted_factor, 
                            lumo_1,lumo_2,false,lumo_min, num_state);
                    }
    
                    s.send((e_mp2_term_os,e_eij_term_os,num_eij_iter,i_state,j_state)).unwrap()
                });

                receiver.into_iter().for_each(|(e_mp2_term_os,e_eij_term_os, num_eij_iter,i_state,j_state)| {
                    e_mp2_os -= e_mp2_term_os;
                    e_bge2_os -= e_eij_term_os;
                    //e_ij[(i_state,j_state)] = (e_mp2_term_os,0.0, e_eij_term_os,0.0,num_eij_iter);
                    //e_ij[(i_state,j_state)].0 += e_mp2_term_os;
                    //e_ij[(i_state,j_state)].2 += e_eij_term_os;
                    //e_ij[(i_state,j_state)].4 += num_eij_iter;
                    let mut eij = &mut eij_01[(i_state,j_state)];
                    //println!("debug ({},{}), before: {:?}", i_state, j_state, eij);
                    eij.0 -= e_mp2_term_os;
                    eij.1 -= e_eij_term_os;
                    //eij.4 += num_eij_iter;
                    //println!("debug after: {:?}", eij);
                });
            }
        }
        if scf_data.mol.ctrl.print_level>1 {
            let num_occ_alpha = if scf_data.mol.num_elec[1] <= 1.0e-6 {0} else {scf_data.homo[0] + 1};
            let num_occ_beta = if scf_data.mol.num_elec[2] <= 1.0e-6 {0} else {scf_data.homo[1] + 1};
            println!("Print the correlation energies for each electron-pair:");
            println!("For (alpha, alpha)");
            for i_state in start_mo..num_occ_alpha {
                for j_state in i_state+1..num_occ_alpha {
                    let (e_mp2_term, e_eij_term) = eij_00[(i_state, j_state)];
                    println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term, e_eij_term);
                }
            };
            println!("For (beta, beta)");
            for i_state in start_mo..num_occ_beta {
                for j_state in i_state+1..num_occ_beta {
                    let (e_mp2_term, e_eij_term) = eij_11[(i_state, j_state)];
                    println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term, e_eij_term);
                }
            }
            println!("For (alpha, beta)");
            for i_state in start_mo..num_occ_alpha {
                for j_state in start_mo..num_occ_beta {
                    let (e_mp2_term, e_eij_term) = eij_01[(i_state, j_state)];
                    println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term, e_eij_term);
                }
            }
        }
    } else {
        panic!("RI3MO should be initialized before the PT2 calculations")
    };
    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok(([e_bge2_ss+e_bge2_os,e_bge2_os,e_bge2_ss],[eij_00,eij_01,eij_11]))

}

pub fn open_shell_sbge2_rayon(scf_data: &crate::scf_io::SCF) -> anyhow::Result<[f64;3]> {
    let ([e_bge2_tot, e_bge2_os,e_bge2_ss],_) = open_shell_sbge2_detailed_rayon(scf_data).unwrap();
    Ok([e_bge2_ss+e_bge2_os,e_bge2_os,e_bge2_ss])
}


pub fn open_shell_eij_serial(
    eri_virt: &MatrixFull<f64>, 
    denominator: &MatrixFull<[f64;2]>, 
    previous_eij_ss: f64,
    previous_eij_os: f64,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
    lumo_i: usize, 
    lumo_j: usize,
    is_same_spin: bool,
    lumo_min: usize, num_state: usize) -> (f64,f64)
{
    //let mut tmp_energy = 0.0f64;
    let mut eij_ss = 0.0f64;
    let mut eij_os = 0.0f64;
    if screening_factor.abs().lt(&1.0E-6) {
        if is_same_spin {
            for i_virt in lumo_i..num_state {
                for j_virt in i_virt+1..num_state {
                    let i_loc_virt = i_virt - lumo_min;
                    let j_loc_virt = j_virt - lumo_min;
                    let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                    let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                    let [dij, occij] = denominator[[i_loc_virt,j_loc_virt]];
                    
                    if occij.abs() > 1.0e-6 {
                        let tmp_energy_term_ss = (vij-vji).powf(2.0f64)/(dij + enhanced_factor*previous_eij_ss)*occij;
                        eij_ss += tmp_energy_term_ss;
                    }
                }
            }
        } else {
            for i_virt in lumo_i..num_state {
                for j_virt in lumo_j..num_state {
                    let i_loc_virt = i_virt - lumo_min;
                    let j_loc_virt = j_virt - lumo_min;
                    let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                    let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                    let [dij,occij] = denominator[[i_loc_virt,j_loc_virt]];
                    

                    if occij.abs() > 1.0e-6 {
                        let tmp_energy_term_os = vij.powf(2.0f64)/(dij + enhanced_factor*previous_eij_os)*occij;
                        eij_os += tmp_energy_term_os;
                    }
                }
            }
        }
    } else {
        if is_same_spin {
            for i_virt in lumo_i..num_state {
                let i_loc_virt = i_virt - lumo_min;
                for j_virt in i_virt+1..num_state {
                    let j_loc_virt = j_virt - lumo_min;
                    let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                    let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                    let [dij,occij] = denominator[[i_loc_virt,j_loc_virt]];

                    if occij.abs() > 1.0e-6 {
                        let screening_enhanced = enhanced_factor*erfc(screening_factor*dij);
                        let tmp_energy_term_ss = (vij-vji).powf(2.0f64)/(dij + screening_enhanced*(previous_eij_ss+shifted_factor))*occij;
                        eij_ss += tmp_energy_term_ss;
                    }
                }
            }
            //println!("debug: mp2_ss={:16.8}, eij_ss={:16.8}", previous_eij_ss, eij_ss);
        } else {
            for i_virt in lumo_i..num_state {
                let i_loc_virt = i_virt - lumo_min;
                for j_virt in lumo_j..num_state {
                    let j_loc_virt = j_virt - lumo_min;
                    let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                    let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                    let [dij,occij] = denominator[[i_loc_virt,j_loc_virt]];

                    if occij.abs() > 1.0e-6 {
                        let screening_enhanced = enhanced_factor*erfc(screening_factor*dij);
                        let tmp_energy_term_os = vij.powf(2.0f64)/(dij + screening_enhanced*(previous_eij_os+shifted_factor))*occij;
                        eij_os += tmp_energy_term_os;
                    }
                }
            }
        }
    }


    (eij_ss, eij_os)
} 

pub fn iterator_open_shell_eij_serial(
    eri_virt: &MatrixFull<f64>, 
    denominator: &MatrixFull<[f64;2]>, 
    init_eij_ss: f64,
    init_eij_os: f64,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
    lumo_i: usize, 
    lumo_j: usize,
    is_same_spin: bool,
    lumo_min: usize, num_state: usize) -> (f64,f64,usize)
{
    let threshold_eij = 1.0E-8;
    let max_iteration = 100_usize;
    let mut delta_eij = [1.0,1.0];
    let mut num_iter = 0_usize;

    let mut eij_ss = 0.0;
    let mut eij_os = 0.0;
    let mut previous_eij_ss = init_eij_ss;
    let mut previous_eij_os = init_eij_os;

    //println!("Electron-pair coupling iterations start for sBGE2:");
    //println!("Input Eij[ss]: {:16.8}, Eij[os]: {:16.8}", init_eij_ss,init_eij_os);

    while ! (abs(delta_eij[1])<threshold_eij || num_iter >= max_iteration) {
        (eij_ss, eij_os) = open_shell_eij_serial(
            eri_virt, 
            denominator, 
            previous_eij_ss, 
            previous_eij_os, 
            enhanced_factor, 
            screening_factor, 
            shifted_factor, 
            lumo_i, 
            lumo_j,
            is_same_spin,
            lumo_min, num_state);
        delta_eij[0] = delta_eij[1];
        delta_eij[1] = (eij_ss-previous_eij_ss)+(eij_os-previous_eij_os);
        if delta_eij[0]*delta_eij[1]<0.0 {
            previous_eij_os = (eij_os + previous_eij_os)*0.5;
            previous_eij_ss = (eij_ss + previous_eij_ss)*0.5;
        } else {
            previous_eij_ss = eij_ss;
            previous_eij_os = eij_os;
        }
        //println!("Eij[ss]: {:16.8}, Eij[os]: {:16.8} on {:3}th step", eij_ss,eij_os, num_iter);
        num_iter += 1;
    }

    (eij_ss,eij_os,num_iter)

}

pub fn close_shell_sbge2_rayon_mpi(scf_data: &crate::scf_io::SCF,mpi_operator:&Option<MPIOperator>) -> anyhow::Result<[f64;3]> {

    if let (Some(mpi_op), Some(mpi_ix)) = (mpi_operator, &scf_data.mol.mpi_data) {

        // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        utilities::omp_set_num_threads_wrapper(1);

        let my_rank = mpi_ix.rank;
        let size = mpi_ix.size;
        let ran_auxbas_loc = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distrubtion should be initalized for the auxiliary basis sets before post-SCF calculations")
        };
        let num_auxbas_loc = ran_auxbas_loc.len();

        let enhanced_factor = 1.0;
        let screening_factor = 1.0;
        let shifted_factor = 0.0;
        //let mut tmp_record = TimeRecords::new();


        let num_auxbas = scf_data.mol.num_auxbas;
        let num_basis = scf_data.mol.num_basis;
        let num_state = scf_data.mol.num_state;
        let homo = scf_data.homo.get(0).unwrap().clone();
        let lumo = scf_data.lumo.get(0).unwrap().clone();
        let start_mo: usize = scf_data.mol.start_mo;
        //let num_occu = homo + 1;
        //let num_occu = lumo;
        let num_occu = if scf_data.mol.num_elec[0] <= 1.0e-6 {0} else {homo + 1};

        let mut e_mp2_ss = 0.0_f64;
        let mut e_mp2_os = 0.0_f64;
        let mut e_bge2_ss = 0.0_f64;
        let mut e_bge2_os = 0.0_f64;

        // fake the detailed output of electron_pair correlation energies
        //let mut eij_00 = MatrixFull::new([num_occu,num_occu], (0.0_f64,0.0_f64));
        //let mut eij_01 = MatrixFull::new([num_occu,num_occu], (0.0_f64,0.0_f64));
        //let mut eij_11 = MatrixFull::new([num_occu,num_occu], (0.0_f64,0.0_f64));
        //let mut eij_00 = MatrixFull::empty();
        //let mut eij_01 = MatrixFull::empty();
        //let mut eij_11 = MatrixFull::empty();

        if let Some(ri3mo_vec) = &scf_data.ri3mo {
            let (rimo, vir_range, occ_range) = &ri3mo_vec[0];

            let mut e_ij = MatrixFull::new([num_occu,num_occu], (0.0_f64,0.0_f64,0.0_f64,0.0_f64,0_usize));

            let lumo_min = vir_range.start;

            let eigenvector = scf_data.eigenvectors.get(0).unwrap();
            let eigenvalues = scf_data.eigenvalues.get(0).unwrap();
            let occupation = scf_data.occupation.get(0).unwrap();

            for i_state in start_mo..num_occu {
                for j_state in i_state..num_occu {

                    let mut num_eij_iter = 0_usize;
                    let mut e_eij_term_ss = 0.0_f64;
                    let mut e_eij_term_os = 0.0_f64;

                    let mut eij_mp2_ss = 0.0_f64;
                    let mut eij_mp2_os = 0.0_f64;
                    let mut eij_bge2_ss = 0.0_f64;
                    let mut eij_bge2_os = 0.0_f64;

                    let i_state_eigen = eigenvalues[i_state];
                    let j_state_eigen = eigenvalues[j_state];
                    let i_state_occ = occupation[i_state]/2.0;
                    let j_state_occ = occupation[j_state]/2.0;
                    let ij_state_eigen = i_state_eigen + j_state_eigen;

                    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {

                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                        // the indices in rimo are shifted
                        let i_loc_state = i_state-occ_range.start;
                        let j_loc_state = j_state-occ_range.start;
                        let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                        let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                        //let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                        //_dgemm(
                        //    &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                        //    &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                        //    &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                        //    1.0,0.0);
                        let mut eri_virt = {
                            let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                            _dgemm(
                                &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                                &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                                &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                1.0,0.0);
                            let mut eri_virt = mpi_reduce(&mpi_op.world, &loc_eri_virt.data_ref().unwrap(), 0, &SystemOperation::sum());
                            mpi_broadcast_vector(&mpi_op.world, &mut eri_virt, 0);
                            MatrixFull::from_vec([vir_range.len(), vir_range.len()], eri_virt).unwrap()
                        };

                        let mut denominator = MatrixFull::new([vir_range.len(),vir_range.len()],[0.0_f64;2]);
                        let loc_virt_pair = mpi_ix.distribution_opposite_spin_virtual_orbital_pair(lumo, lumo, num_state, scf_data.mol.ctrl.pt2_mpi_mode);
                        let (sender, receiver) = channel();
                        loc_virt_pair.par_iter().for_each_with(sender,|s,i_pair| {
                            let mut e_mp2_term_ss = 0.0_f64;
                            let mut e_mp2_term_os = 0.0_f64;
                            //let mut denominator = MatrixFull::new([vir_range.len(),vir_range.len()],[0.0_f64;2]);
                            let i_virt = i_pair[0];
                            let j_virt = i_pair[1];
                            let i_virt_eigen = eigenvalues[i_virt];
                            let i_virt_occ = occupation[i_virt]/2.0;
                            let (double_gap, occ_state, i_loc_virt, j_loc_virt) = {
                                if (1.0 - i_virt_occ).abs() > 1.0e-6 {
                                    let j_virt_eigen = eigenvalues[j_virt];
                                    let j_virt_occ = occupation[j_virt]/2.0;

                                    if (1.0-j_virt_occ).abs() > 1.0e-6 {
                                        let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                        if double_gap.abs()<=1.0E-6 {
                                            println!("Warning: too close to degeneracy");
                                            double_gap = 1.0E-6;
                                        };

                                        // consider fractional occupation
                                        //double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));
                                        let occ_state = (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                        // the indices in rimo are shifted
                                        let i_loc_virt = i_virt-vir_range.start;
                                        let j_loc_virt = j_virt-vir_range.start;

                                        let e_mp2_a = eri_virt[[i_loc_virt,j_loc_virt]];
                                        let e_mp2_b = eri_virt[[j_loc_virt,i_loc_virt]];
                                        e_mp2_term_ss += (e_mp2_a - e_mp2_b) * e_mp2_a / double_gap * occ_state;
                                        e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap * occ_state; 
                                        (double_gap, occ_state, i_loc_virt, j_loc_virt)
                                    } else {
                                        (0.0, 0.0, 0,0)
                                    }
                                } else {
                                    (0.0, 0.0, 0, 0)
                                }
                            };
                            s.send((e_mp2_term_os, e_mp2_term_ss, double_gap, occ_state, i_loc_virt, j_loc_virt)).unwrap()
                        });
                        receiver.iter().for_each(|(e_mp2_term_os,e_mp2_term_ss, double_gap, occ_state,i_loc_virt, j_loc_virt)| {
                            eij_mp2_os -= e_mp2_term_os;
                            eij_mp2_ss -= e_mp2_term_ss;
                            denominator[[i_loc_virt, j_loc_virt]] = [double_gap, occ_state];
                            //e_ij[(i_state,j_state)] = (-e_mp2_term_os,-e_mp2_term_ss, -e_eij_term_os, -e_eij_term_ss,num_eij_iter)
                            
                        });
                        // sum up the ss and os contribution from the mpi tasks.
                        eij_mp2_ss = mpi_reduce(&mpi_op.world, &mut [eij_mp2_ss], 0, &SystemOperation::sum())[0];
                        mpi_broadcast(&mpi_op.world, &mut eij_mp2_ss, 0);
                        eij_mp2_os = mpi_reduce(&mpi_op.world, &mut [eij_mp2_os], 0, &SystemOperation::sum())[0];
                        mpi_broadcast(&mpi_op.world, &mut eij_mp2_os, 0);

                        (eij_bge2_ss, eij_bge2_os,num_eij_iter) = iterator_close_shell_eij_rayon_mpi(&eri_virt, &denominator, 
                            eij_mp2_ss, eij_mp2_os, 
                            enhanced_factor, screening_factor, shifted_factor, 
                            lumo, lumo_min, num_state, &scf_data.mol.mpi_data, mpi_operator);

                        if i_state != j_state {
                            eij_mp2_ss *= 2.0;
                            eij_mp2_os *= 2.0;
                            eij_bge2_ss *= 2.0;
                            eij_bge2_os *= 2.0;
                        };

                        e_mp2_os -= eij_mp2_os;
                        e_mp2_ss -= eij_mp2_ss;
                        e_bge2_os -= eij_bge2_os;
                        e_bge2_ss -= eij_bge2_ss;
                    }
                }
            };
        } else {
            panic!("RI3MO should be initialized before the PT2 calculations")
        };

        // reuse the default omp_num_threads setting
        utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

        //println!("debug pt2 and sbge2: {:?}, {:?}", e_mp2_os+e_mp2_ss, e_bge2_ss+ e_bge2_os);

        //tmp_record.report_all();
        //Ok(([e_bge2_ss+e_bge2_os,e_bge2_os,e_bge2_ss],[eij_00,eij_01,eij_11]))
        Ok([e_bge2_ss+e_bge2_os,e_bge2_os,e_bge2_ss])
    } else {
        let (pt2, eij) = close_shell_sbge2_detailed_rayon(scf_data).unwrap();
        Ok(pt2)
    }

}

pub fn iterator_close_shell_eij_rayon_mpi(
    eri_virt: &MatrixFull<f64>, 
    denominator: &MatrixFull<[f64;2]>, 
    init_eij_ss: f64,
    init_eij_os: f64,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
    lumo: usize,
    lumo_min: usize, num_state: usize, 
    mpi_data: &Option<MPIData>, mpi_operator:&Option<MPIOperator>) -> (f64,f64,usize)
{
    let threshold_eij = 1.0E-8;
    let max_iteration = 100_usize;
    let mut delta_eij = [1.0,1.0];
    let mut num_iter = 0_usize;

    let mut eij_ss = 0.0;
    let mut eij_os = 0.0;
    let mut previous_eij_ss = init_eij_ss;
    let mut previous_eij_os = init_eij_os;

    //println!("Electron-pair coupling iterations start for sBGE2:");
    //println!("Input Eij[ss]: {:16.8}, Eij[os]: {:16.8}", init_eij_ss,init_eij_os);
    while ! (abs(delta_eij[1])<threshold_eij || num_iter >= max_iteration) {
        if let Some(mpi_op) = &mpi_operator {
            (eij_ss, eij_os) = close_shell_eij_rayon_mpi(
                eri_virt, 
                denominator, 
                previous_eij_ss, 
                previous_eij_os, 
                enhanced_factor, 
                screening_factor, 
                shifted_factor, 
                lumo, 
                lumo_min, 
                num_state,
                mpi_data,
                mpi_operator);
        } else {
            (eij_ss, eij_os) = close_shell_eij_serial(
                eri_virt, 
                denominator, 
                previous_eij_ss, 
                previous_eij_os, 
                enhanced_factor, 
                screening_factor, 
                shifted_factor, 
                lumo, 
                lumo_min, 
                num_state);
        }
        delta_eij[0] = delta_eij[1];
        delta_eij[1] = (eij_ss-previous_eij_ss)+(eij_os-previous_eij_os);
        if delta_eij[0]*delta_eij[1]<0.0 {
            previous_eij_os = (eij_os + previous_eij_os)*0.5;
            previous_eij_ss = (eij_ss + previous_eij_ss)*0.5;
        } else {
            previous_eij_ss = eij_ss;
            previous_eij_os = eij_os;
        }
        //println!("Eij[ss]: {:16.8}, Eij[os]: {:16.8} on {:3}th step", eij_ss,eij_os, num_iter);
        num_iter += 1;
    }

    (eij_ss,eij_os,num_iter)

}

pub fn close_shell_eij_rayon_mpi(
    eri_virt: &MatrixFull<f64>, 
    denominator: &MatrixFull<[f64;2]>, 
    previous_eij_ss: f64,
    previous_eij_os: f64,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
    lumo: usize,
    lumo_min: usize, num_state: usize,
    mpi_data: &Option<MPIData>,
    mpi_operator:&Option<MPIOperator>) -> (f64,f64)
{
    if let (Some(mpi_op), Some(mpi_ix)) = (mpi_operator, mpi_data) {
        let my_rank = mpi_ix.rank;
        let size = mpi_ix.size;
        let ran_auxbas_loc = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distrubtion should be initalized for the auxiliary basis sets before post-SCF calculations")
        };
        let num_auxbas_loc = ran_auxbas_loc.len();

        let mut tmp_energy = 0.0f64;
        let mut eij_ss = 0.0f64;
        let mut eij_os = 0.0f64;

        let loc_virt_pair = mpi_ix.distribution_opposite_spin_virtual_orbital_pair(lumo, lumo, num_state, 0);
        if screening_factor.abs().lt(&1.0E-6) {
            let (sender, receiver) = channel();
            loc_virt_pair.par_iter().for_each_with(sender,|s,i_pair| {
                let i_virt = i_pair[0];
                let j_virt = i_pair[1];
                let i_loc_virt = i_virt - lumo_min;
                let j_loc_virt = j_virt - lumo_min;
                let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                let [dij,occ_ij] = denominator[[i_loc_virt,j_loc_virt]];
                
                let (tmp_ss, tmp_os) = if occ_ij.abs() > 1.0e-6 {
                    let tmp_energy_term_ss = (vij-vji)*vij/(dij + enhanced_factor*previous_eij_ss)*occ_ij;
                    let tmp_energy_term_os = vij.powf(2.0f64)/(dij + enhanced_factor*previous_eij_os)*occ_ij;
                    //tmp_energy += tmp_energy_term_os + tmp_energy_term_ss;
                    //eij_ss += tmp_energy_term_ss;
                    //eij_os += tmp_energy_term_os;
                    (tmp_energy_term_ss, tmp_energy_term_os)
                } else {
                    (0.0, 0.0)
                };
                s.send((tmp_ss, tmp_os)).unwrap()
            });
            receiver.into_iter().for_each(|(tmp_ss, tmp_os)| {
                eij_ss += tmp_ss;
                eij_os += tmp_os;
            });

        } else {
            let (sender, receiver) = channel();
            loc_virt_pair.par_iter().for_each_with(sender,|s,i_pair| {
                let i_virt = i_pair[0];
                let j_virt = i_pair[1];
                let i_loc_virt = i_virt - lumo_min;
                let j_loc_virt = j_virt - lumo_min;
                let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                let [dij,occ_ij] = denominator[[i_loc_virt,j_loc_virt]];
                
                let (tmp_ss, tmp_os) = if occ_ij.abs() > 1.0e-6 {

                    let screening_enhanced = enhanced_factor*erfc(screening_factor*dij);
                    let tmp_energy_term_ss = (vij-vji)*vij/(dij + screening_enhanced*(previous_eij_ss+shifted_factor))*occ_ij;
                    let tmp_energy_term_os = vij.powf(2.0f64)/(dij + screening_enhanced*(previous_eij_os+shifted_factor))*occ_ij;

                    (tmp_energy_term_ss, tmp_energy_term_os)
                } else {
                    (0.0, 0.0)
                };
                s.send((tmp_ss, tmp_os)).unwrap()
            });
            receiver.into_iter().for_each(|(tmp_ss, tmp_os)| {
                eij_ss += tmp_ss;
                eij_os += tmp_os;
            });

        }

        // sum up the ss and os contribution from the mpi tasks.
        eij_ss = mpi_reduce(&mpi_op.world, &mut [eij_ss], 0, &SystemOperation::sum())[0];
        mpi_broadcast(&mpi_op.world, &mut eij_ss, 0);
        eij_os = mpi_reduce(&mpi_op.world, &mut [eij_os], 0, &SystemOperation::sum())[0];
        mpi_broadcast(&mpi_op.world, &mut eij_os, 0);

        (eij_ss, eij_os)

    } else {
        close_shell_eij_serial(eri_virt, denominator, previous_eij_ss, previous_eij_os, enhanced_factor, screening_factor, shifted_factor, lumo, lumo_min, num_state)
    }

} 

pub fn open_shell_sbge2_rayon_mpi(scf_data: &crate::scf_io::SCF, mpi_operator:&Option<MPIOperator>) -> anyhow::Result<[f64;3]> {
    if let (Some(mpi_op), Some(mpi_ix)) = (mpi_operator, &scf_data.mol.mpi_data) {

        let my_rank = mpi_ix.rank;
        let size = mpi_ix.size;
        let ran_auxbas_loc = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distrubtion should be initalized for the auxiliary basis sets before post-SCF calculations")
        };
        let num_auxbas_loc = ran_auxbas_loc.len();

        let enhanced_factor = 1.0;
        let screening_factor = 1.0;
        let shifted_factor = 0.0;

        // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        utilities::omp_set_num_threads_wrapper(1);

        let mut e_mp2_ss = 0.0_f64;
        let mut e_mp2_os = 0.0_f64;
        let mut e_bge2_ss = 0.0_f64;
        let mut e_bge2_os = 0.0_f64;
        let num_occu_alpha = if scf_data.mol.num_elec[1] <= 1.0e-6 {0} else {scf_data.homo[0] + 1};
        let num_occu_beta = if scf_data.mol.num_elec[2] <= 1.0e-6 {0} else {scf_data.homo[1] + 1};

        //let num_occu_max = scf_data.homo[0].max(scf_data.homo[1])+1;
        let num_occu_max = num_occu_alpha.max(num_occu_beta);

        //let mut eij_00 = MatrixFull::new([num_occu_max,num_occu_max], (0.0_f64,0.0_f64));
        //let mut eij_01 = MatrixFull::new([num_occu_max,num_occu_max], (0.0_f64,0.0_f64));
        //let mut eij_11 = MatrixFull::new([num_occu_max,num_occu_max], (0.0_f64,0.0_f64));

        if let Some(ri3mo_vec) = &scf_data.ri3mo {

            let start_mo: usize = scf_data.mol.start_mo;
            let num_basis = scf_data.mol.num_basis;
            let num_state = scf_data.mol.num_state;
            let num_auxbas = scf_data.mol.num_auxbas;
            let spin_channel = scf_data.mol.spin_channel;
            let i_spin_pair: [(usize,usize);3] = [(0,0),(0,1),(1,1)];


            for (i_spin_1,i_spin_2) in i_spin_pair {
                if i_spin_1 == i_spin_2 {

                    let i_spin = i_spin_1;
                    let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
                    let eigenvalues = scf_data.eigenvalues.get(i_spin).unwrap();
                    let occupation = scf_data.occupation.get(i_spin).unwrap();

                    let homo = scf_data.homo[i_spin].clone();
                    let lumo = scf_data.lumo[i_spin].clone();
                    //let num_occu = homo + 1;
                    let num_occu = lumo;

                    let (rimo, vir_range, occ_range) = &ri3mo_vec[i_spin];
                    let lumo_min = vir_range.start;

                    for i_state in start_mo..num_occu {
                        for j_state in i_state+1..num_occu {
                            let mut eij_mp2_ss = 0.0_f64;
                            let mut eij_bge2_ss = 0.0_f64;

                            let mut num_eij_iter = 0_usize;
                            //let mut e_mp2_term_ss = 0.0_f64;
                            //let mut e_eij_term_ss = 0.0_f64;

                            let i_state_eigen = eigenvalues[i_state];
                            let j_state_eigen = eigenvalues[j_state];
                            let ij_state_eigen = i_state_eigen + j_state_eigen;
                            let i_state_occ = occupation[i_state];
                            let j_state_occ = occupation[j_state];

                            if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {

                                // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                // the indices in rimo are shifted
                                let i_loc_state = i_state-occ_range.start;
                                let j_loc_state = j_state-occ_range.start;
                                let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                                let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                                let mut eri_virt = {
                                    let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                                    _dgemm(
                                        &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                                        &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                                        &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                        1.0,0.0);
                                    let mut eri_virt = mpi_reduce(&mpi_op.world, &loc_eri_virt.data_ref().unwrap(), 0, &SystemOperation::sum());
                                    mpi_broadcast_vector(&mpi_op.world, &mut eri_virt, 0);
                                    MatrixFull::from_vec([vir_range.len(), vir_range.len()], eri_virt).unwrap()
                                };

                                let mut denominator = MatrixFull::new([vir_range.len(),vir_range.len()],[0.0_f64;2]);
                                let loc_virt_pair = mpi_ix.distribution_same_spin_virtual_orbital_pair(lumo, num_state);
                                let (sender, receiver) = channel();
                                loc_virt_pair.par_iter().for_each_with(sender,|s,i_pair| {
                                    let i_virt = i_pair[0];
                                    let j_virt = i_pair[1];
                                    let i_virt_eigen = eigenvalues[i_virt];
                                    let i_virt_occ = occupation[i_virt];
                                    let j_virt_eigen = eigenvalues[j_virt];
                                    let j_virt_occ = occupation[j_virt];
                                    let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
                                    let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                    if double_gap.abs()<=10E-6 {
                                        println!("Warning: too close to degeneracy");
                                        double_gap = 1.0E-6;
                                    }
                                    let occ_state = (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));
                                    let i_loc_virt = i_virt-vir_range.start;
                                    let j_loc_virt = j_virt-vir_range.start;
                                    let e_mp2_term_ss = if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6{
                                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                        // the indices in rimo are shifted
                                        let e_mp2_a = eri_virt[[i_loc_virt,j_loc_virt]];
                                        let e_mp2_b = eri_virt[[j_loc_virt,i_loc_virt]];
                                        (e_mp2_a - e_mp2_b).powf(2.0) / double_gap * occ_state
                                        //denominator[[i_loc_virt,j_loc_virt]] = [double_gap, occ_state];
                                    } else {
                                        0.0
                                    };
                                    s.send((e_mp2_term_ss, double_gap, occ_state, i_loc_virt, j_loc_virt)).unwrap()
                                });
                                receiver.iter().for_each(|(e_mp2_term_ss, double_gap, occ_state,i_loc_virt, j_loc_virt)| {
                                    eij_mp2_ss -= e_mp2_term_ss;
                                    denominator[[i_loc_virt, j_loc_virt]] = [double_gap, occ_state];
                                    //e_ij[(i_state,j_state)] = (-e_mp2_term_os,-e_mp2_term_ss, -e_eij_term_os, -e_eij_term_ss,num_eij_iter)
                                    
                                });
                                // sum up the ss and os contribution from the mpi tasks.
                                eij_mp2_ss = mpi_reduce(&mpi_op.world, &mut [eij_mp2_ss], 0, &SystemOperation::sum())[0];
                                mpi_broadcast(&mpi_op.world, &mut eij_mp2_ss, 0);

                                (eij_bge2_ss, _, num_eij_iter) = iterator_open_shell_eij_rayon_mpi(&eri_virt, &denominator, 
                                    eij_mp2_ss, 0.0, 
                                    enhanced_factor, screening_factor, shifted_factor, 
                                    lumo,lumo,true,
                                    lumo_min, num_state, 
                                    &scf_data.mol.mpi_data, mpi_operator);

                                //println!("debug: i,j=({},{}), mp2_ss={:16.8}, eij_ss={:16.8}", i_state, j_state, e_mp2_term_ss, e_eij_term_ss);
                                e_mp2_ss -= eij_mp2_ss;
                                e_bge2_ss -= eij_bge2_ss;
                            }
                        }
                    };


                } else {
                    let eigenvector_1 = scf_data.eigenvectors.get(i_spin_1).unwrap();
                    let eigenvalues_1 = scf_data.eigenvalues.get(i_spin_1).unwrap();
                    let occupation_1 = scf_data.occupation.get(i_spin_1).unwrap();
                    let homo_1 = scf_data.homo.get(i_spin_1).unwrap().clone();
                    let lumo_1 = scf_data.lumo.get(i_spin_1).unwrap().clone();
                    //let num_occu_1 = homo_1 + 1;
                    let num_occu_1 = lumo_1;
                    let (rimo_1, vir_range, occ_range) = &ri3mo_vec[i_spin_1];
                    let lumo_min = vir_range.start;

                    let eigenvector_2 = scf_data.eigenvectors.get(i_spin_2).unwrap();
                    let eigenvalues_2 = scf_data.eigenvalues.get(i_spin_2).unwrap();
                    let occupation_2 = scf_data.occupation.get(i_spin_2).unwrap();
                    let homo_2 = scf_data.homo.get(i_spin_2).unwrap().clone();
                    let lumo_2 = scf_data.lumo.get(i_spin_2).unwrap().clone();
                    //let num_occu_2 = homo_2 + 1;
                    let num_occu_2 = lumo_2;
                    let (rimo_2, _, _) = &ri3mo_vec[i_spin_2];

                    for i_state in start_mo..num_occu_1 {
                        for j_state in start_mo..num_occu_2 {

                            let mut eij_mp2_os = 0.0_f64;
                            let mut eij_bge2_os = 0.0_f64;

                            let mut num_eij_iter = 0_usize;

                            let i_state_eigen = eigenvalues_1.get(i_state).unwrap();
                            let j_state_eigen = eigenvalues_2.get(j_state).unwrap();
                            let i_state_occ = occupation_1[i_state];
                            let j_state_occ = occupation_2[j_state];
                        
                            if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                                let ij_state_eigen = i_state_eigen + j_state_eigen;
                                // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                // the indices in rimo are shifted
                                let i_loc_state = i_state-occ_range.start;
                                let j_loc_state = j_state-occ_range.start;
                                let ri_i = rimo_1.get_reducing_matrix(i_loc_state).unwrap();
                                let ri_j = rimo_2.get_reducing_matrix(j_loc_state).unwrap();
                                let mut eri_virt = {
                                    let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                                    _dgemm(
                                        &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                                        &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                                        &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                        1.0,0.0);
                                    let mut eri_virt = mpi_reduce(&mpi_op.world, &loc_eri_virt.data_ref().unwrap(), 0, &SystemOperation::sum());
                                    mpi_broadcast_vector(&mpi_op.world, &mut eri_virt, 0);
                                    MatrixFull::from_vec([vir_range.len(), vir_range.len()], eri_virt).unwrap()
                                };
                                let mut denominator = MatrixFull::new([vir_range.len(),vir_range.len()],[0.0_f64;2]);
                                let loc_virt_pair = mpi_ix.distribution_opposite_spin_virtual_orbital_pair(lumo_1, lumo_2, num_state, scf_data.mol.ctrl.pt2_mpi_mode);
                                let (sender, receiver) = channel();
                                loc_virt_pair.par_iter().for_each_with(sender,|s,i_pair| {
                                    let i_virt = i_pair[0];
                                    let j_virt = i_pair[1];
                                    let i_loc_virt = i_virt-vir_range.start;
                                    let j_loc_virt = j_virt-vir_range.start;
                                    let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                                    let i_virt_occ = occupation_1[i_virt];
                                    let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                                    let j_virt_occ = occupation_2[j_virt];
                                    let (e_mp2_term_os, double_gap, occ_state) = if (1.0 - i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                                        let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
    
                                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                        if double_gap.abs()<=10E-6 {
                                            println!("Warning: too close to degeneracy");
                                            double_gap = 1.0E-6;
                                        };
    
                                        let occ_state = (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));
    
    
                                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                        // the indices in rimo are shifted
                                        let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                        let e_mp2_term_os = e_mp2_a * e_mp2_a / double_gap * occ_state;
                                        //denominator[[i_loc_virt,j_loc_virt]] = [double_gap, occ_state];
                                        (e_mp2_term_os, double_gap, occ_state)
                                    } else {
                                        (0.0, 0.0, 0.0)
                                    };
                                    s.send( (e_mp2_term_os, double_gap, occ_state, i_virt, j_virt, i_loc_virt, j_loc_virt)).unwrap()
                                });

                                receiver.into_iter().for_each(|(e_mp2_term_os, double_gap, occ_state, i_virt, j_virt, i_virt_loc, j_virt_loc)| {
                                    eij_mp2_os += e_mp2_term_os;
                                    denominator[[i_virt_loc, j_virt_loc]] = [double_gap, occ_state];
                                    //e_ij[(i_state, j_state)] = (-e_mp2_term_os, -e_mp2_term_ss, -e_eij_term_os, -e_eij_term_ss, num_eij_iter)
                                });
                                // sum up the ss and os contribution from the mpi tasks.
                                eij_mp2_os = mpi_reduce(&mpi_op.world, &mut [eij_mp2_os], 0, &SystemOperation::sum())[0];
                                mpi_broadcast(&mpi_op.world, &mut eij_mp2_os, 0);


                                (_, eij_bge2_os, num_eij_iter) = iterator_open_shell_eij_rayon_mpi(&eri_virt, &denominator, 
                                    0.0, eij_mp2_os, 
                                    enhanced_factor, screening_factor, shifted_factor, 
                                    lumo_1,lumo_2,false,lumo_min, num_state,
                                    &scf_data.mol.mpi_data, mpi_operator);

                                e_mp2_os -= eij_mp2_os;
                                e_bge2_os -= eij_bge2_os;
                            }
                            //s.send((e_mp2_term_os,e_eij_term_os,num_eij_iter,i_state,j_state)).unwrap()
                        }
                    };

                    //receiver.into_iter().for_each(|(e_mp2_term_os,e_eij_term_os, num_eij_iter,i_state,j_state)| {
                    //    e_mp2_os -= e_mp2_term_os;
                    //    e_bge2_os -= e_eij_term_os;
                    //    //e_ij[(i_state,j_state)] = (e_mp2_term_os,0.0, e_eij_term_os,0.0,num_eij_iter);
                    //    //e_ij[(i_state,j_state)].0 += e_mp2_term_os;
                    //    //e_ij[(i_state,j_state)].2 += e_eij_term_os;
                    //    //e_ij[(i_state,j_state)].4 += num_eij_iter;
                    //    let mut eij = &mut eij_01[(i_state,j_state)];
                    //    //println!("debug ({},{}), before: {:?}", i_state, j_state, eij);
                    //    eij.0 -= e_mp2_term_os;
                    //    eij.1 -= e_eij_term_os;
                    //    //eij.4 += num_eij_iter;
                    //    //println!("debug after: {:?}", eij);
                    //});
                }
            }
            //if scf_data.mol.ctrl.print_level>1 {
            //    let num_occ_alpha = if scf_data.mol.num_elec[1] <= 1.0e-6 {0} else {scf_data.homo[0] + 1};
            //    let num_occ_beta = if scf_data.mol.num_elec[2] <= 1.0e-6 {0} else {scf_data.homo[1] + 1};
            //    println!("Print the correlation energies for each electron-pair:");
            //    println!("For (alpha, alpha)");
            //    for i_state in start_mo..num_occ_alpha {
            //        for j_state in i_state+1..num_occ_alpha {
            //            let (e_mp2_term, e_eij_term) = eij_00[(i_state, j_state)];
            //            println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term, e_eij_term);
            //        }
            //    };
            //    println!("For (beta, beta)");
            //    for i_state in start_mo..num_occ_beta {
            //        for j_state in i_state+1..num_occ_beta {
            //            let (e_mp2_term, e_eij_term) = eij_11[(i_state, j_state)];
            //            println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term, e_eij_term);
            //        }
            //    }
            //    println!("For (alpha, beta)");
            //    for i_state in start_mo..num_occ_alpha {
            //        for j_state in start_mo..num_occ_beta {
            //            let (e_mp2_term, e_eij_term) = eij_01[(i_state, j_state)];
            //            println!("the ({:3},{:3}) elec-pair: (PT2, sBGE2)=({:16.8},{:16.8})", i_state,j_state, e_mp2_term, e_eij_term);
            //        }
            //    }
            //}
        } else {
            panic!("RI3MO should be initialized before the PT2 calculations")
        };
        // reuse the default omp_num_threads setting
        utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

        Ok(([e_bge2_ss+e_bge2_os,e_bge2_os,e_bge2_ss]))
        //panic!("ERROR: the MPI/Rayon parallelization is not yet available for open-shell SBGE2 calculation")
    } else {
        let ([e_bge2_tot, e_bge2_os,e_bge2_ss],_) = open_shell_sbge2_detailed_rayon(scf_data).unwrap();
        Ok([e_bge2_ss+e_bge2_os,e_bge2_os,e_bge2_ss])
    }
}

pub fn iterator_open_shell_eij_rayon_mpi(
    eri_virt: &MatrixFull<f64>, 
    denominator: &MatrixFull<[f64;2]>, 
    init_eij_ss: f64,
    init_eij_os: f64,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
    lumo_i: usize, 
    lumo_j: usize,
    is_same_spin: bool,
    lumo_min: usize, num_state: usize,
    mpi_data: &Option<MPIData>, mpi_operator: &Option<MPIOperator>) -> (f64,f64,usize)
{
    let threshold_eij = 1.0E-8;
    let max_iteration = 100_usize;
    let mut delta_eij = [1.0,1.0];
    let mut num_iter = 0_usize;

    let mut eij_ss = 0.0;
    let mut eij_os = 0.0;
    let mut previous_eij_ss = init_eij_ss;
    let mut previous_eij_os = init_eij_os;

    //println!("Electron-pair coupling iterations start for sBGE2:");
    //println!("Input Eij[ss]: {:16.8}, Eij[os]: {:16.8}", init_eij_ss,init_eij_os);

    while ! (abs(delta_eij[1])<threshold_eij || num_iter >= max_iteration) {
        if let Some(mpi_op) = &mpi_operator {
            (eij_ss, eij_os) = open_shell_eij_rayon_mpi(
                eri_virt, 
                denominator, 
                previous_eij_ss, 
                previous_eij_os, 
                enhanced_factor, 
                screening_factor, 
                shifted_factor, 
                lumo_i, 
                lumo_j,
                is_same_spin,
                lumo_min, num_state,
                mpi_data, mpi_operator);
        } else {
            (eij_ss, eij_os) = open_shell_eij_serial(
                eri_virt, 
                denominator, 
                previous_eij_ss, 
                previous_eij_os, 
                enhanced_factor, 
                screening_factor, 
                shifted_factor, 
                lumo_i, 
                lumo_j,
                is_same_spin,
                lumo_min, num_state);
        }
        delta_eij[0] = delta_eij[1];
        delta_eij[1] = (eij_ss-previous_eij_ss)+(eij_os-previous_eij_os);
        if delta_eij[0]*delta_eij[1]<0.0 {
            previous_eij_os = (eij_os + previous_eij_os)*0.5;
            previous_eij_ss = (eij_ss + previous_eij_ss)*0.5;
        } else {
            previous_eij_ss = eij_ss;
            previous_eij_os = eij_os;
        }
        //println!("Eij[ss]: {:16.8}, Eij[os]: {:16.8} on {:3}th step", eij_ss,eij_os, num_iter);
        num_iter += 1;
    }

    (eij_ss,eij_os,num_iter)

}

pub fn open_shell_eij_rayon_mpi(
    eri_virt: &MatrixFull<f64>, 
    denominator: &MatrixFull<[f64;2]>, 
    previous_eij_ss: f64,
    previous_eij_os: f64,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
    lumo_i: usize, 
    lumo_j: usize,
    is_same_spin: bool,
    lumo_min: usize, num_state: usize,
    mpi_data: &Option<MPIData>, mpi_operator: &Option<MPIOperator>
) -> (f64,f64)
{
    //let mut tmp_energy = 0.0f64;
    if let (Some(mpi_op), Some(mpi_ix)) = (mpi_operator, mpi_data) {
        let my_rank = mpi_ix.rank;
        let size = mpi_ix.size;
        let ran_auxbas_loc = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distrubtion should be initalized for the auxiliary basis sets before post-SCF calculations")
        };
        let num_auxbas_loc = ran_auxbas_loc.len();
        let mut eij_ss = 0.0f64;
        let mut eij_os = 0.0f64;
        if screening_factor.abs().lt(&1.0E-6) {
            if is_same_spin {
                let loc_virt_pair = mpi_ix.distribution_same_spin_virtual_orbital_pair(lumo_i, num_state);
                let (sender, receiver) = channel();
                loc_virt_pair.par_iter().for_each_with(sender,|s,i_pair| {
                //for i_virt in lumo_i..num_state {
                //    for j_virt in i_virt+1..num_state {
                    let i_virt = i_pair[0];
                    let j_virt = i_pair[1];
                    let i_loc_virt = i_virt - lumo_min;
                    let j_loc_virt = j_virt - lumo_min;
                    let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                    let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                    let [dij, occij] = denominator[[i_loc_virt,j_loc_virt]];
                    
                    let tmp_energy_term_ss = if occij.abs() > 1.0e-6 {
                        (vij-vji).powf(2.0f64)/(dij + enhanced_factor*previous_eij_ss)*occij
                        //eij_ss += tmp_energy_term_ss;
                    } else {
                        0.0
                    };
                    s.send(tmp_energy_term_ss).unwrap()
                    //}
                });
                receiver.into_iter().for_each(|tmp_energy_term_ss| {
                    eij_ss += tmp_energy_term_ss;
                });
            } else {
                let loc_virt_pair = mpi_ix.distribution_opposite_spin_virtual_orbital_pair(lumo_i, lumo_j, num_state, 0);
                let (sender, receiver) = channel();
                loc_virt_pair.par_iter().for_each_with(sender,|s,i_pair| {
                //for i_virt in lumo_i..num_state {
                //    for j_virt in lumo_j..num_state {
                    let i_virt = i_pair[0];
                    let j_virt = i_pair[1];
                    let i_loc_virt = i_virt - lumo_min;
                    let j_loc_virt = j_virt - lumo_min;
                    let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                    let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                    let [dij,occij] = denominator[[i_loc_virt,j_loc_virt]];
                    

                    let tmp_energy_term_os = if occij.abs() > 1.0e-6 {
                        vij.powf(2.0f64)/(dij + enhanced_factor*previous_eij_os)*occij
                    } else {
                        0.0
                    };
                    s.send(tmp_energy_term_os).unwrap()
                });
                receiver.into_iter().for_each(|tmp_energy_term_os| {
                    eij_os += tmp_energy_term_os;
                });
            }
        } else {
            if is_same_spin {
                let loc_virt_pair = mpi_ix.distribution_same_spin_virtual_orbital_pair(lumo_i, num_state);
                let (sender, receiver) = channel();
                loc_virt_pair.par_iter().for_each_with(sender,|s,i_pair| {
                //for i_virt in lumo_i..num_state {
                    let i_virt = i_pair[0];
                    let j_virt = i_pair[1];
                    let i_loc_virt = i_virt - lumo_min;
                    //for j_virt in i_virt+1..num_state {
                    let j_loc_virt = j_virt - lumo_min;
                    let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                    let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                    let [dij,occij] = denominator[[i_loc_virt,j_loc_virt]];

                    let tmp_energy_term_ss = if occij.abs() > 1.0e-6 {
                        let screening_enhanced = enhanced_factor*erfc(screening_factor*dij);
                        (vij-vji).powf(2.0f64)/(dij + screening_enhanced*(previous_eij_ss+shifted_factor))*occij
                        //eij_ss += tmp_energy_term_ss;
                    } else {
                        0.0
                    };
                    s.send(tmp_energy_term_ss).unwrap()
                });
                receiver.into_iter().for_each(|tmp_energy_term_ss| {
                        eij_ss += tmp_energy_term_ss;
                });
                //println!("debug: mp2_ss={:16.8}, eij_ss={:16.8}", previous_eij_ss, eij_ss);
            } else {
                let loc_virt_pair = mpi_ix.distribution_opposite_spin_virtual_orbital_pair(lumo_i, lumo_j, num_state, 0);
                let (sender, receiver) = channel();
                loc_virt_pair.par_iter().for_each_with(sender,|s,i_pair| {
                    let i_virt = i_pair[0];
                    let j_virt = i_pair[1];
                    let i_loc_virt = i_virt - lumo_min;
                    let j_loc_virt = j_virt - lumo_min;
                    let vij = eri_virt[[i_loc_virt,j_loc_virt]];
                    let vji = eri_virt[[j_loc_virt,i_loc_virt]];

                    let [dij,occij] = denominator[[i_loc_virt,j_loc_virt]];

                    let tmp_energy_term_os = if occij.abs() > 1.0e-6 {
                        let screening_enhanced = enhanced_factor*erfc(screening_factor*dij);
                        vij.powf(2.0f64)/(dij + screening_enhanced*(previous_eij_os+shifted_factor))*occij
                        //eij_os += tmp_energy_term_os;
                    } else {
                        0.0
                    };
                     s.send(tmp_energy_term_os).unwrap()
                });
                receiver.into_iter().for_each(|tmp_energy_term_os| {
                    eij_os += tmp_energy_term_os;
                });
            }
        }

        // sum up the ss and os contribution from the mpi tasks.
        eij_ss = mpi_reduce(&mpi_op.world, &mut [eij_ss], 0, &SystemOperation::sum())[0];
        mpi_broadcast(&mpi_op.world, &mut eij_ss, 0);
        eij_os = mpi_reduce(&mpi_op.world, &mut [eij_os], 0, &SystemOperation::sum())[0];
        mpi_broadcast(&mpi_op.world, &mut eij_os, 0);

        (eij_ss, eij_os)
    } else {
        open_shell_eij_serial(eri_virt, denominator, previous_eij_ss, previous_eij_os, enhanced_factor, screening_factor, shifted_factor, lumo_i, lumo_j, is_same_spin, lumo_min, num_state)
    }
} 