use std::{sync::mpsc::channel, num};

use libm::{erf, erfc, sqrt};
use num_traits::{abs, Float};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator, IndexedParallelIterator};
use statrs::statistics::Max;
use tensors::{MatrixFull, MathMatrix, BasicMatrix, matrix_blas_lapack::_dgemm_full};

use crate::{scf_io::{SCF,scf, SCFType}, utilities::{self, TimeRecords}};
use crate::constants::{PI, E, INVERSE_THRESHOLD};
use crate::mpi_io::MPIOperator;
use tensors::matrix_blas_lapack::{_dgemm,_dsyev};

use super::{trans_gauss_legendre_grids, gauss_legendre_grids, logarithmic_grid};

pub fn evaluate_spin_response_rayon(scf_data: &SCF, freq: f64) -> anyhow::Result<Vec<MatrixFull<f64>>> {

    let num_auxbas = scf_data.mol.num_auxbas;
    let num_basis = scf_data.mol.num_basis;
    let num_state = scf_data.mol.num_state;
    let start_mo = scf_data.mol.start_mo;
    let spin_channel = scf_data.mol.spin_channel;
    let num_spin = spin_channel as f64;
    let frac_spin_occ = num_spin / 2.0f64;
    let mut spin_polar_freq: Vec<MatrixFull<f64>> = vec![MatrixFull::empty();2];

    if let Some(ri3mo_vec) = &scf_data.ri3mo {
        for i_spin in 0..spin_channel {
            let mut polar_freq = spin_polar_freq.get_mut(i_spin).unwrap();
            *polar_freq = MatrixFull::new([num_auxbas,num_auxbas], 0.0);
            let eigenvector = match scf_data.scftype { SCFType::RHF | SCFType::UHF => scf_data.eigenvectors.get(i_spin).unwrap(),
                SCFType::ROHF => scf_data.semi_eigenvectors.get(i_spin).unwrap()
            };
            let eigenvalues = match scf_data.scftype { SCFType::RHF | SCFType::UHF => scf_data.eigenvalues.get(i_spin).unwrap(),
                SCFType::ROHF => scf_data.semi_eigenvalues.get(i_spin).unwrap()
            };
            let occ_numbers = scf_data.occupation.get(i_spin).unwrap();
            let homo = scf_data.homo.get(i_spin).unwrap().clone();
            let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
            //let num_occu = homo + 1;
            //let num_occu = lumo;
            //let num_occu = scf_data.mol.num_elec.get(i_spin + 1).unwrap().clone() as usize;
            let num_occu = if scf_data.mol.num_elec[i_spin+1] <= 1.0e-6 {0} else {homo + 1};

            let (ri3mo, vir_range, occ_range) = ri3mo_vec.get(i_spin).unwrap();
            
            for j_state in start_mo..num_occu {
                let j_state_eigen = eigenvalues[j_state];
                let j_state_occ = occ_numbers[j_state];
                let j_loc_state = j_state - occ_range.start;
                let rimo_j = ri3mo.get_reducing_matrix(j_loc_state).unwrap();
                let mut tmp_matrix = MatrixFull::new([num_auxbas,vir_range.len()],0.0);

                let num_threads = rayon::current_num_threads();
                let tasks = utilities::balancing(num_state-lumo, num_threads);
                //println!("debug tasks: {:?}", &tasks);

                let (sender, receiver) = channel();

                tasks.par_iter().for_each_with(sender, |s, tasks| {
                    let mut tmp_matrix_loc = MatrixFull::new([num_auxbas,vir_range.len()],0.0);
                    tasks.clone().for_each(|k_state_loc| {
                        let k_state = k_state_loc + lumo;
                        let k_state_eigen = eigenvalues.get(k_state).unwrap();
                        let k_state_occ = occ_numbers.get(k_state).unwrap();

                        let mut energy_gap = j_state_eigen - k_state_eigen;
                        if energy_gap < 1.0e-6 && energy_gap >= 0.0 {
                            energy_gap += 1.0e-6
                        } else if energy_gap >-1.0e-6 && energy_gap < 0.0 {
                            energy_gap += -1.0e-6
                        };
                        //=======================================================================
                        // fractional occupation scheme suggested by Weitao Yang, which was 
                        // derived from the ensemble of Green function
                        //=======================================================================
                        let zeta = 2.0f64*energy_gap / (energy_gap.powf(2.0) + freq*freq)*
                            (j_state_occ*frac_spin_occ)*(1.0f64-k_state_occ*frac_spin_occ);

                        let k_loc_state = k_state - vir_range.start;
                        //timerecords.count_start("submatrix");
                        let from_iter = ri3mo.get_slices(0..num_auxbas, k_loc_state..k_loc_state+1, j_loc_state..j_loc_state+1);
                        let to_iter = tmp_matrix_loc.iter_submatrix_mut(0..num_auxbas,k_loc_state..k_loc_state+1);
                        to_iter.zip(from_iter).for_each(|(to, from)| {
                            *to = from * zeta
                        });

                    });
                    s.send(tmp_matrix_loc).unwrap();
                });

                receiver.into_iter().for_each(|tmp_matrix_loc| {
                    tmp_matrix += tmp_matrix_loc
                });

                _dgemm_full(&tmp_matrix, 'N', &rimo_j, 'T', polar_freq, 1.0, 1.0);
            };
        }
    } else {
        panic!("RI3MO should be initialized before the RPA calculations")
    };

    Ok(spin_polar_freq)

}

fn screening_de_excitation(dij: f64, freq: f64, a: f64, b: f64, sigma: f64, scale: f64) -> f64 {
    let r2 = sqrt(2.0);
    erfc(freq)*scale*0.25*(1.0+erf((dij-a)/sigma/r2))*(1.0+erf(b-dij)/sigma/r2)
}

pub fn evaluate_spin_response_serial(scf_data: &SCF, freq: f64) -> anyhow::Result<Vec<MatrixFull<f64>>> {

    //let mut timerecords = TimeRecords::new();
    //timerecords.new_item("all", "all exclude dgemm");
    //timerecords.new_item("submatrix", "iter submatrix");
    //timerecords.new_item("dgemm", "dgemm");
    let [a,b, sigma, scaling_factor] = if let Some(value) = scf_data.mol.ctrl.rpa_de_excitation_parameters {
        value
    } else {
        [1.0, 1.0, 1.0, 0.0]
    };
    //let [a,b, sigma, scaling_factor] = [0.0, 0.02, 0.02, 0.1];

    let num_auxbas = scf_data.mol.num_auxbas;
    let num_basis = scf_data.mol.num_basis;
    let num_state = scf_data.mol.num_state;
    let start_mo = scf_data.mol.start_mo;
    let spin_channel = scf_data.mol.spin_channel;
    let num_spin = spin_channel as f64;
    let frac_spin_occ = num_spin / 2.0f64;
    //let mut polar_freq = MatrixFull::new([num_auxbas,num_auxbas],0.0);
    let mut spin_polar_freq: Vec<MatrixFull<f64>> = vec![MatrixFull::empty();2];

    if let Some(ri3mo_vec) = &scf_data.ri3mo {
        for i_spin in 0..spin_channel {
            let mut polar_freq = spin_polar_freq.get_mut(i_spin).unwrap();
            *polar_freq = MatrixFull::new([num_auxbas,num_auxbas], 0.0);
            let eigenvector = match scf_data.scftype { SCFType::RHF | SCFType::UHF => scf_data.eigenvectors.get(i_spin).unwrap(),
                SCFType::ROHF => scf_data.semi_eigenvectors.get(i_spin).unwrap()
            }; 
            let eigenvalues = match scf_data.scftype { SCFType::RHF | SCFType::UHF => scf_data.eigenvalues.get(i_spin).unwrap(),
                SCFType::ROHF => scf_data.semi_eigenvalues.get(i_spin).unwrap()
            };
            let occ_numbers = scf_data.occupation.get(i_spin).unwrap();
            let homo = scf_data.homo.get(i_spin).unwrap().clone();
            let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
            //let num_occu = homo + 1;
            //let num_occu = lumo;
            //let num_occu = scf_data.mol.num_elec.get(i_spin + 1).unwrap().clone() as usize;
            let num_occu = if scf_data.mol.num_elec[i_spin+1] <= 1.0e-6 {0} else {homo + 1};

            let (ri3mo, vir_range, occ_range) = ri3mo_vec.get(i_spin).unwrap();

            for j_state in start_mo..num_occu {
                let j_state_eigen = eigenvalues[j_state];
                let j_state_occ = occ_numbers[j_state];
                let j_loc_state = j_state - occ_range.start;
                if j_state_occ >= 1.0e-6 {
                    let rimo_j = ri3mo.get_reducing_matrix(j_loc_state).unwrap();
                    let mut tmp_matrix = MatrixFull::new([num_auxbas,vir_range.len()],0.0);
                    for k_state in lumo..num_state {
                        let k_state_eigen = eigenvalues.get(k_state).unwrap();
                        let k_state_occ = occ_numbers.get(k_state).unwrap();
                        if (1.0-k_state_occ*frac_spin_occ).abs() >=1.0e-6 {
                            //let zeta = num_spin*(j_state_eigen-k_state_eigen) /
                            //    ((j_state_eigen-k_state_eigen).powf(2.0) + freq*freq)*
                            //    (j_state_occ-k_state_occ);
                            let mut energy_gap = j_state_eigen - k_state_eigen;
                            if energy_gap < 1.0e-6 && energy_gap >=0.0 {
                                energy_gap = 1.0e-6
                            } else if energy_gap >-1.0e-6 && energy_gap < 0.0 {
                                energy_gap = -1.0e-6
                            };
                            //=======================================================================
                            // fractional occupation scheme suggested by Weitao Yang, which was 
                            // derived from the ensemble of Green function
                            //=======================================================================
                            let level_shift = screening_de_excitation(energy_gap, freq, a, b, sigma, scaling_factor);
                            //let level_shift = 0.0;
                            let zeta = 2.0f64*(energy_gap+level_shift)/ ((energy_gap+level_shift).powf(2.0) + freq*freq)*
                                (j_state_occ*frac_spin_occ)*(1.0f64-k_state_occ*frac_spin_occ);
                            //=======================================================================
                            // fractional occupation scheme suggested by Xinguo
                            //=======================================================================
                            //let zeta = num_spin*energy_gap / (energy_gap.powf(2.0) + freq*freq)*
                            //    (j_state_occ-k_state_occ);

                            if level_shift > 1.0e-3 {
                                println!("elec_pair: ({:2},{:2}), occ: ({:6.4},{:6.4}), frq: {:12.6}, gap: {:12.6}, lsf: {:12.6}, zeta: {:12.6}", 
                                        j_state, k_state, j_state_occ, k_state_occ, freq, energy_gap, level_shift, zeta);
                            }

                            let k_loc_state = k_state - vir_range.start;
                            //timerecords.count_start("submatrix");
                            let from_iter = ri3mo.get_slices(0..num_auxbas, k_loc_state..k_loc_state+1, j_loc_state..j_loc_state+1);
                            let to_iter = tmp_matrix.iter_submatrix_mut(0..num_auxbas,k_loc_state..k_loc_state+1);
                            to_iter.zip(from_iter).for_each(|(to, from)| {
                                *to = from * zeta
                            });
                        }
                        //timerecords.count("submatrix");
                    }
                    _dgemm_full(&tmp_matrix, 'N', &rimo_j, 'T', polar_freq, 1.0, 1.0);
                }
                //timerecords.count("dgemm");
                //timerecords.count("all");
            };

            //if i_spin == 1 {println!("debug polar_freq_beta_1: {:?}", &polar_freq)};

            //receiver.into_iter().for_each(|loc_polar_freq| {
            //    polar_freq += loc_polar_freq
            //})

            //timerecords.report_all();

        }
    } else {
        panic!("RI3MO should be initialized before the RPA calculations")
    };

    //println!("debug spin_polar_freq: {:?}", &spin_polar_freq);

    Ok(spin_polar_freq)

}

pub fn evaluate_special_radius(polar_freq: &MatrixFull<f64>) -> f64 {

    let (_, eigenvalues, non_sigular) = _dsyev(polar_freq, 'N');

    //let special_radius = eigenvalues.iter().map(|x| x.abs()).max().unwrap();

    //eigenvalues.iter().map(|x| (*x).abs()).collect::<Vec<f64>>().max()
    eigenvalues.iter().fold(0.0, |x, y| x.abs().max(y.abs()))
}

pub fn evaluate_special_radius_only(scf_data: &SCF) -> anyhow::Result<[f64;2]>  {
    let spin_channel = scf_data.mol.spin_channel;
    //let spin_polar_freq = evaluate_spin_response_serial(scf_data, 0.0).unwrap();
    let spin_polar_freq = evaluate_spin_response_rayon(scf_data, 0.0).unwrap();

    let mut special_radius = [0.0f64; 2];
    let mut sc_check = [false; 2];

    for i_spin in 0..spin_channel {
        let polar_freq = spin_polar_freq.get(i_spin).unwrap();
        special_radius[i_spin] = evaluate_special_radius(polar_freq);
        sc_check[i_spin] = special_radius[i_spin] > 0.8f64;
    }

    if spin_channel == 1 {
        sc_check = [false;2];
        let mut tmp_sr = special_radius[0];
        special_radius[1] = tmp_sr;
    }

    Ok(special_radius)
}

pub fn evaluate_osrpa_correlation_rayon_mpi(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) -> anyhow::Result<[f64;3]>  {
    if let (Some(mpi_op), Some(mpi_ix)) = (mpi_operator, &scf_data.mol.mpi_data) {
        panic!("The MPI implementation is not yet available for the SCSRPA evaluation")
        //let (rpa_c, _) = evaluate_osrpa_correlation_detailed_rayon_mpi(scf_data, mpi_op, mpi_ix).unwrap();
        //Ok(rpa_c)
    } else {
        let (rpa_c, _) = evaluate_osrpa_correlation_detailed_rayon(scf_data).unwrap();
        Ok(rpa_c)
    }
}

pub fn evaluate_osrpa_correlation_rayon(scf_data: &SCF) -> anyhow::Result<[f64;3]>  {
    let (rpa_c, _) = evaluate_osrpa_correlation_detailed_rayon(scf_data).unwrap();
    Ok(rpa_c)
}

pub fn evaluate_osrpa_correlation_detailed_rayon(scf_data: &SCF) -> anyhow::Result<([f64;3],[f64;2])>  {

    let mut rpa_c_energy = 0.0_f64;
    let mut rpa_c_energy_os = 0.0_f64;
    let mut rpa_c_energy_ss = 0.0_f64;

    let spin_channel = scf_data.mol.spin_channel;
    let freq_grid_type = scf_data.mol.ctrl.freq_grid_type;
    let num_freq = scf_data.mol.ctrl.frequency_points;
    let max_freq = scf_data.mol.ctrl.freq_cut_off;
    let mut sp = format!("The frequency integration is tabulated by {:3} grids using", num_freq);
    let (omega,weight) = if freq_grid_type==0 {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        trans_gauss_legendre_grids(1.0, num_freq)
    } else if freq_grid_type==1 {
        sp = format!("{} the standard Gauss-Legendre grids",sp);
        gauss_legendre_grids([0.0,max_freq], num_freq)
    } else if freq_grid_type== 2 {
        sp = format!("{} the logarithmic grids",sp);
        logarithmic_grid([0.0,max_freq], num_freq)
    } else {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        trans_gauss_legendre_grids(1.0, num_freq)
    };
    if scf_data.mol.ctrl.print_level>1 {
        println!("{}", sp);
    }

    let num_lambda = scf_data.mol.ctrl.lambda_points;;
    //let (lambda_omega,lambda_weight) = trans_gauss_legendre_grids(1.0, num_lambda);
    let (lambda_omega,lambda_weight) = gauss_legendre_grids([0.0,1.0], num_lambda);

    let mut timerecords = TimeRecords::new();
    timerecords.new_item("evaluate_special_radius", "");
    timerecords.new_item("evaluate_spin_response_serial", "");

    timerecords.count_start("evaluate_spin_response_serial");
    let spin_polar_freq = evaluate_spin_response_serial(scf_data, 0.0).unwrap();
    timerecords.count("evaluate_spin_response_serial");

    let mut special_radius = [0.0f64; 2];
    let mut sc_check = [false; 2];

    timerecords.count_start("evaluate_special_radius");
    for i_spin in 0..spin_channel {
        //println!("debug: i_spin = {}", i_spin);
        let polar_freq = spin_polar_freq.get(i_spin).unwrap();
        //println!("debug polar_freq = {:?}", polar_freq);
        special_radius[i_spin] = evaluate_special_radius(polar_freq);
        sc_check[i_spin] = special_radius[i_spin] > 0.8f64;
    }
    timerecords.count("evaluate_special_radius");

    if scf_data.mol.ctrl.print_level>0 {timerecords.report_all()};

    if spin_channel == 1 {
        sc_check = [false;2];
        let mut tmp_sr = special_radius[0];
        special_radius[1] = tmp_sr;
    }
    if scf_data.mol.ctrl.print_level>0 {
        println!("Special radius of non-interacting response matrix: ({:16.8}, {:16.8})", special_radius[0], special_radius[1]);
    }

    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    let mut per_omp_num_threads = default_omp_num_threads/num_freq;
    if per_omp_num_threads == 0 {per_omp_num_threads = 1};
    //if default_omp_num_threads%num_freq != 0 {per_omp_num_threads += 1};

    let (sender,receiver) = channel();
    omega.par_iter().zip(weight.par_iter())
        .for_each_with(sender, |s, (omega,weight)| {
        utilities::omp_set_num_threads_wrapper(per_omp_num_threads);
        let mut response_freq = evaluate_spin_response_serial(scf_data, *omega).unwrap();
        //if scf_data.mol.spin_channel == 1 {
        //    response_freq *= 2.0;
        //}
        let [rpa_c_integrand,rpa_c_integrand_os, rpa_c_integrand_ss] = 
            evaluate_osrpa_integrand(&mut response_freq, spin_channel, &lambda_omega,&lambda_weight,&sc_check);

        if scf_data.mol.ctrl.print_level>1 {println!(" (freq, weight, rpa_c): {:16.8},{:16.8},{:16.8}, {:16.8}, {:16.8}", 
            omega, weight, rpa_c_integrand, rpa_c_integrand_os, rpa_c_integrand_ss)};

        //rpa_c_energy += rpa_c_integrand*weight;

        s.send((rpa_c_integrand*weight,rpa_c_integrand_os*weight, rpa_c_integrand_ss*weight)).unwrap()

    });

    receiver.into_iter().for_each(|(rpa_c_integrand, rpa_c_integrand_os, rpa_c_integrand_ss)| {
        rpa_c_energy += rpa_c_integrand;
        rpa_c_energy_os += rpa_c_integrand_os;
        rpa_c_energy_ss += rpa_c_integrand_ss;
    });

    //rpa_c_energy = receiver.into_iter().sum();

    rpa_c_energy = rpa_c_energy*0.5/PI;
    rpa_c_energy_os = rpa_c_energy_os*0.5/PI;
    rpa_c_energy_ss = rpa_c_energy_ss*0.5/PI;

    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    // for scs-rpa, higher oder OS terms are combined with the SS term to be the SS+ term
    // SS+ = RPA_total - RPA_OS
    Ok(([rpa_c_energy, rpa_c_energy_os, rpa_c_energy_ss],special_radius))
}

fn evaluate_osrpa_integrand(
    spin_polar_freq: &mut Vec<MatrixFull<f64>>, spin_channel: usize, 
    lambda_omega: &Vec<f64>, lambda_weight: &Vec<f64>,
    sc_check: &[bool;2]
) -> [f64;3] {
    let mut rpa_c_integrand = 0.0;
    let mut rpa_c_integrand_ss = 0.0;
    let mut rpa_c_integrand_os = 0.0;

    if spin_channel ==1 {
        let mut polar_freq = spin_polar_freq.get_mut(0).unwrap();
        //=============================================================
        // for the dRPA calculations
        //=============================================================
        {
            let mut full_polar_freq = polar_freq.clone();
            full_polar_freq *= 2.0;

            let num_auxbas = full_polar_freq.size.get(0).unwrap();

            let trace_v_times_polar = full_polar_freq.iter_diagonal().unwrap()
                .fold(0.0, |acc,value| acc+(*value));

            //let mut tmp_v = polar_freq.get_diagonal_terms_mut().unwrap();
            //tmp_v.iter_mut().for_each(|data| **data -= 1.0);
            full_polar_freq.iter_diagonal_mut().unwrap().for_each(|data| *data -= 1.0);
            //let mut tmp_v = polar_freq.get_diagonal_terms_mut().unwrap();
            //tmp_v.iter_mut().for_each(|data| **data -= 1.0);
            full_polar_freq *= -1.0;

            full_polar_freq = full_polar_freq.to_matrixfullslicemut().lapack_dgetrf().unwrap();

            //let mut det_v_times_polar = v_times_polar.get_diagonal_terms().unwrap().iter()
            //    .fold(1.0,|acc,value| acc*(*value)
            //);
            let mut det_v_times_polar = full_polar_freq.iter_diagonal().unwrap()
                .fold(1.0,|acc,value| acc*(*value)
            );
            if det_v_times_polar<0.0 {println!("WARNING: Determinant of V_TIMES_POLAR is negetive !")};
            //println!("debug {:16.8}, {:16.8}",trace_v_times_polar, det_v_times_polar);
            rpa_c_integrand = det_v_times_polar.abs().log(E) + trace_v_times_polar;
        }
        //=============================================================
        // for the os-RPA and ss-RPA calculations
        //=============================================================
        {
            let num_auxbas = polar_freq.size.get(0).unwrap();

            let trace_v_times_polar = polar_freq.iter_diagonal().unwrap()
                .fold(0.0, |acc,value| acc+2.0*(*value));

            //let mut tmp_v = polar_freq.get_diagonal_terms_mut().unwrap();
            //tmp_v.iter_mut().for_each(|data| **data -= 1.0);
            polar_freq.iter_diagonal_mut().unwrap().for_each(|data| *data -= 1.0);
            //let mut tmp_v = polar_freq.get_diagonal_terms_mut().unwrap();
            //tmp_v.iter_mut().for_each(|data| **data -= 1.0);
            polar_freq.self_multiple(-1.0);

            let v_times_polar = polar_freq.to_matrixfullslicemut().lapack_dgetrf().unwrap();

            //let mut det_v_times_polar = v_times_polar.get_diagonal_terms().unwrap().iter()
            //    .fold(1.0,|acc,value| acc*(*value)
            //);
            let mut det_v_times_polar = v_times_polar.iter_diagonal().unwrap()
                .fold(1.0,|acc,value| acc*(*value)
            );
            if det_v_times_polar<0.0 {println!("WARNING: Determinant of V_TIMES_POLAR is negetive !")};
            //println!("debug {:16.8}, {:16.8}",trace_v_times_polar, det_v_times_polar);
            rpa_c_integrand_os = 2.0*det_v_times_polar.abs().log(E) + trace_v_times_polar;

            // for the close-shell cases, ss-rpa has the same contribution of the os-rpa at the first order
            rpa_c_integrand_ss = rpa_c_integrand_os;

        }
    } else {
        let mut trace_v_times_polar = 0.0f64;
        //=============================================================
        // for the dRPA calculations
        //=============================================================
        {
            let mut polar_freq_alpha = spin_polar_freq.get(0).unwrap();
            let mut polar_freq_beta = spin_polar_freq.get(1).unwrap();
            let mut full_polar_freq = polar_freq_alpha.add(polar_freq_beta).unwrap();

            let num_auxbas = full_polar_freq.size.get(0).unwrap();

            trace_v_times_polar = full_polar_freq.iter_diagonal().unwrap()
                .fold(0.0, |acc,value| acc+(*value));

            full_polar_freq.iter_diagonal_mut().unwrap().for_each(|data| *data -= 1.0);
            full_polar_freq *= -1.0;

            full_polar_freq = full_polar_freq.to_matrixfullslicemut().lapack_dgetrf().unwrap();

            let mut det_v_times_polar = full_polar_freq.iter_diagonal().unwrap()
                .fold(1.0,|acc,value| acc*(*value)
            );
            if det_v_times_polar<0.0 {println!("WARNING: Determinant of V_TIMES_POLAR is negetive !")};
            rpa_c_integrand = det_v_times_polar.abs().log(E) + trace_v_times_polar;
        }
        //=============================================================
        // for the ss-RPA calculations
        //=============================================================
        rpa_c_integrand_ss = trace_v_times_polar;
        for i_spin in 0..spin_channel {
            let mut polar_freq= spin_polar_freq.get(i_spin).unwrap().clone();
            polar_freq.iter_diagonal_mut().unwrap().for_each(|data| *data -= 1.0);
            polar_freq *= -1.0;

            polar_freq = polar_freq.to_matrixfullslicemut().lapack_dgetrf().unwrap();
            let mut det_v_times_polar = polar_freq.iter_diagonal().unwrap()
                .fold(1.0,|acc,value| acc*(*value)
            );
            if det_v_times_polar<0.0 {println!("WARNING: Determinant of V_TIMES_POLAR is negetive !")};
            rpa_c_integrand_ss += det_v_times_polar.abs().log(E);
        }
        //=============================================================
        // for the os-RPA calculations
        //=============================================================
        for i_spin in 0..spin_channel {
            let j_spin = if i_spin == 0 {1usize} else {0usize};
            let rpa_c_integrand_os_i = evaluate_osrpa_response(i_spin, j_spin, 
                &spin_polar_freq, &lambda_omega, &lambda_weight, &sc_check);
            rpa_c_integrand_os += rpa_c_integrand_os_i;
            //println!("debug: osRPA correlation ({}-spin): {:16.8} Ha", i_spin, rpa_c_integrand_os_i);
        }

    }

    [rpa_c_integrand, rpa_c_integrand_os, rpa_c_integrand_ss]
}

pub fn evaluate_osrpa_response(
    i_spin: usize, j_spin:usize,
    spin_polar_freq: &Vec<MatrixFull<f64>>, 
    lambda_omega: &Vec<f64>, lambda_weight: &Vec<f64>, sc_check: &[bool;2]
) -> f64 {

    let mut rpa_c_integrand_spin = 0.0f64;

    let c_osrpa_threshold = 0.8f64;

    let polar_freq_i = spin_polar_freq.get(i_spin).unwrap();
    let polar_freq_j = spin_polar_freq.get(j_spin).unwrap();

    let mut temp_v_times_polar_a = polar_freq_j.clone();
    //temp_v_times_polar_a *= 2.0;
    //temp_v_times_polar_a.iter_diagonal_mut().unwrap().for_each(|x| {*x = *x-1.0f64});
    //temp_v_times_polar_a.iter_mut().for_each(|x| {*x = -1.0f64*x});

    let num_auxbas = temp_v_times_polar_a.size()[0];

    let mut temp_v_times_polar_b = MatrixFull::new([num_auxbas, num_auxbas], 0.0f64);

    let special_radius = if sc_check[j_spin] {
        evaluate_special_radius(polar_freq_j)
    } else {
        0.0f64
    };

    if special_radius < c_osrpa_threshold {
        let mut i_term = 1.0f64;
        _dgemm(
            &temp_v_times_polar_a, (0..num_auxbas,0..num_auxbas), 'N', 
            polar_freq_i, (0..num_auxbas,0..num_auxbas), 'N', 
            &mut temp_v_times_polar_b, (0..num_auxbas,0..num_auxbas), 1.0,0.0);

        let mut delta_trace = temp_v_times_polar_b.iter_diagonal().unwrap().fold(0.0,|acc,x| acc+x);
        delta_trace = -1.0f64/(i_term +1.0f64)*delta_trace;

        rpa_c_integrand_spin += delta_trace;

        while (delta_trace.abs()>1.0e-8 && (i_term as usize) < 300) {
            i_term = i_term + 1.0f64;
            _dgemm(
                polar_freq_j, (0..num_auxbas,0..num_auxbas), 'N', 
                &temp_v_times_polar_b, (0..num_auxbas,0..num_auxbas), 'N', 
                &mut temp_v_times_polar_a, (0..num_auxbas,0..num_auxbas), 1.0,0.0);
                 
            delta_trace = temp_v_times_polar_a.iter_diagonal().unwrap().fold(0.0,|acc,x| acc+x);
            delta_trace = -1.0f64/(i_term +1.0f64)*delta_trace;

            rpa_c_integrand_spin += delta_trace;

            temp_v_times_polar_b.iter_mut().zip(temp_v_times_polar_a.iter()).for_each(|(b,a)| *b = *a);
            //println!("debug: delta_trace = {:16.8} after {:4.1} steps", delta_trace, i_term);
        }
        if (i_term as usize) >=100  {println!("WARNNING: delta_trace = {:16.8} after {:4.1} steps", delta_trace, i_term)};
    } else {
        //if special_radius>=1.0f64 {
        //    println!("Special radius of non-interacting response matrix in the {}-spin channel >=1.0 ({:16.8})",
        //        j_spin,special_radius);
        //    println!("Strong correlation breaks the perturbative OS-type particle-hole summation in the {}-spin channel", j_spin);
        //} else {
        //    println!("Special radius of non-interacting response matrix in the {}-spin channel >={} ({:16.8})",
        //        j_spin,c_osrpa_threshold,special_radius);
        //    println!("Strong correlation is large in the OS-type particle-hole summation in the {}-spin channel", j_spin);
        //}

        //rpa_c_integrand_spin = 0.0f64;
        //let transpose_polar_freq_i = polar_freq_i.transpose();
        rpa_c_integrand_spin = polar_freq_i.iter_diagonal().unwrap().fold(0.0, |acc,a| acc + a);

        let mut polar_transform = MatrixFull::new([num_auxbas,num_auxbas],0.0f64);

        lambda_omega.iter().zip(lambda_weight.iter()).for_each(|(omega,weight)| {
            temp_v_times_polar_b.iter_mut().zip(polar_freq_j.iter()).for_each(|(b,j)| *b = -j*omega);

            temp_v_times_polar_b.iter_diagonal_mut().unwrap().for_each(|x| *x += 1.0f64);

            //temp_v_times_polar_a = temp_v_times_polar_b.to_matrixfullslicemut().lapack_power(-1.0, INVERSE_THRESHOLD).unwrap();
            temp_v_times_polar_a = temp_v_times_polar_b.to_matrixfullslicemut().lapack_inverse().unwrap();

            //polar_transform += temp_v_times_polar_a*(*weight);
            polar_transform.iter_mut().zip(temp_v_times_polar_a.iter()).for_each(|(to,from)| {
                *to += from * weight
            });
        });

        //rpa_c_integrand_spin -= polar_transform.iter_columns_full().zip(transpose_polar_freq_i.iter_columns_full())
        //    .fold(0.0f64, |acc, (a,b)| {
        //        acc + a.iter().zip(b.iter()).fold(0.0,|acc, (a,b)| acc+a*b)
        //    })
        //rpa_c_integrand_spin -= polar_transform.iter().zip(transpose_polar_freq_i.iter())
        //    .fold(0.0f64, |acc, (a,b)| {
        //        acc + a*b
        //    });
        for i_basis in 0..num_auxbas {
            rpa_c_integrand_spin -= polar_transform.iter_column(i_basis).zip(polar_freq_i.iter_row(i_basis))
                .fold(0.0, |acc,(tp,pi)| acc + tp*pi);
        }
        //_dgemm(
        //    &polar_transform, (0..num_auxbas,0..num_auxbas), 'N',
        //    polar_freq_i,(0..num_auxbas,0..num_auxbas),'N',
        //    &mut temp_v_times_polar_b,(0..num_auxbas,0..num_auxbas),1.0,0.0);
        //rpa_c_integrand_spin -= temp_v_times_polar_b.iter_diagonal().unwrap().fold(0.0, |acc,x| acc +x);
    }

    rpa_c_integrand_spin

}

#[test]
fn test_absmax() {
    let dd = vec![1.0,-2.0,3.0,-4.3, -6.0, 5.0];
    let ff = dd.iter().map(|x| (*x).abs()).collect::<Vec<f64>>().max();


    println!("abs_max = {}", ff);

    let dd = vec![1.0,-2.0,3.0,-4.3];
    let md = MatrixFull::from_vec([2,2], dd).unwrap();
    let td = md.transpose();
     
    let sd = md.iter_columns_full().zip(td.iter_columns_full()).fold(0.0,|acc,(m,t)| acc + m.iter().zip(t.iter()).fold(0.0,|acc, (a,b)| acc+a*b));
    println!("{}", sd);

}
