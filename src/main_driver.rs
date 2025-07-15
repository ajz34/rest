#![allow(unused)]
extern crate rest_tensors as tensors;
//extern crate rest_libxc as libxc;
extern crate chrono as time;
extern crate hdf5_metno as hdf5;
use std::{f64, fs::File, io::Write};
use std::path::PathBuf;
use crate::basis_io::ecp::ghost_effective_potential_matrix;
use crate::external_field::num_dipole::numerical_dipole;
use crate::geom_io::GeomUnit;
use num_traits::Pow;
use pyo3::prelude::*;
use autocxx::prelude::*;
use crate::ctrl_io::JobType;
use crate::constants::{ANG, AU2DEBYE};
use crate::scf_io::{scf_without_build, SCFType, SCF};
use tensors::{MathMatrix, MatrixFull};
use crate::{utilities, ri_pt2, ri_rpa, dft, scf_io, post_scf_analysis};

//use rayon;
use crate::constants::EV;
use crate::grad::{formated_force, numerical_force};
use crate::initial_guess::enxc::{effective_nxc_matrix, effective_nxc_tensors};
//use crate::grad::rhf::Gradient;
use crate::initial_guess::sap::*;

use anyhow;
//use crate::isdf::error_isdf;
use crate::dft::DFA4REST;
use crate::post_scf_analysis::mulliken::mulliken_pop;
//use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile};
use crate::scf_io::{initialize_scf, scf};
use time::{DateTime,Local};
use crate::molecule_io::Molecule;
//use crate::isdf::error_isdf;
//use crate::dft::DFA4REST;
use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile, rand_wf_real_space, cube_build, molden_build, post_ai_correction};
use liblbfgs::{lbfgs,Progress};
use crate::mpi_io::{MPIOperator,MPIData};

//use crate::mpi_io::initialization;


pub fn main_driver() -> anyhow::Result<()> {

                                  
    //let mut time_mark = utilities::TimeRecords::new();
    //time_mark.new_item("Overall", "the whole job");
    //time_mark.new_item("SCF", "the scf procedure");




    // VERY IMPORTANCE: introduce mpi_operator:
    let (mpi_operator , mut mpi_data)= MPIData::initialization();

    let ctrl_file = utilities::parse_input().value_of("input_file").unwrap_or("ctrl.in").to_string();
    if ! PathBuf::from(ctrl_file.clone()).is_file() {
        panic!("Input file ({:}) does not exist", ctrl_file);
    }
    let mut mol = Molecule::build(ctrl_file, mpi_data)?;
    if mol.ctrl.print_level>0 {println!("Molecule_name: {}", &mol.geom.name)};
    if mol.ctrl.print_level>=2 {
        println!("{}", mol.ctrl.formated_output_in_toml());
    }
    let mut time_mark = initialize_time_record(&mol);
    time_mark.count_start("Overall");

    if mol.ctrl.deep_pot {
        //let mut scf_data = scf_io::SCF::build(&mut mol);
        let mut effective_hamiltonian = mol.int_ij_matrixupper(String::from("hcore"));
        //effective_hamiltonian.formated_output(5, "full");
        let effective_nxc = effective_nxc_matrix(&mut mol);
        effective_nxc.formated_output(5, "full");
        effective_hamiltonian.data.iter_mut().zip(effective_nxc.data.iter()).for_each(|(to,from)| {*to += from});

        let mut ecp = mol.int_ij_matrixupper(String::from("ecp"));
        //println!("ecp: {:?}", ecp);
        ecp.iter_mut().zip(effective_nxc.iter()).for_each(|(to, from)| {*to -= from});

        ecp.formated_output(5, "full");
        let acc_error = ecp.iter().fold(0.0, |acc, x| {acc + x.abs()});
        println!("acc_error: {}", acc_error);
        
        return Ok(())
    }

    if mol.ctrl.bench_eps {
        let ecp = mol.int_ij_matrixupper(String::from("ecp"));
        let enxc = effective_nxc_matrix(&mut mol);
        let gep = ghost_effective_potential_matrix(
            &mol.cint_env, &mol.cint_atm, &mol.cint_bas, &mol.cint_type, mol.num_basis, 
            &mol.geom.ghost_ep_path, &mol.geom.ghost_ep_pos);

        let d12 = ecp.data.iter().zip(enxc.data.iter()).fold(0.0, |acc, dt| {acc + (dt.0 -dt.1).powf(2.0)});
        let d13 = ecp.data.iter().zip(gep.data.iter()).fold(0.0, |acc, dt| {acc + (dt.0 -dt.1).powf(2.0)});
        let num_data = ecp.data.len() as f64;
        println!("Compare between ECP, ENXC and GEP with the matrix sizes of");
        println!(" {:?}, {:?}, and {:?}, respectively", ecp.size(), enxc.size(), gep.size());
        println!("RMSDs between (ECP, ENXC) and (ECP, GEP): ({:16.8}, {:16.8})", 
            (d12/num_data).powf(0.5), (d13/num_data).powf(0.5)
        );

        return Ok(())
    }


    // initialize the time record
    // initialize the SCF procedure
    time_mark.count_start("SCF");
    let mut scf_data = scf_io::SCF::build(mol,&mpi_operator);
    time_mark.count("SCF");
    // perform the SCF and post SCF evaluation for the specified xc method
    performance_essential_calculations(&mut scf_data, &mut time_mark, &mpi_operator);

    let spin_correction_scheme: Option<String> = scf_data.mol.ctrl.spin_correction_scheme.clone();
    match spin_correction_scheme.as_deref() {
        Some("yamaguchi") => {
            println!("==========================================");
            println!("Now apply the Yamaguchi spin correction.");
            println!("==========================================");
            
            let scf_energy_singlet = scf_data.scf_energy;
            let tot_energy_singlet = scf_data.energies.get("xdh_energy").unwrap()[0]
            + scf_data.energies.get("ai_correction").map_or(0.0, |v| v[0]);
            let [square_spin_singlet, _] = scf_io::evaluate_spin_angular_momentum(&scf_data.density_matrix, &scf_data.ovlp, scf_data.mol.spin_channel, &scf_data.mol.num_elec);

            if scf_data.mol.ctrl.spin == 1.0 && square_spin_singlet >= 1e-3 {
                time_mark.count_start("spin_correction");
                println!("Computing the triplet energy...");
                //scf_data.mol.ctrl.spin = 3.0;
                //scf_data.mol.ctrl.spin_polarization = false;
                scf_data.mol.ctrl.guess_mix = false;
                scf_data.mol.num_elec[1] += 1.0;
                scf_data.mol.num_elec[2] -= 1.0;
                scf_data.mol.ctrl.initial_guess = String::from("inherit");
                scf_data.mol.ctrl.level_shift = Some(0.5);
                scf_data.scftype = SCFType::ROHF;

                //scf(scf_data.mol, &mpi_operator);
                initialize_scf(&mut scf_data, &mpi_operator);
                performance_essential_calculations(&mut scf_data, &mut time_mark, &mpi_operator);

                let scf_energy_triplet = scf_data.scf_energy;
                let tot_energy_triplet = scf_data.energies.get("xdh_energy").unwrap()[0]
                + scf_data.energies.get("ai_correction").map_or(0.0, |v| v[0]);
                let [square_spin_triplet, _] = scf_io::evaluate_spin_angular_momentum(&scf_data.density_matrix, &scf_data.ovlp, scf_data.mol.spin_channel, &scf_data.mol.num_elec);

                let spin_corrction_factor = square_spin_singlet / (square_spin_triplet - square_spin_singlet);
                let scf_energy_gap = scf_energy_triplet - scf_energy_singlet;
                let tot_energy_gap = tot_energy_triplet - tot_energy_singlet;

                let scf_energy_corrected = scf_energy_singlet - scf_energy_gap * spin_corrction_factor;
                let tot_energy_corrected = tot_energy_singlet - tot_energy_gap * spin_corrction_factor;
                
                println!("----------------------------------------------------------------------");
                println!("Report for Yamaguchi spin correction: ");
                println!("Open-shell singlet: scf_energy = {:18.10} Ha, tot_energy = {:18.10} Ha, <s^2> = {:6.3}.", scf_energy_singlet, tot_energy_singlet, square_spin_singlet);
                println!("Triplet: scf_energy = {:18.10} Ha, tot_energy = {:18.10} Ha, <s^2> = {:6.3}.", scf_energy_triplet, tot_energy_triplet, square_spin_triplet);
                println!("Corrected: scf_energy = {:18.10} Ha, tot_energy = {:18.10} Ha.", scf_energy_corrected, tot_energy_corrected);
                println!("----------------------------------------------------------------------");            
                time_mark.count("spin_correction");
            } else {
                println!("Yamaguchi spin correction skipped: either spin is not 0.0 or contamination is negligible.");
            }
        },
        None => {
        },
        Some(other) => {
            println!(
                "Warning: Unrecognized spin correction scheme '{}'.\nOnly 'yamaguchi' is currently supported. Spin correction will be skipped.",
                other
            );            
        }
    }


    let jobtype = scf_data.mol.ctrl.job_type.clone();
    match jobtype {
        JobType::Force => {
            eval_force(&mut scf_data, &mut time_mark, &mpi_operator);
        },
        JobType::NumDipole => {
            time_mark.count_start("numerical dipole");
            if scf_data.mol.ctrl.print_level>0 {
                println!("Numerical dipole calculation invoked");
            }

            let displace = scf_data.mol.ctrl.ndipole_displacement;
            let dp_au = numerical_dipole(&scf_data, displace);
            let dp = dp_au.iter().map(|x| *x * AU2DEBYE).collect::<Vec<f64>>();
            println!("Dipole Moment in A.U. : {:16.8}, {:16.8}, {:16.8}", dp_au[0], dp_au[1], dp_au[2]);
            println!("Dipole Moment in DEBYE: {:16.8}, {:16.8}, {:16.8}", dp[0], dp[1], dp[2]);
        },
        JobType::GeomOpt => {
            let opt_engine = scf_data.mol.ctrl.opt_engine.clone().unwrap_or("lbfgs".to_string());
            if opt_engine == "lbfgs" {
                println!("LBFGS geometry optimization invoked");
                time_mark.count_start("geom_opt");
                if scf_data.mol.ctrl.print_level>0 {
                    println!("Geometry optimization invoked");
                }
                let displace = 0.0013/ANG;

                //let (energy,nforce) = numerical_force(&scf_data, displace);
                //println!("Total atomic forces [a.u.]: ");
                //nforce.formated_output(5, "full");
                //let mut nnforce = nforce.clone();
                //nnforce.iter_mut().for_each(|x| *x *= ANG/EV);
                //println!("Total atomic forces [EV/Ang]: ");
                //nnforce.formated_output(5, "full");

                let mut position = scf_data.mol.geom.position.iter().map(|x| *x).collect::<Vec<f64>>();
                lbfgs().minimize(
                    &mut position, 
                    |x: &[f64], gx: &mut [f64]| {
                        scf_data.mol.geom.position = MatrixFull::from_vec([3,x.len()/3], x.to_vec()).unwrap();
                        if scf_data.mol.ctrl.print_level>0 {
                            println!("Input geometry in this round is:");
                            println!("{}", scf_data.mol.geom.formated_geometry());
                        }
                        scf_data.mol.ctrl.initial_guess = String::from("inherit");
                        initialize_scf(&mut scf_data, &mpi_operator);
                        performance_essential_calculations(&mut scf_data, &mut time_mark, &mpi_operator);
                        let (energy, nforce) = numerical_force(&scf_data, displace, &mpi_operator);
                        gx.iter_mut().zip(nforce.iter()).for_each(|(to, from)| {*to = *from});

                        if scf_data.mol.ctrl.print_level>0 {
                            println!("Output force in this round [a.u.] is:");
                            println!("{}", formated_force(&nforce, &scf_data.mol.geom.elem));
                        }

                        Ok(energy)
                    },
                    |prgr| {
                        println!("Iteration {}, Evaluation: {}", &prgr.niter, &prgr.neval);
                        println!(" xnorm = {}, gnorm = {}, step = {}",
                            &prgr.xnorm, &prgr.gnorm, &prgr.step
                        );
                        false
                    },
                );
                println!("Geometry after relaxation [Ang]:");
                println!("{}", scf_data.mol.geom.formated_geometry());
                time_mark.count("geom_opt");

                time_mark.report("geom_opt");
            } else if opt_engine == "geometric-pyo3" {
                #[cfg(feature = "geometric-pyo3")]
                {
                    //println!("Geometric geometry optimization invoked");
                    time_mark.count_start("geom_opt");
                    if scf_data.mol.ctrl.print_level>0 {
                        println!("Geometry optimization invoked using the optimization engine of geometric-pyo3");
                    }
                    geometric_pyo3_impl::optimize_geometric_pyo3(&mut scf_data, &mut time_mark);
                    println!("Geometry after relaxation [Ang]:");
                    println!("{}", scf_data.mol.geom.formated_geometry());
                    time_mark.count("geom_opt");
                    time_mark.report("geom_opt");
                }
                #[cfg(not(feature = "geometric-pyo3"))]
                panic!("Geometric-Pyo3 feature is not enabled. Please enable it in Cargo.toml.");
            } else {
                panic!("Invalid optimization engine: {}", opt_engine);
            }
        },
        _ => {}
    }

    //let mut grad_data = Gradient::build(&scf_data.mol, &scf_data);

    //grad_data.calc_j(&scf_data.density_matrix);
    //print!("occ, {:?}", scf_data.occupation);

    //time_mark.count("SCF");

    if scf_data.mol.ctrl.restart {
        if let Some(mp_op) = &mpi_operator {
            if mp_op.rank == 0 {
                println!("Rank 0: now save the converged SCF results");
                save_chkfile(&scf_data)
            }
        } else {
            println!("now save the converged SCF results");
            save_chkfile(&scf_data)
        }
    };

    if scf_data.mol.ctrl.check_stab {
        time_mark.new_item("Stability", "the scf stability check");
        time_mark.count_start("Stability");

        scf_data.stability();

        time_mark.count("Stability");
    }

    //====================================
    // Now for post-xc calculations
    //====================================
    if scf_data.mol.ctrl.post_xc.len()>=1 {
        print_out_dfa(&scf_data);
    }

    //====================================
    // Now for post-SCF analysis
    //====================================
    if scf_data.mol.ctrl.print_level > 0 {
        let mulliken = mulliken_pop(&scf_data);
        println!("Mulliken population analysis:");
        let elem_tot = scf_data.mol.geom.elem.clone().into_iter().chain(scf_data.mol.geom.ghost_bs_elem.clone().into_iter()).collect::<Vec<_>>();
        let ghost_atm_start = scf_data.mol.geom.elem.len();
        for (i, (pop, atom)) in mulliken.iter().zip(elem_tot.iter()).enumerate() {
            if i < ghost_atm_start {
                println!("{:3}-{:3}: {:10.6}", i, atom, pop)
            } else {
                println!("{:3}-{:3}: {:10.6}, Ghost Atom", i, atom, pop)
            };
        }
    }

    post_scf_analysis::post_scf_output(&scf_data, &mpi_operator);

    //====================================
    // Now for post-correlation calculations
    //====================================
    if scf_data.mol.ctrl.post_correlation.len()>=1 {
        post_scf_correlation(&mut scf_data);
    }

    time_mark.count("Overall");

    if scf_data.mol.ctrl.print_level > 0 {
        println!("");
        println!("====================================================");
        println!("              REST: Mission accomplished");
        println!("====================================================");
        output_result(&scf_data);
        time_mark.report_all();
    }

    //if let Some(mpi_op) = &mpi_operator {
    //    // I would like to finish the mpi world
    //}


    Ok(())
}


pub fn output_result(scf_data: &scf_io::SCF) {
    println!("The SCF energy        : {:18.10} Ha", 
        //scf_data.mol.ctrl.xc.to_uppercase(),
        scf_data.scf_energy);
    
    let xc_name = scf_data.mol.ctrl.xc.to_lowercase();
    if xc_name.eq("mp2") || xc_name.eq("xyg3") || xc_name.eq("xygjos") || xc_name.eq("r-xdh7") || xc_name.eq("xyg7") || xc_name.eq("zrps") || xc_name.eq("scsrpa") {
        let total_energy = scf_data.energies.get("xdh_energy").unwrap()[0];
        //let post_ai_correction = scf_data.mol.ctrl.post_ai_correction.to_lowercase();
        //let ai_correction = if xc_name.eq("r-xdh7") && post_ai_correction.eq("scc15") {
        //    let ai_correction = scf_data.energies.get("ai_correction").unwrap()[0];
        //    println!("AI Correction         : {:18.10} Ha", ai_correction);
        //    ai_correction
        //} else {
        //    0.0
        //};
        let ai_correction = if let Some(ai_correction) = scf_data.energies.get("ai_correction") {
            ai_correction[0]
        } else {
            0.0
        };
        println!("The (R)-xDH energy    : {:18.10} Ha", total_energy+ ai_correction);
    }
    if xc_name.eq("rpa@pbe") {
        let total_energy = scf_data.energies.get("rpa_energy").unwrap()[0];
        println!("The RPA energy        : {:18.10} Ha", total_energy);
    }

}

/// Perform key SCF and post-SCF calculations
/// Return the total energy of the specfied xc method
/// Assume the initialization of SCF is ready
pub fn performance_essential_calculations(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords, mpi_operator: &Option<MPIOperator>) -> f64 {

    let mut total_energy = 0.0;

    //=================================================================
    // Now evaluate the SCF energy for the given method
    //=================================================================
    time_mark.count_start("SCF");
    scf_without_build(scf_data, mpi_operator);
    //println!("debug time mark SCF turn off");
    time_mark.count("SCF");

    //==================================================================
    // Now evaluate the advanced correction energy for the given method
    //==================================================================
    //let mut time_mark = utilities::TimeRecords::new();
    if let Some(dft_method) = &scf_data.mol.xc_data.dfa_family_pos {
        match dft_method {
            dft::DFAFamily::PT2 | dft::DFAFamily::SBGE2 => {
                //time_mark.new_item("PT2", "the PT2 evaluation");
                time_mark.count_start("PT2");
                ri_pt2::xdh_calculations(scf_data, mpi_operator);
                time_mark.count("PT2");
            },
            dft::DFAFamily::RPA => {
                //time_mark.new_item("RPA", "the RPA evaluation");
                time_mark.count_start("RPA");
                ri_rpa::rpa_calculations(scf_data, mpi_operator);
                time_mark.count("RPA");
            }
            dft::DFAFamily::SCSRPA => {
                //time_mark.new_item("SCS-RPA", "the SCS-RPA evaluation");
                time_mark.count_start("SCS-RPA");
                ri_pt2::xdh_calculations(scf_data, mpi_operator);
                time_mark.count("SCS-RPA");
            }
            _ => {}
        }
    }
    //====================================
    // Now for post ai correction
    //====================================
    if let Some(scc) = post_ai_correction(scf_data, mpi_operator) {
        scf_data.energies.insert("ai_correction".to_string(), scc);
    }

    collect_total_energy(scf_data)

}

pub fn collect_total_energy(scf_data: &SCF) -> f64 {
    //====================================
    // Determine the total energy
    //====================================
    let mut total_energy = scf_data.scf_energy;
    
    let xc_name = scf_data.mol.ctrl.xc.to_lowercase();
    if xc_name.eq("mp2") || xc_name.eq("xyg3") || xc_name.eq("xygjos") || xc_name.eq("r-xdh7") || xc_name.eq("xyg7") || xc_name.eq("zrps") || xc_name.eq("scsrpa") {
        total_energy = scf_data.energies.get("xdh_energy").unwrap()[0];
    } else if xc_name.eq("rpa@pbe") {
        total_energy = scf_data.energies.get("rpa_energy").unwrap()[0];
    }
    if let Some(post_ai_correction) = scf_data.energies.get("ai_correction") {
        total_energy += post_ai_correction[0]
    };

    total_energy

}


fn initialize_time_record(mol: &Molecule) -> utilities::TimeRecords {
    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("Overall", "the whole job");
    time_mark.new_item("SCF", "the scf procedure");
    let jobtype = mol.ctrl.job_type.clone();
    match jobtype {
        JobType::GeomOpt => {
            time_mark.new_item("geom_opt", "geometry optimization");
        },
        JobType::Force => {
            time_mark.new_item("force", "force calculation");
        },
        _ => {}
    };
    if let Some(dft_method) = &mol.xc_data.dfa_family_pos {
        match dft_method {
            dft::DFAFamily::PT2 | dft::DFAFamily::SBGE2 => {
                time_mark.new_item("PT2", "the PT2 evaluation");
            },
            dft::DFAFamily::RPA => {
                time_mark.new_item("RPA", "the RPA evaluation");
            }
            dft::DFAFamily::SCSRPA => {
                time_mark.new_item("SCS-RPA", "the SCS-RPA evaluation");
            }
            _ => {}
        }
    }

    time_mark

}

/* #region force and geomopt utilities */

fn eval_force(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords, mpi_operator: &Option<MPIOperator>) -> (f64, MatrixFull<f64>) {
    // this is a temporary workaround for the force evaluation
    // currently, this framework could not work for post-scf,
    // especially that currently there's no class that represents post-scf computation

    time_mark.count_start("force");
    if scf_data.mol.ctrl.print_level>0 {
        println!("Force calculation invoked");
    }

    let (energy, gradient) = if scf_data.mol.ctrl.numerical_force {
        if scf_data.mol.ctrl.print_level > 1 {
            println!("Gradient evaluation using numerical differentiation");
        }
        let displace = scf_data.mol.ctrl.nforce_displacement / ANG;
        let (energy, nforce) = numerical_force(&scf_data, displace, &mpi_operator);
        println!("------ Output gradient [a.u.] ------");
        println!("{}", formated_force(&nforce, &scf_data.mol.geom.elem));
        println!("------------------------------------");
        (energy, nforce)
    } else {
        // current available analytical gradients methods:
        // 1) numerical force
        // 2) analytical RHF, UHF force
        // 
        // disallow dft and post-scf calculations for force
        if scf_data.mol.ctrl.xc.to_lowercase() != "hf" {
            panic!("Gradient calculation is only available for RHF and UHF");
        }

        if scf_data.mol.ctrl.print_level > 1 {
            println!("Gradient evaluation using Analytical differentiation");
        }

        // Please note that this is only a temporary workaround for RHF/UHF gradients.
        // Totally refactor the following code if necessary if other types of gradients to be implemented.
        let grad_data: Box<dyn crate::grad::traits::GradAPI> = {
            if !scf_data.mol.ctrl.spin_polarization {
                let mut grad_data = crate::grad::rhf::RIRHFGradient::new(&scf_data);
                grad_data.calc();
                Box::new(grad_data)
            } else {
                let mut grad_data = crate::grad::uhf::RIUHFGradient::new(&scf_data);
                grad_data.calc();
                Box::new(grad_data)
            }
        };

        let gradient = grad_data.get_gradient();

        println!("------ Output gradient [a.u.] ------");
        println!("{}", formated_force(&gradient, &scf_data.mol.geom.elem));
        println!("------------------------------------");

        (scf_data.scf_energy, gradient)
    };

    time_mark.count("force");
    time_mark.report("force");

    (energy, gradient)
}

fn eval_force_with_position(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords, mpi_operator: &Option<MPIOperator>, position: &MatrixFull<f64>) -> (f64, MatrixFull<f64>) {
    //scf_data.mol.geom.position = position.clone();
    scf_data.mol.geom.geom_update(&position.data(), GeomUnit::Angstrom);
    if scf_data.mol.ctrl.print_level>0 {
        println!("Input geometry in this round is:");
        println!("{}", scf_data.mol.geom.formated_geometry());
    }
    scf_data.mol.ctrl.initial_guess = String::from("inherit");
    initialize_scf(scf_data, mpi_operator);
    performance_essential_calculations(scf_data, time_mark, mpi_operator);
    let (energy, gradient) = eval_force(scf_data, time_mark, mpi_operator);
    return (energy, gradient);
}

#[cfg(feature = "geometric-pyo3")]
mod geometric_pyo3_impl {
    use super::*;
    use geometric_pyo3::prelude::*;
    use pyo3::prelude::*;

    pub(crate) struct GeometricOptDriver<'a> {
        scf_data: &'a mut SCF,
        time_mark: &'a mut utilities::TimeRecords,
    }

    impl GeomDriverAPI for GeometricOptDriver<'_> {
        fn calc_new(&mut self, coords: &[f64], _dirname: &str) -> GradOutput {
            let coords = coords.to_vec();
            let coords = MatrixFull::from_vec([3, coords.len()/3], coords).unwrap();
            let mpi_operator = None;
            let (scf_data, time_mark) = (&mut self.scf_data, &mut self.time_mark);
            let (energy, mut gradient) = eval_force_with_position(scf_data, time_mark, &mpi_operator, &coords);
            //gradient.formated_output(3, "full");
            //gradient *= -1.0;
            let gradient = gradient.data();
            return GradOutput {
                energy,
                gradient,
            }
        }
    }

    pub(crate) fn optimize_geometric_pyo3(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords) -> PyResult<(f64, MatrixFull<f64>)> {
        pyo3::prepare_freethreaded_python();
        
        let elem = scf_data.mol.geom.elem.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        const BOHR: f64 = crate::constants::BOHR;
        let xyz = scf_data.mol.geom.position.data.iter().map(|x| x * BOHR).collect::<Vec<f64>>();
        //let xyz = scf_data.mol.geom.position.iter().map(|x| *x).collect::<Vec<f64>>();
        let xyzs = vec![xyz];
        let molecule = init_pyo3_molecule(&elem, &xyzs).unwrap();
        
        let optimizer_params = r#"
            convergence_energy   = 1.0e-6  # Eh
            convergence_grms     = 3.0e-4  # Eh/Bohr
            convergence_gmax     = 4.5e-4  # Eh/Bohr
            convergence_drms     = 1.2e-3  # Angstrom
            convergence_dmax     = 1.8e-3  # Angstrom
        "#;
        let input = None;
        let params = tomlstr2py(optimizer_params)?;

        let pyo3_engine_cls = get_pyo3_engine_cls()?;
        let geometric_opt_driver = GeometricOptDriver {
            scf_data,
            time_mark,
        };
        let driver: PyGeomDriver = geometric_opt_driver.into();

        
        let (last_energy, last_coords) = Python::with_gil(|py| -> PyResult<(f64, Vec<f64>)> {
            let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
            custom_engine.call_method1(py, "set_driver", (driver,))?;
            let res = run_optimization(custom_engine, &params, input)?;

            let last_energy = res
                .getattr(py, "qm_energies")?
                .call_method1(py, "__getitem__", (-1,))?
                .extract::<f64>(py)?;

            let last_coords = res
                .getattr(py, "xyzs")?
                .call_method1(py, "__getitem__", (-1,))?
                .call_method0(py, "flatten")?
                .call_method0(py, "tolist")?
                .extract::<Vec<f64>>(py)?;

            Ok((last_energy, last_coords))
        })?;

        let last_coords = MatrixFull::from_vec([3, last_coords.len()/3], last_coords).unwrap();

        return Ok((last_energy, last_coords));
    }
}

/* #endregion */
