use crate::basis_io::ecp::ghost_effective_potential_matrix;
use crate::check_norm::force_state_occupation::adapt_occupation_with_force_projection;
use crate::check_norm::{self, generate_occupation_frac_occ, generate_occupation_integer, generate_occupation_sad, OCCType};
use crate::dft::gen_grids::prune::prune_by_rho;
use crate::dft::{numerical_density, DFTType, Grids};
use crate::geom_io::{calc_nuc_energy, calc_nuc_energy_with_ext_field, calc_nuc_energy_with_point_charges};
use crate::mpi_io::{mpi_broadcast, mpi_broadcast_matrixfull, mpi_broadcast_vector, mpi_reduce, MPIOperator};
use crate::utilities::{create_pool, TimeRecords};
////use blas_src::openblas::dgemm;
mod addons;
mod fchk;
mod pyrest_scf_io;

use mpi::collective::SystemOperation;
use pyo3::{pyclass, pymethods, pyfunction};
use tensors::matrix_blas_lapack::{_dgemm, _dgemm_full, _dgemm_nn, _dgemv, _dinverse, _dspgvx, _dsymm, _dsyrk, _hamiltonian_fast_solver, _power, _power_rayon_for_symmetric_matrix, _dsyevd};
use tensors::{map_full_to_upper, map_upper_to_full, ri, BasicMatUp, BasicMatrix, ERIFold4, ERIFull, MathMatrix, MatrixFull, MatrixFullSlice, MatrixFullSliceMut, MatrixUpper, MatrixUpperSlice, ParMathMatrix, RIFull, TensorSliceMut};
use itertools::{Itertools, iproduct, izip};
use rayon::prelude::*;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Mutex, Arc,mpsc};
use std::thread;
use crossbeam::{channel::{unbounded,bounded},thread::{Scope,scope}};
use std::sync::mpsc::{channel, Receiver};
use crate::isdf::{prepare_for_ri_isdf, init_by_rho, prepare_m_isdf};
use crate::molecule_io::{Molecule, generate_ri3fn_from_rimatr};
use crate::tensors::{TensorOpt,TensorOptMut,TensorSlice};
use crate::{utilities, initial_guess};
use crate::initial_guess::initial_guess;
use crate::external_libs::dftd;
use crate::constants::{INVERSE_THRESHOLD, SPECIES_INFO, SQRT_THRESHOLD};




#[pyclass]
#[derive(Clone)]
pub struct SCF {
    #[pyo3(get,set)]
    pub mol: Molecule,
    pub ovlp: MatrixUpper<f64>,
    pub h_core: MatrixUpper<f64>,
    //pub ijkl: Option<Tensors<f64>>,
    //pub ijkl: Option<ERIFull<f64>>,
    pub ijkl: Option<ERIFold4<f64>>,
    pub ri3fn: Option<RIFull<f64>>,
    pub ri3fn_isdf: Option<RIFull<f64>>,
    pub tab_ao: Option<MatrixFull<f64>>,
    pub m: Option<MatrixFull<f64>>,
    pub rimatr: Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
    pub ri3mo: Option<Vec<(RIFull<f64>,std::ops::Range<usize> , std::ops::Range<usize>)>>,
    #[pyo3(get,set)]
    pub eigenvalues: [Vec<f64>;2],
    //pub eigenvectors: Vec<Tensors<f64>>,
    pub eigenvectors: [MatrixFull<f64>;2],
    //pub density_matrix: Vec<Tensors<f64>>,
    //pub density_matrix: [MatrixFull<f64>;2],
    pub density_matrix: Vec<MatrixFull<f64>>,
    //pub hamiltonian: Vec<Tensors<f64>>,
    pub hamiltonian: [MatrixUpper<f64>;2],
    pub roothaan_hamiltonian: MatrixUpper<f64>,
    pub semi_eigenvalues: [Vec<f64>;2],
    pub semi_eigenvectors: [MatrixFull<f64>;2],
    pub semi_fock: [MatrixFull<f64>;2],
    pub scftype: SCFType,
    #[pyo3(get,set)]
    pub occupation: [Vec<f64>;2],
    #[pyo3(get,set)]
    pub homo: [usize;2],
    #[pyo3(get,set)]
    pub lumo: [usize;2],
    #[pyo3(get,set)]
    pub nuc_energy: f64,
    #[pyo3(get,set)]
    pub scf_energy: f64,
    pub grids: Option<Grids>,
    pub empirical_dispersion_energy: f64,
    pub energies: HashMap<String,Vec<f64>>,
    pub ref_eigenvectors: HashMap<String, ([MatrixFull<f64>;2], [usize;4])>,
}

#[derive(Clone,Copy)]
pub enum SCFType {
    RHF,
    ROHF,
    UHF
}


impl SCF {
    pub fn init_scf(mol: &Molecule) -> SCF {
        let mut scf_data = SCF {
            mol: mol.clone(),
            ovlp: MatrixUpper::new(1,0.0),
            h_core: MatrixUpper::new(1,0.0),
            ijkl: None,
            ri3fn: None,
            ri3fn_isdf: None,
            tab_ao: None,
            m: None,
            rimatr: None,
            ri3mo: None,
            eigenvalues: [vec![],vec![]],
            hamiltonian: [MatrixUpper::empty(),
                          MatrixUpper::empty()],
            roothaan_hamiltonian: MatrixUpper::empty(),
            semi_eigenvalues: [vec![],vec![]],
            semi_eigenvectors: [MatrixFull::empty(), MatrixFull::empty()],
            semi_fock: [MatrixFull::empty(), MatrixFull::empty()],
            eigenvectors: [MatrixFull::empty(),
                           MatrixFull::empty()],
            ref_eigenvectors: HashMap::new(),
            //density_matrix: [MatrixFull::new([1,1],0.0),
            //                     MatrixFull::new([1,1],0.0)],
            density_matrix: vec![MatrixFull::empty(),
                                 MatrixFull::empty()],
            scftype: SCFType::RHF,
            occupation: [vec![],vec![]],
            homo: [0,0],
            lumo: [0,0],
            nuc_energy: 0.0,
            scf_energy: 0.0,
            empirical_dispersion_energy: 0.0,
            grids: None,
            energies: HashMap::new(),
        };

        // at first check the scf type: RHF, ROHF or UHF
        scf_data.scftype = if mol.num_elec[1]==mol.num_elec[2] && ! mol.ctrl.spin_polarization {
            SCFType::RHF
        } else if mol.num_elec[1]!=mol.num_elec[2] && ! mol.ctrl.spin_polarization {
            SCFType::ROHF
        } else {      
            SCFType::UHF
        };
        match &scf_data.scftype {
            SCFType::RHF => {
                if mol.ctrl.print_level>0 {println!("Restricted Hartree-Fock (or Kohn-Sham) algorithm is invoked.")}},
            SCFType::ROHF => {
                if mol.ctrl.print_level>0 {println!("Restricted open shell Hartree-Fock (or Kohn-Sham) algorithm is invoked.")};
                // In ROHF, although the Roothaan Fock matrix is not separated into alpha and beta spin channels, 
                // it is derived based on the density matrices of the alpha and beta spin channels. 
                // Therefore, even though "spin_polarization=False" is specified as input, we handle it as "spin_channel=2".
                scf_data.mol.ctrl.spin_channel=2;
                scf_data.mol.spin_channel=2;
                scf_data.mol.xc_data.spin_channel=2;
            },
            SCFType::UHF => {
                if mol.ctrl.print_level>0 {println!("Unrestricted Hartree-Fock (or Kohn-Sham) algorithm is invoked.")}
            },
        };

        scf_data
    }


    pub fn prepare_necessary_integrals(&mut self, mpi_operator: &Option<MPIOperator>) {
        // prepare standard two, three, and four-center integrals.
        // for ISDF integrals, they needs density grids, and thus should be prepared after the grid initialization

        let print_level = self.mol.ctrl.print_level;

        //========================================
        // For nuclear energy, includin the interaction with the ghost atoms with point charges
        self.nuc_energy = calc_nuc_energy(&self.mol.geom, &self.mol.basis4elem);

        let nuc_energy_pc = calc_nuc_energy_with_point_charges(&self.mol.geom, &self.mol.basis4elem);
        let nuc_energy_ext_field = calc_nuc_energy_with_ext_field(&self.mol.geom, &self.mol.basis4elem);
        self.nuc_energy += nuc_energy_pc;
        self.nuc_energy += nuc_energy_ext_field;

        if print_level>0 {
            println!("Nuc_energy: {:16.8} Hartree",self.nuc_energy);
            if nuc_energy_pc.abs() > 1.0e-4 {
                println!("External potential due to point charges exists: {:16.8} Hartree", &nuc_energy_pc);
            }
            if nuc_energy_ext_field.abs() > 1.0e-10 {
                println!("External dipole field contribution to nuc energy exists: {:16.8} Hartree", &nuc_energy_ext_field);
            }
        }
        //========================================
        // For emperial dispersion correction
        if let Some(empirical_dispersion_name) = &self.mol.ctrl.empirical_dispersion {
            let (engy_disp, grad_disp, sigma_disp) = dftd(self);
            if self.mol.ctrl.print_level>1 { 
                println!("The empirical dispersion energy of {} is {}.", self.mol.ctrl.empirical_dispersion.clone().unwrap().to_uppercase(), engy_disp)
            };
            if self.mol.ctrl.print_level>3 { 
                println!("{:?}, {:?}", &grad_disp, &sigma_disp);
            };
            self.empirical_dispersion_energy = engy_disp;
            
            // empirical dispersion energy added to the nuc_energy
            self.nuc_energy += engy_disp;
        } else {
            if self.mol.ctrl.print_level>1 { 
                println!("no empirical dispersion correction is employed");
            }
        }
        //========================================
        // For two-center integrals
        self.ovlp = self.mol.int_ij_matrixupper(String::from("ovlp"));
        self.h_core = self.mol.int_ij_matrixupper(String::from("hcore"));
        //========================================
        // For ghost effective potential
        if self.mol.geom.ghost_ep_path.len() > 0 {
            let tmp_matr = ghost_effective_potential_matrix(
                &self.mol.cint_env, &self.mol.cint_atm, &self.mol.cint_bas,&self.mol.cint_type, self.mol.num_basis,
                &self.mol.geom.ghost_ep_path, &self.mol.geom.ghost_ep_pos
            );
            self.h_core.iter_mut().zip(tmp_matr.iter()).for_each(|(a,b)| *a += *b);
        } else {
            if self.mol.ctrl.print_level > 0 {
                println!("No ghost effective potential");
            }
        }

        // ========================================
        // For the external field
        if let Some(ext_field_dipole) = &self.mol.geom.ext_field.dipole {
            use crate::external_field::ExtField;
            let mut ext_field = ExtField::empty();
            ext_field.dipole = ext_field_dipole.clone().try_into().unwrap();
            let tmp_matr = ext_field.contribution_2c(&self.mol);
            let tmp_matr = tmp_matr.to_matrixupper();
            self.h_core.iter_mut().zip(tmp_matr.iter()).for_each(|(a,b)| *a += *b);
        }

        // For the ghost point charge term
        if self.mol.geom.ghost_pc_chrg.len() > 0 {
            if self.mol.ctrl.print_level > 0 {
                println!("There are {} point charges specified", self.mol.geom.ghost_pc_chrg.len());
            }
            let tmp_matr = self.mol.int_ij_matrixupper(String::from("point charge"));
            self.h_core.iter_mut().zip(tmp_matr.iter()).for_each(|(a,b)| *a += *b);
        } else {
            if self.mol.ctrl.print_level > 0 {
                println!("No ghost point charges");
            }
        }

        //========================================
        // For four-center integrals
        self.ijkl = if self.mol.ctrl.use_auxbas {
            None
        } else {
            Some(self.mol.int_ijkl_erifold4())
        };
        //========================================
        if self.mol.ctrl.print_level>3 {
            println!("The S matrix:");
            self.ovlp.formated_output(5, "lower");
            let mut kin = self.mol.int_ij_matrixupper(String::from("kinetic"));
            println!("The Kinetic matrix:");
            kin.formated_output(5, "lower");
            println!("The H-core matrix:");
            self.h_core.formated_output(5, "lower");
        }

        if self.mol.ctrl.print_level>4 {
            //(ij|kl)
            if let Some(tmp_eris) = &self.ijkl {
                println!("The four-center ERIs:");
                let mut tmp_num = 0;
                let (i_len,j_len) =  (self.mol.num_basis,self.mol.num_basis);
                let (k_len,l_len) =  (self.mol.num_basis,self.mol.num_basis);
                (0..k_len).into_iter().for_each(|k| {
                    (0..k+1).into_iter().for_each(|l| {
                        (0..i_len).into_iter().for_each(|i| {
                            (0..i+1).into_iter().for_each(|j| {
                                if let Some(tmp_value) = tmp_eris.get(&[i,j,k,l]) {
                                    if tmp_value.abs()>1.0e-1 {
                                        println!("I= {:2} J= {:2} K= {:2} L= {:2} Int= {:16.8}",i+1, j+1, k+1,l+1, tmp_value);
                                        tmp_num+= 1;
                                    }
                                } else {
                                    println!("Error: unknown value for eris[{},{},{},{}]",i,j,k,l)
                                };
                            })
                        })
                    })
                });
                println!("Print out {} ERIs", tmp_num);
            }
        }

        let use_eri = self.mol.use_eri;
        //let use_eri = true;
        let isdf = if use_eri {self.mol.ctrl.eri_type.eq("ri_v") && self.mol.ctrl.use_isdf} else {false};
        let ri3fn_full = if use_eri {self.mol.ctrl.use_auxbas && !self.mol.ctrl.use_ri_symm} else {false};
        let ri3fn_symm = if use_eri {self.mol.ctrl.use_auxbas && self.mol.ctrl.use_ri_symm} else{false};

        // preparing the three-center integrals in the full format
        self.ri3fn = if ri3fn_full && !isdf {
            Some(self.mol.prepare_ri3fn_for_ri_v_full_rayon())
        }else if self.mol.ctrl.isdf_k_only{ 
            Some(self.mol.prepare_ri3fn_for_ri_v_full_rayon())
        }else {
            None
        };

        // preparing the three-center integrals using the symmetry
        self.rimatr = if ri3fn_symm  && ! isdf {

            // initialize the mpi distribution information for num_auxbas and num_baspar
            if let Some(local_mpi) = &mut self.mol.mpi_data {
                let num_auxbas = self.mol.num_auxbas;
                let num_basis = self.mol.num_basis;
                let cint_bas = self.mol.cint_fdqc.clone();
                local_mpi.distribute_rimatr_tasks(num_auxbas, num_basis, cint_bas);
            }

            let (rimatr, basbas2baspar, baspar2basbas) = 
                self.mol.prepare_rimatr_for_ri_v_mpi_rayon(mpi_operator);
            Some((rimatr, basbas2baspar, baspar2basbas))
        } else if ri3fn_symm  && isdf {
            None
        } else {
            None
        };


        // initial eigenvectors and eigenvalues
        let (eigenvectors, eigenvalues,n_found)=self.ovlp.to_matrixupperslicemut().lapack_dspevx().unwrap();

        if (n_found as usize) < self.mol.fdqc_bas.len() {
            if self.mol.ctrl.print_level>0 {
                println!("Overlap matrix is singular:");
                println!("  Using {} out of a possible {} specified basis functions",n_found, self.mol.fdqc_bas.len());
                println!("  Lowest remaining eigenvalue: {:16.8}",eigenvalues[0]);
            }
            self.mol.num_state = n_found as usize;
        } else {
            if self.mol.ctrl.print_level>0 {
                println!("Overlap matrix is nonsigular:");
                println!("  Lowest eigenvalue: {:16.8} with the total number of basis functions: {:6}",eigenvalues[0],self.mol.num_state);
            }
        };

    }

    pub fn prepare_density_grids(&mut self) {

        self.grids = if self.mol.xc_data.is_dfa_scf() || self.mol.ctrl.use_isdf || self.mol.ctrl.initial_guess == "vsap" {
            let grids = Grids::build(&mut self.mol);
            if self.mol.ctrl.print_level>0 {
                println!("Grid size: {:}", grids.coordinates.len());
            }
            Some(grids)
        } else {None};

        if let Some(grids) = &mut self.grids {
            grids.prepare_tabulated_ao(&self.mol);
        }

    }

    pub fn prepare_isdf(&mut self, mpi_operator: &Option<MPIOperator>) {

        let use_eri = self.mol.use_eri;
        let isdf = if use_eri {self.mol.ctrl.eri_type.eq("ri_v") && self.mol.ctrl.use_isdf} else {false};
        let ri3fn_full = if use_eri {self.mol.ctrl.use_auxbas && !self.mol.ctrl.use_ri_symm} else {false};
        let ri3fn_symm = if use_eri {self.mol.ctrl.use_auxbas && self.mol.ctrl.use_ri_symm} else{false};

        if ! isdf {return}
        if let Some(grids) = &self.grids {
            if self.mol.ctrl.use_isdf {
                let init_fock = self.h_core.clone();
                if self.mol.spin_channel==1 {
                    self.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
                } else {
                    let init_fock_beta = init_fock.clone();
                    self.hamiltonian = [init_fock,init_fock_beta];
                };
                (self.eigenvectors,self.eigenvalues, self.mol.num_state) = diagonalize_hamiltonian_outside(&self, mpi_operator);
                (self.occupation, self.homo, self.lumo) = generate_occupation_outside(&self);
                self.density_matrix = generate_density_matrix_outside(&self);

                self.grids = Some(prune_by_rho(grids, &self.density_matrix, self.mol.spin_channel));
                
            };


            self.ri3fn_isdf = if ri3fn_full && isdf && !self.mol.ctrl.isdf_new{
                if let Some(grids) = &self.grids {
                    Some(prepare_for_ri_isdf(self.mol.ctrl.isdf_k_mu, &self.mol, &grids))
                } else {
                    None
                }
            } else {
                None
            };

            (self.tab_ao, self.m) = if isdf && self.mol.ctrl.isdf_new{
                if let Some(grids) = &self.grids {
                    let isdf = prepare_m_isdf(self.mol.ctrl.isdf_k_mu, &self.mol, &grids);
                    (Some(isdf.0), Some(isdf.1))
                } else {
                    (None,None)
                }
            } else {
                (None,None)
            };
        } else {
            panic!("SCF.grids should be initialized before the preparation of ISDF");
        }

    }



    pub fn build(mol: Molecule, mpi_operator: &Option<MPIOperator>) -> SCF {

        let mut new_scf = SCF::init_scf(&mol);
        //new_scf.generate_occupation();

        initialize_scf(&mut new_scf, mpi_operator);

        new_scf

    }


    pub fn generate_occupation(&mut self) {
        (self.occupation, self.homo, self.lumo) = generate_occupation_outside(&self);
    }

    pub fn generate_density_matrix(&mut self) {
        self.density_matrix = generate_density_matrix_outside(&self);
    }

    pub fn generate_vj_with_erifold4(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vj[i_spin] = MatrixUpper::new(npair,0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        let dm_ij = dm[i_spin].get1d(ic*num_basis + jc).unwrap() + 
                                        dm[i_spin].get1d(jc*num_basis + ic).unwrap();
                        let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                        let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                        //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                        //unsafe{daxpy(npair as i32, dm_ij, reduce_ij, 1, vj[i_spin].to_slice_mut(), 1)};
                        vj[i_spin].data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                            *vj_ij += eri_ij*dm_ij
                        });
                        // Rayon parallellism. 
                        //vj[i_spin].data.par_iter_mut().zip(reduce_ij.par_iter()).for_each(|(vj_ij,eri_ij)| {
                        //    *vj_ij += eri_ij*dm_ij
                        //});
                    }
                }
                for jc in (0..num_basis) {
                    let dm_ij = dm[i_spin].get1d(jc*num_basis + jc).unwrap(); 
                    let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    //let reduce_ij = ijkl.get4d_slice([0,0,jc,jc],npair).unwrap();
                    //unsafe{daxpy(npair as i32, *dm_ij, reduce_ij, 1, vj[i_spin].to_slice_mut(), 1)};
                    vj[i_spin].data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                        *vj_ij += eri_ij*dm_ij
                    });
                    //vj[i_spin].data.par_iter_mut().zip(reduce_ij.par_iter()).for_each(|(vj_ij,eri_ij)| {
                    //    *vj_ij += eri_ij*dm_ij
                    //});
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        //let i_spin:usize = 0;
        //for j in (0..num_basis) {
        //    for i in (0..j+1) {
        //        println!("i: {}, j: {}, vj_ij: {}", i,j, vj[0].get2d([i,j]).unwrap());
        //    }
        //}

        vj
    }
    pub fn generate_vj_with_erifold4_sync(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        let num_para: usize = if let Some(num_para) = self.mol.ctrl.num_threads {
            num_para
        } else {
            1
        };
        let num_chunck = if num_basis%num_para==0 {
            (num_basis/num_para,num_basis/num_para)
        } else {
            (num_basis/num_para+1,num_basis-(num_basis/num_para+1)*(num_para-1))
        };
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vj[i_spin] = MatrixUpper::new(npair,0.0f64);
                scope(|s_thread| {
                    let (tx_jc,rx_jc) = unbounded();
                    for f in (0..num_para-1) {
                        let jc_start_thread = f*num_chunck.0;
                        let jc_end_thread = jc_start_thread + num_chunck.0;
                        let tx_jc_thread = tx_jc.clone();
                        let handle = s_thread.spawn(move |_| {
                            let mut vj_thread = MatrixUpper::new(npair,0.0f64);
                            for jc in (jc_start_thread..jc_end_thread) {
                                for ic in (0..jc) {
                                    let dm_ij = dm[i_spin].get1d(ic*num_basis + jc).unwrap() + 
                                                    dm[i_spin].get1d(jc*num_basis + ic).unwrap();
                                    let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                    vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                        *vj_ij += eri_ij*dm_ij
                                    });
                                }
                                let dm_ij = dm[i_spin].get1d(jc*num_basis + jc).unwrap(); 
                                let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                    *vj_ij += eri_ij*dm_ij
                                });
                            }
                            tx_jc_thread.send(vj_thread).unwrap();
                        });
                    }
                    let jc_start_thread = (num_para-1)*num_chunck.0;
                    let jc_end_thread = jc_start_thread + num_chunck.1;
                    let tx_jc_thread = tx_jc;
                    let handle = s_thread.spawn(move |_| {
                        let mut vj_thread = MatrixUpper::new(npair,0.0f64);
                        for jc in (jc_start_thread..jc_end_thread) {
                            for ic in (0..jc) {
                                let dm_ij = dm[i_spin].get1d(ic*num_basis + jc).unwrap() + 
                                                dm[i_spin].get1d(jc*num_basis + ic).unwrap();
                                let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                    *vj_ij += eri_ij*dm_ij
                                });
                            }
                            let dm_ij = dm[i_spin].get1d(jc*num_basis + jc).unwrap(); 
                            let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                            let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                            vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                *vj_ij += eri_ij*dm_ij
                            });
                        }
                        tx_jc_thread.send(vj_thread).unwrap();
                    });
                    for received in rx_jc {
                        vj[i_spin].data.iter_mut()
                            .zip(received.data).for_each(|(i,j)| {*i += j});
                    }

                }).unwrap();
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        vj
    }

    pub fn generate_vj_on_the_fly(&self) -> Vec<MatrixUpper<f64>>{
        let num_shell = self.mol.cint_bas.len();
        //let num_shell = self.mol.cint_fdqc.len();
        let num_basis = self.mol.num_basis;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut vj: Vec<MatrixUpper<f64>> = vec![];
        let mol = &self.mol;
        for i_spin in 0..spin_channel{
            let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
            let mut dm_s = &self.density_matrix[i_spin];
            for k in 0..num_shell{
                let bas_start_k = mol.cint_fdqc[k][0];
                let bas_len_k = mol.cint_fdqc[k][1];

                for l in 0..num_shell{
                    let bas_start_l = mol.cint_fdqc[l][0];
                    let bas_len_l = mol.cint_fdqc[l][1];
                    let mut klij = &mol.int_ijkl_given_kl(k, l);
                    
                    // ao_k & ao_l are index of ao
                    let mut sum =0.0;
                    for ao_k in bas_start_k..bas_start_k+bas_len_k{
                        for ao_l in bas_start_l..bas_start_l+bas_len_l{
                            let mut sum =0.0;
                            let mut index_k = ao_k-bas_start_k;
                            let mut index_l = ao_l-bas_start_l;
                            let mut eri_cd = &klij[index_l * bas_len_k + index_k];
                            let eri_full = eri_cd.to_matrixupper().to_matrixfull().unwrap();
                            let mut v_cd = MatrixFull::new([num_basis, num_basis], 0.0);
                            v_cd.data.iter_mut().zip(dm_s.data.iter()).zip(eri_full.data.iter()).for_each(|((v,p),eri)|{
                                *v = *p * *eri
                            });

                            v_cd.data.iter().for_each(|x|{
                                sum += *x
                            });
                            vj_i[(ao_k,ao_l)] = sum;
                        }
                    }
                }
            }

            vj.push(vj_i.to_matrixupper());
        }
        if spin_channel == 1{
            vj.push(MatrixUpper::new(1, 0.0));       
        }
        vj
    }

    pub fn generate_vj_on_the_fly_par_old(&self) -> Vec<MatrixUpper<f64>>{
        //utilities::omp_set_num_threads_wrapper(1);
        let num_shell = self.mol.cint_bas.len();
        let num_basis = self.mol.num_basis;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut vj: Vec<MatrixUpper<f64>> = vec![];
        let mol = &self.mol;
        for i_spin in 0..spin_channel{
            let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
            let mut dm_s = &self.density_matrix[i_spin];
            let par_tasks = utilities::balancing(num_shell*num_shell, rayon::current_num_threads());
            let (sender, receiver) = channel();
            let mut index = vec![0usize; num_shell*num_shell];
            for i in 0..num_shell*num_shell{index[i] = i};
            index.par_iter().for_each_with(sender,|s,i|{
                let k = i/num_shell;
                let bas_start_k = mol.cint_fdqc[k][0];
                let bas_len_k = mol.cint_fdqc[k][1];
                let l = i%num_shell;
                let bas_start_l = mol.cint_fdqc[l][0];
                let bas_len_l = mol.cint_fdqc[l][1];
                let mut klij = &mol.int_ijkl_given_kl(k, l);
                let mut sum =0.0;
                //let mut out = vec![(0.0, 0usize, 0usize); ];
                let mut out:Vec<(f64, usize, usize)> = Vec::new();
                    for ao_k in bas_start_k..bas_start_k+bas_len_k{
                        for ao_l in bas_start_l..bas_start_l+bas_len_l{
                            let mut sum =0.0;
                            let mut index_k = ao_k-bas_start_k;
                            let mut index_l = ao_l-bas_start_l;
                            let mut eri_cd = &klij[index_l * bas_len_k + index_k];
                            let eri_full = eri_cd.to_matrixupper().to_matrixfull().unwrap();
                            let mut v_cd = MatrixFull::new([num_basis, num_basis], 0.0);
                            v_cd.data.iter_mut().zip(dm_s.data.iter()).zip(eri_full.data.iter()).for_each(|((v,p),eri)|{
                                *v = *p * *eri
                            });

                            v_cd.data.iter().for_each(|x|{
                                sum += *x
                            });
                            out.push((sum, ao_k, ao_l));
                        }
                    }
                s.send(out).unwrap();
            });
            receiver.into_iter().for_each(|out_vec| {
                out_vec.iter().for_each(|(value,index_k,index_l)|{
                    vj_i[(*index_k, *index_l)] = *value
                })
            });


            vj.push(vj_i.to_matrixupper());
        }
        if spin_channel == 1{
            vj.push(MatrixUpper::new(1, 0.0));
        }
        vj

    }

    pub fn generate_vj_on_the_fly_par_new(&self) -> Vec<MatrixUpper<f64>>{
        let num_shell = self.mol.cint_bas.len();
        let num_basis = self.mol.num_basis;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut vj: Vec<MatrixUpper<f64>> = vec![];
        let mol = &self.mol;
        //utilities::omp_set_num_threads_wrapper(1);
        for i_spin in 0..spin_channel{
            let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
            let dm_s = &self.density_matrix[i_spin];
            let par_tasks = utilities::balancing(num_shell*num_shell, rayon::current_num_threads());
            let (sender, receiver) = channel();
            let mut index = Vec::new();
            for l in 0..num_shell {
                for k in 0..l+1 {
                    index.push((k,l))
                }
            };
            index.par_iter().for_each_with(sender,|s,(k,l)|{
                let bas_start_k = mol.cint_fdqc[*k][0];
                let bas_len_k = mol.cint_fdqc[*k][1];
                let bas_start_l = mol.cint_fdqc[*l][0];
                let bas_len_l = mol.cint_fdqc[*l][1];

                let klij = mol.int_ijkl_given_kl_v02(*k, *l);
                let mut sum =0.0;
                //let mut out = vec![(0.0, 0usize, 0usize); ];
                //let mut out:Vec<(f64, usize, usize)> = Vec::new();
                let mut out = MatrixFull::new([bas_len_k, bas_len_l],0.0);
                //for ao_k in bas_start_k..bas_start_k+bas_len_k{
                //    for ao_l in bas_start_l..bas_start_l+bas_len_l{
                out.iter_columns_full_mut().enumerate().for_each(|(loc_l,x)|{
                    x.iter_mut().enumerate().for_each(|(loc_k,elem)|{
                        let ao_k = loc_k + bas_start_k;
                        let ao_l = loc_l + bas_start_l;
                        let eri_cd = klij.get(&[loc_k, loc_l]).unwrap();
                        let mut sum = dm_s.iter_matrixupper().unwrap()
                            .zip(eri_cd.iter_matrixupper().unwrap())
                            .fold(0.0,|sum, (p,eri)| {
                            sum + *p * *eri
                        });

                        let mut diagonal = dm_s.iter_diagonal().unwrap().zip(eri_cd.iter_diagonal().unwrap()).fold(0.0,|diagonal, (p,eri)| {
                            diagonal + *p * *eri
                        });

                        sum = sum*2.0 - diagonal;

                        *elem = sum;
                    })
                });
                s.send((out,*k,*l)).unwrap();
            });
            receiver.into_iter().for_each(|(out,k,l)| {
                let bas_start_k = mol.cint_fdqc[k][0];
                let bas_len_k = mol.cint_fdqc[k][1];
                let bas_start_l = mol.cint_fdqc[l][0];
                let bas_len_l = mol.cint_fdqc[l][1];
                vj_i.copy_from_matr(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l, 
                    &out, 0..bas_len_k,0..bas_len_l);
                //vj_i.iter_submatrix_mut(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l).zip(out.iter())
                //    .for_each(|(to, from)| {*to = *from});
            });
            

            vj.push(vj_i.to_matrixupper());
        }
        if spin_channel == 1{
            vj.push(MatrixUpper::new(1, 0.0));       
        }
        vj
    }

    pub fn generate_vj_on_the_fly_par(&self) -> Vec<MatrixUpper<f64>> {
        vj_on_the_fly_par(&self.mol, &self.density_matrix)
        // IGOR MARK: still has bugs in the batch_by_batch version
        //vj_on_the_fly_par_batch_by_batch(&self.mol, &self.density_matrix)
    }


    pub fn generate_vk_with_erifold4(&mut self, scaling_factor: f64) -> Vec<MatrixFull<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixFull<f64>> = vec![MatrixFull::new([1,1],0.0f64),MatrixFull::new([1,1],0.0f64)];
        let dm = &self.density_matrix;
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixFull::new([num_basis,num_basis],0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                        let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                        //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                        let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                        let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                        let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*num_basis,num_basis).unwrap();
                        let mut kl = 0_usize;
                        for k in (0..num_basis) {
                            // The psuedo-code for the next several ten lines
                            //for l in (0..k) {
                            //    vk_ic[l] += reduce_ij[kl] *dm_jc[k];
                            //    vk_ic[k] += reduce_ij[kl] *dm_jc[l];
                            //    kl += 1;
                            //}
                            vk_ic[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                            vk_ic[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            //============================================
                            vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                            kl += 1;
                        }
                        let mut vk_jc = vk[i_spin].get1d_slice_mut(jc*num_basis,num_basis).unwrap();
                        let mut kl = 0_usize;
                        for k in (0..num_basis) {
                            // The psuedo-code for the next several ten lines
                            //for l in (0..k) {
                            //    vk_jc[l] += reduce_ij[kl] *dm_ic[k];
                            //    vk_jc[k] += reduce_ij[kl] *dm_ic[l];
                            //    kl += 1;
                            //}
                            vk_jc[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            vk_jc[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            //============================================
                            vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                            kl += 1;
                        }
                    }
                }
                for ic in (0..num_basis) {
                    let ijkl_start = (ic*(ic+1)/2+ic)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                    let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*num_basis,num_basis).unwrap();
                    let mut kl = 0_usize;
                    for k in (0..num_basis) {
                        // The psuedo-code for the next several ten lines
                        //for l in (0..k) {
                        //    vk_ic[l] += reduce_ij[kl] *dm_ic[k];
                        //    vk_ic[k] += reduce_ij[kl] *dm_ic[l];
                        //    kl += 1;
                        //}
                        vk_ic[..k].par_iter_mut()
                            .zip(reduce_ij[kl..kl+k].par_iter())
                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                        vk_ic[k] += reduce_ij[kl..kl+k]
                            .iter()
                            .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                        kl += k;
                        //=================================================
                        vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                        kl += 1;
                    }
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }
    pub fn generate_vk_with_erifold4_v02(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixUpper::new(npair,0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                        let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                        //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                        let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                        let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                        let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                        let mut kl = 0_usize;
                        // The psuedo-code for the next several ten lines
                        //for k in (0..num_basis) {
                        //    for l in (0..k) {
                        //        vk_ic[l] += reduce_ij[kl] *dm_jc[k];
                        //        vk_ic[k] += reduce_ij[kl] *dm_jc[l];
                        //        kl += 1;
                        //    }
                        //}
                        //    vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                        //    kl += 1;
                        //==============================================
                        for k in (0..num_basis) {
                            if k<=ic {
                                vk_ic[..k].iter_mut()
                                    .zip(reduce_ij[kl..kl+k].iter())
                                    .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                vk_ic[k] += reduce_ij[kl..kl+k]
                                    .iter()
                                    .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                kl += k;
                                vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                                kl += 1;
                            } else {
                                vk_ic[..ic+1].iter_mut()
                                    .zip(reduce_ij[kl..kl+ic+1].iter())
                                    .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                kl += k+1;
                            }
                            //if ic==4 && k==35 {println!("{}",kl)};
                        }
                        //=================================================
                        // try rayon parallel version
                        //for k in (0..num_basis) {
                        //    if k<=ic {
                        //        vk_ic[..k].iter_mut()
                        //            .zip(reduce_ij[kl..kl+k].iter())
                        //            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                        //        vk_ic[k] += reduce_ij[kl..kl+k]
                        //            .par_iter()
                        //            .zip(dm_jc[..k].par_iter()).map(|(i,j)| i*j).sum::<f64>();
                        //        kl += k;
                        //        vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                        //        kl += 1;我们的描述子是1*10
                        //    } else {
                        //        vk_ic[..ic+1].iter_mut()
                        //            .zip(reduce_ij[kl..kl+ic+1].iter())
                        //            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                        //        kl += k+1;
                        //    }
                        //}
                        //=================================================
                        let mut vk_jc = vk[i_spin].get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                        let mut kl = 0_usize;
                        // The psuedo-code for the next several ten lines
                        //for k in (0..num_basis) {
                        //    for l in (0..k) {
                        //        vk_jc[l] += reduce_ij[kl] *dm_ic[k];
                        //        vk_jc[k] += reduce_ij[kl] *dm_ic[l];
                        //        kl += 1;
                        //    }
                        //    vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                        //    kl += 1;
                        //}
                        for k in (0..num_basis) {
                            if k<=jc {
                                vk_jc[..k].iter_mut()
                                    .zip(reduce_ij[kl..kl+k].iter())
                                    .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                vk_jc[k] += reduce_ij[kl..kl+k]
                                    .iter()
                                    .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                kl += k;
                                vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                                kl += 1;
                            } else {
                                vk_jc[..jc+1].iter_mut()
                                    .zip(reduce_ij[kl..kl+jc+1].iter())
                                    .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                kl += k+1;
                            }
                        }
                        //=================================================
                    }
                }
                for ic in (0..num_basis) {
                    let ijkl_start = (ic*(ic+1)/2+ic)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                    let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                    let mut kl = 0_usize;
                    // The psuedo-code for the next several ten lines
                    //for k in (0..num_basis) {
                    //    for l in (0..k) {
                    //        vk_ic[l] += reduce_ij[kl] *dm_ic[k];
                    //        vk_ic[k] += reduce_ij[kl] *dm_ic[l];
                    //        kl += 1;
                    //    }
                    //    vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                    //    kl += 1;
                    //}
                    for k in (0..num_basis) {
                        if k<=ic {
                            vk_ic[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            vk_ic[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                            kl += 1;
                        } else {
                            vk_ic[..ic+1].iter_mut()
                                .zip(reduce_ij[kl..kl+ic+1].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            kl += k+1;
                        }
                    }
                    //=================================================
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }
    pub fn generate_vk_with_erifold4_sync(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        let num_para: usize = if let Some(num_para) = self.mol.ctrl.num_threads {
            num_para
        } else {
            1
        };
        let num_chunck = if num_basis%num_para==0 {
            (num_basis/num_para,num_basis/num_para)
        } else {
            (num_basis/num_para+1,num_basis-(num_basis/num_para+1)*(num_para-1))
        };
        println!("num_threads: ({},{}),num_chunck: ({},{})",
                num_para,num_basis,
                num_chunck.0,
                num_chunck.1);
                //if num_basis%num_para==0 {num_chunck.0} else {num_basis%num_para});
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixUpper::new(npair,0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        scope(|s_thread| {
                            let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                            let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                            //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                            let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                            let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                            let (tx_ic,rx_ic) = unbounded();
                            let (tx_jc,rx_jc) = unbounded();
                            //println!("Main thread: {:?}",thread::current().id());
                            for f in (0..num_para-1) {
                                let ic_thread = ic;
                                let jc_thread = jc;
                                let k_start_thread = f*num_chunck.0;
                                let k_end_thread = k_start_thread + num_chunck.0;
                                let tx_ic_thread = tx_ic.clone();
                                let tx_jc_thread = tx_jc.clone();
                                let mut kl_thread = k_start_thread*(k_start_thread+1)/2;
                                let handle = s_thread.spawn(move |_| {
                                    let mut vk_ic_thread = vec![0.0;ic_thread+1];
                                    let mut vk_jc_thread = vec![0.0;jc_thread+1];
                                    //let handle_thread = thread::current();
                                    //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                    for k in (k_start_thread..k_end_thread) {
                                        let mut kl_jc_thread = kl_thread;
                                        if k<=ic_thread {
                                            vk_ic_thread[..k].iter_mut()
                                                .zip(reduce_ij[kl_thread..kl_thread+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            vk_ic_thread[k] += reduce_ij[kl_thread..kl_thread+k]
                                                .iter()
                                                .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl_thread += k;
                                            vk_ic_thread[k] += reduce_ij[kl_thread] *dm_jc[k];
                                            kl_thread += 1;
                                        } else {
                                            vk_ic_thread[..ic+1].iter_mut()
                                                .zip(reduce_ij[kl_thread..kl_thread+ic+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            kl_thread += k+1;
                                        }
                                        if k<=jc_thread {
                                            vk_jc_thread[..k].iter_mut()
                                                .zip(reduce_ij[kl_jc_thread..kl_jc_thread+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            vk_jc_thread[k] += reduce_ij[kl_jc_thread..kl_jc_thread+k]
                                                .iter()
                                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl_jc_thread += k;
                                            vk_jc_thread[k] += reduce_ij[kl_jc_thread] *dm_ic[k];
                                            kl_jc_thread += 1;
                                        } else {
                                            vk_jc_thread[..jc+1].iter_mut()
                                                .zip(reduce_ij[kl_jc_thread..kl_jc_thread+jc_thread+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            kl_jc_thread += k+1;
                                        }
                                    }
                                    //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                    tx_ic_thread.send(vk_ic_thread).unwrap();
                                    tx_jc_thread.send(vk_jc_thread).unwrap();
                                });
                                //handles.push(handle);
                            }
                            let ic_thread = ic;
                            let jc_thread = jc;
                            let k_start_thread = (num_para-1)*num_chunck.0;
                            let k_end_thread = k_start_thread+num_chunck.1;
                            //let reduce_ij_thread = reduce_ij.clone();
                            //let dm_ic_thread = dm_ic.clone();
                            //let dm_jc_thread = dm_jc.clone();
                            let mut kl_thread = k_start_thread*(k_start_thread+1)/2;
                            let tx_ic_thread = tx_ic;
                            let tx_jc_thread = tx_jc;
                            let handle = s_thread.spawn(move |_| {
                                let mut vk_ic_thread = vec![0.0;ic_thread+1];
                                let mut vk_jc_thread = vec![0.0;jc_thread+1];
                                //let handle_thread = thread::current();
                                //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                for k in (k_start_thread..k_end_thread) {
                                    let mut kl_jc_thread = kl_thread;
                                    if k<=ic_thread {
                                        vk_ic_thread[..k].iter_mut()
                                            .zip(reduce_ij[kl_thread..kl_thread+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        vk_ic_thread[k] += reduce_ij[kl_thread..kl_thread+k]
                                            .iter()
                                            .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl_thread += k;
                                        vk_ic_thread[k] += reduce_ij[kl_thread] *dm_jc[k];
                                        kl_thread += 1;
                                    } else {
                                        vk_ic_thread[..ic+1].iter_mut()
                                            .zip(reduce_ij[kl_thread..kl_thread+ic+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        kl_thread += k+1;
                                    }
                                    if k<=jc_thread {
                                        vk_jc_thread[..k].iter_mut()
                                            .zip(reduce_ij[kl_jc_thread..kl_jc_thread+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        vk_jc_thread[k] += reduce_ij[kl_jc_thread..kl_jc_thread+k]
                                            .iter()
                                            .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl_jc_thread += k;
                                        vk_jc_thread[k] += reduce_ij[kl_jc_thread] *dm_ic[k];
                                        kl_jc_thread += 1;
                                    } else {
                                        vk_jc_thread[..jc+1].iter_mut()
                                            .zip(reduce_ij[kl_jc_thread..kl_jc_thread+jc_thread+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        kl_jc_thread += k+1;
                                    }
                                }
                                //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                tx_ic_thread.send(vk_ic_thread).unwrap();
                                tx_jc_thread.send(vk_jc_thread).unwrap();
                            });
                            //handles.push(handle);
                            {
                                let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                                for received in rx_ic {
                                    vk_ic.iter_mut()
                                        .zip(received)
                                        .for_each(|(i,j)| {*i += j});
                                }
                            }
                            {
                                let mut vk_jc = vk[i_spin].get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                for received in rx_jc {
                                    vk_jc.iter_mut()
                                        .zip(received)
                                        .for_each(|(i,j)| {*i += j});
                                }
                            }
                        }).unwrap();
                    }
                }
                for ic in (0..num_basis) {
                    let ijkl_start = (ic*(ic+1)/2+ic)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                    let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                    let mut kl = 0_usize;
                    // The psuedo-code for the next several ten lines
                    //for k in (0..num_basis) {
                    //    for l in (0..k) {
                    //        vk_ic[l] += reduce_ij[kl] *dm_ic[k];
                    //        vk_ic[k] += reduce_ij[kl] *dm_ic[l];
                    //        kl += 1;
                    //    }
                    //    vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                    //    kl += 1;
                    //}
                    for k in (0..num_basis) {
                        if k<=ic {
                            vk_ic[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            vk_ic[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                            kl += 1;
                        } else {
                            vk_ic[..ic+1].iter_mut()
                                .zip(reduce_ij[kl..kl+ic+1].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            kl += k+1;
                        }
                    }
                    //=================================================
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }
    pub fn generate_vk_with_erifold4_sync_v02(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        let num_para: usize = if let Some(num_para) = self.mol.ctrl.num_threads {
            num_para
        } else {
            1
        };
        let num_chunck = if num_basis%num_para==0 {
            (num_basis/num_para,num_basis/num_para)
        } else {
            (num_basis/num_para+1,num_basis-(num_basis/num_para+1)*(num_para-1))
        };
        //println!("num_threads: ({},{}),num_chunck: ({},{})",
        //        num_para,num_basis,
        //        num_chunck.0,
        //        num_chunck.1);
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixUpper::new(npair,0.0f64);
                scope(|s_thread| {
                    let (tx_jc,rx_jc) = unbounded();
                    for f in (0..num_para-1) {
                        let jc_start_thread = f*num_chunck.0;
                        let jc_end_thread = jc_start_thread + num_chunck.0;
                        let tx_jc_thread = tx_jc.clone();
                        let handle = s_thread.spawn(move |_| {
                            let mut vk_thread =  MatrixUpper::new(npair,0.0f64);
                            for jc in (jc_start_thread..jc_end_thread) {
                                for ic in (0..jc) {
                                    let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                    //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                                    let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                                    let mut vk_ic = vk_thread.get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                                    let mut kl = 0_usize;
                                    for k in (0..num_basis) {
                                        if k<=ic {
                                            vk_ic[..k].iter_mut()
                                                .zip(reduce_ij[kl..kl+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            vk_ic[k] += reduce_ij[kl..kl+k]
                                                .iter()
                                                .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl += k;
                                            vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                                            kl += 1;
                                        } else {
                                            vk_ic[..ic+1].iter_mut()
                                                .zip(reduce_ij[kl..kl+ic+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            kl += k+1;
                                        }
                                    }
                                    //=================================================
                                    let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                    let mut kl = 0_usize;
                                    for k in (0..num_basis) {
                                        if k<=jc {
                                            vk_jc[..k].iter_mut()
                                                .zip(reduce_ij[kl..kl+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            vk_jc[k] += reduce_ij[kl..kl+k]
                                                .iter()
                                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl += k;
                                            vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                                            kl += 1;
                                        } else {
                                            vk_jc[..jc+1].iter_mut()
                                                .zip(reduce_ij[kl..kl+jc+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            kl += k+1;
                                        }
                                    }
                                    //=================================================
                                }
                                let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                                let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                                let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                let mut kl = 0_usize;
                                for k in (0..num_basis) {
                                    if k<=jc {
                                        vk_jc[..k].iter_mut()
                                            .zip(reduce_ij[kl..kl+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        vk_jc[k] += reduce_ij[kl..kl+k]
                                            .iter()
                                            .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl += k;
                                        vk_jc[k] += reduce_ij[kl] *dm_jc[k];
                                        kl += 1;
                                    } else {
                                        vk_jc[..jc+1].iter_mut()
                                            .zip(reduce_ij[kl..kl+jc+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        kl += k+1;
                                    }
                                }
                            }
                            tx_jc_thread.send(vk_thread);
                        });
                    }
                    let jc_start_thread = (num_para-1)*num_chunck.0;
                    let jc_end_thread = jc_start_thread + num_chunck.1;
                    let tx_jc_thread = tx_jc;
                    let handle = s_thread.spawn(move |_| {
                        let mut vk_thread =  MatrixUpper::new(npair,0.0f64);
                        for jc in (jc_start_thread..jc_end_thread) {
                            for ic in (0..jc) {
                                let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                                let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                                let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                                let mut vk_ic = vk_thread.get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                                let mut kl = 0_usize;
                                for k in (0..num_basis) {
                                    if k<=ic {
                                        vk_ic[..k].iter_mut()
                                            .zip(reduce_ij[kl..kl+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        vk_ic[k] += reduce_ij[kl..kl+k]
                                            .iter()
                                            .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl += k;
                                        vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                                        kl += 1;
                                    } else {
                                        vk_ic[..ic+1].iter_mut()
                                            .zip(reduce_ij[kl..kl+ic+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        kl += k+1;
                                    }
                                }
                                //=================================================
                                let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                let mut kl = 0_usize;
                                for k in (0..num_basis) {
                                    if k<=jc {
                                        vk_jc[..k].iter_mut()
                                            .zip(reduce_ij[kl..kl+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        vk_jc[k] += reduce_ij[kl..kl+k]
                                            .iter()
                                            .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl += k;
                                        vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                                        kl += 1;
                                    } else {
                                        vk_jc[..jc+1].iter_mut()
                                            .zip(reduce_ij[kl..kl+jc+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        kl += k+1;
                                    }
                                }
                                //=================================================
                            }
                            let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                            let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                            //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                            let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                            let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                            let mut kl = 0_usize;
                            for k in (0..num_basis) {
                                if k<=jc {
                                    vk_jc[..k].iter_mut()
                                        .zip(reduce_ij[kl..kl+k].iter())
                                        .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                    vk_jc[k] += reduce_ij[kl..kl+k]
                                        .iter()
                                        .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                    kl += k;
                                    vk_jc[k] += reduce_ij[kl] *dm_jc[k];
                                    kl += 1;
                                } else {
                                    vk_jc[..jc+1].iter_mut()
                                        .zip(reduce_ij[kl..kl+jc+1].iter())
                                        .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                    kl += k+1;
                                }
                            }
                        }
                        tx_jc_thread.send(vk_thread);
                    });
                    for received in rx_jc {
                        vk[i_spin].data.iter_mut()
                            .zip(received.data)
                            .for_each(|(i,j)| {*i += j});
                    }
                }).unwrap();
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }

    pub fn generate_vk_with_isdf_new(&self, scaling_factor: f64) -> Vec<MatrixUpper<f64>>{
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![];
        let spin_channel = self.mol.spin_channel;
        let m = self.m.clone().unwrap();
        let tab_ao = self.tab_ao.clone().unwrap();
        let n_ip = m.size[0];

        for i_spin in 0..spin_channel{
            let mut dm_s = &self.density_matrix[i_spin];
            let nw =  self.homo[i_spin]+1;
            let mut kernel_mid = MatrixFull::new([n_ip,num_basis], 0.0);
            _dgemm(&tab_ao,(0..num_basis, 0..n_ip),'T',
                dm_s,(0..num_basis,0..num_basis),'N',
                &mut kernel_mid, (0..n_ip, 0..num_basis),
                1.0,0.0);

            let mut kernel = MatrixFull::new([n_ip,n_ip], 0.0);
            _dgemm(&kernel_mid,(0..n_ip, 0..num_basis),'N',
            &tab_ao, (0..num_basis, 0..n_ip),'N',
            &mut kernel, (0..n_ip,0..n_ip),
            1.0, 0.0);

            kernel.data.iter_mut().zip(m.data.iter()).for_each(|(x,y)|{
                *x *= *y * scaling_factor
            });

            let mut tmp = MatrixFull::new([num_basis, n_ip], 0.0);
            _dgemm(&tab_ao,(0..num_basis,0..n_ip),'N',
            &kernel,(0..n_ip,0..n_ip),'N',
            &mut tmp, (0..num_basis,0..n_ip),
            1.0, 0.0);
            let mut vk_i = MatrixFull::new([num_basis, num_basis], 0.0);
            _dgemm(&tmp,(0..num_basis,0..n_ip),'N',
            &tab_ao,(0..num_basis,0..n_ip),'T',
            &mut vk_i, (0..num_basis,0..num_basis),
            1.0, 0.0);
            vk.push(vk_i.to_matrixupper());
        }
        vk
    }

    pub fn generate_vk_with_isdf_dm_only(&mut self) -> Vec<MatrixUpper<f64>>{
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![];
        let eigv = &self.eigenvectors;
        let spin_channel = self.mol.spin_channel;
        let m = self.m.clone().unwrap();
        let tab_ao = self.tab_ao.clone().unwrap();
        let n_ip = m.size[0];

        for i_spin in 0..spin_channel{
            let occ_s =  &self.occupation[i_spin];
            let nw =  self.homo[i_spin]+1;

            let mut tab_mo = MatrixFull::new([nw,n_ip], 0.0);
            _dgemm(&eigv[i_spin],(0..num_basis, 0..nw),'T',
                &tab_ao,(0..num_basis,0..n_ip),'N',
                &mut tab_mo, (0..nw, 0..n_ip),
                1.0,0.0);

            let mut zip_m_mo = MatrixFull::new([n_ip,n_ip], 0.0);
            _dgemm(&tab_mo,(0..nw, 0..n_ip),'T',
            &tab_mo, (0..nw, 0..n_ip),'N',
            &mut zip_m_mo, (0..n_ip,0..n_ip),
            1.0, 0.0);

            zip_m_mo.data.iter_mut().zip(m.data.iter()).for_each(|(x,y)|{
                *x *= *y * (-1.0)
            });

            let mut tmp = MatrixFull::new([num_basis, n_ip], 0.0);
            _dgemm(&tab_ao,(0..num_basis,0..n_ip),'N',
            &zip_m_mo,(0..n_ip,0..n_ip),'N',
            &mut tmp, (0..num_basis,0..n_ip),
            1.0, 0.0);

            let mut vk_i = MatrixFull::new([num_basis, num_basis], 0.0);
            _dgemm(&tmp,(0..num_basis,0..n_ip),'N',
            &tab_ao,(0..num_basis,0..n_ip),'T',
            &mut vk_i, (0..num_basis,0..num_basis),
            1.0, 0.0);

            vk.push(vk_i.to_matrixupper());

        }
        vk
    }
    pub fn generate_hf_hamiltonian_erifold4(&mut self) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        //let homo = &self.homo;
        //let vj = if self.mol.ctrl.num_threads>1 {
        //    self.generate_vj_with_erifold4_sync(1.0);
        //} else {
        //    self.generate_vj_with_erifold4(1.0)
        //};
        //let vk = if self.mol.ctrl.num_threads>1 {
        //    self.generate_vk_with_erifold4_sync_v02(-0.5);
        //} else {
        //    self.generate_vk_with_erifold4_v02(-0.5)
        //};
        let vj = self.generate_vj_with_erifold4_sync(1.0);
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        };
        let vk = self.generate_vk_with_erifold4_sync(scaling_factor);
        // let tmp_matrix = &self.h_core;
        // let mut tmp_num = 0;
        // let (i_len,j_len) =  (self.mol.num_basis,self.mol.num_basis);
        // let (k_len,l_len) =  (self.mol.num_basis,self.mol.num_basis);
        // tmp_matrix.data.iter().enumerate().for_each(|value| {
        //     if value.1.abs()>1.0e-1 {
        //         println!("I= {:2} Value= {:16.8}",value.0,value.1);
        //     }
        // });
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[0].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[1].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vk[i_spin].data.par_iter())
                            .for_each(|(h_ij,vk_ij)| {
                                *h_ij += vk_ij
                            });
        };

        if let SCFType::ROHF = self.scftype {
            self.roothaan_hamiltonian = self.generate_roothaan_fock();
        }
    }
    pub fn generate_hf_hamiltonian_ri_v(&mut self, mpi_operator: &Option<MPIOperator>) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let dt1 = time::Local::now();
        let vj = if self.mol.ctrl.isdf_new || self.mol.ctrl.ri_k_only {
            self.generate_vj_on_the_fly_par()
        }else{
            self.generate_vj_with_ri_v_sync(1.0, mpi_operator)
        };


        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        };

        let use_dm_only = self.mol.ctrl.use_dm_only;
        let vk = if self.mol.ctrl.use_isdf && !self.mol.ctrl.isdf_new{
            self.generate_vk_with_isdf(scaling_factor, use_dm_only)
        }else if self.mol.ctrl.isdf_new{
            self.generate_vk_with_isdf_new(scaling_factor)
        }else{
            self.generate_vk_with_ri_v(scaling_factor, use_dm_only, mpi_operator)
        };


        let dt3 = time::Local::now();
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj and Vk matrices cost {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2);
        }
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[0].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[1].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vk[i_spin].data.par_iter())
                            .for_each(|(h_ij,vk_ij)| {
                                *h_ij += vk_ij
                            });
        };

        if let SCFType::ROHF = self.scftype {
            self.roothaan_hamiltonian = self.generate_roothaan_fock();
        }
    }
    pub fn generate_hf_hamiltonian_ri_v_dm_only(&mut self, mpi_operator: &Option<MPIOperator>) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        //let homo = &self.homo;
        let dt1 = time::Local::now();
        let vj = if self.mol.ctrl.isdf_new || self.mol.ctrl.ri_k_only {
            self.generate_vj_on_the_fly_par()
        } else {
            self.generate_vj_with_ri_v_sync(1.0, mpi_operator)
        };
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        };
        let vk = self.generate_vk_with_ri_v(scaling_factor, true, mpi_operator);
        let dt3 = time::Local::now();
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj and Vk matrices cost {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2);
        }
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[0].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[1].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vk[i_spin].data.par_iter())
                            .for_each(|(h_ij,vk_ij)| {
                                *h_ij += vk_ij
                            });
        };

        if let SCFType::ROHF = self.scftype {
            self.roothaan_hamiltonian = self.generate_roothaan_fock();
        }
    }

    pub fn generate_ks_hamiltonian_erifold4(&mut self) -> (f64,f64) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let mut exc_total = 0.0;
        let mut vxc_total = 0.0;
        //let homo = &self.homo;
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
        }
        let dt1 = time::Local::now();
        //let vj = self.generate_vj_with_ri_v_sync(1.0);
        let vj = self.generate_vj_with_erifold4_sync(1.0);
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin].data
                .par_iter_mut()
                .zip(vj[0].data.par_iter())
                .for_each(|(h_ij,vj_ij)| {
                    *h_ij += vj_ij
                });
            self.hamiltonian[i_spin].data
                .par_iter_mut()
                .zip(vj[1].data.par_iter())
                .for_each(|(h_ij,vj_ij)| {
                    *h_ij += vj_ij
                });
        }
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        }*self.mol.xc_data.dfa_hybrid_scf ;
        if ! scaling_factor.eq(&0.0) {
            //let vk = self.generate_vk_with_ri_v(scaling_factor);
            let vk = self.generate_vk_with_erifold4_sync(scaling_factor);
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                    .par_iter_mut()
                    .zip(vk[i_spin].data.par_iter())
                    .for_each(|(h_ij,vk_ij)| {
                        *h_ij += vk_ij
                    });
            };
        }
        let dt3 = time::Local::now();
        if self.mol.xc_data.dfa_compnt_scf.len()!=0 {
            let (exc,vxc) = self.generate_vxc(1.0);
            //println!("{:?}",vxc[0].data);
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                                .par_iter_mut()
                                .zip(vxc[i_spin].data.par_iter())
                                .for_each(|(h_ij,vk_ij)| {
                                    *h_ij += vk_ij
                                });
            };
            exc_total = exc;
            for i_spin in (0..spin_channel) {
                let dm_s = &self.density_matrix[i_spin];
                let dm_upper = dm_s.to_matrixupper();
                vxc_total += SCF::par_energy_contraction(&dm_upper, &vxc[i_spin]);
            }
        }

        let dt4 = time::Local::now();
        
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        let timecost3 = (dt4.timestamp_millis()-dt3.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj, Vk and Vxc matrices cost {:10.2}, {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2, timecost3);
        };

        if let SCFType::ROHF = self.scftype {
            self.roothaan_hamiltonian = self.generate_roothaan_fock();
        }
        (exc_total, vxc_total)

    }
    pub fn generate_ks_hamiltonian_ri_v(&mut self, mpi_operator: &Option<MPIOperator>) -> (f64,f64) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let mut exc_total = 0.0;
        let mut vxc_total = 0.0;
        let mut vk_total = 0.0;
        //let homo = &self.homo;
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
        }
        let dt1 = time::Local::now();
        //let use_eri = self.mol.xc_data.use_eri() || self.mol.xc_dat;
        //let use_eri = true;
        let vj = if self.mol.ctrl.use_ri_vj {
            self.generate_vj_with_ri_v_sync(1.0, mpi_operator)
        } else {
            self.generate_vj_on_the_fly_par()
        };
        //// ==== DEBUG IGOR ====
        //if let Some(mpi_op) = &mpi_operator {
        //    if mpi_op.rank == 0 {
        //        vj[0].formated_output(5, "full");
        //    }
        //} else {
        //    vj[0].formated_output(5, "full");
        //}
        //// ==== DEBUG IGOR ====
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin].data
                .par_iter_mut()
                .zip(vj[0].data.par_iter())
                .for_each(|(h_ij,vj_ij)| {
                    *h_ij += vj_ij
                });
            self.hamiltonian[i_spin].data
                .par_iter_mut()
                .zip(vj[1].data.par_iter())
                .for_each(|(h_ij,vj_ij)| {
                    *h_ij += vj_ij
                });
        }
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        }*self.mol.xc_data.dfa_hybrid_scf ;
        if ! scaling_factor.eq(&0.0) {
            let use_dm_only = self.mol.ctrl.use_dm_only;
            //self.mol.ctrl.use_dm_only
            let vk = self.generate_vk_with_ri_v(scaling_factor, use_dm_only, mpi_operator);
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                    .par_iter_mut()
                    .zip(vk[i_spin].data.par_iter())
                    .for_each(|(h_ij,vk_ij)| {
                        *h_ij += vk_ij
                    });
            };
        }
        let dt3 = time::Local::now();
        if self.mol.xc_data.dfa_compnt_scf.len()!=0 {
            //let (exc,vxc) = self.generate_vxc_rayon_dm_only(1.0);
            let (_, exc,vxc) = if self.mol.ctrl.use_dm_only {
                self.generate_vxc_mpi_rayon_dm_only(1.0, mpi_operator)
            } else {
                self.generate_vxc_mpi_rayon(1.0, mpi_operator)
            };
            //let (exc,vxc) = self.generate_vxc(1.0);
            let _ = utilities::timing(&dt3, Some("evaluate vxc total"));
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                                .par_iter_mut()
                                .zip(vxc[i_spin].data.par_iter())
                                .for_each(|(h_ij,vk_ij)| {
                                    *h_ij += vk_ij
                                });
            };
            exc_total = exc;
            for i_spin in (0..spin_channel) {
                let dm_s = &self.density_matrix[i_spin];
                let dm_upper = dm_s.to_matrixupper();
                vxc_total += SCF::par_energy_contraction(&dm_upper, &vxc[i_spin]);
            }
        };

        let dt4 = time::Local::now();

        if let SCFType::ROHF = self.scftype {
            self.roothaan_hamiltonian = self.generate_roothaan_fock();
            let dt5 = time::Local::now();
            let timecost4 = (dt5.timestamp_millis()-dt4.timestamp_millis()) as f64 /1000.0;
            if self.mol.ctrl.print_level > 2 {
                println!("The evaluation of Roothaan effective Fock Matrix costs {:10.2} seconds.", timecost4);
            }
        }
        
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        let timecost3 = (dt4.timestamp_millis()-dt3.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj, Vk and Vxc matrices cost {:10.2}, {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2, timecost3);
        };
        (exc_total, vxc_total)

    }

//    pub fn generate_ks_hamiltonian_ri_v_dm_only(&mut self, mpi_operator: &Option<MPIOperator>) -> (f64,f64) {
//        let num_basis = self.mol.num_basis;
//        let num_state = self.mol.num_state;
//        let spin_channel = self.mol.spin_channel;
//        let mut exc_total = 0.0;
//        let mut vxc_total = 0.0;
//        let mut vk_total = 0.0;
//        //let homo = &self.homo;
//        for i_spin in (0..spin_channel) {
//            self.hamiltonian[i_spin] = self.h_core.clone();
//        }
//        let dt1 = time::Local::now();
//        let vj = self.generate_vj_with_ri_v_sync(1.0);
//        for i_spin in (0..spin_channel) {
//            self.hamiltonian[i_spin].data
//                .par_iter_mut()
//                .zip(vj[0].data.par_iter())
//                .for_each(|(h_ij,vj_ij)| {
//                    *h_ij += vj_ij
//                });
//            self.hamiltonian[i_spin].data
//                .par_iter_mut()
//                .zip(vj[1].data.par_iter())
//                .for_each(|(h_ij,vj_ij)| {
//                    *h_ij += vj_ij
//                });
//        }
//        let dt2 = time::Local::now();
//        let scaling_factor = match self.scftype {
//            SCFType::RHF => -0.5,
//            _ => -1.0,
//        }*self.mol.xc_data.dfa_hybrid_scf ;
//        if ! scaling_factor.eq(&0.0) {
//            let use_dm_only = self.mol.ctrl.use_dm_only;
//            let vk = if self.mol.ctrl.use_isdf{
//                self.generate_vk_with_isdf(scaling_factor, use_dm_only)
//            }else{
//                self.generate_vk_with_ri_v(scaling_factor, use_dm_only)
//            };
//            for i_spin in (0..spin_channel) {
//                self.hamiltonian[i_spin].data
//                    .par_iter_mut()
//                    .zip(vk[i_spin].data.par_iter())
//                    .for_each(|(h_ij,vk_ij)| {
//                        *h_ij += vk_ij
//                    });
//            };
//        }
//        let dt3 = time::Local::now();
//        if self.mol.xc_data.dfa_compnt_scf.len()!=0 {
//            let (_, exc,vxc) = self.generate_vxc_mpi_rayon_dm_only(1.0, mpi_operator);
//            //let (exc,vxc) = self.generate_vxc(1.0);
//            let _ = utilities::timing(&dt3, Some("evaluate vxc total"));
//            for i_spin in (0..spin_channel) {
//                self.hamiltonian[i_spin].data
//                                .par_iter_mut()
//                                .zip(vxc[i_spin].data.par_iter())
//                                .for_each(|(h_ij,vk_ij)| {
//                                    *h_ij += vk_ij
//                                });
//            };
//            exc_total = exc;
//            for i_spin in (0..spin_channel) {
//                let dm_s = &self.density_matrix[i_spin];
//                let dm_upper = dm_s.to_matrixupper();
//                vxc_total += SCF::par_energy_contraction(&dm_upper, &vxc[i_spin]);
//            }
//        }
//
//        let dt4 = time::Local::now();
//        
//        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
//        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
//        let timecost3 = (dt4.timestamp_millis()-dt3.timestamp_millis()) as f64 /1000.0;
//        if self.mol.ctrl.print_level>2 {
//            println!("The evaluation of Vj, Vk and Vxc matrices cost {:10.2}, {:10.2} and {:10.2} seconds, respectively",
//                      timecost1,timecost2, timecost3);
//        };
//        (exc_total, vxc_total)
//
//    }

    pub fn generate_hf_hamiltonian_for_guess(&mut self) {
        if self.mol.xc_data.dfa_compnt_scf.len() == 0 {
            if self.mol.ctrl.eri_type.eq("analytic") {
                self.generate_hf_hamiltonian_erifold4();
            } else if  self.mol.ctrl.eri_type.eq("ri_v") {
                self.generate_hf_hamiltonian_ri_v_dm_only(&None);
            }
        } else {
            if self.mol.ctrl.eri_type.eq("analytic") {
                self.generate_ks_hamiltonian_erifold4();
            } else if  self.mol.ctrl.eri_type.eq("ri_v") {
                let origin_dm_only = self.mol.ctrl.use_dm_only;
                self.mol.ctrl.use_dm_only = true;
                self.generate_ks_hamiltonian_ri_v(&None);
                self.mol.ctrl.use_dm_only = origin_dm_only;
            }
        }
    }

    pub fn evaluate_hf_total_energy(&self) -> f64 {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut total_energy = self.nuc_energy;
        match self.scftype {
            SCFType::RHF => {
                // D*(H^{core}+F)
                let dm_s = &dm[0];
                let hc = &self.h_core;
                let ht_s = &self.hamiltonian[0];
                let dm_upper = dm_s.to_matrixupper();
                let mut hc_and_ht = hc.clone();
                hc_and_ht.data.par_iter_mut().zip(ht_s.data.par_iter()).for_each(|value| {
                    *value.0 += value.1
                });
                total_energy += SCF::par_energy_contraction(&dm_upper, &hc_and_ht);
            },
            _ => {
                let dm_a = &dm[0];
                let dm_a_upper = dm_a.to_matrixupper();
                let dm_b = &dm[1];
                let dm_b_upper = dm_b.to_matrixupper();
                let mut dm_t_upper = dm_a_upper.clone();
                dm_t_upper.data.par_iter_mut().zip(dm_b_upper.data.par_iter()).for_each(|value| {*value.0+=value.1});

                // Now for D^{tot}*H^{core} term
                total_energy += SCF::par_energy_contraction(&dm_t_upper, &self.h_core);
                // Now for D^{alpha}*F^{alpha} term
                total_energy += SCF::par_energy_contraction(&dm_a_upper, &self.hamiltonian[0]);
                // Now for D^{beta}*F^{beta} term
                total_energy += SCF::par_energy_contraction(&dm_b_upper, &self.hamiltonian[1]);

            },
        }
        total_energy
    }

    pub fn generate_hf_hamiltonian(&mut self, mpi_operator: &Option<MPIOperator>) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let mut exc_total = 0.0;
        let mut vxc_total = 0.0;
        if self.mol.xc_data.dfa_compnt_scf.len() == 0 {
            if self.mol.ctrl.eri_type.eq("analytic") {
                self.generate_hf_hamiltonian_erifold4();
            } else if  self.mol.ctrl.eri_type.eq("ri_v") {
                self.generate_hf_hamiltonian_ri_v(mpi_operator);
            }
        } else {
            if self.mol.ctrl.eri_type.eq("analytic") {
                //panic!("Hybrid DFA is not implemented with analytic ERI.");
                let (tmp_exc_total,tmp_vxc_total) = self.generate_ks_hamiltonian_erifold4();
                exc_total = tmp_exc_total;
                vxc_total = tmp_vxc_total;
            } else {
                let (tmp_exc_total,tmp_vxc_total) = self.generate_ks_hamiltonian_ri_v(mpi_operator);
                exc_total = tmp_exc_total;
                vxc_total = tmp_vxc_total;
            }
        }

        let dm = &self.density_matrix;

        // The following scf energy evaluation follow the formula presented in
        // the quantum chemistry book of Szabo A. and Ostlund N.S. P 150, Formula (3.184)
        self.scf_energy = self.nuc_energy;
        // for DFT calculations, we should replace the exchange-correlation (xc) potential by the xc energy
        //if self.mol.ctrl.print_level>1 {println!("Exc: {:?}, Vxc: {:?}", exc_total, vxc_total)};
        self.scf_energy = self.scf_energy - vxc_total + exc_total;
        //if let Some(local_grids) = &self.grids {
        //    let total_elec = numerical_density(&local_grids, &self.mol, dm, mpi_operator);
        //    if self.mol.ctrl.print_level>1 {
        //        if self.mol.spin_channel==1 {
        //            println!("total electron number: {:16.8}", total_elec[0])
        //        } else {
        //            println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
        //            println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
        //        }
        //    };
        //}
        //println!("==== IGOR debug for Exc[HF]====");
        //let exc_hf = self.evaluate_exact_exchange_ri_v(mpi_operator);
        //println!("Exc[HF] = {:16.8}", exc_hf);
        //println!("==== IGOR debug for Exc[HF]====");

        if self.mol.ctrl.print_level>1 {
            println!("Exc: {:16.8}, Vxc: {:16.8}", exc_total, vxc_total)
        };
        match self.scftype {
            SCFType::RHF => {
                // D*(H^{core}+F)
                let dm_s = &dm[0];
                let hc = &self.h_core;
                let ht_s = &self.hamiltonian[0];
                let dm_upper = dm_s.to_matrixupper();
                let mut hc_and_ht = hc.clone();
                hc_and_ht.data.par_iter_mut().zip(ht_s.data.par_iter()).for_each(|value| {
                    *value.0 += value.1
                });
                self.scf_energy += SCF::par_energy_contraction(&dm_upper, &hc_and_ht);
            },
            _ => {
                let dm_a = &dm[0];
                let dm_a_upper = dm_a.to_matrixupper();
                let dm_b = &dm[1];
                let dm_b_upper = dm_b.to_matrixupper();
                let mut dm_t_upper = dm_a_upper.clone();
                dm_t_upper.data.par_iter_mut().zip(dm_b_upper.data.par_iter()).for_each(|value| {*value.0+=value.1});

                // Now for D^{tot}*H^{core} term
                self.scf_energy += SCF::par_energy_contraction(&dm_t_upper, &self.h_core);
                // Now for D^{alpha}*F^{alpha} term
                self.scf_energy += SCF::par_energy_contraction(&dm_a_upper, &self.hamiltonian[0]);
                // Now for D^{beta}*F^{beta} term
                self.scf_energy += SCF::par_energy_contraction(&dm_b_upper, &self.hamiltonian[1]);

            },
        }

        match self.mol.ctrl.xc_type {
            DFTType::DeepLearning => {
                // For the Deep-Learning DFA (DL-DFA) developed by ShenBi  -- Coded by IGOR/ 2025/3/22
                // 1) evaluate the energy_components with respect to the current xc_data.dfa_paramr_scf
                let energy_components = self.evaluate_energy_components(mpi_operator);
                // 2) evaluate the DL-DFA total energy
                let exc_dldfa = crate::dft::deep_learning::dl_hybrid_xc_energy(&energy_components);
                // 3) update the DL-DFA xc potential 
                let next_dfa_paramr_scf = crate::dft::deep_learning::dl_hybrid_xc_param(&energy_components);
                self.mol.xc_data.dfa_hybrid_scf = next_dfa_paramr_scf[1];
                self.mol.xc_data.dfa_paramr_scf = next_dfa_paramr_scf[2..].to_vec();
            },
            _ => {}
        }
         
    }

        pub fn generate_roothaan_fock(&self) -> MatrixUpper<f64> {
        //generate Roothaan's effective Fock matrix
        // ======== ======== ====== =========
        // space     closed   open   virtual
        // ======== ======== ====== =========
        // closed      Fc      Fb     Fc
        // open        Fb      Fc     Fa
        // virtual     Fc      Fa     Fc
        // ======== ======== ====== =========
        // where Fc = (Fa + Fb) / 2, Roothaan's Fock matrix for core
        let num_basis = self.mol.num_basis;

        let hamiltonian_a: MatrixFull<f64> = self.hamiltonian[0].to_matrixfull().unwrap();
        let hamiltonian_b: MatrixFull<f64> = self.hamiltonian[1].to_matrixfull().unwrap();
        let hamiltonian_c: MatrixFull<f64> = (hamiltonian_a.clone() + hamiltonian_b.clone()) * 0.5;

        // Projector for core, open-shell, and virtual space
        // pc = dm_b * ovlp
        // po = (dm_a - dm_b) * ovlp
        // pv = eye - dm_a * ovlp
        let mut pc = MatrixFull::new([num_basis, num_basis], 0.0);
        let mut po = MatrixFull::new([num_basis, num_basis], 0.0);
        let mut pv = MatrixFull::new([num_basis, num_basis], 0.0);
        let ovlp_full = self.ovlp.to_matrixfull().unwrap();
        
        //_dgemm(&self.density_matrix[1], (0..num_basis,0..num_basis), 'N', &ovlp_full, (0..num_basis,0..num_basis), 'N', &mut pc, (0..num_basis,0..num_basis), 1.0, 0.0);
        _dsymm(&self.density_matrix[1], &ovlp_full, &mut pc, 'L', 'U', 1.0, 0.0);
        
        let dm_o = self.density_matrix[0].clone() - self.density_matrix[1].clone();
        //_dgemm(&dm_o, (0..num_basis,0..num_basis), 'N', &ovlp_full, (0..num_basis,0..num_basis), 'N', &mut po, (0..num_basis,0..num_basis), 1.0, 0.0);
        _dsymm(&dm_o, &ovlp_full, &mut po, 'L', 'U', 1.0, 0.0);
        
        pv.iter_diagonal_mut().unwrap().for_each(|x| *x = 1.0);
        //_dgemm(&self.density_matrix[0], (0..num_basis,0..num_basis), 'N', &ovlp_full, (0..num_basis,0..num_basis), 'N', &mut pv, (0..num_basis,0..num_basis), -1.0, 1.0);
        _dsymm(&self.density_matrix[0], &ovlp_full, &mut pv, 'L', 'U', -1.0, 1.0);
        
        let mut roothaan_fock = MatrixFull::new([num_basis, num_basis], 0.0);
        roothaan_fock += apply_projection_operator(&pc, &hamiltonian_c, &pc) * 0.5;
        roothaan_fock += apply_projection_operator(&po, &hamiltonian_c, &po) * 0.5;
        roothaan_fock += apply_projection_operator(&pv, &hamiltonian_c, &pv) * 0.5;
        roothaan_fock += apply_projection_operator(&po, &hamiltonian_b, &pc);
        roothaan_fock += apply_projection_operator(&po, &hamiltonian_a, &pv);
        roothaan_fock += apply_projection_operator(&pv, &hamiltonian_c, &pc);
        roothaan_fock = roothaan_fock.clone() + roothaan_fock.transpose();
        
        roothaan_fock.to_matrixupper()
    }

    pub fn evaluate_energy_components(&self, mpi_operator: &Option<MPIOperator>) -> Vec<f64> {
        /// 1. the ordering is given by E_noXC, Ex_HF, and a sequence of DFA components in self.xc_data.dfa_compnt_scf
        /// 2. this function should be invoked after generate_hf_hamiltonian()
        let exc_hf = if self.mol.ctrl.eri_type.eq("ri_v") {
            self.evaluate_exact_exchange_ri_v(mpi_operator)
        } else {
            panic!("Only RI_V version has been implemented for SCF::evaluate_energy_components")
        };
        //println!("Exc[HF] = {:16.8}", exc_hf);
        let current_dfa = &self.mol.xc_data;
        let xc_energy_list = if let Some(grids) = &self.grids {
            let xc_code_list = &current_dfa.dfa_compnt_scf;
            self.mol.xc_data.xc_exc_list(
                xc_code_list, 
                grids, 
                &self.density_matrix, 
                &self.eigenvectors, 
                &self.occupation
            )
        } else {
            let code_list = &self.mol.xc_data.dfa_compnt_scf;
            vec![[0.0,0.0];code_list.len()]
        };

        let mut xc_energy_list: Vec<f64> = xc_energy_list.iter().map(|energy| energy[0]+energy[1]).collect();

        if let Some(mpi_world) = mpi_operator {
            let mut tot_xc_list = mpi_reduce(&mpi_world.world, &xc_energy_list, 0, &SystemOperation::sum());
            mpi_broadcast(&mpi_world.world, &mut tot_xc_list, 0);
            xc_energy_list = tot_xc_list;
        }

        let mut exc_total = exc_hf * current_dfa.dfa_hybrid_scf;
        xc_energy_list.iter().zip(current_dfa.dfa_paramr_scf.iter()).for_each(|(xc_energy, &param)| {
            exc_total += xc_energy * param
        });

        let e_noxc = self.scf_energy - exc_total;
        //let mut energy_components = vec![self.scf_energy-exc_hf];

        xc_energy_list.insert(0, exc_hf);
        xc_energy_list.insert(0, e_noxc);

        xc_energy_list
    }

    /// about total energy contraction:
    /// E0 = 1/2*\sum_{i}\sum_{j}a_{ij}*b_{ij}
    /// einsum('ij,ij')
    pub fn par_energy_contraction(a:&MatrixUpper<f64>, b:&MatrixUpper<f64>) -> f64 {
        let (sender, receiver) = channel();
        a.data.par_iter().zip(b.data.par_iter()).for_each_with(sender, |s,(dm,hc)| {
            let mut tmp_scf_energy = 0.0_f64;
            tmp_scf_energy += dm*(hc);
            s.send(tmp_scf_energy).unwrap();
        });
        let mut tmp_energy = 2.0_f64 * receiver.into_iter().sum::<f64>();
        let a_diag = a.get_diagonal_terms().unwrap();
        let b_diag = b.get_diagonal_terms().unwrap();
        let double_count = a_diag.par_iter().zip(b_diag.par_iter()).fold(|| 0.0_f64,|acc,(a,b)| {
            acc + *a*(*b)
        }).sum::<f64>();

        (tmp_energy - double_count) * 0.5
    }

    pub fn evaluate_exact_exchange_ri_v(&self, mpi_operator: &Option<MPIOperator>) -> f64 {
        let mut x_energy = 0.0;
        let use_dm_only = self.mol.ctrl.use_dm_only;
        //let mut vk = self.generate_vk_with_ri_v(1.0, use_dm_only);
        let mut vk = if self.mol.ctrl.use_isdf{
            self.generate_vk_with_isdf(1.0, use_dm_only)
        }else{
            self.generate_vk_with_ri_v(1.0, use_dm_only, mpi_operator)
        };
        let spin_channel = self.mol.spin_channel;
        for i_spin in 0..spin_channel {
            let dm_s = &self.density_matrix[i_spin];
            let dm_upper = dm_s.to_matrixupper();
            x_energy += SCF::par_energy_contraction(&dm_upper, &vk[i_spin]);
        }
        if self.mol.spin_channel==1 {
            // the factor of 0.5 is due to the use of full density matrix for the exchange energy evaluation
            x_energy*-0.5
        } else {
            x_energy*-1.0
        }
    }

    pub fn evaluate_xc_energy(&mut self, iop: usize, mpi_operator: &Option<MPIOperator>) -> f64 {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut exc_spin:Vec<f64> = vec![];
        let dm = &mut self.density_matrix;
        let mo = &mut self.eigenvectors;
        let occ = &mut self.occupation;
        if let Some(grids) = &mut self.grids {
            exc_spin = self.mol.xc_data.xc_exc(grids, spin_channel,dm, mo, occ,iop, mpi_operator);
        }
        let exc:f64 = exc_spin.iter().sum();
        exc
    }

    //pub fn evaluate_xc_energy
       

    pub fn diagonalize_hamiltonian(&mut self, mpi_operator: &Option<MPIOperator>) {
        (self.eigenvectors, self.eigenvalues, self.mol.num_state) = diagonalize_hamiltonian_outside(&self, mpi_operator);
    }

    pub fn semi_diagonalize_hamiltonian(&mut self) {
        (self.semi_eigenvectors, self.semi_eigenvalues, self.semi_fock, self.mol.num_state) = semi_diagonalize_hamiltonian_outside(&self);
    }

    pub fn check_scf_convergence(&self, scftracerecode: &ScfTraceRecord) -> [bool;2] {
        if scftracerecode.num_iter<2 {
            return [false,false]
        }
        let spin_channel = self.mol.spin_channel;
        let num_basis = self.mol.num_basis as f64;
        let max_scf_cycle = self.mol.ctrl.max_scf_cycle;
        let scf_acc_rho   = self.mol.ctrl.scf_acc_rho;   
        let scf_acc_eev   = self.mol.ctrl.scf_acc_eev;  
        let scf_acc_etot  = self.mol.ctrl.scf_acc_etot; 
        let mut flag   = [true,true];

        let cur_index = 1;
        let pre_index = 0;
        let cur_energy = self.scf_energy;
        let pre_energy = scftracerecode.scf_energy;
        let diff_energy = cur_energy-pre_energy;
        let etot_converge = diff_energy.abs()<=scf_acc_etot;
        //scftracerecode.energy_change.push(diff_energy);

        let cur_energy = &self.eigenvalues;
        let pre_energy = &scftracerecode.eigenvalues;
        //let eev_converge = true
        let mut eev_err = 0.0;
        for i_spin in 0..spin_channel {
            //eev_err += cur_energy[i_spin].iter()
            //    .zip(pre_energy[i_spin].iter())
            //    .fold(0.0,|acc,(c,p)| acc + (c-p).powf(2.0));
            // rayon parallel version
            eev_err += cur_energy[i_spin].par_iter()
                .zip(pre_energy[i_spin].par_iter())
                .map(|(c,p)| (c-p).powf(2.0)).sum::<f64>();
        }
        eev_err = eev_err.sqrt();

        let mut dm_err = [0.0;2];
        //let cur_index = &scftracerecode.residual_density.len()-1;
        //let cur_residual_density = &scftracerecode.residual_density[cur_index];
        let cur_dm = &self.density_matrix;
        let pre_dm = &scftracerecode.density_matrix[1];

        for i_spin in 0..spin_channel {
            //dm_err[i_spin] = cur_dm[i_spin].data.iter()
            //    .zip(pre_dm[i_spin].data.iter())
            //    .fold(0.0,|acc,(c,p)| acc + (c-p).powf(2.0)).sqrt()/num_basis;
            // rayon parallel version
            dm_err[i_spin] = cur_dm[i_spin].data.par_iter()
                .zip(pre_dm[i_spin].data.par_iter())
                .map(|(c,p)| (c-p).powf(2.0)).sum::<f64>().sqrt()/num_basis;
            //dm_err[i_spin] = cur_residual_density[i_spin].data.par_iter()
            //    .map(|c| c.powf(2.0)).sum::<f64>().sqrt()/num_basis;
        }

        if spin_channel==1 {
            if self.mol.ctrl.print_level>0 {
                println!("SCF Change: DM {:10.5e}; eev {:10.5e} Ha; etot {:10.5e} Ha",dm_err[0],eev_err,diff_energy)
            };
            flag[0] = diff_energy.abs()<=scf_acc_etot &&
                      dm_err[0] <=scf_acc_rho &&
                      eev_err <= scf_acc_eev
        } else {
            if self.mol.ctrl.print_level>0 {
                println!("SCF Change: DM ({:10.5e},{:10.5e}); eev {:10.5e} Ha; etot {:10.5e} Ha",dm_err[0],dm_err[1],eev_err,diff_energy)
            };
            flag[0] = diff_energy.abs()<=scf_acc_etot &&
                      dm_err[0] <=scf_acc_rho &&
                      dm_err[1] <=scf_acc_rho &&
                      eev_err <= scf_acc_eev
        }

                  
        // Now check if max_scf_cycle is reached or not
        flag[1] = scftracerecode.num_iter >= max_scf_cycle;

        flag
    }

    pub fn formated_eigenvalues(&self,num_state_to_print:usize) {
        let mut cur_num_state_to_print = 0;
        let spin_channel = self.mol.spin_channel;
        match self.scftype {
            SCFType::RHF => {
                println!("{:>8}{:>14}{:>18}",String::from("State"),
                                        String::from("Occupation"),
                                        String::from("Eigenvalue"));
                if self.occupation[0].len() < num_state_to_print {
                    cur_num_state_to_print = self.eigenvalues[0].len();
                } else {
                    cur_num_state_to_print = num_state_to_print;
                }
                for i_state in (0..cur_num_state_to_print) {
                    println!("{:>8}{:>14.5}{:>18.6}",i_state,self.occupation[0][i_state],self.eigenvalues[0][i_state]);
                }
            },
            SCFType::UHF => {
                for i_spin in (0..spin_channel) {
                    if i_spin == 0 {
                        println!("Spin-up eigenvalues");
                        println!(" ");
                    } else {
                        println!(" ");
                        println!("Spin-down eigenvalues");
                        println!(" ");
                    }
                    println!("{:>8}{:>14}{:>18}",String::from("State"),
                                                String::from("Occupation"),
                                                String::from("Eigenvalue"));
                    if self.occupation[i_spin].len() < num_state_to_print {
                        cur_num_state_to_print = self.eigenvalues[i_spin].len();
                        println!("the number of eigenvalues in {} spin channel is {}", i_spin, self.occupation[i_spin].len());
                    } else {
                        cur_num_state_to_print = num_state_to_print;
                        for i_state in (0..cur_num_state_to_print) {
                            println!("{:>8}{:>14.5}{:>18.6}",i_state,self.occupation[i_spin][i_state],self.eigenvalues[i_spin][i_state]);
                        }
                    }
                }
            },
            SCFType::ROHF => {
                println!("{:>8}{:>14}{:>18}",String::from("State"),
                                        String::from("Occupation"),
                                        String::from("Eigenvalue"));
                // combine the occ numbers of alpha and beta channel
                let combined_occupation: Vec<f64> =  self.occupation[0]
                                                        .iter()
                                                        .zip(self.occupation[1].iter())
                                                        .map(|(a, b)| a + b)
                                                        .collect();
                if combined_occupation.len() < num_state_to_print {
                    cur_num_state_to_print = combined_occupation.len();
                } else {
                    cur_num_state_to_print = num_state_to_print;
                }
                for i_state in (0..cur_num_state_to_print) {
                    println!("{:>8}{:>14.5}{:>18.6}",i_state,combined_occupation[i_state],self.eigenvalues[0][i_state]);
                }
            }
        }
    }
    pub fn formated_eigenvectors(&self) {
        let spin_channel = self.mol.spin_channel;
        if spin_channel==1 || matches!(&self.scftype, SCFType::ROHF){
            self.eigenvectors[0].formated_output(5, "full");
        } else {
            (0..spin_channel).into_iter().for_each(|i_spin|{
                if i_spin == 0 {
                    println!("Spin-up eigenvalues");
                    println!(" ");
                } else {
                    println!(" ");
                    println!("Spin-down eigenvalues");
                    println!(" ");
                }
                self.eigenvectors[i_spin].formated_output(5, "full");
            });
        }
    }

    // relevant to RI-V
    pub fn generate_vj_with_ri_v(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {

        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        vj_upper_with_ri_v(&self.ri3fn, dm, spin_channel, scaling_factor)
    }

    pub fn generate_vj_with_ri_v_sync(&mut self, scaling_factor: f64, mpi_operator: &Option<MPIOperator>) -> Vec<MatrixUpper<f64>> {

        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        if self.mol.ctrl.use_ri_symm {
            vj_upper_with_rimatr_sync_mpi(&self.rimatr, dm, spin_channel, scaling_factor, mpi_operator)
        } else {
            //vj_upper_with_ri_v_sync(&self.ri3fn, dm, spin_channel, scaling_factor)
            if self.mol.ctrl.use_isdf && !self.mol.ctrl.isdf_k_only && !self.mol.ctrl.isdf_new{
                vj_upper_with_ri_v_sync(&self.ri3fn_isdf, dm, spin_channel, scaling_factor)
            }else{
                vj_upper_with_ri_v_sync(&self.ri3fn, dm, spin_channel, scaling_factor)
            }
        }
    }

    pub fn generate_vj_with_isdf(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {

        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        vj_upper_with_ri_v(&self.ri3fn_isdf, dm, spin_channel, scaling_factor)
    }

    pub fn generate_vk_with_ri_v(&self, scaling_factor: f64, use_dm_only: bool, mpi_operator: &Option<MPIOperator>) -> Vec<MatrixUpper<f64>> {
        //let num_basis = self.mol.num_basis;
        //let num_state = self.mol.num_state;
        //let num_auxbas = self.mol.num_auxbas;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;

        if self.mol.ctrl.use_ri_symm {
            if use_dm_only {
                let dm = &self.density_matrix;
                vk_upper_with_rimatr_use_dm_only_sync_mpi(&self.rimatr, dm, spin_channel, scaling_factor, mpi_operator)
            } else {
                let eigv = &self.eigenvectors;
                let occupation = &self.occupation;
                let num_elec = &self.mol.num_elec;
                //vk_upper_with_rimatr_sync(&mut self.rimatr, eigv, num_elec, occupation, spin_channel, scaling_factor)
                vk_upper_with_rimatr_sync_mpi(&self.rimatr, eigv, num_elec, occupation, spin_channel, scaling_factor,mpi_operator)
            }
        } else if self.mol.ctrl.isdf_new{
            self.generate_vk_with_isdf_new(scaling_factor)
        }else{
            if use_dm_only {
                let dm = &self.density_matrix;
                vk_upper_with_ri_v_use_dm_only_sync(&self.ri3fn, dm, spin_channel, scaling_factor)
            } else {
                let eigv = &self.eigenvectors;
                vk_upper_with_ri_v_sync(&self.ri3fn, eigv, &self.mol.num_elec, &self.occupation, 
                                        spin_channel, scaling_factor)
            }
        }

    

    }

    pub fn generate_vk_with_isdf(&self, scaling_factor: f64, use_dm_only: bool) -> Vec<MatrixUpper<f64>> {
        //let num_basis = self.mol.num_basis;
        //let num_state = self.mol.num_state;
        //let num_auxbas = self.mol.num_auxbas;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;

        if self.mol.ctrl.use_ri_symm {
            let dm = &self.density_matrix;
            vk_upper_with_rimatr_use_dm_only_sync(&self.rimatr, dm, spin_channel, scaling_factor)
        } else {
            if use_dm_only {
                //println!("use isdf to generate k");
                let dm = &self.density_matrix;
                //&dm[0].formated_output_e(5, "full");
                vk_upper_with_ri_v_use_dm_only_sync(&self.ri3fn_isdf, dm, spin_channel, scaling_factor)
            } else {
                let eigv = &self.eigenvectors;
                vk_upper_with_ri_v_sync(&self.ri3fn_isdf, eigv, &self.mol.num_elec, &self.occupation, 
                                        spin_channel, scaling_factor)
            }
        }
        

    }

    pub fn generate_vxc(&self, scaling_factor: f64) -> (f64, Vec<MatrixUpper<f64>>) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut vxc:Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty();spin_channel];
        let mut exc_spin:Vec<f64> = vec![];
        let mut exc_total:f64 = 0.0;
        let mut vxc_mf:Vec<MatrixFull<f64>> = vec![MatrixFull::empty();spin_channel];
        let dm = &self.density_matrix;
        let mo = &self.eigenvectors;
        let occ = &self.occupation;
        let print_level = self.mol.ctrl.print_level;
        if let Some(grids) = &self.grids {
            let dt0 = utilities::init_timing();
            let (exc,mut vxc_ao) = self.mol.xc_data.xc_exc_vxc(grids, spin_channel,dm, mo, occ, print_level);
            let dt1 = utilities::timing(&dt0, Some("Total vxc_ao time"));
            exc_spin = exc;
            if let Some(ao) = &grids.ao {
                // Evaluate the exchange-correlation energy
                //exc_total = izip!(grids.weights.iter(),exc.data.iter()).fold(0.0,|acc,(w,e)| {
                //    acc + w*e
                //});
                for i_spin in 0..spin_channel {
                    let vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();
                    *vxc_mf_s = MatrixFull::new([num_basis,num_basis],0.0f64);
                    let vxc_ao_s = vxc_ao.get(i_spin).unwrap();
                    _dgemm_full(ao, 'N', vxc_ao_s, 'T', vxc_mf_s, 1.0, 0.0);
                    //vxc_mf_s.lapack_dgemm(ao, vxc_ao_s, 'N', 'T', 1.0, 0.0);
                }
            }
            let dt2 = utilities::timing(&dt1, Some("From vxc_ao to vxc"));
        }


        let dt0 = utilities::init_timing();
        for i_spin in (0..spin_channel) {
            let mut vxc_s = vxc.get_mut(i_spin).unwrap();
            let mut vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();

            vxc_mf_s.self_add(&vxc_mf_s.transpose());
            vxc_mf_s.self_multiple(0.5);
            //vxc_mf_s.formated_output(10, "full");
            *vxc_s = vxc_mf_s.to_matrixupper();
        }

        utilities::timing(&dt0, Some("symmetrize vxc"));

        exc_total = exc_spin.iter().sum();


        if scaling_factor!=1.0f64 {
            exc_total *= scaling_factor;
            for i_spin in (0..spin_channel) {
                vxc[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        (exc_total, vxc)

    }

    pub fn generate_vxc_rayon_dm_only(&self, scaling_factor: f64) -> ([f64;2], f64, Vec<MatrixUpper<f64>>) {
        //In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        //In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        utilities::omp_set_num_threads_wrapper(1);

        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut vxc:Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty();spin_channel];
        let mut exc_spin:Vec<f64> = vec![0.0;spin_channel];
        let mut total_elec = [0.0,0.0];
        let mut exc_total:f64 = 0.0;
        let mut vxc_mf:Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis,num_basis],0.0);spin_channel];
        let dm = &self.density_matrix;
        let mo = &self.eigenvectors;
        let occ = &self.occupation;
        if let Some(grids) = &self.grids {
            let (sender, receiver) = channel();
            grids.parallel_balancing.par_iter().for_each_with(sender,|s,range_grids| {
                // change the return of xc_exc_vxc, directly return vxc_mat [num_basis, num_basis]
                // let (exc,vxc_ao,total_elec) = self.mol.xc_data.xc_exc_vxc_slots_dm_only(range_grids.clone(), grids, spin_channel,dm, mo, occ);
                //exc_spin = exc;
                let (exc, vxc_mf, total_elec) = self.mol.xc_data.xc_exc_vxc_slots_dm_only(range_grids.clone(), grids, spin_channel,dm, mo, occ);
                // let mut vxc_mf: Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis,num_basis],0.0f64);spin_channel];;
                // if let Some(ao) = &grids.ao {
                //     for i_spin in 0..spin_channel {

                //         let vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();
                //         let vxc_ao_s = vxc_ao.get(i_spin).unwrap();
                //         rest_tensors::matrix::matrix_blas_lapack::_dgemm(
                //             ao,(0..num_basis, range_grids.clone()),'N',
                //             vxc_ao_s,(0..num_basis,0..range_grids.len()),'T',
                //             vxc_mf_s, (0..num_basis,0..num_basis),
                //             1.0,0.0);

                //         //vxc_mf_s.to_matrixfullslicemut().lapack_dgemm(
                //         //    &ao.to_matrixfullslice(), 
                //         //    &vxc_ao_s.to_matrixfullslice(),
                //         //    'N', 'T', 1.0, 0.0);
                //     }
                // }
                s.send((vxc_mf,exc,total_elec)).unwrap()
            });
            receiver.into_iter().for_each(|(vxc_mf_local,exc_local,loc_total_elec)| {
                vxc_mf.iter_mut().zip(vxc_mf_local.iter()).for_each(|(to_matr,from_matr)| {
                    to_matr.self_add(from_matr);
                });
                exc_spin.iter_mut().zip(exc_local.iter()).for_each(|(to_exc,from_exc)| {
                    *to_exc += from_exc
                });
                total_elec.iter_mut().zip(loc_total_elec.iter()).for_each(|(to_elec, from_elec)| {
                    *to_elec += from_elec

                })
            })
        }

        //if self.mol.ctrl.print_level>1 {
        //    if spin_channel==1 {
        //        println!("total electron number: {:16.8}", total_elec[0]);
        //    } else {
        //        println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
        //        println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
        //    }
        //}


        for i_spin in (0..spin_channel) {
            let mut vxc_s = vxc.get_mut(i_spin).unwrap();
            let mut vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();

            vxc_mf_s.self_add(&vxc_mf_s.transpose());
            vxc_mf_s.self_multiple(0.5);
            *vxc_s = vxc_mf_s.to_matrixupper();
        }

        exc_total = exc_spin.iter().sum();


        if scaling_factor!=1.0f64 {
            exc_total *= scaling_factor;
            for i_spin in (0..spin_channel) {
                vxc[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

        (total_elec, exc_total, vxc)

    }

    pub fn generate_vxc_mpi_rayon_dm_only(&self, scaling_factor: f64, mpi_operator: &Option<MPIOperator>) -> ([f64;2], f64, Vec<MatrixUpper<f64>>) {
        let (total_elec, tot_exc, tot_xc) = if let Some(mpi_world) = mpi_operator {

            let world = &mpi_world.world;
            let my_rank = mpi_world.rank;

            let (mut total_elec, exc, mut vxc) = self.generate_vxc_rayon_dm_only(scaling_factor);

            let mut tot_exc = mpi_reduce(world, &[exc], 0, &SystemOperation::sum())[0];
            //mpi_broadcast(&world, &mut tot_exc, 0);

            let mut tot_elec = mpi_reduce(world, &total_elec, 0, &SystemOperation::sum());
            total_elec.iter_mut().zip(tot_elec.iter()).for_each(|(to, from)| *to = *from);
            //mpi_broadcast(&world, &mut total_elec, 0);

            //let mut tot_xc: Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty(), MatrixUpper::empty()];
            for i_spin in 0..self.mol.spin_channel {
                let mut result= mpi_reduce(world, vxc[i_spin].data_ref().unwrap(), 0, &SystemOperation::sum());
                let mut xc_spin = vxc.get_mut(i_spin).unwrap();
                //mpi_broadcast_vector(&world, &mut result, 0);
                //if mpi_world.rank==0 {
                    xc_spin.data = result;
                //}
            } 

            (total_elec, tot_exc, vxc)

        } else {
            self.generate_vxc_rayon_dm_only(scaling_factor)
        };

        if self.mol.ctrl.print_level>1 {
            if self.mol.spin_channel==1 {
                println!("total electron number: {:16.8}", total_elec[0]);
            } else {
                println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
                println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
            }
        }

        (total_elec, tot_exc, tot_xc)


    }

    pub fn generate_vxc_mpi_rayon(&self, scaling_factor: f64, mpi_operator: &Option<MPIOperator>) -> ([f64;2], f64, Vec<MatrixUpper<f64>>) {

        let (total_elec, tot_exc, tot_xc) = if let Some(mpi_world) = mpi_operator {

            let world = &mpi_world.world;
            let my_rank = mpi_world.rank;

            let (mut total_elec, exc, mut vxc) = self.generate_vxc_rayon(scaling_factor);

            let mut tot_exc = mpi_reduce(world, &[exc], 0, &SystemOperation::sum())[0];
            mpi_broadcast(&world, &mut tot_exc, 0);

            let mut tot_elec = mpi_reduce(world, &total_elec, 0, &SystemOperation::sum());
            total_elec.iter_mut().zip(tot_elec.iter()).for_each(|(to, from)| *to = *from);
            mpi_broadcast(&world, &mut total_elec, 0);

            //let mut tot_xc: Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty(), MatrixUpper::empty()];
            for i_spin in 0..self.mol.spin_channel {
                let mut result= mpi_reduce(world, vxc[i_spin].data_ref().unwrap(), 0, &SystemOperation::sum());

                let mut xc_spin = vxc.get_mut(i_spin).unwrap();
                //mpi_broadcast_vector(&world, &mut result, 0);
                if mpi_world.rank==0 {
                    xc_spin.data = result;
                } 
            } 

            (total_elec, tot_exc, vxc)

        } else {
            self.generate_vxc_rayon(scaling_factor)
        };

        if self.mol.ctrl.print_level>1 {
            if self.mol.spin_channel==1 {
                println!("total electron number: {:16.8}", total_elec[0]);
            } else {
                println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
                println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
            }
        }

        (total_elec, tot_exc, tot_xc)

    }


    pub fn generate_vxc_rayon(&self, scaling_factor: f64) -> ([f64;2], f64, Vec<MatrixUpper<f64>>) {
        //In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        //In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        utilities::omp_set_num_threads_wrapper(1);

        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut vxc:Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty();spin_channel];
        let mut exc_spin:Vec<f64> = vec![0.0;spin_channel];
        let mut total_elec = [0.0,0.0];
        let mut exc_total:f64 = 0.0;
        let mut vxc_mf:Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis,num_basis],0.0);spin_channel];
        let dm = &self.density_matrix;
        let mo = &self.eigenvectors;
        let occ = &self.occupation;
        if let Some(grids) = &self.grids {
            let (sender, receiver) = channel();
            grids.parallel_balancing.par_iter().for_each_with(sender,|s,range_grids| {
                // change the return value of xc_exc_vxc by vxc_mat [num_basis, num_basis]
                // let (exc,vxc_ao,total_elec) = self.mol.xc_data.xc_exc_vxc_slots(range_grids.clone(), grids, spin_channel,dm, mo, occ);
                //exc_spin = exc;
                let (exc, vxc_mf, total_elec) = self.mol.xc_data.xc_exc_vxc_slots(range_grids.clone(), grids, spin_channel, dm, mo, occ);
                // let mut vxc_mf: Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis,num_basis],0.0f64);spin_channel];;
                // if let Some(ao) = &grids.ao {
                //     for i_spin in 0..spin_channel {

                //         let vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();
                //         let vxc_ao_s = vxc_ao.get(i_spin).unwrap();
                //         rest_tensors::matrix::matrix_blas_lapack::_dgemm(
                //             ao,(0..num_basis, range_grids.clone()),'N',
                //             vxc_ao_s,(0..num_basis,0..range_grids.len()),'T',
                //             vxc_mf_s, (0..num_basis,0..num_basis),
                //             1.0,0.0);

                //         //vxc_mf_s.to_matrixfullslicemut().lapack_dgemm(
                //         //    &ao.to_matrixfullslice(), 
                //         //    &vxc_ao_s.to_matrixfullslice(),
                //         //    'N', 'T', 1.0, 0.0);
                //     }
                // }
                s.send((vxc_mf,exc,total_elec)).unwrap()
            });
            receiver.into_iter().for_each(|(vxc_mf_local,exc_local,loc_total_elec)| {
                vxc_mf.iter_mut().zip(vxc_mf_local.iter()).for_each(|(to_matr,from_matr)| {
                    to_matr.self_add(from_matr);
                });
                exc_spin.iter_mut().zip(exc_local.iter()).for_each(|(to_exc,from_exc)| {
                    *to_exc += from_exc
                });
                total_elec.iter_mut().zip(loc_total_elec.iter()).for_each(|(to_elec, from_elec)| {
                    *to_elec += from_elec

                })
            })
        }

        //if self.mol.ctrl.print_level>1 {
        //    if spin_channel==1 {
        //        println!("total electron number: {:16.8}", total_elec[0]);
        //    } else {
        //        println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
        //        println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
        //    }
        //}


        for i_spin in (0..spin_channel) {
            let mut vxc_s = vxc.get_mut(i_spin).unwrap();
            let mut vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();

            vxc_mf_s.self_add(&vxc_mf_s.transpose());
            vxc_mf_s.self_multiple(0.5);
            *vxc_s = vxc_mf_s.to_matrixupper();
        }

        exc_total = exc_spin.iter().sum();


        if scaling_factor!=1.0f64 {
            exc_total *= scaling_factor;
            for i_spin in (0..spin_channel) {
                vxc[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

        (total_elec,exc_total, vxc)

    }

    pub fn generate_ri3mo_rayon(&mut self, row_range: std::ops::Range<usize>, col_range: std::ops::Range<usize>) {
        if let SCFType::ROHF = self.scftype { //in ROHF case, generate semi-canonical eigenvectors for post SCF calculations.
            self.semi_diagonalize_hamiltonian(); 
        }

        let (mut ri3ao, mut basbas2baspair, mut baspar2basbas) =  if let Some((riao,basbas2baspair, baspar2basbas))=&mut self.rimatr {
            (riao,basbas2baspair, baspar2basbas)
        } else {
            panic!("rimatr should be initialized in the preparation of ri3mo");
        };
        let mut ri3mo: Vec<(RIFull<f64>,std::ops::Range<usize>, std::ops::Range<usize>)> = vec![];
        for i_spin in 0..self.mol.spin_channel {
            let eigenvector = match self.scftype {
                SCFType::ROHF => &self.semi_eigenvectors[i_spin],
                _ => &self.eigenvectors[i_spin],
            };
            ri3mo.push(
                ao2mo_rayon(
                    eigenvector, ri3ao, 
                    row_range.clone(), 
                    col_range.clone()
                ).unwrap()
            )
        }

        //if let Some(my_data)=&self.mol.mpi_data {
        //    //if my_data.rank == 0 {self.eigenvectors[0].formated_output(5, "full")};
        //    let (dd, col, row) = &ri3mo[0];
        //    let ff = dd.get_reducing_matrix(0).unwrap();
        //    ff.iter_columns_full().enumerate().for_each(|(i,x)| {
        //        println!("i: {}", i);
        //        println!("x: {:?}", &x);
        //    })
        //} else {
        //    //self.eigenvectors[0].formated_output(5, "full");
        //    let (dd, col, row) = &ri3mo[0];
        //    let ff = dd.get_reducing_matrix(0).unwrap();
        //    ff.iter_columns_full().enumerate().for_each(|(i,x)| {
        //        println!("i: {}", i);
        //        println!("x: {:?}", &x);
        //    })
        //};

        // deallocate the rimatr to save the memory
        self.rimatr = None;
        self.ri3mo = Some(ri3mo);


    }

}

/// Applies a projection operator to a given matrix. Specifically, it calculates the
/// product \( a^T \cdot b \cdot c \), where \( a \), \( b \), and \( c \) are input matrices.
///
/// # Arguments
/// * `a` - The first matrix (used as a transpose in the calculation).
/// * `b` - The second matrix.
/// * `c` - The third matrix.
/// * `size` - The size of the matrices.
///
/// # Returns
/// A new matrix that represents the result of \( a^T \cdot b \cdot c \).
///
/// # Example
/// ```
/// let a = MatrixFull::new([size, size], ...);
/// let b = MatrixFull::new([size, size], ...);
/// let c = MatrixFull::new([size, size], ...);
/// let result = apply_projection_operator(&a, &b, &c, size);
/// ```
pub fn apply_projection_operator(a: &MatrixFull<f64>, b: &MatrixFull<f64>, c: &MatrixFull<f64>) -> MatrixFull<f64> {
    // Temporary matrix to store intermediate result of a^T * b
    let mut temp: MatrixFull<f64> = MatrixFull::new([a.size[1], b.size[1]], 0.0);
    
    // First multiplication: a^T * b
    _dgemm_full(a, 'T', b,  'N', &mut temp, 1.0, 0.0);
    
    // Second multiplication: (a^T * b) * c
    let mut final_result: MatrixFull<f64> = MatrixFull::new([a.size[1], c.size[1]], 0.0);
    _dgemm_full(&temp, 'N', c, 'N', &mut final_result, 1.0, 0.0);
    
    final_result
}


/// return the occupation range and virtual range for the preparation of ri3mo;
pub fn determine_ri3mo_size_for_pt2_and_rpa(scf_data: &SCF) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let num_state = scf_data.mol.num_state;
    let mut homo = 0_usize;
    let mut lumo = num_state;
    let start_mo = scf_data.mol.start_mo;

    for i_spin in 0..scf_data.mol.spin_channel {

        let i_homo = scf_data.homo.get(i_spin).unwrap().clone();
        let i_lumo = scf_data.lumo.get(i_spin).unwrap().clone();

        homo = homo.max(i_homo);
        lumo = lumo.min(i_lumo);
    }

    (start_mo..homo+1, lumo..num_state)
}


// vj, vk without dependency on SCF struct
//
pub fn vj_upper_with_ri_v(
                    ri3fn: &Option<RIFull<f64>>,
                    dm: &Vec<MatrixFull<f64>>, 
                    spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    if let Some(ri3fn) = ri3fn {
        let num_basis = ri3fn.size[0];
        let num_auxbas = ri3fn.size[2];
        let npair = num_basis*(num_basis+1)/2;
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixUpper::new(npair,0.0f64);
            ri3fn.iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each(|(i,m)| {
                //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu}
                let tmp_mu =
                    m.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                        .fold(0.0_f64,|acc, (m,d)| {
                            acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                        });
                // filter out the upper part of  M_{ij}^{\mu}
                let m_ij_upper = m.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                    .map(|(i,v)| v );

                // fill vj[i_spin] with the contribution from the given {\mu}:
                //
                // M_{ij}^{\mu}*(\sum_{kl}D_{kl}*M_{kl}^{\mu})
                //
                vj_spin.data.iter_mut().zip(m_ij_upper)
                    .for_each(|value| *value.0 += *value.1*tmp_mu); 
            });
        }
    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vj[i_spin].data.iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };
    vj
}
pub fn vj_upper_with_ri_v_sync(
                ri3fn: &Option<RIFull<f64>>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    //// In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    //// In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    //let default_omp_num_threads = unsafe {openblas_get_num_threads()};
    //unsafe{openblas_set_num_threads(1)};
    

    if let Some(ri3fn) = ri3fn {
    let num_basis = ri3fn.size[0];
    let num_auxbas = ri3fn.size[2];
    let npair = num_basis*(num_basis+1)/2;
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixUpper::new(npair,0.0f64);

            let (sender, receiver) = channel();
            ri3fn.par_iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each_with(sender, |s, (i,m)| {
                //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu} for each \mu -> tmp_mu
                let tmp_mu =
                    m.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                        .fold(0.0_f64,|acc, (m,d)| {
                            acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                        });
                // filter out the upper part (ij pair) of M_{ij}^{\mu} for each \mu -> m_ij_upper
                let m_ij_upper = m.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                    .map(|(i,v)| v.clone() ).collect_vec();
                s.send((m_ij_upper,tmp_mu)).unwrap();
            });
            // fill vj[i_spin] with the contribution from the given {\mu}:
            //
            // M_{ij}^{\mu}*(\sum_{kl}D_{kl}*M_{kl}^{\mu})
            //
            receiver.iter().for_each(|(m_ij_upper, tmp_mu)| {
                vj_spin.data.iter_mut().zip(m_ij_upper.iter())
                    .for_each(|value| *value.0 += *value.1*tmp_mu); 
            });


            //vj_spin.data.par_iter_mut().zip(m_ij_upper.par_iter())
            //    .for_each(|value| *value.0 += *value.1*tmp_mu); 
        }
    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    //// reuse the default omp_num_threads setting
    //unsafe{openblas_set_num_threads(default_omp_num_threads)};

    vj
}

pub fn vj_upper_with_rimatr_sync_mpi(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64,
                mpi_operator: &Option<MPIOperator>)  -> Vec<MatrixUpper<f64>> {
    if let Some(mpi_op) = &mpi_operator {
        let mut vj_vec = vj_upper_with_rimatr_sync(ri3fn, dm, spin_channel, scaling_factor);
        for i_spin in 0..spin_channel {
            let vj = &mut vj_vec[i_spin];
            let mut tot_vj = mpi_reduce(&mpi_op.world, vj.data_ref().unwrap(), 0, &SystemOperation::sum());
            mpi_broadcast_vector(&mpi_op.world, &mut tot_vj, 0);
            //if mpi_op.rank == 0 {
                vj.data = tot_vj;
            //}
        }
        vj_vec
    } else {
        vj_upper_with_rimatr_sync(ri3fn, dm, spin_channel, scaling_factor)
    }
}

pub fn vj_upper_with_rimatr_sync(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    vj_upper_with_rimatr_sync_v02(ri3fn,dm,spin_channel,scaling_factor)
}

pub fn vj_upper_with_rimatr_sync_v01(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    //// In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    //// In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    //let default_omp_num_threads = unsafe {openblas_get_num_threads()};
    //unsafe{openblas_set_num_threads(1)};
    
    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = basbas2baspar.size[0];
        let num_baspar = ri3fn.size[0];
        let num_auxbas = ri3fn.size[1];
        //let npair = num_basis*(num_basis+1)/2;
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixUpper::new(num_baspar,0.0f64);
            let dm_s = &dm[i_spin];

            let (sender, receiver) = channel();
            ri3fn.par_iter_columns_full().enumerate().for_each_with(sender, |s, (i,m)| {
                //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu} for each \mu -> tmp_mu
                let riupper = MatrixUpperSlice::from_vec(m);
                let mut tmp_mu =
                    m.iter().zip(dm_s.iter_matrixupper().unwrap()).fold(0.0_f64, |acc,(m,d)| {
                        acc + *m * (*d)
                    });
                //let diagonal_term = riupper.get_diagonal_terms().unwrap()
                //    .iter().zip(dm[i_spin].iter_diagonal().unwrap()).fold(0.0f64, |acc, (v1,v2)| {
                //        acc + *v1*v2
                //});
                let diagonal_term = riupper.iter_diagonal()
                    .zip(dm[i_spin].iter_diagonal().unwrap()).fold(0.0f64, |acc, (v1,v2)| {
                        acc + *v1*v2
                });

                tmp_mu = 2.0_f64*tmp_mu - diagonal_term;

                let m_ij_upper = m.iter().map(|v| *v*tmp_mu).collect_vec();
                s.send(m_ij_upper).unwrap();
            });
            // fill vj[i_spin] with the contribution from the given {\mu}:
            //
            // M_{ij}^{\mu}*(\sum_{kl}D_{kl}*M_{kl}^{\mu})
            //
            receiver.iter().for_each(|(m_ij_upper)| {
                vj_spin.data.iter_mut().zip(m_ij_upper.iter())
                    .for_each(|value| *value.0 += *value.1); 
            });


            //vj_spin.data.par_iter_mut().zip(m_ij_upper.par_iter())
            //    .for_each(|value| *value.0 += *value.1*tmp_mu); 
        }
    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    //// reuse the default omp_num_threads setting
    //unsafe{openblas_set_num_threads(default_omp_num_threads)};
    //vj[0].formated_output(5, "full");

    vj
}

pub fn vj_upper_with_rimatr_sync_v02(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = basbas2baspar.size[0];
        let num_baspar = ri3fn.size[0];
        let num_auxbas = ri3fn.size[1];
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixUpper::new(num_baspar,0.0f64);
            //let mut dm_s = dm[i_spin].clone();
            //dm_s.iter_diagonal_mut().unwrap().for_each(|x| *x = *x/2.0);

            let mut dm_s_upper = MatrixUpper::from_vec(num_baspar,dm[i_spin].iter_matrixupper().unwrap().map(|x| *x).collect_vec()).unwrap();
            dm_s_upper.iter_diagonal_mut().for_each(|x| {*x = *x/2.0});

            let mut tmp_v = vec![0.0;num_auxbas];

            _dgemv(ri3fn, &dm_s_upper.data, &mut tmp_v, 'T', 2.0, 0.0, 1, 1);

            _dgemv(ri3fn, &tmp_v, &mut vj_spin.data, 'N',1.0,0.0,1,1);

        }
    }

    vj
}


// Just for test, no need to use vj_full because it's always symmetric
pub fn vj_full_with_ri_v(
                    ri3fn: &Option<RIFull<f64>>,
                    dm: &Vec<MatrixFull<f64>>, 
                    spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixFull<f64>> {
    
    let mut vj: Vec<MatrixFull<f64>> = vec![MatrixFull::new([1,1],0.0f64),MatrixFull::new([1,1],0.0f64)];
    if let Some(ri3fn) = ri3fn {
        let num_basis = ri3fn.size[0];
        let num_auxbas = ri3fn.size[2];
        //let npair = num_basis*(num_basis+1)/2;
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixFull::new([num_basis, num_basis],0.0f64);
            ri3fn.iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each(|(i,m)| {
                //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu}
                let tmp_mu =
                    m.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                        .fold(0.0_f64,|acc, (m,d)| {
                            acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                        });

                // fill vj[i_spin] with the contribution from the given {\mu}:
                //
                // M_{ij}^{\mu} * (\sum_{kl}D_{kl}*M_{kl}^{\mu})
                //
                vj_spin.data.iter_mut().zip(m)
                    .for_each(|value| *value.0 += *value.1*tmp_mu); 
            });
        }
    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vj[i_spin].data.iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };
    vj
}
pub fn vk_full_fromdm_with_ri_v(
                    ri3fn: &Option<RIFull<f64>>,
                    dm: &Vec<MatrixFull<f64>>, 
                    spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixFull<f64>> {
    
    let mut vk: Vec<MatrixFull<f64>> = vec![MatrixFull::new([1,1],0.0f64),MatrixFull::new([1,1],0.0f64)];
    if let Some(ri3fn) = ri3fn {
        let num_basis = ri3fn.size[0];
        let num_auxbas = ri3fn.size[2];
        //let npair = num_basis*(num_basis+1)/2;
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vk_spin = &mut vk[i_spin];
            *vk_spin = MatrixFull::new([num_basis, num_basis],0.0f64);
            ri3fn.iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each(|(i,m)| {
                //prepare \sum_{l}D_{jl}*M_{kl}^{\mu}
                let mut tmp_mu = MatrixFull::from_vec([num_basis, num_basis], m.to_vec()).unwrap();
                let mut dm_m = MatrixFull::new([num_basis, num_basis], 0.0f64);
                dm_m.lapack_dgemm(&mut dm[i_spin].clone(), &mut tmp_mu, 'N', 'T', 1.0, 0.0);

                // fill vk[i_spin] with the contribution from the given {\mu}:
                //
                // \sum_j M_{ij}^{\mu} * (\sum_{l}D_{jl}*M_{kl}^{\mu})
                //
                vk_spin.lapack_dgemm(&mut tmp_mu.clone(), &mut dm_m, 'N', 'N', 1.0, 1.0);
            });
        }
    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };
    vk
}

pub fn vk_upper_with_ri_v_use_dm_only_sync(
                ri3fn: &Option<RIFull<f64>>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];


    if let Some(ri3fn) = ri3fn {
        let num_basis = dm[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let dm_s = &dm[i_spin];
            //dm_s.formated_output(5, "upper");
            let (sender, receiver) = channel();
            ri3fn.par_iter_auxbas(0..num_auxbas).unwrap().for_each_with(sender,|s, m| {
                let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut reduced_ri3fn = MatrixFullSlice {
                    size:  &[num_basis,num_basis],
                    indicing: &[1,num_basis],
                    data: m,
                };
                //_dgemm(&reduced_ri3fn, (0..num_basis,0..num_basis), 'N', 
                //       dm_s, (0..num_basis,0..num_basis), 'N', 
                //       &mut tmp_mat, (0..num_basis,0..num_basis), 1.0, 0.0);
                //let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                //_dgemm(&tmp_mat, (0..num_basis,0..num_basis), 'N', 
                //       &reduced_ri3fn, (0..num_basis,0..num_basis), 'T', 
                //       &mut vk_sm, (0..num_basis,0..num_basis), 1.0, 0.0);
                //tmp_mat = ri3fn \cdot dm
                _dsymm(&reduced_ri3fn, dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                //vk_sm = ri3fn \cdot dm \cdot ri3fn
                _dsymm(&reduced_ri3fn, &tmp_mat, &mut vk_sm, 'R', 'U', 1.0, 0.0);

                s.send(vk_sm.to_matrixupper()).unwrap();
            });

            receiver.into_iter().for_each(|vk_mu_upper| {
                vk_s.data.par_iter_mut()
                    .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                    *value.0 += *value.1
                })
            });
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}



pub fn vk_upper_with_rimatr_use_dm_only_sync(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {

    vk_upper_with_rimatr_use_dm_only_sync_v02(ri3fn, dm, spin_channel, scaling_factor)
}

pub fn vk_upper_with_rimatr_use_dm_only_sync_v01(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = dm[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let dm_s = &dm[i_spin];
            let (sender, receiver) = channel();
            ri3fn.par_iter_columns_full().for_each_with(sender,|s, m| {
                let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);

                reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

                _dsymm(&reduced_ri3fn, dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                _dsymm(&reduced_ri3fn, &tmp_mat, &mut vk_sm, 'R', 'U', 1.0, 0.0);

                s.send(vk_sm.to_matrixupper()).unwrap();
            });

            receiver.into_iter().for_each(|vk_mu_upper| {
                vk_s.data.iter_mut()
                    .zip(vk_mu_upper.data.iter()).for_each(|value| {
                    *value.0 += *value.1
                })
            });
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

pub fn vk_upper_with_rimatr_use_dm_only_sync_v02(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    //utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = dm[0].size()[0];
        let num_baspair = ri3fn.size()[0];
        let num_auxbas = ri3fn.size()[1];
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            //let dm_s = &dm[i_spin];
            utilities::omp_set_num_threads_wrapper(default_omp_num_threads);
            let dm_s = _power_rayon_for_symmetric_matrix(&dm[i_spin], 0.5, SQRT_THRESHOLD).unwrap();
            utilities::omp_set_num_threads_wrapper(1);
            let batch_num_auxbas = utilities::balancing(num_auxbas, rayon::current_num_threads());
            let (sender, receiver) = channel();
            batch_num_auxbas.par_iter().for_each_with(sender, |s,loc_auxbas| {
                let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                ri3fn.iter_columns(loc_auxbas.clone()).for_each(|m| {
                    reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});
                    //_dsymm(&reduced_ri3fn, dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                    //_dsymm(&reduced_ri3fn, &tmp_mat, &mut vk_sm, 'R', 'U', 1.0, 1.0);
                    _dsymm(&reduced_ri3fn, &dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                    _dsyrk(&tmp_mat, &mut vk_sm, 'U', 'N', 1.0, 1.0)
                });
                s.send(vk_sm.to_matrixupper()).unwrap();
            });

            receiver.into_iter().for_each(|vk_mu_upper| {
                vk_s.data.par_iter_mut()
                    .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                    *value.0 += *value.1
                })
            });
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

pub fn vk_upper_with_rimatr_use_dm_only_sync_mpi(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64, mpi_operator: &Option<MPIOperator>)  -> Vec<MatrixUpper<f64>> {
    if let Some(mpi_op) = &mpi_operator {
        let mut vk_vec = vk_upper_with_rimatr_use_dm_only_sync_v02(ri3fn, dm, spin_channel, scaling_factor);
        for i_spin in 0..spin_channel {
            let vk = &mut vk_vec[i_spin];
            let mut tot_vk = mpi_reduce(&mpi_op.world, vk.data_ref().unwrap(), 0, &SystemOperation::sum());
            mpi_broadcast(&mpi_op.world, &mut tot_vk, 0);
            //if mpi_op.rank == 0 {
                vk.data = tot_vk;
            //}
        };
        vk_vec
    } else {
        vk_upper_with_rimatr_use_dm_only_sync_v02(ri3fn, dm, spin_channel, scaling_factor)
    }
}

pub fn vk_upper_with_rimatr_sync_mpi(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                spin_channel: usize, scaling_factor: f64,
                mpi_operator: &Option<MPIOperator>)  -> Vec<MatrixUpper<f64>> {
    if let Some(mpi_op) = &mpi_operator {
        let mut vk_vec = vk_upper_with_rimatr_sync_v03(ri3fn,eigv,num_elec,occupation,spin_channel,scaling_factor);
        for i_spin in 0..spin_channel {
            let vk = &mut vk_vec[i_spin];
            let mut tot_vk = mpi_reduce(&mpi_op.world, vk.data_ref().unwrap(), 0, &SystemOperation::sum());
            mpi_broadcast(&mpi_op.world, &mut tot_vk, 0);
            //if mpi_op.rank == 0 {
                vk.data = tot_vk;
            //}
        };
        vk_vec
    } else {
        vk_upper_with_rimatr_sync_v03(ri3fn,eigv,num_elec,occupation,spin_channel,scaling_factor)
    }
}
pub fn vk_upper_with_rimatr_sync(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    vk_upper_with_rimatr_sync_v03(ri3fn,eigv,num_elec,occupation,spin_channel,scaling_factor)
}

pub fn vk_upper_with_rimatr_sync_v01(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = eigv[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let eigv_s = &eigv[i_spin];
            let homo_s = occupation[i_spin].iter().enumerate()
                .fold(0_usize,|x, (ob, occ)| {if *occ>1.0e-4 {ob} else {x}});
            let nw = homo_s + 1;
            //let nw = num_elec[i_spin+1].ceil() as usize;
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let occ_s = &occupation[i_spin][0..nw];
                tmp_mat.data.par_chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.par_iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value.sqrt()});
                });
                let reduced_eigv_s = tmp_mat;

                //let dm_s = &dm[i_spin];

                let (sender, receiver) = channel();
                ri3fn.par_iter_columns_full().for_each_with(sender,|s, m| {
                    //let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);

                    reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

                    let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                    //tmp_mc = ri3fn \cdot eigv \cdot occ.sqrt()
                    _dsymm(&reduced_ri3fn, &reduced_eigv_s, &mut tmp_mc, 'L', 'U', 1.0, 0.0);

                    let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    _dsyrk(&tmp_mc, &mut vk_sm, 'U', 'N', 1.0, 0.0);

                    s.send(vk_sm.to_matrixupper()).unwrap();
                });

                receiver.into_iter().for_each(|vk_mu_upper| {
                    vk_s.data.par_iter_mut()
                        .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                        *value.0 += *value.1
                    })
                });
            }
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

/// a new vk version with the parallelization giving to openmk.
pub fn vk_upper_with_rimatr_sync_v02(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {

    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = eigv[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
            //*vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let eigv_s = &eigv[i_spin];
            // now locate the highest obital that has electron with occupation largger than 1.0e-4
            let homo_s = occupation[i_spin].iter().enumerate()
                .fold(0_usize,|x, (ob, occ)| {if *occ>1.0e-4 {ob} else {x}});
            let nw = homo_s + 1;
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let occ_s = &occupation[i_spin][0..nw];
                tmp_mat.data.chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value.sqrt()});
                });
                let reduced_eigv_s = tmp_mat;

                //let dm_s = &dm[i_spin];

                //let (sender, receiver) = channel();

                let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);

                ri3fn.iter_columns_full().for_each(|m| {
                    //let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

                    //tmp_mc = ri3fn \cdot eigv
                    _dsymm(&reduced_ri3fn, &reduced_eigv_s, &mut tmp_mc, 'L', 'U', 1.0, 0.0);

                    _dsyrk(&tmp_mc, &mut vk_sm, 'U', 'N', 1.0, 1.0);

                    //s.send(vk_sm.to_matrixupper()).unwrap();
                });
                *vk_s = vk_sm.to_matrixupper();
            } else {
              *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            }
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    //// reuse the default omp_num_threads setting
    //utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

pub fn vk_upper_with_rimatr_sync_v03(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    //let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty(),MatrixUpper::empty()];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = eigv[0].size()[0];
        let num_baspair = ri3fn.size()[0];
        let num_auxbas = ri3fn.size()[1];
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let eigv_s = if !eigv[i_spin].data.is_empty() {
                &eigv[i_spin]
            } else { // use the eigv[0] again for ROHF case.
                &eigv[0]
            };
            // now locate the highest obital that has electron with occupation largger than 1.0e-4
            let homo_s = occupation[i_spin].iter().enumerate()
                .fold(0_usize,|x, (ob, occ)| {if *occ>1.0e-4 {ob} else {x}});
            let elec_spin = num_elec[i_spin+1].ceil() as usize;
            let nw = if elec_spin == 0 {0} else {homo_s + 1} ;
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let occ_s = &occupation[i_spin][0..nw];
                tmp_mat.data.par_chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.par_iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value.sqrt()});
                });
                let reduced_eigv_s = tmp_mat;

                //let dm_s = &dm[i_spin];

                let batch_num_auxbas = utilities::balancing(num_auxbas, rayon::current_num_threads());
                let (sender, receiver) = channel();
                batch_num_auxbas.par_iter().for_each_with(sender, |s, loc_auxbas| {
                    let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                    ri3fn.iter_columns(loc_auxbas.clone()).for_each(|m| {
                        reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});
                        _dsymm(&reduced_ri3fn, &reduced_eigv_s, &mut tmp_mc, 'L', 'U', 1.0, 0.0);
                        _dsyrk(&tmp_mc, &mut vk_sm, 'U', 'N', 1.0, 1.0);
                    });
                    s.send(vk_sm.to_matrixupper()).unwrap();
                });

                receiver.into_iter().for_each(|vk_mu_upper| {
                    vk_s.data.par_iter_mut()
                        .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                        *value.0 += *value.1
                    })
                });
            }
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

//==========================need to be checked=============================
pub fn vk_upper_with_ri_v_sync(
                ri3fn: &Option<RIFull<f64>>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    
    if let Some(ri3fn) = ri3fn {
        let num_basis = eigv[0].size[0];
        let num_state = eigv[0].size[1];
        let num_auxbas = ri3fn.size[2];
        let npair = num_basis*(num_basis+1)/2;
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(npair,0.0_f64);
            let eigv_s = &eigv[i_spin];
            let nw = num_elec[i_spin+1].ceil() as usize;
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let reduced_eigv_s = tmp_mat;
                let occ_s = &occupation[i_spin][0..nw];
                //let mut tmp_b = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let (sender, receiver) = channel();
                ri3fn.par_iter_auxbas(0..num_auxbas).unwrap().for_each_with(sender, |s, m| {

                    let mut reduced_ri3fn = MatrixFullSlice {
                        size:  &[num_basis,num_basis], 
                        indicing: &[1,num_basis],
                        data: m,
                    };
                    //tmp_mat: copy of related eigenvalue; reduced_ri3fn: certain part of ri3fn
                    let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                    //tmp_mc = ri3fn \cdot eigv
                    tmp_mc.to_matrixfullslicemut().lapack_dgemm(&reduced_ri3fn, &reduced_eigv_s.to_matrixfullslice(), 'N', 'N', 1.0, 0.0);
                    //tmp_mat = tmp_mc (ri3fn \cdot eigv)
                    let mut tmp_mat = tmp_mc.clone();
                    //tmp_mat = tmp_mc * occ
                    tmp_mat.data.chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value});
                    });

                    let mut vk_mu = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    // vk_mu = tmp_mat \cdot tmp_mc.T  ((ri3fn \cdot eigv * occ) \cdot (ri3fn \cdot eigv)^T)
                    vk_mu.lapack_dgemm(&mut tmp_mat, &mut tmp_mc, 'N', 'T', 1.0, 0.0);

                    // filter out the upper part of vk_mu
                    let mut tmp_mat = MatrixUpper::from_vec(npair, vk_mu.data.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                        .map(|(i,v)| v.clone() ).collect_vec()).unwrap();

                    s.send(tmp_mat).unwrap()
                });
                receiver.into_iter().for_each(|vk_mu_upper| {
                    vk_s.data.par_iter_mut()
                        .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                        *value.0 += *value.1
                    })
                });
            }
        }
        //// for each spin channel
        //vk.iter_mut().zip(eigv.iter()).for_each(|(vk_s,eigv_s)| {
        //});

    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);


    vk
}
//=============================================================================

#[derive(Clone)]
pub struct ScfTraceRecord {
    pub num_iter: usize,
    pub mixer: String,
    // the maximum number of stored residual densities
    pub num_max_records: usize,
    pub mix_param: f64,
    pub start_diis_cycle: usize,
    //pub scf_energy : Vec<f64>,
    //pub density_matrix: Vec<[MatrixFull<f64>;2]>,
    //pub eigenvectors: Vec<[MatrixFull<f64>;2]>,
    //pub eigenvalues: Vec<[Vec<f64>;2]>,
    pub scf_energy : f64,
    pub energy_records: Vec<f64>,
    pub prev_hamiltonian: Vec<[MatrixUpper<f64>;2]>,
    pub eigenvectors: [MatrixFull<f64>;2],
    pub eigenvalues: [Vec<f64>;2],
    pub density_matrix: [Vec<MatrixFull<f64>>;2],
    pub target_vector: Vec<[MatrixFull<f64>;2]>,
    pub error_vector: Vec<Vec<f64>>,
}

impl ScfTraceRecord {
    pub fn new(num_max_records: usize, mix_param: f64, mixer: String,start_diis_cycle: usize) -> ScfTraceRecord {
        if num_max_records==0 {
            println!("Error: num_max_records cannot be 0");
        }
        ScfTraceRecord {
            num_iter: 0,
            mixer,
            mix_param,
            num_max_records,
            start_diis_cycle,
            scf_energy : 0.0,
            energy_records: vec![],
            prev_hamiltonian: vec![[MatrixUpper::empty(),MatrixUpper::empty()]],
            eigenvectors: [MatrixFull::new([1,1],0.0),
                              MatrixFull::new([1,1],0.0)],
            eigenvalues: [Vec::<f64>::new(),Vec::<f64>::new()],
            density_matrix: [vec![MatrixFull::new([1,1],0.0),
                              MatrixFull::new([1,1],0.0)],
                             vec![MatrixFull::new([1,1],0.0),
                              MatrixFull::new([1,1],0.0)]],
            target_vector: Vec::<[MatrixFull<f64>;2]>::new(),
            error_vector: Vec::<Vec::<f64>>::new(),
        }
    }
    pub fn initialize(scf: &SCF) -> ScfTraceRecord {
        /// 
        /// Initialize the scf records which should be involked after the initial guess 
        /// 
        let mut tmp_records = ScfTraceRecord::new(
            scf.mol.ctrl.num_max_diis, 
            scf.mol.ctrl.mix_param.clone(), 
            scf.mol.ctrl.mixer.clone(),
            scf.mol.ctrl.start_diis_cycle.clone()
        );
        tmp_records.scf_energy=scf.scf_energy;
        tmp_records.eigenvectors=scf.eigenvectors.clone();
        tmp_records.eigenvalues=scf.eigenvalues.clone();
        tmp_records.density_matrix=[scf.density_matrix.clone(),scf.density_matrix.clone()];
        if tmp_records.mixer.eq(&"ddiis") {
            tmp_records.target_vector.push([scf.density_matrix[0].clone(),scf.density_matrix[1].clone()]);
        }
        tmp_records
    }
    /// This subroutine updates:  
    ///     scf_energy:         from previous to the current value  
    ///     scf_eigenvalues:    from previous to the current value  
    ///     scf_eigenvectors:   from previous to the current value  
    ///     scf_density_matrix: [pre, cur]  
    ///     num_iter  
    /// This subroutine should be called after [`scf.check_scf_convergence`] and before [`self.prepare_next_input`]"
    pub fn update(&mut self, scf: &SCF) {

        let spin_channel = scf.mol.spin_channel;

        // now store the scf energy, eigenvectors and eigenvalues of the last two steps
        //let tmp_data =  self.scf_energy[1].clone();
        self.scf_energy=scf.scf_energy;
        //let tmp_data =  self.eigenvectors[1].clone();
        self.eigenvectors=scf.eigenvectors.clone();
        //let tmp_data =  self.eigenvalues[1].clone();
        self.eigenvalues=scf.eigenvalues.clone();
        let tmp_data =  self.density_matrix[1].clone();
        self.density_matrix=[tmp_data,scf.density_matrix.clone()];

        self.num_iter +=1;
    }
    /// This subroutine prepares the fock matrix for the next step according different mixing algorithm  
    ///
    /// self.mixer =  
    /// * "direct": the output density matrix in the current step `n0[out]` will be used directly 
    ///             to generate the the input fock matrix of the next step
    /// * "linear": the density matrix used in the next step `n1[in]` is a mix between
    ///             the input density matrix in the current step `n0[in]` and `n0[out]`  
    ///             <span style="text-align:right">`n1[in] = alpha*n0[out] + (1-alpha)*n0[in]`</span>  
    ///             <span style="text-align:right">`       = n0[in] + alpha * Rn0            ` </span>  
    ///             where alpha the mixing parameter obtained from self.mix_param
    ///              and `Rn0 = n0[out]-n0[in]` is the density matrix change in the current step.
    ///             `n1[in]` is then be used to generate the input fock matrix of the next step
    /// * "diis":   the input fock matrix of the next step `f1[in] = sum_{i} c_i*f_i[in]`,
    ///            where `f_i[in]` is the input fock matrix of the ith step and 
    ///            c_i is obtained by the diis altogirhm against the error vector
    ///            of the commutator `(f_i[out]*d_i[out]*s-s*d_i[out]*f_i[out])`, where  
    ///            - `f_i[out]` is the ith output fock matrix,   
    ///            - `d_i[out]` is the ith output density matrix,  
    ///            - `s` is the overlap matrix  
    /// * **Ref**: P. Pulay, Improved SCF Convergence Acceleration, JCC, 1982, 3:556-560.
    ///
    pub fn prepare_next_input(&mut self, scf: &mut SCF, mpi_operator: &Option<MPIOperator>) {
        let spin_channel = scf.mol.spin_channel;
        let start_pulay = self.start_diis_cycle;
        //if self.residual_density.len()>=2 {
        let alpha = self.mix_param;
        let beta = 1.0-self.mix_param;
        if self.mixer.eq(&"direct") {
            scf.generate_hf_hamiltonian(mpi_operator);
        }
        else if self.mixer.eq(&"linear") 
            || (self.mixer.eq(&"ddiis") && self.num_iter<start_pulay) 
            || (self.mixer.eq(&"diis") && self.num_iter<start_pulay) 
        {
            let mut alpha = self.mix_param;
            let mut beta = 1.0-alpha;
            // n1[in] = a*n0[out] + (1-a)*n0[in] = n0[out]-(1-a)*Rn0 = n0[in] + a*Rn0
            // Rn0 = n0[out]-n0[in]; the residual density in the current iteration
            // n1[in] is the input density for the next iteration
            for i_spin in (0..spin_channel) {
                let residual_dm = self.density_matrix[1][i_spin].sub(&self.density_matrix[0][i_spin]).unwrap();
                scf.density_matrix[i_spin] = self.density_matrix[0][i_spin]
                    .scaled_add(&residual_dm, alpha)
                    .unwrap();
            }
            scf.generate_hf_hamiltonian(mpi_operator);
        } else if self.mixer.eq(&"diis") && self.num_iter>=start_pulay {
            // 
            // Reference: P. Pulay, Improved SCF Convergence Acceleration, JCC, 1982, 3:556-560.
            // 
            let start_dim = 0usize;
            let mut start_check_oscillation = scf.mol.ctrl.start_check_oscillation;
            //
            // prepare the fock matrix according to the output density matrix of the previous step
            //
            let dt1 = time::Local::now();

            scf.generate_hf_hamiltonian(mpi_operator);


            // update the energy records and check the oscillation
            self.energy_records.push(scf.scf_energy);
            let num_step = self.energy_records.len();
            let oscillation_flag = if num_step >=2 {
                let change_1 = self.energy_records[num_step-1] - self.energy_records[num_step-2];
                num_step > start_check_oscillation && change_1 > 0.0
                //false
            }else {
                false
            };

            let dt2 = time::Local::now();


            // check if the storage of fock matrix reaches the maximum setting
            if self.target_vector.len() == self.num_max_records {
                self.target_vector.remove(0);
                self.error_vector.remove(0);
            };

            //
            // prepare and store the fock matrix in full formate and the error vector in the current step
            //
            //for i_spin in (0..spin_channel) {
            //    //self.target_vector.push([scf.hamiltonian[i_spin].clone(), scf.hamiltonian[i_spin].clone()]);
            //    scf.hamiltonian[i_spin].formated_output(5, "upper");
            //}
            let (cur_error_vec, cur_target) = generate_diis_error_vector(&scf.hamiltonian, &scf.ovlp, &mut self.density_matrix, spin_channel);
            self.error_vector.push(cur_error_vec);
            self.target_vector.push(cur_target);


            // solve the DIIS against the error vector
            if let Some(coeff) = diis_solver(&self.error_vector, &self.error_vector.len()) {
                // now extrapolate the fock matrix for the next step
                (0..spin_channel).into_iter().for_each(|i_spin| {
                    let mut next_hamiltonian = MatrixFull::new(self.target_vector[0][i_spin].size.clone(),0.0);
                    coeff.iter().enumerate().for_each(|(i,value)| {
                        next_hamiltonian.self_scaled_add(&self.target_vector[i+start_dim][i_spin], *value);
                    });
                    let next_hamiltonian = next_hamiltonian.to_matrixupper();

                    if oscillation_flag {
                        if scf.mol.ctrl.print_level>0 {
                            println!("Energy increase is detected. Turn on the linear mixing algorithm with (H[DIIS, i-1] + H[DIIS, i+1]).");
                            let length = self.energy_records.len();
                            println!("Prev_Energies: ({:16.8}, {:16.8})", self.energy_records[length-2], self.energy_records[length-1]);
                        }
                        let mut alpha: f64 = self.mix_param;
                        let mut beta = 1.0-alpha;
                        scf.hamiltonian[i_spin].data.par_iter_mut().zip(self.prev_hamiltonian[0][i_spin].data.par_iter()).zip(next_hamiltonian.data.par_iter())
                        .for_each(|((to, prev), new)| {
                            *to = prev*beta + new*alpha;
                        });
                    } else {
                        scf.hamiltonian[i_spin] = next_hamiltonian;
                    }
                });

                // update the previous hamiltonian list to make sure the first item is H[DIIS, i-1]
                // and the second term is H[DIIS, i]
                if self.prev_hamiltonian.len() == 2 {self.prev_hamiltonian.remove(0);};
                self.prev_hamiltonian.push(scf.hamiltonian.clone());



            } else {
                let mut alpha = self.mix_param;
                let mut beta = 1.0-alpha;
                if scf.mol.ctrl.print_level>0 {
                    println!("WARNING: fail to obtain the DIIS coefficients. Turn to use the linear mixing algorithm, and re-invoke DIIS  8 steps later");
                }
                for i_spin in (0..spin_channel) {
                    let residual_dm = self.density_matrix[1][i_spin].sub(&self.density_matrix[0][i_spin]).unwrap();
                    scf.density_matrix[i_spin] = self.density_matrix[0][i_spin]
                        .scaled_add(&residual_dm, alpha)
                        .unwrap();
                }
                scf.generate_hf_hamiltonian(mpi_operator);
                //let index = self.target_vector.len()-1;
                //self.target_vector.remove(index);
                //self.error_vector.remove(index);
                self.start_diis_cycle = self.num_iter + 8;
                self.target_vector =  Vec::<[MatrixFull<f64>;2]>::new();
                self.error_vector =  Vec::<Vec::<f64>>::new();
            }
            let dt3 = time::Local::now();
            let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
            let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
            if scf.mol.ctrl.print_level>2 {
                println!("Hamiltonian: generation by {:10.2}s and DIIS extrapolation by {:10.2}s", timecost1,timecost2);
            }
            
        };

        // now consider if level_shift is applied
        // at present only a constant level shift is implemented for both spin channels and for the whole SCF procedure
        if let Some(level_shift) = scf.mol.ctrl.level_shift {
            let ovlp = &scf.ovlp;
            let dm_scaling_factor = match scf.scftype {
                SCFType::RHF | SCFType::ROHF=> 0.5,
                SCFType::UHF => 1.0,
                };
                match scf.scftype {
                    SCFType::RHF => {
                        let mut fock = scf.hamiltonian.get_mut(0).unwrap();
                        let dm = scf.density_matrix.get(0).unwrap();
                        level_shift_fock(fock, ovlp, level_shift, dm, dm_scaling_factor);
                    },
                    SCFType::UHF => {
                        for i_spin in 0..scf.mol.spin_channel {
                            let mut fock = scf.hamiltonian.get_mut(i_spin).unwrap();
                            let dm = scf.density_matrix.get(i_spin).unwrap();
                            level_shift_fock(fock, ovlp, level_shift, dm, dm_scaling_factor);
                        }
                    },
                    SCFType::ROHF => {
                        let mut fock = &mut scf.roothaan_hamiltonian;
                        let dm = scf.density_matrix[0].clone() + scf.density_matrix[1].clone();
                        level_shift_fock(fock, ovlp, level_shift, &dm, dm_scaling_factor);
                    }
                }
        }
    }
}

pub fn generate_diis_error_vector(hamiltonian: &[MatrixUpper<f64>;2], 
                                ovlp: &MatrixUpper<f64>, 
                                density_matrix: &mut [Vec<MatrixFull<f64>>;2],
                                spin_channel: usize) -> (Vec<f64>, [MatrixFull<f64>;2]) {
            let mut cur_error = [
                MatrixFull::new([1,1],0.0),
                MatrixFull::new([1,1],0.0)
            ];
            let mut cur_target = [hamiltonian[0].to_matrixfull().unwrap(),
                 hamiltonian[1].to_matrixfull().unwrap()];

            let mut full_ovlp = ovlp.to_matrixfull().unwrap();

            // now generte the error as the commutator of [fds-sdf]
            (0..spin_channel).into_iter().for_each(|i_spin| {
                cur_error[i_spin] = cur_target[i_spin]
                    .ddot(&mut density_matrix[1][i_spin]).unwrap();
                cur_error[i_spin] = cur_error[i_spin].ddot(&mut full_ovlp).unwrap();
                let mut dsf = full_ovlp.ddot(&mut density_matrix[1][i_spin]).unwrap();
                dsf = dsf.ddot(&mut cur_target[i_spin]).unwrap();
                cur_error[i_spin].self_sub(&dsf);

                // //transfer to an orthorgonal basis
                // let mut tmp_mat = cur_error[i_spin].ddot(&mut sqrt_inv_ovlp).unwrap();
                // cur_error[i_spin].lapack_dgemm(
                //   &mut sqrt_inv_ovlp, &mut tmp_mat,
                //   'T', 'N',
                //   1.0,0.0);
            });

            let mut norm = 0.0;
            (0..spin_channel).for_each(|i_spin| {
                let dd = cur_error[i_spin].data.par_iter().fold(|| 0.0, |acc, x| {
                    acc + x*x
                }).sum::<f64>();
                norm += dd
            });
            //println!("diis-norm : {:16.8}",norm.sqrt());

            ([cur_error[0].data.clone(),cur_error[1].data.clone()].concat(),
            cur_target)

}

pub fn diis_solver(em: &Vec<Vec<f64>>,
                   num_vec:&usize) -> Option<Vec<f64>> {

    let dim_vec = em.len();
    let start_dim = if (em.len()>=*num_vec) {em.len()-*num_vec} else {0};
    let dim = if (em.len()>=*num_vec) {*num_vec} else {em.len()};
    let mut coeff = Vec::<f64>::new();
    //let mut norm_rdm = [Vec::<f64>::new(),Vec::<f64>::new()];
    let mut odm = MatrixFull::new([1,1],0.0);
    //let mut inv_opta = MatrixFull::new([dim,dim],0.0);
    //let mut sum_inv_norm_rdm = [0.0,0.0];
    let mut sum_inv_norm_rdm = 0.0_f64;

    // now prepare the norm matrix of the residual density matrix
    let mut opta = MatrixFull::new([dim,dim],0.0);
    (start_dim..dim_vec).into_iter().for_each(|i| {
        (start_dim..dim_vec).into_iter().for_each(|j| {
            let mut inv_norm_rdm = em[i].iter()
                .zip(em[j].iter())
                .fold(0.0,|c,(d,e)| {c + d*e});
            opta.set2d([i-start_dim,j-start_dim],inv_norm_rdm);
            //sum_inv_norm_rdm += inv_norm_rdm;
        })
    });
    let inv_opta = if let Some(inv_opta) = _dinverse(&mut opta) {
        //println!("diis_solver: _dinverse");
        inv_opta
    //} else if let Some(inv_opta) = opta.lapack_power(-1.0, INVERSE_THRESHOLD) {
    //    println!("diis_solver: lapack_power");
    //    inv_opta
    } else {
        //println!("diis_solver: none");
        return None
    };
    sum_inv_norm_rdm = inv_opta.data.iter().sum::<f64>().powf(-1.0f64);

    // now prepare the coefficients for the pulay mixing
    coeff = vec![sum_inv_norm_rdm;dim];
    //coeff.iter().for_each(|i| {println!("coeff: {}",i)});
    (0..dim).zip(coeff.iter_mut()).for_each(|(i,value)| {
        //println!("{:?}",*value);
        *value *= inv_opta.get2d_slice([0,i], dim)
                 .unwrap()
                 .iter()
                 .sum::<f64>();
    });

    Some(coeff)

}

pub fn scf(mol:Molecule, mpi_operator: &Option<MPIOperator>) -> anyhow::Result<SCF> {
    let dt0 = time::Local::now();

    let mut scf_data = SCF::build(mol, mpi_operator);

    scf_without_build(&mut scf_data, mpi_operator);

    let dt2 = time::Local::now();
    if scf_data.mol.ctrl.print_level>0 {
        println!("the job costs {:16.2} seconds",(dt2.timestamp_millis()-dt0.timestamp_millis()) as f64 /1000.0)
    };

    //if scf_data.empirical_dispersion_energy != 0.0 {
    //    scf_data.scf_energy += scf_data.empirical_dispersion_energy
    //}

    Ok(scf_data)
}

fn ao2mo_rayon<'a, T, P>(eigenvector: &T, rimat_chunk: &P, row_dim: std::ops::Range<usize>, column_dim: std::ops::Range<usize>)
-> anyhow::Result<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)>
    where T: BasicMatrix<'a, f64>+std::marker::Sync,
          P: BasicMatrix<'a, f64>
{
    ao2mo_rayon_v02(eigenvector, rimat_chunk, row_dim, column_dim)
    //let mut 
    //ri_ao2mo_f
}

fn ao2mo_rayon_v01<'a, T, P>(eigenvector: &T, rimat_chunk: &P, row_dim: std::ops::Range<usize>, column_dim: std::ops::Range<usize>)
-> anyhow::Result<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)>
    where T: BasicMatrix<'a, f64>+std::marker::Sync,
          P: BasicMatrix<'a, f64>
{
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    let num_basis = eigenvector.size()[0];
    let num_state = eigenvector.size()[1];
    let num_bpair = rimat_chunk.size()[0];
    let num_auxbs = rimat_chunk.size()[1];
    let num_loc_row = row_dim.len();
    let num_loc_col = column_dim.len();
    let mut rimo = RIFull::new([num_auxbs, num_loc_row, num_loc_col],0.0);
    let (sender, receiver) = channel();

    rimat_chunk.data_ref().unwrap().par_chunks_exact(num_bpair).enumerate().for_each_with(sender, |s, (i_auxbs, m)| {
        let mut loc_ri3mo = MatrixFull::new([row_dim.len(), column_dim.len()],0.0_f64);
        let mut reduced_ri = MatrixFull::new([num_basis, num_basis], 0.0_f64);
        reduced_ri.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

        let mut tmp_mat = MatrixFull::new([num_basis,num_state], 0.0_f64);
        _dsymm(&reduced_ri, eigenvector, &mut tmp_mat, 'L', 'U', 1.0, 0.0);

        _dgemm(
            &tmp_mat, ((0..num_basis),row_dim.clone()), 'T',
            eigenvector, ((0..num_basis),column_dim.clone()), 'N',
            &mut loc_ri3mo, (0..row_dim.len(), 0..column_dim.len()),
            1.0, 0.0
        );
        s.send((loc_ri3mo, i_auxbs)).unwrap()
    });
    receiver.into_iter().for_each(|(loc_ri3mo, i_auxbs)| {
        rimo.copy_from_matr(0..num_loc_row, 0..num_loc_col, i_auxbs, 2, &loc_ri3mo, 0..num_loc_row, 0..num_loc_col)
    });

    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok((rimo, row_dim, column_dim))
}

fn ao2mo_rayon_v02<'a, T, P>(eigenvector: &T, rimat_chunk: &P, row_dim: std::ops::Range<usize>, column_dim: std::ops::Range<usize>)
-> anyhow::Result<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)>
    where T: BasicMatrix<'a, f64>+std::marker::Sync,
          P: BasicMatrix<'a, f64>
{
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    utilities::omp_set_num_threads_wrapper(1);

    let num_basis = eigenvector.size()[0];
    let num_state = eigenvector.size()[1];
    let num_bpair = rimat_chunk.size()[0];
    let num_auxbs = rimat_chunk.size()[1];
    let num_loc_row = row_dim.len();
    let num_loc_col = column_dim.len();
    let mut rimo = RIFull::new([num_auxbs, num_loc_row, num_loc_col],0.0);
    let (sender, receiver) = channel();

    rimat_chunk.data_ref().unwrap().par_chunks_exact(num_bpair).enumerate().for_each_with(sender, |s, (i_auxbs, m)| {
        let mut loc_ri3mo = MatrixFull::new([row_dim.len(), column_dim.len()],0.0_f64);
        let mut reduced_ri = MatrixFull::new([num_basis, num_basis], 0.0_f64);
        reduced_ri.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

        let mut tmp_mat = MatrixFull::new([num_basis,num_state], 0.0_f64);
        _dsymm(&reduced_ri, eigenvector, &mut tmp_mat, 'L', 'U', 1.0, 0.0);

        _dgemm(
            &tmp_mat, ((0..num_basis),row_dim.clone()), 'T',
            eigenvector, ((0..num_basis),column_dim.clone()), 'N',
            &mut loc_ri3mo, (0..row_dim.len(), 0..column_dim.len()),
            1.0, 0.0
        );
        s.send((loc_ri3mo, i_auxbs)).unwrap()
    });
    receiver.into_iter().for_each(|(loc_ri3mo, i_auxbs)| {
        rimo.copy_from_matr(0..num_loc_row, 0..num_loc_col, i_auxbs, 2, &loc_ri3mo, 0..num_loc_row, 0..num_loc_col)
    });

    utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok((rimo, row_dim, column_dim))
}

pub fn level_shift_fock(fock: &mut MatrixUpper<f64>, ovlp: &MatrixUpper<f64>, level_shift: f64,  dm: &MatrixFull<f64>, dm_scaling_factor: f64) {
    // FC = SCE
    // F' = F + SC \Lambda C^\dagger S
    // F' = F + LF * (S - SDS) 

    let num_basis = dm.size()[0];

    let mut tmp_s = MatrixFull::new([num_basis, num_basis], 0.0);
    tmp_s.iter_matrixupper_mut().unwrap().zip(ovlp.data.iter()).for_each(|(to, from)| {*to = *from});
    let mut tmp_s2 = tmp_s.clone();
    let mut tmp_s3 = tmp_s.clone();
    _dsymm(&tmp_s, dm, &mut tmp_s2, 'L', 'U', -dm_scaling_factor, 0.0);
    _dsymm(&tmp_s, &mut tmp_s2, &mut tmp_s3, 'R', 'U', 1.0, 1.0);
    fock.data.iter_mut().zip(tmp_s3.iter_matrixupper().unwrap()).for_each(|(to, from)| {*to += *from*level_shift});
}

pub fn diagonalize_hamiltonian_outside(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) -> ([MatrixFull<f64>;2], [Vec<f64>;2], usize) {
    let mut eigenvectors = [MatrixFull::empty(),MatrixFull::empty()];
    let mut eigenvalues = [Vec::new(),Vec::new()];
    let mut num_state = 0;

    // perform diagonalization within the first mpi task at present.
    // NOTE:: to fully utilize all CPU resources, a scalapack memory distribution is necessary.
    if let Some(mpi_io) = mpi_operator {
        if mpi_io.rank == 0 {
            (eigenvectors, eigenvalues, num_state) = diagonalize_hamiltonian_outside_fast(scf_data);
        }
        for i_spin in 0..scf_data.mol.spin_channel {
            mpi_broadcast_vector(&mpi_io.world, &mut eigenvalues[i_spin], 0);
            mpi_broadcast_matrixfull(&mpi_io.world, &mut eigenvectors[i_spin], 0);
        }
        mpi_broadcast(&mpi_io.world, &mut num_state, 0);


    } else {
        (eigenvectors, eigenvalues, num_state) = diagonalize_hamiltonian_outside_fast(scf_data);
    }

    //println!("diagonalize_hamiltonian_outside: num_state {}", num_state);


    (eigenvectors, eigenvalues, scf_data.mol.num_state)
}

pub fn diagonalize_hamiltonian_outside_fast(scf_data: &SCF)  -> ([MatrixFull<f64>;2], [Vec<f64>;2], usize) {
    let spin_channel = scf_data.mol.spin_channel;
    let mut num_state = scf_data.mol.num_state;
    let dt1 = time::Local::now();

    let mut eigenvectors = [MatrixFull::empty(),MatrixFull::empty()];
    let mut eigenvalues = [Vec::new(),Vec::new()];

    match scf_data.scftype {
        SCFType::RHF | SCFType::UHF => {
            for i_spin in (0..spin_channel) {
                let (eigenvector_spin, eigenvalue_spin)=
                    _hamiltonian_fast_solver(&scf_data.hamiltonian[i_spin], &scf_data.ovlp, &mut num_state).unwrap();
                    //self.hamiltonian[i_spin].to_matrixupperslicemut()
                    //.lapack_dspgvx(self.ovlp.to_matrixupperslicemut(),num_state).unwrap();
                eigenvectors[i_spin] = eigenvector_spin;
                eigenvalues[i_spin] = eigenvalue_spin;
            }
        },
        SCFType::ROHF => {
            // diagonalize Roothaan Fock matrix
            let (eigenvector, eigenvalue)=
                _hamiltonian_fast_solver(&scf_data.roothaan_hamiltonian, &scf_data.ovlp, &mut num_state).unwrap();
            eigenvectors[0] = eigenvector;
            eigenvalues[0] = eigenvalue;
        }
    };
    (eigenvectors, eigenvalues, num_state)
}

pub fn diagonalize_hamiltonian_outside_rayon(scf_data: &SCF) -> ([MatrixFull<f64>;2], [Vec<f64>;2], usize) {
    let spin_channel = scf_data.mol.spin_channel;
    let num_state = scf_data.mol.num_state;
    //println!("diagonalize_hamiltonian_outside_rayon: num_state {}", num_state);
    let dt1 = time::Local::now();
    let mut num_state_out = num_state;

    let mut eigenvectors = [MatrixFull::empty(),MatrixFull::empty()];
    let mut eigenvalues = [Vec::new(),Vec::new()];

    match scf_data.scftype {
        SCFType::ROHF => {
            // diagonalize Roothaan Fock matrix
            let (eigenvector, eigenvalue, tmp_num_state_out)=
                _dspgvx(&scf_data.roothaan_hamiltonian, &scf_data.ovlp, num_state).unwrap();
            eigenvectors[0] = eigenvector;
            eigenvalues[0] = eigenvalue;
            if tmp_num_state_out < num_state_out {
                num_state_out = tmp_num_state_out;
            }
        },
        SCFType::RHF | SCFType::UHF => {
            for i_spin in (0..spin_channel) {
                let (eigenvector_spin, eigenvalue_spin, tmp_num_state_out)=
                    _dspgvx(&scf_data.hamiltonian[i_spin], &scf_data.ovlp, num_state).unwrap();
                    //self.hamiltonian[i_spin].to_matrixupperslicemut()
                    //.lapack_dspgvx(self.ovlp.to_matrixupperslicemut(),num_state).unwrap();
                eigenvectors[i_spin] = eigenvector_spin;
                eigenvalues[i_spin] = eigenvalue_spin;
                if tmp_num_state_out < num_state_out {
                    num_state_out = tmp_num_state_out;
                }
            }
        }
    }

    (eigenvectors,eigenvalues, num_state_out)
}

pub fn semi_diagonalize_hamiltonian_outside(scf_data: &SCF) -> ([MatrixFull<f64>; 2], [Vec<f64>; 2], [MatrixFull<f64>; 2], usize) {
    // get the semi-canonical orbitals for RO-xDH calculations
    // See Knowles et al., Chem. Phys. Lett. 186(2), 130–136 (1991)
    let num_state = scf_data.mol.num_state;
    //let num_basis = scf_data.mol.num_basis; 
    let d_idx = 0..scf_data.lumo[1];
    let s_idx = scf_data.lumo[1]..scf_data.lumo[0];
    let v_idx = scf_data.lumo[0]..num_state;
    let ds_idx = 0..scf_data.lumo[0];
    let sv_idx = scf_data.lumo[1]..num_state;

    let fock = [scf_data.hamiltonian[0].to_matrixfull().unwrap(), scf_data.hamiltonian[1].to_matrixfull().unwrap()];

    let idx_list = [ds_idx, v_idx, d_idx, sv_idx];
    let fock_list = [&fock[0], &fock[0], &fock[1], &fock[1]];

    let mut semi_eigenvectors_list = [
        MatrixFull::new([num_state, scf_data.lumo[0]], 0.0),
        MatrixFull::new([num_state, num_state - scf_data.lumo[0]], 0.0),
        MatrixFull::new([num_state, scf_data.lumo[1]], 0.0),
        MatrixFull::new([num_state, num_state - scf_data.lumo[1]], 0.0),
    ];

    for (i, idx) in idx_list.iter().enumerate() {
        let c_slice = scf_data.eigenvectors[0].to_matrixfullslice_columns(idx.clone());
        let c = MatrixFull::from_vec(c_slice.size, c_slice.data.to_vec()).unwrap();

        let fock = apply_projection_operator(&c, fock_list[i], &c);
        let (eigenvectors, _eigenvalues, _) = _dsyevd(&fock, 'V');

        _dgemm_full(&c, 'N', &eigenvectors.unwrap(), 'N', &mut semi_eigenvectors_list[i], 1.0, 0.0);
    }

    let mut semi_eigenvectors: [MatrixFull<f64>; 2] = [
        semi_eigenvectors_list[0].clone(),
        semi_eigenvectors_list[2].clone(),
        ];

    semi_eigenvectors[0].append_column(&semi_eigenvectors_list[1]); // [ c_ds | c_v ]
    semi_eigenvectors[1].append_column(&semi_eigenvectors_list[3]); // [ c_d  | c_sv ]

    let mut semi_fock = [MatrixFull::new([num_state, num_state], 0.0), MatrixFull::new([num_state, num_state], 0.0)];
    let mut semi_eigenvalues = [Vec::new(), Vec::new()];

    for i_spin in 0..2 {
        let fock_tmp = apply_projection_operator(&semi_eigenvectors[i_spin], &fock[i_spin], &semi_eigenvectors[i_spin]);
    
        let diag_terms: Vec<f64> = fock_tmp.get_diagonal_terms().unwrap().into_iter().map(|&x| x).collect();
        
        semi_fock[i_spin] = fock_tmp;
        semi_eigenvalues[i_spin] = diag_terms;
    }
    
    (semi_eigenvectors, semi_eigenvalues, semi_fock, num_state)
}

pub fn generate_occupation_outside(scf_data: &SCF) -> ([Vec<f64>;2], [usize;2], [usize;2]) {
    let mut occ = [vec![],vec![]];
    let mut homo = [0,0];
    let mut lumo = [0,0];
    match scf_data.mol.ctrl.occupation_type {
        OCCType::INTEGER => {
            (occ,homo,lumo) = generate_occupation_integer(&scf_data.mol,&scf_data.scftype);
        },
        OCCType::ATMSAD => {
            (occ,homo,lumo) = generate_occupation_sad(scf_data.mol.geom.elem.get(0).unwrap(),scf_data.mol.num_state, scf_data.mol.ecp_electrons);
        },
        OCCType::FRAC => {
            (occ,homo,lumo) = generate_occupation_frac_occ(&scf_data.mol,&scf_data.scftype, &scf_data.eigenvalues, scf_data.mol.ctrl.frac_tolerant);
        }
    }

    let mut force_occ = scf_data.mol.ctrl.force_state_occupation.clone();

    if force_occ.len()>0 {
        adapt_occupation_with_force_projection(
        &mut occ, &mut homo, &mut lumo,
        &mut force_occ, 
        &scf_data.scftype, 
        &scf_data.eigenvectors, 
        &scf_data.ovlp, 
        &scf_data.ref_eigenvectors);
        if scf_data.mol.ctrl.print_level>=2 {
            let mut window = [occ[0].len()-1,0];
            force_occ.iter().map(|x| x.get_check_window())
                .for_each(|[x,y]| {
                    if x< window[0] {window[0] = x};
                    if y>window[1] {window[1]=y}
                });
            println!("Occupation in Alpha Channel ({}-{}):", window[0], window[1]);
            let mut output = String::new();
            &occ[0][window[0]..window[1]].iter().enumerate().for_each(|(li,x)| {
                output = format!("{} ({:4}, {:6.3})", output, li+window[0], x);
                if (li+1)%5 == 0 {
                    output = format!("{}\n", output);
                }
            });
            println!("{}",output);
            if scf_data.mol.spin_channel == 2{
                println!("Occupation in Beta Channel: ({}-{}):", window[0], window[1]);
                let mut output = String::new();
                &occ[1][window[0]..window[1]].iter().enumerate().for_each(|(li,x)| {
                    output = format!("{} ({:4}, {:6.3})", output, li+window[0], x);
                    if (li+1)%5 == 0 {
                        output = format!("{}\n", output);
                    }
                });
                println!("{}",output);
            }
        }
    }



    (occ, homo, lumo)
}

pub fn generate_density_matrix_outside(scf_data: &SCF) -> Vec<MatrixFull<f64>>{

    let num_basis = scf_data.mol.num_basis;
    let num_state = scf_data.mol.num_state;
    let spin_channel = scf_data.mol.spin_channel;
    let homo = &scf_data.homo;
    //println!("homo: {:?}", &homo);
    let mut dm = vec![
        MatrixFull::empty(),
        MatrixFull::empty()
        ];
    (0..spin_channel).into_iter().for_each(|i_spin| {
        let mut dm_s = &mut dm[i_spin];
        *dm_s = MatrixFull::new([num_basis,num_basis],0.0);
        let eigv_s = if let SCFType::ROHF = scf_data.scftype {
            &scf_data.eigenvectors[0]
        } else {
            &scf_data.eigenvectors[i_spin]
        };
        let occ_s =  &scf_data.occupation[i_spin];

        let nw =  scf_data.homo[i_spin]+1;
        //println!("number of occupied orbitals from dm generation: {}", nw);

        let mut weight_eigv = MatrixFull::new([num_basis, num_state],0.0_f64);
        //let mut weight_eigv = eigv_s.clone();
        weight_eigv.par_iter_columns_mut(0..nw).unwrap().zip(eigv_s.par_iter_columns(0..nw).unwrap())
            .for_each(|value| {
                value.0.into_iter().zip(value.1.into_iter()).for_each(|value| {
                    *value.0 = *value.1
                })
            });
        // prepare weighted eigenvalue matrix wC
        weight_eigv.par_iter_columns_mut(0..nw).unwrap().zip(occ_s[0..nw].par_iter()).for_each(|(we,occ)| {
        //weight_eigv.data.chunks_exact_mut(weight_eigv.size[0]).zip(occ_s.iter()).for_each(|(we,occ)| {
            we.iter_mut().for_each(|c| *c = *c*occ);
        });

        // dm = wC*C^{T}
        _dgemm_full(&weight_eigv,'N',eigv_s, 'T',dm_s, 1.0, 0.0);
        //dm_s.lapack_dgemm(&mut weight_eigv, eigv_s, 'N', 'T', 1.0, 0.0);
        //dm_s.formated_output(5, "full");
    });
    //if let SCFType::ROHF = scf_data.scftype {dm[1]=dm[0].clone()};
    //scf_data.density_matrix = dm;

    dm

}

pub fn initialize_scf(scf_data: &mut SCF, mpi_operator: &Option<MPIOperator>) {

    // update the corresponding geometry information, which is crucial 
    // for preparing the following integrals accurately
    let position = &scf_data.mol.geom.position;
    scf_data.mol.cint_env = scf_data.mol.update_geom_poisition_in_cint_env(position);

    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("Overall", "SCF Preparation");
    time_mark.count_start("Overall");

    time_mark.new_item("CInt", "Two, Three, and Four-center integrals");
    time_mark.count_start("CInt");
    scf_data.prepare_necessary_integrals(mpi_operator);
    time_mark.count("CInt");


    time_mark.new_item("DFT Grids", "Initialization of the tabulated Grids and AOs");
    time_mark.count_start("DFT Grids");
    scf_data.prepare_density_grids();
    time_mark.count("DFT Grids");

    time_mark.new_item("ISDF", "ISDF initialization");
    time_mark.count_start("ISDF");
    scf_data.prepare_isdf(mpi_operator);
    time_mark.count("ISDF");

    time_mark.new_item("InitGuess", "Prepare initial guess");
    time_mark.count_start("InitGuess");
    initial_guess(scf_data, mpi_operator);
    if ! scf_data.mol.ctrl.atom_sad && scf_data.mol.ctrl.print_level>2 {
        println!("Initial density matrix by Atom SAD:");
        scf_data.density_matrix[0].formated_output(5, "full");
    }
    //println!("======== IGOR debug  for xc components after Initial Guess =======");
    //let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
    //let post_xc_energy = if let Some(grids) = &scf_data.grids {
    //    dfa.post_xc_exc(&scf_data.mol.ctrl.post_xc, grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
    //} else {
    //    vec![[0.0,0.0]]
    //};
    //post_xc_energy.iter().zip(scf_data.mol.ctrl.post_xc.iter()).for_each(|(energy, name)| {
    //    println!("{:<16}: {:16.8} Ha", name, energy[0]+energy[1]);
    //});
    //println!("======= IGOR debug ========");
    time_mark.count("InitGuess");

    time_mark.count("Overall");
    if scf_data.mol.ctrl.print_level>=2 {
        time_mark.report_all();
    }

}

pub fn scf_without_build(scf_data: &mut SCF, mpi_operator: &Option<MPIOperator>) {
    scf_data.generate_hf_hamiltonian(mpi_operator);

    let mut scf_records=ScfTraceRecord::initialize(&scf_data);

    if scf_data.mol.ctrl.print_level>0 {println!("The total energy: {:20.10} Ha by the initial guess",scf_data.scf_energy)};
    //let mut scf_continue = true;
    if scf_data.mol.ctrl.noiter {
        println!("Warning: the SCF iteration is skipped!");
        return;
    }

    // now prepare the input density matrix for the first iteration and initialize the records
    scf_data.diagonalize_hamiltonian(mpi_operator);
    scf_data.generate_occupation();

    if scf_data.mol.ctrl.guess_mix {
        for (i_spin, &theta_deg) in scf_data.mol.ctrl.guess_mix_theta_deg.iter().enumerate() {
            if theta_deg <= 0.0 || theta_deg > 45.0 {
                println!(
                    "WARNING: theta for spin {} = {:.1}° is outside the recommended range (0°–45°); mixing may be ineffective or unstable.",
                    i_spin, theta_deg
                );
            }
    
            let (cos_theta, sin_theta) = {
                let rad = theta_deg.to_radians();
                (rad.cos(), rad.sin())
            };
    
            let homo = scf_data.homo[i_spin];
            let lumo = scf_data.lumo[i_spin];
            let eigenvector_mut = scf_data.eigenvectors.get_mut(i_spin).unwrap();
            let homo_vec: Vec<f64> = eigenvector_mut.iter_column(homo).cloned().collect();
            let lumo_vec: Vec<f64> = eigenvector_mut.iter_column(lumo).cloned().collect();
    
            let (mixed_homo_vec, mixed_lumo_vec) = (
                homo_vec.iter().zip(&lumo_vec).map(|(h, l)| cos_theta * h + sin_theta * l).collect::<Vec<f64>>(),
                homo_vec.iter().zip(&lumo_vec).map(|(h, l)| -sin_theta * h + cos_theta * l).collect::<Vec<f64>>(),
            );
    
            for (val, slot) in mixed_homo_vec.iter().zip(eigenvector_mut.iter_column_mut(homo)) {
                *slot = *val;
            }
            for (val, slot) in mixed_lumo_vec.iter().zip(eigenvector_mut.iter_column_mut(lumo)) {
                *slot = *val;
            }
        }
    }
    
    

    scf_data.generate_density_matrix();
    scf_records.update(&scf_data);

    //println!("======== IGOR debug for xc components =======");
    //let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
    //let post_xc_energy = if let Some(grids) = &scf_data.grids {
    //    dfa.post_xc_exc(&scf_data.mol.ctrl.post_xc, grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
    //} else {
    //    vec![[0.0,0.0]]
    //};
    //post_xc_energy.iter().zip(scf_data.mol.ctrl.post_xc.iter()).for_each(|(energy, name)| {
    //    println!("{:<16}: {:16.8} Ha", name, energy[0]+energy[1]);
    //});
    //println!("======= IGOR debug for xc components ========");

    let mut scf_converge = [false;2];
    while ! (scf_converge[0] || scf_converge[1]) {
        let dt1 = time::Local::now();

        scf_records.prepare_next_input(scf_data, mpi_operator);

        let dt1_1 = time::Local::now();

        scf_data.diagonalize_hamiltonian(mpi_operator);
        let dt1_2 = time::Local::now();
        scf_data.generate_occupation();
        scf_data.generate_density_matrix();
        let dt1_3 = time::Local::now();
        scf_converge = scf_data.check_scf_convergence(&scf_records);
        let dt1_4 = time::Local::now();
        scf_records.update(&scf_data);
        let dt1_5 = time::Local::now();


        let dt2 = time::Local::now();
        let timecost = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        if scf_data.mol.ctrl.print_level>0 {
            if scf_data.mol.spin_channel == 2 {
                let [square_spin, spin_z] = evaluate_spin_angular_momentum(&scf_data.density_matrix, &scf_data.ovlp, scf_data.mol.spin_channel, &scf_data.mol.num_elec);
                println!("Energy: {:18.10} Ha with <S^2> = {:6.3} and <2S+1> = {:6.3} after {:4} iterations (in {:10.2} seconds).",
                     scf_records.scf_energy,
                     square_spin, spin_z,
                     scf_records.num_iter-1,
                     timecost)
            } else {
                println!("Energy: {:18.10} Ha after {:4} iterations (in {:10.2} seconds).",
                     scf_records.scf_energy,
                     scf_records.num_iter-1,
                     timecost)
            }
        };
        if scf_data.mol.ctrl.print_level>1 {
            println!("Detailed timing info in this SCF step:");
            let timecost = (dt1_1.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
            println!("prepare_next_input:      {:10.2}s", timecost);
            let timecost = (dt1_2.timestamp_millis()-dt1_1.timestamp_millis()) as f64 /1000.0;
            println!("diagonalize_hamiltonian: {:10.2}s", timecost);
            let timecost = (dt1_3.timestamp_millis()-dt1_2.timestamp_millis()) as f64 /1000.0;
            println!("generate_density_matrix: {:10.2}s", timecost);
            let timecost = (dt1_4.timestamp_millis()-dt1_3.timestamp_millis()) as f64 /1000.0;
            println!("check_scf_convergence:   {:10.2}s", timecost);
            let timecost = (dt1_5.timestamp_millis()-dt1_4.timestamp_millis()) as f64 /1000.0;
            println!("scf_records.update:      {:10.2}s", timecost);
        }
    }
    if scf_converge[0] {
        if scf_data.mol.ctrl.print_level>0 {println!("SCF is converged after {:4} iterations.", scf_records.num_iter-1)};
        // Level shift is disabled before the final diagonalization to ensure accurate eigenvalues.
        // Formatted printing of eigenvalues and eigenvectors is now performed after re-diagonalizing the HF Hamiltonian.

        match scf_data.mol.ctrl.occupation_type {
            OCCType::FRAC => {
                let (occupation, homo, lumo) = check_norm::generate_occupation_integer(&scf_data.mol, &scf_data.scftype);
                scf_data.occupation = occupation;
                scf_data.homo = homo;
                scf_data.lumo = lumo;
                scf_data.generate_density_matrix();
                scf_data.generate_hf_hamiltonian(mpi_operator);
                scf_data.diagonalize_hamiltonian(mpi_operator);

            }
            _ => {
                scf_data.generate_hf_hamiltonian(mpi_operator); 
                scf_data.diagonalize_hamiltonian(mpi_operator); 
            }
        }

        if scf_data.mol.ctrl.print_level>1 {
            scf_data.formated_eigenvalues((scf_data.homo.iter().max().unwrap()+4).min(scf_data.mol.num_state));
        }
        if scf_data.mol.ctrl.print_level>3 {
            scf_data.formated_eigenvectors();
        }
        // not yet implemented. Just an empty subroutine
    } else {
        //if scf_data.mol.ctrl.restart {save_chkfile(&scf_data)};
        println!("SCF does not converge within {:03} iterations",scf_records.num_iter);
    }

}

pub fn vj_on_the_fly_par(mol: &Molecule, dm: &Vec<MatrixFull<f64>>) -> Vec<MatrixUpper<f64>>{

    let num_shell = mol.cint_bas.len();
    let num_basis = mol.num_basis;
    let spin_channel = mol.spin_channel;

    // establish the map between matrixupper and matrixfull
    let matupp_length = (mol.num_basis+1)*mol.num_basis/2;
    let matrixupper_index = map_upper_to_full(matupp_length).unwrap();

    let mut dm_upper = Vec::new();
    let mut dm_diagonal = Vec::new();
    for i_spin in 0..spin_channel {
        dm_upper.push(dm[i_spin].to_matrixupper());
        dm_diagonal.push(dm[i_spin].iter_diagonal().unwrap().map(|x| *x).collect::<Vec<f64>>())
    }
    //let dm_upper = dm.iter().map(|dm_s| dm_s.to_matrixupper()).collect::<Vec<MatrixUpper<f64>>>();
    //let dm_diagonal  = dm.iter().map(|dm_s| 
    //    dm_s.iter_diagonal().unwrap().map(|x| *x).collect::<Vec<f64>>()
    //).collect::<Vec<Vec<f64>>>();

    //let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1, 0.0), MatrixUpper::new(1, 0.0)];
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty(), MatrixUpper::empty()];

    let mut vj_full: Vec<MatrixFull<f64>> = vec![
        MatrixFull::new([num_basis, num_basis], 0.0),
        if spin_channel==2 {
            MatrixFull::new([num_basis, num_basis], 0.0)
        } else {
            MatrixFull::empty()
        }
    ];


    // initialize the parallel tasks
    let mut index = Vec::new();
    for l in 0..num_shell {
        for k in 0..l+1 {
            index.push([mol.cint_fdqc[k][1]*mol.cint_fdqc[l][1],k,l])
        }
    };
    index = index.iter().sorted_by(|a,b| Ord::cmp(&a[0], &b[0]))
        .map(|x| *x).collect::<Vec<[usize;3]>>();

    let half_length = index.len()/2;
    let is_odd = index.len()%2;

    // re-arrange the tasks that mixed universally according the work loading.
    let mut index_new = Vec::new();
    if is_odd==1 {index_new.push(index[half_length])};
    index[0..half_length].iter().zip(index[half_length+is_odd..index.len()].iter().rev()).for_each(|(task1, task2)| {
        index_new.push(*task1);
        index_new.push(*task2);
    });

    let par_tasks = utilities::balancing(index_new.len(), rayon::current_num_threads());
    let (sender, receiver) = channel();

    par_tasks.par_iter().for_each_with(sender, |s,task_range| {
        //rayon::current_thread_index();
        let mut cint_data = mol.initialize_cint(false);
        cint_data.cint2e_optimizer_rust();

        let mut out_submatrix = Vec::new();

        index_new[task_range.clone()].iter().for_each(|[weight,k,l]| {
            let bas_start_k = mol.cint_fdqc[*k][0];
            let bas_len_k = mol.cint_fdqc[*k][1];
            let bas_start_l = mol.cint_fdqc[*l][0];
            let bas_len_l = mol.cint_fdqc[*l][1];


            let klij = mol.int_ijkl_given_kl_v03(*k, *l, &matrixupper_index, &mut cint_data);
            let mut out = vec![
                MatrixFull::new([bas_len_k, bas_len_l],0.0),
                if spin_channel==2 {
                    MatrixFull::new([bas_len_k, bas_len_l],0.0)
                } else {
                    MatrixFull::empty()
                }
            ];
            for i_spin in 0..spin_channel {
                let dm_s_upper = &dm_upper[i_spin];
                let dm_s_diagonal = &dm_diagonal[i_spin];
                let mut out_s = &mut out[i_spin];

                out_s.iter_columns_full_mut().enumerate().for_each(|(loc_l,x)|{
                    x.iter_mut().enumerate().for_each(|(loc_k,elem)|{
                        let ao_k = loc_k + bas_start_k;
                        let ao_l = loc_l + bas_start_l;
                        let eri_cd = klij.get(&[loc_k, loc_l]).unwrap();
                        let mut sum = dm_s_upper.data.iter().zip(eri_cd.iter())
                            .fold(0.0,|sum, (p,eri)| {
                            sum + *p * *eri
                        });
                        let mut diagonal = dm_s_diagonal.iter().zip(eri_cd.iter_diagonal()).fold(0.0,|diagonal, (p,eri)| {
                            diagonal + *p * *eri
                        });
                        sum = sum*2.0 - diagonal;

                        *elem = sum;
                    });
                });
            }

            out_submatrix.push((out,*k,*l));

        });

        cint_data.final_c2r();
        s.send(out_submatrix).unwrap();
    });

    receiver.into_iter().for_each(|out_submatrix| {
        out_submatrix.into_iter().for_each(|(out,k,l)| {
        let bas_start_k = mol.cint_fdqc[k][0];
        let bas_len_k = mol.cint_fdqc[k][1];
        let bas_start_l = mol.cint_fdqc[l][0];
        let bas_len_l = mol.cint_fdqc[l][1];
        for i_spin in 0..spin_channel {
            let mut vj_s = &mut vj_full[i_spin];
            let out_s = &out[i_spin];
            vj_s.copy_from_matr(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l, 
                out_s, 0..bas_len_k,0..bas_len_l);
            //vj_i.iter_submatrix_mut(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l).zip(out.iter())
            //    .for_each(|(to, from)| {*to = *from});
        }
        });
    });

    for i_spin in 0..spin_channel {
        vj[i_spin] = vj_full[i_spin].to_matrixupper();
    }

    vj
}

pub fn vj_on_the_fly_par_batch_by_batch(mol: &Molecule, dm: &Vec<MatrixFull<f64>>) -> Vec<MatrixUpper<f64>>{

    let matrixupper_index = map_upper_to_full((mol.num_basis+1)*mol.num_basis/2).unwrap();
    let batch_length = mol.ctrl.batch_size;
    let mut batches = Vec::new();
    let mut total_length = matrixupper_index.size as i32;
    let mut start = 0;
    while total_length >=0 {
        batches.push([start, batch_length]);
        start += batch_length;
        total_length -= (batch_length as i32);
    }
    let ind = batches.len()-1;
    if total_length < 0 {
        let [start, mut batch_length] = batches.pop().unwrap();
        batch_length -= (total_length.abs() as usize);
        if batch_length > 0 {
            batches.push([start,batch_length])
        }
    }

    //println!("{:?}", &batches);

    let num_shell = mol.cint_bas.len();
    let num_basis = mol.num_basis;
    let spin_channel = mol.spin_channel;
    //let dm = &self.density_matrix;
    let mut vj: Vec<MatrixUpper<f64>> = vec![];
    //let mol = &self.mol;
    //utilities::omp_set_num_threads_wrapper(1);
    for i_spin in 0..spin_channel{
        let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
        let dm_s_upper = dm[i_spin].to_matrixupper();
        let dm_s_diagnoal = dm[i_spin].iter_diagonal().unwrap().map(|x| *x).collect::<Vec<f64>>();
        let par_tasks = utilities::balancing(num_shell*num_shell, rayon::current_num_threads());
        let (sender, receiver) = channel();
        let mut index = Vec::new();
        for l in 0..num_shell {
            for k in 0..l+1 {
                index.push((k,l))
            }
        };

        //utilities::balancing_type_02(num_tasks, num_threads, per_communication);
        index.par_iter().for_each_with(sender,|s,(k,l)|{
            let bas_start_k = mol.cint_fdqc[*k][0];
            let bas_len_k = mol.cint_fdqc[*k][1];
            let bas_start_l = mol.cint_fdqc[*l][0];
            let bas_len_l = mol.cint_fdqc[*l][1];

            let mut out = MatrixFull::new([bas_len_k, bas_len_l],0.0);
            let mut output = String::new();
            for [start, batch_length] in &batches {
                //if *k==0 && *l == 0 {
                //    println!("batch: ({},{})",start, batch_length);
                //}
                let batch = (*start..*start+batch_length);
                let [str_row,str_col] = matrixupper_index[batch.start];
                let [end_row,end_col] = matrixupper_index[batch.end-1];
                let diag_list = if end_row < end_col {str_col..end_col} else {str_col..(end_col+1)};
                let klij = mol.int_ijkl_given_kl_batch(*k, *l, batch.clone(), &matrixupper_index);
                //if *k==0 && *l == 0 && (batch.start == 60|| batch.start==75) {
                //    println!("{:?}", &klij[(0,0)].data);
                //}
                let mut sum = 0.0;
                out.iter_columns_full_mut().enumerate().for_each(|(loc_l,x)|{
                    x.iter_mut().enumerate().for_each(|(loc_k,elem)|{
                        let ao_k = loc_k + bas_start_k;
                        let ao_l = loc_l + bas_start_l;
                        let eri_cd = &klij[(loc_k, loc_l)];
                        let mut sum = dm_s_upper.data[batch.clone()].iter().zip(eri_cd.iter())
                            .fold(0.0,|sum, (p,eri)| {
                        //let mut sum = dm_s_upper.data[batch.clone()].iter().enumerate().zip(eri_cd.iter())
                        //    .fold(0.0,|sum, ((i,p),eri)| {
                            //if ao_k == 0 && ao_l ==0 {
                            //    output = format!("{}, ({},{:8.4},{:8.4})", output, i+batch.start, p, eri);
                            //}
                            sum + *p * *eri
                        });

                        let mut diagonal = 0.0;
                        diag_list.clone().for_each(|i| {

                            let p = dm_s_diagnoal[i];

                            let i_in_mu = (i+1)*i/2 + i;
                            let i_in_smu = i_in_mu - eri_cd.global_range.start;

                            let eri = eri_cd.data[i_in_smu];

                            diagonal += p*eri;

                            //if ao_k == 0 && ao_l ==0 {
                            //    output = format!("{}, ({},{:8.4},{:8.4})", output, i, p, eri);
                            //}


                        });

                        *elem += sum*2.0 - diagonal;

                    });
                });

            }
            //if *k==0 && *l == 0 {
            //    println!("{}",output);
            //}
            //if *k==0 && *l==0 {
            //    out.formated_output(5, "full");
            //}

            s.send((out,*k,*l)).unwrap();
        });
        receiver.into_iter().for_each(|(out,k,l)| {
            let bas_start_k = mol.cint_fdqc[k][0];
            let bas_len_k = mol.cint_fdqc[k][1];
            let bas_start_l = mol.cint_fdqc[l][0];
            let bas_len_l = mol.cint_fdqc[l][1];
            vj_i.copy_from_matr(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l, 
                &out, 0..bas_len_k,0..bas_len_l);
            //vj_i.iter_submatrix_mut(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l).zip(out.iter())
            //    .for_each(|(to, from)| {*to = *from});
        });
        

        vj.push(vj_i.to_matrixupper());
    }
    if spin_channel == 1{
        vj.push(MatrixUpper::new(1, 0.0));       
    }
    vj
}

// evaluate the expectation value of squre spin angular momentum operator (S^2) as well as the expectation value of spin angular momentum operator along z axis
// the expectation value of S^2 is given by 3/4*Tr[(scf_data.density_matrix[0]] + scf_data.density_matrix[1])\dot scf_data.ovlp.to_matrixfull().unwrap()]
// the expectation value of S_z^2  is given by 1/2*Tr[(scf_data.density_matrix[0] -scf_data.[1])\dot scf_data.ovlp.to_matrixfull().unwrap()]
// use MatrixFull::iter_diagonal() to get the diagonal elements of a matrix for the `Tr` evaluation
pub fn evaluate_spin_angular_momentum_wrong(dm: &Vec<MatrixFull<f64>>, ovlp: &MatrixUpper<f64>,spin_channel: usize) -> [f64;2] {
    let mut tr_spin_angular_momentum = [0.0;2];
    let ovlp_full = ovlp.to_matrixfull().unwrap();

    if spin_channel==1 {
        let dm_s = dm.get(0).unwrap();
        let mut tmp_matrix = dm_s.clone();
        _dgemm_full(dm_s, 'N', &ovlp_full, 'N', &mut tmp_matrix, 1.0, 0.0);

        [0.75*tmp_matrix.iter_diagonal().unwrap().fold(0.0, |acc, val| acc + val),0.0]
    } else {
        let mut tr_spin_angular_momentum = [0.0;2];
        for i_spin in 0..spin_channel {
            let dm_s = dm.get(i_spin).unwrap();
            let mut tmp_matrix = dm_s.clone();
            _dgemm_full(dm_s, 'N', &ovlp_full, 'N', &mut tmp_matrix, 1.0, 0.0);
            tr_spin_angular_momentum[i_spin] = tmp_matrix.iter_diagonal().unwrap().fold(0.0, |acc, val| acc + val);
        }

        [0.75*(tr_spin_angular_momentum[0]+tr_spin_angular_momentum[1]),
         0.50*(tr_spin_angular_momentum[0]-tr_spin_angular_momentum[1])]
    }
} 


pub fn evaluate_spin_angular_momentum(dm: &Vec<MatrixFull<f64>>, ovlp: &MatrixUpper<f64>, spin_channel: usize, num_elec: &[f64;3]) -> [f64;2] {

    let n_a = num_elec[1];
    let n_b = num_elec[2];
    let ms = (n_a-n_b)*0.5;
    let mut s2 = ms*(ms+1.0);
    let mut mlpy = 2.0*ms + 1.0;

    if spin_channel == 2 {
        let ovlp_full = ovlp.to_matrixfull().unwrap();

        s2 += n_b;

        let mut tmp_matr_a = dm[0].clone();
        let mut tmp_matr_b = dm[0].clone();

        _dgemm_full(&ovlp_full, 'N', &dm[0], 'N', &mut tmp_matr_a, 1.0, 0.0);
        _dgemm_full(&dm[1], 'N', &tmp_matr_a, 'N', &mut tmp_matr_b, 1.0, 0.0);
        _dgemm_full(&ovlp_full, 'N', &tmp_matr_b, 'N', &mut tmp_matr_a, 1.0, 0.0);

        s2 -= tmp_matr_a.iter_diagonal().unwrap().fold(0.0, |acc, x| acc+x);

        mlpy = (1.0 + 4.0* s2).powf(0.5);

    }

    [s2, mlpy]

}



#[test]
fn test_max() {
    println!("{}, {}, {}",1,2,std::cmp::max(1, 2));
}
