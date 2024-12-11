#[cfg(feature = "cuda")]
mod ump2_cuda {
    use crate::scf_io::SCF;
    use candle_core::IndexOp;

    pub fn kernel_open_shell_mp2_gpu_tf32(
        cderi_ovu: &[ndarray::ArrayView3<f64>],
        occ_energy: &[ndarray::ArrayView1<f64>],
        vir_energy: &[ndarray::ArrayView1<f64>],
        occ_occupation: Option<&[ndarray::ArrayView1<f64>]>,
        vir_occupation: Option<&[ndarray::ArrayView1<f64>]>,
        batch_occ: Option<usize>,
    ) -> anyhow::Result<[f64; 3]> {
        use candle_core::{DType, Device, IndexOp, Tensor};
        use ndarray::prelude::*;

        // set reduced precision
        let reduced_precision = candle_core::cuda::gemm_reduced_precision_f32();
        candle_core::cuda::set_gemm_reduced_precision_f32(true);

        // dimension specification and sanity check
        assert!(cderi_ovu.len() == 2);
        assert!(occ_energy.len() == 2);
        assert!(vir_energy.len() == 2);
        let (nocc_a, nvir_a, naux_a) = cderi_ovu[0].dim();
        let (nocc_b, nvir_b, naux_b) = cderi_ovu[1].dim();
        assert!(naux_a == naux_b);
        let nocc = [nocc_a, nocc_b];
        let nvir = [nvir_a, nvir_b];
        let naux = naux_a;
        let batch_occ = batch_occ.unwrap_or(nocc_a.max(nocc_b));
        let is_frac_occ = occ_occupation.is_some() && vir_occupation.is_some();

        let device = Device::new_cuda(0)?;
        let mut e_corr_mp2 = vec![Tensor::zeros(&[], DType::F64, &device)?; 3];

        // [spin_ia, spin_jb, spin_ijab]
        let spins_iter_indices = [[0, 0, 0], [1, 1, 1], [0, 1, 2]];
        for [spin_i, spin_j, spin_ij] in spins_iter_indices {
            let same_spin = spin_i == spin_j;

            // virtual orbital energy denominator
            let d_vv = {
                let d_vv = -&vir_energy[spin_i].view().insert_axis(Axis(1)) - &vir_energy[spin_j].view().insert_axis(Axis(0));
                let d_vv = d_vv.as_standard_layout().to_owned();
                Tensor::from_slice(d_vv.as_slice().unwrap(), (nvir[spin_i], nvir[spin_j]), &device)?.to_dtype(DType::F32)?
            };

            // virtual occupation multiplier
            let n_vv = match is_frac_occ {
                true => {
                    let frac_occ_a = -&vir_occupation.unwrap()[spin_i] + 1.0;
                    let frac_occ_b = -&vir_occupation.unwrap()[spin_j] + 1.0;
                    let n_vv = &frac_occ_a.view().insert_axis(Axis(1)) * &frac_occ_b.view().insert_axis(Axis(0));
                    let n_vv = n_vv.as_standard_layout().to_owned();
                    Some(Tensor::from_slice(n_vv.as_slice().unwrap(), (nvir[spin_i], nvir[spin_j]), &device)?.to_dtype(DType::F32)?)
                },
                false => None,
            };

            // batch processing (i, j) of occupied orbitals
            let ptr_i_max = nocc[spin_i];
            for ptr_i in (0..ptr_i_max).step_by(batch_occ) {
                let nbatch_i = std::cmp::min(batch_occ, ptr_i_max - ptr_i);
                let cderi_ovu_batch_i = {
                    let tsr = cderi_ovu[spin_i].slice(s![ptr_i..ptr_i + nbatch_i, .., ..]);
                    let tsr = tsr.as_standard_layout(); // assure c-contiguous order
                    let tsr = tsr.as_slice().unwrap();
                    Tensor::from_slice(tsr, (nbatch_i, nvir[spin_i], naux), &device)?.to_dtype(DType::F32)?
                };
                // ptr_j will be half-batched if same-spin, otherwise full-batched
                let ptr_j_max = if same_spin { ptr_i + nbatch_i } else { nocc[spin_j] };
                for ptr_j in (0..ptr_j_max).step_by(batch_occ) {
                    let nbatch_j = std::cmp::min(batch_occ, ptr_j_max - ptr_j);
                    let cderi_ovu_batch_j = if (ptr_j == ptr_i && same_spin) {
                        &cderi_ovu_batch_i
                    } else {
                        let tsr = cderi_ovu[spin_j].slice(s![ptr_j..ptr_j + nbatch_j, .., ..]);
                        let tsr = tsr.as_standard_layout(); // assure c-contiguous order
                        let tsr = tsr.as_slice().unwrap();
                        &Tensor::from_slice(tsr, (nbatch_j, nvir[spin_j], naux), &device)?.to_dtype(DType::F32)?
                    };

                    // compute MP2 energy for each pair of (i, j)
                    for i in ptr_i..ptr_i + nbatch_i {
                        let j_max = if same_spin { (ptr_j + nbatch_j).min(i + 1) } else { ptr_j + nbatch_j };
                        for j in ptr_j..j_max {
                            // matmul part of MP2 energy
                            let cderi_ivu = cderi_ovu_batch_i.i(i - ptr_i)?;
                            let cderi_jvu = cderi_ovu_batch_j.i(j - ptr_j)?;
                            let g_ab = cderi_ivu.matmul(&cderi_jvu.transpose(0, 1)?)?;

                            // asymmetrize if same spin
                            let g_ab = match same_spin {
                                true => (&g_ab - &g_ab.transpose(0, 1)?)?,
                                false => g_ab,
                            };

                            // dominator and multiplier part of MP2 energy
                            let e_i = occ_energy[spin_i][i] as f64;
                            let e_j = occ_energy[spin_j][j] as f64;
                            let d_ab = (e_i + e_j + &d_vv)?;

                            let d_ab = match is_frac_occ {
                                true => {
                                    let n_i = occ_occupation.unwrap()[spin_i][i] as f64;
                                    let n_j = occ_occupation.unwrap()[spin_j][j] as f64;
                                    let n_ab = (n_i * n_j * n_vv.as_ref().unwrap())?;
                                    (&d_ab / &n_ab)?
                                },
                                false => d_ab,
                            };
                            let t_ab = (&g_ab / &d_ab)?;

                            let factor = match same_spin {
                                true => match i == j {
                                    true => 1.0,
                                    false => 0.5,
                                },
                                false => 1.0,
                            };

                            let e_corr_increment = (factor * (&g_ab * &t_ab)?.sum_all()?.to_dtype(DType::F64)?)?;
                            e_corr_mp2[spin_ij] = (&e_corr_mp2[spin_ij] + e_corr_increment)?;
                        }
                    }
                }
            }
        }

        // collect MP2 energy into CPU
        let e_corr_mp2 = e_corr_mp2.into_iter().map(|t| t.to_scalar::<f64>().unwrap()).collect::<Vec<f64>>();
        let e_os = e_corr_mp2[2];
        let e_ss = e_corr_mp2[0] + e_corr_mp2[1];

        // restore reduced precision
        candle_core::cuda::set_gemm_reduced_precision_f32(reduced_precision);

        // return MP2 energy
        return Ok([e_os + e_ss, e_os, e_ss]);
    }

    pub fn open_shell_pt2_cuda(scf_data: &SCF) -> anyhow::Result<[f64; 3]> {
        // TODO:
        // - GPU before ri3mo
        // - `batch_occ` fine grain control
        // - `cderi_ovu` to be contiguous
        //      (currently redundant ao2mo is performed in generate_ri3mo_rayon, which affects contiguous of cderi_ovu if exactly truncates)

        use ndarray::prelude::*;

        // get required data for MP2 calculation
        if scf_data.ri3mo.is_none() {
            panic!("RI3MO should be initialized before the PT2 calculations")
        }

        let ri3mo_vec = scf_data.ri3mo.as_ref().unwrap();
        let mut cderi_ovu = vec![];
        let mut cderi_ovu_truncated = vec![];
        let mut occ_energy = vec![];
        let mut vir_energy = vec![];
        let mut occ_occupation = vec![];
        let mut vir_occupation = vec![];

        // for borrow rule, we must store cderi_ovu views first, then handle sliceing
        for spin in [0, 1] {
            let (rimo, _, _) = &ri3mo_vec[spin];
            let mut cderi_ovu_shape = rimo.size.clone();
            cderi_ovu_shape.reverse();
            let cderi_ovu_spin = ArrayView3::from_shape(cderi_ovu_shape, &rimo.data)?;
            cderi_ovu.push(cderi_ovu_spin);
        }

        for spin in [0, 1] {
            // truncates ri3mo for each spin
            let homo = scf_data.homo[spin];
            let lumo = scf_data.lumo[spin];
            let num_occu = if scf_data.mol.num_elec[spin + 1] <= 1.0e-6 { 0 } else { homo + 1 };
            let start_mo = scf_data.mol.start_mo;
            let num_state = scf_data.mol.num_state;
            // essential inputs
            let (rimo, vir_range, occ_range) = &ri3mo_vec[spin];
            let actual_occ_range = start_mo..num_occu;
            let actual_vir_range = lumo..num_state;
            let eigenvalues = &scf_data.eigenvalues[spin];
            let occ_energy_spin = ArrayView1::from(&eigenvalues[actual_occ_range.clone()]);
            let vir_energy_spin = ArrayView1::from(&eigenvalues[actual_vir_range.clone()]);
            // truncate cderi_ovu to correct occ-vir configuration
            let slc_truncate = s![actual_occ_range.start..actual_occ_range.end, (actual_vir_range.start - vir_range.start..actual_vir_range.end - vir_range.start), ..];
            let cderi_ovu_spin_truncated = cderi_ovu[spin].slice(slc_truncate);

            cderi_ovu_truncated.push(cderi_ovu_spin_truncated);
            occ_energy.push(occ_energy_spin);
            vir_energy.push(vir_energy_spin);

            // fractional occupation number handling
            let occupation = &scf_data.occupation[spin];
            let occ_occupation_spin = ArrayView1::from(&occupation[actual_occ_range.clone()]);
            let vir_occupation_spin = ArrayView1::from(&occupation[actual_vir_range.clone()]);
            occ_occupation.push(occ_occupation_spin);
            vir_occupation.push(vir_occupation_spin);
        }
        let occ_occupation_view = &occ_occupation.iter().map(|x| x.view()).collect::<Vec<_>>();
        let vir_occupation_view = &vir_occupation.iter().map(|x| x.view()).collect::<Vec<_>>();

        let result = kernel_open_shell_mp2_gpu_tf32(&cderi_ovu_truncated, &occ_energy, &vir_energy, Some(occ_occupation_view), Some(vir_occupation_view), Some(32))?;
        return Ok(result);
    }
}

#[cfg(not(feature = "cuda"))]
mod ump2_cuda {
    use crate::scf_io::SCF;

    pub fn open_shell_pt2_cuda(scf_data: &SCF) -> anyhow::Result<[f64; 3]> {
        panic!("CUDA support is not enabled")
    }
}

pub use ump2_cuda::*;

#[cfg(test)]
mod debug_h2o {
    use super::*;
    use crate::ctrl_io::InputKeywords;
    use crate::molecule_io::Molecule;
    use crate::ri_pt2::open_shell_pt2_rayon;
    use crate::scf_io::{self, determine_ri3mo_size_for_pt2_and_rpa, scf_without_build, SCF};

    #[test]
    fn test_usual_case() {
        let mut scf_data = initialize_h2o();
        let (occ_range, vir_range) = determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);
        let result_cpu = open_shell_pt2_rayon(&scf_data).unwrap();
        let result_gpu = open_shell_pt2_cuda(&scf_data).unwrap();
        println!("{:?}", result_cpu);
        println!("{:?}", result_gpu);
        // check the same result from CPU and GPU
        assert!((result_cpu[0] - result_gpu[0]).abs() < 1e-6);
        assert!((result_cpu[1] - result_gpu[1]).abs() < 1e-6);
        assert!((result_cpu[2] - result_gpu[2]).abs() < 1e-6);
        // check absolute value
        assert!((result_gpu[0] - -0.21227236266306992).abs() < 1e-6);
        assert!((result_gpu[1] - -0.16586930450532872).abs() < 1e-6);
        assert!((result_gpu[2] - -0.04640305815774121).abs() < 1e-6);
    }

    #[test]
    fn test_frac_occ() {
        let mut scf_data = initialize_h2o();
        // assigning fake occupation numbers
        let fake_occ = vec![1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1];
        let fake_occ = vec![1.7, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1, 0.5];
        for i in 0..fake_occ.len() {
            scf_data.occupation[0][i] = fake_occ[i];
        }
        let (occ_range, vir_range) = determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);
        let result_cpu = open_shell_pt2_rayon(&scf_data).unwrap();
        let result_gpu = open_shell_pt2_cuda(&scf_data).unwrap();
        println!("{:?}", result_cpu);
        println!("{:?}", result_gpu);
        // check the same result from CPU and GPU
        assert!((result_cpu[0] - result_gpu[0]).abs() < 1e-6);
        assert!((result_cpu[1] - result_gpu[1]).abs() < 1e-6);
        assert!((result_cpu[2] - result_gpu[2]).abs() < 1e-6);
        // check absolute value
        assert!((result_gpu[0] - -0.23616002126608224).abs() < 1e-6);
        assert!((result_gpu[1] - -0.18596335283378365).abs() < 1e-6);
        assert!((result_gpu[2] - -0.05019666843229859).abs() < 1e-6);
    }

    fn initialize_h2o() -> SCF {
        let input_token = r##"
            [ctrl]
                print_level =          2
                xc =                   "mp2"
                basis_path =           "basis-set-pool/def2-TZVP"
                auxbas_path =          "basis-set-pool/def2-SVP-JKFIT"
                basis_type =           "spheric"
                eri_type =             "ri-v"
                auxbas_type =          "spheric"
                charge =               1.0
                spin =                 2.0
                spin_polarization =    true
                num_threads =          4

            [geom]
                name = "H2O"
                unit = "Angstrom"
                position = """
                    O  0.0  0.0  0.0
                    H  0.0  0.0  1.0
                    H  0.0  1.0  0.0
                """
            "##;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (mut ctrl, mut geom) = InputKeywords::parse_ctl_from_json(&keys).unwrap();
        let mol = Molecule::build_native(ctrl, geom).unwrap();
        let mut scf_data = scf_io::SCF::build(mol);
        scf_without_build(&mut scf_data);
        return scf_data;
    }
}
