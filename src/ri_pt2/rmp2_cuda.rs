#[cfg(feature = "cuda")]
mod rmp2_cuda {
    use crate::scf_io::SCF;

    /// Compute RMP2 energy using CUDA with reduced precision.
    ///
    /// # Arguments
    ///
    /// * `cderi_ovu`: Cholesky decomposed ERIs in the MO basis.
    ///     Shape: (nocc, nvir, naux), c-contiguous order
    /// * `occ_energy`: Occupied orbital energies.
    /// * `vir_energy`: Virtual orbital energies.
    /// * `occ_occupation`: Fractional occupation of occupied orbitals. This occupation number should be renormalized to maximum 1 for each orbital.
    /// * `vir_occupation`: Fractional occupation of virtual orbitals. This occupation number should be renormalized to maximum 1 for each orbital.
    /// * `batch_occ`: Number of occupied orbitals to process in a single batch. Default is number of occupied orbitals (all orbitals are read into GPU in one batch).
    pub fn kernel_close_shell_mp2_gpu_tf32(
        cderi_ovu: ndarray::ArrayView3<f64>,
        occ_energy: ndarray::ArrayView1<f64>,
        vir_energy: ndarray::ArrayView1<f64>,
        occ_occupation: Option<ndarray::ArrayView1<f64>>,
        vir_occupation: Option<ndarray::ArrayView1<f64>>,
        batch_occ: Option<usize>,
    ) -> anyhow::Result<[f64; 3]> {
        use candle_core::{DType, Device, IndexOp, Tensor};
        use ndarray::prelude::*;

        // set reduced precision
        let reduced_precision = candle_core::cuda::gemm_reduced_precision_f32();
        candle_core::cuda::set_gemm_reduced_precision_f32(true);

        // dimension specification and sanity check
        let (nocc, nvir, naux) = cderi_ovu.dim();
        assert_eq!(nocc, occ_energy.len());
        assert_eq!(nvir, vir_energy.len());
        let batch_occ = batch_occ.unwrap_or(nocc);

        let device = Device::new_cuda(0)?;

        // virtual orbital energy denominator
        let d_vv = {
            let d_vv = -&vir_energy.view().insert_axis(Axis(1)) - &vir_energy.view().insert_axis(Axis(0));
            let d_vv = d_vv.as_standard_layout().to_owned();
            Tensor::from_slice(d_vv.as_slice().unwrap(), (nvir, nvir), &device)?.to_dtype(DType::F32)?
        };

        // virtual occupation multiplier
        let is_frac_occ = occ_occupation.is_some() && vir_occupation.is_some();
        let n_vv = match is_frac_occ {
            true => {
                let tmp = -&vir_occupation.unwrap() + 1.0;
                let n_vv = &tmp.view().insert_axis(Axis(1)) * &tmp.view().insert_axis(Axis(0));
                let n_vv = n_vv.as_standard_layout().to_owned();
                Some(Tensor::from_slice(n_vv.as_slice().unwrap(), (nvir, nvir), &device)?.to_dtype(DType::F32)?)
            },
            false => None,
        };

        // two bi-orthogonal parts of MP2 energy
        let mut e_bi1 = Tensor::zeros(&[], DType::F64, &device)?;
        let mut e_bi2 = Tensor::zeros(&[], DType::F64, &device)?;

        // batch processing (i, j) of occupied orbitals
        for ptr_i in (0..nocc).step_by(batch_occ) {
            let nbatch_i = std::cmp::min(batch_occ, nocc - ptr_i);
            let cderi_ovu_batch_i = {
                let tsr = cderi_ovu.slice(s![ptr_i..ptr_i + nbatch_i, .., ..]);
                let tsr = tsr.as_standard_layout(); // assure c-contiguous order
                let tsr = tsr.as_slice().unwrap();
                Tensor::from_slice(tsr, (nbatch_i, nvir, naux), &device)?.to_dtype(DType::F32)?
            };

            for ptr_j in (0..ptr_i + nbatch_i).step_by(batch_occ) {
                let nbatch_j = std::cmp::min(batch_occ, ptr_i + nbatch_i - ptr_j);
                let cderi_ovu_batch_j = if (ptr_j == ptr_i) {
                    &cderi_ovu_batch_i
                } else {
                    let tsr = cderi_ovu.slice(s![ptr_j..ptr_j + nbatch_j, .., ..]);
                    let tsr = tsr.as_standard_layout(); // assure c-contiguous order
                    let tsr = tsr.as_slice().unwrap();
                    &Tensor::from_slice(tsr, (nbatch_j, nvir, naux), &device)?.to_dtype(DType::F32)?
                };

                // compute MP2 energy for each pair of (i, j)
                for i in ptr_i..ptr_i + nbatch_i {
                    for j in ptr_j..std::cmp::min(ptr_j + nbatch_j, i + 1) {
                        // matmul part of MP2 energy
                        let cderi_ivu = cderi_ovu_batch_i.i(i - ptr_i)?;
                        let cderi_jvu = cderi_ovu_batch_j.i(j - ptr_j)?;
                        let g_ab = cderi_ivu.matmul(&cderi_jvu.transpose(0, 1)?)?;

                        // denominator and multiplier part of MP2 energy
                        let e_i = occ_energy[i] as f64;
                        let e_j = occ_energy[j] as f64;
                        let d_ab = (e_i + e_j + &d_vv)?;

                        let d_ab = match is_frac_occ {
                            true => {
                                let n_i = occ_occupation.as_ref().unwrap()[i];
                                let n_j = occ_occupation.as_ref().unwrap()[j];
                                let n_ab = (n_i * n_j * n_vv.as_ref().unwrap())?;
                                (&d_ab / &n_ab)?
                            },
                            false => d_ab,
                        };

                        let t_ab = (&g_ab / d_ab)?;
                        let factor = if i != j { 2.0 } else { 1.0 };

                        // accumulate MP2 energy
                        e_bi1 = (e_bi1 + (factor * (&g_ab * &t_ab)?.sum_all()?)?.to_dtype(DType::F64)?)?;
                        e_bi2 = (e_bi2 + (factor * (&g_ab * &t_ab.transpose(0, 1)?)?.sum_all()?.to_dtype(DType::F64)?)?)?;
                    }
                }
            }
        }

        // get MP2 energy from GPU tensor
        let e_bi1 = e_bi1.to_scalar::<f64>()?;
        let e_bi2 = e_bi2.to_scalar::<f64>()?;
        let e_os = e_bi1;
        let e_ss = e_bi1 - e_bi2;

        // restore reduced precision
        candle_core::cuda::set_gemm_reduced_precision_f32(reduced_precision);

        return Ok([e_os + e_ss, e_os, e_ss]);
    }

    pub fn close_shell_pt2_cuda(scf_data: &SCF) -> anyhow::Result<[f64; 3]> {
        // TODO:
        // - GPU before ri3mo
        // - `batch_occ` fine grain control

        use ndarray::prelude::*;

        // get required data for MP2 calculation
        if scf_data.ri3mo.is_none() {
            panic!("RI3MO should be initialized before the PT2 calculations")
        }

        let ri3mo_vec = scf_data.ri3mo.as_ref().unwrap();
        let (rimo, vir_range, occ_range) = &ri3mo_vec[0];
        let eigenvalues = &scf_data.eigenvalues[0];

        // essential inputs
        let occ_energy = ArrayView1::from(&eigenvalues[occ_range.clone()]);
        let vir_energy = ArrayView1::from(&eigenvalues[vir_range.clone()]);
        let mut cderi_ovu_shape = rimo.size.clone();
        cderi_ovu_shape.reverse(); // fortran-to-c order
        let cderi_ovu = ArrayView3::from_shape(cderi_ovu_shape, &rimo.data)?;

        // fractional occupation number handling
        let occupation = &scf_data.occupation[0];
        let occ_occupation = &ArrayView1::from(&occupation[occ_range.clone()]) / 2.0;
        let vir_occupation = &ArrayView1::from(&occupation[vir_range.clone()]) / 2.0;
        let occ_occupation_view = Some(occ_occupation.view());
        let vir_occupation_view = Some(vir_occupation.view());

        let result = kernel_close_shell_mp2_gpu_tf32(cderi_ovu, occ_energy, vir_energy, occ_occupation_view, vir_occupation_view, Some(32))?;
        return Ok(result);
    }
}

#[cfg(not(feature = "cuda"))]
mod rmp2_cuda {
    use crate::scf_io::SCF;

    pub fn close_shell_pt2_cuda(scf_data: &SCF) -> anyhow::Result<[f64; 3]> {
        panic!("CUDA support is not enabled")
    }
}

pub use rmp2_cuda::*;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod debug_nh3 {
    use super::*;
    use crate::ctrl_io::InputKeywords;
    use crate::molecule_io::Molecule;
    use crate::ri_pt2::close_shell_pt2_rayon;
    use crate::scf_io::{self, determine_ri3mo_size_for_pt2_and_rpa, scf_without_build, SCF};

    #[test]
    fn test_usual_case() {
        let mut scf_data = initialize_nh3();
        let (occ_range, vir_range) = determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);
        let result_cpu = close_shell_pt2_rayon(&scf_data).unwrap();
        let result_gpu = close_shell_pt2_cuda(&scf_data).unwrap();
        println!("{:?}", result_cpu);
        println!("{:?}", result_gpu);
        // check the same result from CPU and GPU
        assert!((result_cpu[0] - result_gpu[0]).abs() < 1e-6);
        assert!((result_cpu[1] - result_gpu[1]).abs() < 1e-6);
        assert!((result_cpu[2] - result_gpu[2]).abs() < 1e-6);
        // check absolute value
        assert!((result_gpu[0] - -0.31670803412052667).abs() < 1e-6);
        assert!((result_gpu[1] - -0.24939630820846492).abs() < 1e-6);
        assert!((result_gpu[2] - -0.06731172591206175).abs() < 1e-6);
    }

    #[test]
    fn test_frac_occ() {
        let mut scf_data = initialize_nh3();
        // assigning fake occupation numbers
        let fake_occ = vec![1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1];
        for i in 0..fake_occ.len() {
            scf_data.occupation[0][i] = fake_occ[i];
        }
        let (occ_range, vir_range) = determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);
        let result_cpu = close_shell_pt2_rayon(&scf_data).unwrap();
        let result_gpu = close_shell_pt2_cuda(&scf_data).unwrap();
        println!("{:?}", result_cpu);
        println!("{:?}", result_gpu);
        // check the same result from CPU and GPU
        assert!((result_cpu[0] - result_gpu[0]).abs() < 1e-6);
        assert!((result_cpu[1] - result_gpu[1]).abs() < 1e-6);
        assert!((result_cpu[2] - result_gpu[2]).abs() < 1e-6);
        // check absolute value
        assert!((result_gpu[0] - -0.10070463208846368).abs() < 1e-6);
        assert!((result_gpu[1] - -0.08139191909431057).abs() < 1e-6);
        assert!((result_gpu[2] - -0.01931271299415312).abs() < 1e-6);
    }

    fn initialize_nh3() -> SCF {
        let input_token = r##"
            [ctrl]
                print_level =          2
                xc =                   "mp2"
                basis_path =           "basis-set-pool/def2-TZVP"
                auxbas_path =          "basis-set-pool/def2-SVP-JKFIT"
                basis_type =           "spheric"
                eri_type =             "ri-v"
                auxbas_type =          "spheric"
                charge =               0.0
                spin =                 1.0
                spin_polarization =    false
                num_threads =          4

            [geom]
                name = "NH3"
                unit = "Angstrom"
                position = """
                    N  0.0  0.0  0.0
                    H  0.0  1.5  1.0
                    H  1.4  1.1  0.0
                    H  1.2  0.0  1.3
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

#[cfg(test)]
#[cfg(feature = "cuda")]
mod debug_c20h42 {
    use super::*;
    use crate::ctrl_io::InputKeywords;
    use crate::molecule_io::Molecule;
    use crate::ri_pt2::close_shell_pt2_rayon;
    use crate::scf_io::{self, determine_ri3mo_size_for_pt2_and_rpa};
    use crate::scf_io::{scf_without_build, SCF};
    use std::time::Instant;

    #[test]
    #[ignore = "Stress test"]
    fn test() {
        let start = Instant::now();
        let mut scf_data = initialize_c20h42();
        println!("Elapsed time (SCF): {:?}", start.elapsed());

        let start = Instant::now();
        let (occ_range, vir_range) = determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);
        println!("Elapsed time (RI3MO): {:?}", start.elapsed());

        let start = Instant::now();
        let result = close_shell_pt2_cuda(&scf_data).unwrap();
        println!("{:?}", result);
        println!("Elapsed time (MP2 cuda): {:?}", start.elapsed());

        let start = Instant::now();
        let result = close_shell_pt2_rayon(&scf_data).unwrap();
        println!("{:?}", result);
        println!("Elapsed time (MP2 rayon): {:?}", start.elapsed());

        // timing results
        // Elapsed time (RI3MO): 5.815747779s
        // Elapsed time (MP2 cuda): 4.547107079s
        // Elapsed time (MP2 rayon): 12.716702551s
    }

    fn initialize_c20h42() -> SCF {
        let input_token = r##"
            [ctrl]
                print_level =          2
                xc =                   "mp2"
                basis_path =           "basis-set-pool/def2-TZVP"
                auxbas_path =          "basis-set-pool/def2-SVP-JKFIT"
                eri_type =             "ri-v"
                charge =               0.0
                spin =                 1.0
                spin_polarization =    false
                initial_guess =        "sad"
                num_threads =          16

            [geom]
                name = "C20H42"
                unit = "Angstrom"
                position = """
                    C          0.92001        0.06420        0.06012
                    C          2.43575        0.07223        0.05487
                    C          2.95953        0.07735       -1.37280
                    C          4.47850        0.08347       -1.38184
                    C          4.99051        0.09045       -2.81325
                    C          6.50633        0.09509       -2.82168
                    C          7.01829        0.10275       -4.24936
                    C          8.53627        0.10527       -4.25456
                    C          9.04692        0.11614       -5.68273
                    C         10.56346        0.11661       -5.69047
                    C         11.07492        0.12997       -7.11418
                    C         12.59092        0.13132       -7.12186
                    C         13.10157        0.14462       -8.55008
                    C         14.61964        0.14607       -8.55548
                    C         15.13124        0.16035       -9.98248
                    C         16.64733        0.16206       -9.99128
                    C         17.15935        0.17513      -11.42267
                    C         18.67833        0.17723      -11.43172
                    C         19.20209        0.19029      -12.85934
                    C         20.71785        0.19235      -12.86461
                    H          0.54201        0.06057        1.08613
                    H          0.53500       -0.82407       -0.45173
                    H          0.52556        0.94987       -0.44900
                    H          2.80958       -0.81016        0.58802
                    H          2.80026        0.95682        0.59082
                    H          2.58241        0.95886       -1.90489
                    H          2.58965       -0.80493       -1.90864
                    H          4.85987       -0.80023       -0.85536
                    H          4.85295        0.96646       -0.84930
                    H          4.61266        0.97436       -3.34144
                    H          4.61753       -0.79132       -3.34848
                    H          6.88668       -0.78995       -2.29450
                    H          6.88220        0.97538       -2.28518
                    H          6.64237        0.98886       -4.77569
                    H          6.64440       -0.77705       -4.78613
                    H          8.91291       -0.78065       -3.72931
                    H          8.91067        0.98489       -3.71692
                    H          8.67103        1.00375       -6.20773
                    H          8.67135       -0.76270       -6.22160
                    H         10.93921       -0.77082       -5.16486
                    H         10.93901        0.99501       -5.14890
                    H         10.69770        1.01663       -7.64012
                    H         10.69983       -0.74916       -7.65530
                    H         12.96781       -0.75568       -6.59750
                    H         12.96549        1.01075       -6.58231
                    H         12.72473        1.03172       -9.07390
                    H         12.72689       -0.73395       -9.08896
                    H         14.99584       -0.74031       -8.03132
                    H         14.99314        1.02558       -8.01607
                    H         14.75220        1.04672      -10.50702
                    H         14.75461       -0.71855      -10.52233
                    H         17.02394       -0.72486       -9.46726
                    H         17.02159        1.04075       -9.45197
                    H         16.78025        1.06253      -11.94450
                    H         16.78260       -0.70411      -11.95980
                    H         19.05301       -0.70877      -10.90541
                    H         19.05066        1.05497      -10.89014
                    H         18.83173        1.07790      -13.38621
                    H         18.83408       -0.68904      -13.40152
                    H         21.10641        1.07540      -12.34646
                    H         21.09584        0.20174      -13.89058
                    H         21.10877       -0.69850      -12.36183
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
