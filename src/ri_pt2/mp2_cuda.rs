#[cfg(feature = "cuda")]
mod mp2_cuda {
    use candle_core::IndexOp;
    use crate::scf_io::SCF;
    
    /// Compute RMP2 energy using CUDA with reduced precision.
    pub fn kernel_close_shell_mp2_gpu_tf32(
        cderi_ovu: ndarray::ArrayView3::<f64>,
        occ_energy: ndarray::ArrayView1::<f64>,
        vir_energy: ndarray::ArrayView1::<f64>,
        batch_occ: Option<usize>,
    ) -> anyhow::Result<[f64; 3]> {
        use candle_core::{Device, DType, Tensor};
        use ndarray::prelude::*;
    
        // set reduced precision
        let reduced_precision = candle_core::cuda::gemm_reduced_precision_f32();
        candle_core::cuda::set_gemm_reduced_precision_f32(true);
    
        let (nocc, nvir, naux) = cderi_ovu.dim();
        assert_eq!(nocc, occ_energy.len());
        assert_eq!(nvir, vir_energy.len());
        let batch_occ = batch_occ.unwrap_or(nocc);
    
        let device = Device::new_cuda(0)?;
        let d_vv = - &vir_energy.clone().insert_axis(Axis(0)) - &vir_energy.clone().insert_axis(Axis(1));
        let d_vv = d_vv.as_standard_layout().to_owned();
        let d_vv_device = Tensor::from_slice(d_vv.as_slice().unwrap(), (nvir, nvir), &device)?.to_dtype(DType::F32)?;
    
        let mut e_bi1 = Tensor::zeros(&[], DType::F64, &device)?;
        let mut e_bi2 = Tensor::zeros(&[], DType::F64, &device)?;
        
        for ptr_i in (0..nocc).step_by(batch_occ) {
            let nbatch_i = std::cmp::min(batch_occ, nocc - ptr_i);
            let cderi_ovu_batch_i = Tensor::from_slice(
                cderi_ovu.slice(s![ptr_i..ptr_i+nbatch_i, .., ..]).as_slice().unwrap(),
                (nbatch_i, nvir, naux), &device
            )?.to_dtype(DType::F32)?;
    
            for ptr_j in (0..ptr_i+nbatch_i).step_by(batch_occ) {
                let nbatch_j = std::cmp::min(batch_occ, ptr_i + nbatch_i - ptr_j);
                let cderi_ovu_batch_j = Tensor::from_slice(
                    cderi_ovu.slice(s![ptr_j..ptr_j+nbatch_j, .., ..]).as_slice().unwrap(),
                    (nbatch_j, nvir, naux), &device
                )?.to_dtype(DType::F32)?;
    
                for i in ptr_i .. ptr_i+nbatch_i {
                    for j in ptr_j .. std::cmp::min(ptr_j + nbatch_j, i + 1) {
                        let cderi_ivu = cderi_ovu_batch_i.i(i - ptr_i)?;
                        let cderi_jvu = cderi_ovu_batch_j.i(j - ptr_j)?;
                        let g_ab = cderi_ivu.matmul(&cderi_jvu.transpose(0, 1)?)?;
                        let e_i = occ_energy[i] as f64;
                        let e_j = occ_energy[j] as f64;
                        let d_ab = ((&d_vv_device + e_i)? + e_j)?;
                        let t_ab = (&g_ab / d_ab)?;
                        let factor = if i != j { 2.0 } else { 1.0 };
    
                        e_bi1 = (e_bi1 + (factor * (&g_ab * &t_ab)?.sum_all()?)?.to_dtype(DType::F64)?)?;
                        e_bi2 = (e_bi2 + (factor * (&g_ab * &t_ab.transpose(0, 1)?)?.sum_all()?.to_dtype(DType::F64)?)?)?;
                    }
                }
            }
        }
        
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
        // 1. fraction occupation MP2
        // 2. GPU before ri3mo
        // 3. various workflow control options
        // 4. `batch_occ` fine grain control

        use ndarray::s;

        // get required data for MP2 calculation
        if let Some(ri3mo_vec) = &scf_data.ri3mo {
            let (rimo, vir_range, occ_range) = &ri3mo_vec[0];

            let eigenvector = scf_data.eigenvectors.get(0).unwrap();
            let eigenvalues = scf_data.eigenvalues.get(0).unwrap();
            
            let occ_energy = ndarray::ArrayView1::from(&eigenvalues[occ_range.clone()]);
            let vir_energy = ndarray::ArrayView1::from(&eigenvalues[vir_range.clone()]);
            let mut cderi_ovu_shape = rimo.size.clone();
            cderi_ovu_shape.reverse();  // fortran-to-c order
            let cderi_ovu = ndarray::ArrayView3::from_shape(cderi_ovu_shape, &rimo.data)?;
            let result = kernel_close_shell_mp2_gpu_tf32(cderi_ovu, occ_energy, vir_energy, Some(32))?;
            return Ok(result);
        } else {
            panic!("RI3MO should be initialized before the PT2 calculations")
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod mp2_cuda {
    pub fn close_shell_pt2_cuda(scf_data: &SCF) -> anyhow::Result<[f64; 3]> {
        panic!("CUDA support is not enabled")
    }
}

pub use mp2_cuda::*;

#[cfg(test)]
mod debug_nh3 {
    use super::*;
    use crate::ctrl_io::InputKeywords;
    use crate::molecule_io::Molecule;
    use crate::ri_pt2::close_shell_pt2_rayon;
    use crate::scf_io::{self, determine_ri3mo_size_for_pt2_and_rpa};
    use crate::scf_io::{scf_without_build, SCF};

    #[test]
    fn test() {
        let mut scf_data = initialize_nh3();
        let (occ_range, vir_range) = determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);
        let result = close_shell_pt2_cuda(&scf_data).unwrap();
        println!("{:?}", result);
        let result = close_shell_pt2_rayon(&scf_data).unwrap();
        println!("{:?}", result);
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
     guessfile =            "none"
     chkfile =              "none"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     external_grids =       "none"
     initial_guess=         "sad"
     mixer =                "diis"
     num_max_diis =         8
     start_diis_cycle =     3
     mix_param =            0.8
     max_scf_cycle =        100
     scf_acc_rho =          1.0e-10
     scf_acc_eev =          1.0e-10
     scf_acc_etot =         1.0e-11
     num_threads =          16

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
mod debug_c12h26 {
    use super::*;
    use std::time::Instant;
    use crate::ctrl_io::InputKeywords;
    use crate::molecule_io::Molecule;
    use crate::ri_pt2::close_shell_pt2_rayon;
    use crate::scf_io::{self, determine_ri3mo_size_for_pt2_and_rpa};
    use crate::scf_io::{scf_without_build, SCF};

    #[test]
    fn test() {
        let start = Instant::now();
        let mut scf_data = initialize_c12h26();
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

        /*
            Tested on Rayon 7945HX (16 threads) + RTX 4060

            Elapsed time (RI3MO): 1.246392849s
            [-2.254752243251664, -1.768394118171839, -0.4863581250798248]
            Elapsed time (MP2 cuda): 834.194864ms
            [-2.2547687994450776, -1.7684088667125075, -0.48635993273257017]
            Elapsed time (MP2 rayon): 7.106851207s
         */
    }

    fn initialize_c12h26() -> SCF {
        let input_token = r##"
[ctrl]
     print_level =          2
     xc =                   "mp2"
     basis_path =           "basis-set-pool/def2-TZVP"
     auxbas_path =          "basis-set-pool/def2-SVP-JKFIT"
     basis_type =           "spheric"
     eri_type =             "ri-v"
     auxbas_type =          "spheric"
     guessfile =            "none"
     chkfile =              "none"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     external_grids =       "none"
     initial_guess=         "sad"
     mixer =                "diis"
     num_max_diis =         8
     start_diis_cycle =     3
     mix_param =            0.8
     max_scf_cycle =        100
     scf_acc_rho =          1.0e-6
     scf_acc_eev =          1.0e-9
     scf_acc_etot =         1.0e-11
     num_threads =          16

[geom]
    name = "C12H26"
    unit = "Angstrom"
    position = """
        C          0.99590        0.00874        0.02912
        C          2.51497        0.01491        0.04092
        C          3.05233        0.96529        1.10755
        C          4.57887        0.97424        1.12090
        C          5.10652        1.92605        2.19141
        C          6.63201        1.93944        2.20819
        C          7.15566        2.89075        3.28062
        C          8.68124        2.90423        3.29701
        C          9.20897        3.85356        4.36970
        C         10.73527        3.86292        4.38316
        C         11.27347        4.81020        5.45174
        C         12.79282        4.81703        5.46246
        H          0.62420       -0.67624       -0.73886
        H          0.60223        1.00743       -0.18569
        H          0.59837       -0.31563        0.99622
        H          2.88337        0.31565       -0.94674
        H          2.87961       -1.00160        0.22902
        H          2.67826        0.66303        2.09347
        H          2.68091        1.98005        0.91889
        H          4.95608        1.27969        0.13728
        H          4.95349       -0.03891        1.31112
        H          4.72996        1.62033        3.17524
        H          4.73142        2.93925        2.00202
        H          7.00963        2.24697        1.22537
        H          7.00844        0.92672        2.39728
        H          6.77826        2.58280        4.26344
        H          6.77905        3.90354        3.09209
        H          9.05732        3.21220        2.31361
        H          9.05648        1.89051        3.48373
        H          8.83233        3.54509        5.35255
        H          8.83406        4.86718        4.18229
        H         11.10909        4.16750        3.39797
        H         11.10701        2.84786        4.56911
        H         10.90576        4.50686        6.43902
        H         10.90834        5.82716        5.26681
        H         13.18649        3.81699        5.67312
        H         13.16432        5.49863        6.23161
        H         13.18931        5.14445        4.49536
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

