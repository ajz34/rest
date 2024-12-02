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
        use std::time::{Instant, Duration};
    
        // set reduced precision
        let reduced_precision = candle_core::cuda::gemm_reduced_precision_f32();
        candle_core::cuda::set_gemm_reduced_precision_f32(true);

        let time_tot = Instant::now();
        let mut time_load = Duration::from_secs_f64(0.0);
        let mut time_matmul = Duration::from_secs_f64(0.0);
        let mut time_sum = Duration::from_secs_f64(0.0);
    
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
            let current = Instant::now();
            let nbatch_i = std::cmp::min(batch_occ, nocc - ptr_i);
            let cderi_ovu_batch_i = Tensor::from_slice(
                cderi_ovu.slice(s![ptr_i..ptr_i+nbatch_i, .., ..]).as_slice().unwrap(),
                (nbatch_i, nvir, naux), &device
            )?.to_dtype(DType::F32)?;
            device.synchronize();
            time_load += current.elapsed();
    
            for ptr_j in (0..ptr_i+nbatch_i).step_by(batch_occ) {
                let nbatch_j = std::cmp::min(batch_occ, ptr_i + nbatch_i - ptr_j);
                let current = Instant::now();
                let cderi_ovu_batch_j = if (ptr_j == ptr_i) { &cderi_ovu_batch_i } else {
                    &Tensor::from_slice(
                        cderi_ovu.slice(s![ptr_j..ptr_j+nbatch_j, .., ..]).as_slice().unwrap(),
                        (nbatch_j, nvir, naux), &device
                    )?.to_dtype(DType::F32)?
                };
                device.synchronize();
                time_load += current.elapsed();
    
                for i in ptr_i .. ptr_i+nbatch_i {
                    for j in ptr_j .. std::cmp::min(ptr_j + nbatch_j, i + 1) {
                        let current = Instant::now();
                        let cderi_ivu = cderi_ovu_batch_i.i(i - ptr_i)?;
                        let cderi_jvu = cderi_ovu_batch_j.i(j - ptr_j)?;
                        let g_ab = cderi_ivu.matmul(&cderi_jvu.transpose(0, 1)?)?;
                        device.synchronize();
                        time_matmul += current.elapsed();

                        let current = Instant::now();
                        let e_i = occ_energy[i] as f64;
                        let e_j = occ_energy[j] as f64;
                        let d_ab = ((&d_vv_device + e_i)? + e_j)?;
                        let t_ab = (&g_ab / d_ab)?;
                        let factor = if i != j { 2.0 } else { 1.0 };
    
                        e_bi1 = (e_bi1 + (factor * (&g_ab * &t_ab)?.sum_all()?)?.to_dtype(DType::F64)?)?;
                        e_bi2 = (e_bi2 + (factor * (&g_ab * &t_ab.transpose(0, 1)?)?.sum_all()?.to_dtype(DType::F64)?)?)?;
                        device.synchronize();
                        time_sum += current.elapsed();
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

        println!("Total time: {:?}", time_tot.elapsed());
        println!("Load time: {:?}", time_load);
        println!("Matmul time: {:?}", time_matmul);
        println!("Sum time: {:?}", time_sum);
    
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
    use crate::scf_io::SCF;

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
mod debug_c20h42 {
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

        /*
            Elapsed time (RI3MO): 5.834809235s
            Total time: 4.428279844s
            Load time: 283.471219ms
            Matmul time: 1.314239139s
            Sum time: 2.616146412s
            [-3.7516710824339645, -2.9355218441569377, -0.8161492382770268]
            Elapsed time (MP2 cuda): 4.43012143s
            [-3.751694492142329, -2.935539784813922, -0.8161547073284072]
            Elapsed time (MP2 rayon): 12.629729127s
         */
    }

    fn initialize_c20h42() -> SCF {
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

