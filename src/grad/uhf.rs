#![doc = include_str!("rhf-grad-doc.md")]

use super::rhf::*;
use crate::constants::AUXBAS_THRESHOLD;
use crate::grad::traits::GradAPI;
use crate::scf_io;
use crate::scf_io::SCF;
use crate::Molecule;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use rest_libcint::prelude::*;
use rstsr::prelude::*;
use std::collections::HashMap;
use tensors::{matrix_blas_lapack::_power_rayon_for_symmetric_matrix, MatrixFull};

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;
type TsrView<'a, T> = TensorView<'a, T, DeviceBLAS, IxD>;
type TsrMut<'a, T> = TensorMut<'a, T, DeviceBLAS, IxD>;

/// Gradient structure and values for UHF method.
pub struct RIUHFGradient<'a> {
    pub scf_data: &'a SCF,
    pub flags: RIHFGradientFlags,
    pub result: HashMap<String, MatrixFull<f64>>,
}

impl GradAPI for RIUHFGradient<'_> {
    fn get_gradient(&self) -> MatrixFull<f64> {
        self.result.get("de").unwrap().clone()
    }

    fn get_energy(&self) -> f64 {
        self.scf_data.scf_energy
    }
}

impl RIUHFGradient<'_> {
    pub fn new(scf_data: &SCF) -> RIUHFGradient<'_> {
        // check SCF type
        match scf_data.scftype {
            scf_io::SCFType::UHF => {}
            _ => panic!("SCFtype is not sutiable for UHF gradient."),
        };

        // flags
        let mut flags = RIHFGradientFlagsBuilder::default();
        flags.factor_j(Some(1.0));
        flags.factor_k(Some(1.0));
        flags.auxbasis_response(scf_data.mol.ctrl.auxbasis_response);
        flags.print_level(scf_data.mol.ctrl.print_level);
        let flags = flags.build().unwrap();

        RIUHFGradient {
            scf_data,
            flags,
            result: HashMap::new(),
        }
    }

    /// Derivatives of nuclear repulsion energy with reference to nuclear coordinates
    pub fn calc_de_nuc(&mut self) -> &mut Self {
        let mol = &self.scf_data.mol;
        let de_nuc = calc_de_nuc(mol);
        self.result.insert("de_nuc".into(), de_nuc);
        return self;
    }

    pub fn calc_de_ovlp(&mut self) -> &mut Self {
        // preparation
        let mol = &self.scf_data.mol;
        let mut cint_data = mol.initialize_cint(false);
        let device = rt::DeviceBLAS::default();

        // orbital informations
        let mo_coeff = get_mo_coeff(self.scf_data, &device);
        let mo_occ = get_mo_occ(self.scf_data, &device);
        let mo_energy = get_mo_energy(self.scf_data, &device);

        // tsr_int1e_ipovlp
        let tsr_int1e_ipovlp = {
            let (out, mut shape) = cint_data.integral_s1::<int1e_ipovlp>(None);
            rt::asarray((out, shape, &device))
        };

        let mut dme0 = get_dme0(mo_coeff[0].view(), mo_occ[0].view(), mo_energy[0].view());
        dme0 += get_dme0(mo_coeff[1].view(), mo_occ[1].view(), mo_energy[1].view());
        let dao_ovlp = get_grad_dao_ovlp(tsr_int1e_ipovlp.view(), dme0.view());

        // de_ovlp
        let natm = mol.geom.elem.len();
        let mut de_ovlp = rt::zeros(([3, natm], &device));
        let ao_slice = mol.aoslice_by_atom();

        for atm in 0..natm {
            let [_, _, p0, p1] = ao_slice[atm];
            *&mut de_ovlp.i_mut((.., atm)) += dao_ovlp.i((p0..p1)).sum_axes(0);
        }

        // note f-contiguous transpose
        let de_ovlp_raw = de_ovlp.into_raw_parts().0.into_cpu_vec().unwrap();
        let de_ovlp = MatrixFull::from_vec([3, natm], de_ovlp_raw).unwrap();
        self.result.insert("de_ovlp".into(), de_ovlp);
        return self;
    }

    pub fn calc_de_hcore(&mut self) -> &mut Self {
        let mol = &self.scf_data.mol;
        let natm = mol.geom.elem.len();
        let device = rt::DeviceBLAS::default();

        let dm = get_dm(self.scf_data, &device);
        let dm = &dm[0] + &dm[1];
        let mut de_hcore: Tsr<f64> = rt::zeros(([3, natm], &device));
        let mut gen_deriv_hcore = generator_deriv_hcore(&self.scf_data);
        for atm in 0..natm {
            *&mut de_hcore.i_mut((.., atm)) += (gen_deriv_hcore(atm) * &dm).sum_axes([0, 1]);
        }

        let de_hcore_raw = de_hcore.into_raw_parts().0.into_cpu_vec().unwrap();
        let de_hcore = MatrixFull::from_vec([3, natm], de_hcore_raw).unwrap();
        self.result.insert("de_hcore".into(), de_hcore);
        return self;
    }

    pub fn calc_de_jk(&mut self) -> &mut Self {
        let mut time_records = crate::utilities::TimeRecords::new();
        time_records.new_item("de-jk preparation 1", "de-jk preparation 1");
        time_records.new_item("de-jk preparation power", "de-jk preparation power");
        time_records.new_item("de-jk preparation 2", "de-jk preparation 2");
        time_records.new_item("de-jk batch int", "de-jk batch int");
        time_records.new_item("de-jk batch 1", "de-jk batch 1");
        time_records.new_item("de-jk batch 2", "de-jk batch 2");
        time_records.new_item("de-jk batch 3", "de-jk batch 3");
        time_records.new_item("de-jk batch 4", "de-jk batch 4");
        time_records.new_item("de-jk batch 5", "de-jk batch 5");

        time_records.count_start("de-jk preparation 1");

        let mol = &self.scf_data.mol;
        let auxmol = mol.make_auxmol_fake();
        let natm = mol.geom.elem.len();
        let mut cint_data = mol.initialize_cint(true);
        let mut cint_data_aux = auxmol.initialize_cint(false);
        let n_basis_shell = mol.cint_bas.len() as i32;
        let n_auxbas_shell = mol.cint_aux_bas.len() as i32;
        let device = rt::DeviceBLAS::default();

        // density matrix and triu-packed density matrix
        let dm = get_dm(self.scf_data, &device);
        let dm = &dm[0] + &dm[1];
        let dm_tp = pack_triu_tilde(dm.view());

        // orbital coefficients
        let mo_coeff = get_mo_coeff(self.scf_data, &device);
        let mo_occ = get_mo_occ(self.scf_data, &device);
        let nao = dm.shape()[0];
        let nocc = [
            mo_occ[0].iter().map(|&x| if x > 0.0 { 1 } else { 0 }).sum::<usize>(),
            mo_occ[1].iter().map(|&x| if x > 0.0 { 1 } else { 0 }).sum::<usize>(),
        ];
        let occ_coeff = [mo_coeff[0].i((.., ..nocc[0])), mo_coeff[1].i((.., ..nocc[1]))];

        // eigen-decomposed ERI
        let ederi_utp = {
            let tsr = self.scf_data.rimatr.as_ref().unwrap();
            rt::asarray((&tsr.0.data, tsr.0.size, &device))
        };
        let naux = ederi_utp.shape()[0];

        time_records.count_start("de-jk preparation power");
        // tsr_int2c2e_l: J^-1/2
        let tsr_int2c2e_l_inv = {
            let shl_slices = vec![
                [n_basis_shell, n_basis_shell + n_auxbas_shell],
                [n_basis_shell, n_basis_shell + n_auxbas_shell],
            ];
            let (out, shape) = cint_data.integral_s1::<int2c2e>(Some(&shl_slices));
            let out = MatrixFull::from_vec(shape.try_into().unwrap(), out).unwrap();
            let out = _power_rayon_for_symmetric_matrix(&out, -0.5, AUXBAS_THRESHOLD).unwrap();
            rt::asarray((out.data, out.size, &device))
        };
        time_records.count("de-jk preparation power");

        // tsr_int2c2e_ip1
        let tsr_int2c2e_ip1 = {
            let shl_slices = vec![
                [n_basis_shell, n_basis_shell + n_auxbas_shell],
                [n_basis_shell, n_basis_shell + n_auxbas_shell],
            ];
            let (out, shape) = cint_data.integral_s1::<int2c2e_ip1>(Some(&shl_slices));
            rt::asarray((out, shape, &device))
        };

        // shell partition of int3c2e
        let ao_loc = cint_data.cgto_loc();
        let aux_loc = &ao_loc[(n_basis_shell as usize)..];
        
        // available memory in MB, if not set, will be calculated from system
        let sys_info = sysinfo::System::new_all();
        let mem_avail = self.flags.max_memory.map(|max_memory| {
            let pid = sysinfo::get_current_pid().unwrap();
            let used_memory = sys_info.process(pid).unwrap().memory() as f64 / 1024.0 / 1024.0;
            max_memory - used_memory
        });
        let aux_batch_size = calc_batch_size::<f64>(8 * nao * nao, mem_avail, None, Some(naux * (nocc[0] * nocc[0] + nocc[1] * nocc[1])));
        let aux_batch_size = aux_batch_size.min(216);
        let aux_partition = balance_partition(aux_loc, aux_batch_size);

        time_records.count("de-jk preparation 1");

        // preparation finished

        time_records.count_start("de-jk preparation 2");

        // temporaries for de_jaux, de_kaux
        let mut itm_j = rt::full(([], f64::NAN, &device));
        let mut dao_j = rt::full(([], f64::NAN, &device));
        let mut daux_j = rt::full(([], f64::NAN, &device));
        if self.flags.factor_j.is_some() {
            itm_j = get_itm_j(tsr_int2c2e_l_inv.view(), ederi_utp.view(), dm_tp.view());
            dao_j = rt::zeros(([nao, 3], &device));
        }
        if self.flags.factor_j.is_some() && self.flags.auxbasis_response {
            daux_j = get_grad_daux_j_int2c2e_ip1(tsr_int2c2e_ip1.view(), itm_j.view());
        }

        let mut itm_k_occtp = [rt::full(([], f64::NAN, &device)), rt::full(([], f64::NAN, &device))];
        let mut dao_k = rt::full(([], f64::NAN, &device));
        let mut daux_k = rt::full(([], f64::NAN, &device));
        if self.flags.factor_k.is_some() {
            itm_k_occtp = [
                get_itm_k_occtp(tsr_int2c2e_l_inv.view(), ederi_utp.view(), occ_coeff[0].view()),
                get_itm_k_occtp(tsr_int2c2e_l_inv.view(), ederi_utp.view(), occ_coeff[1].view()),
            ];
            dao_k = rt::zeros(([nao, 3], &device));
        }
        if self.flags.factor_k.is_some() && self.flags.auxbasis_response {
            let itm_k_aux = get_itm_k_aux(itm_k_occtp[0].view_mut()) + get_itm_k_aux(itm_k_occtp[1].view_mut());
            daux_k = get_grad_daux_k_int2c2e_ip1(tsr_int2c2e_ip1.view(), itm_k_aux.view());
        }

        time_records.count("de-jk preparation 2");

        let mut idx_aux_start = 0;
        for [shl0, shl1] in aux_partition {
            let shl_naux = aux_loc[shl1] - aux_loc[shl0];
            let shl_slices = vec![
                [0, n_basis_shell],
                [0, n_basis_shell],
                [n_basis_shell + shl0 as i32, n_basis_shell + shl1 as i32],
            ];
            let (p0, p1) = (idx_aux_start, idx_aux_start + shl_naux);

            time_records.count_start("de-jk batch int");
            // int3c2e_ip1
            let tsr_int3c2e_ip1 = {
                let (out, shape) = cint_data.integral_s1::<int3c2e_ip1>(Some(&shl_slices));
                rt::asarray((out, shape, &device))
            };

            // int3c2e_ip2
            let mut tsr_int3c2e_ip2 = rt::full(([], f64::NAN, &device));
            if self.flags.auxbasis_response {
                tsr_int3c2e_ip2 = {
                    let (out, shape) = cint_data.integral_s2ij::<int3c2e_ip2>(Some(&shl_slices));
                    rt::asarray((out, shape, &device))
                };
            }
            time_records.count("de-jk batch int");

            if self.flags.factor_j.is_some() {
                time_records.count_start("de-jk batch 1");
                *&mut dao_j += get_grad_dao_j_int3c2e_ip1(tsr_int3c2e_ip1.view(), dm.view(), itm_j.i(p0..p1));
                time_records.count("de-jk batch 1");

                if self.flags.auxbasis_response {
                    time_records.count_start("de-jk batch 2");
                    *&mut daux_j.i_mut((p0..p1)) += get_grad_daux_j_int3c2e_ip2(tsr_int3c2e_ip2.view(), dm_tp.view(), itm_j.i(p0..p1));
                    time_records.count("de-jk batch 2");
                }
            }

            if self.flags.factor_k.is_some() {
                time_records.count_start("de-jk batch 3");
                let mut itm_k_ao = get_itm_k_ao(itm_k_occtp[0].i((.., p0..p1)), occ_coeff[0].view())
                    + get_itm_k_ao(itm_k_occtp[1].i((.., p0..p1)), occ_coeff[1].view());
                time_records.count("de-jk batch 3");

                time_records.count_start("de-jk batch 4");
                *&mut dao_k += get_grad_dao_k_int3c2e_ip1(tsr_int3c2e_ip1.view(), itm_k_ao.view());
                time_records.count("de-jk batch 4");

                if self.flags.auxbasis_response {
                    time_records.count_start("de-jk batch 5");
                    *&mut daux_k.i_mut((p0..p1)) += get_grad_daux_k_int3c2e_ip2(tsr_int3c2e_ip2.view(), itm_k_ao.view());
                    time_records.count("de-jk batch 5");
                }
            }

            idx_aux_start += shl_naux;
        }

        if self.scf_data.mol.ctrl.print_level >= 2 {
            time_records.report_all();
        }

        // de_j, de_k, de_jaux, de_kaux
        let mut de_j = rt::full(([3, natm], f64::NAN, &device));
        let mut de_k = rt::full(([3, natm], f64::NAN, &device));
        let mut de_jaux = rt::full(([3, natm], f64::NAN, &device));
        let mut de_kaux = rt::full(([3, natm], f64::NAN, &device));
        let ao_slice = mol.aoslice_by_atom();
        let aux_slice = mol.make_auxmol_fake().aoslice_by_atom();

        for atm in 0..natm {
            let [_, _, p0, p1] = ao_slice[atm].clone().try_into().unwrap();
            if self.flags.factor_j.is_some() {
                *&mut de_j.i_mut((.., atm)).assign(dao_j.i((p0..p1)).sum_axes(0));
            }
            if self.flags.factor_k.is_some() {
                *&mut de_k.i_mut((.., atm)).assign(dao_k.i((p0..p1)).sum_axes(0));
            }

            let [_, _, p0, p1] = aux_slice[atm].clone().try_into().unwrap();
            if self.flags.factor_j.is_some() && self.flags.auxbasis_response {
                *&mut de_jaux.i_mut((.., atm)).assign(daux_j.i((p0..p1)).sum_axes(0));
            }
            if self.flags.factor_k.is_some() && self.flags.auxbasis_response {
                *&mut de_kaux.i_mut((.., atm)).assign(daux_k.i((p0..p1)).sum_axes(0));
            }
        }

        if let Some(factor_j) = self.flags.factor_j {
            de_j *= factor_j;
            de_jaux *= factor_j;
        }
        if let Some(factor_k) = self.flags.factor_k {
            de_k *= -1.0 * factor_k;
            de_kaux *= -1.0 * factor_k;
        }

        let de_j = {
            let de_j_raw = de_j.into_raw_parts().0.into_cpu_vec().unwrap();
            MatrixFull::from_vec([3, natm], de_j_raw).unwrap()
        };
        let de_jaux = {
            let de_jaux_raw = de_jaux.into_raw_parts().0.into_cpu_vec().unwrap();
            MatrixFull::from_vec([3, natm], de_jaux_raw).unwrap()
        };
        let de_k = {
            let de_k_raw = de_k.into_raw_parts().0.into_cpu_vec().unwrap();
            MatrixFull::from_vec([3, natm], de_k_raw).unwrap()
        };
        let de_kaux = {
            let de_kaux_raw = de_kaux.into_raw_parts().0.into_cpu_vec().unwrap();
            MatrixFull::from_vec([3, natm], de_kaux_raw).unwrap()
        };

        if self.flags.factor_j.is_some() {
            self.result.insert("de_j".into(), de_j);
        }
        if self.flags.factor_k.is_some() {
            self.result.insert("de_k".into(), de_k);
        }
        if self.flags.factor_j.is_some() && self.flags.auxbasis_response {
            self.result.insert("de_jaux".into(), de_jaux);
        }
        if self.flags.factor_k.is_some() && self.flags.auxbasis_response {
            self.result.insert("de_kaux".into(), de_kaux);
        }

        return self;
    }

    pub fn calc(&mut self) -> &MatrixFull<f64> {
        let mut time_records = crate::utilities::TimeRecords::new();
        time_records.new_item("uhf grad", "uhf grad");
        time_records.new_item("uhf grad calc_de_nuc", "uhf grad calc_de_nuc");
        time_records.new_item("uhf grad calc_de_ovlp", "uhf grad calc_de_ovlp");
        time_records.new_item("uhf grad calc_de_hcore", "uhf grad calc_de_hcore");
        time_records.new_item("uhf grad calc_de_jk", "uhf grad calc_de_jk");

        time_records.count_start("uhf grad");

        time_records.count_start("uhf grad calc_de_nuc");
        self.calc_de_nuc();
        time_records.count("uhf grad calc_de_nuc");

        time_records.count_start("uhf grad calc_de_ovlp");
        self.calc_de_ovlp();
        time_records.count("uhf grad calc_de_ovlp");

        time_records.count_start("uhf grad calc_de_hcore");
        self.calc_de_hcore();
        time_records.count("uhf grad calc_de_hcore");

        if self.flags.factor_j.is_some() || self.flags.factor_k.is_some() {
            time_records.count_start("uhf grad calc_de_jk");
            self.calc_de_jk();
            time_records.count("uhf grad calc_de_jk");
        }

        time_records.count("uhf grad");

        let mut de = self.result.get("de_nuc").unwrap().clone();
        de += self.result.get("de_ovlp").unwrap().clone();
        de += self.result.get("de_hcore").unwrap().clone();
        self.result.get("de_j").map(|x| de += x.clone());
        self.result.get("de_k").map(|x| de += x.clone());
        self.result.get("de_jaux").map(|x| de += x.clone());
        self.result.get("de_kaux").map(|x| de += x.clone());
        self.result.insert("de".into(), de);

        if self.scf_data.mol.ctrl.print_level >= 2 {
            time_records.report_all();
        }

        return self.result.get("de").unwrap();
    }
}

fn get_mo_coeff(scf_data: &SCF, device: &DeviceBLAS) -> Vec<Tsr<f64>> {
    // This can be reconsidered if 3-D (spin, ao, mo) is better.
    // Currently, vector of 2-D (ao, mo) is used.
    let mut result = vec![];
    for spin in [0, 1] {
        let mo_coeff = &scf_data.eigenvectors[spin];
        result.push(rt::asarray((&mo_coeff.data, mo_coeff.size, device)).to_owned());
    }
    return result;
}

fn get_mo_occ(scf_data: &SCF, device: &DeviceBLAS) -> Vec<Tsr<f64>> {
    // This can be reconsidered if 2-D (spin, mo) is better.
    // Currently, vector of 1-D (mo) is used.
    let mut result = vec![];
    for spin in [0, 1] {
        let mo_occ = &scf_data.occupation[spin];
        result.push(rt::asarray((mo_occ, [mo_occ.len()], device)).to_owned());
    }
    return result;
}

fn get_mo_energy(scf_data: &SCF, device: &DeviceBLAS) -> Vec<Tsr<f64>> {
    // This can be reconsidered if 2-D (spin, mo) is better.
    // Currently, vector of 1-D (ao, mo) is used.
    let mut result = vec![];
    for spin in [0, 1] {
        let mo_energy = &scf_data.eigenvalues[spin];
        result.push(rt::asarray((mo_energy, [mo_energy.len()], device)).to_owned());
    }
    return result;
}

fn get_dm(scf_data: &SCF, device: &DeviceBLAS) -> Vec<Tsr<f64>> {
    // This can be reconsidered if 3-D (spin, ao, ao) is better.
    // Currently, vector of 2-D (ao, ao) is used.
    let mut result = vec![];
    for spin in [0, 1] {
        let dm = &scf_data.density_matrix[spin];
        result.push(rt::asarray((&dm.data, dm.size, device)).to_owned());
    }
    return result;
}

#[cfg(test)]
#[allow(non_snake_case)]
mod debug {
    use super::*;
    use crate::ctrl_io::InputKeywords;
    use crate::scf_io::scf_without_build;

    #[test]
    fn test_nh3() {
        let scf_data = initialize_nh3();
        let time = std::time::Instant::now();
        let scf_grad = test_with_scf(&scf_data);
        println!("Time elapsed: {:?}", time.elapsed());

        let de = scf_grad.result.get("de").unwrap().clone();
        let de = rt::asarray((de.data, de.size));
        #[rustfmt::skip]
        let de_ref = vec![
            -0.0023559249, -0.0046636209, -0.0080803853,
            -0.0123568700,  0.0105089687,  0.0115916374,
             0.0120656988,  0.0087569741, -0.0235104578,
             0.0026470960, -0.0146023219,  0.0199992057,
        ];
        let de_ref = rt::asarray((&de_ref, [3, 4]));
        assert!((de_ref - de).abs().max_all() < 1.0e-5);
    }

    #[test]
    fn test_hi() {
        let scf_data = initialize_hi();
        let time = std::time::Instant::now();
        let scf_grad = test_with_scf(&scf_data);
        println!("Time elapsed: {:?}", time.elapsed());

        let de = scf_grad.result.get("de").unwrap().clone();
        let de = rt::asarray((de.data, de.size));
        #[rustfmt::skip]
        let de_ref = vec![
            -0.0000000000, 0.0000000000,  0.0021577095,
             0.0000000000, 0.0000000000, -0.0021577095,
        ];
        let de_ref = rt::asarray((&de_ref, [3, 2]));
        assert!((de_ref - de).abs().max_all() < 1.0e-5);
    }

    #[test]
    #[ignore = "stress test"]
    fn test_c12h26() {
        let scf_data = initialize_c12h26();
        let time = std::time::Instant::now();
        test_with_scf(&scf_data);
        println!("Time elapsed: {:?}", time.elapsed());
    }

    fn test_with_scf(scf_data: &SCF) -> RIUHFGradient {
        let mut scf_grad = RIUHFGradient::new(&scf_data);
        scf_grad.calc();

        println!("=== de ===");
        let de = scf_grad.result.get("de").unwrap().clone();
        let de = rt::asarray((de.data, de.size));
        println!("{:12.6}", de.t());

        println!("=== de_nuc ===");
        scf_grad.result.get("de_nuc").map(|de| {
            let de = rt::asarray((&de.data, de.size));
            println!("{:12.6}", de.t());
        });

        println!("=== de_ovlp ===");
        scf_grad.result.get("de_ovlp").map(|de| {
            let de = rt::asarray((&de.data, de.size));
            println!("{:12.6}", de.t());
        });

        println!("=== de_hcore ===");
        scf_grad.result.get("de_hcore").map(|de| {
            let de = rt::asarray((&de.data, de.size));
            println!("{:12.6}", de.t());
        });

        println!("=== de_j ===");
        scf_grad.result.get("de_j").map(|de| {
            let de = rt::asarray((&de.data, de.size));
            println!("{:12.6}", de.t());
        });

        println!("=== de_k ===");
        scf_grad.result.get("de_k").map(|de| {
            let de = rt::asarray((&de.data, de.size));
            println!("{:12.6}", de.t());
        });

        println!("=== de_jaux ===");
        scf_grad.result.get("de_jaux").map(|de| {
            let de = rt::asarray((&de.data, de.size));
            println!("{:12.6}", de.t());
        });

        println!("=== de_kaux ===");
        scf_grad.result.get("de_kaux").map(|de| {
            let de = rt::asarray((&de.data, de.size));
            println!("{:12.6}", de.t());
        });

        return scf_grad;
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
     charge =               2.0
     spin =                 3.0
     spin_polarization =    true
     auxbasis_response =    true
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
        let mol = Molecule::build_native(ctrl, geom, None).unwrap();
        let mut scf_data = scf_io::SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);
        return scf_data;
    }

    fn initialize_hi() -> SCF {
        let input_token = r##"
[ctrl]
     # 设置程序输出的程度，缺省为1
     print_level =               2
     # 设置Rayon和OpenMP的并行数，缺省为1
     num_threads =               16
     # 设置使用的电子结构方法，缺省为HF
     xc =                        "mp2" 
     basis_path =                "basis-set-pool/def2-TZVP"
     auxbas_path =               "basis-set-pool/def2-SV(P)-JKFIT"
     charge =                    -1.0
     spin =                      2.0
     spin_polarization =         true
     mixer =                     "diis"
     num_max_diis =              8
     start_diis_cycle =          1
     mix_param =                 0.6
     max_scf_cycle =             100
     initial_guess =             "hcore"
     auxbasis_response =         true

[geom]
    name = "HI"
    unit = "Angstrom"
    position = """
        H   0.0  0.0  0.0
        I   0.0  0.0  3.0
    """
"##;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (mut ctrl, mut geom) = InputKeywords::parse_ctl_from_json(&keys).unwrap();
        let mol = Molecule::build_native(ctrl, geom, None).unwrap();
        let mut scf_data = scf_io::SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);
        return scf_data;
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
     charge =               2.0
     spin =                 3.0
     spin_polarization =    true
     external_grids =       "none"
     initial_guess=         "sad"
     mixer =                "diis"
     num_max_diis =         8
     start_diis_cycle =     3
     mix_param =            0.8
     max_scf_cycle =        10
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
        let mol = Molecule::build_native(ctrl, geom, None).unwrap();
        let mut scf_data = scf_io::SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);
        return scf_data;
    }
}
