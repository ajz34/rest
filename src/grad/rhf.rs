#![doc = include_str!("rhf-grad-doc.md")]

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

#[non_exhaustive]
#[derive(derive_builder::Builder)]
pub struct RIHFGradientFlags {
    /// Print level for debugging.
    #[builder(default = 0)]
    pub print_level: usize,

    /// Memory available for calculation (in MB).
    /// Note that this value does not count the memory used by the program itself.
    #[builder(default = "None")]
    pub max_memory: Option<f64>,

    /// Perform response of auxiliary basis.
    #[builder(default = true)]
    pub auxbasis_response: bool,

    /// Factor of J contribution.
    /// Usually set to 1.0.
    #[builder(default = "Some(1.0)")]
    pub factor_j: Option<f64>,

    /// Factor of K contribution.
    /// For RHF, it should be set to 1.0; for KS, it depends on exchange coefficient.
    #[builder(default = "Some(1.0)")]
    pub factor_k: Option<f64>,
}

/// Gradient structure and values for RHF method.
///
/// Field `result` contains
/// - `de`: total derivative of energy with respect to nuclear coordinates, in unit a.u.
/// - components contributed to total derivative, including
///   `de_nuc`, `de_hcore`, `de_ovlp`, `de_j`, `de_k`, `de_jaux`, `de_kaux`.
pub struct RIRHFGradient<'a> {
    pub scf_data: &'a SCF,
    pub flags: RIHFGradientFlags,
    pub result: HashMap<String, MatrixFull<f64>>,
}

impl GradAPI for RIRHFGradient<'_> {
    fn get_gradient(&self) -> MatrixFull<f64> {
        self.result.get("de").unwrap().clone()
    }

    fn get_energy(&self) -> f64 {
        self.scf_data.scf_energy
    }
}

impl RIRHFGradient<'_> {
    pub fn new(scf_data: &SCF) -> RIRHFGradient<'_> {
        // check SCF type
        match scf_data.scftype {
            scf_io::SCFType::RHF => {}
            _ => panic!("SCFtype is not sutiable for RHF gradient."),
        };

        // flags
        let mut flags = RIHFGradientFlagsBuilder::default();
        flags.factor_j(Some(1.0));
        flags.factor_k(Some(1.0));
        flags.auxbasis_response(scf_data.mol.ctrl.auxbasis_response);
        flags.print_level(scf_data.mol.ctrl.print_level);
        flags.max_memory(scf_data.mol.ctrl.max_memory);
        let flags = flags.build().unwrap();

        RIRHFGradient {
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
        let device = DeviceBLAS::default();

        // orbital informations
        let mo_coeff = get_mo_coeff(self.scf_data, &device);
        let mo_occ = get_mo_occ(self.scf_data, &device);
        let mo_energy = get_mo_energy(self.scf_data, &device);

        // tsr_int1e_ipovlp
        let tsr_int1e_ipovlp = {
            let (out, mut shape) = cint_data.integral_s1::<int1e_ipovlp>(None);
            rt::asarray((out, shape, &device))
        };

        let dme0 = get_dme0(mo_coeff.view(), mo_occ.view(), mo_energy.view());
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
        let device = DeviceBLAS::default();

        let dm = get_dm(self.scf_data, &device);
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
        let device = DeviceBLAS::default();

        // density matrix and triu-packed density matrix
        let dm = get_dm(self.scf_data, &device);
        let dm_tp = pack_triu_tilde(dm.view());

        // orbital coefficients
        let mo_coeff = get_mo_coeff(self.scf_data, &device);
        let mo_occ = get_mo_occ(self.scf_data, &device);
        let nao = dm.shape()[0];
        let nocc = mo_occ.iter().map(|&x| if x > 0.0 { 1 } else { 0 }).sum::<usize>();
        let weighted_occ_coeff = mo_coeff.i((.., 0..nocc)) * mo_occ.mapv(f64::sqrt).i((None, 0..nocc));

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
        let aux_batch_size = calc_batch_size::<f64>(8 * nao * nao, mem_avail, None, Some(naux * nocc * nocc));
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

        let mut itm_k_occtp = rt::full(([], f64::NAN, &device));
        let mut dao_k = rt::full(([], f64::NAN, &device));
        let mut daux_k = rt::full(([], f64::NAN, &device));
        if self.flags.factor_k.is_some() {
            itm_k_occtp = get_itm_k_occtp(tsr_int2c2e_l_inv.view(), ederi_utp.view(), weighted_occ_coeff.view());
            dao_k = rt::zeros(([nao, 3], &device));
        }
        if self.flags.factor_k.is_some() && self.flags.auxbasis_response {
            let itm_k_aux = get_itm_k_aux(itm_k_occtp.view_mut());
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
                let itm_k_ao = get_itm_k_ao(itm_k_occtp.i((.., p0..p1)), weighted_occ_coeff.view());
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

        if self.flags.print_level >= 2 {
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
            de_k *= -0.5 * factor_k;
            de_kaux *= -0.5 * factor_k;
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
        time_records.new_item("rhf grad", "rhf grad");
        time_records.new_item("rhf grad calc_de_nuc", "rhf grad calc_de_nuc");
        time_records.new_item("rhf grad calc_de_ovlp", "rhf grad calc_de_ovlp");
        time_records.new_item("rhf grad calc_de_hcore", "rhf grad calc_de_hcore");
        time_records.new_item("rhf grad calc_de_jk", "rhf grad calc_de_jk");

        time_records.count_start("rhf grad");

        time_records.count_start("rhf grad calc_de_nuc");
        self.calc_de_nuc();
        time_records.count("rhf grad calc_de_nuc");

        time_records.count_start("rhf grad calc_de_ovlp");
        self.calc_de_ovlp();
        time_records.count("rhf grad calc_de_ovlp");

        time_records.count_start("rhf grad calc_de_hcore");
        self.calc_de_hcore();
        time_records.count("rhf grad calc_de_hcore");

        if self.flags.factor_j.is_some() || self.flags.factor_k.is_some() {
            time_records.count_start("rhf grad calc_de_jk");
            self.calc_de_jk();
            time_records.count("rhf grad calc_de_jk");
        }

        time_records.count("rhf grad");

        let mut de = self.result.get("de_nuc").unwrap().clone();
        de += self.result.get("de_ovlp").unwrap().clone();
        de += self.result.get("de_hcore").unwrap().clone();
        self.result.get("de_j").map(|x| de += x.clone());
        self.result.get("de_k").map(|x| de += x.clone());
        self.result.get("de_jaux").map(|x| de += x.clone());
        self.result.get("de_kaux").map(|x| de += x.clone());
        self.result.insert("de".into(), de);

        if self.flags.print_level >= 2 {
            time_records.report_all();
        }

        return self.result.get("de").unwrap();
    }
}

pub fn generator_deriv_hcore<'a>(scf_data: &'a SCF) -> impl FnMut(usize) -> Tsr<f64> + 'a {
    let mut mol = scf_data.mol.clone();
    let mut cint_data = mol.initialize_cint(false);
    let device = DeviceBLAS::default();

    let necp_by_atom = {
        let basis4elem = Molecule::collect_basis(&scf_data.mol.ctrl, &mut scf_data.mol.geom.clone()).0;
        basis4elem
            .iter()
            .map(|i| if let Some(num_ecp) = i.ecp_electrons { num_ecp } else { 0 })
            .collect::<Vec<usize>>()
    };
    let has_ecp = necp_by_atom.iter().any(|&x| x > 0);

    let tsr_int1e_ipkin = {
        let (out, shape) = cint_data.integral_s1::<int1e_ipkin>(None);
        rt::asarray((out, shape, &device))
    };

    let tsr_int1e_ipnuc = {
        let (out, shape) = cint_data.integral_s1::<int1e_ipnuc>(None);
        rt::asarray((out, shape, &device))
    };

    let mut h1 = -(tsr_int1e_ipkin + tsr_int1e_ipnuc);

    if has_ecp {
        let (out, shape) = cint_data.integral_ecp_s1::<ECPscalar_ipnuc>(None);
        let tsr_int1e_ecp_ipnuc = rt::asarray((out, shape, &device));
        h1 -= tsr_int1e_ecp_ipnuc;
    }
    let aoslice_by_atom = mol.aoslice_by_atom();
    let charge_by_atom = crate::geom_io::get_charge(&mol.geom.elem);

    move |atm_id| {
        let [_, _, p0, p1] = aoslice_by_atom[atm_id];
        mol.with_rinv_at_nucleus(atm_id, |mol| {
            let mut cint_data = mol.initialize_cint(false);

            let tsr_int1e_iprinv = {
                let (out, shape) = cint_data.integral_s1::<int1e_iprinv>(None);
                rt::asarray((out, shape, &device))
            };

            let mut vrinv = -((&charge_by_atom)[atm_id] - (&necp_by_atom)[atm_id] as f64) * tsr_int1e_iprinv;

            if has_ecp && necp_by_atom[atm_id] > 0 {
                let (out, shape) = cint_data.integral_ecp_s1::<ECPscalar_iprinv>(None);
                let tsr_int1e_ecp_iprinv = rt::asarray((out, shape, &device));
                vrinv += tsr_int1e_ecp_iprinv;
            }

            *&mut vrinv.i_mut((p0..p1)) += &h1.i((p0..p1));
            (&vrinv + vrinv.swapaxes(0, 1)).into_contig(FlagOrder::F)
        })
    }
}

/* #region utilities */

/// Calculate batch size within possible memory.
///
/// For example, if we want to compute tensor (100, 100, 100), but only 50,000 memory available, then this tensor should be splited into 20 batches.
///
/// ``flop`` in parameters is number of data, not refers to FLOPs.
///
/// This function requires generic `<T>`, which determines size of data.
///
/// # Parameters
///
/// - `unit_flop`: Number of data for unit operation. For example, for a tensor with shape (110, 120, 130), the 1st dimension is indexable from outer programs, then a unit operation handles 120x130 = 15,600 data. Then we call this function with ``unit_flop = 15600``. This value will be set to 1 if too small.
/// - `mem_avail`: Memory available in MB. By default, it will check available memory in os system.
/// - `mem_factor`: factor for mem_avail, to avoid all memory consumed; should be smaller than 1, recommended 0.7.
/// - `pre_flop`: Number of data preserved in memory. Unit in number.
pub fn calc_batch_size<T>(unit_flop: usize, mem_avail: Option<f64>, mem_factor: Option<f64>, pre_flop: Option<usize>) -> usize {
    let nbytes_dtype = std::mem::size_of::<T>();
    let unit_flop = unit_flop.max(1);
    let unit_mb = (unit_flop * nbytes_dtype) as f64 / 1024.0 / 1024.0;
    let pre_mb = pre_flop.unwrap_or(0) as f64 * nbytes_dtype as f64 / 1024.0 / 1024.0;
    let mem_factor = mem_factor.unwrap_or(0.7);
    let mem_avail_mb = mem_avail.unwrap_or_else(|| {
        let sys = sysinfo::System::new_all();
        (sys.total_memory() - sys.used_memory()) as f64 / 1024.0 / 1024.0
    }) * mem_factor;
    let max_mb = mem_avail_mb - pre_mb;

    if unit_mb > max_mb {
        println!("[Warn] Memory overflow when preparing batch number.");
        println!("Current memory available {:10.3} MB, minimum required {:10.3} MB", max_mb, unit_mb);
    }
    let batch_size = (max_mb / unit_mb).max(1.0).to_usize().unwrap();
    return batch_size;
}

/// Balance partition of indices.
///
/// This function is used to balance partition of indices, so that each partition has similar size.
/// This function mostly applied in shell-to-basis partition splitting.
///
/// # Parameters
///
/// - `indices`: List of indices to be partitioned. We assume this array is sorted and no elements are the same value.
/// - `batch_size`: Maximum size of each partition.
///
/// # Example
///
/// ```rust
/// let indices = [1, 3, 6, 7, 10, 15, 16, 19];
/// let partitions = balance_partition(&indices, 4);
/// // A info of `[Warn] Batch size is too small: 15 - 10 > 4` will be printed.
/// assert_eq!(partitions, [[0, 1], [1, 3], [3, 4], [4, 5], [5, 7]]);
/// ```
pub fn balance_partition(indices: &[usize], batch_size: usize) -> Vec<[usize; 2]> {
    if batch_size == 0 {
        panic!("Batch size should not be zero.");
    }
    // handle special case
    if indices.len() <= 1 {
        return vec![];
    }

    let mut partitions = vec![0];
    for idx in 1..indices.len() {
        let last = indices[partitions.last().unwrap().clone()];
        if indices[idx] - last > batch_size {
            if indices[idx - 1] == last {
                println!("[Warn] Batch size is too small: {} - {} > {}", indices[idx], last, batch_size);
                partitions.push(idx);
            } else {
                partitions.push(idx - 1);
                let last = indices[partitions.last().unwrap().clone()];
                if indices[idx] - last > batch_size {
                    println!("[Warn] Batch size is too small: {} - {} > {}", indices[idx], indices[idx - 1], batch_size);
                    partitions.push(idx);
                }
            }
        }
    }
    if partitions.last().unwrap().clone() != indices.len() - 1 {
        partitions.push(indices.len() - 1);
    }

    assert!(partitions.len() >= 2);
    let mut result = vec![];
    for i in 0..partitions.len() - 1 {
        result.push([partitions[i], partitions[i + 1]]);
    }
    return result;
}

fn get_mo_coeff(scf_data: &SCF, device: &DeviceBLAS) -> Tsr<f64> {
    let mo_coeff = &scf_data.eigenvectors[0];
    return rt::asarray((&mo_coeff.data, mo_coeff.size, device)).to_owned();
}

fn get_mo_occ(scf_data: &SCF, device: &DeviceBLAS) -> Tsr<f64> {
    let mo_occ = &scf_data.occupation[0];
    return rt::asarray((mo_occ, [mo_occ.len()], device)).to_owned();
}

fn get_mo_energy(scf_data: &SCF, device: &DeviceBLAS) -> Tsr<f64> {
    let mo_energy = &scf_data.eigenvalues[0];
    return rt::asarray((mo_energy, [mo_energy.len()], device)).to_owned();
}

fn get_dm(scf_data: &SCF, device: &DeviceBLAS) -> Tsr<f64> {
    let dm = &scf_data.density_matrix[0];
    return rt::asarray((&dm.data, dm.size, device)).to_owned();
}

/* #endregion */

/* #region Matrix Algorithms in RI-RHF code */

pub fn calc_de_nuc(mol: &Molecule) -> MatrixFull<f64> {
    let device = DeviceBLAS::default();

    let natm = mol.geom.elem.len();
    let coords = (0..natm).map(|i| mol.geom.get_coord(i)).flatten().collect::<Vec<f64>>();
    let coords = rt::asarray((&coords, [3, natm], &device));

    let charges_by_atom = crate::geom_io::get_charge(&mol.geom.elem);
    let necp_by_atom = mol
        .basis4elem
        .iter()
        .map(|i| if let Some(num_ecp) = i.ecp_electrons { num_ecp as f64 } else { 0.0 })
        .collect::<Vec<f64>>();
    let charges = rt::asarray((charges_by_atom, &device)) - rt::asarray((necp_by_atom, &device));

    let nuc_z = charges;
    let nuc_v = coords.i((.., None, ..)) - coords.i((.., .., None));
    let nuc_inf = rt::full(([natm], f64::INFINITY, &device));
    let nuc_rinv: Tsr<f64> = 1.0 / (nuc_v.l2_norm_axes(0) + rt::diag(&nuc_inf));

    let tmp = -nuc_z.i((None, .., None)) * nuc_z.i((None, None, ..)) * nuc_rinv.mapv(|x| x.powi(3)).i((None, .., ..)) * nuc_v;
    let de_nuc = tmp.sum_axes(1);

    let de_nuc = {
        let de_nuc_raw = de_nuc.into_raw_parts().0.into_cpu_vec().unwrap();
        MatrixFull::from_vec([3, natm], de_nuc_raw).unwrap()
    };

    return de_nuc;
}

pub fn pack_triu_tilde(dm: TsrView<f64>) -> Tsr<f64> {
    // Pack the lower triangular part of a matrix into a 1D array
    // and non-diagonal values are multiplied by 2.
    assert_eq!(dm.ndim(), 2);
    assert_eq!(dm.shape()[0], dm.shape()[1]);
    let nao = dm.shape()[0];
    let mut dm_triu: Tsr<f64> = 2.0 * dm.pack_triu();
    for i in 0..nao {
        dm_triu[[(i + 2) * (i + 1) / 2 - 1]] *= 0.5;
    }
    return dm_triu;
}

pub fn get_dme0(mo_coeff: TsrView<f64>, mo_occ: TsrView<f64>, mo_energy: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    return (&mo_coeff * mo_occ.i((None, ..)) * mo_energy.i((None, ..))) % mo_coeff.t();
}

pub fn get_grad_dao_ovlp(tsr_int1e_ipovlp: TsrView<f64>, dme0: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int1e_ipovlp.f_prefer());
    assert!(dme0.f_prefer());
    return 2.0 * (tsr_int1e_ipovlp * dme0.i((.., .., None))).sum_axes(1);
}

pub fn get_itm_j(tsr_int2c2e_l_inv: TsrView<f64>, ederi_utp: TsrView<f64>, dm_tp: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    return dm_tp % ederi_utp % tsr_int2c2e_l_inv;
}

pub fn get_grad_daux_j_int2c2e_ip1(tsr_int2c2e_ip1: TsrView<f64>, itm_j: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    let naux = itm_j.shape()[0];
    let device = tsr_int2c2e_ip1.device().clone();
    let mut daux_j_int2c2e_ip1 = rt::zeros(([naux, 3], &device));
    for t in 0..3 {
        *&mut daux_j_int2c2e_ip1.i_mut((.., t)) += &itm_j * (tsr_int2c2e_ip1.i((.., .., t)) % &itm_j);
    }
    return daux_j_int2c2e_ip1;
}

pub fn get_grad_dao_j_int3c2e_ip1(tsr_int3c2e_ip1: TsrView<f64>, dm: TsrView<f64>, itm_j: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int3c2e_ip1.f_prefer());

    let nao = dm.shape()[0];
    let naux = itm_j.shape()[0];
    let device = dm.device().clone();

    let mut dao_j_int3c2e_ip1: Tsr<f64> = rt::zeros(([nao, 3], &device));
    for t in 0..3 {
        let tmp1 = tsr_int3c2e_ip1.i((.., .., .., t)).reshape([nao * nao, naux]) % &itm_j;
        *&mut dao_j_int3c2e_ip1.i_mut((.., t)) += (&tmp1.reshape([nao, nao]) * &dm).sum_axes(1);
    }
    dao_j_int3c2e_ip1 *= -2.0;
    return dao_j_int3c2e_ip1;
}

pub fn get_grad_daux_j_int3c2e_ip2(tsr_int3c2e_ip2: TsrView<f64>, dm_tp: TsrView<f64>, itm_j: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int3c2e_ip2.f_prefer());

    let naux = itm_j.shape()[0];
    let tmp1 = dm_tp % tsr_int3c2e_ip2.reshape((-1, naux * 3));
    return -1.0 * (&tmp1.reshape((naux, 3)) * itm_j.i((.., None)));
}

pub fn get_itm_k_occtp(tsr_int2c2e_l_inv: TsrView<f64>, ederi_utp: TsrView<f64>, weighted_occ_coeff: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int2c2e_l_inv.f_prefer());
    assert!(ederi_utp.f_prefer());
    assert!(weighted_occ_coeff.f_prefer());

    let nocc = weighted_occ_coeff.shape()[1];
    let naux = ederi_utp.shape()[1];
    let nocc_tp = nocc * (nocc + 1) / 2;
    let device = tsr_int2c2e_l_inv.device().clone();
    let mut tmp: Tsr<f64> = unsafe { rt::empty(([nocc_tp, naux], &device)) };
    (0..naux).into_par_iter().for_each(|p| {
        let ederi_bb = ederi_utp.i((.., p)).unpack_triu(FlagSymm::Sy);
        let ederi_oo = weighted_occ_coeff.t() % ederi_bb % &weighted_occ_coeff;

        let mut tmp = unsafe { tmp.force_mut() };
        tmp.i_mut((.., p)).assign(ederi_oo.pack_triu());
    });
    let itm_k_occ = tmp % tsr_int2c2e_l_inv;
    return itm_k_occ;
}

pub fn get_itm_k_aux(mut itm_k_occtp: TsrMut<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(itm_k_occtp.f_prefer());

    let naux = itm_k_occtp.shape()[1];
    let nocc_tp = itm_k_occtp.shape()[0];
    let nocc = ((2 * nocc_tp) as f64).sqrt().floor().to_usize().unwrap();
    assert_eq!(nocc * (nocc + 1) / 2, nocc_tp);
    let device = itm_k_occtp.device().clone();

    // modify diag elements in-place
    for i in 0..nocc {
        let idx = (i + 2) * (i + 1) / 2 - 1;
        *&mut itm_k_occtp.i_mut(idx) *= f64::sqrt(0.5);
    }

    let itm_k_aux = 2.0 * (itm_k_occtp.t() % &itm_k_occtp);

    // modify back diag elements in-place
    for i in 0..nocc {
        let idx = (i + 2) * (i + 1) / 2 - 1;
        *&mut itm_k_occtp.i_mut(idx) *= f64::sqrt(2.0);
    }

    return itm_k_aux;
}

pub fn get_itm_k_ao(itm_k_occtp: TsrView<f64>, weighted_occ_coeff: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(itm_k_occtp.f_prefer());
    assert!(weighted_occ_coeff.f_prefer());

    let nao = weighted_occ_coeff.shape()[0];
    let naux = itm_k_occtp.shape()[1];
    let device = weighted_occ_coeff.device().clone();

    let itm_k_ao = unsafe { rt::empty(([nao, nao, naux], &device)) };
    (0..naux).into_par_iter().for_each(|p| {
        let mut itm_k_ao = unsafe { itm_k_ao.force_mut() };
        let itm_k_occ_p = itm_k_occtp.i((.., p)).unpack_triu(FlagSymm::Sy);
        itm_k_ao
            .i_mut((.., .., p))
            .assign(&weighted_occ_coeff % itm_k_occ_p % weighted_occ_coeff.t());
    });
    return itm_k_ao;
}

pub fn get_grad_daux_k_int2c2e_ip1(tsr_int2c2e_ip1: TsrView<f64>, itm_k_aux: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int2c2e_ip1.f_prefer());
    assert!(itm_k_aux.f_prefer());

    return (tsr_int2c2e_ip1 * itm_k_aux.i((.., .., None))).sum_axes(1);
}

pub fn get_grad_dao_k_int3c2e_ip1(tsr_int3c2e_ip1: TsrView<f64>, itm_k_ao: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int3c2e_ip1.f_prefer());
    assert!(itm_k_ao.f_prefer());

    let naux = itm_k_ao.shape()[2];
    let nao = tsr_int3c2e_ip1.shape()[0];
    let device = tsr_int3c2e_ip1.device().clone();

    let mut tmp = unsafe { rt::empty(([nao, 3, naux], &device)) };
    (0..naux).into_par_iter().for_each(|p| {
        let mut tmp = unsafe { tmp.force_mut() };
        for t in 0..3 {
            tmp.i_mut((.., t, p))
                .assign(-2.0 * (tsr_int3c2e_ip1.i((.., .., p, t)) * itm_k_ao.i((.., .., p))).sum_axes(1));
        }
    });
    return tmp.sum_axes(-1);
}

pub fn get_grad_daux_k_int3c2e_ip2(tsr_int3c2e_ip2: TsrView<f64>, itm_k_ao: TsrView<f64>) -> Tsr<f64> {
    // see module level documentation for details
    assert!(tsr_int3c2e_ip2.f_prefer());
    assert!(itm_k_ao.f_prefer());

    let naux = itm_k_ao.shape()[2];
    let nao = itm_k_ao.shape()[0];
    let device = tsr_int3c2e_ip2.device().clone();

    // modify diag elements in-place
    let daux_k_int3c2e_ip2 = rt::zeros(([naux, 3], &device));
    (0..naux).into_par_iter().for_each(|p| {
        let mut itm_k_ao_p = itm_k_ao.i((.., .., p)).pack_triu();
        for u in (0..nao) {
            let idx = (u + 2) * (u + 1) / 2 - 1;
            itm_k_ao_p[[idx]] *= 0.5;
        }
        let tmp = -2.0 * (itm_k_ao_p % tsr_int3c2e_ip2.i((.., p)));

        let mut daux_k_int3c2e_ip2 = unsafe { daux_k_int3c2e_ip2.force_mut() };
        *&mut daux_k_int3c2e_ip2.i_mut((p)) += tmp;
    });
    return daux_k_int3c2e_ip2;
}

/* #endregion */

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
            -0.1137786866, -0.1161365056, -0.1125150713,
             0.0004215289,  0.0659156136,  0.0553809300,
             0.0630962392,  0.0488708661, -0.0140011525,
             0.0502609185,  0.0013500258,  0.0711352938,
        ];
        let de_ref = rt::asarray((&de_ref, [3, 4]));
        println!("Maximum Error {:?}", (&de_ref - &de).abs().max_all());
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
            -0.0000000000, -0.0000000000, -0.0515725566,
             0.0000000000, -0.0000000000,  0.0515725566,
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

    fn test_with_scf(scf_data: &SCF) -> RIRHFGradient {
        let mut scf_grad = RIRHFGradient::new(scf_data);
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
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
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
     charge =                    0.0
     spin =                      1.0
     spin_polarization =         false
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
     auxbasis_response =    true
     max_memory =           10240

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
