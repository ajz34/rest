use crate::scf_io::SCF;
use crate::Molecule;
use crate::{constants::AUXBAS_THRESHOLD, scf_io};
use ndarray::{prelude::*, Zip};
use num_traits::ToPrimitive;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
use rest_libcint::prelude::*;
use std::mem;
use std::ops::AddAssign;
use tensors::{matrix_blas_lapack::_power_rayon, MatrixFull, RIFull};

/// Gradient structure and values for RHF method.
pub struct RIRHFGradient<'a> {
    pub scf_data: &'a SCF,
    pub de: MatrixFull<f64>,
    pub de_nuc: MatrixFull<f64>,
    pub de_ovlp: MatrixFull<f64>,
    pub de_hcore: MatrixFull<f64>,
    pub de_j: MatrixFull<f64>,
    pub de_k: MatrixFull<f64>,
    pub de_jaux: MatrixFull<f64>,
    pub de_kaux: MatrixFull<f64>,
}

impl<'a> RIRHFGradient<'a> {
    pub fn new(scf_data: &'a SCF) -> RIRHFGradient<'a> {
        match scf_data.scftype {
            scf_io::SCFType::RHF => {}
            _ => panic!("SCFtype is not sutiable for RHF gradient."),
        };
        RIRHFGradient {
            scf_data,
            de: MatrixFull::empty(),
            de_nuc: MatrixFull::empty(),
            de_ovlp: MatrixFull::empty(),
            de_hcore: MatrixFull::empty(),
            de_j: MatrixFull::empty(),
            de_k: MatrixFull::empty(),
            de_jaux: MatrixFull::empty(),
            de_kaux: MatrixFull::empty(),
        }
    }

    /// Derivatives of nuclear repulsion energy with reference to nuclear coordinates
    pub fn calc_de_nuc(&mut self) -> &mut Self {
        let mol = &self.scf_data.mol;
        self.de_nuc = mol.geom.calc_nuc_energy_deriv();
        return self;
    }

    pub fn calc_de_ovlp(&mut self) -> &mut Self {
        // preparation
        let mol = &self.scf_data.mol;
        let mut cint_data = mol.initialize_cint(false);

        // dme0
        let mo_coeff = {
            let mo_coeff = &self.scf_data.eigenvectors[0];
            ArrayView2::from_shape(mo_coeff.size.f(), &mo_coeff.data).unwrap()
        };
        let mo_occ = {
            let mo_occ = &self.scf_data.occupation[0];
            ArrayView1::from_shape(mo_occ.len(), &mo_occ).unwrap()
        };
        let mo_energy = {
            let mo_energy = &self.scf_data.eigenvalues[0];
            ArrayView1::from_shape(mo_energy.len(), &mo_energy).unwrap()
        };
        let dme0 = get_dme0(mo_coeff, mo_occ, mo_energy);

        // tsr_int1e_ipovlp
        let (out, shape) = cint_data.integral_s1::<int1e_ipovlp>(None);
        let out = Array::from_shape_vec(shape.f(), out).unwrap();
        let tsr_int1e_ipovlp = out.into_dimensionality::<Ix3>().unwrap();

        // dao_ovlp
        let dao_ovlp = get_grad_dao_ovlp(tsr_int1e_ipovlp.view(), dme0.view());

        // de_ovlp
        let natm = mol.geom.elem.len();
        let mut de_ovlp = Array2::<f64>::zeros([3, natm].f());
        let ao_slice = mol.aoslice_by_atom();

        for atm in 0..natm {
            let [_, _, p0, p1] = ao_slice[atm].clone().try_into().unwrap();
            de_ovlp
                .index_axis_mut(Axis(1), atm)
                .assign(&dao_ovlp.slice(s![p0..p1, ..]).sum_axis(Axis(0)));
        }
        let de_ovlp = MatrixFull::from_vec([3, natm], de_ovlp.into_raw_vec()).unwrap();
        self.de_ovlp = de_ovlp;
        return self;
    }

    pub fn calc_de_hcore(&mut self) -> &mut Self {
        let mol = &self.scf_data.mol;
        let natm = mol.geom.elem.len();

        let dm = {
            let dm = &self.scf_data.density_matrix[0];
            ArrayView2::from_shape(dm.size.f(), &dm.data).unwrap()
        };

        let mut de_hcore = Array2::<f64>::zeros([3, natm].f());
        let mut gen_deriv_hcore = generator_deriv_hcore(&self.scf_data);
        for atm in 0..natm {
            de_hcore.slice_mut(s![.., atm]).assign(
                &(gen_deriv_hcore(atm) * dm.insert_axis(Axis(2)))
                    .sum_axis(Axis(0))
                    .sum_axis(Axis(0)), // sum first two axis
            );
        }

        self.de_hcore =
            MatrixFull::from_vec(de_hcore.dim().into(), de_hcore.into_raw_vec()).unwrap();
        return self;
    }

    pub fn calc_de_jk(&mut self) -> &mut Self {
        let mol = &self.scf_data.mol;
        let auxmol = mol.make_auxmol_fake();
        let natm = mol.geom.elem.len();
        let mut cint_data = mol.initialize_cint(true);
        let mut cint_data_aux = auxmol.initialize_cint(false);
        let n_basis_shell = mol.cint_bas.len() as i32;
        let n_auxbas_shell = mol.cint_aux_bas.len() as i32;

        // density matrix and trilu-packed density matrix
        let dm = {
            let dm = &self.scf_data.density_matrix[0];
            ArrayView2::from_shape(dm.size.f(), &dm.data).unwrap()
        };
        let dm_tp = pack_tril_tilde(dm.view());

        // orbital coefficients
        let mo_coeff = {
            let mo_coeff = &self.scf_data.eigenvectors[0];
            ArrayView2::from_shape(mo_coeff.size.f(), &mo_coeff.data).unwrap()
        };
        let mo_occ = {
            let mo_occ = &self.scf_data.occupation[0];
            ArrayView1::from_shape(mo_occ.len(), &mo_occ).unwrap()
        };
        let nao = dm.nrows();
        let nocc = mo_occ
            .iter()
            .map(|&x| if x > 0.0 { 1 } else { 0 })
            .sum::<usize>();
        // we assume all occupied orbitals are set to first columns
        // and we merge occupation numbers into occ_coeff.
        let weighted_occ_coeff = mo_coeff.slice(s![.., 0..nocc]).into_owned()
            * mo_occ.slice(s![0..nocc]).mapv(f64::sqrt);

        // eigen-decomposed ERI
        let ederi_all = self.scf_data.rimatr.as_ref().unwrap();
        let ederi_tpu = ArrayView2::from_shape(ederi_all.0.size.f(), &ederi_all.0.data).unwrap();
        let naux = ederi_tpu.ncols();

        // tsr_int2c2e_l: J^-1/2
        let tsr_int2c2e_l_inv = {
            let shl_slices = vec![
                [n_basis_shell, n_basis_shell + n_auxbas_shell],
                [n_basis_shell, n_basis_shell + n_auxbas_shell],
            ];
            let (out, shape) = cint_data.integral_s1::<int2c2e>(Some(&shl_slices));
            let out = MatrixFull::from_vec(shape.try_into().unwrap(), out).unwrap();
            let out = _power_rayon(&out, -0.5, AUXBAS_THRESHOLD).unwrap();
            Array2::from_shape_vec(out.size.f(), out.data).unwrap()
        };

        // tsr_int2c2e_ip1
        let tsr_int2c2e_ip1 = {
            let shl_slices = vec![
                [n_basis_shell, n_basis_shell + n_auxbas_shell],
                [n_basis_shell, n_basis_shell + n_auxbas_shell],
            ];
            let (out, shape) = cint_data.integral_s1::<int2c2e_ip1>(Some(&shl_slices));
            let out = Array::from_shape_vec(shape.f(), out).unwrap();
            out.into_dimensionality::<Ix3>().unwrap()
        };

        // shell partition of int3c2e
        let ao_loc = cint_data.cgto_loc();
        let aux_loc = &ao_loc[(n_basis_shell as usize)..];
        let aux_batch_size =
            calc_batch_size::<f64>(5 * nao * nao, None, None, Some(naux * nocc * nocc));
        let aux_partition = balance_partition(aux_loc, aux_batch_size);

        // preparation finished

        // temporaries for de_jaux, de_kaux
        let itm_jaux = get_itm_jaux(tsr_int2c2e_l_inv.view(), ederi_tpu.view(), dm_tp.view());
        let itm_kaux_occ = get_itm_kaux_occ(
            tsr_int2c2e_l_inv.view(),
            ederi_tpu.view(),
            weighted_occ_coeff.view(),
        );
        let itm_kaux_aux = get_itm_kaux_aux(itm_kaux_occ.view());

        let mut dao_j = Array2::<f64>::zeros([nao, 3].f());
        let mut daux_j = get_grad_daux_j_int2c2e_ip1(tsr_int2c2e_ip1.view(), itm_jaux.view());
        let mut dao_k = Array2::<f64>::zeros([nao, 3].f());
        let mut daux_k = get_grad_daux_k_int2c2e_ip1(tsr_int2c2e_ip1.view(), itm_kaux_aux.view());

        let mut idx_aux_start = 0;
        for [shl0, shl1] in aux_partition {
            let shl_naux = aux_loc[shl1] - aux_loc[shl0];
            let shl_slices = vec![
                [0, n_basis_shell],
                [0, n_basis_shell],
                [n_basis_shell + shl0 as i32, n_basis_shell + shl1 as i32],
            ];
            let (p0, p1) = (idx_aux_start, idx_aux_start + shl_naux);

            // int3c2e_ip1
            let (out, shape) = cint_data.integral_s1::<int3c2e_ip1>(Some(&shl_slices));
            let out = Array::from_shape_vec(shape.f(), out).unwrap();
            let tsr_int3c2e_ip1 = out.into_dimensionality::<Ix4>().unwrap();

            // itm_kaux_ao
            let itm_kaux_ao = get_itm_kaux_ao(
                itm_kaux_occ.slice(s![.., .., p0..p1]),
                weighted_occ_coeff.view(),
            );

            dao_j.add_assign(&get_grad_dao_j_int3c2e_ip1(
                tsr_int3c2e_ip1.view(),
                dm.view(),
                itm_jaux.view(),
            ));
            dao_k.add_assign(&get_grad_dao_k_int3c2e_ip1(
                tsr_int3c2e_ip1.view(),
                itm_kaux_ao.view(),
            ));

            // int3c2e_ip2
            let (out, shape) = cint_data.integral_s2ij::<int3c2e_ip2>(Some(&shl_slices));
            let out = Array::from_shape_vec(shape.f(), out).unwrap();
            let tsr_int3c2e_ip2 = out.into_dimensionality::<Ix3>().unwrap();

            daux_j
                .slice_mut(s![p0..p1, ..])
                .add_assign(&get_grad_daux_j_int3c2e_ip2(
                    tsr_int3c2e_ip2.view(),
                    dm_tp.view(),
                    itm_jaux.slice(s![p0..p1]),
                ));
            daux_k
                .slice_mut(s![p0..p1, ..])
                .add_assign(&get_grad_daux_k_int3c2e_ip2(
                    tsr_int3c2e_ip2.view(),
                    itm_kaux_ao.view(),
                ));

            idx_aux_start += shl_naux;
        }

        // de_j, de_k, de_jaux, de_kaux
        let mut de_j = Array2::<f64>::zeros([3, natm].f());
        let mut de_k = Array2::<f64>::zeros([3, natm].f());
        let mut de_jaux = Array2::<f64>::zeros([3, natm].f());
        let mut de_kaux = Array2::<f64>::zeros([3, natm].f());
        let ao_slice = mol.aoslice_by_atom();
        let aux_slice = mol.make_auxmol_fake().aoslice_by_atom();

        for atm in 0..natm {
            let [_, _, p0, p1] = ao_slice[atm].clone().try_into().unwrap();
            de_j.index_axis_mut(Axis(1), atm)
                .add_assign(&dao_j.slice(s![p0..p1, ..]).sum_axis(Axis(0)));
            de_k.index_axis_mut(Axis(1), atm)
                .add_assign(&dao_k.slice(s![p0..p1, ..]).sum_axis(Axis(0)));
            let [_, _, p0, p1] = aux_slice[atm].clone().try_into().unwrap();
            de_jaux
                .index_axis_mut(Axis(1), atm)
                .add_assign(&daux_j.slice(s![p0..p1, ..]).sum_axis(Axis(0)));
            de_kaux
                .index_axis_mut(Axis(1), atm)
                .add_assign(&daux_k.slice(s![p0..p1, ..]).sum_axis(Axis(0)));
        }

        de_k *= -0.5;
        de_kaux *= -0.5;

        let de_j = MatrixFull::from_vec([3, natm], de_j.into_raw_vec()).unwrap();
        let de_jaux = MatrixFull::from_vec([3, natm], de_jaux.into_raw_vec()).unwrap();
        let de_k = MatrixFull::from_vec([3, natm], de_k.into_raw_vec()).unwrap();
        let de_kaux = MatrixFull::from_vec([3, natm], de_kaux.into_raw_vec()).unwrap();

        self.de_j = de_j;
        self.de_k = de_k;
        self.de_jaux = de_jaux;
        self.de_kaux = de_kaux;

        return self;
    }

    fn calc(&mut self) -> &MatrixFull<f64> {
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

        time_records.count_start("rhf grad calc_de_jk");
        self.calc_de_jk();
        time_records.count("rhf grad calc_de_jk");

        time_records.count("rhf grad");

        self.de = self.de_nuc.clone()
            + self.de_ovlp.clone()
            + self.de_hcore.clone()
            + self.de_j.clone()
            + self.de_k.clone()
            + self.de_jaux.clone()
            + self.de_kaux.clone();

        if self.scf_data.mol.ctrl.print_level >= 2 {
            time_records.report_all();
        }

        return &self.de;
    }
}

pub fn generator_deriv_hcore<'a>(scf_data: &'a SCF) -> impl FnMut(usize) -> Array3<f64> + 'a {
    let mut mol = scf_data.mol.clone();
    let mut cint_data = mol.initialize_cint(false);

    let (out, shape) = cint_data.integral_s1::<int1e_ipkin>(None);
    let out = Array::from_shape_vec(shape.f(), out).unwrap();
    let tsr_int1e_ipkin = out.into_dimensionality::<Ix3>().unwrap();

    let (out, shape) = cint_data.integral_s1::<int1e_ipnuc>(None);
    let out = Array::from_shape_vec(shape.f(), out).unwrap();
    let tsr_int1e_ipnuc = out.into_dimensionality::<Ix3>().unwrap();

    let h1 = -(tsr_int1e_ipkin + tsr_int1e_ipnuc);
    let aoslice_by_atom = mol.aoslice_by_atom();
    let charge_by_atom = crate::geom_io::get_charge(&mol.geom.elem);

    move |atm_id| {
        let [_, _, p0, p1] = aoslice_by_atom[atm_id].clone().try_into().unwrap();
        mol.with_rinv_at_nucleus(atm_id, |mol| {
            let mut cint_data = mol.initialize_cint(false);

            let (out, shape) = cint_data.integral_s1::<int1e_iprinv>(None);
            let out = Array::from_shape_vec(shape.f(), out).unwrap();
            let tsr_int1e_iprinv = out.into_dimensionality::<Ix3>().unwrap();

            let mut vrinv = -(&charge_by_atom)[atm_id] * tsr_int1e_iprinv;
            vrinv
                .slice_mut(s![p0..p1, .., ..])
                .add_assign(&h1.slice(s![p0..p1, .., ..]));
            vrinv += &vrinv.clone().permuted_axes([1, 0, 2]);
            return ndarray_to_colmajor(vrinv);
        })
    }
}

/* #region utilities */

/// Transform ndarray c-contiguous to f-contiguous.
///
/// Utility function. Only clone data when input is not f-contiguous.
pub fn ndarray_to_colmajor<A, D>(arr: Array<A, D>) -> Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    let arr = arr.reversed_axes(); // data not copied
    if arr.is_standard_layout() {
        // arr is f-contiguous = reversed arr is c-contiguous
        // CowArray `into_owned` will not copy if own data, but will copy if it represents view
        // So, though `arr.as_standard_layout().reversed_axes().into_owned()` works, it clones data instead of move it
        return arr.reversed_axes(); // data not copied
    } else {
        // arr is not f-contiguous
        // make reversed arr c-contiguous, then reverse arr again
        return arr.as_standard_layout().reversed_axes().into_owned();
    }
}

#[allow(non_snake_case)]
impl Molecule {
    /// Make an auxiliary molecule from a molecule for calculation.
    /// The auxmol is very similar to the origin mole,
    /// except its basis-related infomation is cloned from auxbasis information of the original mole.
    pub fn make_auxmol_fake(&self) -> Molecule {
        // this code is copied from TYGao's code
        let mut auxmol = self.clone();
        auxmol.num_basis = auxmol.num_auxbas.clone();
        auxmol.fdqc_bas = auxmol.fdqc_aux_bas.clone();
        auxmol.cint_fdqc = auxmol.cint_aux_fdqc.clone();
        auxmol.cint_bas = auxmol.cint_aux_bas.clone();
        auxmol.cint_atm = auxmol.cint_aux_atm.clone();
        auxmol.cint_env = auxmol.cint_aux_env.clone();
        return auxmol;
    }

    pub fn aoslice_by_atom(&self) -> Vec<[usize; 4]> {
        use rest_libcint::cint;

        let ATOM_OF = cint::ATOM_OF as usize;

        let cint_data = self.initialize_cint(false);
        let cint_bas = self.cint_bas.clone();

        let ao_loc = cint_data.cgto_loc();
        let natm = self.geom.elem.len();
        let nbas = cint_bas.len();
        let mut aoslice = vec![[0; 4]; natm];

        // the following code should assume that atoms in `cint_bas` has been sorted by atom index
        let delimiter = (0..(nbas - 1))
            .into_iter()
            .filter(|&idx| cint_bas[idx + 1][ATOM_OF] != cint_bas[idx][ATOM_OF])
            .collect::<Vec<usize>>();
        if delimiter.len() != natm - 1 {
            unimplemented!(
                "Missing basis in atoms. Currently it should be internal problem in program."
            );
        }
        let shl_idx = 0;
        for atm in 0..natm {
            let shl0 = if atm == 0 { 0 } else { delimiter[atm - 1] + 1 };
            let shl1 = if atm == natm - 1 {
                nbas
            } else {
                delimiter[atm] + 1
            };
            let p0 = ao_loc[shl0];
            let p1 = ao_loc[shl1];
            aoslice[atm] = [shl0, shl1, p0, p1];
        }

        // todo: currently we have not consider missing basis in atom
        return aoslice;
    }
}

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
fn calc_batch_size<T>(
    unit_flop: usize,
    mem_avail: Option<f64>,
    mem_factor: Option<f64>,
    pre_flop: Option<usize>,
) -> usize {
    let nbytes_dtype = mem::size_of::<T>();
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
        println!(
            "Current memory available {:10.3} MB, minimum required {:10.3} MB",
            max_mb, unit_mb
        );
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
                println!(
                    "[Warn] Batch size is too small: {} - {} > {}",
                    indices[idx], last, batch_size
                );
                partitions.push(idx);
            } else {
                partitions.push(idx - 1);
                let last = indices[partitions.last().unwrap().clone()];
                if indices[idx] - last > batch_size {
                    println!(
                        "[Warn] Batch size is too small: {} - {} > {}",
                        indices[idx],
                        indices[idx - 1],
                        batch_size
                    );
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

/* #endregion */

/* #region Matrix Algorithms in RI-RHF code */

pub fn get_dme0<'a>(
    mo_coeff: ArrayView2<'a, f64>,
    mo_occ: ArrayView1<'a, f64>,
    mo_energy: ArrayView1<'a, f64>,
) -> Array2<f64> {
    // energy-weighted density matrix
    // E_{\mu \nu} &= C_{\mu i} \varepsilon_i n_i C_{\nu i}

    // python code: mo_coeff * mo_occ * mo_energy @ mo_coeff.T

    let dme0 = (mo_coeff.to_owned() * mo_occ.insert_axis(Axis(0)) * mo_energy.insert_axis(Axis(0)))
        .dot(&mo_coeff.t());
    return ndarray_to_colmajor(dme0);
}

pub fn get_grad_dao_ovlp<'a>(
    tsr_int1e_ipovlp: ArrayView3<'a, f64>,
    dme0: ArrayView2<'a, f64>,
) -> Array2<f64> {
    // AO derivative matrix contribution of ovlp
    // \Delta_{t \mu} &\leftarrow - 2 (\partial_t \mu | \nu) E_{\mu \nu}

    // python code (PySCF convention):
    // np.einsum("tuv, uv -> tu", int1e_ipovlp, dme0)
    // equiv code:
    // 2 * (int1e_ipovlp * dme0).sum(axis=-1)
    let dao_ovlp =
        2.0 * (tsr_int1e_ipovlp.to_owned() * dme0.insert_axis(Axis(2))).sum_axis(Axis(1));
    return ndarray_to_colmajor(dao_ovlp);
}

pub fn pack_tril_tilde(dm: ArrayView2<'_, f64>) -> Array1<f64> {
    // Pack the lower triangular part of a matrix into a 1D array
    // and non-diagonal values are multiplied by 2.
    // D_{\mathrm{tp} (\mu \nu)} \mathop{\tilde{\bowtie}} D_{\mu \nu}

    // python code
    // nao = dm.shape[-1]
    // dm_tril = 2 * lib.pack_tril(dm)
    // indices_diag = [(i + 2) * (i + 1) // 2 - 1 for i in range(nao)]
    // dm_tril[indices_diag] *= 0.5
    // return dm_tril
    assert_eq!(dm.ncols(), dm.nrows());
    let nao = dm.ncols();
    let mut dm_tril = Vec::with_capacity((nao * (nao + 1)) / 2);
    for i in 0..nao {
        for j in 0..=i {
            if i == j {
                dm_tril.push(dm[[i, j]]);
            } else {
                dm_tril.push(2.0 * dm[[i, j]]);
            }
        }
    }
    Array1::from_vec(dm_tril)
}

pub fn get_itm_jaux(
    tsr_int2c2e_l_inv: ArrayView2<f64>,
    cderi_tpu: ArrayView2<f64>,
    dm_tp: ArrayView1<f64>,
) -> Array1<f64> {
    // Temporary half-contracted auxiliary array of coulomb integrals
    // \mathscr{J}_P = (\mathbf{J}^{-1/2})_{QP} Y_{Q, \mathrm{tp} (\mu \nu)} D_{\mathrm{tp} (\mu \nu)}

    // python code (PySCF convention)
    // tmp1 = cderi_utp @ dm_tp
    // itm_jaux = scipy.linalg.solve(int2c2e_l, tmp1)

    // in this function, we does not use `int2c2e_l` directly, but its inverse.

    let tmp1 = dm_tp.dot(&cderi_tpu);
    let itm_jaux = tsr_int2c2e_l_inv.dot(&tmp1);
    return itm_jaux;
}

pub fn get_grad_daux_j_int2c2e_ip1(
    tsr_int2c2e_ip1: ArrayView3<f64>,
    itm_jaux: ArrayView1<f64>,
) -> Array2<f64> {
    // \Delta_{tP} \leftarrow \mathscr{J}_P (\partial_t P_A | Q) \mathscr{J}_Q

    // python code (PySCF convention)
    // naux = itm_jaux.size
    // daux_j_int2c2e_ip1 = np.zeros((3, naux))
    // for t in range(3):
    //     daux_j_int2c2e_ip1[t] = (itm_jaux @ int2c2e_ip1[t]) * itm_jaux

    let naux = itm_jaux.len();
    let mut daux_j_int2c2e_ip1 = Array2::zeros([naux, 3].f());
    for t in 0..3 {
        let res = tsr_int2c2e_ip1.index_axis(Axis(2), t).dot(&itm_jaux) * itm_jaux;
        daux_j_int2c2e_ip1.index_axis_mut(Axis(1), t).assign(&res);
    }
    return daux_j_int2c2e_ip1;
}

pub fn get_grad_dao_j_int3c2e_ip1(
    tsr_int3c2e_ip1: ArrayView4<f64>,
    dm: ArrayView2<f64>,
    itm_jaux: ArrayView1<f64>,
) -> Array2<f64> {
    // \Delta_{t \mu} \leftarrow - 2 D_{\mu \nu} (\partial_t \mu \nu | P) \mathscr{J}_P

    // python code (PySCF convention)
    // nao = dm.shape[-1]
    // naux = itm_jaux.size
    // dao_j_int3c2e_ip1 = np.zeros((3, nao))
    // for t in range(3):
    //     tmp1 = itm_jaux @ int3c2e_ip1_pubb[t].reshape(naux, nao**2)
    //     # dao_j_int3c2e_ip1[t] = (tmp1.reshape((nao, nao)) * dm).sum(axis=0)
    //     dao_j_int3c2e_ip1[t] = np.einsum("vu, vu -> u", tmp1.reshape((nao, nao)), dm)
    // dao_j_int3c2e_ip1 *= -2

    let nao = dm.nrows();
    let naux = itm_jaux.len();
    let mut dao_j_int3c2e_ip1 = Array2::zeros([nao, 3].f());
    for t in 0..3 {
        let tmp = tsr_int3c2e_ip1.index_axis(Axis(3), t);
        let tmp = tmp
            .to_shape(((nao * nao, naux), ndarray::Order::F))
            .unwrap();
        let tmp = tmp.dot(&itm_jaux);
        let tmp = tmp
            .to_shape(((nao, nao), ndarray::Order::F))
            .unwrap()
            .into_owned();
        let tmp = (tmp * dm).sum_axis(Axis(1));
        dao_j_int3c2e_ip1.index_axis_mut(Axis(1), t).assign(&tmp);
    }
    dao_j_int3c2e_ip1 *= -2.0;
    return dao_j_int3c2e_ip1;
}

pub fn get_grad_daux_j_int3c2e_ip2(
    tsr_int3c2e_ip2: ArrayView3<f64>,
    dm_tp: ArrayView1<f64>,
    itm_jaux: ArrayView1<f64>,
) -> Array2<f64> {
    // \Delta_{t P} \leftarrow - D_{\mu \nu} (\mu \nu | \partial_t P_A) \mathscr{J}_P

    // python code (PySCF convention)
    // naux = itm_jaux.size
    // tmp1 = int3c2e_ip2_putp.reshape((3 * naux, -1)) @ dm_tp
    // daux_j_int3c2e_ip2 = tmp1.reshape((3, naux)) * itm_jaux
    // daux_j_int3c2e_ip2 *= -1

    let naux = itm_jaux.len();
    let nao_tp = dm_tp.len();
    let tmp = tsr_int3c2e_ip2
        .to_shape(((nao_tp, naux * 3), ndarray::Order::F))
        .unwrap();
    let tmp = dm_tp.dot(&tmp);
    let tmp = tmp
        .to_shape(((naux, 3), ndarray::Order::F))
        .unwrap()
        .into_owned();
    let daux_j_int3c2e_ip2 = -tmp * itm_jaux.insert_axis(Axis(1));
    return daux_j_int3c2e_ip2;
}

pub fn get_itm_kaux_occ(
    tsr_int2c2e_l_inv: ArrayView2<f64>,
    cderi_tpu: ArrayView2<f64>,
    occ_coeff: ArrayView2<f64>,
) -> Array3<f64> {
    // Temporary half-contracted auxiliary array of exchange integrals
    // \mathscr{K}_{P, ij} = (\mathbf{J}^{-1/2})_{QP} Y_{Q, \mu \nu} C_{\mu i} C_{\nu j}

    // python code (PySCF convention)
    // nocc = occ_coeff.shape[-1]
    // tmp1 = ao2mo._ao2mo.nr_e2(cderi_utp, occ_coeff, (0, nocc, 0, nocc), aosym="s2", mosym="s2")
    // itm_kaux_occ = scipy.linalg.solve(int2c2e_decomp, tmp1, lower=False)

    // in this function, we does not use `int2c2e_l` directly, but its inverse.
    // also, we do not apply triangular-packed and parallel by auxiliary index, so this function is somehow low-efficient.

    let nocc = occ_coeff.ncols();
    let nao = occ_coeff.nrows();
    let naux = cderi_tpu.ncols();
    let mut tmp = Array3::<f64>::zeros([nocc, nocc, naux].f());
    let mut tmp_full = Array2::<f64>::zeros([nao, nao].f());
    for p in 0..naux {
        let mut tmp_full = Array2::zeros([nao, nao].f());
        for u in 0..nao {
            for v in 0..=u {
                let val = cderi_tpu[[u * (u + 1) / 2 + v, p]];
                tmp_full[[u, v]] = val;
                tmp_full[[v, u]] = val;
            }
        }
        let tmp_partial = occ_coeff.t().dot(&tmp_full).dot(&occ_coeff);
        tmp.slice_mut(s![.., .., p]).assign(&tmp_partial);
    }
    let tmp = tmp
        .to_shape(((nocc * nocc, naux), ndarray::Order::F))
        .unwrap();
    let itm_kaux_occ = tmp.dot(&tsr_int2c2e_l_inv);
    let itm_kaux_occ = itm_kaux_occ
        .to_shape(((nocc, nocc, naux), ndarray::Order::F))
        .unwrap()
        .into_owned();
    return itm_kaux_occ;
}

pub fn get_itm_kaux_aux(itm_kaux_occ: ArrayView3<f64>) -> Array2<f64> {
    // \mathscr{K}_{PQ} = \mathscr{K}_{P, ij} \mathscr{K}_{Q, ij}

    // we do not apply triangular-packed, so this function is somehow low-efficient.
    let (nocc, _, naux) = itm_kaux_occ.dim();
    let tmp = itm_kaux_occ
        .to_shape(((nocc * nocc, naux), ndarray::Order::F))
        .unwrap();
    return ndarray_to_colmajor(tmp.t().dot(&tmp));
}

pub fn get_itm_kaux_ao(itm_kaux_occ: ArrayView3<f64>, occ_coeff: ArrayView2<f64>) -> Array3<f64> {
    // \mathscr{K}_{P, \mu \nu} = \mathscr{K}_{P, ij} C_{\mu i} C_{\nu j}

    // we do not apply triangular-packed and parallel by auxiliary index, so this function is somehow low-efficient.
    let (_, _, naux) = itm_kaux_occ.dim();
    let (nao, nocc) = occ_coeff.dim();
    let mut itm_kaux_ao = Array3::<f64>::zeros((nao, nao, naux).f());
    for p in 0..naux {
        itm_kaux_ao.index_axis_mut(Axis(2), p).assign(
            &occ_coeff
                .dot(&itm_kaux_occ.index_axis(Axis(2), p))
                .dot(&occ_coeff.t()),
        )
    }
    return itm_kaux_ao;
}

pub fn get_grad_daux_k_int2c2e_ip1(
    tsr_int2c2e_ip1: ArrayView3<f64>,
    itm_kaux_aux: ArrayView2<f64>,
) -> Array2<f64> {
    // \Delta_{tP} \leftarrow (\partial_t P | Q) \mathscr{K}_{PQ}

    // python code
    // np.einsum("tQP, QP -> tP", int2c2e_ip1, itm_kaux_aux)
    let naux = itm_kaux_aux.nrows();
    let mut result = Array2::<f64>::zeros((naux, 3).f());
    for t in 0..3 {
        result.index_axis_mut(Axis(1), t).assign(
            &(tsr_int2c2e_ip1.index_axis(Axis(2), t).into_owned() * itm_kaux_aux).sum_axis(Axis(1)),
        );
    }
    return result;
}

pub fn get_grad_dao_k_int3c2e_ip1(
    tsr_int3c2e_ip1: ArrayView4<f64>,
    itm_kaux_ao: ArrayView3<f64>,
) -> Array2<f64> {
    // \Delta_{t \mu} \leftarrow - 2 (\partial_t \mu \nu | P) \mathscr{K}_{P, \mu \nu}

    // python code (PySCF convention)
    // itm_kaux_ao = lib.unpack_tril(itm_kaux_aotp)
    // nao = int3c2e_ip1_pubb.shape[-1]
    // dao_k_int3c2e_ip1 = np.zeros((3, nao))
    // for t in range(3):
    //     # dao_k_int3c2e_ip1[t] = (int3c2e_ip1_pubb[t] * itm_kaux_ao).sum(axis=(0, 1))
    //     dao_k_int3c2e_ip1[t] = np.einsum("Pvu, Pvu -> u", int3c2e_ip1_pubb[t], itm_kaux_ao)
    // dao_k_int3c2e_ip1 *= -2

    // we do not apply triangular-packed and parallel by auxiliary index, so this function is somehow low-efficient.

    let (nao, _, naux) = itm_kaux_ao.dim();
    let itm_kaux_ao = itm_kaux_ao
        .to_shape(((nao, nao * naux), ndarray::Order::F))
        .unwrap();
    let tsr_int3c2e_ip1 = tsr_int3c2e_ip1
        .to_shape(((nao, nao * naux, 3), ndarray::Order::F))
        .unwrap();
    let mut result = Array2::<f64>::zeros((nao, 3).f());
    for t in 0..3 {
        let tmp =
            (itm_kaux_ao.to_owned() * tsr_int3c2e_ip1.index_axis(Axis(2), t)).sum_axis(Axis(1));
        result.index_axis_mut(Axis(1), t).assign(&tmp);
    }
    let result = -2.0 * result;
    return result;
}

pub fn get_grad_daux_k_int3c2e_ip2(
    tsr_int3c2e_ip2: ArrayView3<f64>,
    itm_kaux_ao: ArrayView3<f64>,
) -> Array2<f64> {
    // \Delta_{t P} \leftarrow - (\mu \nu | \partial_t P) \mathscr{K}_{P, \mu \nu}

    // python code (PySCF convention)
    // naux, naotp = itm_kaux_aotp.shape
    // nao = int(np.floor(np.sqrt(naotp * 2)))
    // assert nao * (nao + 1) // 2 == naotp
    // # modify tensor in-place
    // indices_diag = [(i + 2) * (i + 1) // 2 - 1 for i in range(nao)]
    // itm_kaux_aotp[:, indices_diag] *= 0.5
    // daux_k_int3c2e_ip2 = np.zeros((3, naux))
    // for t in range(3):
    //     daux_k_int3c2e_ip2[t] = 2 * np.einsum("Pu, Pu -> P", int3c2e_ip2_putp[t], itm_kaux_aotp)
    // itm_kaux_aotp[:, indices_diag] *= 2
    // daux_k_int3c2e_ip2 *= -1

    // we do not apply triangular-packed and parallel by auxiliary index, so this function is somehow low-efficient.

    let (nao, _, naux) = itm_kaux_ao.dim();
    let naotp = nao * (nao + 1) / 2;
    let mut result = Array2::<f64>::zeros((naux, 3).f());
    let mut kaux_aotp = Array1::<f64>::zeros(naotp);
    for p in 0..naux {
        // pack itm_kaux_ao
        for u in 0..nao {
            for v in 0..u {
                kaux_aotp[[u * (u + 1) / 2 + v]] = 2.0 * itm_kaux_ao[[u, v, p]];
            }
            kaux_aotp[[u * (u + 1) / 2 + u]] = itm_kaux_ao[[u, u, p]];
        }
        for t in 0..3 {
            let tmp = (tsr_int3c2e_ip2.slice(s![.., p, t]).to_owned() * kaux_aotp.view()).sum();
            result[[p, t]] = tmp;
        }
    }
    let result = -result;
    return result;
}

/* #endregion */

#[cfg(test)]
#[allow(non_snake_case)]
mod debug_nh3 {
    use super::*;
    use crate::ctrl_io::InputKeywords;
    use crate::scf_io::scf_without_build;

    #[test]
    fn test() {
        let mut scf_data = initialize_nh3();
        let mut scf_grad = RIRHFGradient::new(&scf_data);
        scf_grad.calc();

        println!("=== de ===");
        let de = scf_grad.de.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_nuc ===");
        let de = scf_grad.de_nuc.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_ovlp ===");
        let de = scf_grad.de_ovlp.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_hcore ===");
        let de = scf_grad.de_hcore.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_j ===");
        let de = scf_grad.de_j.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_k ===");
        let de = scf_grad.de_k.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_jaux ===");
        let de = scf_grad.de_jaux.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_kaux ===");
        let de = scf_grad.de_kaux.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());
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
#[allow(non_snake_case)]
mod stress_grad_c12h26 {
    use super::*;
    use crate::ctrl_io::InputKeywords;
    use crate::scf_io::scf_without_build;

    #[test]
    fn test() {
        let mut scf_data = initialize_c12h26();
        let mut scf_grad = RIRHFGradient::new(&scf_data);
        scf_grad.calc();

        println!("=== de ===");
        let de = scf_grad.de.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_nuc ===");
        let de = scf_grad.de_nuc.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_ovlp ===");
        let de = scf_grad.de_ovlp.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_hcore ===");
        let de = scf_grad.de_hcore.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_j ===");
        let de = scf_grad.de_j.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_k ===");
        let de = scf_grad.de_k.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_jaux ===");
        let de = scf_grad.de_jaux.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());

        println!("=== de_kaux ===");
        let de = scf_grad.de_kaux.clone();
        let de = Array2::from_shape_vec(de.size.f(), de.data).unwrap();
        println!("{:12.6?}", de.t());
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
