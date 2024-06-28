use std::ops::AddAssign;

use ndarray::{prelude::*, StrideShape};
use tensors::{MatrixFull, RIFull};
use crate::scf_io;
use crate::scf_io::SCF;
use crate::Molecule;
use rest_libcint::prelude::*;

/// Gradient structure and values for RHF method.
pub struct RIRHFGradient<'a> {
    pub scf_data: &'a SCF,
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
            scf_io::SCFType::RHF => {},
            _ => panic!("SCFtype is not sutiable for RHF gradient.")
        };
        RIRHFGradient {
            scf_data,
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
            de_ovlp.index_axis_mut(Axis(1), atm)
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
                .sum_axis(Axis(0)).sum_axis(Axis(0))  // sum first two axis
            );
        }

        self.de_hcore = MatrixFull::from_vec(de_hcore.dim().into(), de_hcore.into_raw_vec()).unwrap();
        return self;
    }

}

pub fn generator_deriv_hcore<'a> (
    scf_data: &'a SCF
) -> impl FnMut(usize) -> Array3<f64> + 'a
{
    let mut mol = scf_data.mol.clone();
    let mut cint_data = mol.initialize_cint(false);

    let (out, shape) = cint_data.integral_s1::<int1e_ipkin>(None);
    let out = Array::from_shape_vec(shape.f(), out).unwrap();
    let tsr_int1e_ipkin = out.into_dimensionality::<Ix3>().unwrap();

    let (out, shape) = cint_data.integral_s1::<int1e_ipnuc>(None);
    let out = Array::from_shape_vec(shape.f(), out).unwrap();
    let tsr_int1e_ipnuc = out.into_dimensionality::<Ix3>().unwrap();

    let h1 = - (tsr_int1e_ipkin + tsr_int1e_ipnuc);
    let aoslice_by_atom = mol.aoslice_by_atom();
    let charge_by_atom = crate::geom_io::get_charge(&mol.geom.elem);

    move | atm_id | {
        let [_, _, p0, p1] = aoslice_by_atom[atm_id].clone().try_into().unwrap();
        mol.with_rinv_at_nucleus(atm_id, | mol | {
            let mut cint_data = mol.initialize_cint(false);
            
            let (out, shape) = cint_data.integral_s1::<int1e_iprinv>(None);
            let out = Array::from_shape_vec(shape.f(), out).unwrap();
            let tsr_int1e_iprinv = out.into_dimensionality::<Ix3>().unwrap();

            let mut vrinv = - (&charge_by_atom)[atm_id] * tsr_int1e_iprinv;
            vrinv.slice_mut(s![p0..p1, .., ..]).add_assign(&h1.slice(s![p0..p1, .., ..]));
            vrinv += &vrinv.clone().permuted_axes([1, 0, 2]);
            return ndarray_to_colmajor(vrinv);
        })
    }
}

/* #region utilities */

/// Transform ndarray c-contiguous to f-contiguous.
/// 
/// Utility function. Only clone data when input is not f-contiguous. 
pub fn ndarray_to_colmajor<A, D> (arr: Array<A, D>) -> Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    let arr = arr.reversed_axes();  // data not copied
    if arr.is_standard_layout() {
        // arr is f-contiguous = reversed arr is c-contiguous
        // CowArray `into_owned` will not copy if own data, but will copy if it represents view
        // So, though `arr.as_standard_layout().reversed_axes().into_owned()` works, it clones data instead of move it
        return arr.reversed_axes();  // data not copied
    } else {
        // arr is not f-contiguous
        // make reversed arr c-contiguous, then reverse arr again
        return arr.as_standard_layout().reversed_axes().into_owned();
    }
}

#[allow(non_snake_case)]
impl Molecule {
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
        let delimiter = (0..(nbas-1)).into_iter()
            .filter(|&idx| { cint_bas[idx+1][ATOM_OF] != cint_bas[idx][ATOM_OF] })
            .collect::<Vec<usize>>();
        if delimiter.len() != natm - 1 {
            unimplemented!("Missing basis in atoms. Currently it should be internal problem in program.");
        }
        let shl_idx = 0;
        for atm in 0..natm {
            let shl0 = if atm == 0 { 0 } else { delimiter[atm-1] + 1 };
            let shl1 = if atm == natm-1 { nbas } else { delimiter[atm] + 1 };
            let p0 = ao_loc[shl0];
            let p1 = ao_loc[shl1];
            aoslice[atm] = [shl0, shl1, p0, p1];
        }

        // todo: currently we have not consider missing basis in atom
        return aoslice;
    }
}

/* #endregion */

/* #region Matrix Algorithms in RI-RHF code */

pub fn get_dme0<'a> (
    mo_coeff: ArrayView2<'a, f64>,
    mo_occ: ArrayView1<'a, f64>,
    mo_energy: ArrayView1<'a, f64>,
) -> Array2<f64>
{
    // energy-weighted density matrix
    // E_{\mu \nu} &= C_{\mu i} \varepsilon_i n_i C_{\nu i}

    // python code: mo_coeff * mo_occ * mo_energy @ mo_coeff.T

    let dme0 = (
        mo_coeff.to_owned()
        * mo_occ.insert_axis(Axis(0))
        * mo_energy.insert_axis(Axis(0))
    ).dot(&mo_coeff.t());
    return ndarray_to_colmajor(dme0);
}

pub fn get_grad_dao_ovlp<'a> (
    tsr_int1e_ipovlp: ArrayView3<'a, f64>,
    dme0: ArrayView2<'a, f64>,
) -> Array2<f64>
{
    // AO derivative matrix contribution of ovlp
    // \Delta_{t \mu} &\leftarrow - 2 (\partial_t \mu | \nu) E_{\mu \nu}

    // python code (PySCF convention):
    // np.einsum("tuv, uv -> tu", int1e_ipovlp, dme0)
    // equiv code:
    // 2 * (int1e_ipovlp * dme0).sum(axis=-1)
    let dao_ovlp = 2.0 * (
        tsr_int1e_ipovlp.to_owned()
        * dme0.insert_axis(Axis(2))
    ).sum_axis(Axis(1));
    return ndarray_to_colmajor(dao_ovlp);
}

/* #endregion */

#[cfg(test)]
#[allow(non_snake_case)] 
mod debug_nh3 {
    use super::*;
    use crate::scf_io::scf_without_build;
    use crate::ctrl_io::InputKeywords;

    #[test]
    fn test()
    {
        let mut scf_data = initialize_nh3();
        let mut scf_grad = RIRHFGradient::new(&scf_data);
        
        println!("=== de_nuc ===");
        scf_grad.calc_de_nuc();
        let de_nuc = scf_grad.de_nuc.clone();
        let de_nuc = Array2::from_shape_vec(de_nuc.size.f(), de_nuc.data).unwrap();
        println!("{:12.6?}", de_nuc.t());

        println!("=== de_ovlp ===");
        scf_grad.calc_de_ovlp();
        let de_ovlp = scf_grad.de_ovlp.clone();
        let de_ovlp = Array2::from_shape_vec(de_ovlp.size.f(), de_ovlp.data).unwrap();
        println!("{:12.6?}", de_ovlp.t());

        println!("=== deriv_hcore ===");
        scf_grad.calc_de_hcore();
        let de_hcore = scf_grad.de_hcore.clone();
        let de_hcore = Array2::from_shape_vec(de_hcore.size.f(), de_hcore.data).unwrap();
        println!("{:12.6?}", de_hcore.t());
    }

    fn initialize_nh3() -> SCF
    {
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
    use crate::scf_io::scf_without_build;
    use crate::ctrl_io::InputKeywords;

    #[test]
    fn test()
    {
        let mut scf_data = initialize_c12h26();
    }

    fn initialize_c12h26() -> SCF
    {
        let input_token = r##"
[ctrl]
     print_level =          2
     xc =                   "mp2"
     basis_path =           "basis-set-pool/cc-pVDZ"
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