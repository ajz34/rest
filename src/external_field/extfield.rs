//! External field module
//! 
//! Current implemented external field is only the dipole field.

use rest_libcint::prelude::*;
use tensors::{MatrixFull, RIFull, BasicMatrixOpt};
use crate::molecule_io::Molecule;

/// External field description
/// 
/// Currently only dipole field is implemented.
/// - dipole: Dipole field vector [x, y, z]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ExtField<T> {
    pub dipole: Option<[T; 3]>,
    // pub quadrupole: Option<[T; 9]>,
    // pub octupole: Option<[T; 27]>,
}

impl ExtField<f64> {
    pub fn empty() -> Self {
        Self { dipole: None }
    }

    /// Contribution to 2c integral
    pub fn contribution_2c(self, mol: &Molecule) -> MatrixFull<f64> {
        // handle dipole
        let nao = mol.num_basis;
        let mut cint = mol.initialize_cint(false);
        let mut result = MatrixFull::new([nao, nao], 0.0);

        if let Some(dipole) = self.dipole {
            // handle dipole field
            // println!("Dipole field: {:?}", dipole);
            let dipole_norm = dipole.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            // println!("Dipole norm: {:?}", dipole_norm);
            if dipole_norm < 10.0 * f64::EPSILON {
                println!("Dipole field is close to zero, ignore it");
            } else {
                let tsr_int1e_r = {
                    let (out, shape) = cint.integral_s1::<int1e_r>(None);
                    RIFull::from_vec(shape.try_into().unwrap(), out).unwrap()
                };
                for t in (0..3) {
                    let tsr_int1e_t = tsr_int1e_r.get_reducing_matrix(t).unwrap();
                    result += tsr_int1e_t.to_matrixfull().unwrap() * (-dipole[t]);
                }
            }
        }

        return result;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ctrl_io::InputKeywords;
    use crate::scf_io::scf_without_build;
    use rstsr::prelude::*;
    use tensors::BasicMatrix;

    #[test]
    fn test_nh3() {
        let mut mol = initialize_nh3();
        let mut ext_field = ExtField::empty();
        ext_field.dipole = Some([0.0, 0.0, 1.0]);
        let tsr = {
            let tsr = ext_field.contribution_2c(&mol);
            let shape = tsr.size();
            let vec = tsr.data;
            rt::asarray((vec, shape))
        };
        println!("2c integral: {:16.10?}", tsr);
    }

    fn initialize_nh3() -> Molecule {
        let input_token = r##"
[ctrl]
     xc =                   "hf"
     basis_path =           "basis-set-pool/def2-TZVP"
     auxbas_path =          "basis-set-pool/def2-SVP-JKFIT"
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
        Molecule::build_native(ctrl, geom, None).unwrap()
    }
}
