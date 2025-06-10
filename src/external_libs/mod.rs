mod ffi_mokit;
use crate::external_libs::ffi_mokit::*;
use crate::geom_io::get_charge;
use std::ffi::{c_double, c_int, c_char, CStr, CString};
use crate::scf_io::SCF;

pub fn py2fch(
    fchname: String,
    nbf: usize,
    nif: usize,
    eigenvector:&[f64], 
    ab: char,
    eigenvalues:&[f64],
    natorb: usize,
    gen_density: usize) 
{
    //let fchname_cstring = CString::new(fchname).expect("CString::new failed");
    let fchname_chars:&Vec<c_char> = &fchname.chars().map(|c| c as c_char).collect();
    unsafe{rest2fch_(
        //fchname_cstring.as_ptr(),
        fchname_chars.as_ptr(),
        &(fchname.len() as i32),
        &(nbf as i32),
        &(nif as i32),
        eigenvector.as_ptr(), 
        &(ab as c_char),
        eigenvalues.as_ptr(), 
        &(natorb as i32),
        &(gen_density as i32)
    )
    }
}

pub fn dftd(scf_data: &SCF) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    // "d3" or "d3bj", then use dftd3_atm;
    // "d4" use dftd4,
    // else panic on invalid input
    if let Some(tmp_emprical) = &scf_data.mol.ctrl.empirical_dispersion {
        if tmp_emprical == "d3" || tmp_emprical == "d3bj" {
            return dftd3_atm(scf_data);
        } else if tmp_emprical == "d4" {
            return dftd4_atm(scf_data); 
        } else {
            panic!("Invalid input for empirical_dispersion: {}.\nDo not invoke the empirical dispersion evaluation!", tmp_emprical);
        }
    } else {
        println!("No empirical_dispersion.");
        return (0.0, None, None);
    }
}

pub fn dftd3_atm(scf_data: &SCF) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    #[cfg(feature = "dftd3")]
    {
        use dftd3::prelude::*;

        let numbers = get_charge(&scf_data.mol.geom.elem).iter().map(|x| *x as usize).collect::<Vec<usize>>();
        let positions = &scf_data.mol.geom.position.data;
        let lattice = None;
        let periodic = None;
        let d3_model = DFTD3Model::new(&numbers, positions, lattice, periodic);

        let xc = scf_data.mol.ctrl.xc.as_str();
        let version = scf_data.mol.ctrl.empirical_dispersion.clone().unwrap();
        // handle special case: d3 -> d3zero
        let version = if version == "d3" {
            "d3zero".to_string()
        } else {
            version
        };
        let params = dftd3_load_param(&version, xc, true);
        d3_model.get_dispersion(&params, true).into()
    }
    #[cfg(not(feature = "dftd3"))]
    {
        println!("Do not invoke the empirical dispersion evaluation!");
        panic!("dftd3 is not enabled in the build.");
    }
}

pub fn dftd4_atm(scf_data: &SCF) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    #[cfg(feature = "dftd4")]
    {
        use dftd4::prelude::*;

        let numbers = get_charge(&scf_data.mol.geom.elem).iter().map(|x| *x as usize).collect::<Vec<usize>>();
        let positions = &scf_data.mol.geom.position.data;
        let charges = None;
        let lattice = None;
        let periodic = None;
        let d4_model = DFTD4Model::new(&numbers, positions, charges, lattice, periodic);

        let xc = scf_data.mol.ctrl.xc.as_str();
        let params = DFTD4Param::load_rational_damping(xc, true);
        d4_model.get_dispersion(&params, true).into()
    }
    #[cfg(not(feature = "dftd4"))]
    {
        println!("Do not invoke the empirical dispersion evaluation!");
        panic!("dftd4 is not enabled in the build.");
    }
}
