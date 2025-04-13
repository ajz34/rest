use crate::main_driver::performance_essential_calculations;
use crate::scf_io::{initialize_scf, SCF};

pub fn numerical_dipole(scf_data: &SCF, displace: f64) -> [f64; 3] {
    let mut num_dipole_elec = [0.0; 3];

    // three-point stencil
    let stencil = [
        [-displace, 0.0, 0.0],
        [displace, 0.0, 0.0],
        [0.0, -displace, 0.0],
        [0.0, displace, 0.0],
        [0.0, 0.0, -displace],
        [0.0, 0.0, displace],
    ];

    let mut stencil_result = [0.0; 6];

    print!("Evaluating Numerical Dipole\nStencil index: ");

    for n in 0..6 {
        print!("{n}|");
        let mut time_mark = crate::utilities::TimeRecords::new();
        let mut new_scf = scf_data.clone();
        new_scf.mol.ctrl.initial_guess = String::from("inherit");
        new_scf.mol.ctrl.print_level = 0;
        new_scf.mol.ctrl.ext_field_dipole = Some(stencil[n]);
        initialize_scf(&mut new_scf, &None);
        let eng = performance_essential_calculations(&mut new_scf, &mut time_mark, &None);
        stencil_result[n] = eng;
    }
    println!();

    // calculate the dipole moment of electronic contribution
    for n in 0..3 {
        num_dipole_elec[n] = - (stencil_result[n * 2 + 1] - stencil_result[n * 2]) / (2.0 * displace);
    }

    // calculate the dipole moment of nuclear contribution
    let mut dipole_nuc = [0.0; 3];
    let mol = &scf_data.mol;
    let natm = mol.geom.elem.len();
    let coords = (0..natm).map(|i| mol.geom.get_coord(i)).flatten().collect::<Vec<f64>>();

    let charges_by_atom = crate::geom_io::get_charge(&mol.geom.elem);
    let necp_by_atom = mol
        .basis4elem
        .iter()
        .map(|i| if let Some(num_ecp) = i.ecp_electrons { num_ecp as f64 } else { 0.0 })
        .collect::<Vec<f64>>();
    let charges = charges_by_atom.iter().zip(necp_by_atom.iter()).map(|(q, e)| q - e).collect::<Vec<f64>>();

    for i in 0..natm {
        let charge = charges[i];
        for t in 0..3 {
            dipole_nuc[t] += charge * coords[i * 3 + t];
        }
    }

    // calculate the total dipole moment
    let mut num_dipole_tot = [0.0; 3];
    for t in 0..3 {
        num_dipole_tot[t] = num_dipole_elec[t] + dipole_nuc[t];
    }

    return num_dipole_tot;
}

#[cfg(test)]
#[allow(non_snake_case)]
mod debug {
    use super::*;
    use crate::ctrl_io::InputKeywords;
    use crate::molecule_io::Molecule;
    use crate::scf_io::{self, scf_without_build};

    #[test]
    fn test_nh3() {
        let scf_data = initialize_nh3();
        let time = std::time::Instant::now();
        let num_dip = numerical_dipole(&scf_data, 3e-4);
        println!("Time elapsed: {:?}", time.elapsed());
        println!("Num Dipole: {:?}", num_dip);
    }

    #[test]
    fn test_hi() {
        let scf_data = initialize_hi();
        let time = std::time::Instant::now();
        let num_dip = numerical_dipole(&scf_data, 3e-4);
        println!("Time elapsed: {:?}", time.elapsed());
        println!("Num Dipole: {:?}", num_dip);
    }

    fn initialize_nh3() -> SCF {
        let input_token = r##"
[ctrl]
     print_level =          2
     job_type =             "numerical dipole"
     xc =                   "hf"
     basis_path =           "basis-set-pool/def2-TZVP"
     auxbas_path =          "basis-set-pool/def2-SVP-JKFIT"
     guessfile =            "none"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
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
     print_level =          2
     xc =                   "hf"
     basis_path =           "basis-set-pool/def2-TZVP"
     auxbas_path =          "basis-set-pool/def2-SV(P)-JKFIT"
     guessfile =            "none"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     num_threads =          16

[geom]
    name = "HI"
    unit = "Angstrom"
    position = """
        H  0.0  0.0  0.0
        I  0.0  0.0  3.0
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
