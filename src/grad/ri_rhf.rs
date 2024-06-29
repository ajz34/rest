

#[test]
fn test_ri_rhf() {
    use crate::ctrl_io::InputKeywords;
    use crate::molecule_io::Molecule;
    use crate::scf_io;
    use crate::scf_without_build;

    println!("Hello, rust!");
    let ctrl = r###"
[ctrl]
    print_level =               1
    num_threads =               8
    xc =                        "HF"
    basis_path =                "/home/a/rest_pack/basis-set-pool/def2-SVP"
    auxbas_path =               "/home/a/rest_pack/basis-set-pool/def2-SVP-JKFIT"
    use_ri_symm =               true
    use_dm_only =               true
    charge =                    0.0
    spin =                      1.0
    spin_polarization =         false
    mixer =                     "diis"
    mix_param =                 0.6
    initial_guess =             "hcore"
    chkfile =                   "none"

[geom]
    name = "NH3"
    unit = "angstrom"
    position = '''
       N  -2.1988391019      1.8973746268      0.0000000000
       H  -1.1788391019      1.8973746268      0.0000000000
       H  -2.5388353987      1.0925460144     -0.5263586446
       H  -2.5388400276      2.7556271745     -0.4338224694 '''
    "###;
    
    let ctrl_json = toml::from_str::<serde_json::Value>(&ctrl.to_string()[..]).unwrap();
    let (mut ctrl, mut geom) = InputKeywords::parse_ctl_from_json(&ctrl_json).unwrap();
    let mut mol = Molecule::build_native(ctrl, geom).unwrap();
    let mut mf = scf_io::SCF::build(mol);
    scf_without_build(&mut mf);
}
