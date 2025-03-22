use crate::scf_io::SCF;
use crate::dft::DFA4REST;

//trait DeepLearningXC {
//    fn evaluate_model_energy(&self, scf_data: &SCF) -> f64;
//    fn update_xc_potential(&self, scf_data: &SCF ) -> DFA4REST;
//}
//
//// at current implementation, fake by B3LYP
//struct BSHybrid;
//
//impl DeepLearningXC for BSHybrid {
//    fn evaluate_model_energy(&self, scf_data: &SCF) -> f64 {
//        0.0
//    }
//
//    fn update_xc_potential(&self, scf_data: &SCF) -> DFA4REST {
//        let spin_channel = scf_data.mol.spin_channel;
//        let print_level = scf_data.mol.ctrl.print_level;
//        let mut dfa4rest = DFA4REST::new_xc(spin_channel, print_level);
//        //let mut gradient = Vec::new();
//        //for atom in scf_data.atoms.iter() {
//        //    gradient.push(atom.gradient);
//        //}
//        //gradient
//
//        dfa4rest
//    }
//}



pub fn dl_hybrid_xc_energy(energy_components: &Vec<f64>) -> f64 {
    //fake by b3lyp
    let paralist = [
        1.00,   //E_noXC
        0.20,   //Ex_HF
        0.08,   //Ex_LDA
        0.72,   //Ex_B88
        0.19,   //Ec_LDA
        0.81    //Ec_LYP
    ];

    let energy = energy_components.iter().zip(paralist.iter()).fold(0.0, |acc, (x, y)| {
        acc + x * y
    });
    
    energy
}

pub fn dl_hybrid_xc_param(energy_components: &Vec<f64>) -> Vec<f64> {
    //fake by b3lyp
    let paralist = vec![
        1.00,   //E_noXC
        0.20,   //Ex_HF
        0.08,   //Ex_LDA
        0.72,   //Ex_B88
        0.19,   //Ec_LDA
        0.81    //Ec_LYP
    ];

    paralist
}