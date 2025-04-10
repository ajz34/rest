//! Trait abstraction for gradient computation.

use tensors::MatrixFull;

pub trait GradAPI {
    /// Get the gradient of the system.
    /// 
    /// Output dimension: [3, natm]
    ///
    /// This function may not perform actual calculations.
    /// May called after the gradient has been computed and stored in struct;
    /// then use this function to get gradient for next steps (printing, geomopt, etc.).
    fn get_gradient(&self) -> MatrixFull<f64>;

    /// Get the energy of the system.
    fn get_energy(&self) -> f64;
}
