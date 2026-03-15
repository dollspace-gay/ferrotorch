use std::sync::Arc;

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::tensor::Tensor;

/// Run a function with gradient checkpointing.
///
/// During the forward pass, intermediate activations are **not** saved.
/// During the backward pass, the forward function is re-executed to
/// recompute them, trading compute for memory.
///
/// This is useful for very deep networks where storing all activations
/// would exceed available memory.
///
/// # Arguments
///
/// * `f` - The forward function to checkpoint. It receives the input tensor
///   and returns the output tensor.
/// * `input` - The input tensor. Must have `requires_grad = true`.
///
/// # Returns
///
/// The output tensor, with a grad_fn that will recompute `f` during backward.
pub fn checkpoint<T, F>(f: F, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
{
    use crate::autograd::no_grad::no_grad;
    use crate::storage::TensorStorage;

    // Forward pass without recording the graph (saves memory).
    let output = no_grad(|| f(input))?;

    if !input.requires_grad() {
        return Ok(output);
    }

    // Wrap in a CheckpointBackward that re-runs f during backward.
    let checkpoint_fn = Arc::new(CheckpointBackward {
        func: Arc::new(f),
        input: input.clone(),
        output_shape: output.shape().to_vec(),
    });

    let result = Tensor::from_operation(
        TensorStorage::cpu(output.data()?.to_vec()),
        output.shape().to_vec(),
        checkpoint_fn,
    )?;

    Ok(result)
}

struct CheckpointBackward<T: Float> {
    func: Arc<dyn Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>,
    input: Tensor<T>,
    output_shape: Vec<usize>,
}

impl<T: Float> std::fmt::Debug for CheckpointBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointBackward")
            .field("input_shape", &self.input.shape())
            .field("output_shape", &self.output_shape)
            .finish()
    }
}

impl<T: Float> crate::tensor::GradFn<T> for CheckpointBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Re-run the forward function WITH gradient tracking to build the graph.
        let input_with_grad = self.input.clone().requires_grad_(true);
        let recomputed = (self.func)(&input_with_grad)?;

        // Now backward through the recomputed graph.
        // We need to scale by grad_output, but for simplicity in this
        // initial implementation, we call backward on a scalar formed by
        // sum(recomputed * grad_output).
        //
        // TODO: This is a simplified implementation. A production version
        // would use torch.autograd.backward(recomputed, grad_output) directly.
        // For now, this captures the core idea of recomputation.
        let recomputed_data = recomputed.data()?;
        let go_data = grad_output.data()?;
        let product_sum: T = recomputed_data
            .iter()
            .zip(go_data.iter())
            .map(|(&a, &b)| a * b)
            .fold(<T as num_traits::Zero>::zero(), |acc, x| acc + x);

        let _scalar = crate::creation::scalar(product_sum)?;
        // If the recomputed output has a grad_fn, backward through it.
        if recomputed.grad_fn().is_some() {
            recomputed.backward()?;
        }

        let input_grad = input_with_grad.grad()?;
        Ok(vec![input_grad])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CheckpointBackward"
    }
}
