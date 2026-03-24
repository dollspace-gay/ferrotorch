//! Model export via [`ExportedProgram`] — the `torch.export` equivalent.
//!
//! An [`ExportedProgram`] captures a module's computation graph along with its
//! parameters, input/output specifications, and dynamic shape constraints.
//! Unlike [`TracedModule`](crate::module::TracedModule), an exported program is
//! fully self-contained: it embeds parameter data and can be serialized to a
//! portable binary format that does not depend on the original Rust module.
//!
//! # Dynamic shapes
//!
//! By default, all tensor dimensions are treated as static (fixed at the values
//! observed during tracing). [`DynamicShapeSpec`] allows marking specific
//! dimensions as dynamic, enabling the exported program to accept inputs with
//! varying sizes along those axes (e.g. variable batch size or sequence
//! length).
//!
//! # Example
//!
//! ```ignore
//! use ferrotorch_jit::export::{export, DynamicShapeSpec, ExportedProgram};
//!
//! let program = export(&model, &[example_input], None)?;
//! let outputs = program.run(&[real_input])?;
//!
//! program.save("model.ftep")?;
//! let loaded = ExportedProgram::load("model.ftep")?;
//! ```

use std::collections::HashMap;
use std::path::Path;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

use ferrotorch_nn::module::Module;

use crate::graph::{IrGraph, IrOpKind};
use crate::interpreter::interpret;
use crate::trace::trace;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to the export pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExportError {
    #[error("export: {message}")]
    Export { message: String },

    #[error("export: input validation failed — {message}")]
    InputValidation { message: String },

    #[error("export: constraint violated — {message}")]
    ConstraintViolation { message: String },

    #[error("export: serialization failed — {message}")]
    Serialization { message: String },

    #[error("export: deserialization failed — {message}")]
    Deserialization { message: String },

    #[error("export: unsupported op '{op}' cannot be serialized")]
    UnsupportedOp { op: String },

    #[error(transparent)]
    Core(#[from] FerrotorchError),
}

// ---------------------------------------------------------------------------
// DType (portable, serializable dtype tag)
// ---------------------------------------------------------------------------

/// Portable data-type tag for exported programs.
///
/// This mirrors the element types supported by the tensor system but is
/// decoupled from the generic `T: Float` parameter so it can be stored
/// in serialized metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    Float32 = 0,
    Float64 = 1,
}

impl DType {
    /// Return the size in bytes of a single element of this type.
    pub fn element_size(self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float64 => 8,
        }
    }

    /// Infer the dtype from `std::mem::size_of::<T>()`.
    pub fn from_float<T: Float>() -> Result<Self, ExportError> {
        match std::mem::size_of::<T>() {
            4 => Ok(DType::Float32),
            8 => Ok(DType::Float64),
            other => Err(ExportError::Export {
                message: format!("unsupported element size {other} bytes"),
            }),
        }
    }

    fn to_tag(self) -> u8 {
        self as u8
    }

    fn from_tag(tag: u8) -> Result<Self, ExportError> {
        match tag {
            0 => Ok(DType::Float32),
            1 => Ok(DType::Float64),
            other => Err(ExportError::Deserialization {
                message: format!("unknown dtype tag {other}"),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// DimSpec — static or dynamic dimension
// ---------------------------------------------------------------------------

/// Specification for a single tensor dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum DimSpec {
    /// Fixed dimension — must match exactly at runtime.
    Static(usize),
    /// Dynamic dimension — any value in `[min, max]` is accepted at runtime.
    Dynamic {
        name: String,
        min: usize,
        max: usize,
    },
}

// ---------------------------------------------------------------------------
// InputSpec / OutputSpec
// ---------------------------------------------------------------------------

/// Specification for a single input tensor of the exported program.
#[derive(Debug, Clone)]
pub struct InputSpec {
    /// Human-readable name (e.g. `"input_0"`).
    pub name: String,
    /// Per-dimension specification (static or dynamic).
    pub shape: Vec<DimSpec>,
    /// Element data type.
    pub dtype: DType,
}

/// Specification for a single output tensor of the exported program.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    /// Human-readable name (e.g. `"output_0"`).
    pub name: String,
    /// Shape observed during tracing.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DType,
}

// ---------------------------------------------------------------------------
// ShapeConstraint
// ---------------------------------------------------------------------------

/// A relationship constraint between two dimensions across inputs.
///
/// For example: "batch dimension of input 0 must equal batch dimension of
/// input 1" would be:
///
/// ```ignore
/// ShapeConstraint {
///     dim_a: (0, 0),  // input 0, dim 0
///     dim_b: (1, 0),  // input 1, dim 0
///     relation: ConstraintRelation::Equal,
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeConstraint {
    /// `(input_index, dimension_index)` for the first operand.
    pub dim_a: (usize, usize),
    /// `(input_index, dimension_index)` for the second operand.
    pub dim_b: (usize, usize),
    /// The required relationship.
    pub relation: ConstraintRelation,
}

/// The kind of relationship enforced by a [`ShapeConstraint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintRelation {
    Equal,
    LessThan,
    GreaterThan,
}

// ---------------------------------------------------------------------------
// ExportMetadata
// ---------------------------------------------------------------------------

/// Metadata attached to an exported program.
#[derive(Debug, Clone)]
pub struct ExportMetadata {
    /// Semantic version of the export format.
    pub format_version: u32,
    /// Optional description of the model.
    pub description: String,
    /// Name of the framework that produced this export.
    pub producer: String,
}

impl Default for ExportMetadata {
    fn default() -> Self {
        Self {
            format_version: 1,
            description: String::new(),
            producer: "ferrotorch".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// DynamicShapeSpec
// ---------------------------------------------------------------------------

/// Specification of which input dimensions are dynamic.
///
/// Maps `(input_index, dimension_index)` to a symbolic name. Dimensions not
/// listed are treated as static.
#[derive(Debug, Clone, Default)]
pub struct DynamicShapeSpec {
    /// `input_index -> (dim_index -> symbolic_name)`.
    pub specs: HashMap<usize, HashMap<usize, String>>,
}

impl DynamicShapeSpec {
    /// Create an empty specification (all dimensions static).
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a specific dimension of a specific input as dynamic.
    ///
    /// Returns `&mut Self` for chaining.
    pub fn set_dynamic(&mut self, input_idx: usize, dim_idx: usize, name: &str) -> &mut Self {
        self.specs
            .entry(input_idx)
            .or_default()
            .insert(dim_idx, name.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// ExportedProgram
// ---------------------------------------------------------------------------

/// A fully self-contained, serializable model export.
///
/// Contains the computation graph, embedded parameters, input/output specs,
/// and shape constraints. Can be executed directly via [`run`](Self::run) or
/// serialized to disk via [`save`](Self::save).
#[derive(Debug, Clone)]
pub struct ExportedProgram {
    /// The traced computation graph.
    graph: IrGraph,
    /// Input specifications (name, shape constraints, dtype).
    input_specs: Vec<InputSpec>,
    /// Output specifications.
    output_specs: Vec<OutputSpec>,
    /// Named parameters/buffers embedded in the graph as constant data.
    /// Maps parameter name to flattened f32 data.
    state_dict: HashMap<String, Vec<f32>>,
    /// Shape constraints (guards) validated before execution.
    constraints: Vec<ShapeConstraint>,
    /// Metadata.
    metadata: ExportMetadata,
}

impl ExportedProgram {
    // -- Constructor ---------------------------------------------------------

    /// Construct an `ExportedProgram` from its constituent parts.
    ///
    /// This is the primary non-tracing constructor, useful for building
    /// programs from hand-constructed graphs or for deserialization.
    pub fn from_parts(
        graph: IrGraph,
        input_specs: Vec<InputSpec>,
        output_specs: Vec<OutputSpec>,
        state_dict: HashMap<String, Vec<f32>>,
        constraints: Vec<ShapeConstraint>,
        metadata: ExportMetadata,
    ) -> Self {
        Self {
            graph,
            input_specs,
            output_specs,
            state_dict,
            constraints,
            metadata,
        }
    }

    // -- Accessors ----------------------------------------------------------

    /// The computation graph.
    pub fn graph(&self) -> &IrGraph {
        &self.graph
    }

    /// Input specifications.
    pub fn input_specs(&self) -> &[InputSpec] {
        &self.input_specs
    }

    /// Output specifications.
    pub fn output_specs(&self) -> &[OutputSpec] {
        &self.output_specs
    }

    /// Embedded state dict.
    pub fn state_dict(&self) -> &HashMap<String, Vec<f32>> {
        &self.state_dict
    }

    /// Shape constraints.
    pub fn constraints(&self) -> &[ShapeConstraint] {
        &self.constraints
    }

    /// Metadata.
    pub fn metadata(&self) -> &ExportMetadata {
        &self.metadata
    }

    // -- Constraint management ---------------------------------------------

    /// Add a shape constraint (guard) to be validated before each execution.
    pub fn add_constraint(&mut self, constraint: ShapeConstraint) {
        self.constraints.push(constraint);
    }

    // -- Input validation --------------------------------------------------

    /// Validate that the provided inputs satisfy all specs and constraints.
    ///
    /// Returns `Ok(())` if all checks pass, or a descriptive error.
    pub fn validate_inputs(&self, inputs: &[Tensor<f32>]) -> Result<(), ExportError> {
        // Check input count.
        let expected_count = self.input_specs.len();
        if inputs.len() != expected_count {
            return Err(ExportError::InputValidation {
                message: format!("expected {} inputs, got {}", expected_count, inputs.len()),
            });
        }

        // Check each input against its spec.
        for (i, (input, spec)) in inputs.iter().zip(self.input_specs.iter()).enumerate() {
            let shape = input.shape();
            if shape.len() != spec.shape.len() {
                return Err(ExportError::InputValidation {
                    message: format!(
                        "input '{}' (index {i}): expected {} dimensions, got {}",
                        spec.name,
                        spec.shape.len(),
                        shape.len()
                    ),
                });
            }

            for (d, (actual, dim_spec)) in shape.iter().zip(spec.shape.iter()).enumerate() {
                match dim_spec {
                    DimSpec::Static(expected) => {
                        if actual != expected {
                            return Err(ExportError::InputValidation {
                                message: format!(
                                    "input '{}' (index {i}), dim {d}: expected static size {expected}, got {actual}",
                                    spec.name
                                ),
                            });
                        }
                    }
                    DimSpec::Dynamic { name, min, max } => {
                        if actual < min || actual > max {
                            return Err(ExportError::InputValidation {
                                message: format!(
                                    "input '{}' (index {i}), dim {d} ('{name}'): value {actual} \
                                     outside dynamic range [{min}, {max}]",
                                    spec.name
                                ),
                            });
                        }
                    }
                }
            }
        }

        // Validate constraints.
        self.validate_constraints(inputs)?;

        Ok(())
    }

    /// Check all shape constraints against concrete input tensors.
    fn validate_constraints(&self, inputs: &[Tensor<f32>]) -> Result<(), ExportError> {
        for constraint in &self.constraints {
            let (inp_a, dim_a) = constraint.dim_a;
            let (inp_b, dim_b) = constraint.dim_b;

            if inp_a >= inputs.len() || inp_b >= inputs.len() {
                return Err(ExportError::ConstraintViolation {
                    message: format!(
                        "constraint references input index {} or {}, but only {} inputs provided",
                        inp_a,
                        inp_b,
                        inputs.len()
                    ),
                });
            }

            let shape_a = inputs[inp_a].shape();
            let shape_b = inputs[inp_b].shape();

            if dim_a >= shape_a.len() || dim_b >= shape_b.len() {
                return Err(ExportError::ConstraintViolation {
                    message: format!(
                        "constraint references dim {} of input {} (ndim={}) or dim {} of input {} (ndim={})",
                        dim_a, inp_a, shape_a.len(), dim_b, inp_b, shape_b.len()
                    ),
                });
            }

            let val_a = shape_a[dim_a];
            let val_b = shape_b[dim_b];

            let satisfied = match constraint.relation {
                ConstraintRelation::Equal => val_a == val_b,
                ConstraintRelation::LessThan => val_a < val_b,
                ConstraintRelation::GreaterThan => val_a > val_b,
            };

            if !satisfied {
                let rel_str = match constraint.relation {
                    ConstraintRelation::Equal => "==",
                    ConstraintRelation::LessThan => "<",
                    ConstraintRelation::GreaterThan => ">",
                };
                return Err(ExportError::ConstraintViolation {
                    message: format!(
                        "input[{}].shape[{}] ({}) {} input[{}].shape[{}] ({}) is not satisfied",
                        inp_a, dim_a, val_a, rel_str, inp_b, dim_b, val_b
                    ),
                });
            }
        }

        Ok(())
    }

    // -- Execution ---------------------------------------------------------

    /// Execute the exported program with the given inputs.
    ///
    /// Validates inputs against specs and constraints, then interprets the
    /// graph.
    pub fn run(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>, ExportError> {
        self.validate_inputs(inputs)?;
        let result = interpret(&self.graph, inputs)?;
        Ok(vec![result])
    }

    // -- Serialization -----------------------------------------------------

    /// Serialize the exported program to a portable binary file.
    ///
    /// # Binary format
    ///
    /// | Section | Content |
    /// |---------|---------|
    /// | Magic | `b"FTEP"` (4 bytes) |
    /// | Version | `u32` LE |
    /// | Graph | IR graph binary (length-prefixed) |
    /// | Metadata | JSON string (length-prefixed) |
    /// | Input specs | count + per-spec data |
    /// | Output specs | count + per-spec data |
    /// | State dict | count + per-entry (name + data) |
    /// | Constraints | count + per-constraint data |
    pub fn save(&self, path: &Path) -> Result<(), ExportError> {
        let mut buf = Vec::with_capacity(4096);

        // Magic.
        buf.extend_from_slice(b"FTEP");

        // Format version.
        buf.extend_from_slice(&1u32.to_le_bytes());

        // Graph (IR serialization).
        let graph_bytes = self.graph.serialize();
        buf.extend_from_slice(&(graph_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&graph_bytes);

        // Metadata as JSON.
        let meta_json = format!(
            "{{\"format_version\":{},\"description\":{},\"producer\":{}}}",
            self.metadata.format_version,
            json_escape(&self.metadata.description),
            json_escape(&self.metadata.producer),
        );
        let meta_bytes = meta_json.as_bytes();
        buf.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(meta_bytes);

        // Input specs.
        buf.extend_from_slice(&(self.input_specs.len() as u32).to_le_bytes());
        for spec in &self.input_specs {
            write_input_spec(&mut buf, spec);
        }

        // Output specs.
        buf.extend_from_slice(&(self.output_specs.len() as u32).to_le_bytes());
        for spec in &self.output_specs {
            write_output_spec(&mut buf, spec);
        }

        // State dict.
        buf.extend_from_slice(&(self.state_dict.len() as u32).to_le_bytes());
        // Sort keys for deterministic output.
        let mut keys: Vec<&String> = self.state_dict.keys().collect();
        keys.sort();
        for key in keys {
            let data = &self.state_dict[key];
            write_string(&mut buf, key);
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            for &v in data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // Constraints.
        buf.extend_from_slice(&(self.constraints.len() as u32).to_le_bytes());
        for c in &self.constraints {
            write_constraint(&mut buf, c);
        }

        std::fs::write(path, &buf).map_err(|e| ExportError::Serialization {
            message: format!("failed to write {}: {e}", path.display()),
        })
    }

    /// Deserialize an exported program from a binary file previously produced
    /// by [`save`](Self::save).
    pub fn load(path: &Path) -> Result<Self, ExportError> {
        let data = std::fs::read(path).map_err(|e| ExportError::Deserialization {
            message: format!("failed to read {}: {e}", path.display()),
        })?;
        Self::from_bytes(&data)
    }

    /// Deserialize from a byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ExportError> {
        let mut r = ReadCursor::new(data);

        // Magic.
        let magic = r.read_bytes(4)?;
        if magic != b"FTEP" {
            return Err(ExportError::Deserialization {
                message: format!("invalid magic bytes {:?} (expected FTEP)", magic),
            });
        }

        // Version.
        let version = r.read_u32()?;
        if version != 1 {
            return Err(ExportError::Deserialization {
                message: format!("unsupported format version {version} (expected 1)"),
            });
        }

        // Graph.
        let graph_len = r.read_u32()? as usize;
        let graph_bytes = r.read_bytes(graph_len)?;
        let graph = IrGraph::deserialize(graph_bytes).map_err(|e| {
            ExportError::Deserialization {
                message: format!("failed to deserialize IR graph: {e}"),
            }
        })?;

        // Metadata.
        let meta_len = r.read_u32()? as usize;
        let meta_bytes = r.read_bytes(meta_len)?;
        let meta_str = std::str::from_utf8(meta_bytes).map_err(|e| {
            ExportError::Deserialization {
                message: format!("metadata is not valid UTF-8: {e}"),
            }
        })?;
        let metadata = parse_metadata_json(meta_str)?;

        // Input specs.
        let input_count = r.read_u32()? as usize;
        let mut input_specs = Vec::with_capacity(input_count);
        for _ in 0..input_count {
            input_specs.push(read_input_spec(&mut r)?);
        }

        // Output specs.
        let output_count = r.read_u32()? as usize;
        let mut output_specs = Vec::with_capacity(output_count);
        for _ in 0..output_count {
            output_specs.push(read_output_spec(&mut r)?);
        }

        // State dict.
        let state_count = r.read_u32()? as usize;
        let mut state_dict = HashMap::with_capacity(state_count);
        for _ in 0..state_count {
            let name = read_string(&mut r)?;
            let data_len = r.read_u32()? as usize;
            let mut values = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                values.push(r.read_f32()?);
            }
            state_dict.insert(name, values);
        }

        // Constraints.
        let constraint_count = r.read_u32()? as usize;
        let mut constraints = Vec::with_capacity(constraint_count);
        for _ in 0..constraint_count {
            constraints.push(read_constraint(&mut r)?);
        }

        Ok(ExportedProgram {
            graph,
            input_specs,
            output_specs,
            state_dict,
            constraints,
            metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// export() — the main public API
// ---------------------------------------------------------------------------

/// Trace a module and produce an [`ExportedProgram`].
///
/// This is the `torch.export.export()` equivalent. It:
/// 1. Traces the module with example inputs to capture the computation graph.
/// 2. Extracts parameters from the module and embeds them as constants.
/// 3. If `dynamic_shapes` is provided, marks the specified dimensions as
///    dynamic with min/max bounds inferred from the example inputs.
/// 4. Validates that all operations in the graph are exportable.
///
/// # Arguments
///
/// * `module` — The module to export. Must implement `Module<f32>`.
/// * `example_inputs` — Concrete input tensors used for tracing. At least one
///   must have `requires_grad = true`.
/// * `dynamic_shapes` — Optional dynamic shape specification. When `None`, all
///   dimensions are treated as static.
///
/// # Errors
///
/// Returns an error if tracing fails, an operation cannot be exported, or
/// shape constraints are invalid.
pub fn export(
    module: &dyn Module<f32>,
    example_inputs: &[Tensor<f32>],
    dynamic_shapes: Option<&DynamicShapeSpec>,
) -> Result<ExportedProgram, ExportError> {
    // Ensure at least one input has requires_grad so tracing works.
    // We clone inputs and enable grad on the first one if none have it.
    let mut trace_inputs: Vec<Tensor<f32>> = Vec::with_capacity(example_inputs.len());
    let mut any_grad = false;
    for inp in example_inputs {
        if inp.requires_grad() {
            any_grad = true;
        }
        trace_inputs.push(inp.clone());
    }

    if !any_grad && !trace_inputs.is_empty() {
        trace_inputs[0] = trace_inputs[0].clone().requires_grad_(true);
    }

    // Trace the module's forward pass.
    let graph = trace(
        |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            module.forward(&inputs[0])
        },
        &trace_inputs,
    )?;

    // Validate all ops are exportable (no FusedElementwise which can't be
    // round-tripped through the export format).
    validate_exportable(&graph)?;

    // Extract parameters from the module and embed them.
    let named_params = module.named_parameters();
    let mut state_dict = HashMap::with_capacity(named_params.len());
    for (name, param) in &named_params {
        let tensor = param.tensor();
        let data = tensor.data().map_err(|e| ExportError::Export {
            message: format!("failed to read parameter '{name}': {e}"),
        })?;
        state_dict.insert(name.clone(), data.to_vec());
    }

    // Build input specs.
    let dtype = DType::Float32;
    let mut input_specs = Vec::with_capacity(example_inputs.len());
    for (i, inp) in example_inputs.iter().enumerate() {
        let shape = inp.shape();
        let dim_specs: Vec<DimSpec> = shape
            .iter()
            .enumerate()
            .map(|(d, &size)| {
                if let Some(dyn_shapes) = dynamic_shapes {
                    if let Some(dim_map) = dyn_shapes.specs.get(&i) {
                        if let Some(name) = dim_map.get(&d) {
                            // Dynamic: min = 1, max = 2 * observed (heuristic
                            // matching PyTorch's behavior of allowing >=1).
                            // The user can always refine via constraints.
                            let max_val = if size == 0 { 0 } else { size.max(1) * 2 };
                            return DimSpec::Dynamic {
                                name: name.clone(),
                                min: if size == 0 { 0 } else { 1 },
                                max: max_val,
                            };
                        }
                    }
                }
                DimSpec::Static(size)
            })
            .collect();

        input_specs.push(InputSpec {
            name: format!("input_{i}"),
            shape: dim_specs,
            dtype,
        });
    }

    // Build output specs from the graph's output values.
    let mut output_specs = Vec::with_capacity(graph.output_values.len());
    for (i, &out_id) in graph.output_values.iter().enumerate() {
        let shape = graph
            .values
            .iter()
            .find(|v| v.id == out_id)
            .map(|v| v.shape.clone())
            .unwrap_or_default();
        output_specs.push(OutputSpec {
            name: format!("output_{i}"),
            shape,
            dtype,
        });
    }

    Ok(ExportedProgram {
        graph,
        input_specs,
        output_specs,
        state_dict,
        constraints: Vec::new(),
        metadata: ExportMetadata::default(),
    })
}

/// Trace a function (instead of a Module) and produce an [`ExportedProgram`].
///
/// This is useful when the computation is defined as a closure rather than a
/// module with named parameters.
pub fn export_function<F>(
    f: F,
    example_inputs: &[Tensor<f32>],
    dynamic_shapes: Option<&DynamicShapeSpec>,
) -> Result<ExportedProgram, ExportError>
where
    F: Fn(&[Tensor<f32>]) -> FerrotorchResult<Tensor<f32>>,
{
    let mut trace_inputs: Vec<Tensor<f32>> = Vec::with_capacity(example_inputs.len());
    let mut any_grad = false;
    for inp in example_inputs {
        if inp.requires_grad() {
            any_grad = true;
        }
        trace_inputs.push(inp.clone());
    }

    if !any_grad && !trace_inputs.is_empty() {
        trace_inputs[0] = trace_inputs[0].clone().requires_grad_(true);
    }

    let graph = trace(&f, &trace_inputs)?;
    validate_exportable(&graph)?;

    let dtype = DType::Float32;
    let mut input_specs = Vec::with_capacity(example_inputs.len());
    for (i, inp) in example_inputs.iter().enumerate() {
        let shape = inp.shape();
        let dim_specs: Vec<DimSpec> = shape
            .iter()
            .enumerate()
            .map(|(d, &size)| {
                if let Some(dyn_shapes) = dynamic_shapes {
                    if let Some(dim_map) = dyn_shapes.specs.get(&i) {
                        if let Some(name) = dim_map.get(&d) {
                            let max_val = if size == 0 { 0 } else { size.max(1) * 2 };
                            return DimSpec::Dynamic {
                                name: name.clone(),
                                min: if size == 0 { 0 } else { 1 },
                                max: max_val,
                            };
                        }
                    }
                }
                DimSpec::Static(size)
            })
            .collect();

        input_specs.push(InputSpec {
            name: format!("input_{i}"),
            shape: dim_specs,
            dtype,
        });
    }

    let mut output_specs = Vec::with_capacity(graph.output_values.len());
    for (i, &out_id) in graph.output_values.iter().enumerate() {
        let shape = graph
            .values
            .iter()
            .find(|v| v.id == out_id)
            .map(|v| v.shape.clone())
            .unwrap_or_default();
        output_specs.push(OutputSpec {
            name: format!("output_{i}"),
            shape,
            dtype,
        });
    }

    Ok(ExportedProgram {
        graph,
        input_specs,
        output_specs,
        state_dict: HashMap::new(),
        constraints: Vec::new(),
        metadata: ExportMetadata::default(),
    })
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Verify that all operations in the graph are exportable.
///
/// FusedElementwise nodes cannot be serialized through the export format
/// because they contain nested op lists. They should be un-fused before
/// export.
fn validate_exportable(graph: &IrGraph) -> Result<(), ExportError> {
    for node in &graph.nodes {
        if let IrOpKind::FusedElementwise { .. } = &node.op {
            return Err(ExportError::UnsupportedOp {
                op: "FusedElementwise".to_string(),
            });
        }
    }
    Ok(())
}

// ===========================================================================
// Serialization helpers
// ===========================================================================

// -- Read cursor -----------------------------------------------------------

struct ReadCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ReadCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], ExportError> {
        if self.remaining() < n {
            return Err(ExportError::Deserialization {
                message: format!(
                    "unexpected EOF at offset {} (need {} bytes, have {})",
                    self.pos,
                    n,
                    self.remaining()
                ),
            });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, ExportError> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    fn read_u32(&mut self) -> Result<u32, ExportError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f32(&mut self) -> Result<f32, ExportError> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

// -- Write helpers ---------------------------------------------------------

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn read_string(r: &mut ReadCursor<'_>) -> Result<String, ExportError> {
    let len = r.read_u32()? as usize;
    let bytes = r.read_bytes(len)?;
    String::from_utf8(bytes.to_vec()).map_err(|e| ExportError::Deserialization {
        message: format!("invalid UTF-8 in string: {e}"),
    })
}

fn write_dim_spec(buf: &mut Vec<u8>, spec: &DimSpec) {
    match spec {
        DimSpec::Static(size) => {
            buf.push(0); // tag: static
            buf.extend_from_slice(&(*size as u32).to_le_bytes());
        }
        DimSpec::Dynamic { name, min, max } => {
            buf.push(1); // tag: dynamic
            write_string(buf, name);
            buf.extend_from_slice(&(*min as u32).to_le_bytes());
            buf.extend_from_slice(&(*max as u32).to_le_bytes());
        }
    }
}

fn read_dim_spec(r: &mut ReadCursor<'_>) -> Result<DimSpec, ExportError> {
    let tag = r.read_u8()?;
    match tag {
        0 => {
            let size = r.read_u32()? as usize;
            Ok(DimSpec::Static(size))
        }
        1 => {
            let name = read_string(r)?;
            let min = r.read_u32()? as usize;
            let max = r.read_u32()? as usize;
            Ok(DimSpec::Dynamic { name, min, max })
        }
        other => Err(ExportError::Deserialization {
            message: format!("unknown DimSpec tag {other}"),
        }),
    }
}

fn write_input_spec(buf: &mut Vec<u8>, spec: &InputSpec) {
    write_string(buf, &spec.name);
    buf.extend_from_slice(&(spec.shape.len() as u32).to_le_bytes());
    for dim in &spec.shape {
        write_dim_spec(buf, dim);
    }
    buf.push(spec.dtype.to_tag());
}

fn read_input_spec(r: &mut ReadCursor<'_>) -> Result<InputSpec, ExportError> {
    let name = read_string(r)?;
    let ndim = r.read_u32()? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(read_dim_spec(r)?);
    }
    let dtype = DType::from_tag(r.read_u8()?)?;
    Ok(InputSpec { name, shape, dtype })
}

fn write_output_spec(buf: &mut Vec<u8>, spec: &OutputSpec) {
    write_string(buf, &spec.name);
    buf.extend_from_slice(&(spec.shape.len() as u32).to_le_bytes());
    for &dim in &spec.shape {
        buf.extend_from_slice(&(dim as u32).to_le_bytes());
    }
    buf.push(spec.dtype.to_tag());
}

fn read_output_spec(r: &mut ReadCursor<'_>) -> Result<OutputSpec, ExportError> {
    let name = read_string(r)?;
    let ndim = r.read_u32()? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(r.read_u32()? as usize);
    }
    let dtype = DType::from_tag(r.read_u8()?)?;
    Ok(OutputSpec { name, shape, dtype })
}

fn write_constraint(buf: &mut Vec<u8>, c: &ShapeConstraint) {
    buf.extend_from_slice(&(c.dim_a.0 as u32).to_le_bytes());
    buf.extend_from_slice(&(c.dim_a.1 as u32).to_le_bytes());
    buf.extend_from_slice(&(c.dim_b.0 as u32).to_le_bytes());
    buf.extend_from_slice(&(c.dim_b.1 as u32).to_le_bytes());
    let rel_tag: u8 = match c.relation {
        ConstraintRelation::Equal => 0,
        ConstraintRelation::LessThan => 1,
        ConstraintRelation::GreaterThan => 2,
    };
    buf.push(rel_tag);
}

fn read_constraint(r: &mut ReadCursor<'_>) -> Result<ShapeConstraint, ExportError> {
    let a0 = r.read_u32()? as usize;
    let a1 = r.read_u32()? as usize;
    let b0 = r.read_u32()? as usize;
    let b1 = r.read_u32()? as usize;
    let rel_tag = r.read_u8()?;
    let relation = match rel_tag {
        0 => ConstraintRelation::Equal,
        1 => ConstraintRelation::LessThan,
        2 => ConstraintRelation::GreaterThan,
        other => {
            return Err(ExportError::Deserialization {
                message: format!("unknown ConstraintRelation tag {other}"),
            })
        }
    };
    Ok(ShapeConstraint {
        dim_a: (a0, a1),
        dim_b: (b0, b1),
        relation,
    })
}

// -- JSON helpers ----------------------------------------------------------

/// Minimal JSON string escaping (no external dependency needed).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Parse the minimal JSON metadata object.
fn parse_metadata_json(s: &str) -> Result<ExportMetadata, ExportError> {
    // We produce a very constrained JSON shape, so we parse it with simple
    // string matching rather than pulling in a JSON crate.
    let extract = |key: &str| -> Result<String, ExportError> {
        let needle = format!("\"{}\":", key);
        let start = s.find(&needle).ok_or_else(|| ExportError::Deserialization {
            message: format!("metadata missing key '{key}'"),
        })?;
        let after_key = start + needle.len();
        let rest = &s[after_key..];

        // Numeric values.
        if rest.starts_with(|c: char| c.is_ascii_digit()) {
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            return Ok(rest[..end].to_string());
        }

        // String values.
        if rest.starts_with('"') {
            // Find the closing quote (handling escaped quotes).
            let content = &rest[1..]; // skip opening quote
            let mut escaped = false;
            let mut end = 0;
            for (i, c) in content.char_indices() {
                if escaped {
                    escaped = false;
                    continue;
                }
                if c == '\\' {
                    escaped = true;
                    continue;
                }
                if c == '"' {
                    end = i;
                    break;
                }
            }
            let raw = &content[..end];
            // Unescape.
            let unescaped = raw
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t");
            return Ok(unescaped);
        }

        Err(ExportError::Deserialization {
            message: format!("unexpected value format for key '{key}'"),
        })
    };

    let format_version: u32 = extract("format_version")?
        .parse()
        .map_err(|e| ExportError::Deserialization {
            message: format!("invalid format_version: {e}"),
        })?;
    let description = extract("description")?;
    let producer = extract("producer")?;

    Ok(ExportMetadata {
        format_version,
        description,
        producer,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{IrGraph, IrOpKind};
    use ferrotorch_core::error::FerrotorchResult;
    use ferrotorch_core::grad_fns::arithmetic::{add, mul};
    use ferrotorch_core::grad_fns::reduction::sum;
    use ferrotorch_core::storage::TensorStorage;
    use ferrotorch_core::tensor::Tensor;

    // -- Test helpers -------------------------------------------------------

    fn grad_vec(data: Vec<f32>) -> Tensor<f32> {
        let n = data.len();
        Tensor::from_storage(TensorStorage::cpu(data), vec![n], true)
            .unwrap()
            .requires_grad_(true)
    }

    fn tensor_1d(data: &[f32]) -> Tensor<f32> {
        ferrotorch_core::from_vec(data.to_vec(), &[data.len()]).unwrap()
    }

    fn tensor_2d(data: &[f32], rows: usize, cols: usize) -> Tensor<f32> {
        ferrotorch_core::from_vec(data.to_vec(), &[rows, cols]).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: got {a}, expected {e} (diff {})",
                (a - e).abs()
            );
        }
    }

    /// Build a simple ExportedProgram from a hand-built graph: y = x + x.
    fn make_add_self_program() -> ExportedProgram {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        ExportedProgram {
            graph: g,
            input_specs: vec![InputSpec {
                name: "input_0".to_string(),
                shape: vec![DimSpec::Static(3)],
                dtype: DType::Float32,
            }],
            output_specs: vec![OutputSpec {
                name: "output_0".to_string(),
                shape: vec![3],
                dtype: DType::Float32,
            }],
            state_dict: HashMap::new(),
            constraints: Vec::new(),
            metadata: ExportMetadata::default(),
        }
    }

    /// Build an ExportedProgram with two inputs and a dynamic batch dim:
    /// y = a + b, where dim 0 of both inputs is dynamic.
    fn make_dynamic_program() -> ExportedProgram {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![4, 3]);
        let b = g.add_input(vec![4, 3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![4, 3]]);
        g.set_outputs(vec![add_outs[0]]);

        ExportedProgram {
            graph: g,
            input_specs: vec![
                InputSpec {
                    name: "input_0".to_string(),
                    shape: vec![
                        DimSpec::Dynamic {
                            name: "batch".to_string(),
                            min: 1,
                            max: 16,
                        },
                        DimSpec::Static(3),
                    ],
                    dtype: DType::Float32,
                },
                InputSpec {
                    name: "input_1".to_string(),
                    shape: vec![
                        DimSpec::Dynamic {
                            name: "batch".to_string(),
                            min: 1,
                            max: 16,
                        },
                        DimSpec::Static(3),
                    ],
                    dtype: DType::Float32,
                },
            ],
            output_specs: vec![OutputSpec {
                name: "output_0".to_string(),
                shape: vec![4, 3],
                dtype: DType::Float32,
            }],
            state_dict: HashMap::new(),
            constraints: vec![ShapeConstraint {
                dim_a: (0, 0),
                dim_b: (1, 0),
                relation: ConstraintRelation::Equal,
            }],
            metadata: ExportMetadata::default(),
        }
    }

    // -- DynamicShapeSpec tests ----------------------------------------------

    #[test]
    fn test_dynamic_shape_spec_new_is_empty() {
        let spec = DynamicShapeSpec::new();
        assert!(spec.specs.is_empty());
    }

    #[test]
    fn test_dynamic_shape_spec_set_dynamic() {
        let mut spec = DynamicShapeSpec::new();
        spec.set_dynamic(0, 0, "batch");
        spec.set_dynamic(0, 1, "seq_len");
        spec.set_dynamic(1, 0, "batch");

        assert_eq!(spec.specs.len(), 2); // two input indices
        assert_eq!(spec.specs[&0].len(), 2); // two dynamic dims for input 0
        assert_eq!(spec.specs[&1].len(), 1); // one dynamic dim for input 1
        assert_eq!(spec.specs[&0][&0], "batch");
        assert_eq!(spec.specs[&0][&1], "seq_len");
        assert_eq!(spec.specs[&1][&0], "batch");
    }

    #[test]
    fn test_dynamic_shape_spec_chaining() {
        let mut spec = DynamicShapeSpec::new();
        spec.set_dynamic(0, 0, "batch")
            .set_dynamic(0, 1, "seq");

        assert_eq!(spec.specs[&0].len(), 2);
    }

    // -- DimSpec tests -------------------------------------------------------

    #[test]
    fn test_dim_spec_static() {
        let dim = DimSpec::Static(42);
        assert_eq!(dim, DimSpec::Static(42));
    }

    #[test]
    fn test_dim_spec_dynamic() {
        let dim = DimSpec::Dynamic {
            name: "batch".into(),
            min: 1,
            max: 128,
        };
        match &dim {
            DimSpec::Dynamic { name, min, max } => {
                assert_eq!(name, "batch");
                assert_eq!(*min, 1);
                assert_eq!(*max, 128);
            }
            _ => panic!("expected Dynamic"),
        }
    }

    // -- DType tests ---------------------------------------------------------

    #[test]
    fn test_dtype_element_size() {
        assert_eq!(DType::Float32.element_size(), 4);
        assert_eq!(DType::Float64.element_size(), 8);
    }

    #[test]
    fn test_dtype_from_float() {
        assert_eq!(DType::from_float::<f32>().unwrap(), DType::Float32);
        assert_eq!(DType::from_float::<f64>().unwrap(), DType::Float64);
    }

    #[test]
    fn test_dtype_tag_round_trip() {
        for dtype in [DType::Float32, DType::Float64] {
            let tag = dtype.to_tag();
            let restored = DType::from_tag(tag).unwrap();
            assert_eq!(dtype, restored);
        }
    }

    #[test]
    fn test_dtype_from_invalid_tag() {
        assert!(DType::from_tag(99).is_err());
    }

    // -- ExportedProgram::validate_inputs tests --------------------------------

    #[test]
    fn test_validate_inputs_correct() {
        let program = make_add_self_program();
        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        assert!(program.validate_inputs(&[input]).is_ok());
    }

    #[test]
    fn test_validate_inputs_wrong_count() {
        let program = make_add_self_program();
        let err = program.validate_inputs(&[]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("expected 1 inputs, got 0"), "got: {msg}");
    }

    #[test]
    fn test_validate_inputs_wrong_ndim() {
        let program = make_add_self_program();
        let input = tensor_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let err = program.validate_inputs(&[input]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("expected 1 dimensions, got 2"), "got: {msg}");
    }

    #[test]
    fn test_validate_inputs_wrong_static_size() {
        let program = make_add_self_program();
        let input = tensor_1d(&[1.0, 2.0]); // size 2 instead of 3
        let err = program.validate_inputs(&[input]).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("expected static size 3, got 2"),
            "got: {msg}"
        );
    }

    #[test]
    fn test_validate_inputs_dynamic_in_range() {
        let program = make_dynamic_program();
        // batch=4 is within [1, 16]
        let a = tensor_2d(&[1.0; 12], 4, 3);
        let b = tensor_2d(&[2.0; 12], 4, 3);
        assert!(program.validate_inputs(&[a, b]).is_ok());
    }

    #[test]
    fn test_validate_inputs_dynamic_different_batch_in_range() {
        let mut program = make_dynamic_program();
        // Remove the equal constraint so we can test pure range validation.
        program.constraints.clear();

        // batch=8 for both is within [1, 16]
        let a = tensor_2d(&[1.0; 24], 8, 3);
        let b = tensor_2d(&[2.0; 24], 8, 3);
        assert!(program.validate_inputs(&[a, b]).is_ok());
    }

    #[test]
    fn test_validate_inputs_dynamic_out_of_range() {
        let mut program = make_dynamic_program();
        program.constraints.clear();

        // batch=0 is below min=1
        let a = ferrotorch_core::from_vec(Vec::<f32>::new(), &[0, 3]).unwrap();
        let b = ferrotorch_core::from_vec(Vec::<f32>::new(), &[0, 3]).unwrap();
        let err = program.validate_inputs(&[a, b]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("outside dynamic range"), "got: {msg}");
    }

    #[test]
    fn test_validate_inputs_constraint_satisfied() {
        let program = make_dynamic_program();
        // Both batch=4, equal constraint satisfied.
        let a = tensor_2d(&[1.0; 12], 4, 3);
        let b = tensor_2d(&[2.0; 12], 4, 3);
        assert!(program.validate_inputs(&[a, b]).is_ok());
    }

    #[test]
    fn test_validate_inputs_constraint_violated() {
        let program = make_dynamic_program();
        // Different batch sizes violate the Equal constraint.
        let a = tensor_2d(&[1.0; 12], 4, 3);
        let b = tensor_2d(&[2.0; 9], 3, 3);
        let err = program.validate_inputs(&[a, b]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not satisfied"), "got: {msg}");
    }

    // -- ExportedProgram::run tests ------------------------------------------

    #[test]
    fn test_run_add_self() {
        let program = make_add_self_program();
        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let outputs = program.run(&[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_close(outputs[0].data().unwrap(), &[2.0, 4.0, 6.0], 1e-6);
    }

    #[test]
    fn test_run_with_dynamic_batch() {
        let program = make_dynamic_program();
        let a = tensor_2d(&[1.0; 12], 4, 3);
        let b = tensor_2d(&[2.0; 12], 4, 3);
        let outputs = program.run(&[a, b]).unwrap();
        assert_eq!(outputs.len(), 1);
        let data = outputs[0].data().unwrap();
        assert_eq!(data.len(), 12);
        for &v in data {
            assert_close(&[v], &[3.0], 1e-6);
        }
    }

    #[test]
    fn test_run_rejects_invalid_inputs() {
        let program = make_add_self_program();
        let err = program.run(&[]).unwrap_err();
        assert!(err.to_string().contains("expected 1 inputs"));
    }

    // -- add_constraint tests ------------------------------------------------

    #[test]
    fn test_add_constraint() {
        let mut program = make_add_self_program();
        assert!(program.constraints.is_empty());

        program.add_constraint(ShapeConstraint {
            dim_a: (0, 0),
            dim_b: (0, 0),
            relation: ConstraintRelation::Equal,
        });
        assert_eq!(program.constraints.len(), 1);
    }

    // -- ConstraintRelation tests --------------------------------------------

    #[test]
    fn test_constraint_less_than() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![2, 3]);
        let b = g.add_input(vec![4, 3]);
        let (_, outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![4, 3]]);
        g.set_outputs(vec![outs[0]]);

        let program = ExportedProgram {
            graph: g,
            input_specs: vec![
                InputSpec {
                    name: "a".into(),
                    shape: vec![
                        DimSpec::Dynamic {
                            name: "d".into(),
                            min: 1,
                            max: 100,
                        },
                        DimSpec::Static(3),
                    ],
                    dtype: DType::Float32,
                },
                InputSpec {
                    name: "b".into(),
                    shape: vec![
                        DimSpec::Dynamic {
                            name: "d".into(),
                            min: 1,
                            max: 100,
                        },
                        DimSpec::Static(3),
                    ],
                    dtype: DType::Float32,
                },
            ],
            output_specs: vec![OutputSpec {
                name: "out".into(),
                shape: vec![4, 3],
                dtype: DType::Float32,
            }],
            state_dict: HashMap::new(),
            constraints: vec![ShapeConstraint {
                dim_a: (0, 0),
                dim_b: (1, 0),
                relation: ConstraintRelation::LessThan,
            }],
            metadata: ExportMetadata::default(),
        };

        // 2 < 4 -> satisfied
        let a = tensor_2d(&[1.0; 6], 2, 3);
        let b = tensor_2d(&[1.0; 12], 4, 3);
        assert!(program.validate_inputs(&[a, b]).is_ok());

        // 4 < 4 -> not satisfied
        let a2 = tensor_2d(&[1.0; 12], 4, 3);
        let b2 = tensor_2d(&[1.0; 12], 4, 3);
        assert!(program.validate_inputs(&[a2, b2]).is_err());
    }

    #[test]
    fn test_constraint_greater_than() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![4, 3]);
        let b = g.add_input(vec![2, 3]);
        let (_, outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![4, 3]]);
        g.set_outputs(vec![outs[0]]);

        let program = ExportedProgram {
            graph: g,
            input_specs: vec![
                InputSpec {
                    name: "a".into(),
                    shape: vec![
                        DimSpec::Dynamic {
                            name: "d".into(),
                            min: 1,
                            max: 100,
                        },
                        DimSpec::Static(3),
                    ],
                    dtype: DType::Float32,
                },
                InputSpec {
                    name: "b".into(),
                    shape: vec![
                        DimSpec::Dynamic {
                            name: "d".into(),
                            min: 1,
                            max: 100,
                        },
                        DimSpec::Static(3),
                    ],
                    dtype: DType::Float32,
                },
            ],
            output_specs: vec![OutputSpec {
                name: "out".into(),
                shape: vec![4, 3],
                dtype: DType::Float32,
            }],
            state_dict: HashMap::new(),
            constraints: vec![ShapeConstraint {
                dim_a: (0, 0),
                dim_b: (1, 0),
                relation: ConstraintRelation::GreaterThan,
            }],
            metadata: ExportMetadata::default(),
        };

        // 4 > 2 -> satisfied
        let a = tensor_2d(&[1.0; 12], 4, 3);
        let b = tensor_2d(&[1.0; 6], 2, 3);
        assert!(program.validate_inputs(&[a, b]).is_ok());

        // 2 > 4 -> not satisfied
        let a2 = tensor_2d(&[1.0; 6], 2, 3);
        let b2 = tensor_2d(&[1.0; 12], 4, 3);
        assert!(program.validate_inputs(&[a2, b2]).is_err());
    }

    // -- Serialization round-trip tests --------------------------------------

    #[test]
    fn test_save_load_round_trip() {
        let program = make_add_self_program();

        let dir = std::env::temp_dir().join("ferrotorch_test_export_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.ftep");

        program.save(&path).unwrap();
        assert!(path.exists());

        let loaded = ExportedProgram::load(&path).unwrap();

        // Verify structure.
        assert_eq!(loaded.graph.node_count(), program.graph.node_count());
        assert_eq!(loaded.graph.value_count(), program.graph.value_count());
        assert_eq!(loaded.input_specs.len(), program.input_specs.len());
        assert_eq!(loaded.output_specs.len(), program.output_specs.len());
        assert_eq!(loaded.constraints.len(), program.constraints.len());
        assert_eq!(
            loaded.metadata.format_version,
            program.metadata.format_version
        );
        assert_eq!(loaded.metadata.producer, program.metadata.producer);

        // Verify the loaded program produces the same result.
        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let outputs = loaded.run(&[input]).unwrap();
        assert_close(outputs[0].data().unwrap(), &[2.0, 4.0, 6.0], 1e-6);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_save_load_with_dynamic_shapes_and_constraints() {
        let program = make_dynamic_program();

        let dir = std::env::temp_dir().join("ferrotorch_test_export_dyn");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dynamic.ftep");

        program.save(&path).unwrap();
        let loaded = ExportedProgram::load(&path).unwrap();

        // Verify dynamic dim specs.
        assert_eq!(loaded.input_specs.len(), 2);
        match &loaded.input_specs[0].shape[0] {
            DimSpec::Dynamic { name, min, max } => {
                assert_eq!(name, "batch");
                assert_eq!(*min, 1);
                assert_eq!(*max, 16);
            }
            other => panic!("expected Dynamic, got {other:?}"),
        }
        match &loaded.input_specs[0].shape[1] {
            DimSpec::Static(3) => {}
            other => panic!("expected Static(3), got {other:?}"),
        }

        // Verify constraints.
        assert_eq!(loaded.constraints.len(), 1);
        assert_eq!(loaded.constraints[0].dim_a, (0, 0));
        assert_eq!(loaded.constraints[0].dim_b, (1, 0));
        assert_eq!(
            loaded.constraints[0].relation,
            ConstraintRelation::Equal
        );

        // Execute with valid inputs.
        let a = tensor_2d(&[1.0; 12], 4, 3);
        let b = tensor_2d(&[2.0; 12], 4, 3);
        let outputs = loaded.run(&[a, b]).unwrap();
        assert_eq!(outputs.len(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_save_load_with_state_dict() {
        let mut program = make_add_self_program();
        program
            .state_dict
            .insert("layer.weight".to_string(), vec![1.0, 2.0, 3.0]);
        program
            .state_dict
            .insert("layer.bias".to_string(), vec![0.1, 0.2]);

        let dir = std::env::temp_dir().join("ferrotorch_test_export_sd");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("with_state.ftep");

        program.save(&path).unwrap();
        let loaded = ExportedProgram::load(&path).unwrap();

        assert_eq!(loaded.state_dict.len(), 2);
        assert_close(
            &loaded.state_dict["layer.weight"],
            &[1.0, 2.0, 3.0],
            1e-7,
        );
        assert_close(&loaded.state_dict["layer.bias"], &[0.1, 0.2], 1e-7);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_invalid_magic() {
        let dir = std::env::temp_dir().join("ferrotorch_test_export_bad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.ftep");
        std::fs::write(&path, b"NOPE_garbage").unwrap();

        let err = ExportedProgram::load(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid magic"), "got: {msg}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_nonexistent_file() {
        let path = std::path::Path::new("/tmp/ferrotorch_does_not_exist.ftep");
        let err = ExportedProgram::load(path).unwrap_err();
        assert!(err.to_string().contains("failed to read"));
    }

    #[test]
    fn test_from_bytes_truncated() {
        let program = make_add_self_program();
        let dir = std::env::temp_dir().join("ferrotorch_test_export_trunc");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trunc.ftep");
        program.save(&path).unwrap();

        let full = std::fs::read(&path).unwrap();
        // Truncate to just the magic + version.
        let truncated = &full[..8];
        let err = ExportedProgram::from_bytes(truncated).unwrap_err();
        assert!(err.to_string().contains("EOF") || err.to_string().contains("unexpected"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- export_function tests -----------------------------------------------

    #[test]
    fn test_export_function_add_sum() {
        let a = grad_vec(vec![1.0, 2.0, 3.0]);
        let b = grad_vec(vec![4.0, 5.0, 6.0]);

        let program = export_function(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let s = add(&inputs[0], &inputs[1])?;
                sum(&s)
            },
            &[a, b],
            None,
        )
        .unwrap();

        assert_eq!(program.input_specs.len(), 2);
        assert_eq!(program.output_specs.len(), 1);

        // All dims should be static.
        for spec in &program.input_specs {
            for dim in &spec.shape {
                assert!(matches!(dim, DimSpec::Static(_)));
            }
        }
    }

    #[test]
    fn test_export_function_with_dynamic_shapes() {
        let a = grad_vec(vec![1.0, 2.0, 3.0]);
        let b = grad_vec(vec![4.0, 5.0, 6.0]);

        let mut dyn_shapes = DynamicShapeSpec::new();
        dyn_shapes.set_dynamic(0, 0, "len");
        dyn_shapes.set_dynamic(1, 0, "len");

        let program = export_function(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let s = add(&inputs[0], &inputs[1])?;
                sum(&s)
            },
            &[a, b],
            Some(&dyn_shapes),
        )
        .unwrap();

        // Dim 0 of both inputs should be Dynamic.
        match &program.input_specs[0].shape[0] {
            DimSpec::Dynamic { name, min, max } => {
                assert_eq!(name, "len");
                assert_eq!(*min, 1);
                assert_eq!(*max, 6); // 3 * 2
            }
            other => panic!("expected Dynamic, got {other:?}"),
        }
        match &program.input_specs[1].shape[0] {
            DimSpec::Dynamic { name, min, max } => {
                assert_eq!(name, "len");
                assert_eq!(*min, 1);
                assert_eq!(*max, 6);
            }
            other => panic!("expected Dynamic, got {other:?}"),
        }
    }

    #[test]
    fn test_export_function_run_produces_correct_result() {
        let a = grad_vec(vec![1.0, 2.0, 3.0]);
        let b = grad_vec(vec![4.0, 5.0, 6.0]);

        let program = export_function(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let product = mul(&inputs[0], &inputs[1])?;
                sum(&product)
            },
            &[a, b],
            None,
        )
        .unwrap();

        let a_input = tensor_1d(&[1.0, 2.0, 3.0]);
        let b_input = tensor_1d(&[4.0, 5.0, 6.0]);
        let outputs = program.run(&[a_input, b_input]).unwrap();
        // sum([4, 10, 18]) = 32
        assert_close(outputs[0].data().unwrap(), &[32.0], 1e-5);
    }

    // -- validate_exportable tests -------------------------------------------

    #[test]
    fn test_validate_exportable_rejects_fused() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, outs) = g.add_node(
            IrOpKind::FusedElementwise {
                ops: vec![IrOpKind::Add, IrOpKind::Relu],
            },
            vec![x],
            vec![vec![3]],
        );
        g.set_outputs(vec![outs[0]]);

        let err = validate_exportable(&g).unwrap_err();
        assert!(err.to_string().contains("FusedElementwise"));
    }

    #[test]
    fn test_validate_exportable_accepts_normal_ops() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![3]]);
        g.set_outputs(vec![relu_outs[0]]);

        assert!(validate_exportable(&g).is_ok());
    }

    // -- ExportMetadata tests ------------------------------------------------

    #[test]
    fn test_metadata_default() {
        let meta = ExportMetadata::default();
        assert_eq!(meta.format_version, 1);
        assert_eq!(meta.producer, "ferrotorch");
        assert!(meta.description.is_empty());
    }

    // -- JSON helper tests ---------------------------------------------------

    #[test]
    fn test_json_escape_basic() {
        assert_eq!(json_escape("hello"), "\"hello\"");
        assert_eq!(json_escape(""), "\"\"");
        assert_eq!(json_escape("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_escape("a\\b"), "\"a\\\\b\"");
        assert_eq!(json_escape("a\nb"), "\"a\\nb\"");
    }

    #[test]
    fn test_metadata_json_round_trip() {
        let meta = ExportMetadata {
            format_version: 1,
            description: "test model with \"quotes\"".to_string(),
            producer: "ferrotorch".to_string(),
        };

        let json = format!(
            "{{\"format_version\":{},\"description\":{},\"producer\":{}}}",
            meta.format_version,
            json_escape(&meta.description),
            json_escape(&meta.producer),
        );

        let parsed = parse_metadata_json(&json).unwrap();
        assert_eq!(parsed.format_version, 1);
        assert_eq!(parsed.description, "test model with \"quotes\"");
        assert_eq!(parsed.producer, "ferrotorch");
    }

    // -- End-to-end: save, load, run -----------------------------------------

    #[test]
    fn test_end_to_end_export_save_load_run() {
        let a = grad_vec(vec![1.0, 2.0, 3.0]);
        let b = grad_vec(vec![4.0, 5.0, 6.0]);

        let program = export_function(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let s = add(&inputs[0], &inputs[1])?;
                sum(&s)
            },
            &[a, b],
            None,
        )
        .unwrap();

        let dir = std::env::temp_dir().join("ferrotorch_test_export_e2e");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("e2e.ftep");

        program.save(&path).unwrap();
        let loaded = ExportedProgram::load(&path).unwrap();

        let a_input = tensor_1d(&[1.0, 2.0, 3.0]);
        let b_input = tensor_1d(&[4.0, 5.0, 6.0]);
        let outputs = loaded.run(&[a_input, b_input]).unwrap();
        // sum([5, 7, 9]) = 21
        assert_close(outputs[0].data().unwrap(), &[21.0], 1e-5);

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- Accessors -----------------------------------------------------------

    #[test]
    fn test_accessors() {
        let program = make_add_self_program();
        assert_eq!(program.graph().node_count(), 2);
        assert_eq!(program.input_specs().len(), 1);
        assert_eq!(program.output_specs().len(), 1);
        assert!(program.state_dict().is_empty());
        assert!(program.constraints().is_empty());
        assert_eq!(program.metadata().producer, "ferrotorch");
    }

    // -- Error display tests -------------------------------------------------

    #[test]
    fn test_export_error_display() {
        let e = ExportError::Export {
            message: "test".into(),
        };
        assert_eq!(e.to_string(), "export: test");

        let e = ExportError::InputValidation {
            message: "bad".into(),
        };
        assert!(e.to_string().contains("input validation"));

        let e = ExportError::ConstraintViolation {
            message: "fail".into(),
        };
        assert!(e.to_string().contains("constraint violated"));

        let e = ExportError::Serialization {
            message: "io".into(),
        };
        assert!(e.to_string().contains("serialization"));

        let e = ExportError::Deserialization {
            message: "bad".into(),
        };
        assert!(e.to_string().contains("deserialization"));

        let e = ExportError::UnsupportedOp {
            op: "Fused".into(),
        };
        assert!(e.to_string().contains("Fused"));
    }
}
