/*!
 * ──────────────────────────────────────────────────────────────
 *  PyO3 Python Bindings for Tensor Protocol
 * ──────────────────────────────────────────────────────────────
 *  Thin PyO3 façade around the UniFFI `TensorNode`.
 *  Compile with: `--features python` (or `python,torch` for Torch support)
 * ──────────────────────────────────────────────────────────────
 */
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use numpy::{ToPyArray, PyArray1};
use pyo3_asyncio::tokio::future_into_py;
use pyo3::PyObject;      

use crate::{TensorNode, TensorData, TensorMetadata, CHUNK_SIZE};

/// Helper: convert Python buffer (NumPy, bytes, etc.) to Vec<u8>
/// Accepts either a NumPy array of u8 or a Python bytes object.
/// Returns a Vec<u8> containing the raw data.
fn pybytes_to_vec(_py: Python<'_>, obj: &PyAny) -> PyResult<Vec<u8>> {
    // Try to downcast to a NumPy array of u8
    if let Ok(arr) = obj.downcast::<PyArray1<u8>>() {
        // SAFETY: We're immediately converting to owned Vec, so no lifetime issues
        unsafe { Ok(arr.as_slice()?.to_vec()) }
    // Try to downcast to Python bytes
    } else if let Ok(bytes) = obj.downcast::<PyBytes>() {
        Ok(bytes.as_bytes().to_vec())
    // Otherwise, error out
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected bytes-like or NumPy u8 array",
        ))
    }
}

// =========== Python-exposed wrappers ============

/// Python class representing tensor data (with metadata)
#[pyclass]
struct PyTensorData {
    inner: TensorData,
}

#[pymethods]
impl PyTensorData {
    /// Create a new PyTensorData from Python (data, shape, dtype, requires_grad)
    /// - `data`: bytes-like or NumPy array
    /// - `shape`: list of i64
    /// - `dtype`: string (e.g. "float32")
    /// - `requires_grad`: bool
    #[new]
    fn new(py: Python<'_>, data: &PyAny, shape: Vec<i64>, dtype: &str, requires_grad: bool)
        -> PyResult<Self>
    {
        // Convert the Python data to Vec<u8>
        let raw = pybytes_to_vec(py, data)?;
        Ok(Self {
            inner: TensorData {
                metadata: TensorMetadata {
                    shape,
                    dtype: dtype.to_string(),
                    requires_grad,
                },
                data: raw,
            },
        })
    }

    /// Return the tensor data as a NumPy array (read-only, zero-copy)
    fn as_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<u8>> {
        Ok(self.inner.data.as_slice().to_pyarray(py))
    }

    /// Return the tensor data as raw Python bytes
    fn as_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.inner.data)
    }

    /// Get the shape of the tensor (list of i64)
    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.inner.metadata.shape.clone()
    }

    /// Get the dtype of the tensor (string)
    #[getter]
    fn dtype(&self) -> &str {
        &self.inner.metadata.dtype
    }

    /// Get whether the tensor requires gradients (bool)
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.metadata.requires_grad
    }

    // #[cfg(feature = "torch")]
    // fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
    //     use tch::{Tensor, Kind, Device};
    //     // let len = self.inner.data.len();
    //     let len = self.inner.data.len() / std::mem::size_of::<f32>();
    //     // Create a 1D tensor of u8 from the raw data
    //     let t = Tensor::from_slice(&self.inner.data)
    //             .to_kind(Kind::Uint8)
    //             .to_device(Device::Cpu)
    //             .reshape(&[len as i64]);
    //     // Leak the tensor into Python (ownership transferred)
    //     Ok(t.into_py(py))
    // }

    #[cfg(feature = "torch")]
    fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        use numpy::{PyArray1, PyArrayDyn};

        // 1. Re-interpret internal bytes as f32 slice (no copy).
        let n_floats = self.inner.data.len() / std::mem::size_of::<f32>();
        let float_slice = unsafe {
            std::slice::from_raw_parts(
                self.inner.data.as_ptr() as *const f32,
                n_floats,
            )
        };

        // 2.  Build a NumPy array that BORROWS the same data
        let arr1d: &PyArray1<f32> = PyArray1::from_slice(py, float_slice);
        //    Convert Vec<i64> → slice of usize (ndarray shape expects usize)
        let dims: Vec<usize> =
            self.inner.metadata.shape.iter().map(|&d| d as usize).collect();
        //    Reshape to the original tensor shape.
        let arr: &PyArrayDyn<f32> = arr1d.reshape(dims)?;

        // 3. Call torch.from_numpy(arr) – zero-copy share.
        let torch = py.import("torch")?;
        let tensor = torch.getattr("from_numpy")?.call1((arr,))?;

        Ok(tensor.to_object(py))   // return owned torch.Tensor to Python
    }
}

/// Python class representing a tensor node (protocol endpoint)
#[pyclass]
struct PyTensorNode {
    inner: TensorNode,
}

#[pymethods]
impl PyTensorNode {
    /// Create a new PyTensorNode (optionally with config, here always None)
    #[new]
    fn new() -> Self {
        Self { inner: crate::create_node(None) }
    }

    /// Start the node (async, returns awaitable)
    fn start<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            inner.start().await.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    /// Get the node's address (async, returns awaitable)
    fn get_node_addr<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            inner.get_node_addr().await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    /// Register a tensor under a given name
    fn register_tensor(&self, name: &str, tensor: &PyTensorData) -> PyResult<()> {
        self.inner.register_tensor(name.to_string(), tensor.inner.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Send a tensor to a peer (async, returns awaitable)
    /// - `peer`: address string
    /// - `name`: tensor name
    /// - `tensor`: PyTensorData
    fn send_tensor<'py>(
        &self,
        py: Python<'py>,
        peer: &str,
        name: &str,
        tensor: &PyTensorData,
    ) -> PyResult<&'py PyAny> {
        let inner = self.inner.clone();
        let peer = peer.to_string();
        let name = name.to_string();
        let td   = tensor.inner.clone();
        future_into_py(py, async move {
            inner.send_tensor_direct(peer, name, td).await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    /// Receive a tensor (async, returns awaitable)
    /// Returns Some(PyTensorData) if a tensor is received, or None.
    fn receive_tensor<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            match inner.receive_tensor().await {
                Ok(Some(t)) => Ok(Some(PyTensorData { inner: t })),
                Ok(None)    => Ok(None),
                Err(e)      => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
            }
        })
    }

    /// Shut down the node (cleanup resources)
    fn shutdown(&self) -> PyResult<()> {
        self.inner.shutdown().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get the current connection pool size (async, returns awaitable)
    fn pool_size<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            inner.get_pool_size().await
                .map(|v| v as u64)   // Python int
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }
    // }   
}

// Optional Torch helper (needs `--features torch`)
// #[cfg(feature = "torch")]
// #[pymethods]
// impl PyTensorData {
    /// Convert to `torch.Tensor` via DLPack (zero-copy)
    /// Only available if compiled with the "torch" feature.

// ---------- module entry ----------
/// Python module definition for `tensor_protocol`
#[pymodule]
fn tensor_iroh(_py: Python, m: &PyModule) -> PyResult<()> {
    // Initialize the Tokio runtime for PyO3 async support (multi-threaded)
    // pyo3_asyncio::tokio::init_multi_thread();
    // Expose PyTensorNode and PyTensorData classes to Python
    m.add_class::<PyTensorNode>()?;
    m.add_class::<PyTensorData>()?;
    // Expose CHUNK_SIZE constant for convenience
    m.add("CHUNK_SIZE", CHUNK_SIZE)?;
    Ok(())
}
