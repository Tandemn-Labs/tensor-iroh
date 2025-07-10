use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use iroh::{Endpoint, NodeAddr};
use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::endpoint::{BindError, ConnectError, ConnectionError, WriteError};
use iroh_base::ticket::{NodeTicket, ParseError};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc};
use tracing::{debug, error, info, warn};
use iroh::Watcher;
uniffi::setup_scaffolding!();

// ALPN for our tensor protocol
const TENSOR_ALPN: &[u8] = b"tensor-iroh/direct/0";

// Error types
#[derive(Debug, Error, uniffi::Error)]
#[uniffi(flat_error)]
pub enum TensorError {
    #[error("IO error: {message}")]
    Io { message: String },
    #[error("Serialization error: {message}")]
    Serialization { message: String },
    #[error("Connection error: {message}")]
    Connection { message: String },
    #[error("Protocol error: {message}")]
    Protocol { message: String },
}

impl From<anyhow::Error> for TensorError {
    fn from(err: anyhow::Error) -> Self {
        TensorError::Protocol {
            message: err.to_string(),
        }
    }
}

impl From<std::io::Error> for TensorError {
    fn from(err: std::io::Error) -> Self {
        TensorError::Io {
            message: err.to_string(),
        }
    }
}

impl From<postcard::Error> for TensorError {
    fn from(err: postcard::Error) -> Self {
        TensorError::Serialization {
            message: err.to_string(),
        }
    }
}

impl From<quinn::WriteError> for TensorError {
    fn from(err: quinn::WriteError) -> Self {
        TensorError::Io {
            message: err.to_string(),
        }
    }
}

impl From<WriteError> for TensorError {
    fn from(err: WriteError) -> Self {
        TensorError::Io {
            message: err.to_string(),
        }
    }
}

impl From<ConnectionError> for TensorError {
    fn from(err: ConnectionError) -> Self {
        TensorError::Connection {
            message: err.to_string(),
        }
    }
}

impl From<ParseError> for TensorError {
    fn from(err: ParseError) -> Self {
        TensorError::Connection {
            message: err.to_string(),
        }
    }
}

// Data structures
#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct TensorMetadata {
    pub shape: Vec<i64>,
    pub dtype: String,
    pub requires_grad: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct TensorData {
    pub metadata: TensorMetadata,
    pub data: Vec<u8>,
}

// Protocol message types
#[derive(Debug, Serialize, Deserialize)]
enum TensorMessage {
    Request { tensor_name: String },
    Response { tensor_name: String, data: TensorData },
    Error { message: String },
}

// Protocol handler for incoming tensor requests
#[derive(Debug, Clone)]
struct TensorProtocolHandler {
    tensor_store: Arc<Mutex<HashMap<String, TensorData>>>,
    receiver_tx: Arc<Mutex<Option<mpsc::UnboundedSender<(String, String, TensorData)>>>>,
}

impl TensorProtocolHandler {
    fn new() -> Self {
        Self {
            tensor_store: Arc::new(Mutex::new(HashMap::new())),
            receiver_tx: Arc::new(Mutex::new(None)),
        }
    }

    fn set_receiver(&self, tx: mpsc::UnboundedSender<(String, String, TensorData)>) {
        *self.receiver_tx.lock().unwrap() = Some(tx);
    }

    fn register_tensor(&self, name: String, tensor: TensorData) {
        self.tensor_store.lock().unwrap().insert(name, tensor);
    }
}

impl ProtocolHandler for TensorProtocolHandler {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        let peer_id = connection.remote_node_id()?.to_string();
        debug!("Accepted tensor connection from {}", peer_id);

        let (mut send, mut recv) = connection.accept_bi().await?;

        // Read the request message
        // let request_bytes = recv.read_to_end(1024).await?;
        let request_bytes = recv.read_to_end(1024).await.map_err(AcceptError::from_err)?;
        let message: TensorMessage = postcard::from_bytes(&request_bytes)
            .map_err(|e| AcceptError::from_err(e))?;

        match message {
            TensorMessage::Request { tensor_name } => {
                debug!("Received request for tensor: {}", tensor_name);

                let response = {
                    let store = self.tensor_store.lock().unwrap();
                    match store.get(&tensor_name) {
                        Some(tensor_data) => TensorMessage::Response {
                            tensor_name: tensor_name.clone(),
                            data: tensor_data.clone(),
                        },
                        None => TensorMessage::Error {
                            message: format!("Tensor '{}' not found", tensor_name),
                        },
                    }
                };

                // Send response
                let response_bytes = postcard::to_allocvec(&response)
                    .map_err(|e| AcceptError::from_err(e))?;
                // send.write_all(&response_bytes).await?;
                send.write_all(&response_bytes).await.map_err(AcceptError::from_err)?;
                send.finish().map_err(AcceptError::from_err)?;
            }
            TensorMessage::Response { tensor_name, data } => {
                // This is an incoming tensor from a peer
                debug!("Received tensor data: {}", tensor_name);
                
                if let Some(tx) = self.receiver_tx.lock().unwrap().as_ref() {
                    let _ = tx.send((peer_id, tensor_name, data));
                }
            }
            TensorMessage::Error { message } => {
                warn!("Received error from peer: {}", message);
            }
        }

        connection.closed().await;
        Ok(())
    }
}

// Main TensorNode implementation
#[derive(uniffi::Object)]
pub struct TensorNode {
    endpoint: Arc<Mutex<Option<Endpoint>>>,
    router: Arc<Mutex<Option<Router>>>,
    handler: Arc<TensorProtocolHandler>,
    receiver_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<(String, String, TensorData)>>>>,
}

#[uniffi::export]
impl TensorNode {
    #[uniffi::constructor]
    pub fn new(_storage_path: Option<String>) -> Self {
        let handler = Arc::new(TensorProtocolHandler::new());
        let (tx, rx) = mpsc::unbounded_channel();
        handler.set_receiver(tx);

        Self {
            endpoint: Arc::new(Mutex::new(None)),
            router: Arc::new(Mutex::new(None)),
            handler,
            receiver_rx: Arc::new(Mutex::new(Some(rx))),
        }
    }

    #[uniffi::method(async_runtime = "tokio")]
    pub async fn start(&self) -> Result<(), TensorError> {
        info!("Starting tensor node...");

        // Create endpoint
        // this is the official doc example, so use it 
        let endpoint = Endpoint::builder()
            .discovery_n0()
            .bind()
            .await
            .map_err(|e: BindError| {
                TensorError::Connection { message: e.to_string() }
            })?;

        

        // Create router with our protocol handler
        let router = Router::builder(endpoint.clone())
            .accept(TENSOR_ALPN, self.handler.clone())
            .spawn();

        // Store references using interior mutability
        *self.endpoint.lock().unwrap() = Some(endpoint);
        *self.router.lock().unwrap() = Some(router);

        info!("Tensor node started successfully");
        Ok(())
    }

    #[uniffi::method(async_runtime = "tokio")]
    pub async fn send_tensor_direct(
        &self,
        peer_addr: String,
        tensor_name: String,
        tensor: TensorData,
    ) -> Result<(), TensorError> {
        let endpoint = {
            let endpoint_guard = self.endpoint.lock().unwrap();
            endpoint_guard.as_ref()
                .ok_or_else(|| TensorError::Protocol { message: "Node not started".to_string() })?
                .clone()
        };

        debug!("Sending tensor '{}' to {}", tensor_name, peer_addr);

        // Parse peer address
        let ticket: NodeTicket = peer_addr.parse()?;
        let node_addr: NodeAddr = ticket.into();

        // Connect to peer
        let connection = endpoint.connect(node_addr, TENSOR_ALPN).await
            .map_err(|e: ConnectError| {TensorError::Connection { message: e.to_string() }})?;


        // Open stream and send tensor
        let (mut send, mut _recv) = connection.open_bi().await?;

        // Send the tensor data directly as a response message
        let message = TensorMessage::Response {
            tensor_name: tensor_name.clone(),
            data: tensor,
        };

        let message_bytes = postcard::to_allocvec(&message)?;
        send.write_all(&message_bytes).await?;
        send.finish().map_err(|e| TensorError::Connection { message: e.to_string() })?;

        debug!("Tensor '{}' sent successfully", tensor_name);
        Ok(())
    }

    #[uniffi::method(async_runtime = "tokio")]
    pub async fn receive_tensor(&self) -> Result<Option<TensorData>, TensorError> {
        let mut receiver_guard = self.receiver_rx.lock().unwrap();
        if let Some(rx) = receiver_guard.as_mut() {
            match rx.try_recv() {
                Ok((peer_id, tensor_name, tensor_data)) => {
                    debug!("Received tensor '{}' from {}", tensor_name, peer_id);
                    Ok(Some(tensor_data))
                }
                Err(mpsc::error::TryRecvError::Empty) => Ok(None),
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    Err(TensorError::Protocol { message: "Receiver disconnected".to_string() })
                }
            }
        } else {
            Err(TensorError::Protocol { message: "Node not started".to_string() })
        }
    }

    #[uniffi::method(async_runtime = "tokio")]
    pub async fn get_node_addr(&self) -> Result<String, TensorError> {
        let endpoint = {
            let endpoint_guard = self.endpoint.lock().unwrap();
            endpoint_guard.as_ref()
                .ok_or_else(|| TensorError::Protocol {
                    message: "Node not started".into(),
                })?
                .clone()
        };
        // the following is the official doc example, so use it 
        // we need to get the node addr from the endpoint
        // this is not a future?
        let result = endpoint.node_addr().initialized().await
            .map_err(|_err| TensorError::Protocol { message: "Discovery watcher error".into() })?;
        
        let addr = result.relay_url.ok_or_else(|| TensorError::Protocol { message: "Address not available".into() })?;

        Ok(format!("{:?}", addr))
    }

    #[uniffi::method]
    pub fn register_tensor(&self, name: String, tensor: TensorData) -> Result<(), TensorError> {
        debug!("Registering tensor: {}", name);
        self.handler.register_tensor(name, tensor);
        Ok(())
    }

    #[uniffi::method]
    pub fn shutdown(&self) -> Result<(), TensorError> {
        info!("Shutting down tensor node...");
        // Router will be dropped automatically, triggering shutdown
        Ok(())
    }
}

// Free function for creating nodes
#[uniffi::export]
pub fn create_node(_storage_path: Option<String>) -> TensorNode {
    TensorNode::new(_storage_path)
}

// Callback trait for receiving tensors
#[uniffi::export(with_foreign)]
pub trait TensorReceiveCallback: Send + Sync + 'static {
    fn on_tensor_received(&self, peer_id: String, tensor_name: String, tensor: TensorData);
} 
