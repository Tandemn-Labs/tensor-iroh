/*
 * TENSOR PROTOCOL - Direct peer-to-peer tensor streaming using Iroh
 * 
 * This library implements a custom protocol for sending machine learning tensors
 * (like PyTorch tensors) directly between computers over the internet using Iroh's
 * networking stack.
 * 
 * Think of this like a specialized file sharing system, but instead of sharing
 * regular files, we're sharing AI model data (tensors) in real-time.
 */

// ============================================================================
// IMPORTS - All the external code we need to use
// ============================================================================

use std::{
    collections::HashMap,           // For storing tensors by name (like a dictionary)
    sync::{Arc, Mutex},            // For thread-safe sharing of data
};

use anyhow::Result;                // A flexible error handling type
use iroh::{Endpoint, NodeAddr};    // Iroh's networking types - Endpoint is like a "network connection manager"
use iroh::endpoint::Connection;    // Represents a connection to another peer
use iroh::protocol::{AcceptError, ProtocolHandler, Router}; // For handling incoming connections
use iroh::endpoint::{BindError, ConnectError, ConnectionError, WriteError}; // Different error types
use iroh_base::ticket::{NodeTicket, ParseError}; // For parsing peer addresses from strings
use serde::{Deserialize, Serialize}; // The "architect" for serialization (converting data to bytes)
use thiserror::Error;             // Makes creating custom error types easier
use tokio::sync::{mpsc};          // For async message passing between parts of our program
use tracing::{debug, error, info, warn}; // For logging what's happening (like print statements but better)
use iroh::Watcher;                // For watching network changes

// This tells UniFFI (the cross-language binding generator) to set up the scaffolding
// It's like telling the "translator" to get ready to translate our Rust code
uniffi::setup_scaffolding!();

// ============================================================================
// CONSTANTS - Fixed values used throughout the program
// ============================================================================

// This is like a "protocol ID" - it tells other computers what kind of data we're sending
// Think of it like saying "I speak Tensor Protocol version 0" when connecting
const TENSOR_ALPN: &[u8] = b"tensor-iroh/direct/0";

// ============================================================================
// ERROR TYPES - All the different ways things can go wrong
// ============================================================================

// This defines all the possible error types our program might encounter
// It's like creating a list of all the different ways a delivery could fail
#[derive(Debug, Error, uniffi::Error)] // Debug allows us to print the error, Error makes a 
// custom error type, uniffi::Error makes it compatible with UniFFI and allows export to other languages
#[uniffi(flat_error)]  // Tells UniFFI how to expose this to other languages (flat is to avoid nested structs)
pub enum TensorError {
    #[error("IO error: {message}")]           // Network or file problems
    Io { message: String },
    #[error("Serialization error: {message}")]  // Problems converting data to/from bytes
    Serialization { message: String },
    #[error("Connection error: {message}")]   // Problems connecting to other peers
    Connection { message: String },
    #[error("Protocol error: {message}")]     // Problems with our custom protocol
    Protocol { message: String },
}

// ============================================================================
// ERROR CONVERSION IMPLEMENTATIONS - The "Universal Adapters"
// ============================================================================
// These are like universal adapters that convert specific errors into our TensorError type
// Remember: Our functions can only return TensorError, but libraries return their own error types
// These implementations let us use the ? operator to automatically convert errors

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

// ============================================================================
// DATA STRUCTURES - The custom data types we use
// ============================================================================

// This describes the "shape" and properties of a tensor (like an AI model's data)
// Think of it like a label on a box that tells you what's inside
// these are a part of the Public Facing API. It allows us to define the shape and properties of a tensor
// and then use it to send and receive tensors.
#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)] // Debug allows us to print the struct,
// Clone allows us to copy it, Serialize and Deserialize allow us to convert it to and from bytes,
// uniffi::Record allows us to export it to other languages
pub struct TensorMetadata {
    pub shape: Vec<i64>,        // The dimensions (e.g., [3, 224, 224] for a 3-channel 224x224 image)
    pub dtype: String,          // The data type (e.g., "float32", "int64")
    pub requires_grad: bool,    // Whether this tensor needs gradients for training
}

// This is the actual tensor data - both the metadata (description) and the raw bytes
// Think of it like a package: the metadata is the shipping label, data is what's inside
#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct TensorData {
    pub metadata: TensorMetadata,  // Information about the tensor
    pub data: Vec<u8>,            // The actual tensor data as raw bytes
}

// ============================================================================
// PROTOCOL MESSAGE TYPES - The different types of messages peers can send
// ============================================================================

// These are the different types of messages our protocol can send
// It's like defining the different types of letters you can send in the mail
// This is an internal type that is not exposed to the public facing API.
#[derive(Debug, Serialize, Deserialize)] // Debug allows us to print the struct,
// Serialize and Deserialize allow us to convert it to and from bytes
// Clone is missing here because it will be BIG to copy a BIG tensor(expensive af)
enum TensorMessage {
    // A request asking for a specific tensor by name
    Request { tensor_name: String },
    // A response containing the requested tensor data
    Response { tensor_name: String, data: TensorData },
    // An error message when something goes wrong
    Error { message: String },
}

// ============================================================================
// PROTOCOL HANDLER - Handles incoming connections and messages
// ============================================================================

// This is the "receptionist" that handles incoming connections from other peers
// It knows how to respond to requests and where to store/retrieve tensors
#[derive(Debug, Clone)]
struct TensorProtocolHandler {
    // A thread-safe storage for tensors (like a filing cabinet)
    tensor_store: Arc<Mutex<HashMap<String, TensorData>>>, // this is literally where the data lives
    // it is a thread safe hashmap that is used to store the tensors. 
    // hashmap because the speed of retrieval is O(1)
    
    // A way to send received tensors to the main application
    // Think of it like a conveyor belt that delivers incoming packages
    receiver_tx: Arc<Mutex<Option<mpsc::UnboundedSender<(String, String, TensorData)>>>>,
}

impl TensorProtocolHandler {
    // Creates a new protocol handler (like hiring a new receptionist)
    fn new() -> Self {
        Self {
            tensor_store: Arc::new(Mutex::new(HashMap::new())),
            receiver_tx: Arc::new(Mutex::new(None)),
        }
    }

    // Sets up the channel for delivering received tensors to the main application
    fn set_receiver(&self, tx: mpsc::UnboundedSender<(String, String, TensorData)>) {
        *self.receiver_tx.lock().unwrap() = Some(tx);
    }

    // Stores a tensor in our local storage (like filing a document)
    fn register_tensor(&self, name: String, tensor: TensorData) {
        self.tensor_store.lock().unwrap().insert(name, tensor);
    }
}

// This implements the actual protocol logic - what happens when someone connects to us
impl ProtocolHandler for TensorProtocolHandler {
    // This function is called whenever another peer connects to us
    // It's like answering the door when someone knocks
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        // Get the ID of who's connecting to us
        let peer_id = connection.remote_node_id()?.to_string();
        debug!("Accepted tensor connection from {}", peer_id);

        // Set up a two-way communication channel
        // send = for sending data to the peer, recv = for receiving data from the peer
        let (mut send, mut recv) = connection.accept_bi().await?;

        // Read the incoming message (up to 1024 bytes)
        // this part is GOOD for control plane (not much bytes)
        let request_bytes = recv.read_to_end(1024).await.map_err(AcceptError::from_err)?;
        
        // Convert the raw bytes back into a TensorMessage using postcard
        // This is like opening an envelope and reading the letter inside
        // this part is GOOD for control plane (not much bytes)
        let message: TensorMessage = postcard::from_bytes(&request_bytes)
            .map_err(|e| AcceptError::from_err(e))?;

        // Handle different types of messages
        match message {
            // Someone is asking us for a tensor
            TensorMessage::Request { tensor_name } => {
                debug!("Received request for tensor: {}", tensor_name);

                // Look up the tensor in our storage
                let response = {
                    let store = self.tensor_store.lock().unwrap();
                    match store.get(&tensor_name) {
                        // We have the tensor - send it back
                        Some(tensor_data) => TensorMessage::Response {
                            tensor_name: tensor_name.clone(),
                            data: tensor_data.clone(),
                        },
                        // We don't have the tensor - send an error
                        None => TensorMessage::Error {
                            message: format!("Tensor '{}' not found", tensor_name),
                        },
                    }
                };

                // Convert our response back to bytes using postcard
                let response_bytes = postcard::to_allocvec(&response)
                    .map_err(|e| AcceptError::from_err(e))?;
                
                // Send the response back to the peer
                send.write_all(&response_bytes).await.map_err(AcceptError::from_err)?;
                send.finish().map_err(AcceptError::from_err)?;
            }
            
            // Someone is sending us a tensor (unsolicited)
            TensorMessage::Response { tensor_name, data } => {
                debug!("Received tensor data: {}", tensor_name);
                
                // If we have a receiver channel set up, forward the tensor to the main application
                if let Some(tx) = self.receiver_tx.lock().unwrap().as_ref() {
                    let _ = tx.send((peer_id, tensor_name, data));
                }
            }
            
            // Someone sent us an error message
            TensorMessage::Error { message } => {
                warn!("Received error from peer: {}", message);
            }
        }

        // Wait for the connection to close gracefully
        connection.closed().await;
        Ok(())
    }
}

// ============================================================================
// MAIN TENSORNODE IMPLEMENTATION - The primary interface for our protocol
// ============================================================================

// This is the main class that users will interact with
// Think of it as the "control panel" for our tensor networking system
#[derive(uniffi::Object)]
pub struct TensorNode {
    // The network endpoint (like our "network interface card")
    endpoint: Arc<Mutex<Option<Endpoint>>>,
    
    // The router that handles incoming connections (like a switchboard operator)
    router: Arc<Mutex<Option<Router>>>,
    
    // Our protocol handler (the "receptionist")
    handler: Arc<TensorProtocolHandler>,
    
    // Channel for receiving tensors from other peers
    receiver_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<(String, String, TensorData)>>>>,
}

#[uniffi::export]  // Makes these methods available to other programming languages
impl TensorNode {
    // Constructor - creates a new TensorNode (like building a new office)
    #[uniffi::constructor]
    pub fn new(_storage_path: Option<String>) -> Self {
        // Create the protocol handler
        let handler = Arc::new(TensorProtocolHandler::new());
        
        // Create a channel for receiving tensors
        // tx = transmitter (sender), rx = receiver
        let (tx, rx) = mpsc::unbounded_channel();
        handler.set_receiver(tx);

        Self {
            // We use Arc<Mutex<Option<>>> for thread-safe lazy initialization
            // This means "maybe we have an endpoint, maybe we don't, but if we do, it's thread-safe"
            endpoint: Arc::new(Mutex::new(None)),
            router: Arc::new(Mutex::new(None)),
            handler,
            receiver_rx: Arc::new(Mutex::new(Some(rx))),
        }
    }

    // Starts the tensor node (like "opening for business")
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn start(&self) -> Result<(), TensorError> {
        info!("Starting tensor node...");

        // Create the network endpoint using Iroh's builder pattern
        // This is like setting up our network connection and getting a phone number
        let endpoint = Endpoint::builder()
            .discovery_n0()  // Use Iroh's discovery system to find other peers
            .bind()          // Actually create and bind the endpoint
            .await
            .map_err(|e: BindError| {
                TensorError::Connection { message: e.to_string() }
            })?;

        // Create a router that will handle incoming connections
        // This is like hiring a receptionist to answer calls
        let router = Router::builder(endpoint.clone())
            .accept(TENSOR_ALPN, self.handler.clone())  // Accept connections for our protocol
            .spawn();  // Start the router in the background

        // Store the endpoint and router using interior mutability
        // This is thread-safe because of the Arc<Mutex<>>
        *self.endpoint.lock().unwrap() = Some(endpoint);
        *self.router.lock().unwrap() = Some(router);

        info!("Tensor node started successfully");
        Ok(())
    }

    // Sends a tensor directly to another peer (like mailing a package)
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn send_tensor_direct(
        &self,
        peer_addr: String,      // The address of the peer (like a mailing address)
        tensor_name: String,    // A name for the tensor
        tensor: TensorData,     // The actual tensor data
    ) -> Result<(), TensorError> {
        // Get our endpoint (make sure we're started)
        let endpoint = {
            let endpoint_guard = self.endpoint.lock().unwrap();
            endpoint_guard.as_ref()
                .ok_or_else(|| TensorError::Protocol { message: "Node not started".to_string() })?
                .clone()
        };

        debug!("Sending tensor '{}' to {}", tensor_name, peer_addr);

        // Parse the peer address from a string into a NodeAddr
        // This is like converting a written address into GPS coordinates
        let ticket: NodeTicket = peer_addr.parse()?;
        let node_addr: NodeAddr = ticket.into();

        // Connect to the peer
        // This is like making a phone call to the other peer
        let connection = endpoint.connect(node_addr, TENSOR_ALPN).await
            .map_err(|e: ConnectError| {TensorError::Connection { message: e.to_string() }})?;

        // Open a bidirectional stream (like opening a two-way conversation)
        let (mut send, mut _recv) = connection.open_bi().await?;

        // Create a message containing our tensor
        let message = TensorMessage::Response {
            tensor_name: tensor_name.clone(),
            data: tensor,
        };

        // Convert the message to bytes using postcard (pack it in an envelope)
        let message_bytes = postcard::to_allocvec(&message)?;
        
        // Send the bytes over the network
        send.write_all(&message_bytes).await?;
        
        // Signal that we're done sending (like sealing the envelope)
        send.finish().map_err(|e| TensorError::Connection { message: e.to_string() })?;

        debug!("Tensor '{}' sent successfully", tensor_name);
        Ok(())
    }

    // Checks if we've received any tensors from other peers (like checking the mailbox)
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn receive_tensor(&self) -> Result<Option<TensorData>, TensorError> {
        let mut receiver_guard = self.receiver_rx.lock().unwrap();
        if let Some(rx) = receiver_guard.as_mut() {
            match rx.try_recv() {
                // We got a tensor!
                Ok((peer_id, tensor_name, tensor_data)) => {
                    debug!("Received tensor '{}' from {}", tensor_name, peer_id);
                    Ok(Some(tensor_data))
                }
                // No tensors waiting
                Err(mpsc::error::TryRecvError::Empty) => Ok(None),
                // The channel is broken
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    Err(TensorError::Protocol { message: "Receiver disconnected".to_string() })
                }
            }
        } else {
            Err(TensorError::Protocol { message: "Node not started".to_string() })
        }
    }

    // Gets our node's address so others can connect to us (like getting our phone number)
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
        
        // Wait for our address to be initialized by the discovery system
        let result = endpoint.node_addr().initialized().await
            .map_err(|_err| TensorError::Protocol { message: "Discovery watcher error".into() })?;
        
        // Extract the relay URL (this is how other peers can reach us)
        let addr = result.relay_url.ok_or_else(|| TensorError::Protocol { message: "Address not available".into() })?;

        Ok(format!("{:?}", addr))
    }

    // Stores a tensor locally so others can request it (like putting something in storage)
    #[uniffi::method]
    pub fn register_tensor(&self, name: String, tensor: TensorData) -> Result<(), TensorError> {
        debug!("Registering tensor: {}", name);
        self.handler.register_tensor(name, tensor);
        Ok(())
    }

    // Shuts down the node (like closing the office)
    #[uniffi::method]
    pub fn shutdown(&self) -> Result<(), TensorError> {
        info!("Shutting down tensor node...");
        // When the router is dropped, it automatically shuts down
        Ok(())
    }
}

// ============================================================================
// CONVENIENCE FUNCTIONS AND TRAITS
// ============================================================================

// A convenience function for creating nodes from other languages
#[uniffi::export]
pub fn create_node(_storage_path: Option<String>) -> TensorNode {
    TensorNode::new(_storage_path)
}

// A trait for callbacks when tensors are received
// This allows other languages to register callback functions
#[uniffi::export(with_foreign)]
pub trait TensorReceiveCallback: Send + Sync + 'static {
    fn on_tensor_received(&self, peer_id: String, tensor_name: String, tensor: TensorData);
} 
