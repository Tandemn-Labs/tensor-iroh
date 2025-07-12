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
const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024; // 100MB limit to prevent memory exhaustion


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
        let peer_id_result = connection.remote_node_id();
        println!("🚪 [ACCEPT] Incoming connection...");
        
        let peer_id = peer_id_result?.to_string();
        println!("🚪 [ACCEPT] Connection from peer: {}", peer_id);
        debug!("Accepted tensor connection from {}", peer_id);

        println!("📡 [ACCEPT] Setting up bidirectional stream...");
        // Set up a two-way communication channel
        // send = for sending data to the peer, recv = for receiving data from the peer
        let (mut send, mut recv) = connection.accept_bi().await?;
        println!("✅ [ACCEPT] Bidirectional stream established");

        println!("📥 [ACCEPT] Reading incoming message (max 100MB)...");
        // Read the incoming message dynamically with size limit
        // iroh-quinn's read_to_end takes a size_limit parameter and returns Vec<u8>
        let request_bytes = recv.read_to_end(MAX_MESSAGE_SIZE).await.map_err(AcceptError::from_err)?;
        println!("✅ [ACCEPT] Read {} bytes from peer", request_bytes.len());
        
        println!("🗜️ [ACCEPT] Deserializing message with postcard...");
        // Convert the raw bytes back into a TensorMessage using postcard
        // This is like opening an envelope and reading the letter inside
        // this part is GOOD for control plane (not much bytes)
        let message: TensorMessage = postcard::from_bytes(&request_bytes)
            .map_err(|e| {
                println!("❌ [ACCEPT] Deserialization failed: {:?}", e);
                AcceptError::from_err(e)
            })?;
        println!("✅ [ACCEPT] Message deserialized successfully");

        // Handle different types of messages
        match message {
            // Someone is asking us for a tensor
            TensorMessage::Request { tensor_name } => {
                println!("📥 [ACCEPT] Received REQUEST for tensor: {}", tensor_name);
                debug!("Received request for tensor: {}", tensor_name);

                println!("🔍 [ACCEPT] Looking up tensor in storage...");
                // Look up the tensor in our storage
                let response = {
                    let store = self.tensor_store.lock().unwrap();
                    match store.get(&tensor_name) {
                        // We have the tensor - send it back
                        Some(tensor_data) => {
                            println!("✅ [ACCEPT] Found tensor '{}' (size: {} bytes)", tensor_name, tensor_data.data.len());
                            TensorMessage::Response {
                                tensor_name: tensor_name.clone(),
                                data: tensor_data.clone(),
                            }
                        }
                        // We don't have the tensor - send an error
                        None => {
                            println!("❌ [ACCEPT] Tensor '{}' not found", tensor_name);
                            TensorMessage::Error {
                                message: format!("Tensor '{}' not found", tensor_name),
                            }
                        }
                    }
                };

                println!("🗜️ [ACCEPT] Serializing response...");
                // Convert our response back to bytes using postcard
                let response_bytes = postcard::to_allocvec(&response)
                    .map_err(|e| {
                        println!("❌ [ACCEPT] Response serialization failed: {:?}", e);
                        AcceptError::from_err(e)
                    })?;
                
                println!("📤 [ACCEPT] Sending response ({} bytes)...", response_bytes.len());
                // Send the response back to the peer
                send.write_all(&response_bytes).await.map_err(AcceptError::from_err)?;
                send.finish().map_err(AcceptError::from_err)?;
                println!("✅ [ACCEPT] Response sent successfully");
            }
            
            // Someone is sending us a tensor (unsolicited)
            TensorMessage::Response { tensor_name, data } => {
                println!("📥 [ACCEPT] Received RESPONSE with tensor: {} (size: {} bytes)", tensor_name, data.data.len());
                debug!("Received tensor data: {}", tensor_name);
                
                println!("📨 [ACCEPT] Forwarding tensor to receiver channel...");
                // If we have a receiver channel set up, forward the tensor to the main application
                if let Some(tx) = self.receiver_tx.lock().unwrap().as_ref() {
                    let send_result = tx.send((peer_id.clone(), tensor_name.clone(), data));
                    match send_result {
                        Ok(_) => println!("✅ [ACCEPT] Tensor forwarded to receiver channel"),
                        Err(e) => println!("❌ [ACCEPT] Failed to forward tensor: {:?}", e),
                    }
                } else {
                    println!("⚠️ [ACCEPT] No receiver channel available");
                }
            }
            
            // Someone sent us an error message
            TensorMessage::Error { message } => {
                println!("❌ [ACCEPT] Received ERROR from peer: {}", message);
                warn!("Received error from peer: {}", message);
            }
        }

        println!("⏳ [ACCEPT] Waiting for connection to close...");
        // Wait for the connection to close gracefully
        connection.closed().await;
        println!("🔒 [ACCEPT] Connection closed gracefully");
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
        println!("🏗️ [NEW] Creating new TensorNode...");
        
        println!("🏗️ [NEW] Creating protocol handler...");
        // Create the protocol handler
        let handler = Arc::new(TensorProtocolHandler::new());
        
        println!("🏗️ [NEW] Creating receiver channel...");
        // Create a channel for receiving tensors
        // tx = transmitter (sender), rx = receiver
        let (tx, rx) = mpsc::unbounded_channel();
        handler.set_receiver(tx);
        
        println!("✅ [NEW] TensorNode created successfully");

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
        println!("🚀 [START] Starting tensor node...");
        info!("Starting tensor node...");

        println!("🔧 [START] Creating endpoint builder...");
        // Create the network endpoint using Iroh's builder pattern
        // This is like setting up our network connection and getting a phone number
        let endpoint_result = Endpoint::builder()
            .discovery_n0()  // Use Iroh's discovery system to find other peers
            .bind()          // Actually create and bind the endpoint
            .await;
            
        println!("🔧 [START] Endpoint builder result: {:?}", endpoint_result.is_ok());
        
        let endpoint = endpoint_result.map_err(|e: BindError| {
            println!("❌ [START] Endpoint bind error: {}", e);
            TensorError::Connection { message: e.to_string() }
        })?;
        
        println!("✅ [START] Endpoint created successfully");

        println!("🔧 [START] Creating router...");
        // Create a router that will handle incoming connections
        // This is like hiring a receptionist to answer calls
        let router = Router::builder(endpoint.clone())
            .accept(TENSOR_ALPN, self.handler.clone())  // Accept connections for our protocol
            .spawn();  // Start the router in the background
            
        println!("✅ [START] Router created and spawned");

        println!("🔧 [START] Storing endpoint and router...");
        // Store the endpoint and router using interior mutability
        // This is thread-safe because of the Arc<Mutex<>>
        *self.endpoint.lock().unwrap() = Some(endpoint);
        *self.router.lock().unwrap() = Some(router);
        
        println!("✅ [START] Endpoint and router stored successfully");

        println!("🎉 [START] Tensor node started successfully");
        info!("Tensor node started successfully");
        Ok(())
    }

    // Sends a tensor directly to another peer (like mailing a package)
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn send_tensor_direct(
        &self,
        peer_addr: String,      // The NodeTicket of the peer (contains all addressing info)
        tensor_name: String,    // A name for the tensor
        tensor: TensorData,     // The actual tensor data
    ) -> Result<(), TensorError> {
        println!("📤 [SEND] Sending tensor '{}' to {} (size: {} bytes)", tensor_name, peer_addr, tensor.data.len());
        
        // println!("🔒 [SEND] Acquiring endpoint lock...");
        // Get our endpoint (make sure we're started)
        let endpoint = {
            let endpoint_guard = self.endpoint.lock().unwrap();
            // println!("🔒 [SEND] Endpoint lock acquired");
            
            endpoint_guard.as_ref()
                .ok_or_else(|| {
                    println!("❌ [SEND] Endpoint is None - node not started");
                    TensorError::Protocol { message: "Node not started".to_string() }
                })?
                .clone()
        };
        
        // println!("✅ [SEND] Endpoint acquired successfully");

        debug!("Sending tensor '{}' to {}", tensor_name, peer_addr);

        println!("🔍 [SEND] Parsing peer NodeTicket: {}", peer_addr);
        // ✅ FIX: Parse as NodeTicket instead of expecting a specific format
        // This automatically handles both relay URLs and direct addresses
        let ticket: NodeTicket = peer_addr.parse().map_err(|e| {
            println!("❌ [SEND] Failed to parse NodeTicket: {:?}", e);
            e
        })?;
        let node_addr: NodeAddr = ticket.into();
        
        println!("✅ [SEND] NodeTicket parsed successfully");
        // println!("🔍 [SEND] Peer node_id: {}", node_addr.node_id.fmt_short());
        // println!("🔍 [SEND] Peer relay_url: {:?}", node_addr.relay_url);
        // println!("🔍 [SEND] Peer direct_addresses: {} found", node_addr.direct_addresses.len());

        // println!("🔗 [SEND] Connecting to peer...");
        // Connect to the peer
        // Iroh will automatically try both relay and direct addresses
        let connection = endpoint.connect(node_addr, TENSOR_ALPN).await
            .map_err(|e: ConnectError| {
                println!("❌ [SEND] Connection failed: {}", e);
                TensorError::Connection { message: e.to_string() }
            })?;
        
        println!("✅ [SEND] Connected to peer successfully");

        println!("📡 [SEND] Opening bidirectional stream...");
        // Open a bidirectional stream (like opening a two-way conversation)
        let (mut send, mut _recv) = connection.open_bi().await.map_err(|e| {
            println!("❌ [SEND] Failed to open bidirectional stream: {:?}", e);
            TensorError::Connection { message: e.to_string() }
        })?;
        
        println!("✅ [SEND] Bidirectional stream opened");

        // println!("📦 [SEND] Creating message...");
        // Create a message containing our tensor
        let message = TensorMessage::Response {
            tensor_name: tensor_name.clone(),
            data: tensor,
        };

        println!("🗜️ [SEND] Serializing message with postcard...");
        // Convert the message to bytes using postcard (pack it in an envelope)
        let message_bytes = postcard::to_allocvec(&message).map_err(|e| {
            println!("❌ [SEND] Serialization failed: {:?}", e);
            e
        })?;
        
        println!("✅ [SEND] Message serialized successfully (size: {} bytes)", message_bytes.len());
        
        // println!("📡 [SEND] Writing message bytes to stream...");
        // Send the bytes over the network
        send.write_all(&message_bytes).await.map_err(|e| {
            println!("❌ [SEND] Failed to write message bytes: {:?}", e);
            e
        })?;
        
        println!("🏁 [SEND] Finishing send stream...");
        // Signal that we're done sending (like sealing the envelope)
        send.finish().map_err(|e| {
            println!("❌ [SEND] Failed to finish send stream: {}", e);
            TensorError::Connection { message: e.to_string() }
        })?;

        println!("🎉 [SEND] Tensor '{}' sent successfully", tensor_name);
        debug!("Tensor '{}' sent successfully", tensor_name);
        Ok(())
    }

    // Checks if we've received any tensors from other peers (like checking the mailbox)
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn receive_tensor(&self) -> Result<Option<TensorData>, TensorError> {
        println!("📬 [RECEIVE] Checking for received tensors...");
        
        // println!("🔒 [RECEIVE] Acquiring receiver lock...");
        let mut receiver_guard = self.receiver_rx.lock().unwrap();
        // println!("🔒 [RECEIVE] Receiver lock acquired");
        
        if let Some(rx) = receiver_guard.as_mut() {
            println!("✅ [RECEIVE] Receiver channel found, trying to receive...");
            match rx.try_recv() {
                // We got a tensor!
                Ok((peer_id, tensor_name, tensor_data)) => {
                    println!("🎉 [RECEIVE] Received tensor '{}' from {} (size: {} bytes)", tensor_name, peer_id, tensor_data.data.len());
                    debug!("Received tensor '{}' from {}", tensor_name, peer_id);
                    Ok(Some(tensor_data))
                }
                // No tensors waiting
                Err(mpsc::error::TryRecvError::Empty) => {
                    println!("📭 [RECEIVE] No tensors waiting");
                    Ok(None)
                }
                // The channel is broken
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    println!("❌ [RECEIVE] Receiver channel disconnected");
                    Err(TensorError::Protocol { message: "Receiver disconnected".to_string() })
                }
            }
        } else {
            println!("❌ [RECEIVE] No receiver channel - node not started");
            Err(TensorError::Protocol { message: "Node not started".to_string() })
        }
    }

    // Gets our node's address so others can connect to us (like getting our phone number)
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn get_node_addr(&self) -> Result<String, TensorError> {
        println!("📞 [GET_ADDR] Getting node address...");
        
        println!("🔒 [GET_ADDR] Acquiring endpoint lock...");
        let endpoint = {
            let endpoint_guard = self.endpoint.lock().unwrap();
            println!("🔒 [GET_ADDR] Endpoint lock acquired");
            
            let endpoint_ref = endpoint_guard.as_ref()
                .ok_or_else(|| {
                    println!("❌ [GET_ADDR] Endpoint is None - node not started");
                    TensorError::Protocol {
                        message: "Node not started".into(),
                    }
                })?;
            println!("✅ [GET_ADDR] Endpoint found, cloning...");
            endpoint_ref.clone()
        };
        
        println!("🔍 [GET_ADDR] Calling endpoint.node_addr().initialized()...");
        // Wait for our address to be initialized by the discovery system
        let result = endpoint.node_addr().initialized().await
            .map_err(|err| {
                println!("❌ [GET_ADDR] Discovery watcher error: {:?}", err);
                TensorError::Protocol { message: "Discovery watcher error".into() }
            })?;
        
        println!("✅ [GET_ADDR] Got initialized result: {:?}", result);
        // println!("🔍 [GET_ADDR] Checking addresses...");
        // println!("🔍 [GET_ADDR] relay_url is_some: {}", result.relay_url.is_some());
        // println!("🔍 [GET_ADDR] direct_addresses count: {}", result.direct_addresses.len());
        
        if let Some(ref relay_url) = result.relay_url {
            println!("✅ [GET_ADDR] Found relay_url: {:?}", relay_url);
        }
        
        for (i, direct_addr) in result.direct_addresses.iter().enumerate() {
            println!("✅ [GET_ADDR] Direct address {}: {}", i, direct_addr);
        }
        
        // ✅ FIX: Create a proper NodeTicket with ALL address info (relay + direct)
        // This works whether we have relay servers, direct addresses, or both!
        if result.relay_url.is_none() && result.direct_addresses.is_empty() {
            println!("❌ [GET_ADDR] No addressing information available");
            return Err(TensorError::Protocol { 
                message: "No relay URL or direct addresses available".into() 
            });
        }
        
        // Create a NodeTicket containing the complete addressing information
        let node_ticket = NodeTicket::new(result);
        let ticket_string = node_ticket.to_string();
        
        println!("🎉 [GET_ADDR] Created NodeTicket: {}", ticket_string);
        Ok(ticket_string)
    }

    // Stores a tensor locally so others can request it (like putting something in storage)
    #[uniffi::method]
    pub fn register_tensor(&self, name: String, tensor: TensorData) -> Result<(), TensorError> {
        println!("📋 [REGISTER] Registering tensor '{}' (size: {} bytes)", name, tensor.data.len());
        println!("📋 [REGISTER] Tensor metadata: shape={:?}, dtype={}, requires_grad={}", 
                 tensor.metadata.shape, tensor.metadata.dtype, tensor.metadata.requires_grad);
        
        debug!("Registering tensor: {}", name);
        self.handler.register_tensor(name.clone(), tensor);
        
        println!("✅ [REGISTER] Tensor '{}' registered successfully", name);
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
