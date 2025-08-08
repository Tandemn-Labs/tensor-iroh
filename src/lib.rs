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
    time::{Duration, Instant},
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
use tracing::{debug, error, info, warn, trace}; // For logging what's happening (like print statements but better)
use iroh::Watcher;                // For watching network changes

use tokio::sync::{Mutex as AsyncMutex, RwLock};    //

// log only errors or warnings for production performance
use std::sync::Once;
static INIT_TRACING: Once = Once::new();

fn init_tracing() {
    INIT_TRACING.call_once(|| {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)  // Changed from ERROR to WARN for important warnings only
            .init();
    });
}

// This tells UniFFI (the cross-language binding generator) to set up the scaffolding
// It's like telling the "translator" to get ready to translate our Rust code
uniffi::setup_scaffolding!();

// ============================================================================
// CONSTANTS - Fixed values used throughout the program
// ============================================================================

// This is like a "protocol ID" - it tells other computers what kind of data we're sending
// Think of it like saying "I speak Tensor Protocol version 0" when connecting
const TENSOR_ALPN: &[u8] = b"tensor-iroh/direct/0";
const CHUNK_SIZE: usize = 1024 * 1024; // 1MB chunks to avoid excessive chunking for small messages
const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024; // 100MB total limit

// Optimized connection pool settings
const MAX_CONNECTIONS: usize = 50;  // Increased from 10 for better concurrency
const CONNECTION_IDLE_TIMEOUT: Duration = Duration::from_secs(600);  // 10 minutes instead of 5


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


impl From<ConnectError> for TensorError {
    fn from(err: ConnectError) -> Self {
        TensorError::Connection { message: err.to_string() }
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

// Add the new ReceivedTensor struct after TensorData
#[derive(Debug, Clone, uniffi::Record)]
pub struct ReceivedTensor {
    pub name: String,
    pub data: TensorData,
}

// ============================================================================
// CONNECTION POOL - A simple implementation of a connection pool
// ============================================================================
// this is a struct for the TensorNode's PooledConnection
// The idea of the PooledConnection is to keep a list of connections to other peers
// and to be able to reuse them.
// It is a simple implementation of a connection pool. Look into it more
#[derive(Debug)] // no need to provide Clone access to it
pub struct PooledConnection {
    connections: Connection, // this stores the actual object of Iroh Connection
    last_used: Arc<Mutex<Instant>>, // this stores the last time the connection was used
    is_idle: Arc<Mutex<bool>>, // this stores whether the connection is idle
}

#[derive(Debug)] // no need to provide Clone access to it
pub struct ConnectionPool{
    connections: Arc<AsyncMutex<HashMap<String, PooledConnection>>>, // this stores the connections to other peers 
    max_idle_time: Duration, // this stores the maximum idle time of the connection
    max_connections: usize, // this stores the maximum number of connections to other peers
}

impl ConnectionPool {
    // Creates a new connection pool with specified limits
    // max_connections: Maximum number of concurrent connections to maintain
    // max_idle_time: How long a connection can sit unused before being closed
    fn new(max_connections: usize, max_idle_time: Duration) -> Self {
        Self {
            // Initialize with an empty HashMap wrapped in Arc<Mutex<>> for thread-safe access
            // Arc allows multiple threads to share ownership of the HashMap
            // Mutex ensures only one thread can modify the HashMap at a time
            connections: Arc::new(AsyncMutex::new(HashMap::new())),
            max_idle_time,
            max_connections,
        }
    }

    async fn get_connection(&self, peer_id: &str, endpoint: &Endpoint, node_addr: &NodeAddr) -> Result<Connection, TensorError> {
        // Fast-path: try to reuse an existing, still-fresh connection
        {
            let mut conns = self.connections.lock().await;
            if let Some(pc) = conns.get_mut(peer_id) {
                let elapsed_since_last_use = pc.last_used.lock().unwrap().elapsed();
                if elapsed_since_last_use < self.max_idle_time {
                    // Mark busy again and update last used
                    *pc.is_idle.lock().unwrap() = false;
                    *pc.last_used.lock().unwrap() = Instant::now();
                    return Ok(pc.connections.clone());
                } else {
                    // Stale entry; remove so we create a fresh connection below
                    conns.remove(peer_id);
                }
            }
        }

        // Slow-path: create new connection without holding the pool lock
        let connection = endpoint
            .connect(node_addr.clone(), TENSOR_ALPN)
            .await
            .map_err(|e| TensorError::Connection { message: e.to_string() })?;

        // Insert into pool as busy (not idle) and update last_used
        {
            let mut conns = self.connections.lock().await;
            conns.insert(
                peer_id.to_string(),
                PooledConnection {
                    connections: connection.clone(),
                    last_used: Arc::new(Mutex::new(Instant::now())),
                    is_idle: Arc::new(Mutex::new(false)),
                },
            );
        }

        Ok(connection)
    }
    
    // Returns a connection to the pool when it's no longer needed
    // This allows the connection to be reused by other operations
    // async fn return_connection(&self, peer_id: &str) {
    //     if let Some(pooled_conn) = self.connections.lock().unwrap().get(peer_id) {
        // Mark the connection as idle so it can be reused
    //         *pooled_conn.is_idle.lock().unwrap() = true;
    //         // Update the last used time
    //         *pooled_conn.last_used.lock().unwrap() = Instant::now();
    //     }
    // }

    async fn return_connection(&self, peer_id: &str) {
        let conns = self.connections.lock().await;
        if let Some(pc) = conns.get(peer_id) {
            *pc.is_idle.lock().unwrap() = true;
            *pc.last_used.lock().unwrap() = Instant::now();
        }
    }

    // Cleans up expired connections from the pool
    // This should be called periodically to prevent memory leaks
    async fn cleanup_expired_connections(&self) {
        // let mut connections = self.connections.lock().unwrap();
        let mut connections = self.connections.lock().await;
        let now = Instant::now();
        
        // Remove connections that have been idle for too long
        connections.retain(|_, pooled_conn| {
            let last_used = *pooled_conn.last_used.lock().unwrap();
            let is_idle = *pooled_conn.is_idle.lock().unwrap();
            
            // Keep the connection if it's not idle or hasn't exceeded max idle time
            !is_idle || (now.duration_since(last_used) < self.max_idle_time)
        });
    }   
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
        let peer_id = peer_id_result?.to_string();
        trace!("🚪 [ACCEPT] Accepted tensor connection from {}", peer_id);

        // Process multiple streams on this single QUIC connection
        loop {
            let (mut send, mut recv) = match connection.accept_bi().await {
                Ok(pair) => {
                    trace!("🔗 [ACCEPT] Bidirectional stream established with {}", peer_id);
                    pair
                }
                Err(e) => {
                    // Connection closed or errored; end handler
                    trace!("🔒 [ACCEPT] Connection closed or no more streams from {}: {:?}", peer_id, e);
                    break;
                }
            };

            // Read message type (0 = single, >0 = chunked)
            let mut type_buf = [0u8; 4];
            recv.read_exact(&mut type_buf).await.map_err(AcceptError::from_err)?;
            let message_type = u32::from_le_bytes(type_buf);

            let request_bytes = if message_type == 0 {
                // Single message
                receive_length_prefixed_message(&mut recv).await?
            } else {
                // Chunked message
                receive_chunked_message(&mut recv, message_type).await?
            };
            trace!("🔍 [ACCEPT] Read {} bytes from peer {}", request_bytes.len(), peer_id);

            // Deserialize the message
            let message: TensorMessage = postcard::from_bytes(&request_bytes)
                .map_err(|e| {
                    error!("❌ [ACCEPT] Deserialization failed: {:?}", e);
                    AcceptError::from_err(e)
                })?;
            trace!("✅ [ACCEPT] Message deserialized successfully");

            // Handle different types of messages
            match message {
                // Someone is asking us for a tensor
                TensorMessage::Request { tensor_name } => {
                    info!("📥 [ACCEPT] Received REQUEST for tensor: {}", tensor_name);
                    debug!("Received request for tensor: {}", tensor_name);

                    debug!("🔍 [ACCEPT] Looking up tensor in storage...");
                    // Look up the tensor in our storage
                    let response = {
                        let store = self.tensor_store.lock().unwrap();
                        match store.get(&tensor_name) {
                            // We have the tensor - send it back
                            Some(tensor_data) => {
                                info!("✅ [ACCEPT] Found tensor '{}' (size: {} bytes)", tensor_name, tensor_data.data.len());
                                TensorMessage::Response {
                                    tensor_name: tensor_name.clone(),
                                    data: tensor_data.clone(),
                                }
                            }
                            // We don't have the tensor - send an error
                            None => {
                                warn!("❌ [ACCEPT] Tensor '{}' not found", tensor_name);
                                TensorMessage::Error {
                                    message: format!("Tensor '{}' not found", tensor_name),
                                }
                            }
                        }
                    };

                    debug!("🗜️ [ACCEPT] Serializing response...");
                    // Convert our response back to bytes using postcard
                    let response_bytes = postcard::to_allocvec(&response)
                        .map_err(|e| {
                            error!("❌ [ACCEPT] Response serialization failed: {:?}", e);
                            AcceptError::from_err(e)
                        })?;

                    debug!("📤 [ACCEPT] Sending response ({} bytes)...", response_bytes.len());
                    // Send the response back to the peer
                    send.write_all(&response_bytes).await.map_err(AcceptError::from_err)?;
                    send.finish().map_err(AcceptError::from_err)?;
                    info!("✅ [ACCEPT] Response sent successfully");
                }

                // Someone is sending us a tensor (unsolicited)
                TensorMessage::Response { tensor_name, data } => {
                    info!("📥 [ACCEPT] Received RESPONSE with tensor: {} (size: {} bytes)", tensor_name, data.data.len());
                    debug!("Received tensor data: {}", tensor_name);

                    debug!("📨 [ACCEPT] Forwarding tensor to receiver channel...");
                    // If we have a receiver channel set up, forward the tensor to the main application
                    if let Some(tx) = self.receiver_tx.lock().unwrap().as_ref() {
                        let send_result = tx.send((peer_id.clone(), tensor_name.clone(), data));
                        match send_result {
                            Ok(_) => info!("✅ [ACCEPT] Tensor forwarded to receiver channel"),
                            Err(e) => error!("❌ [ACCEPT] Failed to forward tensor: {:?}", e),
                        }
                    } else {
                        warn!("⚠️ [ACCEPT] No receiver channel available");
                    }

                    // We didn't use our send half; close it to tidy up
                    let _ = send.finish();
                }

                // Someone sent us an error message
                TensorMessage::Error { message } => {
                    warn!("❌ [ACCEPT] Received ERROR from peer: {}", message);
                    let _ = send.finish();
                }
            }
        }

        // Done handling this connection
        Ok(())
    }
}

// ============================================================================
// MAIN TENSORNODE IMPLEMENTATION - The primary interface for our protocol
// ============================================================================

// This is the main class that users will interact with
// Think of it as the "control panel" for our tensor networking system
#[derive(uniffi::Object, Clone)]
pub struct TensorNode {
    // The network endpoint (like our "network interface card")
    endpoint: Arc<Mutex<Option<Endpoint>>>,
    
    // The router that handles incoming connections (like a switchboard operator)
    router: Arc<Mutex<Option<Router>>>,
    
    // Our protocol handler (the "receptionist")
    handler: Arc<TensorProtocolHandler>,
    
    // Channel for receiving tensors from other peers
    receiver_rx: Arc<AsyncMutex<Option<mpsc::UnboundedReceiver<(String, String, TensorData)>>>>,

    // Adding the connection Pool here so that the TensorNode can use it
    // connection_pool: Arc<RwLock<ConnectionPool>>, // making sure the entire ConnectionPool
    connection_pool: Arc<ConnectionPool>, // making sure the entire ConnectionPool is thread safe
    // struct is thread safe, and hence we have to add RwLock and Arc to it. 
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

        // Create the connection pool with optimized settings
        let connection_pool = Arc::new(ConnectionPool::new(
            MAX_CONNECTIONS,
            CONNECTION_IDLE_TIMEOUT));

        Self {
            // We use Arc<Mutex<Option<>>> for thread-safe lazy initialization
            // This means "maybe we have an endpoint, maybe we don't, but if we do, it's thread-safe"
            endpoint: Arc::new(Mutex::new(None)),
            router: Arc::new(Mutex::new(None)),
            handler,
            receiver_rx: Arc::new(AsyncMutex::new(Some(rx))),
            connection_pool, // send the connection pool to the TensorNode 
        }
    }

    // Starts the tensor node (like "opening for business")
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn start(&self) -> Result<(), TensorError> {
        init_tracing(); // Initialize logging at WARN level only
        
        // Create the network endpoint using Iroh's builder pattern
        // This is like setting up our network connection and getting a phone number
        let endpoint_result = Endpoint::builder()
            .discovery_n0()  // Use Iroh's discovery system to find other peers
            .bind()          // Actually create and bind the endpoint
            .await;
            
        let endpoint = endpoint_result.map_err(|e: BindError| {
            error!("❌ [START] Endpoint bind error: {}", e);
            TensorError::Connection { message: e.to_string() }
        })?;
        
        trace!("✅ [START] Endpoint created successfully");

        // Create a router that will handle incoming connections
        // This is like hiring a receptionist to answer calls
        let router = Router::builder(endpoint.clone())
            .accept(TENSOR_ALPN, self.handler.clone())  // Accept connections for our protocol
            .spawn();  // Start the router in the background

        // Store the endpoint and router using interior mutability
        // This is thread-safe because of the Arc<Mutex<>>
        *self.endpoint.lock().unwrap() = Some(endpoint);
        *self.router.lock().unwrap() = Some(router);
        
        trace!("🎉 [START] Tensor node started successfully");
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
        trace!("🔍 [SEND] Sending tensor '{}' to {} (size: {} bytes)", tensor_name, peer_addr, tensor.data.len());
        
        // Get our endpoint (make sure we're started)
        let endpoint = {
            let endpoint_guard = self.endpoint.lock().unwrap();
            
            endpoint_guard.as_ref()
                .ok_or_else(|| {
                    error!("❌ [SEND] Endpoint is None - node not started");
                    TensorError::Protocol { message: "Node not started".to_string() }
                })?
                .clone()
        };

        info!("🔍 [SEND] Parsing peer NodeTicket: {}", peer_addr);
        // ✅ FIX: Parse as NodeTicket instead of expecting a specific format
        // This automatically handles both relay URLs and direct addresses
        let ticket: NodeTicket = peer_addr.parse().map_err(|e| {
            error!("❌ [SEND] Failed to parse NodeTicket: {:?}", e);
            e
        })?;
        let node_addr: NodeAddr = ticket.into();
        
        info!("✅ [SEND] NodeTicket parsed successfully");
        // info!("🔍 [SEND] Peer node_id: {}", node_addr.node_id.fmt_short());
        // info!("🔍 [SEND] Peer relay_url: {:?}", node_addr.relay_url);
        // info!("🔍 [SEND] Peer direct_addresses: {} found", node_addr.direct_addresses.len());

        // Connect to the peer
        // Iroh will automatically try both relay and direct addresses
        // Get connection from pool
        trace!("🔗 [SEND] Getting Peer From the existing pool (if it exists)... ");
        let peer_id = node_addr.node_id.to_string();
        // let connection = {
        //     let mut pool = self.connection_pool.read().await; // get the connection pool in an async way
        //     pool.get_connection(&peer_id, &endpoint, &node_addr).await?
        // };

        let connection = self.connection_pool.get_connection(&peer_id, &endpoint, &node_addr).await?;

        // let connection = endpoint.connect(node_addr, TENSOR_ALPN).await
        //     .map_err(|e: ConnectError| {
        //         error!("❌ [SEND] Connection failed: {}", e);
        //         TensorError::Connection { message: e.to_string() }
        //     })?;
        
        trace!("✅ [SEND] Connected to peer successfully");

        trace!("📡 [SEND] Opening bidirectional stream...");
        // Open a bidirectional stream (like opening a two-way conversation)
        let (mut send, mut _recv) = connection.open_bi().await.map_err(|e| {
            error!("❌ [SEND] Failed to open bidirectional stream: {:?}", e);
            TensorError::Connection { message: e.to_string() }
        })?;

        // Create a message containing our tensor
        let message = TensorMessage::Response {
            tensor_name: tensor_name.clone(),
            data: tensor,
        };

        // Convert the message to bytes using postcard (pack it in an envelope)
        let message_bytes = postcard::to_allocvec(&message).map_err(|e| {
            error!("❌ [SEND] Serialization failed: {:?}", e);
            e
        })?;
        
        trace!("✅ [SEND] Message serialized successfully (size: {} bytes)", message_bytes.len());
        
        // Check if we need chunking
        if message_bytes.len() > CHUNK_SIZE {
            trace!("Using chunked protocol for {} bytes", message_bytes.len());
            send_chunked_message(&mut send, &message_bytes).await?;
        } else {
            trace!("Using length-prefixed protocol for {} bytes", message_bytes.len());
            send_length_prefixed_message(&mut send, &message_bytes).await?;
        }
        
        trace!("🏁 [SEND] Finishing send stream...");
        send.finish().map_err(|e| {
            TensorError::Connection { message: e.to_string() }
        })?;
        
        // tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // info!("⏳ [SEND] Waiting for connection to close gracefully...");
        // connection.closed().await;
        // info!("🔒 [SEND] Connection closed gracefully");
        
        // Return connection to pool instead of dropping it
        // {
        //     let mut pool = self.connection_pool.write().await;
        //     pool.return_connection(&peer_id, connection).await;
        // }  
        // drop(connection);

        self.connection_pool.return_connection(&peer_id).await;


        trace!("🎉 [SEND] Tensor '{}' sent successfully", tensor_name);
        trace!("Tensor '{}' sent successfully", tensor_name);
        Ok(())
    }

    // Checks if we've received any tensors from other peers (like checking the mailbox)
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn receive_tensor(&self) -> Result<Option<ReceivedTensor>, TensorError> {
        info!("📬 [RECEIVE] Checking for received tensors...");
        
        // info!("🔒 [RECEIVE] Acquiring receiver lock...");
        let mut receiver_guard = self.receiver_rx.lock().await;
        // info!("🔒 [RECEIVE] Receiver lock acquired");
        
        if let Some(rx) = receiver_guard.as_mut() {
            info!("✅ [RECEIVE] Receiver channel found, trying to receive...");
            match rx.try_recv() {
                // We got a tensor!
                Ok((peer_id, tensor_name, tensor_data)) => {
                    info!("🎉 [RECEIVE] Received tensor '{}' from {} (size: {} bytes)", 
                        tensor_name, peer_id, tensor_data.data.len());
                    debug!("Received tensor '{}' from {}", tensor_name, peer_id);
                    Ok(Some(ReceivedTensor {
                        name: tensor_name,
                        data: tensor_data,
                    }))
                }
                // No tensors waiting
                Err(mpsc::error::TryRecvError::Empty) => {
                    // info!("📭 [RECEIVE] No tensors waiting");
                    Ok(None)
                }
                // The channel is broken
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    error!("❌ [RECEIVE] Receiver channel disconnected");
                    Err(TensorError::Protocol { message: "Receiver disconnected".to_string() })
                }
            }
        } else {
            error!("❌ [RECEIVE] No receiver channel - node not started");
            Err(TensorError::Protocol { message: "Node not started".to_string() })
        }
    }

    // Wait for the next tensor (async, no busy-polling)
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn wait_for_tensor(&self) -> Result<ReceivedTensor, TensorError> {
        info!("📬 [RECEIVE] Waiting for next tensor...");
        let mut receiver_guard = self.receiver_rx.lock().await;
        if let Some(rx) = receiver_guard.as_mut() {
            match rx.recv().await {
                Some((_peer_id, tensor_name, tensor_data)) => {
                    info!(
                        "🎉 [RECEIVE] Received tensor '{}' (size: {} bytes)",
                        tensor_name,
                        tensor_data.data.len()
                    );
                    Ok(ReceivedTensor { name: tensor_name, data: tensor_data })
                }
                None => {
                    error!("❌ [RECEIVE] Receiver channel disconnected");
                    Err(TensorError::Protocol { message: "Receiver disconnected".to_string() })
                }
            }
        } else {
            error!("❌ [RECEIVE] No receiver channel - node not started");
            Err(TensorError::Protocol { message: "Node not started".to_string() })
        }
    }

    // Gets our node's address so others can connect to us (like getting our phone number)
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn get_node_addr(&self) -> Result<String, TensorError> {
        info!("📞 [GET_ADDR] Getting node address...");
        
        info!("🔒 [GET_ADDR] Acquiring endpoint lock...");
        let endpoint = {
            let endpoint_guard = self.endpoint.lock().unwrap();
            info!("🔒 [GET_ADDR] Endpoint lock acquired");
            
            let endpoint_ref = endpoint_guard.as_ref()
                .ok_or_else(|| {
                    error!("❌ [GET_ADDR] Endpoint is None - node not started");
                    TensorError::Protocol {
                        message: "Node not started".into(),
                    }
                })?;
            info!("✅ [GET_ADDR] Endpoint found, cloning...");
            endpoint_ref.clone()
        };
        
        info!("🔍 [GET_ADDR] Calling endpoint.node_addr().initialized()...");
        // Wait for our address to be initialized by the discovery system
        let result = endpoint.node_addr().initialized().await
            .map_err(|err| {
                error!("❌ [GET_ADDR] Discovery watcher error: {:?}", err);
                TensorError::Protocol { message: "Discovery watcher error".into() }
            })?;
        
        info!("✅ [GET_ADDR] Got initialized result: {:?}", result);
        // info!("🔍 [GET_ADDR] Checking addresses...");
        // info!("🔍 [GET_ADDR] relay_url is_some: {}", result.relay_url.is_some());
        // info!("🔍 [GET_ADDR] direct_addresses count: {}", result.direct_addresses.len());
        
        if let Some(ref relay_url) = result.relay_url {
            info!("✅ [GET_ADDR] Found relay_url: {:?}", relay_url);
        }
        
        for (i, direct_addr) in result.direct_addresses.iter().enumerate() {
            info!("✅ [GET_ADDR] Direct address {}: {}", i, direct_addr);
        }
        
        // ✅ FIX: Create a proper NodeTicket with ALL address info (relay + direct)
        // This works whether we have relay servers, direct addresses, or both!
        if result.relay_url.is_none() && result.direct_addresses.is_empty() {
            error!("❌ [GET_ADDR] No addressing information available");
            return Err(TensorError::Protocol { 
                message: "No relay URL or direct addresses available".into() 
            });
        }
        
        // Create a NodeTicket containing the complete addressing information
        let node_ticket = NodeTicket::new(result);
        let ticket_string = node_ticket.to_string();
        
        error!("🎉 [GET_ADDR] Created NodeTicket: {}", ticket_string);
        Ok(ticket_string)
    }

    // Stores a tensor locally so others can request it (like putting something in storage)
    #[uniffi::method]
    pub fn register_tensor(&self, name: String, tensor: TensorData) -> Result<(), TensorError> {
        info!("📋 [REGISTER] Registering tensor '{}' (size: {} bytes)", name, tensor.data.len());
        info!("📋 [REGISTER] Tensor metadata: shape={:?}, dtype={}, requires_grad={}", 
                 tensor.metadata.shape, tensor.metadata.dtype, tensor.metadata.requires_grad);
        
        debug!("Registering tensor: {}", name);
        self.handler.register_tensor(name.clone(), tensor);
        
        info!("✅ [REGISTER] Tensor '{}' registered successfully", name);
        Ok(())
    }

    // Shuts down the node (like closing the office)
    #[uniffi::method]
    pub fn shutdown(&self) -> Result<(), TensorError> {
        info!("🔒 [SHUTDOWN] Shutting down tensor node...");
        
        // Take and drop the router (stops accepting new connections)
        if let Some(router) = self.router.lock().unwrap().take() {
            info!("✅ [SHUTDOWN] Router stopped");
            drop(router);
        }
        
        // Take and drop the endpoint (closes all connections)
        if let Some(endpoint) = self.endpoint.lock().unwrap().take() {
            info!("✅ [SHUTDOWN] Endpoint closed");
            drop(endpoint);
        }
        
        // Take and drop the receiver channel (closes the channel)
        {
            let mut receiver_guard = self.receiver_rx.blocking_lock();
            if let Some(receiver) = receiver_guard.take() {
                info!("✅ [SHUTDOWN] Receiver channel closed");
                drop(receiver);
            }
        }
        
        info!("🎉 [SHUTDOWN] Tensor node shutdown complete");
        Ok(())
    }

    /// Number of currently-pooled connections
    #[uniffi::method(async_runtime = "tokio")]
    pub async fn get_pool_size(&self) -> Result<u32, TensorError> {
        // grab the map inside the pool and return its len()
        let conns = self.connection_pool.connections.lock().await;
        Ok(conns.len() as u32)
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

// Compression helpers
fn compress_data(data: &[u8]) -> Result<Vec<u8>, TensorError> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

fn decompress_data(data: &[u8]) -> Result<Vec<u8>, TensorError> {
    use flate2::read::GzDecoder;
    use std::io::Read;
    
    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

// New helper functions
async fn send_chunked_message(
    send: &mut iroh::endpoint::SendStream,
    data: &[u8],
) -> Result<(), TensorError> {
    trace!("Starting chunked send of {} bytes", data.len());
    
    // Send number of chunks first (this becomes the message_type)
    let num_chunks = (data.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    send.write_all(&(num_chunks as u32).to_le_bytes()).await?;
    
    // Send total length
    let total_len = data.len() as u32;
    send.write_all(&total_len.to_le_bytes()).await?;
    
    trace!("Sent headers: {} chunks, {} total bytes", num_chunks, total_len);
    
    // Send each chunk with its size
    for (chunk_idx, chunk) in data.chunks(CHUNK_SIZE).enumerate() {
        let chunk_size = chunk.len() as u32;
        send.write_all(&chunk_size.to_le_bytes()).await?;
        send.write_all(chunk).await?;
        trace!("Sent chunk {}/{}: {} bytes", chunk_idx + 1, num_chunks, chunk_size);
    }
    
    // Note: Stream will be finished by the caller
    trace!("All chunks sent successfully");
    
    Ok(())
}

async fn send_length_prefixed_message(
    send: &mut iroh::endpoint::SendStream,
    data: &[u8],
) -> Result<(), TensorError> {
    trace!("Sending {} bytes", data.len());
    // Send 0 to indicate non-chunked
    send.write_all(&0u32.to_le_bytes()).await?;
    // Send length
    send.write_all(&(data.len() as u32).to_le_bytes()).await?;
    // Send data
    send.write_all(data).await?;
    trace!("Data sent successfully");
    Ok(())
}

async fn receive_length_prefixed_message(
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<Vec<u8>, AcceptError> {
    // Read length
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await.map_err(AcceptError::from_err)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    
    if len > MAX_MESSAGE_SIZE {
        return Err(AcceptError::from_err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Message too large"
        )));
    }
    
    // Read data
    let mut data = vec![0u8; len];
    recv.read_exact(&mut data).await.map_err(AcceptError::from_err)?;
    Ok(data)
}

async fn receive_chunked_message(
    recv: &mut iroh::endpoint::RecvStream,
    num_chunks: u32,
) -> Result<Vec<u8>, AcceptError> {
    trace!("Reading total length...");
    // Read total length
    let mut total_len_buf = [0u8; 4];
    recv.read_exact(&mut total_len_buf).await.map_err(AcceptError::from_err)?;
    let total_len = u32::from_le_bytes(total_len_buf) as usize;
    
    trace!("Total length: {} bytes, {} chunks", total_len, num_chunks);
    
    if total_len > MAX_MESSAGE_SIZE {
        return Err(AcceptError::from_err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Message too large"
        )));
    }
    
    let mut data = Vec::with_capacity(total_len);
    
    // Read each chunk
    for chunk_idx in 0..num_chunks {
        trace!("Reading chunk {}/{}", chunk_idx + 1, num_chunks);
        
        // Read chunk size
        let mut chunk_size_buf = [0u8; 4];
        recv.read_exact(&mut chunk_size_buf).await.map_err(AcceptError::from_err)?;
        let chunk_size = u32::from_le_bytes(chunk_size_buf) as usize;
        
        trace!("Chunk {} size: {} bytes", chunk_idx + 1, chunk_size);
        
        // Read chunk data
        let mut chunk_data = vec![0u8; chunk_size];
        recv.read_exact(&mut chunk_data).await.map_err(AcceptError::from_err)?;
        data.extend_from_slice(&chunk_data);
        
        trace!("Chunk {} read successfully, total so far: {} bytes", chunk_idx + 1, data.len());
    }
    
    trace!("All chunks read successfully! Total: {} bytes", data.len());
    Ok(data)
} 

#[cfg(feature = "python")]
pub mod pyo3_mod;