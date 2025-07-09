use std::sync::Arc;
use tokio::sync::Mutex;
use iroh_net::{Endpoint, NodeAddr};
use iroh_net::key::SecretKey;
use tokio::io::{AsyncReadExt, AsyncWriteExt}; 
use serde::{Deserialize, Serialize};
use anyhow::Result;

// 1. --- FFI Bridge Setup ---
// This macro and the UDL file are what UniFFI uses to generate the
// Python-to-Rust bridge code.
uniffi::include_scaffolding!("tensor_protocol");

// 2. --- Protocol Message Definition ---
// This defines the "language" our nodes will speak to each other.
// It's a simple enum that can represent metadata, data chunks, or the end of a stream.
#[derive(Serialize, Deserialize, Debug)]
enum Message {
    Header(TensorMetadata),
    DataChunk(Vec<u8>),
    End,
}

// Metadata about the tensor being sent.
#[derive(Serialize, Deserialize, Debug, uniffi::Record)]
pub struct TensorMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
}

// 3. --- Callback for Receiving Tensors in Python ---
// This defines a "trait" or "interface" that a Python object must implement.
// Our Rust code will call the `on_tensor` method when it successfully receives a tensor.
#[uniffi::export(with_foreign)]
pub trait TensorCallback: Send + Sync {
    fn on_tensor(&self, metadata: TensorMetadata, data: Vec<u8>);
}


// 4. --- The Core Node Object ---
// This is the main object Python will interact with. It holds the Iroh endpoint
// and all the logic for sending and receiving.
#[derive(uniffi::Object)]
pub struct TensorNode {
    // We wrap the endpoint in an Arc<Mutex<Option<...>>> to allow it to be
    // initialized asynchronously after the object is created.
    endpoint: Arc<Mutex<Option<Endpoint>>>,
}

#[uniffi::export]
impl TensorNode {
    // The constructor is simple: it just creates the placeholder.
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            endpoint: Arc::new(Mutex::new(None)),
        })
    }

    // Starts the underlying Iroh node, binding to a random port.
    // pub async fn start(&self) -> Result<(), anyhow::Error> {
    pub async fn start(&self) -> Result<()> {
        let mut endpoint_guard = self.endpoint.lock().await;
        if endpoint_guard.is_some() {
            return Ok(()); // Already started
        }
        let secret_key = SecretKey::generate();
        let endpoint = Endpoint::builder()
            .secret_key(secret_key)
            .bind() // Bind to a random available port
            .await?;
        *endpoint_guard = Some(endpoint);
        Ok(())
    }

    // Returns the unique NodeId and addresses of this node, needed for others to connect.
    // pub async fn node_addr(&self) -> Result<String, anyhow::Error> {
    pub async fn node_addr(&self) -> Result<String> {
        let endpoint_guard = self.endpoint.lock().await;
        let endpoint = endpoint_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Node not started"))?;
        let addr = endpoint.node_addr().await?;
        Ok(addr.to_string())
    }

    // The "server" part. This listens for incoming tensor streams.
    // pub async fn listen(&self, callback: Arc<dyn TensorCallback>) -> Result<(), anyhow::Error> {
    pub async fn listen(&self, callback: Arc<dyn TensorCallback>) -> Result<()> {
        let endpoint_guard = self.endpoint.lock().await;
        let endpoint = endpoint_guard.clone().ok_or_else(|| anyhow::anyhow!("Node not started"))?;

        // Spawn a background task to accept incoming connections.
        tokio::spawn(async move {
            // The ALPN identifies our specific protocol.
            while let Some(connecting) = endpoint.accept(b"tensor-iroh/0.1").await {
                let callback = callback.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_incoming_stream(connecting, callback).await {
                        // In a real app, you'd use a logging framework like `tracing`.
                        println!("[ERROR] Failed to handle incoming stream: {:?}", e);
                    }
                });
            }
        });

        Ok(())
    }

    // The "client" part. This connects to a peer and sends a tensor.
    pub async fn send_tensor(   
        &self,
        peer_addr_str: String,
        metadata: TensorMetadata,
        data: Vec<u8>,
    ) -> Result<()> {
        let endpoint_guard = self.endpoint.lock().await;
        let endpoint = endpoint_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Node not started"))?;

        // Parse the peer's address string into a NodeAddr.
        let peer_addr: iroh_net::NodeAddr = peer_addr_str.parse()?;

        // Establish a direct QUIC connection.
        let connection = endpoint.connect(peer_addr, b"tensor-iroh/0.1").await?;
        let (mut send_stream, _recv_stream) = connection.open_bi().await?;

        // 1. Send the header.
        let header_bytes = postcard::to_stdvec(&Message::Header(metadata))?;
        send_stream.write_all(&(header_bytes.len() as u32).to_be_bytes()).await?;
        send_stream.write_all(&header_bytes).await?;

        // 2. Send the data in chunks.
        for chunk in data.chunks(16 * 1024) { // 16KB chunks
            let chunk_bytes = postcard::to_stdvec(&Message::DataChunk(chunk.to_vec()))?;
            send_stream.write_all(&(chunk_bytes.len() as u32).to_be_bytes()).await?;
            send_stream.write_all(&chunk_bytes).await?;
        }

        // 3. Send the end message.
        let end_bytes = postcard::to_stdvec(&Message::End)?;
        send_stream.write_all(&(end_bytes.len() as u32).to_be_bytes()).await?;
        send_stream.write_all(&end_bytes).await?;

        // Gracefully close the sending side of the stream.
        send_stream.finish().await?;

        Ok(())
    }
}


// 5. --- Stream Handling Logic ---
// This helper function runs in its own task for each new connection.
async fn handle_incoming_stream(
    connecting: iroh_net::Connecting,
    callback: Arc<dyn TensorCallback>,
) -> Result<()> {
    let connection = connecting.await?;
    let (_send_stream, mut recv_stream) = connection.accept_bi().await?;

    let mut tensor_data: Vec<u8> = Vec::new();
    let mut tensor_metadata: Option<TensorMetadata> = None;

    loop {
        // Read the 4-byte length prefix.
        let mut len_buf = [0u8; 4];
        recv_stream.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        // Read the message payload.
        let mut msg_buf = vec![0u8; len];
        recv_stream.read_exact(&mut msg_buf).await?;
        let message: Message = postcard::from_bytes(&msg_buf)?;

        match message {
            Message::Header(metadata) => {
                tensor_metadata = Some(metadata);
            }
            Message::DataChunk(chunk) => {
                tensor_data.extend_from_slice(&chunk);
            }
            Message::End => {
                if let Some(metadata) = tensor_metadata {
                    // We have the complete tensor, so we trigger the callback.
                    callback.on_tensor(metadata, tensor_data);
                }
                break; // End of stream.
            }
        }
    }

    Ok(())
} 