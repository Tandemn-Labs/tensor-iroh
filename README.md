# Tensor-Iroh

Goal is to create a light-weight Iroh Python Bindings acting as a general purpose peer to peer protocol for tensors, while pushing for zero-copy regimes (hence trying to integrate Cap'N'Proto)

Currently the script has experimental/monkey-patched code that works for Tandemn's use-case.

┌─────────────────┐    HTTP     ┌─────────────────┐
│  Central Server │◄─────────────┤   Web Client    │
│   (FastAPI)     │──────────────►│   (Browser)     │
└─────────────────┘              └─────────────────┘
          │
          │ Iroh Gossip + HTTP
          │
    ┌─────▼─────┐
    │  Iroh P2P │
    │  Network  │
    └─────┬─────┘
          │
   ┌──────┼──────┐
   │      │      │
┌──▼──┐ ┌─▼──┐ ┌─▼──┐
│Peer1│ │Peer2│ │Peer3│
│Embed│ │Trans│ │Output│
└─────┘ └────┘ └─────┘

**System Design Summary**

The system is designed as a **centralized orchestrator with decentralized, peer-to-peer (P2P) data transfer**.

- **central_server_demo.py (The Orchestrator):** This acts as the brain of the operation. It doesn't perform any heavy computation itself. Its main jobs are:
    1. **Peer Registry:** To keep track of all available worker nodes (peers) via a heartbeat mechanism.
    2. **Job Initiation:** To accept inference requests from a user via a standard HTTP REST API.
    3. **Instruction Broadcasting:** To kick off a job by broadcasting an initial instruction to all peers using **Iroh Gossip**.
    4. **Result Collection:** To receive the final result of the computation from the last peer in the pipeline, also via an HTTP endpoint.
- **peer_node_demo.py (The Worker):** This is the workhorse. Multiple instances of this script run simultaneously, each forming a node in the distributed network. Each peer is responsible for:
    1. **Announcing Itself:** Registering with the central server on startup to become available for work.
    2. **Peer Discovery:** Learning about other peers from the server to enable direct P2P communication.
    3. **Processing a Task:** Simulating the work of a single model layer (e.g., embedding, transformer, or output).
    1. **P2P Data Transfer:** Passing its processed data (the "hidden state") directly to the next peer in the pipeline using **Iroh Blobs and Gossip**, without going through the central server.
    1. **Final Reporting:** If it's the last peer, it sends the final result directly back to the server's HTTP endpoint.
    4. 1. **Processing a Task:** Simulating the work of a single model layer (e.g., embedding, transformer, or output).
    5. 1. **P2P Data Transfer:** Passing its processed data (the "hidden state") directly to the next peer in the pipeline using **Iroh Blobs and Gossip**, without going through the central server.
    6. 1. **Final Reporting:** If it's the last peer, it sends the final result directly back to the server's HTTP endpoint.

**Protocols Used**
- HTTP/REST API: Used for client-to-server communication.
- Iroh Gossip: A P2P pub/sub protocol. It's used for broadcasting small, low-latency control messages, such as the initial job trigger and references to large data blobs.
- Iroh Blobs & Tickets: The mechanism for large data transfer. Instead of sending large tensors through the gossip network (which is inefficient), a peer uploads the data as a "blob" to its local Iroh node. It then creates a "ticket"—a small, shareable string containing the blob's hash and the peer's network address. 

**Network Flow Maps**

Phase 1: Startup and Peer Discovery
Server Starts: central_server_demo.py launches. It starts its Iroh node and an HTTP server. Peers Start: Multiple peer_node_demo.py instances are started. Heartbeat & Registration: Each peer sends an HTTP POST to the server's /heartbeat endpoint. This message contains its unique Iroh node_id and network addresses. Peer Discovery: The server receives the heartbeat, adds the peer to its internal peer_table, and crucially, sends back a list of all other known peers. Mesh Formation: The peer receives this list and adds the server and all other peers to its Iroh routing table. Now, every node in the network knows how to contact every other node directly for P2P communication.

Phase 2: Inference Request and Job Trigger
User Request: A user sends an HTTP POST to the server's /infer endpoint to start a job. The server generates a unique request_id and creates an instruction payload. This payload defines the pipeline (e.g., peer_1 -> peer_2 -> peer_3). The server broadcasts this instruction on the TRIGGER_TOPIC gossip channel. This single message is efficiently sent out to all connected peers.

Phase 3: Peer-to-Peer Pipeline Execution
The first peer in the pipeline (peer_1) receives the trigger message. It processes the initial data (simulating an embedding layer). Peer_1 converts its output tensor into bytes and stores it as an Iroh Blob.  It creates a Blob Ticket and broadcasts a new gossip message on the hidden_state_topic. This message contains the request_id and the ticket. The next peer in the pipeline (peer_2) receives the message with the ticket. Peer_2 uses the ticket to establish a direct P2P connection to peer_1 and downloads the hidden state blob. The central server is not involved in this transfer. Peer_2 processes the data, creates a new blob and ticket, and forwards it to peer_3 using the same gossip-and-blob mechanism.The last peer (peer_3) processes the data it received from peer_2.

Phase 4: Completion
HTTP Result to Server: Since it's the last in the chain, it does not use gossip. Instead, it sends the final result back to the central server via an HTTP POST to the /completion endpoint. This avoids broadcasting the final result to all peers. 

Job Marked as Complete: The server receives the completion data and updates the job's status. The user can now fetch the result by polling the /status/{request_id} endpoint.

**Monkeypatches Applied to make it work** (Have to confirm all of this)
Well, this protocol is not perfect. There might be things in Iroh (rust) that need to be modified and brought to FFI Python Bindings to help us. But here are some problems - 

1 - The Problem: "Simulated" Direct Messaging via Broadcast: 
The Monkey-Patch (Python): Broadcasting a message to a gossip topic and having all but one peer ignore it based on a field in the payload.
The "Better Way" (Rust): Direct Peer-to-Peer Connections & One-Shot Messages (have to confirm this)

How it Works in Rust:
Instead of using the gossip pub/sub system for the hidden state transfer, a peer would do the following:
node.connect(target_node_id): Use this function to create a direct, stream-based QUIC connection to the next peer in the pipeline. This is a dedicated, one-to-one communication channel. Send Data: Once the connection is established, you can send the raw hidden state tensor (or any data) directly over this stream. This is far more efficient than the gossip-and-blob-ticket dance for frequent, targeted messages. Close Connection: The connection can be closed after the transfer. This is essentially a lightweight, secure RPC mechanism built into the networking layer. It avoids network-wide chatter entirely. The reason it isn't in the FFI is likely because managing connection state and async streams across the FFI boundary is complex.

2 - Instead of /HTTP, use a DHT.


**What we ultimately want to achieve** (have to confirm this as well)
Currently we know that gossip is good for sending small data points in the grid, and blob is battle tested for big data points. The disadvantage of the first is that everyone in the entire grid needs to get a message - have to make it targeted pub/sub style. The disadvantage of second is that blob creates a hash that has to be sent from one node to another via gossip, which is hard for developer to do it himself. 
Can they be combined? 
Ultimate aim - have a .send(NODE_ID,DataType/TensorSize etc) and have a .recieve(NODE_ID,DataType/TensorSize etc). The data-type is handled in the back. 
