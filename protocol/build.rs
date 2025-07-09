fn main() {
    uniffi::generate_scaffolding("./src/tensor_protocol.udl")
        .expect("failed to generate uniffi scaffolding");
}