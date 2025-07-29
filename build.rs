fn main() {
    uniffi::generate_scaffolding("./src/tensor_iroh.udl")
        .expect("failed to generate uniffi scaffolding");
}