#[tokio::main]
async fn main() {
    if let Err(err) = candle_vllm_server::run().await {
        eprintln!("candle-vllm error: {err:?}");
    }
}
