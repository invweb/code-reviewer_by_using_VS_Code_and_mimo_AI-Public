use candle_core::Tensor;
use candle_transformers::models::quantized_llama as llama;

pub trait QuantizedModel {
    fn forward(&mut self, tokens: &[u32], start_pos: usize) -> candle_core::Result<Tensor>;
}

impl QuantizedModel for llama::ModelWeights {
    fn forward(&mut self, tokens: &[u32], start_pos: usize) -> candle_core::Result<Tensor> {
        let xs = Tensor::new(tokens, &candle_core::Device::Cpu)?.unsqueeze(0)?;
        self.forward(&xs, start_pos)
    }
}
