use dfdx::prelude::{Module as DfdxModule, *};
use rand::{rngs::StdRng, SeedableRng};

use luminal::{module::Module, prelude::*};
use luminal_nn::{Linear, ReLU};

use crate::{binary_test, unary_test, CudaCompiler};
luminal::test_imports!();

// Test helper for quantized operations
fn quantize_f32_to_int8(data: &[f32], scale: f32) -> Vec<i8> {
    data.iter()
        .map(|&x| {
            let quantized = (x / scale).round() as i32;
            quantized.clamp(-127, 127) as i8
        })
        .collect()
}

fn dequantize_int8_to_f32(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&x| x as f32 * scale).collect()
}

// Test quantized linear layer
#[test]
fn test_quantized_linear() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    
    const IN_FEATURES: usize = 256;
    const OUT_FEATURES: usize = 128;
    const BATCH_SIZE: usize = 2;
    
    // Generate random input and weights
    let input_data = random_vec_rng(IN_FEATURES * BATCH_SIZE, &mut rng);
    let weight_data = random_vec_rng(IN_FEATURES * OUT_FEATURES, &mut rng);
    
    // Quantize weights to int8
    let weight_scale = weight_data.iter().fold(0.0, |max, &x| max.max(x.abs())) / 127.0;
    let quantized_weights = quantize_f32_to_int8(&weight_data, weight_scale);
    
    // Create regular linear layer for comparison
    let regular_linear = Linear::new(IN_FEATURES, OUT_FEATURES, false, &mut cx);
    regular_linear.weight.set(weight_data.clone());
    
    let input = cx.tensor((BATCH_SIZE, IN_FEATURES)).set(input_data.clone());
    let mut regular_output = regular_linear.forward(input).retrieve();
    
    // Create quantized linear layer
    // Note: In practice, you'd need to implement a quantized linear layer that uses your kernel
    // This is a simplified test structure
    let quantized_linear = Linear::new(IN_FEATURES, OUT_FEATURES, false, &mut cx);
    let dequantized_weights = dequantize_int8_to_f32(&quantized_weights, weight_scale);
    quantized_linear.weight.set(dequantized_weights);
    
    let input_quantized = cx.tensor((BATCH_SIZE, IN_FEATURES)).set(input_data);
    let mut quantized_output = quantized_linear.forward(input_quantized).retrieve();
    
    cx.compile(CudaCompiler::<f32>::default(), (&mut regular_output, &mut quantized_output));
    cx.execute();
    
    // Allow some tolerance for quantization error
    assert_close_precision(&regular_output.data(), &quantized_output.data(), 1e-2);
}

// Test quantized matmul with different sizes
#[test]
fn test_quantized_matmul_small() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    
    let a_data = random_vec_rng(32 * 64, &mut rng);
    let b_data = random_vec_rng(64 * 16, &mut rng);
    
    // Quantize matrix B
    let b_scale = b_data.iter().fold(0.0, |max, &x| max.max(x.abs())) / 127.0;
    let quantized_b = quantize_f32_to_int8(&b_data, b_scale);
    let dequantized_b = dequantize_int8_to_f32(&quantized_b, b_scale);
    
    let a = cx.tensor((32, 64)).set(a_data.clone());
    let b_regular = cx.tensor((64, 16)).set(b_data.clone());
    let b_quantized = cx.tensor((64, 16)).set(dequantized_b);
    
    let mut regular_result = a.matmul(b_regular).retrieve();
    let mut quantized_result = a.matmul(b_quantized).retrieve();
    
    cx.compile(CudaCompiler::<f32>::default(), (&mut regular_result, &mut quantized_result));
    cx.execute();
    
    assert_close_precision(&regular_result.data(), &quantized_result.data(), 1e-2);
}

#[test]
fn test_quantized_matmul_large() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    
    let a_data = random_vec_rng(512 * 256, &mut rng);
    let b_data = random_vec_rng(256 * 512, &mut rng);
    
    // Quantize matrix B
    let b_scale = b_data.iter().fold(0.0, |max, &x| max.max(x.abs())) / 127.0;
    let quantized_b = quantize_f32_to_int8(&b_data, b_scale);
    let dequantized_b = dequantize_int8_to_f32(&quantized_b, b_scale);
    
    let a = cx.tensor((512, 256)).set(a_data.clone());
    let b_regular = cx.tensor((256, 512)).set(b_data.clone());
    let b_quantized = cx.tensor((256, 512)).set(dequantized_b);
    
    let mut regular_result = a.matmul(b_regular).retrieve();
    let mut quantized_result = a.matmul(b_quantized).retrieve();
    
    cx.compile(CudaCompiler::<f32>::default(), (&mut regular_result, &mut quantized_result));
    cx.execute();
    
    assert_close_precision(&regular_result.data(), &quantized_result.data(), 1e-2);
}

// Test quantized operations with activation functions
#[test]
fn test_quantized_relu_linear() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    
    const IN_DIM: usize = 128;
    const HIDDEN_DIM: usize = 256;
    const OUT_DIM: usize = 64;
    const BATCH_SIZE: usize = 4;
    
    let input_data = random_vec_rng(BATCH_SIZE * IN_DIM, &mut rng);
    let w1_data = random_vec_rng(IN_DIM * HIDDEN_DIM, &mut rng);
    let w2_data = random_vec_rng(HIDDEN_DIM * OUT_DIM, &mut rng);
    
    // Quantize second layer weights
    let w2_scale = w2_data.iter().fold(0.0, |max, &x| max.max(x.abs())) / 127.0;
    let quantized_w2 = quantize_f32_to_int8(&w2_data, w2_scale);
    let dequantized_w2 = dequantize_int8_to_f32(&quantized_w2, w2_scale);
    
    let model_regular = (
        Linear::new(IN_DIM, HIDDEN_DIM, false, &mut cx),
        ReLU,
        Linear::new(HIDDEN_DIM, OUT_DIM, false, &mut cx),
    );
    model_regular.0.weight.set(w1_data.clone());
    model_regular.2.weight.set(w2_data.clone());
    
    let model_quantized = (
        Linear::new(IN_DIM, HIDDEN_DIM, false, &mut cx),
        ReLU,
        Linear::new(HIDDEN_DIM, OUT_DIM, false, &mut cx),
    );
    model_quantized.0.weight.set(w1_data);
    model_quantized.2.weight.set(dequantized_w2);
    
    let input = cx.tensor((BATCH_SIZE, IN_DIM)).set(input_data);
    let mut regular_output = model_regular.forward(input).retrieve();
    
    let input_quantized = cx.tensor((BATCH_SIZE, IN_DIM)).set(input_data);
    let mut quantized_output = model_quantized.forward(input_quantized).retrieve();
    
    cx.compile(CudaCompiler::<f32>::default(), (&mut regular_output, &mut quantized_output));
    cx.execute();
    
    assert_close_precision(&regular_output.data(), &quantized_output.data(), 1e-2);
}

// Test quantized batch operations
#[test]
fn test_quantized_batch_matmul() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    
    const BATCH_SIZE: usize = 8;
    const M: usize = 64;
    const N: usize = 32;
    const K: usize = 128;
    
    let a_data = random_vec_rng(BATCH_SIZE * M * K, &mut rng);
    let b_data = random_vec_rng(BATCH_SIZE * K * N, &mut rng);
    
    // Quantize matrix B
    let b_scale = b_data.iter().fold(0.0, |max, &x| max.max(x.abs())) / 127.0;
    let quantized_b = quantize_f32_to_int8(&b_data, b_scale);
    let dequantized_b = dequantize_int8_to_f32(&quantized_b, b_scale);
    
    let a = cx.tensor((BATCH_SIZE, M, K)).set(a_data.clone());
    let b_regular = cx.tensor((BATCH_SIZE, K, N)).set(b_data.clone());
    let b_quantized = cx.tensor((BATCH_SIZE, K, N)).set(dequantized_b);
    
    let mut regular_result = a.matmul(b_regular).retrieve();
    let mut quantized_result = a.matmul(b_quantized).retrieve();
    
    cx.compile(CudaCompiler::<f32>::default(), (&mut regular_result, &mut quantized_result));
    cx.execute();
    
    assert_close_precision(&regular_result.data(), &quantized_result.data(), 1e-2);
}

// Test quantized operations with different quantization scales
#[test]
fn test_quantized_different_scales() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    
    const SIZE: usize = 128;
    
    let data = random_vec_rng(SIZE * SIZE, &mut rng);
    
    // Test different quantization scales
    let scales = [0.01, 0.1, 1.0, 10.0];
    
    for &scale in &scales {
        let quantized = quantize_f32_to_int8(&data, scale);
        let dequantized = dequantize_int8_to_f32(&quantized, scale);
        
        let original = cx.tensor((SIZE, SIZE)).set(data.clone());
        let quantized_tensor = cx.tensor((SIZE, SIZE)).set(dequantized);
        
        // Test that quantized tensor can still be used in operations
        let mut result = original.matmul(quantized_tensor).sum(1).retrieve();
        
        cx.compile(CudaCompiler::<f32>::default(), &mut result);
        cx.execute();
        
        // Result should be finite and reasonable
        assert!(result.data().iter().all(|&x| x.is_finite()));
    }
}

// Test quantized operations with mathematical functions
#[test]
fn test_quantized_with_math_ops() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    
    const SIZE: usize = 256;
    
    let a_data = random_vec_rng(SIZE, &mut rng);
    let b_data = random_vec_rng(SIZE, &mut rng);
    
    // Quantize b
    let b_scale = b_data.iter().fold(0.0, |max, &x| max.max(x.abs())) / 127.0;
    let quantized_b = quantize_f32_to_int8(&b_data, b_scale);
    let dequantized_b = dequantize_int8_to_f32(&quantized_b, b_scale);
    
    let a = cx.tensor(SIZE).set(a_data.clone());
    let b_regular = cx.tensor(SIZE).set(b_data.clone());
    let b_quantized = cx.tensor(SIZE).set(dequantized_b);
    
    // Test various mathematical operations with quantized input
    let mut add_regular = (a.clone() + b_regular.clone()).sin().retrieve();
    let mut add_quantized = (a.clone() + b_quantized.clone()).sin().retrieve();
    
    let mut mul_regular = (a.clone() * b_regular.clone()).sqrt().retrieve();
    let mut mul_quantized = (a.clone() * b_quantized.clone()).sqrt().retrieve();
    
    cx.compile(
        CudaCompiler::<f32>::default(),
        (&mut add_regular, &mut add_quantized, &mut mul_regular, &mut mul_quantized),
    );
    cx.execute();
    
    assert_close_precision(&add_regular.data(), &add_quantized.data(), 1e-2);
    assert_close_precision(&mul_regular.data(), &mul_quantized.data(), 1e-2);
}

// Test quantized operations in a more complex computational graph
#[test]
fn test_quantized_complex_graph() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    
    const BATCH_SIZE: usize = 2;
    const IN_DIM: usize = 64;
    const HIDDEN_DIM: usize = 128;
    const OUT_DIM: usize = 32;
    
    let input_data = random_vec_rng(BATCH_SIZE * IN_DIM, &mut rng);
    let weight1_data = random_vec_rng(IN_DIM * HIDDEN_DIM, &mut rng);
    let weight2_data = random_vec_rng(HIDDEN_DIM * OUT_DIM, &mut rng);
    
    // Quantize both weight matrices with different scales
    let w1_scale = weight1_data.iter().fold(0.0, |max, &x| max.max(x.abs())) / 127.0;
    let w2_scale = weight2_data.iter().fold(0.0, |max, &x| max.max(x.abs())) / 127.0;
    
    let quantized_w1 = quantize_f32_to_int8(&weight1_data, w1_scale);
    let quantized_w2 = quantize_f32_to_int8(&weight2_data, w2_scale);
    
    let dequantized_w1 = dequantize_int8_to_f32(&quantized_w1, w1_scale);
    let dequantized_w2 = dequantize_int8_to_f32(&quantized_w2, w2_scale);
    
    // Regular model
    let input_regular = cx.tensor((BATCH_SIZE, IN_DIM)).set(input_data.clone());
    let w1_regular = cx.tensor((IN_DIM, HIDDEN_DIM)).set(weight1_data.clone());
    let w2_regular = cx.tensor((HIDDEN_DIM, OUT_DIM)).set(weight2_data.clone());
    
    let mut regular_output = input_regular
        .matmul(w1_regular)
        .relu()
        .matmul(w2_regular)
        .softmax(1)
        .retrieve();
    
    // Quantized model
    let input_quantized = cx.tensor((BATCH_SIZE, IN_DIM)).set(input_data);
    let w1_quantized = cx.tensor((IN_DIM, HIDDEN_DIM)).set(dequantized_w1);
    let w2_quantized = cx.tensor((HIDDEN_DIM, OUT_DIM)).set(dequantized_w2);
    
    let mut quantized_output = input_quantized
        .matmul(w1_quantized)
        .relu()
        .matmul(w2_quantized)
        .softmax(1)
        .retrieve();
    
    cx.compile(CudaCompiler::<f32>::default(), (&mut regular_output, &mut quantized_output));
    cx.execute();
    
    // For softmax output, we need slightly higher tolerance
    assert_close_precision(&regular_output.data(), &quantized_output.data(), 1e-1);
}

// Test edge cases for quantization
#[test]
fn test_quantization_edge_cases() {
    let mut cx = Graph::new();
    
    // Test with very small values
    let small_data = vec![0.001, 0.002, -0.001, 0.0005];
    let small_scale = 0.001 / 127.0;
    let quantized_small = quantize_f32_to_int8(&small_data, small_scale);
    let dequantized_small = dequantize_int8_to_f32(&quantized_small, small_scale);
    
    let small_tensor = cx.tensor(4).set(small_data.clone());
    let dequantized_tensor = cx.tensor(4).set(dequantized_small);
    
    let mut small_result = small_tensor.sin().retrieve();
    let mut dequantized_result = dequantized_tensor.sin().retrieve();
    
    cx.compile(CudaCompiler::<f32>::default(), (&mut small_result, &mut dequantized_result));
    cx.execute();
    
    assert_close_precision(&small_result.data(), &dequantized_result.data(), 1e-3);
    
    // Test with large values
    let large_data = vec![1000.0, 2000.0, -1000.0, 500.0];
    let large_scale = 2000.0 / 127.0;
    let quantized_large = quantize_f32_to_int8(&large_data, large_scale);
    let dequantized_large = dequantize_int8_to_f32(&quantized_large, large_scale);
    
    let large_tensor = cx.tensor(4).set(large_data.clone());
    let dequantized_large_tensor = cx.tensor(4).set(dequantized_large);
    
    let mut large_result = large_tensor.sqrt().retrieve();
    let mut dequantized_large_result = dequantized_large_tensor.sqrt().retrieve();
    
    cx.compile(CudaCompiler::<f32>::default(), (&mut large_result, &mut dequantized_large_result));
    cx.execute();
    
    assert_close_precision(&large_result.data(), &dequantized_large_result.data(), 1e-1);
}

// Test quantized operations with reduction ops
#[test]
fn test_quantized_reductions() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    
    const ROWS: usize = 64;
    const COLS: usize = 128;
    
    let data = random_vec_rng(ROWS * COLS, &mut rng);
    
    // Quantize the data
    let scale = data.iter().fold(0.0, |max, &x| max.max(x.abs())) / 127.0;
    let quantized = quantize_f32_to_int8(&data, scale);
    let dequantized = dequantize_int8_to_f32(&quantized, scale);
    
    let original = cx.tensor((ROWS, COLS)).set(data.clone());
    let quantized_tensor = cx.tensor((ROWS, COLS)).set(dequantized);
    
    // Test various reduction operations
    let mut sum_regular = original.sum(1).retrieve();
    let mut sum_quantized = quantized_tensor.sum(1).retrieve();
    
    let mut mean_regular = original.mean(0).retrieve();
    let mut mean_quantized = quantized_tensor.mean(0).retrieve();
    
    let mut max_regular = original.max(1).retrieve();
    let mut max_quantized = quantized_tensor.max(1).retrieve();
    
    cx.compile(
        CudaCompiler::<f32>::default(),
        (&mut sum_regular, &mut sum_quantized, &mut mean_regular, &mut mean_quantized, &mut max_regular, &mut max_quantized),
    );
    cx.execute();
    
    assert_close_precision(&sum_regular.data(), &sum_quantized.data(), 1e-2);
    assert_close_precision(&mean_regular.data(), &mean_quantized.data(), 1e-2);
    assert_close_precision(&max_regular.data(), &max_quantized.data(), 1e-2);
}