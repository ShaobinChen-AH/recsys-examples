# test_kv_quantization.py
import torch
import sys
sys.path.insert(0, 'examples/hstu')

from configs import get_kvcache_config, get_inference_hstu_config
from modules.kv_cache_quantization import create_kv_quantizer, KVCacheQuantizer

def test_quantization():
    """测试量化/反量化功能"""
    print("=" * 60)
    print("测试 1: 量化/反量化功能")
    print("=" * 60)
    
    # 创建测试数据
    num_pages = 4
    page_size = 32
    num_heads = 8
    head_dim = 128
    
    # 创建模拟 KV cache 数据
    kv_data = torch.randn(
        num_pages, 2, page_size, num_heads, head_dim,
        dtype=torch.bfloat16, device='cuda'
    )
    
    print(f"原始数据形状: {kv_data.shape}")
    print(f"原始数据大小: {kv_data.numel() * 2} bytes")  # bfloat16 = 2 bytes
    
    # 创建 quantizer
    quantizer = KVCacheQuantizer(
        head_dim=head_dim,
        num_heads=num_heads,
        num_layers=1,
        quantization_bits=4,
        use_random_rotation=True,
        rotation_group_size=128,
        dtype=torch.bfloat16
    ).cuda()
    
    # 量化
    quantized_indices, scales, zero_points = quantizer.quantize(kv_data, layer_idx=0)
    
    print(f"量化后索引形状: {quantized_indices.shape}")
    print(f"量化后大小: {quantized_indices.numel()} bytes (4-bit packed)")
    print(f"压缩比: {(kv_data.numel() * 2) / quantized_indices.numel():.2f}x")
    
    # 反量化
    dequantized = quantizer.dequantize(quantized_indices, scales, zero_points, layer_idx=0)
    
    # 计算误差
    error = torch.abs(kv_data.float() - dequantized.float()).mean()
    max_error = torch.abs(kv_data.float() - dequantized.float()).max()
    
    print(f"平均量化误差: {error.item():.6f}")
    print(f"最大量化误差: {max_error.item():.6f}")
    
    assert error < 0.1, "量化误差过大!"
    print("✓ 量化/反量化测试通过")
    
    return True

def test_config_integration():
    """测试配置集成"""
    print("\n" + "=" * 60)
    print("测试 2: 配置集成")
    print("=" * 60)
    
    hstu_config = get_inference_hstu_config(
        hidden_size=512,
        num_layers=4,
        num_attention_heads=8,
        head_dim=64,
        max_batch_size=16,
        max_seq_len=128,
        dtype=torch.bfloat16,
    )
    
    kv_config = get_kvcache_config(
        blocks_in_primary_pool=128,
        page_size=32,
        offload_chunksize=128,
        enable_kv_quantization=True,
        kv_quantization_bits=4,
        use_random_rotation=True,
        rotation_group_size=128,
    )
    
    quantizer = create_kv_quantizer(hstu_config, kv_config)
    
    if quantizer is not None:
        print(f"✓ Quantizer 创建成功")
        print(f"  - 量化比特数: {quantizer.quantization_bits}")
        print(f"  - 使用随机旋转: {quantizer.use_random_rotation}")
        print(f"  - 压缩比: {16 // quantizer.quantization_bits}x")
    else:
        print("✗ Quantizer 创建失败")
        return False
    
    return True

def test_bandwidth_improvement():
    """测试带宽改善"""
    print("\n" + "=" * 60)
    print("测试 3: CPU-GPU 传输带宽测试")
    print("=" * 60)
    
    import time
    
    num_pages = 100
    page_size = 32
    num_heads = 8
    head_dim = 128
    
    # 创建测试数据
    kv_data = torch.randn(
        num_pages, 2, page_size, num_heads, head_dim,
        dtype=torch.bfloat16, device='cuda'
    )
    
    # 创建量化器
    quantizer = KVCacheQuantizer(
        head_dim=head_dim,
        num_heads=num_heads,
        num_layers=1,
        quantization_bits=4,
        use_random_rotation=False,  # 简化测试
        dtype=torch.bfloat16
    ).cuda()
    
    # 量化数据
    quantized_indices, scales, zero_points = quantizer.quantize(kv_data, layer_idx=0)
    
    # 测试原始数据 H2D 传输时间
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = kv_data.cpu()
        torch.cuda.synchronize()
    h2d_time_original = (time.time() - start) / 10
    
    # 测试量化数据 H2D 传输时间
    start = time.time()
    for _ in range(10):
        _ = quantized_indices.cpu()
        _ = scales.cpu()
        _ = zero_points.cpu()
        torch.cuda.synchronize()
    h2d_time_quantized = (time.time() - start) / 10
    
    original_size = kv_data.numel() * 2  # bytes
    quantized_size = (quantized_indices.numel() + 
                     scales.numel() * 4 + 
                     zero_points.numel() * 4)  # bytes
    
    print(f"原始数据大小: {original_size / 1024 / 1024:.2f} MB")
    print(f"量化后大小: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"大小减少: {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"\n原始数据 H2D 传输时间: {h2d_time_original*1000:.2f} ms")
    print(f"量化数据 H2D 传输时间: {h2d_time_quantized*1000:.2f} ms")
    print(f"传输加速比: {h2d_time_original/h2d_time_quantized:.2f}x")
    
    return True

if __name__ == "__main__":
    try:
        test_quantization()
        test_config_integration()
        test_bandwidth_improvement()
        
        print("\n" + "=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

