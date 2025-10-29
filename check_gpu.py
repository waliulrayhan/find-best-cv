"""
GPU Verification Script for CV Screening Model
Checks if GPU is available and provides detailed information
"""

import torch
import sys
from pathlib import Path

def check_gpu_availability():
    """Check GPU availability and print detailed information"""
    
    print("="*80)
    print("GPU AVAILABILITY CHECK")
    print("="*80)
    
    # PyTorch version
    print(f"\n📦 PyTorch Version: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n🎮 CUDA Available: {cuda_available}")
    
    if cuda_available:
        # CUDA version
        print(f"⚡ CUDA Version: {torch.version.cuda}")
        
        # GPU device count
        device_count = torch.cuda.device_count()
        print(f"🖥️  GPU Device Count: {device_count}")
        
        # GPU details for each device
        for i in range(device_count):
            print(f"\n--- GPU Device {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            
            # Memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"Total Memory: {total_memory:.2f} GB")
            
            # Current memory usage
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"Allocated Memory: {allocated:.2f} GB")
                print(f"Cached Memory: {cached:.2f} GB")
        
        # Current device
        current_device = torch.cuda.current_device()
        print(f"\n🔧 Current Device: {current_device}")
        
        # CUDNN
        cudnn_available = torch.backends.cudnn.is_available()
        print(f"🚀 cuDNN Available: {cudnn_available}")
        if cudnn_available:
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
        
        # Test GPU computation
        print("\n🧪 Testing GPU Computation...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("   ✅ GPU computation test PASSED")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ GPU computation test FAILED: {e}")
        
        print("\n" + "="*80)
        print("✅ GPU IS AVAILABLE AND WORKING!")
        print("="*80)
        print("\nYour model will use GPU for training and inference.")
        print(f"Recommended batch size for {torch.cuda.get_device_name(0)}: 8-16")
        
    else:
        print("\n" + "="*80)
        print("⚠️  WARNING: GPU NOT AVAILABLE")
        print("="*80)
        print("\nPossible reasons:")
        print("1. PyTorch not installed with CUDA support")
        print("   Solution: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("2. NVIDIA drivers not installed or outdated")
        print("   Solution: Update NVIDIA drivers from https://www.nvidia.com/Download/index.aspx")
        print("3. CUDA toolkit not installed")
        print("   Solution: Install CUDA 12.1 from NVIDIA")
        
    return cuda_available

def test_model_on_gpu():
    """Test if model can be loaded on GPU"""
    if not torch.cuda.is_available():
        print("\n⚠️  Cannot test model on GPU - CUDA not available")
        return False
    
    print("\n" + "="*80)
    print("TESTING MODEL ON GPU")
    print("="*80)
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        print("\n📥 Loading BERT model on GPU...")
        model_name = "distilbert-base-uncased"
        model = AutoModel.from_pretrained(model_name)
        model = model.to('cuda')
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("✅ Model loaded on GPU successfully!")
        
        # Test inference
        print("\n🧪 Testing inference on GPU...")
        test_text = "This is a test resume text"
        inputs = tokenizer(test_text, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✅ Inference test PASSED!")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        
        # Clean up
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model test FAILED: {e}")
        return False

if __name__ == "__main__":
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Test model if GPU is available
    if gpu_available:
        test_model_on_gpu()
    
    print("\n" + "="*80)
    print("GPU CHECK COMPLETE")
    print("="*80)
