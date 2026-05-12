import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Simple benchmark
    print("Running matrix multiplication benchmark...")
    a = torch.randn(5000, 5000).cuda()
    b = torch.randn(5000, 5000).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = torch.matmul(a, b)
    end.record()

    torch.cuda.synchronize()
    print(f"Matrix multiplication (5000x5000) took: {start.elapsed_time(end):.2f} ms")
else:
    print("CUDA is NOT available.")
