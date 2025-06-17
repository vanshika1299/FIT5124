import torch
from a3_mnist_fixed import My_MNIST

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test loading the saved model
try:
    model = My_MNIST()
    model.load_state_dict(torch.load('mnist_cnn.pt'))
    model.eval()
    print("✅ Successfully loaded target model!")

    # Test with dummy input
    dummy_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output = model(dummy_input)
        prediction = torch.argmax(output, dim=1)
    print(f"✅ Model works! Dummy prediction: {prediction.item()}")

except Exception as e:
    print(f"❌ Error: {e}")