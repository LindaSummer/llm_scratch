# ROCm DevContainer for LLM Development

This devcontainer is configured for LLM development on AMD GPUs using ROCm.

## Features

- **Base Image**: `rocm/pytorch:latest` with ROCm support
- **Python**: 3.11 with conda environment
- **GPU Access**: Configured for AMD GPUs with proper device mounting
- **Extensions**: Python, Pylance, Jupyter, and development tools
- **Pre-installed packages**: tiktoken, matplotlib, tqdm, and other ML libraries

## Requirements

### Host System
- AMD GPU with ROCm drivers installed
- Docker with GPU support
- VS Code with Remote-Containers extension

### AMD Cloud Setup
Make sure your AMD cloud instance has:
```bash
# Install ROCm drivers (if not already installed)
sudo apt update
sudo apt install rocm-dev rocm-libs

# Verify GPU is detected
rocm-smi
```

## Usage

1. Open this folder in VS Code
2. Press `Ctrl+Shift+P` and select "Dev Containers: Reopen in Container"
3. Wait for the container to build and start
4. Verify PyTorch can see your AMD GPU:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Environment Variables

- `HSA_OVERRIDE_GFX_VERSION`: Set to match your GPU architecture
- `PYTORCH_ROCM_ARCH`: Architecture for PyTorch compilation
- `HIP_VISIBLE_DEVICES`: Controls which GPU devices are visible

## Ports

- **8888**: Jupyter Lab/Notebook
- **6006**: TensorBoard

## Troubleshooting

### GPU Not Detected
- Ensure ROCm drivers are properly installed on host
- Check that `/dev/kfd` and `/dev/dri` devices exist
- Verify user is in `video` group

### Memory Issues
- Adjust `--shm-size` in runArgs if needed
- Monitor GPU memory with `rocm-smi`

### PyTorch Issues
- Update `HSA_OVERRIDE_GFX_VERSION` to match your GPU
- Check ROCm compatibility with your GPU model