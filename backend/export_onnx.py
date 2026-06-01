"""
export_onnx.py — Convert trained CNN+TCN PyTorch model to ONNX format for Vercel deployment.
Run: python export_onnx.py
Output: models/cnn_tcn_model.onnx
"""
import os, sys, torch
import numpy as np

# Add backend dir to path
sys.path.insert(0, os.path.dirname(__file__))
from cnn_tcn_model import CNNTCNModel

MODEL_IN  = os.path.join(os.path.dirname(__file__), 'models', 'cnn_tcn_model.pth')
MODEL_OUT = os.path.join(os.path.dirname(__file__), 'models', 'cnn_tcn_model.onnx')

def main():
    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(MODEL_IN, map_location='cpu', weights_only=False)
    n_channels    = int(ckpt.get('n_channels',    19))
    epoch_samples = int(ckpt.get('epoch_samples', 256))
    print(f"Loaded checkpoint: n_channels={n_channels}, epoch_samples={epoch_samples}")
    print(f"  val_acc={ckpt.get('val_acc', '?')}  val_auc={ckpt.get('val_auc', '?')}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = CNNTCNModel(n_channels=n_channels, n_timepoints=epoch_samples)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ── Dummy input: (batch=1, channels=19, timesteps=256) ───────────────────
    dummy_input = torch.randn(1, n_channels, epoch_samples)

    # Verify forward pass works
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Forward pass OK — output shape: {out.shape}")

    # ── Export to ONNX ───────────────────────────────────────────────────────
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_OUT,
        opset_version=17,
        input_names=['eeg_epoch'],          # (batch, channels, timesteps)
        output_names=['logits'],            # (batch, 2)
        dynamic_axes={
            'eeg_epoch': {0: 'batch_size'},
            'logits':    {0: 'batch_size'},
        },
        do_constant_folding=True,
    )

    size_kb = os.path.getsize(MODEL_OUT) / 1024
    print(f"\nExported: {MODEL_OUT}  ({size_kb:.1f} KB)")

    # ── Quick validation with onnxruntime ────────────────────────────────────
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(MODEL_OUT, providers=['CPUExecutionProvider'])
        result = sess.run(['logits'], {'eeg_epoch': dummy_input.numpy()})
        import torch.nn.functional as F
        probs = F.softmax(torch.tensor(result[0]), dim=1).numpy()
        print(f"ONNX validation OK — probs: {probs}")
        print("ONNX export successful!")
    except ImportError:
        print("onnxruntime not installed, skipping validation. Install: pip install onnxruntime")

if __name__ == '__main__':
    main()
