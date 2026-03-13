import argparse
import inspect
import os

import torch

def load_model(model_path, device):
    try:
        model = torch.jit.load(model_path, map_location=device)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {model_path}") from exc

def export_rknn(model, onnx_output, rknn_output, target_platform="rk3568"):
    onnx_path = onnx_output.strip() if onnx_output else ""
    if not onnx_path:
        base, _ = os.path.splitext(rknn_output)
        onnx_path = f"{base}.onnx"

    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    export_kwargs = dict(
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Newer PyTorch versions support the `dynamo` kwarg; older ones raise TypeError.
    try:
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
    except (TypeError, ValueError):
        pass

    try:
        torch.onnx.export(model_cpu, dummy_input, onnx_path, **export_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "ONNX export failed. Install deps with: pip install onnx onnxscript"
        ) from exc

    print(f"saved_onnx: {onnx_path}")

    try:
        from rknn.api import RKNN
    except Exception as exc:
        raise RuntimeError("RKNN export requires rknn-toolkit2. Please install it first.") from exc

    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(target_platform=target_platform)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with code {ret}")

        # Some toolkit versions require explicit input_size_list.
        try:
            ret = rknn.load_onnx(model=onnx_path, input_size_list=[[1, 3, 32, 32]])
        except TypeError:
            ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed with code {ret}")

        ret = rknn.export_rknn(rknn_output)
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed with code {ret}")
    finally:
        rknn.release()

    print(f"saved_rknn: {rknn_output}")

args = argparse.ArgumentParser(description="Convert PyTorch model to RKNN format.")
args.add_argument("--pth-folder", type=str, required=True, help="Path to input .pth model file.")
args.add_argument("--onnx-folder", type=str, default="", help="Path to output .onnx file (optional). If not provided, will be derived from RKNN output path.")
args.add_argument("--rknn-folder", type=str, required=True, help="Path to output .rknn file.")
args.add_argument("--target-platform", type=str, default="rk3588", help="RKNN target platform.")

args = args.parse_args()

device = torch.device("cpu")

for filename in os.listdir(args.pth_folder):
    if filename.endswith(".pth"):
        pth_input = os.path.join(args.pth_folder, filename)
        os.makedirs(args.rknn_folder, exist_ok=True)
        os.makedirs(args.onnx_folder, exist_ok=True) if args.onnx_folder else None
        onnx_output = os.path.join(args.onnx_folder, os.path.splitext(filename)[0] + ".onnx") if args.onnx_folder else ""
        rknn_output = os.path.join(args.rknn_folder, os.path.splitext(filename)[0] + ".rknn")
        print(f"Processing {pth_input}...")
        model = load_model(pth_input, device)
        export_rknn(model, onnx_output, rknn_output, target_platform=args.target_platform)

# model = load_model(args.pth_input, device)

# export_rknn(model, args.onnx_output, args.rknn_output, target_platform=args.target_platform)
# python pth2onnx_rknn.py --pth-folder models/rasp/hwnasbench/ --onnx-folder models/jetson/hwnasbench --rknn-folder models/orange/hwnasbench
# python pth2onnx_rknn.py --pth-folder models/rasp/nasbench101/ --onnx-folder models/jetson/nasbench101 --rknn-folder models/orange/nasbench101
# python pth2onnx_rknn.py --pth-folder models/rasp/nasbench201/ --onnx-folder models/jetson/nasbench201 --rknn-folder models/orange/nasbench201