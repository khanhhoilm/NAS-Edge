import argparse
import os
import pickle
import time
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from tqdm import tqdm
import requests
import json

SERVER_IP = "localhost"
SERVER_PORT = "8000"
CONFIG = {
    "server_url": f"http://{SERVER_IP}:{SERVER_PORT}",
    "data_dir": "./data",
    "warmup_steps": 50,
    "inference_sample_limit": 1000,  # set to None for no limit (process entire test set)
}


class Compose:
    def __init__(self, transforms_list):
        self.transforms_list = list(transforms_list)

    def __call__(self, x):
        for t in self.transforms_list:
            x = t(x)
        return x

class ToTensor:
    """Convert HWC uint8/float numpy array to CHW float32 torch tensor.

    Mimics torchvision.transforms.ToTensor() for numpy input.
    """

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            t = x
        else:
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            if x.ndim != 3 or x.shape[2] != 3:
                raise ValueError(f"ToTensor expects HWC with 3 channels, got shape {x.shape}")
            t = torch.from_numpy(x)

        if t.ndim != 3:
            raise ValueError(f"ToTensor expects 3D input, got {tuple(t.shape)}")

        # Ensure CHW
        if t.shape[0] == 3 and t.shape[-1] != 3:
            chw = t
        else:
            chw = t.permute(2, 0, 1)

        chw = chw.contiguous()

        if chw.dtype == torch.uint8:
            return chw.to(dtype=torch.float32).div(255.0)
        return chw.to(dtype=torch.float32)

class Normalize:
    """Normalize tensor image with mean/std.

    Expects CHW float tensor.
    """

    def __init__(self, mean, std):
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("Normalize expects mean/std of length 3")
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Normalize expects torch.Tensor, got {type(x)}")
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"Normalize expects CHW with 3 channels, got shape {tuple(x.shape)}")
        return (x - self.mean.to(device=x.device)) / self.std.to(device=x.device)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model (.onnx or .pth) on CIFAR-10")
    parser.add_argument("--folder_path", type=str, default="./models", help="Path to folder containing .onnx or .pth files")
    parser.add_argument("--output_json", type=str, default="results.json", help="Path to output JSON file for results")
    # parser.add_argument("--model_file", type=str, default="sample_cifar10_cnn.pth", help="Path to .onnx or .pth file")
    # parser.add_argument("--server_url", type=str, default="http://192.168.50.186:8000", help="http://<pc_ip>:8000 base URL of measurement server")
    # parser.add_argument("--warmup_steps", type=int, default=200, help="Number of samples for warm-up (no measurement)")
    # parser.add_argument("--data_dir", type=str, default="./data", help="CIFAR-10 data directory (expects cifar-10-batches-py/test_batch)")
    return parser.parse_args()


class CIFAR10TestLocal(Dataset):
    """Minimal CIFAR-10 test Dataset that reads pre-downloaded python 'test_batch'.

    Expects: <data_dir>/cifar-10-batches-py/test_batch
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        batch_path = os.path.join(data_dir, "cifar-10-batches-py", "test_batch")
        if not os.path.exists(batch_path):
            raise FileNotFoundError(
                f"Missing CIFAR-10 file: {batch_path}. "
                "Expected dataset under ./data/cifar-10-batches-py (already downloaded)."
            )

        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")

        data = batch[b"data"]  # (10000, 3072) uint8
        labels = batch[b"labels"]  # list[int]
        self.images_uint8_chw = data.reshape(-1, 3, 32, 32).astype(np.uint8, copy=False)
        self.labels_int64 = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return int(self.labels_int64.shape[0])

    def __getitem__(self, idx):
        img_chw = self.images_uint8_chw[idx]
        img_hwc = np.transpose(img_chw, (1, 2, 0))  # HWC uint8 for torchvision transforms
        target = int(self.labels_int64[idx])
        if self.transform is not None:
            img_hwc = self.transform(img_hwc)
        return img_hwc, target


def notify(action: str, payload=None, raise_flag=True):
    '''
    Helper to send start/stop signals to measurement server

    Parameters:
        action: 'start' or 'stop'
        payload: dict to send as JSON body
        raise_flag: if True, raise exception on non-200 response, otherwise just print error
    '''
    if not CONFIG.get("server_url"):
        raise ValueError("CONFIG['server_url'] is not set. Cannot send notify signal.")
    try:
        resp = requests.post(f"{CONFIG['server_url']}/{action}", json=payload or {}, timeout=5)
        if resp.ok:
            return resp.json()
        if raise_flag:
            resp.raise_for_status()
        else:
            print(f"monitor {action} failed: {resp.status_code} {resp.text}")
    except Exception as exc:
        if raise_flag:
            raise exc
        else:                
            print(f"monitor {action} exception: {exc}")    
    return None

# def main():
def process_inference(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    hashid = os.path.basename(model_path).replace(ext, '').split("_")[0]  # extract hashid from filename (assumes format <hashid>_*.pth or .rknn)

    test_transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # Use local, pre-downloaded CIFAR-10 (no download step)
    test_dataset = CIFAR10TestLocal(data_dir=CONFIG["data_dir"], transform=test_transform)
    limit = CONFIG.get("inference_sample_limit")
    if limit:
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, range(limit))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    print(f"model_hash: {hashid}")
    print(f"model_path: {model_path}")
    print(f"num_images: {len(test_dataset)}")

    try:
        process = psutil.Process(os.getpid())
        correct = 0
        total_latency_ms = 0
        num_samples = 0
        peak_memory_mb = 0
        ram_usage_mb = 0
        swap_usage_mb = 0

        if ext == ".pth":
            # from model_architecture import BasicCNN
            # PyTorch model
            print("Loading PyTorch model (.pth)...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # model = BasicCNN(num_classes=10)
            # model.load_state_dict(torch.load(model_path, map_location=device))
            model = torch.jit.load(model_path, map_location=device)
            model.to(device)
            model.eval()
            print("Model loaded.")
            print("start time:", time.time())
            energy_result = None
            infer_start_ts = None

            # check server health before starting measurement
            if CONFIG.get("server_url"):
                try:
                    health_resp = requests.get(f"{CONFIG['server_url']}/health", timeout=5)
                    if not health_resp.ok:
                        print(f"monitor server health check failed: {health_resp.status_code} {health_resp.text}")
                        return
                except Exception as exc:
                    print(f"monitor server health check exception: {exc}")
                    return
        
            # send signal to server to start monitoring (aligning with first inference)
            notify("start", {"hash_id": f"{hashid}", "timestamp": time.time()})

            for idx, (data, target) in enumerate(tqdm(test_loader)):
                # get timestamp after warmup_steps
                if idx == CONFIG["warmup_steps"]:
                    infer_start_ts = time.time()

                if CONFIG["inference_sample_limit"] is not None and idx >= CONFIG["inference_sample_limit"]:
                    print(f"Reached inference sample limit of {CONFIG['inference_sample_limit']}, stopping inference loop.")
                    break

                data = data.to(device)
                start = time.time()
                with torch.no_grad():
                    output = model(data)
                latency_ms = (time.time() - start) * 1000
                total_latency_ms += latency_ms
                num_samples += 1
                pred = output.argmax(dim=1).cpu().numpy()
                correct += (pred == target.numpy()).sum()
                curr_mem = process.memory_info().rss / (1024 * 1024)
                peak_memory_mb = max(peak_memory_mb, curr_mem)
                curr_ram_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
                ram_usage_mb = max(ram_usage_mb, curr_ram_usage_mb)
                curr_swap_usage_mb = psutil.swap_memory().used / (1024 * 1024)
                swap_usage_mb = max(swap_usage_mb, curr_swap_usage_mb)

            infer_end_ts = time.time()

            energy_result = notify("stop", { "hash_id": f"{hashid}", "start_ts": infer_start_ts, "end_ts": infer_end_ts})

            print("end time:", infer_end_ts)
            print(energy_result)
        
        elif ext == ".rknn":
            @dataclass
            class RKNNRunner:
                api: str  # 'rknnlite' or 'rknn'
                obj: object  # RKNNLite or RKNN instance
                data_format: str

                def inference(self, input_data):
                    if isinstance(input_data, (list, tuple)):
                        inputs = list(input_data)
                    else:
                        inputs = [input_data]

                    data_formats = [self.data_format] * len(inputs)
                    try:
                        return self.obj.inference(inputs=inputs, data_format=data_formats)
                    except Exception as exc:
                        raise RuntimeError(
                            "RKNN inference failed (check that each element in 'inputs' is a numpy array, not a nested list)"
                        ) from exc

                def release(self):
                    if self.obj is not None:
                        self.obj.release()


            def create_runner(model_path: str, data_format: str, target: str, core_mask: str) -> RKNNRunner:
                # Prefer rknnlite (device runtime). Fallback to rknn-toolkit2.
                try:
                    from rknnlite.api import RKNNLite

                    rknn_lite = RKNNLite()
                    ret = rknn_lite.load_rknn(model_path)
                    if ret != 0:
                        raise RuntimeError(f"load_rknn failed: {ret}")

                    init_kwargs = {}
                    if core_mask:
                        core_map = {
                            "0": RKNNLite.NPU_CORE_0,
                            "1": RKNNLite.NPU_CORE_1,
                            "2": RKNNLite.NPU_CORE_2,
                            "3": RKNNLite.NPU_CORE_0_1_2,
                        }
                        if core_mask not in core_map:
                            raise ValueError("--core_mask must be one of: 0,1,2,3")
                        init_kwargs["core_mask"] = core_map[core_mask]

                    ret = rknn_lite.init_runtime(**init_kwargs)
                    if ret != 0:
                        raise RuntimeError(f"init_runtime failed: {ret}")

                    return RKNNRunner(api="rknnlite", obj=rknn_lite, data_format=data_format)
                except Exception:
                    pass

                from rknn.api import RKNN

                rknn = RKNN(verbose=False)
                ret = rknn.load_rknn(model_path)
                if ret != 0:
                    raise RuntimeError(f"load_rknn failed: {ret}")

                init_kwargs = {}
                if target:
                    init_kwargs["target"] = target
                ret = rknn.init_runtime(**init_kwargs)
                if ret != 0:
                    raise RuntimeError(f"init_runtime failed: {ret}")

                return RKNNRunner(api="rknn", obj=rknn, data_format=data_format)
            
            # RKNN model for Orange Pi NPU
            print("Loading RKNN model (.rknn)...")
            
            runner = create_runner(
                model_path=model_path,
                data_format="nhwc",
                target="rv1109",
                core_mask="3"
            )

            print("Model loaded.")
            print("start time:", time.time())
            energy_result = None
            infer_start_ts = None

            if CONFIG["server_url"]:
                try:
                    health_resp = requests.get(f"{CONFIG['server_url']}/health", timeout=5)
                    if not health_resp.ok:
                        print(f"monitor server health check failed: {health_resp.status_code} {health_resp.text}")
                        return
                except Exception as exc:
                    print(f"monitor server health check exception: {exc}")
                    return
        
            notify("start", {"hash_id": f"{hashid}", "timestamp": time.time()})

            for idx, (data, target) in enumerate(tqdm(test_loader)):
                if idx == CONFIG["warmup_steps"]:
                    infer_start_ts = time.time()
                
                if CONFIG["inference_sample_limit"] is not None and idx >= CONFIG["inference_sample_limit"]:
                    print(f"Reached inference sample limit of {CONFIG['inference_sample_limit']}, stopping inference loop.")
                    break

                # Prepare input for RKNN (convert to numpy, remove batch dimension)
                input_data = data.numpy().astype(np.float32)
                input_data = np.transpose(input_data, (0, 2, 3, 1)) # convert CHW to HWC if needed, RKNN expects NHWC or NCHW depending on how model was exported
                
                start = time.time()
                outputs = runner.inference(input_data)
                if outputs is None:
                    raise RuntimeError("RKNN inference returned None")
                latency_ms = (time.time() - start) * 1000
                total_latency_ms += latency_ms
                num_samples += 1
                
                pred = np.argmax(outputs[0], axis=1)
                correct += (pred == target.numpy()).sum()
                curr_mem = process.memory_info().rss / (1024 * 1024)
                peak_memory_mb = max(peak_memory_mb, curr_mem)
                curr_ram_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
                ram_usage_mb = max(ram_usage_mb, curr_ram_usage_mb)
                curr_swap_usage_mb = psutil.swap_memory().used / (1024 * 1024)
                swap_usage_mb = max(swap_usage_mb, curr_swap_usage_mb)

            infer_end_ts = time.time()
            print("start time:", infer_start_ts)
            print("end time:", infer_end_ts)
            energy_result = notify("stop", {"hash_id": f"{hashid}", "start_ts": infer_start_ts, "end_ts": infer_end_ts})
            runner.release()
            print(energy_result)
        elif ext == ".onnx":
            # ONNX model (not supported for inference in this script, only for export in pth2onnx_rknn.py)
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider']
            session = ort.InferenceSession(model_path, providers=providers)
            input_name = session.get_inputs()[0].name
            print("Model loaded.")
            print("start time:", time.time())
            energy_result = None
            infer_start_ts = None

            # check server health before starting measurement
            if CONFIG.get("server_url"):
                try:
                    health_resp = requests.get(f"{CONFIG['server_url']}/health", timeout=5)
                    if not health_resp.ok:
                        print(f"monitor server health check failed: {health_resp.status_code} {health_resp.text}")
                        return
                except Exception as exc:
                    print(f"monitor server health check exception: {exc}")
                    return
        
            # send signal to server to start monitoring (aligning with first inference)
            notify("start", {"hash_id": f"{hashid}", "timestamp": time.time()})

            for idx, (data, target) in enumerate(tqdm(test_loader)):
                # get timestamp after warmup_steps
                if idx == CONFIG["warmup_steps"]:
                    infer_start_ts = time.time()

                if CONFIG["inference_sample_limit"] is not None and idx >= CONFIG["inference_sample_limit"]:
                    print(f"Reached inference sample limit of {CONFIG['inference_sample_limit']}, stopping inference loop.")
                    break

                # inference
                # data = data.to(device)
                data_np = data.numpy()
                start = time.time()
                # with torch.no_grad():
                    # output = model(data)
                output = session.run(None, {input_name: data_np})
                latency_ms = (time.time() - start) * 1000
                total_latency_ms += latency_ms
                num_samples += 1
                # print(output)
                # pred = output.argmax(dim=1).cpu().numpy()
                pred = output[0].argmax(axis=1)
                correct += (pred == target.numpy()).sum()
                curr_mem = process.memory_info().rss / (1024 * 1024)
                peak_memory_mb = max(peak_memory_mb, curr_mem)
                curr_ram_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
                ram_usage_mb = max(ram_usage_mb, curr_ram_usage_mb)
                curr_swap_usage_mb = psutil.swap_memory().used / (1024 * 1024)
                swap_usage_mb = max(swap_usage_mb, curr_swap_usage_mb)

            infer_end_ts = time.time()

            energy_result = notify("stop", { "hash_id": f"{hashid}", "start_ts": infer_start_ts, "end_ts": infer_end_ts})
            print("end time:", infer_end_ts)
            print(energy_result)
        else:
            raise ValueError("Unsupported model file extension. Only .pth is supported (no .onnx)")

        accuracy = 100.0 * correct / len(test_dataset)
        avg_latency_ms = total_latency_ms / num_samples if num_samples > 0 else 0
        energy_metrics = (energy_result or {}).get("result", {}) if isinstance(energy_result, dict) else {}
        ram_usage_output = energy_metrics.get("RAM_usage_MB", ram_usage_mb)
        swap_usage_output = energy_metrics.get("swap_usage_MB", swap_usage_mb)
        peak_memory_output = energy_metrics.get("peak_memory_MB", peak_memory_mb)
        def as_float_or_default(value, default_value=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default_value)

        return {
            hashid: {
                "accuracy_percent": accuracy,
                "avg_latency_ms": avg_latency_ms,
                "RAM_usage_MB": as_float_or_default(ram_usage_output, ram_usage_mb),
                "swap_usage_MB": as_float_or_default(swap_usage_output, swap_usage_mb),
                "peak_memory_MB": as_float_or_default(peak_memory_output, peak_memory_mb),
                "energy_result": energy_result,
            }
        }
    except Exception as e:
        raise e
if __name__ == "__main__":
    args = parse_args()
    folder_path = args.folder_path
    output_json = args.output_json
    print(f"Scanning folder: {folder_path} for .pth, .rknn, and .onnx files...")
    results = {}
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            import json
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {output_json} is not a valid JSON file. Starting with empty results.")
                results = {}
       
    for filename in os.listdir(folder_path):
        if filename.endswith(".pth") or filename.endswith(".rknn") or filename.endswith(".onnx"):
            model_path = os.path.join(folder_path, filename)
            hashid = os.path.basename(model_path).split("_")[0]
            
            if hashid in results:
                print(f"Skipping {model_path} since hashid {hashid} already in results.")
                continue
            try:
                result = process_inference(model_path)
                results.update(result)
                print(f"Completed processing {model_path}. Saving results to {output_json}...")
                with open(output_json, "w") as f:
                    json.dump(results, f, indent=4)

                print(f"Waiting 15 seconds before continuing to ensure measurement server is ready for next run...")
                time.sleep(15)
            except Exception as exc:
                print(f"Error processing {model_path}: {exc}")