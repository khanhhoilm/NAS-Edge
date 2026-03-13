# EnMeEdge: Inference + Energy Measurement

This folder contains tools to benchmark model inference and estimate energy on edge devices.

- `server.py`: measurement server on host machine with FNB USB power meter.
- `inference_on_edge.py`: inference client on edge device (`.pth`, `.onnx`, `.rknn`) and reports start/stop timestamps to server.

## 1. Overview

Flow:

1. Edge device sends `POST /start` to server before inference.
2. Server starts sampling power from FNB meter.
3. Edge runs inference on CIFAR-10 test samples.
4. Edge sends `POST /stop` with start/end timestamps.
5. Server computes average power and energy in the requested time window.
6. Edge stores metrics in output JSON.

## 2. Requirements

### Hardware

- 1 host machine connected to FNB device (FNB48/FNB58/FNB48S/C1 supported in code).
- 1 edge device to run inference.

### Software

Install dependencies for server and client environments as needed.

- Common: `numpy`, `requests`, `tqdm`, `psutil`, `flask`, `pyusb`.
- PyTorch models (`.pth`): `torch`.
- ONNX models (`.onnx`): `onnxruntime`.
- RKNN models (`.rknn`): `rknnlite` (preferred) or `rknn-toolkit2` runtime.

Example install (adjust to your environment):

```bash
pip install numpy requests tqdm psutil flask pyusb
```

## 3. Dataset Expectation

`inference_on_edge.py` expects CIFAR-10 python files at:

`./data/cifar-10-batches-py/test_batch`

If you run from repository root, this path already matches current project structure.

## 4. Start Measurement Server (Host)

Run on the host connected to FNB:

```bash
python scripts/EnMeEdge/server.py --host 0.0.0.0 --port 8000 --time-interval 0.01 --raw-file pc_raw.csv
```

Optional health check:

```bash
curl http://<host_ip>:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Notes:

- `server.py` performs an initialization stage to estimate idle power offset before normal measurement.
- Keep server running while edge inference jobs are executed.

## 5. Configure Edge Client

Edit constants at top of `inference_on_edge.py`:

- `SERVER_IP`: host IP running `server.py`.
- `SERVER_PORT`: default `8000`.
- `CONFIG["data_dir"]`: CIFAR-10 directory root.
- `CONFIG["warmup_steps"]`: warmup samples before measurement start timestamp.
- `CONFIG["inference_sample_limit"]`: number of samples to process (`None` for full test set).

Important:

- For `.pth` loading, this script uses `torch.jit.load(...)`. Input must be TorchScript model.
- `.rknn` path uses `rknnlite` first, then falls back to `rknn` API.

## 6. Run Inference on Edge

Basic usage:

```bash
python scripts/EnMeEdge/inference_on_edge.py \
	--folder_path <path_to_models_folder> \
	--output_json <output_results.json>
```

The script scans all model files in folder with extensions:

- `.pth`
- `.onnx`
- `.rknn`

It skips models whose hash ID (prefix before first `_` in filename) already exists in output JSON.

Example:

```bash
python scripts/EnMeEdge/inference_on_edge.py \
	--folder_path topk_nasbench_models/orange/rknn/nasbench201 \
	--output_json measure_energy_results/orange/nasbench201_36epochs_cifar10.json
```

## 7. Output Format

Each model contributes one key by hash ID:

```json
{
	"<hashid>": {
		"accuracy_percent": 88.12,
		"avg_latency_ms": 4.73,
		"RAM_usage_MB": 1234.56,
		"swap_usage_MB": 0.0,
		"peak_memory_MB": 345.67,
		"energy_result": {
			"status": "stopped",
			"result": {
				"start_time": 1710000000.1,
				"end_time": 1710000004.9,
				"duration": 4.8,
				"average_power_W": 2.35,
				"power_offset_W": 0.75,
				"energy_mWh": 2.13
			}
		}
	}
}
```

## 8. Troubleshooting

- `tracker not ready` on `/start`:
	Server is still initializing or busy. Wait and retry.

- `no active run` on `/stop`:
	`/start` was not accepted or run already stopped.

- `Missing CIFAR-10 file ... test_batch`:
	Fix `CONFIG["data_dir"]` or run command from repository root.

- `monitor server health check failed`:
	Verify server is running and edge can reach `<SERVER_IP>:<SERVER_PORT>`.

- `load_rknn/init_runtime failed`:
	Ensure RKNN runtime version matches target device and model compatibility.

- Permission errors with USB device on host:
	Check udev/device permissions for FNB access.

## 9. Practical Tips

- Keep `warmup_steps` > 0 to avoid cold-start bias.
- Run one model per thermal state condition for fair comparison.
- Preserve raw logs (`--raw-file`) if you need to audit power traces later.
