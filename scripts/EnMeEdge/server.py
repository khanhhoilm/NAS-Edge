"""
Monitoring server: /start enables logging, /stop computes energy using server
timestamps (srv_start_ts to srv_end_ts). The device may send dev_start_ts/dev_end_ts
for reference, but windowing is based on server time to avoid clock skew.
"""

import argparse
import threading
import time
from flask import Flask, jsonify, request
import sys
import time
import usb.core
import usb.util
import threading

#FNB48
VID = 0x0483
PID_FNB48 = 0x003A

PID_C1 = 0x003B

# FNB58
VID_FNB58 = 0x2E3C
PID_FNB58 = 0x5558

# FNB48S
VID_FNB48S = 0x2E3C
PID_FNB48S = 0x0049


class FNBTool:
    def __init__(self,time_interval=0.01,file_path="energy.csv"):
        self._time_interval = time_interval
        self._file_path = file_path
        self._find_device()
        assert self._dev, "Device not found"
        self._interface_hid_num = self._find_hid_interface_num()
        self._ensure_all_interfaces_not_busy()

        cfg = self._dev.get_active_configuration()
        intf = cfg[(self._interface_hid_num, 0)]
        self._ep_out = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT,
        )

        self._ep_in = usb.util.find_descriptor(
            intf,
            # match the first IN endpoint
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN,
        )

        assert self._ep_in
        assert self._ep_out
        self._running=False
        self._energy_init=0
        self._power=0
        self._energy=0
        self._n_measurements=0
        self._measure_thread=None
        self._inference_name=""
        self._start_time=0
        self._time_offset=0
        self._raw_data=[]
        self.status = "ready"
        self._request_data()
        self._init()

    def _find_device(self):
        self._is_fnb58_or_fnb48s = False
        self._dev = usb.core.find(idVendor=VID, idProduct=PID_FNB48)
        if self._dev is None:
            self._dev = usb.core.find(idVendor=VID, idProduct=PID_C1)
        if self._dev is None:
            self._dev = usb.core.find(idVendor=VID_FNB58, idProduct=PID_FNB58)
            if self._dev:
                self._is_fnb58_or_fnb48s = True
        if self._dev is None:
            self._dev = usb.core.find(idVendor=VID_FNB48S, idProduct=PID_FNB48S)
            if self._dev:
                self._is_fnb58_or_fnb48s = True

    def _find_hid_interface_num(self):
        for cfg in self._dev:
            for interface in cfg:
                if interface.bInterfaceClass == 0x03:
                    return interface.bInterfaceNumber

    def _ensure_all_interfaces_not_busy(self):
        for cfg in self._dev:
            for interface in cfg:
                if self._dev.is_kernel_driver_active(interface.bInterfaceNumber):
                    try:
                        self._dev.detach_kernel_driver(interface.bInterfaceNumber)
                    except usb.core.USBError as e:
                        print(f"Could not detatch kernel driver from interface({interface.bInterfaceNumber}): {e}", file=sys.stderr)
                        sys.exit(1)

    def _request_data(self):
        self._ep_out.write(b"\xaa\x81" + b"\x00" * 61 + b"\x8e")
        time.sleep(0.1)
        self._ep_out.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
        time.sleep(0.1)
        if not self._is_fnb58_or_fnb48s:
            self._ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")
            
    def _decode(self, data):
        packet_type = data[1]
        if packet_type != 0x04:
            return None

        for i in range(4):
            offset = 2 + 15 * i
            voltage = (
                              data[offset + 3] * 256 * 256 * 256
                              + data[offset + 2] * 256 * 256
                              + data[offset + 1] * 256
                              + data[offset + 0]
                      ) / 100000
            current = (
                              data[offset + 7] * 256 * 256 * 256
                              + data[offset + 6] * 256 * 256
                              + data[offset + 5] * 256
                              + data[offset + 4]
                      ) / 100000
            return  voltage * current

    def start(self,name):
        self._inference_name=name
        self._request_data()
        assert not self._running, "Measurement process is running, you have to stop before start"
        self._running = True
        self._power=0
        self._energy=0
        self._n_measurements=0
        self._measure_thread = threading.Thread(target=self._read_data)
        self._measure_thread.start()
        self._start_time=time.time()
        self._raw_data=[]
        self.status = "measuring"


    def stop(self):
        assert self._running, "Measurement process is not running"
        self._running = False
        self._measure_thread=None
        self.status = "ready"

    def _read_data(self):
        continue_time = time.time()
        while self._running:
            if time.time() >= continue_time:
                continue_time = time.time() + 1
                self._ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")
            data = self._ep_in.read(size_or_buffer=64, timeout=1000)
            power = self._decode(data)
            if power is not None:
                self._power+=power
                self._n_measurements+=1
            time.sleep(self._time_interval)

            self._raw_data.append({
                'timestamp': time.time(),
                'power': power
            })

    def _init(self):
        self._running = True
        self._measure_thread = threading.Thread(target=self._read_data)
        self._measure_thread.start()
        self.status = "initializing"
        time.sleep(20)
        self._running = False
        self._power_init=self._power/self._n_measurements
        self.status = "ready"
        print(f"FNB initialized. Power offset: {self._power_init:.4f} W")

    @property
    def power_init(self):
        return getattr(self, "_power_init", 0)

    def compute_energy_in_window(self, start_time, end_time, stop=True):
        '''Get energy consumption in start_time and end_time, return None if no samples in the window'''
        if not self._raw_data:
            raise ValueError("No data collected")

        if self._time_offset:
            # tracker._time_offset = edge_timestamp - srv_timestamp
            start_time -= self._time_offset
            end_time -= self._time_offset

        
        if stop and self._running:
            self.stop()

        samples_in_window = [d for d in self._raw_data if start_time <= d['timestamp'] <= end_time]

        print(f"Computing energy for window {start_time} - {end_time}.")
        print(f"Min timestamp: {min(d['timestamp'] for d in self._raw_data):.4f}, Max timestamp: {max(d['timestamp'] for d in self._raw_data):.4f}, Samples in window: {len(samples_in_window)}")

        if not samples_in_window:
            raise ValueError("No samples in the specified time window")
        
        # calculate average power in the window, then calculate energy based on duration and power offset
        sum_power = 0
        n_power = 0
        for d in samples_in_window:
            if d['power'] is not None:
                sum_power += d['power']
                n_power += 1
        avg_power = sum_power / n_power if n_power > 0 else 0
        
        # energy = power * duration, where power is average power in the window minus the power offset (idle power), duration is end_time - start_time
        duration = end_time - start_time
        energy_mWh = (avg_power - self._power_init) * duration / 3.6  # Convert to mWh

        # return detailed results including start_time, end_time, duration, average power, power offset, and energy consumption
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'average_power_W': avg_power,
            'power_offset_W': self._power_init,
            'energy_mWh': energy_mWh
        }



app = Flask(__name__)

def create_app(raw_file: str, time_interval: float):

    tracker = FNBTool(file_path=raw_file, time_interval=time_interval)

    srv_start_ts = None
    srv_end_ts = None
    
    edge_start_ts = None
    edge_end_ts = None

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/start")
    def start():
        srv_timestamp = time.time()
        payload = request.get_json(force=True, silent=True) or {}
        print(f"Received start signal with payload: {payload}")
        hashid = payload.get("hash_id") or payload.get("run") or payload.get("name") or "run"
        edge_timestamp = payload.get("timestamp") 

        # lưu sự chênh lêch giữa thời gian server và edge để hiệu chỉnh sau này

        tracker._time_offset = edge_timestamp - srv_timestamp

        print(f"time offset (edge - server): {tracker._time_offset:.4f} seconds")

        if tracker.status != "ready":
            return jsonify({"status": "error", "message": "tracker not ready"}), 400

        tracker.start(hashid)
        return jsonify({"status": "started", "hash_id": hashid}), 200

    @app.post("/stop")
    def stop():
        payload = request.get_json(force=True, silent=True) or {}
        if tracker.status != "measuring":
            return jsonify({"status": "error", "message": "no active run"}), 400

        hashid = payload.get("hash_id") or tracker._inference_name or "run"
        start_ts = payload.get("start_ts")
        end_ts = payload.get("end_ts")

        result = tracker.compute_energy_in_window(start_ts, end_ts)

        return jsonify({"status": "stopped", "result": result})

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="FNB monitor server (raw logging)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--raw-file", default="pc_raw.csv", help="CSV output path for raw samples")
    parser.add_argument("--time-interval", type=float, default=0.01, help="Sampling interval for FNB (seconds)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_app(args.raw_file, args.time_interval)
    app.run(host=args.host, port=args.port)
