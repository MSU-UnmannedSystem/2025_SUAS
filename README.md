# MSU Unmanned Systems: 2025-2026 Autonomous Flight Software

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![NVIDIA Jetson](https://img.shields.io/badge/Hardware-Jetson_Orin_Nano-76B900.svg)](https://developer.nvidia.com/embedded-computing)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-FFD700.svg)](https://github.com/ultralytics/ultralytics)

This repository contains the primary flight, telemetry, and computer vision source code for the Michigan State University Unmanned Systems competition drone. 

## System Architecture
The software is engineered to run on an **NVIDIA Jetson Orin Nano** edge-compute module, interfacing directly with a Cube flight controller via `dronekit`. 
* **Computer Vision:** Utilizes highly optimized YOLOv11 models exported to TensorRT/ONNX formats for real-time, low-latency object detection and geolocation target mapping.
* **Flight Logic:** Implements automated pathfinding (A* / DFS) and ArduPilot telemetry logging.
* **Hardware Interfacing:** Manages GPIO triggers, camera pipelines, and rescue-drop mechanisms.

## Repository Structure
* `/inference`: Core computer vision pipelines, camera ingestion, and YOLO engine execution.
* `/train`: Dataset configurations and local model training scripts.
* `/listener`: Telemetry logging and simulated flight data streams.
* `/drone_simulator`: Integration testing environment for wired loggers and camera configurations.
* `/pathfinding`: Algorithmic flight routing and grid searches.

## Dependencies
* Python 3.6 (DroneKit specific environments)
* Python 3.11 (Primary runtime)
* `dronekit`, `opencv-python`, `ultralytics`, `torch`