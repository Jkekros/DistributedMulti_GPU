"""
============================================================
Cluster Health Check & Monitoring
Run from the master (Windows, RTX 5090) to verify all nodes
are reachable, GPUs are functional, and cluster is ready.
Workers: Windows + DirectML (RX 7900 XTX)
============================================================
"""

import os
import sys
import socket
import time
import argparse
import yaml
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: paramiko for SSH-based checks
try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_port(ip, port, timeout=5):
    """Check if a TCP port is reachable."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_node_ssh(ip, user, timeout=10):
    """Check if a node is reachable via SSH and get GPU info."""
    if not HAS_PARAMIKO:
        return {"reachable": False, "error": "paramiko not installed"}

    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ip, username=user, timeout=timeout)

        # Detect GPU type
        # Try nvidia-smi first (CUDA), then DirectML (AMD)
        gpu_info = None

        # NVIDIA check
        stdin, stdout, stderr = client.exec_command(
            "nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader 2>NUL"
        )
        nvidia_output = stdout.read().decode().strip()
        if nvidia_output:
            gpu_info = {"type": "cuda", "details": nvidia_output}

        # DirectML check (AMD GPU on Windows)
        if not gpu_info:
            # Check for AMD GPU via PowerShell
            stdin, stdout, stderr = client.exec_command(
                'powershell -Command "Get-CimInstance Win32_VideoController | '
                'Where-Object {$_.Name -match \'AMD|Radeon\'} | '
                'Select-Object -ExpandProperty Name"'
            )
            amd_output = stdout.read().decode().strip()
            if amd_output:
                gpu_info = {"type": "directml", "details": amd_output}

        # Check PyTorch + DirectML
        pytorch_info = ""
        for cmd in [
            '"%%USERPROFILE%%\\gpu-cluster-env\\Scripts\\python.exe" -c '
            '"import torch; '
            'dml=False; '
            'exec(\'try:\\n import torch_directml; dml=True\\nexcept: pass\'); '
            'print(f\'torch={torch.__version__},cuda={torch.cuda.is_available()},directml={dml}\')"',
        ]:
            stdin, stdout, stderr = client.exec_command(cmd)
            output = stdout.read().decode().strip()
            if output and 'torch=' in output:
                pytorch_info = output
                break

        # System info
        stdin, stdout, stderr = client.exec_command(
            'powershell -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB"'
        )
        ram = stdout.read().decode().strip()
        if ram:
            try:
                ram = f"{float(ram):.0f} GB"
            except ValueError:
                pass

        stdin, stdout, stderr = client.exec_command("hostname")
        hostname = stdout.read().decode().strip()

        client.close()

        return {
            "reachable": True,
            "hostname": hostname,
            "gpu": gpu_info,
            "pytorch": pytorch_info,
            "ram": ram,
        }
    except Exception as e:
        return {"reachable": False, "error": str(e)}


def run_health_check(config_path, ssh_user=None):
    config = load_config(config_path)
    cluster = config["cluster"]

    print("=" * 60)
    print("  GPU Cluster Health Check")
    print("  Master: RTX 5090 (CUDA, Windows)")
    print("  Workers: RX 7900 XTX (DirectML, Windows)")
    print("=" * 60)
    print()

    all_nodes = []

    # Master node
    master = cluster["master"]
    all_nodes.append({
        "name": f"MASTER ({master['hostname']})",
        "ip": master["ip"],
        "role": "master",
        "gpu_type": master.get("gpu_type", "rtx5090"),
        "expected_backend": master.get("gpu_backend", "cuda"),
    })

    # Worker nodes
    for i, worker in enumerate(cluster["workers"]):
        all_nodes.append({
            "name": f"WORKER-{i+1} ({worker['hostname']})",
            "ip": worker["ip"],
            "role": "worker",
            "gpu_type": worker.get("gpu_type", "rx7900xtx"),
            "expected_backend": worker.get("gpu_backend", "rocm"),
        })

    # Check each node
    results = []
    dist_port = cluster["master"]["port"]

    for node in all_nodes:
        print(f"Checking {node['name']} ({node['ip']})...")

        # Port check (SSH)
        ssh_ok = check_port(node["ip"], 22)
        status = "OK" if ssh_ok else "FAIL"
        print(f"  SSH (port 22):        [{status}]")

        # Port check (distributed)
        if node["role"] == "master":
            dist_ok = check_port(node["ip"], dist_port)
            status = "OK" if dist_ok else "NOT LISTENING (normal before launch)"
            print(f"  Dist port ({dist_port}):   [{status}]")

        # SSH-based deep check
        node_result = {"node": node["name"], "ip": node["ip"], "ssh": ssh_ok}

        if ssh_ok and ssh_user and HAS_PARAMIKO:
            info = check_node_ssh(node["ip"], ssh_user)
            node_result.update(info)

            if info["reachable"]:
                print(f"  Hostname:             {info.get('hostname', 'N/A')}")
                print(f"  RAM:                  {info.get('ram', 'N/A')}")
                if info.get("gpu"):
                    gpu = info["gpu"]
                    print(f"  GPU Backend:          {gpu['type'].upper()}")
                    print(f"  GPU Details:          {gpu['details'][:80]}")

                    # Verify correct backend
                    if gpu["type"] != node["expected_backend"]:
                        print(f"  [WARN] Expected {node['expected_backend']}, got {gpu['type']}")
                else:
                    print(f"  GPU:                  NOT DETECTED")

                print(f"  PyTorch:              {info.get('pytorch', 'N/A')}")
            else:
                print(f"  SSH deep check:       FAILED ({info.get('error', 'unknown')})")
        elif not HAS_PARAMIKO and ssh_user:
            print(f"  SSH deep check:       SKIPPED (install paramiko: pip install paramiko)")

        results.append(node_result)
        print()

    # Summary
    print("=" * 60)
    print("  CLUSTER SUMMARY")
    print("=" * 60)

    total = len(results)
    reachable = sum(1 for r in results if r.get("ssh", False))

    print(f"  Total nodes:       {total}")
    print(f"  Reachable (SSH):   {reachable}/{total}")
    num_workers = len(cluster.get('workers', []))
    print(f"  Master (CUDA):     1x RTX 5090 (Windows)")
    print(f"  Workers (DML):     {num_workers}x RX 7900 XTX (Windows + DirectML)")
    print(f"  Backend:           gloo")
    print(f"  World size:        {config['training']['world_size']}")

    if reachable == total:
        print(f"\n  STATUS: CLUSTER READY")
    else:
        unreachable = [r["node"] for r in results if not r.get("ssh", False)]
        print(f"\n  STATUS: {total - reachable} NODE(S) UNREACHABLE")
        for name in unreachable:
            print(f"    - {name}")

    print("=" * 60)
    return reachable == total


def main():
    parser = argparse.ArgumentParser(description="Cluster Health Check")
    parser.add_argument("--config", type=str, default="cluster_config.yaml")
    parser.add_argument("--ssh-user", type=str, default=None,
                        help="SSH username for deep checks (requires paramiko)")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    success = run_health_check(config_path, args.ssh_user)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
