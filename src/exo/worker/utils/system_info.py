import socket
import sys
from pathlib import Path
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.profiling import NetworkInterfaceInfo


def _read_file_safe(path: Path) -> str | None:
    """Read a file and return its stripped contents, or None on failure."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None


def _get_linux_friendly_name() -> str | None:
    """
    Retrieve a human-friendly hostname on Linux.

    Checks /etc/machine-info for PRETTY_HOSTNAME first, then falls back
    to /etc/hostname.
    """
    machine_info = _read_file_safe(Path("/etc/machine-info"))
    if machine_info:
        for line in machine_info.splitlines():
            if line.startswith("PRETTY_HOSTNAME="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value

    return _read_file_safe(Path("/etc/hostname"))


async def get_friendly_name() -> str:
    """
    Get the friendly name (computer name) of this machine.

    On macOS, uses scutil to retrieve the ComputerName.
    On Linux, checks /etc/machine-info for PRETTY_HOSTNAME, then /etc/hostname.
    On Windows, uses socket.gethostname() which returns the NetBIOS name.
    Falls back to socket.gethostname() on other platforms or errors.
    """
    hostname = socket.gethostname()

    if sys.platform == "darwin":
        try:
            process = await run_process(["scutil", "--get", "ComputerName"])
        except CalledProcessError:
            return hostname
        return process.stdout.decode("utf-8", errors="replace").strip() or hostname

    if sys.platform == "linux":
        linux_name = _get_linux_friendly_name()
        return linux_name if linux_name else hostname

    # Windows and other platforms: socket.gethostname() is sufficient
    return hostname


def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(name=iface, ip_address=service.address)
                    )
                case _:
                    pass

    return interfaces_info


def _get_linux_model() -> str:
    """
    Retrieve the product/model name on Linux via DMI.

    Falls back to 'Linux Machine' if DMI info is unavailable.
    """
    dmi_paths = [
        Path("/sys/devices/virtual/dmi/id/product_name"),
        Path("/sys/class/dmi/id/product_name"),
    ]
    for path in dmi_paths:
        value = _read_file_safe(path)
        if value and value.lower() not in ("", "to be filled by o.e.m.", "system product name"):
            return value
    return "Linux Machine"


def _get_linux_chip() -> str:
    """
    Retrieve CPU model name from /proc/cpuinfo on Linux.

    Falls back to 'Unknown CPU' if parsing fails.
    """
    cpuinfo = _read_file_safe(Path("/proc/cpuinfo"))
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
    return "Unknown CPU"


async def _get_windows_model() -> str:
    """
    Retrieve the system model on Windows via wmic.

    Falls back to 'Windows PC' if wmic fails.
    """
    try:
        process = await run_process(
            ["wmic", "computersystem", "get", "model", "/value"],
        )
        output = process.stdout.decode("utf-8", errors="replace").strip()
        for line in output.splitlines():
            if line.startswith("Model="):
                value = line.split("=", 1)[1].strip()
                if value and value.lower() not in ("", "system product name", "to be filled by o.e.m."):
                    return value
    except (CalledProcessError, OSError):
        pass
    return "Windows PC"


async def _get_windows_chip() -> str:
    """
    Retrieve the CPU name on Windows via wmic.

    Falls back to 'Unknown CPU' if wmic fails.
    """
    try:
        process = await run_process(
            ["wmic", "cpu", "get", "name", "/value"],
        )
        output = process.stdout.decode("utf-8", errors="replace").strip()
        for line in output.splitlines():
            if line.startswith("Name="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    except (CalledProcessError, OSError):
        pass
    return "Unknown CPU"


async def get_model_and_chip() -> tuple[str, str]:
    """
    Get system model and chip/CPU information.

    On macOS, uses system_profiler to retrieve Model Name and Chip.
    On Linux, reads DMI info for model and /proc/cpuinfo for CPU.
    On Windows, uses wmic to query system and CPU info.
    Returns ("Unknown Model", "Unknown Chip") on other platforms or errors.
    """
    if sys.platform == "linux":
        return (_get_linux_model(), _get_linux_chip())

    if sys.platform == "win32":
        model = await _get_windows_model()
        chip = await _get_windows_chip()
        return (model, chip)

    if sys.platform != "darwin":
        return ("Unknown Model", "Unknown Chip")

    try:
        process = await run_process(
            [
                "system_profiler",
                "SPHardwareDataType",
            ]
        )
    except CalledProcessError:
        return ("Unknown Model", "Unknown Chip")

    output = process.stdout.decode().strip()

    model_line = next(
        (line for line in output.split("\n") if "Model Name" in line), None
    )
    model = model_line.split(": ")[1] if model_line else "Unknown Model"

    chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
    chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

    return (model, chip)
