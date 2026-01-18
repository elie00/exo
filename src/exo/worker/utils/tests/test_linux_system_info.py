"""Tests for Linux system identity functions in system_info.py."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.worker.utils.system_info import (
    _get_linux_chip,
    _get_linux_friendly_name,
    _get_linux_model,
    _read_file_safe,
    get_friendly_name,
    get_model_and_chip,
)


class TestReadFileSafe:
    """Tests for the _read_file_safe helper."""

    def test_returns_none_on_missing_file(self, tmp_path: Path):
        """Returns None when file does not exist."""
        result = _read_file_safe(tmp_path / "nonexistent.txt")
        assert result is None

    def test_reads_file_contents(self, tmp_path: Path):
        """Reads and strips file contents."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("  hello world  \n")
        result = _read_file_safe(test_file)
        assert result == "hello world"

    def test_returns_none_on_empty_file(self, tmp_path: Path):
        """Returns None when file is empty or whitespace only."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("   \n")
        result = _read_file_safe(test_file)
        assert result is None


class TestGetLinuxFriendlyName:
    """Tests for _get_linux_friendly_name."""

    def test_returns_pretty_hostname(self, tmp_path: Path):
        """Returns PRETTY_HOSTNAME from machine-info when present."""
        machine_info = tmp_path / "machine-info"
        machine_info.write_text('PRETTY_HOSTNAME="My Server"\n')

        with patch(
            "exo.worker.utils.system_info._read_file_safe",
            side_effect=lambda p: machine_info.read_text()
            if "machine-info" in str(p)
            else None,
        ):
            with patch(
                "exo.worker.utils.system_info.Path",
                side_effect=lambda p: tmp_path / Path(p).name,
            ):
                result = _get_linux_friendly_name()
                assert result == "My Server"

    def test_returns_hostname_fallback(self, tmp_path: Path):
        """Falls back to /etc/hostname when PRETTY_HOSTNAME is absent."""
        hostname_file = tmp_path / "hostname"
        hostname_file.write_text("myhost\n")

        def mock_read(p: Path) -> str | None:
            if "machine-info" in str(p):
                return None
            if "hostname" in str(p):
                return hostname_file.read_text().strip()
            return None

        with patch("exo.worker.utils.system_info._read_file_safe", side_effect=mock_read):
            result = _get_linux_friendly_name()
            assert result == "myhost"

    def test_returns_none_when_no_files(self):
        """Returns None when no identity files exist."""
        with patch("exo.worker.utils.system_info._read_file_safe", return_value=None):
            result = _get_linux_friendly_name()
            assert result is None


class TestGetLinuxModel:
    """Tests for _get_linux_model."""

    def test_returns_dmi_product_name(self, tmp_path: Path):
        """Returns product name from DMI when available."""
        product_file = tmp_path / "product_name"
        product_file.write_text("Dell PowerEdge R750\n")

        def mock_read(p: Path) -> str | None:
            if "product_name" in str(p):
                return product_file.read_text().strip()
            return None

        with patch("exo.worker.utils.system_info._read_file_safe", side_effect=mock_read):
            result = _get_linux_model()
            assert result == "Dell PowerEdge R750"

    def test_ignores_placeholder_values(self):
        """Ignores placeholder values like 'To Be Filled By O.E.M.'."""
        with patch(
            "exo.worker.utils.system_info._read_file_safe",
            return_value="To Be Filled By O.E.M.",
        ):
            result = _get_linux_model()
            assert result == "Linux Machine"

    def test_returns_fallback_when_unavailable(self):
        """Returns 'Linux Machine' when DMI info is unavailable."""
        with patch("exo.worker.utils.system_info._read_file_safe", return_value=None):
            result = _get_linux_model()
            assert result == "Linux Machine"


class TestGetLinuxChip:
    """Tests for _get_linux_chip."""

    def test_returns_cpu_model_name(self, tmp_path: Path):
        """Returns CPU model name from /proc/cpuinfo."""
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text(
            "processor\t: 0\n"
            "vendor_id\t: GenuineIntel\n"
            "model name\t: Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz\n"
        )

        def mock_read(p: Path) -> str | None:
            if "cpuinfo" in str(p):
                return cpuinfo.read_text()
            return None

        with patch("exo.worker.utils.system_info._read_file_safe", side_effect=mock_read):
            result = _get_linux_chip()
            assert result == "Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz"

    def test_returns_fallback_when_unavailable(self):
        """Returns 'Unknown CPU' when /proc/cpuinfo is unavailable."""
        with patch("exo.worker.utils.system_info._read_file_safe", return_value=None):
            result = _get_linux_chip()
            assert result == "Unknown CPU"


class TestGetFriendlyName:
    """Tests for get_friendly_name async function."""

    @pytest.mark.asyncio
    async def test_linux_returns_linux_name(self):
        """On Linux, uses _get_linux_friendly_name."""
        with (
            patch("exo.worker.utils.system_info.sys.platform", "linux"),
            patch(
                "exo.worker.utils.system_info._get_linux_friendly_name",
                return_value="My Linux Box",
            ),
        ):
            result = await get_friendly_name()
            assert result == "My Linux Box"

    @pytest.mark.asyncio
    async def test_linux_fallback_to_hostname(self):
        """On Linux, falls back to socket.gethostname when Linux name is None."""
        with (
            patch("exo.worker.utils.system_info.sys.platform", "linux"),
            patch(
                "exo.worker.utils.system_info._get_linux_friendly_name", return_value=None
            ),
            patch("exo.worker.utils.system_info.socket.gethostname", return_value="myhost"),
        ):
            result = await get_friendly_name()
            assert result == "myhost"

    @pytest.mark.asyncio
    async def test_darwin_uses_scutil(self):
        """On macOS, uses scutil to get ComputerName."""
        mock_process = MagicMock()
        mock_process.stdout = b"John's MacBook Pro\n"

        with (
            patch("exo.worker.utils.system_info.sys.platform", "darwin"),
            patch(
                "exo.worker.utils.system_info.run_process",
                new_callable=AsyncMock,
                return_value=mock_process,
            ),
        ):
            result = await get_friendly_name()
            assert result == "John's MacBook Pro"

    @pytest.mark.asyncio
    async def test_other_platform_returns_hostname(self):
        """On unsupported platforms, returns socket.gethostname."""
        with (
            patch("exo.worker.utils.system_info.sys.platform", "win32"),
            patch("exo.worker.utils.system_info.socket.gethostname", return_value="DESKTOP-ABC"),
        ):
            result = await get_friendly_name()
            assert result == "DESKTOP-ABC"


class TestGetModelAndChip:
    """Tests for get_model_and_chip async function."""

    @pytest.mark.asyncio
    async def test_linux_returns_linux_identity(self):
        """On Linux, uses Linux identity functions."""
        with (
            patch("exo.worker.utils.system_info.sys.platform", "linux"),
            patch(
                "exo.worker.utils.system_info._get_linux_model",
                return_value="Dell PowerEdge R750",
            ),
            patch(
                "exo.worker.utils.system_info._get_linux_chip",
                return_value="Intel Xeon Platinum 8380",
            ),
        ):
            model, chip = await get_model_and_chip()
            assert model == "Dell PowerEdge R750"
            assert chip == "Intel Xeon Platinum 8380"

    @pytest.mark.asyncio
    async def test_unsupported_platform_returns_unknown(self):
        """On unsupported platforms, returns unknown values."""
        with patch("exo.worker.utils.system_info.sys.platform", "freebsd"):
            model, chip = await get_model_and_chip()
            assert model == "Unknown Model"
            assert chip == "Unknown Chip"
