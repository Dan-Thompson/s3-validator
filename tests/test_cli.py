#!/usr/bin/env python3
"""Check file and pipe input against an independent CRC64/NVME oracle."""

import base64
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def expected_checksums(data, sizes):
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x9A6C9329AC4BC9B5 if crc & 1 else 0)
        table.append(crc)
    crc = 0xFFFFFFFFFFFFFFFF
    position = 0
    for size in sizes:
        for byte in memoryview(data)[position:size]:
            crc = table[(crc ^ byte) & 255] ^ (crc >> 8)
        position = size
        value = crc ^ 0xFFFFFFFFFFFFFFFF
        yield size, base64.b64encode(value.to_bytes(8, "big")) + b"\n"


def run(binary, *args, **kwargs):
    return subprocess.run([str(binary), *map(str, args)], capture_output=True,
                          check=True, timeout=60, **kwargs).stdout


def check_large_inputs(binary, path):
    # Larger than the bounded pipeline queue, with enough data to reuse buffers.
    # Distinct chunk contents make out-of-order CRC combination detectable.
    chunk_size = 4 << 20
    data = b"".join(bytes([index]) * chunk_size for index in range(20))
    data += bytes([20]) * 37
    # Generated once with expected_checksums(), avoiding an 80 MiB Python
    # byte-by-byte oracle pass every time this integration test runs.
    expected = b"7jm0yIraBvY=\n"
    path.write_bytes(data)
    assert run(binary, path) == expected, "large file queue wrap"
    assert run(binary, input=data) == expected, "large pipe"
    with path.open("rb") as redirected:
        assert run(binary, stdin=redirected) == expected, "large redirected stdin"

    if sys.platform.startswith("linux") and hasattr(os, "sched_getaffinity"):
        cpu = min(os.sched_getaffinity(0))

        def restrict_to_one_cpu():
            os.sched_setaffinity(0, {cpu})

        assert run(binary, path, preexec_fn=restrict_to_one_cpu) == expected, \
            "large file with one available CPU"
        return True
    return False


def main():
    binary = Path(sys.argv[1]).resolve()
    assert run(binary, input=b"123456789") == b"rosUhgp5mIg=\n"
    sizes = [0, 1, 7, 8, 9, 15, 16, 17, 255, 256, 257,
             (1 << 20) - 1, 1 << 20, (1 << 20) + 1,
             (4 << 20) - 1, 4 << 20, (4 << 20) + 1,
             (8 << 20) + 19, (17 << 20) + 37]
    data = (bytes(range(256)) * ((sizes[-1] + 255) // 256))[:sizes[-1]]
    with tempfile.TemporaryDirectory(prefix="crc64nvme-test-") as directory:
        path = Path(directory) / "input.bin"
        for size, expected in expected_checksums(data, sizes):
            path.write_bytes(data[:size])
            assert run(binary, path) == expected, ("file", size)
            assert run(binary, input=data[:size]) == expected, ("pipe", size)
        invalid_paths = [Path(directory) / "missing", Path(directory)]
        if hasattr(os, "mkfifo"):
            fifo = Path(directory) / "fifo"
            os.mkfifo(fifo)
            invalid_paths.append(fifo)
        for invalid in invalid_paths:
            result = subprocess.run([str(binary), str(invalid)], capture_output=True,
                                    timeout=60)
            assert result.returncode != 0 and not result.stdout, invalid
        with open(os.devnull, "wb") as unreadable:
            result = subprocess.run([str(binary)], stdin=unreadable, capture_output=True,
                                    timeout=60)
            assert result.returncode != 0 and not result.stdout, "stdin read error"
        checked_affinity = check_large_inputs(binary, path)
    affinity = ", one-CPU affinity" if checked_affinity else ""
    print(f"CRC64/NVME CLI tests passed ({len(sizes)} file/pipe sizes, "
          f"large queue wrap and stdin, errors{affinity})")


if __name__ == "__main__":
    main()
