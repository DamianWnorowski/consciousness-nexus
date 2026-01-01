"""Packet Capture Integration

Packet capture for deep network analysis and troubleshooting.
"""

from __future__ import annotations

import os
import time
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import queue

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class CaptureFormat(str, Enum):
    """Packet capture file format."""
    PCAP = "pcap"
    PCAPNG = "pcapng"
    JSON = "json"
    TEXT = "text"


@dataclass
class CaptureFilter:
    """Filter for packet capture.

    BPF (Berkeley Packet Filter) style filtering.

    Attributes:
        host: Filter by host IP
        net: Filter by network CIDR
        port: Filter by port
        protocol: Filter by protocol (tcp, udp, icmp)
        src_host: Filter by source host
        dst_host: Filter by destination host
        src_port: Filter by source port
        dst_port: Filter by destination port
        expression: Raw BPF expression
    """
    host: Optional[str] = None
    net: Optional[str] = None
    port: Optional[int] = None
    protocol: Optional[str] = None
    src_host: Optional[str] = None
    dst_host: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    expression: Optional[str] = None

    def to_bpf(self) -> str:
        """Convert to BPF filter expression."""
        if self.expression:
            return self.expression

        parts = []

        if self.protocol:
            parts.append(self.protocol.lower())

        if self.host:
            parts.append(f"host {self.host}")

        if self.net:
            parts.append(f"net {self.net}")

        if self.src_host:
            parts.append(f"src host {self.src_host}")

        if self.dst_host:
            parts.append(f"dst host {self.dst_host}")

        if self.port:
            parts.append(f"port {self.port}")

        if self.src_port:
            parts.append(f"src port {self.src_port}")

        if self.dst_port:
            parts.append(f"dst port {self.dst_port}")

        return " and ".join(parts) if parts else ""


@dataclass
class PacketInfo:
    """Information about a captured packet.

    Attributes:
        timestamp: Capture timestamp
        src_ip: Source IP address
        dst_ip: Destination IP address
        src_port: Source port
        dst_port: Destination port
        protocol: Protocol name
        length: Packet length in bytes
        info: Protocol-specific info
        layers: Parsed protocol layers
        raw_hex: Raw packet in hex
    """
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int = 0
    dst_port: int = 0
    protocol: str = "UNKNOWN"
    length: int = 0
    info: str = ""
    layers: List[str] = field(default_factory=list)
    raw_hex: Optional[str] = None

    @classmethod
    def from_tshark_json(cls, data: Dict[str, Any]) -> PacketInfo:
        """Create PacketInfo from tshark JSON output."""
        source = data.get("_source", {})
        layers = source.get("layers", {})

        # Extract frame info
        frame = layers.get("frame", {})
        timestamp_str = frame.get("frame.time_epoch", "0")
        try:
            timestamp = datetime.fromtimestamp(float(timestamp_str))
        except (ValueError, TypeError):
            timestamp = datetime.now()

        length = int(frame.get("frame.len", 0))

        # Extract IP info
        ip = layers.get("ip", {})
        src_ip = ip.get("ip.src", "")
        dst_ip = ip.get("ip.dst", "")

        # Extract transport info
        protocol = "IP"
        src_port = 0
        dst_port = 0
        info = ""

        if "tcp" in layers:
            tcp = layers["tcp"]
            protocol = "TCP"
            src_port = int(tcp.get("tcp.srcport", 0))
            dst_port = int(tcp.get("tcp.dstport", 0))
            flags = tcp.get("tcp.flags_tree", {})
            flag_parts = []
            if flags.get("tcp.flags.syn") == "1":
                flag_parts.append("SYN")
            if flags.get("tcp.flags.ack") == "1":
                flag_parts.append("ACK")
            if flags.get("tcp.flags.fin") == "1":
                flag_parts.append("FIN")
            if flags.get("tcp.flags.rst") == "1":
                flag_parts.append("RST")
            info = " ".join(flag_parts)

        elif "udp" in layers:
            udp = layers["udp"]
            protocol = "UDP"
            src_port = int(udp.get("udp.srcport", 0))
            dst_port = int(udp.get("udp.dstport", 0))

        elif "icmp" in layers:
            icmp = layers["icmp"]
            protocol = "ICMP"
            icmp_type = icmp.get("icmp.type", "")
            info = f"Type: {icmp_type}"

        # Extract application layer
        layer_names = list(layers.keys())

        # Check for HTTP
        if "http" in layers:
            http = layers["http"]
            protocol = "HTTP"
            method = http.get("http.request.method", "")
            uri = http.get("http.request.uri", "")
            if method:
                info = f"{method} {uri}"

        # Check for DNS
        if "dns" in layers:
            dns = layers["dns"]
            protocol = "DNS"
            qry_name = dns.get("dns.qry.name", "")
            if qry_name:
                info = f"Query: {qry_name}"

        # Check for TLS
        if "tls" in layers:
            protocol = "TLS"
            tls = layers["tls"]
            content_type = tls.get("tls.record.content_type", "")
            info = f"Content Type: {content_type}"

        return cls(
            timestamp=timestamp,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            length=length,
            info=info,
            layers=layer_names,
        )


@dataclass
class CaptureStats:
    """Statistics for a capture session.

    Attributes:
        packets_captured: Total packets captured
        packets_dropped: Packets dropped by kernel
        bytes_captured: Total bytes captured
        capture_duration: Duration of capture
        protocol_breakdown: Packets by protocol
        top_talkers: Top source IPs by packet count
        top_destinations: Top destination IPs by packet count
        start_time: Capture start time
        end_time: Capture end time
    """
    packets_captured: int = 0
    packets_dropped: int = 0
    bytes_captured: int = 0
    capture_duration: timedelta = field(default_factory=lambda: timedelta())
    protocol_breakdown: Dict[str, int] = field(default_factory=dict)
    top_talkers: List[tuple] = field(default_factory=list)
    top_destinations: List[tuple] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class CaptureSession:
    """A packet capture session.

    Attributes:
        session_id: Unique session identifier
        interface: Network interface
        filter: Capture filter
        output_file: Output file path
        format: Capture format
        max_packets: Maximum packets to capture
        max_duration: Maximum capture duration
        status: Current status
        stats: Capture statistics
    """
    session_id: str
    interface: str
    filter: Optional[CaptureFilter] = None
    output_file: Optional[Path] = None
    format: CaptureFormat = CaptureFormat.PCAPNG
    max_packets: int = 0
    max_duration: Optional[timedelta] = None
    status: str = "created"
    stats: CaptureStats = field(default_factory=CaptureStats)

    _process: Optional[subprocess.Popen] = field(default=None, repr=False)
    _start_time: Optional[datetime] = field(default=None, repr=False)


class PacketCapture:
    """Packet capture manager.

    Manages packet capture sessions using tcpdump or tshark.

    Usage:
        capture = PacketCapture()

        # Start a capture session
        session = capture.start_capture(
            interface="eth0",
            filter=CaptureFilter(port=80, protocol="tcp"),
            max_packets=1000,
        )

        # Get captured packets
        for packet in capture.read_packets(session):
            process_packet(packet)

        # Stop capture
        capture.stop_capture(session.session_id)

        # Get statistics
        stats = capture.get_stats(session.session_id)
    """

    def __init__(
        self,
        capture_dir: Optional[str] = None,
        namespace: str = "consciousness",
    ):
        self.capture_dir = Path(capture_dir) if capture_dir else Path(tempfile.gettempdir()) / "captures"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace

        self._sessions: Dict[str, CaptureSession] = {}
        self._lock = threading.Lock()
        self._packet_queues: Dict[str, queue.Queue] = {}

        # Detect available tools
        self._tcpdump_path = self._find_executable("tcpdump")
        self._tshark_path = self._find_executable("tshark")
        self._dumpcap_path = self._find_executable("dumpcap")

        # Prometheus metrics
        self.captures_started = Counter(
            f"{namespace}_packet_captures_started_total",
            "Total packet captures started",
            ["interface"],
        )

        self.packets_captured = Counter(
            f"{namespace}_packets_captured_total",
            "Total packets captured",
            ["session_id", "protocol"],
        )

        self.capture_bytes = Counter(
            f"{namespace}_capture_bytes_total",
            "Total bytes captured",
            ["session_id"],
        )

        self.active_captures = Gauge(
            f"{namespace}_active_captures",
            "Number of active capture sessions",
        )

        self.capture_dropped_packets = Counter(
            f"{namespace}_capture_dropped_packets_total",
            "Packets dropped during capture",
            ["session_id"],
        )

    def _find_executable(self, name: str) -> Optional[str]:
        """Find executable in PATH."""
        import shutil
        path = shutil.which(name)
        if path:
            logger.debug(f"Found {name} at {path}")
        return path

    def start_capture(
        self,
        interface: str,
        filter: Optional[CaptureFilter] = None,
        max_packets: int = 0,
        max_duration: Optional[timedelta] = None,
        output_format: CaptureFormat = CaptureFormat.PCAPNG,
        session_id: Optional[str] = None,
    ) -> CaptureSession:
        """Start a packet capture session.

        Args:
            interface: Network interface to capture on
            filter: Optional packet filter
            max_packets: Maximum packets (0 = unlimited)
            max_duration: Maximum duration
            output_format: Output file format
            session_id: Optional session ID

        Returns:
            CaptureSession
        """
        if not session_id:
            session_id = f"capture_{int(time.time() * 1000)}"

        # Create output file
        ext = "pcapng" if output_format == CaptureFormat.PCAPNG else "pcap"
        output_file = self.capture_dir / f"{session_id}.{ext}"

        session = CaptureSession(
            session_id=session_id,
            interface=interface,
            filter=filter,
            output_file=output_file,
            format=output_format,
            max_packets=max_packets,
            max_duration=max_duration,
        )

        # Build capture command
        if self._dumpcap_path:
            cmd = self._build_dumpcap_command(session)
        elif self._tcpdump_path:
            cmd = self._build_tcpdump_command(session)
        else:
            raise RuntimeError("No packet capture tool available (tcpdump, tshark, or dumpcap)")

        logger.info(f"Starting capture: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            session._process = process
            session._start_time = datetime.now()
            session.status = "running"
            session.stats.start_time = session._start_time

            with self._lock:
                self._sessions[session_id] = session
                self._packet_queues[session_id] = queue.Queue(maxsize=10000)

            self.captures_started.labels(interface=interface).inc()
            self.active_captures.inc()

            # Start background packet processing if duration limited
            if max_duration:
                threading.Thread(
                    target=self._auto_stop_capture,
                    args=(session_id, max_duration.total_seconds()),
                    daemon=True,
                ).start()

        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            session.status = "error"
            raise

        return session

    def _build_tcpdump_command(self, session: CaptureSession) -> List[str]:
        """Build tcpdump command."""
        cmd = [
            self._tcpdump_path,
            "-i", session.interface,
            "-w", str(session.output_file),
            "-n",  # Don't resolve hostnames
        ]

        if session.max_packets > 0:
            cmd.extend(["-c", str(session.max_packets)])

        if session.filter:
            bpf = session.filter.to_bpf()
            if bpf:
                cmd.append(bpf)

        return cmd

    def _build_dumpcap_command(self, session: CaptureSession) -> List[str]:
        """Build dumpcap command."""
        cmd = [
            self._dumpcap_path,
            "-i", session.interface,
            "-w", str(session.output_file),
        ]

        if session.max_packets > 0:
            cmd.extend(["-c", str(session.max_packets)])

        if session.max_duration:
            cmd.extend(["-a", f"duration:{int(session.max_duration.total_seconds())}"])

        if session.filter:
            bpf = session.filter.to_bpf()
            if bpf:
                cmd.extend(["-f", bpf])

        return cmd

    def _auto_stop_capture(self, session_id: str, duration: float):
        """Auto-stop capture after duration."""
        time.sleep(duration)
        self.stop_capture(session_id)

    def stop_capture(self, session_id: str) -> Optional[CaptureStats]:
        """Stop a capture session.

        Args:
            session_id: Session ID to stop

        Returns:
            Final CaptureStats or None
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

        if session._process:
            session._process.terminate()
            try:
                session._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                session._process.kill()

        session.status = "stopped"
        session.stats.end_time = datetime.now()
        if session._start_time:
            session.stats.capture_duration = session.stats.end_time - session._start_time

        # Read final statistics from file
        if session.output_file and session.output_file.exists():
            session.stats = self._analyze_capture_file(session.output_file, session.stats)

        self.active_captures.dec()

        return session.stats

    def _analyze_capture_file(
        self,
        file_path: Path,
        stats: CaptureStats,
    ) -> CaptureStats:
        """Analyze a capture file for statistics."""
        if not self._tshark_path:
            return stats

        try:
            # Get packet count and size
            result = subprocess.run(
                [
                    self._tshark_path,
                    "-r", str(file_path),
                    "-T", "fields",
                    "-e", "frame.len",
                    "-e", "frame.protocols",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                stats.packets_captured = len(lines)

                from collections import defaultdict
                protocol_counts: Dict[str, int] = defaultdict(int)
                total_bytes = 0

                for line in lines:
                    parts = line.split("\t")
                    if len(parts) >= 1:
                        try:
                            total_bytes += int(parts[0])
                        except ValueError:
                            pass
                    if len(parts) >= 2:
                        protocols = parts[1].split(":")
                        for proto in protocols:
                            protocol_counts[proto.upper()] += 1

                stats.bytes_captured = total_bytes
                stats.protocol_breakdown = dict(protocol_counts)

        except Exception as e:
            logger.warning(f"Failed to analyze capture file: {e}")

        return stats

    def read_packets(
        self,
        session: CaptureSession,
        live: bool = True,
    ) -> Iterator[PacketInfo]:
        """Read packets from a capture session.

        Args:
            session: Capture session
            live: If True, read live from running capture

        Yields:
            PacketInfo for each packet
        """
        if not self._tshark_path:
            logger.error("tshark not available for packet reading")
            return

        if live and session.status == "running" and session._process:
            # Live capture - process output
            yield from self._read_live_packets(session)
        elif session.output_file and session.output_file.exists():
            # Read from file
            yield from self._read_pcap_file(session.output_file)

    def _read_live_packets(self, session: CaptureSession) -> Iterator[PacketInfo]:
        """Read packets from live capture."""
        cmd = [
            self._tshark_path,
            "-i", session.interface,
            "-T", "json",
            "-l",  # Line buffered
        ]

        if session.filter:
            bpf = session.filter.to_bpf()
            if bpf:
                cmd.extend(["-f", bpf])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            import json
            buffer = ""

            while session.status == "running":
                line = process.stdout.readline()
                if not line:
                    break

                buffer += line

                # Try to parse complete JSON objects
                try:
                    if buffer.strip().startswith("["):
                        buffer = buffer.strip()[1:]

                    if buffer.strip().endswith(","):
                        buffer = buffer.strip()[:-1]

                    if buffer.strip():
                        data = json.loads(buffer)
                        packet = PacketInfo.from_tshark_json(data)
                        self.packets_captured.labels(
                            session_id=session.session_id,
                            protocol=packet.protocol,
                        ).inc()
                        self.capture_bytes.labels(
                            session_id=session.session_id,
                        ).inc(packet.length)
                        yield packet
                        buffer = ""
                except json.JSONDecodeError:
                    continue

            process.terminate()

        except Exception as e:
            logger.error(f"Error reading live packets: {e}")

    def _read_pcap_file(self, file_path: Path) -> Iterator[PacketInfo]:
        """Read packets from pcap file."""
        import json

        cmd = [
            self._tshark_path,
            "-r", str(file_path),
            "-T", "json",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                packets = json.loads(result.stdout)
                for pkt_data in packets:
                    yield PacketInfo.from_tshark_json(pkt_data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tshark output: {e}")
        except Exception as e:
            logger.error(f"Error reading pcap file: {e}")

    def get_session(self, session_id: str) -> Optional[CaptureSession]:
        """Get a capture session by ID.

        Args:
            session_id: Session ID

        Returns:
            CaptureSession or None
        """
        with self._lock:
            return self._sessions.get(session_id)

    def get_active_sessions(self) -> List[CaptureSession]:
        """Get all active capture sessions.

        Returns:
            List of active CaptureSession objects
        """
        with self._lock:
            return [s for s in self._sessions.values() if s.status == "running"]

    def get_stats(self, session_id: str) -> Optional[CaptureStats]:
        """Get statistics for a capture session.

        Args:
            session_id: Session ID

        Returns:
            CaptureStats or None
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.stats
        return None

    def cleanup_old_captures(self, max_age: timedelta = timedelta(hours=24)):
        """Clean up old capture files.

        Args:
            max_age: Maximum age of files to keep
        """
        cutoff = datetime.now() - max_age

        for file in self.capture_dir.iterdir():
            if file.is_file():
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime < cutoff:
                    try:
                        file.unlink()
                        logger.info(f"Cleaned up old capture: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {file}: {e}")

    def get_interfaces(self) -> List[Dict[str, Any]]:
        """Get available network interfaces.

        Returns:
            List of interface information dicts
        """
        interfaces = []

        if self._dumpcap_path:
            try:
                result = subprocess.run(
                    [self._dumpcap_path, "-D"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if line:
                            parts = line.split(". ", 1)
                            if len(parts) == 2:
                                interfaces.append({
                                    "index": parts[0],
                                    "name": parts[1].split(" ")[0],
                                    "description": parts[1],
                                })
            except Exception as e:
                logger.warning(f"Failed to list interfaces: {e}")

        return interfaces

    def is_available(self) -> bool:
        """Check if packet capture is available.

        Returns:
            True if capture tools are available
        """
        return bool(self._tcpdump_path or self._dumpcap_path)
