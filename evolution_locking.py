#!/usr/bin/env python3
"""
🔒 EVOLUTION LOCKING MECHANISM
==============================

Distributed locking system to prevent concurrent evolution processes.
"""

import asyncio
import json
import time
import os
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import threading
from contextlib import contextmanager

class LockType(Enum):
    """Types of evolution locks"""
    EXCLUSIVE = "exclusive"        # Only one process can hold
    SHARED = "shared"             # Multiple processes can hold for read-only
    INTENT = "intent"             # Intention to acquire exclusive lock

class LockStatus(Enum):
    """Lock acquisition status"""
    ACQUIRED = "acquired"
    WAITING = "waiting"
    TIMEOUT = "timeout"
    CONFLICT = "conflict"
    ERROR = "error"

@dataclass
class EvolutionLock:
    """Evolution process lock"""
    lock_id: str
    process_id: str
    lock_type: LockType
    resource: str
    acquired_at: float
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    heartbeat: float = field(default_factory=time.time)

@dataclass
class LockRequest:
    """Request to acquire a lock"""
    process_id: str
    lock_type: LockType
    resource: str
    timeout: float = 30.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LockResponse:
    """Response to lock acquisition request"""
    status: LockStatus
    lock: Optional[EvolutionLock] = None
    message: str = ""
    wait_time: float = 0.0

class EvolutionLockManager:
    """Distributed lock manager for evolution processes"""

    def __init__(self, lock_dir: str = "locks", heartbeat_interval: float = 5.0):
        self.lock_dir = lock_dir
        self.heartbeat_interval = heartbeat_interval
        self.held_locks: Dict[str, EvolutionLock] = {}
        self.waiting_requests: Dict[str, LockRequest] = {}

        # Create lock directory
        os.makedirs(lock_dir, exist_ok=True)

        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def _get_lock_file(self, resource: str) -> str:
        """Get lock file path for resource"""
        # Create consistent hash for resource name
        resource_hash = hashlib.md5(resource.encode()).hexdigest()[:8]
        return os.path.join(self.lock_dir, f"lock_{resource_hash}.json")

    def _read_lock_file(self, lock_file: str) -> Optional[EvolutionLock]:
        """Read lock information from file"""
        if not os.path.exists(lock_file):
            return None

        try:
            with open(lock_file, 'r') as f:
                data = json.load(f)

            # Check if lock is still valid
            if data.get('expires_at') and time.time() > data['expires_at']:
                # Lock expired, remove file
                os.remove(lock_file)
                return None

            return EvolutionLock(
                lock_id=data['lock_id'],
                process_id=data['process_id'],
                lock_type=LockType(data['lock_type']),
                resource=data['resource'],
                acquired_at=data['acquired_at'],
                expires_at=data.get('expires_at'),
                metadata=data.get('metadata', {}),
                heartbeat=data.get('heartbeat', time.time())
            )
        except (json.JSONDecodeError, KeyError):
            # Corrupted lock file, remove it
            if os.path.exists(lock_file):
                os.remove(lock_file)
            return None

    def _write_lock_file(self, lock_file: str, lock: EvolutionLock):
        """Write lock information to file"""
        data = {
            'lock_id': lock.lock_id,
            'process_id': lock.process_id,
            'lock_type': lock.lock_type.value,
            'resource': lock.resource,
            'acquired_at': lock.acquired_at,
            'expires_at': lock.expires_at,
            'metadata': lock.metadata,
            'heartbeat': lock.heartbeat
        }

        # Cross-platform atomic write
        temp_file = lock_file + '.tmp'
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Use os.replace for atomic move (Python 3.3+)
            os.replace(temp_file, lock_file)
        except OSError:
            # Fallback for older Python versions
            try:
                os.rename(temp_file, lock_file)
            except OSError:
                # Last resort - direct write (not atomic but better than nothing)
                with open(lock_file, 'w') as f:
                    json.dump(data, f, indent=2)
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass

    def _remove_lock_file(self, lock_file: str):
        """Remove lock file"""
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except OSError:
            pass  # File may have been removed by another process

    def _can_acquire_lock(self, resource: str, lock_type: LockType,
                         requesting_process: str) -> Tuple[bool, str]:
        """Check if lock can be acquired"""
        lock_file = self._get_lock_file(resource)
        existing_lock = self._read_lock_file(lock_file)

        if not existing_lock:
            return True, "Resource is free"

        # Check if requesting process already holds the lock
        if existing_lock.process_id == requesting_process:
            return True, "Process already holds the lock"

        # Check lock compatibility
        if lock_type == LockType.EXCLUSIVE:
            # Exclusive locks conflict with any existing lock
            return False, f"Resource held by process {existing_lock.process_id}"

        elif lock_type == LockType.SHARED:
            # Shared locks are compatible with other shared locks
            if existing_lock.lock_type == LockType.SHARED:
                return True, "Compatible with existing shared lock"
            else:
                return False, f"Incompatible with {existing_lock.lock_type.value} lock"

        elif lock_type == LockType.INTENT:
            # Intent locks are compatible with shared locks but not exclusive
            if existing_lock.lock_type == LockType.SHARED:
                return True, "Intent compatible with shared lock"
            else:
                return False, f"Incompatible with {existing_lock.lock_type.value} lock"

        return False, "Unknown lock compatibility issue"

    async def acquire_lock(self, request: LockRequest) -> LockResponse:
        """Acquire a lock with timeout"""
        start_time = time.time()
        lock_file = self._get_lock_file(request.resource)

        while time.time() - start_time < request.timeout:
            can_acquire, reason = self._can_acquire_lock(
                request.resource, request.lock_type, request.process_id
            )

            if can_acquire:
                # Create lock
                lock = EvolutionLock(
                    lock_id=f"{request.process_id}_{request.resource}_{int(time.time())}",
                    process_id=request.process_id,
                    lock_type=request.lock_type,
                    resource=request.resource,
                    acquired_at=time.time(),
                    expires_at=time.time() + 3600 if request.lock_type == LockType.EXCLUSIVE else None,
                    metadata=request.metadata
                )

                # Write lock file
                self._write_lock_file(lock_file, lock)
                self.held_locks[lock.lock_id] = lock

                return LockResponse(
                    status=LockStatus.ACQUIRED,
                    lock=lock,
                    message=f"Lock acquired: {reason}"
                )

            # Wait before retrying
            await asyncio.sleep(0.1)

        # Timeout
        return LockResponse(
            status=LockStatus.TIMEOUT,
            message=f"Timeout waiting for lock on {request.resource}",
            wait_time=time.time() - start_time
        )

    def release_lock(self, lock_id: str, process_id: str) -> bool:
        """Release a held lock"""
        lock = self.held_locks.get(lock_id)
        if not lock:
            return False

        # Verify ownership
        if lock.process_id != process_id:
            return False

        # Remove lock file
        lock_file = self._get_lock_file(lock.resource)
        self._remove_lock_file(lock_file)

        # Remove from held locks
        del self.held_locks[lock_id]

        return True

    def force_release_lock(self, resource: str) -> bool:
        """Force release lock (admin operation)"""
        lock_file = self._get_lock_file(resource)
        self._remove_lock_file(lock_file)

        # Remove from held locks if present
        locks_to_remove = [lid for lid, lock in self.held_locks.items()
                          if lock.resource == resource]
        for lock_id in locks_to_remove:
            del self.held_locks[lock_id]

        return True

    def list_locks(self) -> List[EvolutionLock]:
        """List all active locks"""
        locks = []

        # Check all lock files
        if os.path.exists(self.lock_dir):
            for filename in os.listdir(self.lock_dir):
                if filename.startswith('lock_') and filename.endswith('.json'):
                    lock_file = os.path.join(self.lock_dir, filename)
                    lock = self._read_lock_file(lock_file)
                    if lock:
                        locks.append(lock)

        return locks

    def _heartbeat_worker(self):
        """Background thread to update heartbeats"""
        while True:
            try:
                current_time = time.time()

                # Update heartbeats for held locks
                for lock in self.held_locks.values():
                    lock.heartbeat = current_time

                    # Update lock file with new heartbeat
                    lock_file = self._get_lock_file(lock.resource)
                    self._write_lock_file(lock_file, lock)

                time.sleep(self.heartbeat_interval)

            except Exception as e:
                print(f"Heartbeat error: {e}")
                time.sleep(self.heartbeat_interval)

    def _cleanup_worker(self):
        """Background thread to clean up expired locks"""
        while True:
            try:
                current_time = time.time()

                # Check all lock files for expired locks
                if os.path.exists(self.lock_dir):
                    for filename in os.listdir(self.lock_dir):
                        if filename.startswith('lock_') and filename.endswith('.json'):
                            lock_file = os.path.join(self.lock_dir, filename)
                            lock = self._read_lock_file(lock_file)

                            if lock:
                                # Check for expired exclusive locks
                                if (lock.lock_type == LockType.EXCLUSIVE and
                                    lock.expires_at and current_time > lock.expires_at):
                                    print(f"[+] Removing expired lock: {lock.lock_id}")
                                    self._remove_lock_file(lock_file)

                                # Check for stale heartbeats (no heartbeat for 30 seconds)
                                elif current_time - lock.heartbeat > 30:
                                    print(f"[+] Removing stale lock (no heartbeat): {lock.lock_id}")
                                    self._remove_lock_file(lock_file)

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                print(f"Cleanup error: {e}")
                time.sleep(10)

@contextmanager
def evolution_lock(lock_manager: EvolutionLockManager, request: LockRequest):
    """Context manager for evolution locks"""
    response = None

    try:
        # Acquire lock
        response = asyncio.run(lock_manager.acquire_lock(request))

        if response.status != LockStatus.ACQUIRED:
            raise RuntimeError(f"Failed to acquire lock: {response.message}")

        yield response.lock

    finally:
        # Release lock
        if response and response.lock:
            lock_manager.release_lock(response.lock.lock_id, request.process_id)

class EvolutionLockGuard:
    """Guard class for protecting evolution operations with locks"""

    def __init__(self, lock_manager: EvolutionLockManager):
        self.lock_manager = lock_manager

    async def protect_evolution(self, process_id: str, operation: str,
                               operation_func, *args, **kwargs):
        """Protect evolution operation with appropriate locking"""

        # Determine lock type based on operation
        lock_configs = {
            'trigger_evolution': (LockType.EXCLUSIVE, 'evolution_system'),
            'stop_evolution': (LockType.EXCLUSIVE, 'evolution_system'),
            'modify_contracts': (LockType.EXCLUSIVE, 'evolution_contracts'),
            'read_status': (LockType.SHARED, 'evolution_status'),
            'view_logs': (LockType.SHARED, 'evolution_logs'),
            'force_rollback': (LockType.EXCLUSIVE, 'evolution_system'),
            'system_config': (LockType.EXCLUSIVE, 'system_config'),
        }

        lock_type, resource = lock_configs.get(operation, (LockType.EXCLUSIVE, 'evolution_system'))

        # Create lock request
        request = LockRequest(
            process_id=process_id,
            lock_type=lock_type,
            resource=resource,
            timeout=60.0,  # 1 minute timeout
            metadata={'operation': operation}
        )

        # Acquire lock
        print(f"[*] Acquiring {lock_type.value} lock for {operation}...")
        response = await self.lock_manager.acquire_lock(request)

        if response.status != LockStatus.ACQUIRED:
            raise RuntimeError(f"Cannot proceed with {operation}: {response.message}")

        try:
            print(f"[+] Lock acquired, executing {operation}...")
            # Execute operation
            result = await operation_func(*args, **kwargs)
            print(f"[+] Operation {operation} completed successfully")
            return result

        finally:
            # Release lock
            self.lock_manager.release_lock(response.lock.lock_id, process_id)
            print(f"[+] Lock released for {operation}")

def main():
    """CLI interface for lock management"""
    import argparse

    parser = argparse.ArgumentParser(description="Evolution Lock Manager")
    parser.add_argument("--acquire", nargs=3, metavar=('PROCESS_ID', 'LOCK_TYPE', 'RESOURCE'),
                       help="Acquire lock (exclusive/shared/intent)")
    parser.add_argument("--release", nargs=2, metavar=('LOCK_ID', 'PROCESS_ID'),
                       help="Release lock")
    parser.add_argument("--list", action="store_true", help="List all active locks")
    parser.add_argument("--force-release", metavar='RESOURCE',
                       help="Force release lock for resource (admin)")
    parser.add_argument("--test-concurrent", action="store_true",
                       help="Test concurrent lock acquisition")

    args = parser.parse_args()

    lock_manager = EvolutionLockManager()

    if args.acquire:
        process_id, lock_type_str, resource = args.acquire
        lock_type = LockType(lock_type_str)

        request = LockRequest(
            process_id=process_id,
            lock_type=lock_type,
            resource=resource,
            timeout=10.0
        )

        async def acquire():
            response = await lock_manager.acquire_lock(request)
            print(f"Lock status: {response.status.value}")
            if response.lock:
                print(f"Lock ID: {response.lock.lock_id}")
            print(f"Message: {response.message}")

        asyncio.run(acquire())

    elif args.release:
        lock_id, process_id = args.release
        success = lock_manager.release_lock(lock_id, process_id)
        print(f"Release {'successful' if success else 'failed'}")

    elif args.list:
        locks = lock_manager.list_locks()
        if locks:
            for lock in locks:
                print(f"Lock: {lock.lock_id} | Process: {lock.process_id} | Type: {lock.lock_type.value} | Resource: {lock.resource}")
        else:
            print("No active locks")

    elif args.force_release:
        success = lock_manager.force_release_lock(args.force_release)
        print(f"Force release {'successful' if success else 'failed'}")

    elif args.test_concurrent:
        # Test concurrent lock acquisition
        async def test_concurrent():
            import asyncio

            async def acquire_lock(process_id: str):
                request = LockRequest(
                    process_id=process_id,
                    lock_type=LockType.EXCLUSIVE,
                    resource="test_resource",
                    timeout=5.0
                )

                start_time = time.time()
                response = await lock_manager.acquire_lock(request)
                elapsed = time.time() - start_time

                print(f"Process {process_id}: {response.status.value} (took {elapsed:.2f}s)")

                if response.status == LockStatus.ACQUIRED:
                    await asyncio.sleep(2)  # Hold lock for 2 seconds
                    lock_manager.release_lock(response.lock.lock_id, process_id)

            # Launch multiple concurrent requests
            tasks = []
            for i in range(3):
                tasks.append(acquire_lock(f"process_{i}"))

            await asyncio.gather(*tasks)

        asyncio.run(test_concurrent())

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
