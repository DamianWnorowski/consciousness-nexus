"""
Async Utilities for Consciousness Computing Suite
=================================================

Provides async utilities, task management, rate limiting, and concurrency
control for high-performance consciousness computing operations.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import aiohttp
import backoff

T = TypeVar('T')

@dataclass
class AsyncTask:
    """Represents an asynchronous task with metadata"""
    id: str
    coroutine: Awaitable[Any]
    priority: int = 0
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AsyncTaskManager:
    """
    Advanced async task manager with priority queuing, resource management,
    and performance monitoring for consciousness computing operations.
    """

    def __init__(self, max_concurrent: int = 10, queue_size: int = 1000):
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size

        # Task management
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=queue_size)
        self.active_tasks: Dict[str, AsyncTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)

        # Resource management
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Monitoring
        self.task_counter = 0
        self.success_count = 0
        self.error_count = 0

        # Control
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []

    async def start(self, num_workers: int = 3):
        """Start the task manager with specified number of workers"""
        if self.running:
            return

        self.running = True

        # Start worker tasks
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self.worker_tasks.append(worker)

        print(f"AsyncTaskManager started with {num_workers} workers")

    async def stop(self):
        """Stop the task manager and wait for completion"""
        if not self.running:
            return

        self.running = False

        # Cancel worker tasks
        for worker in self.worker_tasks:
            worker.cancel()

        # Wait for cancellation
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        print(f"AsyncTaskManager stopped. Processed {self.task_counter} tasks")

    async def submit_task(self, coroutine: Awaitable[T], priority: int = 0,
                         timeout: Optional[float] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a task for execution"""

        task_id = f"task_{self.task_counter}"
        self.task_counter += 1

        task = AsyncTask(
            id=task_id,
            coroutine=coroutine,
            priority=priority,
            timeout=timeout,
            metadata=metadata or {}
        )

        try:
            # Use negative priority for PriorityQueue (lower number = higher priority)
            await self.task_queue.put((-priority, task))
            return task_id
        except asyncio.QueueFull:
            raise RuntimeError("Task queue is full")

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""

        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'status': 'running',
                'started_at': task.started_at,
                'runtime': time.time() - (task.started_at or time.time()),
                'metadata': task.metadata
            }

        # Check completed tasks
        for completed in self.completed_tasks:
            if completed.id == task_id:
                return {
                    'status': 'completed' if completed.error is None else 'failed',
                    'started_at': completed.started_at,
                    'completed_at': completed.completed_at,
                    'runtime': (completed.completed_at or time.time()) - (completed.started_at or 0),
                    'result': completed.result if completed.error is None else None,
                    'error': str(completed.error) if completed.error else None,
                    'metadata': completed.metadata
                }

        return None

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a task to complete and return its result"""

        start_time = time.time()

        while True:
            status = await self.get_task_status(task_id)
            if status is None:
                raise ValueError(f"Task {task_id} not found")

            if status['status'] in ['completed', 'failed']:
                if status['status'] == 'failed':
                    raise RuntimeError(f"Task {task_id} failed: {status['error']}")
                return status['result']

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

            await asyncio.sleep(0.1)  # Brief pause before checking again

    async def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics"""
        return {
            'running': self.running,
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'total_submitted': self.task_counter,
            'success_rate': self.success_count / max(self.task_counter, 1),
            'error_rate': self.error_count / max(self.task_counter, 1),
            'avg_queue_wait_time': self._calculate_avg_wait_time()
        }

    async def _worker_loop(self, worker_id: int):
        """Worker loop for processing tasks"""
        print(f"Worker {worker_id} started")

        while self.running:
            try:
                # Get task from queue
                priority, task = await self.task_queue.get()

                # Mark task as started
                task.started_at = time.time()
                self.active_tasks[task.id] = task

                # Execute task with semaphore for concurrency control
                async with self.semaphore:
                    try:
                        if task.timeout:
                            result = await asyncio.wait_for(task.coroutine, timeout=task.timeout)
                        else:
                            result = await task.coroutine

                        task.result = result
                        self.success_count += 1

                    except Exception as e:
                        task.error = e
                        self.error_count += 1
                        print(f"Task {task.id} failed: {e}")

                # Mark task as completed
                task.completed_at = time.time()

                # Move to completed tasks
                self.completed_tasks.append(task)
                del self.active_tasks[task.id]

                # Mark queue task as done
                self.task_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                continue

        print(f"Worker {worker_id} stopped")

    def _calculate_avg_wait_time(self) -> float:
        """Calculate average queue wait time"""
        if not self.completed_tasks:
            return 0.0

        total_wait = 0.0
        count = 0

        for task in self.completed_tasks:
            if task.started_at and task.created_at:
                wait_time = task.started_at - task.created_at
                total_wait += wait_time
                count += 1

        return total_wait / max(count, 1)

class RateLimiter:
    """
    Advanced rate limiter with burst handling, adaptive throttling,
    and consciousness-aware rate adjustment.
    """

    def __init__(self, requests_per_second: float = 10.0, burst_limit: int = 20):
        self.requests_per_second = requests_per_second
        self.burst_limit = burst_limit

        # Request tracking
        self.request_times: deque = deque(maxlen=1000)
        self.burst_count = 0
        self.last_burst_reset = time.time()

        # Adaptive parameters
        self.adaptive_mode = True
        self.success_rate_history = deque(maxlen=100)
        self.adjustment_interval = 60  # seconds

        # Thread safety
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire permission to make a request"""
        async with self.lock:
            current_time = time.time()

            # Reset burst counter if needed
            if current_time - self.last_burst_reset >= 1.0:
                self.burst_count = 0
                self.last_burst_reset = current_time

            # Check burst limit
            if self.burst_count >= self.burst_limit:
                return False

            # Check rate limit
            if len(self.request_times) >= self.requests_per_second:
                oldest_request = self.request_times[0]
                if current_time - oldest_request < 1.0:
                    return False

            # Record request
            self.request_times.append(current_time)
            self.burst_count += 1

            return True

    async def wait_for_slot(self, timeout: float = 30.0) -> bool:
        """Wait for an available slot within timeout"""
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            if await self.acquire():
                return True
            await asyncio.sleep(0.1)

        return False

    def update_success_rate(self, success: bool):
        """Update success rate for adaptive throttling"""
        if self.adaptive_mode:
            self.success_rate_history.append(1.0 if success else 0.0)

            # Adjust rate based on success rate every adjustment_interval
            if len(self.success_rate_history) >= 10:
                avg_success_rate = sum(self.success_rate_history) / len(self.success_rate_history)

                if avg_success_rate < 0.8:  # Low success rate
                    self.requests_per_second = max(1.0, self.requests_per_second * 0.8)
                elif avg_success_rate > 0.95:  # High success rate
                    self.requests_per_second = min(100.0, self.requests_per_second * 1.1)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        current_time = time.time()

        # Calculate current rate
        recent_requests = [t for t in self.request_times if current_time - t < 60.0]
        current_rate = len(recent_requests) / 60.0

        return {
            'target_rps': self.requests_per_second,
            'current_rps': current_rate,
            'burst_count': self.burst_count,
            'burst_limit': self.burst_limit,
            'adaptive_mode': self.adaptive_mode,
            'success_rate': sum(self.success_rate_history) / max(len(self.success_rate_history), 1),
            'queue_size': len(self.request_times)
        }

class HTTPClient:
    """
    Advanced HTTP client with retry logic, connection pooling,
    and consciousness-aware request optimization.
    """

    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries

        # Connection pooling
        self.connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool size
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True
        )

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None

        # Request tracking
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        if not self.session:
            raise RuntimeError("HTTP client not initialized. Use 'async with' context manager.")

        self.request_count += 1

        try:
            async with self.session.request(method, url, **kwargs) as response:
                result = {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'url': str(response.url)
                }

                if response.content_type == 'application/json':
                    result['json'] = await response.json()
                else:
                    result['text'] = await response.text()

                if response.ok:
                    self.success_count += 1
                else:
                    self.error_count += 1

                return result

        except Exception:
            self.error_count += 1
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get HTTP client statistics"""
        total_requests = self.request_count
        success_rate = self.success_count / max(total_requests, 1)
        error_rate = self.error_count / max(total_requests, 1)

        return {
            'total_requests': total_requests,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'error_rate': error_rate,
            'connection_pool_size': self.connector.limit if self.connector else 0
        }

class BatchProcessor:
    """
    Batch processing utility for handling multiple items efficiently
    with configurable batch sizes and parallel processing.
    """

    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.task_manager = AsyncTaskManager(max_concurrent=max_concurrent)

    async def process_batch(self, items: List[Any],
                           processor_func: Callable[[Any], Awaitable[T]],
                           progress_callback: Optional[Callable[[int, int], None]] = None) -> List[T]:
        """Process items in batches with progress tracking"""

        results = []
        total_items = len(items)

        await self.task_manager.start()

        try:
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i + self.batch_size]

                # Submit batch tasks
                task_ids = []
                for item in batch:
                    task_id = await self.task_manager.submit_task(
                        processor_func(item)
                    )
                    task_ids.append(task_id)

                # Wait for batch completion
                batch_results = []
                for task_id in task_ids:
                    result = await self.task_manager.wait_for_task(task_id)
                    batch_results.append(result)

                results.extend(batch_results)

                # Progress callback
                if progress_callback:
                    processed = min(i + len(batch), total_items)
                    progress_callback(processed, total_items)

        finally:
            await self.task_manager.stop()

        return results

    def set_batch_size(self, size: int):
        """Set batch size"""
        self.batch_size = size

    def set_max_concurrent(self, max_concurrent: int):
        """Set maximum concurrent operations"""
        self.max_concurrent = max_concurrent
        self.task_manager = AsyncTaskManager(max_concurrent=max_concurrent)
