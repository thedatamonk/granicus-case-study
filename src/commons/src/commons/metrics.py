import psutil
from prometheus_client import Counter, Gauge, Histogram


class ServiceMetrics:
    def __init__(self, service_name: str) -> None:
        self.service_name = service_name

        # 1. Request counter (for RPS calculation)
        self.request_counter = Counter(
            "http_requests_total",
            "Total HTTP Requests",
            ["service", "method", "endpoint", "status"]
        )

        self.response_time = Histogram(
            'http_request_duration_seconds',
            'HTTP request latency',
            ['service', 'method', 'endpoint'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )

        # 3. CPU utilization gauge
        self.cpu_usage = Gauge(
            'process_cpu_usage_percent',
            'Current CPU usage percentage',
            ['service']
        )

        # 4. Memory utilization gauge
        self.memory_usage = Gauge(
            'process_memory_usage_bytes',
            'Current memory usage in bytes',
            ['service']
        )

    def track_request(self, method: str, endpoint: str, status: int):
        """Increment request counter"""
        self.request_counter.labels(
            service=self.service_name,
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()

    def track_latency(self, method: str, endpoint: str, duration: float):
        self.response_time.labels(
            service=self.service_name,
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def update_system_metrics(self):
        """Update CPU and memory metrics"""
        # CPU percentage for current process
        cpu_percent = psutil.Process().cpu_percent(interval=0.1)
        self.cpu_usage.labels(service=self.service_name).set(cpu_percent)
        
        # Memory in bytes for current process
        memory_bytes = psutil.Process().memory_info().rss
        self.memory_usage.labels(service=self.service_name).set(memory_bytes)
