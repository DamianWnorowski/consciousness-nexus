//! Benchmarks for the circuit breaker
//!
//! Run with: cargo bench

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use consciousness_circuit_breaker::{
    CircuitBreaker,
    core::config::ConfigBuilder,
    core::state::AtomicStateMachine,
    ultrathought::trampoline::Trampoline,
};
use tokio::runtime::Runtime;

fn state_machine_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_machine");

    let state_machine = AtomicStateMachine::new(5, 3);

    group.bench_function("record_success", |b| {
        b.iter(|| {
            black_box(state_machine.record_success());
        })
    });

    group.bench_function("record_failure", |b| {
        b.iter(|| {
            black_box(state_machine.record_failure());
        })
    });

    group.bench_function("current_state", |b| {
        b.iter(|| {
            black_box(state_machine.current_state());
        })
    });

    group.finish();
}

fn circuit_breaker_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_breaker");
    let rt = Runtime::new().unwrap();

    // Create circuit breaker
    let config = ConfigBuilder::new()
        .name("bench")
        .failure_threshold(5)
        .success_threshold(3)
        .reset_timeout(Duration::from_secs(30))
        .no_bulkhead()
        .build()
        .unwrap();

    let breaker = Arc::new(CircuitBreaker::new(config));

    group.throughput(Throughput::Elements(1));

    group.bench_function("successful_call", |b| {
        b.to_async(&rt).iter(|| {
            let breaker = breaker.clone();
            async move {
                black_box(
                    breaker
                        .call(|| async { Ok::<_, std::io::Error>(42) })
                        .await,
                )
            }
        })
    });

    group.bench_function("get_stats", |b| {
        b.iter(|| {
            black_box(breaker.stats());
        })
    });

    group.finish();
}

fn trampoline_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("trampoline");

    for depth in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("deep_recursion", depth),
            depth,
            |b, &depth| {
                b.iter(|| {
                    fn deep(n: u64) -> Trampoline<u64> {
                        if n == 0 {
                            Trampoline::done(0)
                        } else {
                            Trampoline::cont(move || deep(n - 1).map(|x| x + 1))
                        }
                    }
                    black_box(deep(depth).run())
                })
            },
        );
    }

    group.finish();
}

fn concurrent_access_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");
    let rt = Runtime::new().unwrap();

    let config = ConfigBuilder::new()
        .name("concurrent-bench")
        .failure_threshold(100)
        .no_bulkhead()
        .build()
        .unwrap();

    let breaker = Arc::new(CircuitBreaker::new(config));

    for num_tasks in [1, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("parallel_calls", num_tasks),
            num_tasks,
            |b, &num_tasks| {
                b.to_async(&rt).iter(|| {
                    let breaker = breaker.clone();
                    async move {
                        let mut handles = Vec::new();
                        for _ in 0..num_tasks {
                            let b = breaker.clone();
                            handles.push(tokio::spawn(async move {
                                for _ in 0..100 {
                                    let _ = b.call(|| async { Ok::<_, std::io::Error>(()) }).await;
                                }
                            }));
                        }
                        for h in handles {
                            let _ = h.await;
                        }
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    state_machine_benchmark,
    circuit_breaker_benchmark,
    trampoline_benchmark,
    concurrent_access_benchmark,
);

criterion_main!(benches);
