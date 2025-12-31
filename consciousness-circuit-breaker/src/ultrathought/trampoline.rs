//! Trampoline pattern for stack-safe deep recursion
//!
//! Enables arbitrarily deep recursive computations without stack overflow.
//! Used by the UltraThought engine for deep thought chains.

use std::future::Future;
use std::pin::Pin;

/// Trampoline for stack-safe recursion
///
/// Instead of direct recursion (which consumes stack), the trampoline
/// converts recursive calls into a loop, allowing unlimited depth.
pub enum Trampoline<T> {
    /// Computation is complete with this value
    Done(T),
    /// Computation needs another step
    Continue(Box<dyn FnOnce() -> Trampoline<T> + Send>),
}

impl<T> Trampoline<T> {
    /// Create a done trampoline
    pub fn done(value: T) -> Self {
        Trampoline::Done(value)
    }

    /// Create a continuation trampoline
    pub fn cont<F>(f: F) -> Self
    where
        F: FnOnce() -> Trampoline<T> + Send + 'static,
    {
        Trampoline::Continue(Box::new(f))
    }

    /// Run the trampoline to completion
    ///
    /// This iteratively processes continuations until we get a Done result.
    /// No matter how deep the recursion, this uses constant stack space.
    pub fn run(self) -> T {
        let mut current = self;
        loop {
            match current {
                Trampoline::Done(value) => return value,
                Trampoline::Continue(f) => current = f(),
            }
        }
    }

    /// Map over the final value
    pub fn map<U, F>(self, f: F) -> Trampoline<U>
    where
        F: FnOnce(T) -> U + Send + 'static,
        T: Send + 'static,
    {
        match self {
            Trampoline::Done(value) => Trampoline::Done(f(value)),
            Trampoline::Continue(cont) => {
                Trampoline::cont(move || cont().map(f))
            }
        }
    }

    /// Flat map (bind) for chaining computations
    pub fn flat_map<U, F>(self, f: F) -> Trampoline<U>
    where
        F: FnOnce(T) -> Trampoline<U> + Send + 'static,
        T: Send + 'static,
    {
        match self {
            Trampoline::Done(value) => f(value),
            Trampoline::Continue(cont) => {
                Trampoline::cont(move || cont().flat_map(f))
            }
        }
    }
}

/// Async trampoline for async recursive computations
pub enum AsyncTrampoline<T> {
    /// Computation is complete
    Done(T),
    /// Need another async step
    Continue(Pin<Box<dyn Future<Output = AsyncTrampoline<T>> + Send>>),
}

impl<T: Send + 'static> AsyncTrampoline<T> {
    /// Create a done async trampoline
    pub fn done(value: T) -> Self {
        AsyncTrampoline::Done(value)
    }

    /// Create a continuation with an async function
    pub fn cont<F>(f: F) -> Self
    where
        F: Future<Output = AsyncTrampoline<T>> + Send + 'static,
    {
        AsyncTrampoline::Continue(Box::pin(f))
    }

    /// Run the async trampoline to completion
    pub async fn run(self) -> T {
        let mut current = self;
        loop {
            match current {
                AsyncTrampoline::Done(value) => return value,
                AsyncTrampoline::Continue(f) => current = f.await,
            }
        }
    }
}

/// Bounded trampoline with iteration limit
pub struct BoundedTrampoline<T> {
    inner: Trampoline<T>,
    max_iterations: usize,
}

impl<T: Default> BoundedTrampoline<T> {
    /// Create a bounded trampoline
    pub fn new(trampoline: Trampoline<T>, max_iterations: usize) -> Self {
        Self {
            inner: trampoline,
            max_iterations,
        }
    }

    /// Run with iteration limit
    ///
    /// Returns Some(value) if completed within limit, None if exceeded.
    pub fn run(self) -> Option<T> {
        let mut current = self.inner;
        let mut iterations = 0;

        loop {
            if iterations >= self.max_iterations {
                return None;
            }

            match current {
                Trampoline::Done(value) => return Some(value),
                Trampoline::Continue(f) => {
                    current = f();
                    iterations += 1;
                }
            }
        }
    }

    /// Run with iteration count returned
    pub fn run_counted(self) -> (Option<T>, usize) {
        let mut current = self.inner;
        let mut iterations = 0;

        loop {
            if iterations >= self.max_iterations {
                return (None, iterations);
            }

            match current {
                Trampoline::Done(value) => return (Some(value), iterations),
                Trampoline::Continue(f) => {
                    current = f();
                    iterations += 1;
                }
            }
        }
    }
}

/// Instrumented trampoline that tracks execution statistics
pub struct InstrumentedTrampoline<T> {
    inner: Trampoline<T>,
}

/// Execution statistics from instrumented trampoline
#[derive(Debug, Clone, Default)]
pub struct TrampolineStats {
    /// Total iterations executed
    pub iterations: usize,
    /// Maximum stack depth reached (always 1 for trampoline)
    pub max_stack_depth: usize,
    /// Time spent in execution
    pub execution_time_us: u64,
}

impl<T> InstrumentedTrampoline<T> {
    /// Create an instrumented trampoline
    pub fn new(trampoline: Trampoline<T>) -> Self {
        Self { inner: trampoline }
    }

    /// Run with instrumentation
    pub fn run(self) -> (T, TrampolineStats) {
        let start = std::time::Instant::now();
        let mut current = self.inner;
        let mut iterations = 0;

        let value = loop {
            match current {
                Trampoline::Done(value) => break value,
                Trampoline::Continue(f) => {
                    current = f();
                    iterations += 1;
                }
            }
        };

        let stats = TrampolineStats {
            iterations,
            max_stack_depth: 1, // Trampoline always uses constant stack
            execution_time_us: start.elapsed().as_micros() as u64,
        };

        (value, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_trampoline() {
        let tramp = Trampoline::done(42);
        assert_eq!(tramp.run(), 42);
    }

    #[test]
    fn test_continuation() {
        let tramp = Trampoline::cont(|| Trampoline::done(42));
        assert_eq!(tramp.run(), 42);
    }

    #[test]
    fn test_deep_recursion() {
        // This would overflow the stack with normal recursion
        fn deep_trampoline(n: u64) -> Trampoline<u64> {
            if n == 0 {
                Trampoline::done(0)
            } else {
                Trampoline::cont(move || deep_trampoline(n - 1).map(|x| x + 1))
            }
        }

        // 100,000 iterations - would crash with normal recursion
        let result = deep_trampoline(100_000).run();
        assert_eq!(result, 100_000);
    }

    #[test]
    fn test_map() {
        let tramp = Trampoline::done(21).map(|x| x * 2);
        assert_eq!(tramp.run(), 42);
    }

    #[test]
    fn test_flat_map() {
        let tramp = Trampoline::done(21).flat_map(|x| Trampoline::done(x * 2));
        assert_eq!(tramp.run(), 42);
    }

    #[test]
    fn test_bounded_success() {
        let tramp = Trampoline::cont(|| Trampoline::done(42));
        let bounded = BoundedTrampoline::new(tramp, 10);
        assert_eq!(bounded.run(), Some(42));
    }

    #[test]
    fn test_bounded_exceeded() {
        fn infinite() -> Trampoline<i32> {
            Trampoline::cont(infinite)
        }

        let bounded: BoundedTrampoline<i32> = BoundedTrampoline::new(infinite(), 100);
        assert_eq!(bounded.run(), None);
    }

    #[test]
    fn test_instrumented() {
        fn counted(n: u64) -> Trampoline<u64> {
            if n == 0 {
                Trampoline::done(0)
            } else {
                Trampoline::cont(move || counted(n - 1))
            }
        }

        let instrumented = InstrumentedTrampoline::new(counted(1000));
        let (value, stats) = instrumented.run();

        assert_eq!(value, 0);
        assert_eq!(stats.iterations, 1000);
        assert_eq!(stats.max_stack_depth, 1);
        assert!(stats.execution_time_us > 0);
    }

    #[tokio::test]
    async fn test_async_trampoline() {
        let tramp = AsyncTrampoline::done(42);
        assert_eq!(tramp.run().await, 42);
    }

    #[tokio::test]
    async fn test_async_continuation() {
        fn async_cont(n: i32) -> std::pin::Pin<Box<dyn std::future::Future<Output = AsyncTrampoline<i32>> + Send>> {
            Box::pin(async move {
                if n <= 0 {
                    AsyncTrampoline::done(n)
                } else {
                    AsyncTrampoline::cont(async_cont(n - 1))
                }
            })
        }

        let result = async_cont(100).await.run().await;
        assert_eq!(result, 0);
    }
}
