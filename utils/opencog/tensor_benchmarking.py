"""
Tensor Benchmarking Framework for Phase 3 Neural-Symbolic Synthesis.

This module implements comprehensive benchmarking for neural-symbolic operations,
performance analysis, and memory efficiency testing.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
import json
import statistics
from abc import ABC, abstractmethod

# Import our GGML kernels
from utils.opencog.ggml_kernels import (
    GGMLKernelManager, KernelOperation, KernelType, 
    NeuralSymbolicSignature, create_neural_symbolic_operation
)


class BenchmarkType(Enum):
    """Types of benchmarks to run."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    SCALABILITY = "scalability"
    COMPARATIVE = "comparative"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    benchmark_type: BenchmarkType
    name: str
    description: str
    iterations: int = 10
    warmup_iterations: int = 3
    timeout_seconds: float = 30.0
    memory_tracking: bool = True
    detailed_logging: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    success: bool
    execution_times: List[float]
    memory_usage: List[int]  # Memory usage in bytes
    throughput: float  # Operations per second
    accuracy_score: float  # Accuracy metric if applicable
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def avg_execution_time(self) -> float:
        """Average execution time."""
        return statistics.mean(self.execution_times) if self.execution_times else 0.0
    
    @property
    def std_execution_time(self) -> float:
        """Standard deviation of execution times."""
        return statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0.0
    
    @property
    def min_execution_time(self) -> float:
        """Minimum execution time."""
        return min(self.execution_times) if self.execution_times else 0.0
    
    @property
    def max_execution_time(self) -> float:
        """Maximum execution time."""
        return max(self.execution_times) if self.execution_times else 0.0
    
    @property
    def avg_memory_usage(self) -> float:
        """Average memory usage in MB."""
        return statistics.mean(self.memory_usage) / (1024 * 1024) if self.memory_usage else 0.0


class BenchmarkSuite(ABC):
    """Abstract base class for benchmark suites."""
    
    def __init__(self, name: str):
        self.name = name
        self.benchmarks: List[BenchmarkConfig] = []
        self.results: List[BenchmarkResult] = []
    
    @abstractmethod
    def setup_benchmarks(self) -> List[BenchmarkConfig]:
        """Setup benchmark configurations."""
        pass
    
    @abstractmethod
    async def run_benchmark(self, config: BenchmarkConfig, kernel_manager: GGMLKernelManager) -> BenchmarkResult:
        """Run a single benchmark."""
        pass


class NeuralSymbolicBenchmarkSuite(BenchmarkSuite):
    """Benchmark suite for neural-symbolic operations."""
    
    def __init__(self):
        super().__init__("neural_symbolic_benchmarks")
        self.setup_benchmarks()
    
    def setup_benchmarks(self) -> List[BenchmarkConfig]:
        """Setup neural-symbolic benchmark configurations."""
        self.benchmarks = [
            BenchmarkConfig(
                benchmark_type=BenchmarkType.EXECUTION_TIME,
                name="symbolic_reasoning_time",
                description="Measure execution time of symbolic reasoning operations",
                iterations=20,
                parameters={
                    'operation_type': KernelType.SYMBOLIC_REASONING,
                    'symbols_count': [5, 10, 20, 50],
                    'rules_count': [2, 5, 10, 15]
                }
            ),
            BenchmarkConfig(
                benchmark_type=BenchmarkType.EXECUTION_TIME,
                name="neural_embedding_time",
                description="Measure execution time of neural embedding operations",
                iterations=15,
                parameters={
                    'operation_type': KernelType.NEURAL_EMBEDDING,
                    'text_lengths': [10, 50, 100, 500],
                    'embedding_dims': [128, 256, 512, 768]
                }
            ),
            BenchmarkConfig(
                benchmark_type=BenchmarkType.EXECUTION_TIME,
                name="attention_fusion_time",
                description="Measure execution time of attention fusion operations",
                iterations=15,
                parameters={
                    'operation_type': KernelType.ATTENTION_FUSION,
                    'input_sizes': [10, 50, 100, 200],
                    'attention_heads': [1, 4, 8, 12]
                }
            ),
            BenchmarkConfig(
                benchmark_type=BenchmarkType.THROUGHPUT,
                name="overall_throughput",
                description="Measure overall throughput of neural-symbolic operations",
                iterations=10,
                parameters={
                    'batch_sizes': [1, 5, 10, 20],
                    'mixed_operations': True
                }
            ),
            BenchmarkConfig(
                benchmark_type=BenchmarkType.MEMORY_USAGE,
                name="memory_efficiency",
                description="Measure memory usage of neural-symbolic operations",
                iterations=10,
                memory_tracking=True,
                parameters={
                    'operation_scales': ['small', 'medium', 'large'],
                    'track_peak_memory': True
                }
            ),
            BenchmarkConfig(
                benchmark_type=BenchmarkType.SCALABILITY,
                name="scalability_test",
                description="Test scalability with increasing operation complexity",
                iterations=5,
                parameters={
                    'complexity_levels': [1, 2, 4, 8, 16],
                    'measure_degradation': True
                }
            )
        ]
        return self.benchmarks
    
    async def run_benchmark(self, config: BenchmarkConfig, kernel_manager: GGMLKernelManager) -> BenchmarkResult:
        """Run a single neural-symbolic benchmark."""
        execution_times = []
        memory_usage = []
        successful_ops = 0
        total_ops = 0
        
        try:
            # Warmup iterations
            for _ in range(config.warmup_iterations):
                await self._run_single_operation(config, kernel_manager)
            
            # Actual benchmark iterations
            for iteration in range(config.iterations):
                start_memory = self._get_memory_usage() if config.memory_tracking else 0
                start_time = time.time()
                
                success = await self._run_single_operation(config, kernel_manager)
                
                end_time = time.time()
                end_memory = self._get_memory_usage() if config.memory_tracking else 0
                
                execution_times.append(end_time - start_time)
                memory_usage.append(max(0, end_memory - start_memory))
                
                if success:
                    successful_ops += 1
                total_ops += 1
            
            # Calculate throughput
            total_time = sum(execution_times)
            throughput = total_ops / total_time if total_time > 0 else 0.0
            
            # Calculate accuracy
            accuracy_score = successful_ops / total_ops if total_ops > 0 else 0.0
            
            return BenchmarkResult(
                config=config,
                success=True,
                execution_times=execution_times,
                memory_usage=memory_usage,
                throughput=throughput,
                accuracy_score=accuracy_score,
                metadata={
                    'successful_operations': successful_ops,
                    'total_operations': total_ops,
                    'warmup_iterations': config.warmup_iterations
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                config=config,
                success=False,
                execution_times=execution_times,
                memory_usage=memory_usage,
                throughput=0.0,
                accuracy_score=0.0,
                error_message=str(e)
            )
    
    async def _run_single_operation(self, config: BenchmarkConfig, kernel_manager: GGMLKernelManager) -> bool:
        """Run a single operation based on benchmark configuration."""
        try:
            if config.benchmark_type in [BenchmarkType.EXECUTION_TIME, BenchmarkType.THROUGHPUT]:
                operation = self._create_test_operation(config)
                await kernel_manager.execute_operation(operation)
                return True
                
            elif config.benchmark_type == BenchmarkType.MEMORY_USAGE:
                # Run multiple operations to test memory usage
                operations = [self._create_test_operation(config) for _ in range(5)]
                for op in operations:
                    await kernel_manager.execute_operation(op)
                return True
                
            elif config.benchmark_type == BenchmarkType.SCALABILITY:
                # Run operations with increasing complexity
                for complexity in config.parameters.get('complexity_levels', [1]):
                    operation = self._create_scaled_operation(config, complexity)
                    await kernel_manager.execute_operation(operation)
                return True
                
        except Exception:
            return False
        
        return True
    
    def _create_test_operation(self, config: BenchmarkConfig) -> KernelOperation:
        """Create a test operation based on benchmark configuration."""
        op_type = config.parameters.get('operation_type', KernelType.SYMBOLIC_REASONING)
        
        if op_type == KernelType.SYMBOLIC_REASONING:
            symbols_count = config.parameters.get('symbols_count', [10])[0]
            rules_count = config.parameters.get('rules_count', [5])[0]
            
            symbols = [f'symbol_{i}' for i in range(symbols_count)]
            rules = []
            for i in range(rules_count):
                conditions = symbols[:min(2, len(symbols))]
                conclusion = f'derived_{i}'
                rules.append({'conditions': conditions, 'conclusion': conclusion})
            
            return create_neural_symbolic_operation(
                f"benchmark_reasoning_{time.time()}",
                KernelType.SYMBOLIC_REASONING,
                atoms_strength=0.8,
                confidence=0.9,
                features=0.7,
                parameters={'symbols': symbols, 'rules': rules}
            )
            
        elif op_type == KernelType.NEURAL_EMBEDDING:
            text_length = config.parameters.get('text_lengths', [100])[0]
            embedding_dim = config.parameters.get('embedding_dims', [256])[0]
            
            text_inputs = [f'benchmark_text_{i}' * (text_length // 15) for i in range(3)]
            
            return create_neural_symbolic_operation(
                f"benchmark_embedding_{time.time()}",
                KernelType.NEURAL_EMBEDDING,
                atoms_strength=0.6,
                confidence=0.8,
                features=0.9,
                parameters={'text_inputs': text_inputs, 'embedding_dim': embedding_dim}
            )
            
        elif op_type == KernelType.ATTENTION_FUSION:
            input_size = config.parameters.get('input_sizes', [50])[0]
            
            neural_inputs = [0.5 + 0.1 * i for i in range(input_size)]
            symbolic_inputs = [f'atom_{i}' for i in range(input_size)]
            
            return create_neural_symbolic_operation(
                f"benchmark_fusion_{time.time()}",
                KernelType.ATTENTION_FUSION,
                atoms_strength=0.7,
                confidence=0.85,
                features=0.8,
                parameters={'neural_inputs': neural_inputs, 'symbolic_inputs': symbolic_inputs}
            )
        
        # Default fallback
        return create_neural_symbolic_operation(
            f"benchmark_default_{time.time()}",
            KernelType.SYMBOLIC_REASONING,
            atoms_strength=0.7,
            confidence=0.8,
            features=0.6,
            parameters={'symbols': ['test'], 'rules': []}
        )
    
    def _create_scaled_operation(self, config: BenchmarkConfig, complexity: int) -> KernelOperation:
        """Create an operation with scaled complexity."""
        base_symbols = 5 * complexity
        base_rules = 2 * complexity
        
        symbols = [f'scaled_symbol_{i}' for i in range(base_symbols)]
        rules = []
        for i in range(base_rules):
            conditions = symbols[:min(complexity, len(symbols))]
            conclusion = f'scaled_derived_{i}'
            rules.append({'conditions': conditions, 'conclusion': conclusion})
        
        return create_neural_symbolic_operation(
            f"benchmark_scaled_{complexity}_{time.time()}",
            KernelType.SYMBOLIC_REASONING,
            atoms_strength=0.8,
            confidence=0.9,
            features=0.7,
            parameters={'symbols': symbols, 'rules': rules}
        )
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # Fallback: simple approximation based on object count
            import gc
            return len(gc.get_objects()) * 100  # Rough approximation


class TensorBenchmarkManager:
    """Manages tensor benchmarking operations."""
    
    def __init__(self, kernel_manager: GGMLKernelManager):
        self.kernel_manager = kernel_manager
        self.benchmark_suites: List[BenchmarkSuite] = []
        self.results_history: List[Dict[str, Any]] = []
        
        # Register default benchmark suites
        self.register_suite(NeuralSymbolicBenchmarkSuite())
    
    def register_suite(self, suite: BenchmarkSuite):
        """Register a benchmark suite."""
        self.benchmark_suites.append(suite)
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all registered benchmark suites."""
        all_results = {}
        start_time = time.time()
        
        print("🎯 Starting Tensor Benchmarking Suite")
        print("=" * 50)
        
        for suite in self.benchmark_suites:
            print(f"\n📊 Running benchmark suite: {suite.name}")
            suite_results = await self.run_benchmark_suite(suite)
            all_results[suite.name] = suite_results
            
            # Print summary for this suite
            self._print_suite_summary(suite.name, suite_results)
        
        total_time = time.time() - start_time
        
        # Create comprehensive report
        report = {
            'timestamp': time.time(),
            'total_execution_time': total_time,
            'benchmark_suites': all_results,
            'summary': self._generate_summary(all_results),
            'kernel_performance': self.kernel_manager.get_performance_stats()
        }
        
        self.results_history.append(report)
        
        print(f"\n✅ All benchmarks completed in {total_time:.2f}s")
        return report
    
    async def run_benchmark_suite(self, suite: BenchmarkSuite) -> List[BenchmarkResult]:
        """Run a single benchmark suite."""
        results = []
        
        for config in suite.benchmarks:
            print(f"  🔄 Running: {config.name}")
            
            try:
                result = await asyncio.wait_for(
                    suite.run_benchmark(config, self.kernel_manager),
                    timeout=config.timeout_seconds
                )
                results.append(result)
                
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"    {status} - {result.avg_execution_time:.4f}s avg")
                
            except asyncio.TimeoutError:
                result = BenchmarkResult(
                    config=config,
                    success=False,
                    execution_times=[],
                    memory_usage=[],
                    throughput=0.0,
                    accuracy_score=0.0,
                    error_message=f"Timeout after {config.timeout_seconds}s"
                )
                results.append(result)
                print(f"    ⏰ TIMEOUT - {config.timeout_seconds}s")
            
            except Exception as e:
                result = BenchmarkResult(
                    config=config,
                    success=False,
                    execution_times=[],
                    memory_usage=[],
                    throughput=0.0,
                    accuracy_score=0.0,
                    error_message=str(e)
                )
                results.append(result)
                print(f"    ❌ ERROR - {str(e)[:50]}")
        
        suite.results = results
        return results
    
    def _print_suite_summary(self, suite_name: str, results: List[BenchmarkResult]):
        """Print a summary of benchmark suite results."""
        successful = sum(1 for r in results if r.success)
        total = len(results)
        
        print(f"\n📈 {suite_name} Summary:")
        print(f"   Benchmarks: {successful}/{total} passed")
        
        if successful > 0:
            avg_times = [r.avg_execution_time for r in results if r.success]
            avg_throughput = [r.throughput for r in results if r.success and r.throughput > 0]
            avg_accuracy = [r.accuracy_score for r in results if r.success]
            
            if avg_times:
                print(f"   Avg execution time: {statistics.mean(avg_times):.4f}s")
            if avg_throughput:
                print(f"   Avg throughput: {statistics.mean(avg_throughput):.2f} ops/s")
            if avg_accuracy:
                print(f"   Avg accuracy: {statistics.mean(avg_accuracy):.2%}")
    
    def _generate_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of all benchmark results."""
        total_benchmarks = 0
        successful_benchmarks = 0
        all_execution_times = []
        all_throughputs = []
        all_accuracies = []
        
        for suite_name, results in all_results.items():
            for result in results:
                total_benchmarks += 1
                if result.success:
                    successful_benchmarks += 1
                    all_execution_times.append(result.avg_execution_time)
                    if result.throughput > 0:
                        all_throughputs.append(result.throughput)
                    all_accuracies.append(result.accuracy_score)
        
        summary = {
            'total_benchmarks': total_benchmarks,
            'successful_benchmarks': successful_benchmarks,
            'success_rate': successful_benchmarks / total_benchmarks if total_benchmarks > 0 else 0.0,
            'overall_avg_execution_time': statistics.mean(all_execution_times) if all_execution_times else 0.0,
            'overall_avg_throughput': statistics.mean(all_throughputs) if all_throughputs else 0.0,
            'overall_avg_accuracy': statistics.mean(all_accuracies) if all_accuracies else 0.0,
            'performance_grade': self._calculate_performance_grade(all_execution_times, all_throughputs, all_accuracies)
        }
        
        return summary
    
    def _calculate_performance_grade(self, execution_times: List[float], throughputs: List[float], accuracies: List[float]) -> str:
        """Calculate an overall performance grade."""
        score = 0
        
        # Execution time score (lower is better)
        if execution_times:
            avg_time = statistics.mean(execution_times)
            if avg_time < 0.001:
                score += 30
            elif avg_time < 0.01:
                score += 25
            elif avg_time < 0.1:
                score += 20
            elif avg_time < 1.0:
                score += 15
            else:
                score += 10
        
        # Throughput score (higher is better)
        if throughputs:
            avg_throughput = statistics.mean(throughputs)
            if avg_throughput > 1000:
                score += 30
            elif avg_throughput > 100:
                score += 25
            elif avg_throughput > 10:
                score += 20
            elif avg_throughput > 1:
                score += 15
            else:
                score += 10
        
        # Accuracy score
        if accuracies:
            avg_accuracy = statistics.mean(accuracies)
            score += int(avg_accuracy * 40)  # Max 40 points for 100% accuracy
        
        # Convert to letter grade
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        else:
            return "F"
    
    def export_results(self, filepath: str):
        """Export benchmark results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)
    
    def get_comparative_analysis(self) -> Dict[str, Any]:
        """Get comparative analysis of benchmark results over time."""
        if len(self.results_history) < 2:
            return {"message": "Need at least 2 benchmark runs for comparison"}
        
        current = self.results_history[-1]
        previous = self.results_history[-2]
        
        current_summary = current['summary']
        previous_summary = previous['summary']
        
        comparison = {
            'execution_time_change': current_summary['overall_avg_execution_time'] - previous_summary['overall_avg_execution_time'],
            'throughput_change': current_summary['overall_avg_throughput'] - previous_summary['overall_avg_throughput'],
            'accuracy_change': current_summary['overall_avg_accuracy'] - previous_summary['overall_avg_accuracy'],
            'success_rate_change': current_summary['success_rate'] - previous_summary['success_rate'],
            'grade_change': f"{previous_summary['performance_grade']} → {current_summary['performance_grade']}"
        }
        
        return comparison


async def demo_tensor_benchmarking():
    """Demonstrate tensor benchmarking capabilities."""
    print("🎯 Tensor Benchmarking Demo - Phase 3")
    print("=" * 50)
    
    # Initialize kernel manager and benchmarking
    kernel_manager = GGMLKernelManager()
    benchmark_manager = TensorBenchmarkManager(kernel_manager)
    
    # Run all benchmarks
    results = await benchmark_manager.run_all_benchmarks()
    
    print("\n📊 Final Results Summary:")
    summary = results['summary']
    print(f"   Performance Grade: {summary['performance_grade']}")
    print(f"   Success Rate: {summary['success_rate']:.1%}")
    print(f"   Avg Execution Time: {summary['overall_avg_execution_time']:.4f}s")
    print(f"   Avg Throughput: {summary['overall_avg_throughput']:.2f} ops/s")
    print(f"   Avg Accuracy: {summary['overall_avg_accuracy']:.1%}")
    
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_tensor_benchmarking())