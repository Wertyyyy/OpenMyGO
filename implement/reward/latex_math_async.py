import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
from typing import Tuple, Optional
import asyncio

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from data_service.typing.grpo_data import GRPOData


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Helper function for process isolation (must be at module level for pickling)
# ============================================================================

def _verify_in_process(completion_text: str, answer: str) -> Tuple[float, str]:
    """
    Worker function that runs in a separate process.
    This must be a module-level function to be picklable.
    
    Args:
        completion_text: The completion text to verify
        answer: The ground truth answer
        
    Returns:
        tuple: (reward_score, debug_message)
    """
    try:
        completion = parse(
            completion_text,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        answer_parsed = parse(answer)

        is_correct = verify(answer_parsed, completion)
        return float(is_correct), f"Answer: {answer_parsed}, Completion: {completion}"
    except Exception as e:
        return 0.0, f"Error in process: {e}"


# ============================================================================
# Async Process-Isolated Version
# ============================================================================

class RewardImpl:
    """
    LaTeX math verification reward function with async + process isolation.
    
    This version provides:
    - Async interface for integration with async code
    - Process-level isolation for each verification
    - Configurable timeout with guaranteed termination
    - Process pool reuse for better performance
    - Graceful degradation on timeout/errors
    
    Note: Despite being async, this class is named RewardImpl to maintain
    consistency with the module loading convention in data_server.py.
    """

    def __init__(
        self,
        timeout: float = 6.0,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize async LaTeX math reward function with process isolation.

        Args:
            timeout: Maximum time (in seconds) to wait for verification.
                    Default: 6.0 seconds
            max_workers: Maximum number of worker processes in the pool.
                        If None, defaults to min(4, cpu_count)
        """
        self.timeout = timeout
        
        # Limit max_workers to avoid too many processes
        if max_workers is None:
            max_workers = min(4, mp.cpu_count() or 1)
        
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        logger.info(
            f"Initialized LaTeXMathReward (Async + Process Isolated) - "
            f"timeout={self.timeout}s, max_workers={self.max_workers}"
        )

    async def __call__(self, grpo_data: GRPOData) -> Tuple[float, str]:
        """
        Async verification with process isolation.

        Args:
            grpo_data: The GRPOData object containing conversation and solution

        Returns:
            tuple: (reward_score, debug_message)
                - reward_score: 1.0 if correct, 0.0 if incorrect or timeout
                - debug_message: String describing the verification result
        """
        try:
            completion_text = grpo_data.conversation.messages[-1].content
            answer = grpo_data.solution

            # Get current event loop
            loop = asyncio.get_event_loop()
            
            # Submit to process pool via event loop
            try:
                reward_score, debug_msg = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        _verify_in_process,
                        completion_text,
                        answer
                    ),
                    timeout=self.timeout
                )
                return reward_score, debug_msg
            except asyncio.TimeoutError:
                logger.warning(
                    f"Async verification timeout after {self.timeout}s for "
                    f"prompt_idx={grpo_data.prompt_idx}, response_idx={grpo_data.response_idx}"
                )
                return 0.0, f"Timeout after {self.timeout}s"
                
        except Exception as e:
            logger.error(f"Error in async LaTeX math reward: {e}")
            return 0.0, f"Error: {e}"

    def __del__(self):
        """Clean up process pool on deletion."""
        if hasattr(self, 'executor'):
            try:
                self.executor.shutdown(wait=False)
            except Exception as e:
                logger.error(f"Error shutting down executor: {e}")


# ============================================================================
# Performance Testing
# ============================================================================

if __name__ == "__main__":
    import unittest
    from data_service.typing.message import Conversation, Message

    def create_test_grpo_data(completion: str, solution: str) -> GRPOData:
        """Helper function to create GRPOData for testing."""
        conversation = Conversation(messages=[
            Message(role="user", content="Test question"),
            Message(role="assistant", content=completion)
        ])
        return GRPOData(
            prompt_idx=0,
            response_idx=0,
            conversation=conversation,
            solution=solution,
            stop_reason="stop",
            prompt_token_ids=[1, 2, 3],
            response_token_ids=[4, 5, 6],
        )

    # Test cases for correctness
    class TestLatexMathAsyncCorrectness(unittest.TestCase):
        """Test correctness of async implementation."""

        def setUp(self):
            """Set up test reward instances"""
            self.async_reward = RewardImpl(timeout=5.0)

        async def _test_async(self, completion: str, solution: str, expected_score: float):
            """Helper to test async reward"""
            grpo_data = create_test_grpo_data(completion, solution)
            score, msg = await self.async_reward(grpo_data)
            self.assertEqual(score, expected_score)
            return score, msg

        def test_correct_simple_boxed(self):
            """Test simple boxed answer"""
            asyncio.run(self._test_async('The answer is \\boxed{42}', '\\boxed{42}', 1.0))

        def test_correct_fraction(self):
            """Test fraction answer"""
            asyncio.run(self._test_async(
                'The answer is \\boxed{\\frac{1}{2}}',
                '\\boxed{\\frac{1}{2}}',
                1.0
            ))

        def test_wrong_answer(self):
            """Test wrong answer"""
            asyncio.run(self._test_async('The answer is \\boxed{42}', '\\boxed{100}', 0.0))

        def test_complex_expression(self):
            """Test complex mathematical expression"""
            asyncio.run(self._test_async(
                'The answer is \\boxed{2\\sqrt{3}}',
                '\\boxed{2\\sqrt{3}}',
                1.0
            ))

        def test_equivalent_forms(self):
            """Test equivalent mathematical forms"""
            asyncio.run(self._test_async('The answer is \\boxed{0.5}', '\\boxed{\\frac{1}{2}}', 1.0))

        def tearDown(self):
            """Clean up resources"""
            if hasattr(self, 'async_reward'):
                del self.async_reward

    # Performance comparison tests
    class TestLatexMathAsyncPerformance(unittest.TestCase):
        """Test performance with concurrent execution."""

        def setUp(self):
            """Set up test data"""
            self.test_cases = [
                ('\\boxed{42}', '\\boxed{42}'),
                ('\\boxed{100}', '\\boxed{100}'),
                ('\\boxed{\\frac{1}{2}}', '\\boxed{\\frac{1}{2}}'),
                ('\\boxed{2\\sqrt{3}}', '\\boxed{2\\sqrt{3}}'),
                ('\\boxed{x^2 + y^2}', '\\boxed{x^2 + y^2}'),
                ('\\boxed{0}', '\\boxed{0}'),
                ('\\boxed{-1}', '\\boxed{-1}'),
                ('\\boxed{\\pi}', '\\boxed{\\pi}'),
                ('\\boxed{e^{i\\pi}}', '\\boxed{e^{i\\pi}}'),
                ('\\boxed{\\infty}', '\\boxed{\\infty}'),
            ]
            
            # Create more test cases by repeating
            self.test_cases = self.test_cases * 10  # 100 test cases

        def test_sequential_execution(self):
            """Test sequential execution (like in data_server)"""
            async def run_test():
                reward_impl = RewardImpl(timeout=6.0, max_workers=4)
                
                start_time = time.time()
                results = []
                
                for completion, solution in self.test_cases:
                    grpo_data = create_test_grpo_data(completion, solution)
                    score, msg = await reward_impl(grpo_data)
                    results.append(score)
                
                elapsed_time = time.time() - start_time
                
                print(f"\n{'='*70}")
                print("Async Sequential Execution Performance:")
                print(f"  Total cases: {len(self.test_cases)}")
                print(f"  Total time: {elapsed_time:.4f}s")
                print(f"  Average time per case: {elapsed_time/len(self.test_cases)*1000:.2f}ms")
                print(f"  Throughput: {len(self.test_cases)/elapsed_time:.2f} cases/sec")
                print(f"  Correct results: {sum(results)}/{len(results)}")
                print(f"{'='*70}")
                
                del reward_impl
            
            asyncio.run(run_test())

        def test_concurrent_execution(self):
            """Test fully concurrent execution"""
            async def run_test():
                reward_impl = RewardImpl(timeout=6.0, max_workers=4)
                
                start_time = time.time()
                
                # Create all tasks
                tasks = []
                for completion, solution in self.test_cases:
                    grpo_data = create_test_grpo_data(completion, solution)
                    tasks.append(reward_impl(grpo_data))
                
                # Run all tasks concurrently
                results = await asyncio.gather(*tasks)
                
                elapsed_time = time.time() - start_time
                
                print(f"\n{'='*70}")
                print("Async Concurrent Execution Performance:")
                print(f"  Total cases: {len(self.test_cases)}")
                print(f"  Total time: {elapsed_time:.4f}s")
                print(f"  Average time per case: {elapsed_time/len(self.test_cases)*1000:.2f}ms")
                print(f"  Throughput: {len(self.test_cases)/elapsed_time:.2f} cases/sec")
                print(f"  Correct results: {sum(score for score, _ in results)}/{len(results)}")
                print(f"  Speedup vs sequential: {len(self.test_cases)/elapsed_time:.2f}x")
                print(f"{'='*70}")
                
                del reward_impl
            
            asyncio.run(run_test())

        def test_timeout_behavior(self):
            """Test timeout behavior"""
            async def run_test():
                # Use very short timeout to potentially trigger timeouts
                reward_impl = RewardImpl(timeout=0.01, max_workers=1)
                
                complex_expr = '\\boxed{' + '+'.join([f'x^{i}' for i in range(100)]) + '}'
                grpo_data = create_test_grpo_data(complex_expr, complex_expr)
                
                start_time = time.time()
                score, msg = await reward_impl(grpo_data)
                elapsed_time = time.time() - start_time
                
                print(f"\n{'='*70}")
                print("Timeout Test:")
                print("  Configured timeout: 0.01s")
                print(f"  Actual elapsed time: {elapsed_time:.4f}s")
                print(f"  Result: score={score}, msg='{msg}'")
                print(f"  Timeout triggered: {'Yes' if 'Timeout' in msg or 'timeout' in msg.lower() else 'No'}")
                print(f"{'='*70}")
                
                del reward_impl
            
            asyncio.run(run_test())

    # Run tests
    print("\n" + "="*70)
    print("LATEX MATH REWARD (ASYNC) - CORRECTNESS AND PERFORMANCE TESTS")
    print("="*70)
    
    # Run correctness tests first
    print("\n### CORRECTNESS TESTS ###")
    correctness_suite = unittest.TestLoader().loadTestsFromTestCase(TestLatexMathAsyncCorrectness)
    correctness_runner = unittest.TextTestRunner(verbosity=2)
    correctness_runner.run(correctness_suite)
    
    # Run performance tests
    print("\n### PERFORMANCE TESTS ###")
    performance_suite = unittest.TestLoader().loadTestsFromTestCase(TestLatexMathAsyncPerformance)
    performance_runner = unittest.TextTestRunner(verbosity=2)
    performance_runner.run(performance_suite)

