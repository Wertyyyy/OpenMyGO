import logging
from data_service.typing.grpo_data import GRPOData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RewardImpl:
    """
    Length penalty reward function (no correctness checking).
    
    This can be combined with other reward functions like latex_math.
    
    Punishment rules:
    - Safe length: configurable (default 8192 tokens, no penalty, returns 0.0)
    - Linear penalty from safe_length to max_penalty_length (0.0 to -1.0)
    - Fixed penalty of -1.0 at and beyond max_penalty_length
    """
    
    def __init__(self, safe_length, max_penalty_length):
        """
        Initialize length penalty reward function.
        
        Args:
            safe_length: Maximum response length with no penalty (default: 8192)
            max_penalty_length: Length at which maximum penalty is applied (default: 16384)
        """
        self.safe_length = safe_length
        self.max_penalty_length = max_penalty_length
        
        logger.info(
            f"Initialized LengthPenaltyReward: "
            f"safe_length={self.safe_length}, max_penalty_length={self.max_penalty_length}"
        )
    
    def __call__(self, grpo_data: GRPOData):
        """
        Calculate length penalty for a given response.
        
        Args:
            grpo_data: The GRPOData object containing response information
            
        Returns:
            tuple: (penalty_score, debug_message)
        """
        try:
            response_length = grpo_data.response_length
            
            if response_length <= self.safe_length:
                # No penalty for responses within safe length
                penalty = 0.0
                msg = f"Response length {response_length} <= {self.safe_length} (no penalty)"
            elif response_length >= self.max_penalty_length:
                # Fixed penalty for responses at or beyond max penalty length
                penalty = -1.0
                msg = f"Response length {response_length} >= {self.max_penalty_length} (max penalty -1.0)"
            else:
                # Linear penalty between safe length and max penalty length
                penalty = -(response_length - self.safe_length) / (
                    self.max_penalty_length - self.safe_length
                )
                msg = (
                    f"Response length {response_length} in "
                    f"[{self.safe_length}, {self.max_penalty_length}] "
                    f"(penalty {penalty:.4f})"
                )
            
            return penalty, msg
            
        except Exception as e:
            logger.error(f"Error in length penalty reward: {e}")
            return 0.0, f"Error: {e}"


if __name__ == "__main__":
    import unittest
    from data_service.typing.message import Conversation, Message

    def create_test_grpo_data(response_token_ids: list) -> GRPOData:
        """Helper function to create GRPOData for testing."""
        conversation = Conversation(messages=[
            Message(role="user", content="Test question"),
            Message(role="assistant", content="Test answer")
        ])
        return GRPOData(
            prompt_idx=0,
            response_idx=0,
            conversation=conversation,
            solution="Test solution",
            stop_reason="stop",
            prompt_token_ids=[1, 2, 3],
            response_token_ids=response_token_ids,
        )

    class TestLengthPenalty(unittest.TestCase):
        
        def setUp(self):
            """Set up test reward instance"""
            self.reward_impl = RewardImpl(safe_length=8192, max_penalty_length=16384)

        def test_very_short_response(self):
            """Very short response should have no penalty"""
            response_token_ids = list(range(100))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("no penalty", msg)

        def test_short_response(self):
            """Short response (1000 tokens) should have no penalty"""
            response_token_ids = list(range(1000))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("no penalty", msg)

        def test_medium_response(self):
            """Medium response (5000 tokens) should have no penalty"""
            response_token_ids = list(range(5000))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("no penalty", msg)

        def test_at_safe_length(self):
            """Response at exactly safe length should have no penalty"""
            response_token_ids = list(range(8192))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("no penalty", msg)

        def test_slightly_over_safe_length(self):
            """Response slightly over safe length should have small penalty"""
            # 9192 tokens = 8192 + 1000
            response_token_ids = list(range(9192))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            # Expected penalty: -(9192 - 8192) / 8192 = -1000/8192 ≈ -0.122
            expected_penalty = -(9192 - 8192) / 8192
            self.assertAlmostEqual(penalty, expected_penalty, places=4)
            self.assertLess(penalty, 0.0)
            self.assertGreater(penalty, -0.2)

        def test_quarter_way_to_max(self):
            """Response 1/4 way between safe and max should have -0.25 penalty"""
            # 10240 tokens = 8192 + (16384-8192)/4 = 8192 + 2048
            response_token_ids = list(range(10240))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            # Expected penalty: -(10240 - 8192) / 8192 = -2048/8192 = -0.25
            self.assertAlmostEqual(penalty, -0.25, places=4)

        def test_halfway_to_max(self):
            """Response halfway between safe and max should have -0.5 penalty"""
            # 12288 tokens = (8192 + 16384) / 2
            response_token_ids = list(range(12288))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            # Expected penalty: -(12288 - 8192) / 8192 = -4096/8192 = -0.5
            self.assertAlmostEqual(penalty, -0.5, places=4)

        def test_three_quarters_to_max(self):
            """Response 3/4 way between safe and max should have -0.75 penalty"""
            # 14336 tokens = 8192 + 3*(16384-8192)/4 = 8192 + 6144
            response_token_ids = list(range(14336))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            # Expected penalty: -(14336 - 8192) / 8192 = -6144/8192 = -0.75
            self.assertAlmostEqual(penalty, -0.75, places=4)

        def test_just_below_max(self):
            """Response just below max penalty length should have penalty close to -1.0"""
            # 16383 tokens
            response_token_ids = list(range(16383))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            # Expected penalty: -(16383 - 8192) / 8192 ≈ -0.9998
            expected_penalty = -(16383 - 8192) / 8192
            self.assertAlmostEqual(penalty, expected_penalty, places=4)
            self.assertGreater(penalty, -1.0)
            self.assertLess(penalty, -0.99)

        def test_at_max_penalty_length(self):
            """Response at exactly max penalty length should have -1.0 penalty"""
            response_token_ids = list(range(16384))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, -1.0)
            self.assertIn("max penalty", msg)

        def test_beyond_max_penalty_length(self):
            """Response beyond max penalty length should have fixed -1.0 penalty"""
            response_token_ids = list(range(20000))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, -1.0)
            self.assertIn("max penalty", msg)

        def test_way_beyond_max(self):
            """Response way beyond max should still have fixed -1.0 penalty"""
            response_token_ids = list(range(50000))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, -1.0)
            self.assertIn("max penalty", msg)

        def test_penalty_is_monotonic(self):
            """Penalty should increase monotonically with length"""
            lengths = [8192, 9000, 10000, 12000, 14000, 16000, 16384, 20000]
            penalties = []
            
            for length in lengths:
                response_token_ids = list(range(length))
                grpo_data = create_test_grpo_data(response_token_ids)
                penalty, _ = self.reward_impl(grpo_data)
                penalties.append(penalty)
            
            # Check that penalties are non-increasing (more negative or equal)
            for i in range(len(penalties) - 1):
                self.assertLessEqual(penalties[i+1], penalties[i])

        def test_empty_response(self):
            """Empty response should have no penalty"""
            response_token_ids = []
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("no penalty", msg)

    unittest.main()

