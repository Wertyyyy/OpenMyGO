import logging
from data_service.typing.grpo_data import GRPOData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RewardImpl:
    """
    Conditional length penalty reward function.
    
    Only applies length penalty when a specified reward (e.g., 'accuracy') is <= 0.
    This allows penalizing long incorrect responses while not penalizing long correct ones.
    
    Punishment rules (when condition is met):
    - Safe length: configurable (default 8192 tokens, no penalty, returns 0.0)
    - Linear penalty from safe_length to max_penalty_length (0.0 to -1.0)
    - Fixed penalty of -1.0 at and beyond max_penalty_length
    """
    
    def __init__(
        self, 
        condition_key,
        safe_length, 
        max_penalty_length
    ):
        """
        Initialize conditional length penalty reward function.
        
        Args:
            condition_key: The reward key to check (e.g., 'accuracy'). 
                          Length penalty only applies if this reward <= 0.
            safe_length: Maximum response length with no penalty (default: 8192)
            max_penalty_length: Length at which maximum penalty is applied (default: 16384)
        """
        self.condition_key = condition_key
        self.safe_length = safe_length
        self.max_penalty_length = max_penalty_length
        
        logger.info(
            f"Initialized ConditionalLengthPenaltyReward: "
            f"condition_key='{self.condition_key}', "
            f"safe_length={self.safe_length}, "
            f"max_penalty_length={self.max_penalty_length}"
        )
    
    def __call__(self, grpo_data: GRPOData):
        """
        Calculate conditional length penalty for a given response.
        
        Args:
            grpo_data: The GRPOData object containing response information
            
        Returns:
            tuple: (penalty_score, debug_message)
        """
        try:
            # Check if condition reward exists and its value
            if not grpo_data.rewards:
                logger.warning("No rewards available, skipping length penalty")
                return 0.0, "No rewards available, skipping length penalty"
            
            if self.condition_key not in grpo_data.rewards:
                logger.warning(
                    f"Condition key '{self.condition_key}' not found in rewards, skipping length penalty"
                )
                return 0.0, f"Condition key '{self.condition_key}' not found in rewards, skipping length penalty"
            
            condition_reward = grpo_data.rewards[self.condition_key]
            
            # Only apply length penalty if condition reward <= 0
            if condition_reward > 0:
                msg = (
                    f"Condition reward '{self.condition_key}'={condition_reward:.4f} > 0, "
                    f"skipping length penalty"
                )
                return 0.0, msg
            
            # Condition met, apply length penalty
            response_length = grpo_data.response_length
            
            if response_length <= self.safe_length:
                # No penalty for responses within safe length
                penalty = 0.0
                msg = (
                    f"Condition reward '{self.condition_key}'={condition_reward:.4f} <= 0, "
                    f"but response length {response_length} <= {self.safe_length} (no penalty)"
                )
            elif response_length >= self.max_penalty_length:
                # Fixed penalty for responses at or beyond max penalty length
                penalty = -1.0
                msg = (
                    f"Condition reward '{self.condition_key}'={condition_reward:.4f} <= 0, "
                    f"response length {response_length} >= {self.max_penalty_length} (max penalty -1.0)"
                )
            else:
                # Linear penalty between safe length and max penalty length
                penalty = -(response_length - self.safe_length) / (
                    self.max_penalty_length - self.safe_length
                )
                msg = (
                    f"Condition reward '{self.condition_key}'={condition_reward:.4f} <= 0, "
                    f"response length {response_length} in "
                    f"[{self.safe_length}, {self.max_penalty_length}] "
                    f"(penalty {penalty:.4f})"
                )
            
            return penalty, msg
            
        except Exception as e:
            logger.error(f"Error in conditional length penalty reward: {e}")
            return 0.0, f"Error: {e}"


if __name__ == "__main__":
    import unittest
    from data_service.typing.message import Conversation, Message

    def create_test_grpo_data(
        response_token_ids: list, 
        rewards: dict = None
    ) -> GRPOData:
        """Helper function to create GRPOData for testing."""
        conversation = Conversation(messages=[
            Message(role="user", content="Test question"),
            Message(role="assistant", content="Test answer")
        ])
        grpo_data = GRPOData(
            prompt_idx=0,
            response_idx=0,
            conversation=conversation,
            solution="Test solution",
            stop_reason="stop",
            prompt_token_ids=[1, 2, 3],
            response_token_ids=response_token_ids,
        )
        if rewards:
            grpo_data.rewards = rewards
        return grpo_data

    class TestConditionalLengthPenalty(unittest.TestCase):
        
        def setUp(self):
            """Set up test reward instance"""
            self.reward_impl = RewardImpl(
                condition_key="accuracy",
                safe_length=8192, 
                max_penalty_length=16384
            )

        def test_no_rewards_available(self):
            """Should skip penalty when no rewards available"""
            response_token_ids = list(range(10000))
            grpo_data = create_test_grpo_data(response_token_ids)
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("No rewards available", msg)

        def test_condition_key_not_found(self):
            """Should skip penalty when condition key not in rewards"""
            response_token_ids = list(range(10000))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"other_reward": 0.5}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("not found in rewards", msg)

        def test_positive_condition_reward_no_penalty(self):
            """Should skip penalty when condition reward > 0"""
            response_token_ids = list(range(10000))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"accuracy": 1.0}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("skipping length penalty", msg)

        def test_zero_condition_reward_applies_penalty(self):
            """Should apply penalty when condition reward = 0"""
            response_token_ids = list(range(10000))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"accuracy": 0.0}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            # 10000 tokens with accuracy=0 should get penalty
            expected_penalty = -(10000 - 8192) / (16384 - 8192)
            self.assertAlmostEqual(penalty, expected_penalty, places=4)
            self.assertLess(penalty, 0.0)

        def test_negative_condition_reward_applies_penalty(self):
            """Should apply penalty when condition reward < 0"""
            response_token_ids = list(range(10000))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"accuracy": -1.0}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            # 10000 tokens with accuracy=-1 should get penalty
            expected_penalty = -(10000 - 8192) / (16384 - 8192)
            self.assertAlmostEqual(penalty, expected_penalty, places=4)
            self.assertLess(penalty, 0.0)

        def test_short_response_negative_accuracy(self):
            """Short response with negative accuracy should have no penalty"""
            response_token_ids = list(range(1000))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"accuracy": -1.0}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("no penalty", msg)

        def test_at_safe_length_negative_accuracy(self):
            """Response at safe length with negative accuracy should have no penalty"""
            response_token_ids = list(range(8192))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"accuracy": 0.0}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("no penalty", msg)

        def test_slightly_over_safe_length_negative_accuracy(self):
            """Response slightly over safe length with negative accuracy should have small penalty"""
            response_token_ids = list(range(9192))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"accuracy": -1.0}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            expected_penalty = -(9192 - 8192) / (16384 - 8192)
            self.assertAlmostEqual(penalty, expected_penalty, places=4)
            self.assertLess(penalty, 0.0)

        def test_halfway_to_max_zero_accuracy(self):
            """Response halfway to max with zero accuracy should have -0.5 penalty"""
            response_token_ids = list(range(12288))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"accuracy": 0.0}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            expected_penalty = -(12288 - 8192) / (16384 - 8192)
            self.assertAlmostEqual(penalty, expected_penalty, places=4)

        def test_at_max_penalty_length_negative_accuracy(self):
            """Response at max penalty length with negative accuracy should have -1.0 penalty"""
            response_token_ids = list(range(16384))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"accuracy": -1.0}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, -1.0)
            self.assertIn("max penalty", msg)

        def test_beyond_max_penalty_length_negative_accuracy(self):
            """Response beyond max penalty length with negative accuracy should have -1.0 penalty"""
            response_token_ids = list(range(20000))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"accuracy": -1.0}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, -1.0)
            self.assertIn("max penalty", msg)

        def test_long_correct_response_no_penalty(self):
            """Long correct response (accuracy > 0) should have no penalty"""
            response_token_ids = list(range(20000))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"accuracy": 1.0}
            )
            
            penalty, msg = self.reward_impl(grpo_data)
            
            self.assertEqual(penalty, 0.0)
            self.assertIn("skipping length penalty", msg)

        def test_penalty_monotonic_for_negative_accuracy(self):
            """Penalty should increase monotonically with length when accuracy <= 0"""
            lengths = [8192, 9000, 10000, 12000, 14000, 16000, 16384, 20000]
            penalties = []
            
            for length in lengths:
                response_token_ids = list(range(length))
                grpo_data = create_test_grpo_data(
                    response_token_ids, 
                    rewards={"accuracy": 0.0}
                )
                penalty, _ = self.reward_impl(grpo_data)
                penalties.append(penalty)
            
            # Check that penalties are non-increasing (more negative or equal)
            for i in range(len(penalties) - 1):
                self.assertLessEqual(penalties[i+1], penalties[i])

        def test_custom_condition_key(self):
            """Should work with custom condition keys"""
            reward_impl = RewardImpl(
                condition_key="latex_valid",
                safe_length=8192,
                max_penalty_length=16384
            )
            
            response_token_ids = list(range(10000))
            grpo_data = create_test_grpo_data(
                response_token_ids, 
                rewards={"latex_valid": -1.0}
            )
            
            penalty, msg = reward_impl(grpo_data)
            
            expected_penalty = -(10000 - 8192) / (16384 - 8192)
            self.assertAlmostEqual(penalty, expected_penalty, places=4)

    unittest.main()

