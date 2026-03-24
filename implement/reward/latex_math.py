import logging

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from data_service.typing.grpo_data import GRPOData


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RewardImpl:
    """
    LaTeX math verification reward function (synchronous version).

    Uses math_verify library to parse and verify mathematical expressions.
    This is the standard synchronous implementation.
    """

    def __init__(self):
        """Initialize LaTeX math reward function."""
        logger.info("Initialized LaTeXMathReward (Synchronous)")

    def __call__(self, grpo_data: GRPOData):
        """
        Verify if the generated answer matches the ground truth.

        Args:
            grpo_data: The GRPOData object containing conversation and solution

        Returns:
            tuple: (reward_score, debug_message)
                - reward_score: 1.0 if correct, 0.0 if incorrect
                - debug_message: String describing the verification result
        """
        try:
            # Extract completion from conversation and answer from solution
            completion_text = grpo_data.conversation.messages[-1].content
            answer = grpo_data.solution

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
            answer = parse(answer)

            is_correct = verify(answer, completion)
            return float(is_correct), f"Answer: {answer}, Completion: {completion}"
        except Exception as e:
            logger.error(f"Error in LaTeX math reward: {e}")
            return 0.0, f"Error: {e}"


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

    class TestLatexMath(unittest.TestCase):

        def setUp(self):
            """Set up test reward instance"""
            self.reward_impl = RewardImpl()

        def test_correct_simple_boxed(self):
            """Test simple boxed answer"""
            grpo_data = create_test_grpo_data('The answer is \\boxed{42}', '\\boxed{42}')
            score, msg = self.reward_impl(grpo_data)
            self.assertEqual(score, 1.0)

        def test_correct_fraction(self):
            """Test fraction answer"""
            grpo_data = create_test_grpo_data(
                'The answer is \\boxed{\\frac{1}{2}}',
                '\\boxed{\\frac{1}{2}}'
            )
            score, msg = self.reward_impl(grpo_data)
            self.assertEqual(score, 1.0)

        def test_wrong_answer(self):
            """Test wrong answer"""
            grpo_data = create_test_grpo_data('The answer is \\boxed{42}', '\\boxed{100}')
            score, msg = self.reward_impl(grpo_data)
            self.assertEqual(score, 0.0)

        def test_equivalent_forms(self):
            """Test equivalent mathematical forms"""
            grpo_data = create_test_grpo_data('The answer is \\boxed{0.5}', '\\boxed{\\frac{1}{2}}')
            score, msg = self.reward_impl(grpo_data)
            self.assertEqual(score, 1.0)

        def test_complex_expression(self):
            """Test complex mathematical expression"""
            grpo_data = create_test_grpo_data(
                'The answer is \\boxed{2\\sqrt{3}}',
                '\\boxed{2\\sqrt{3}}'
            )
            score, msg = self.reward_impl(grpo_data)
            self.assertEqual(score, 1.0)

    unittest.main()
