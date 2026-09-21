import unittest
from hybrid_topic.backends import _initial_prompt, _residual_prompt
from hybrid_topic import Topic

class PromptContractTests(unittest.TestCase):
    def test_optional_direction_reaches_both_generation_stages(self):
        direction = "重点分析消费者遇到的问题"
        topic = Topic("T1", "Battery", "Battery life", ("battery", "charge"))
        initial = _initial_prompt(["text"], topic_instruction=direction)
        residual = _residual_prompt([topic], ["new text"], topic_instruction=direction)
        self.assertIn(direction, initial)
        self.assertIn(direction, residual)
        self.assertIn("Battery life", residual)
        self.assertIn("new text", residual)


    def test_omitting_direction_preserves_original_prompt(self):
        original = _initial_prompt(["text"])
        self.assertEqual(original, _initial_prompt(["text"], topic_instruction=None))
        self.assertEqual(original, _initial_prompt(["text"], topic_instruction="  "))
        self.assertNotIn("Analysis direction", original)


