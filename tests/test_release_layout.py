"""Public imports and package layout after retirement of research coordinators."""
import importlib.util
import unittest
from hybrid_topic import Topic, GenerationResult

class ReleaseLayoutTests(unittest.TestCase):
    def test_public_topic_types_do_not_require_retired_coordinator(self):
        topic = Topic('T1', 'Weather', 'Weather and rain', ('rain',))
        self.assertEqual(GenerationResult((topic,), '', '').topics[0].name, 'Weather')
        self.assertIsNone(importlib.util.find_spec('hybrid_topic.phase2_workflow'))
        self.assertIsNone(importlib.util.find_spec('hybrid_topic.unsupervised_selection'))
