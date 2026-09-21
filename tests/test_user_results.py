import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from hybrid_topic import TopicResult, read_documents
from hybrid_topic.types import Topic


class UserResultTests(unittest.TestCase):
    def setUp(self):
        self.documents = ["电池续航很短", "今天发布新手机", "不知道说什么"]
        self.codebook = [
            Topic("T1", "电池", "电池使用体验", ("电池", "续航")),
            Topic("T2", "新品", "产品发布", ("手机", "发布")),
            Topic("T3", "价格", "产品定价", ("价格", "费用")),
        ]

    def make_result(self, **kwargs):
        return TopicResult(
            self.documents, self.codebook, [0, 1, -1], **kwargs
        )

    def test_tables_preserve_unassigned_and_zero_support_topics(self):
        result = self.make_result(scores=[0.8, 0.4, 0.0])
        self.assertEqual(result.get_topic_info()["count"].tolist(), [1, 1, 0])
        info = result.get_document_info()
        self.assertEqual(info["topic_id"].tolist(), [0, 1, -1])
        self.assertEqual(info["assigned"].tolist(), [True, True, False])
        self.assertIsNone(info.iloc[2]["topic_name"])
        self.assertAlmostEqual(result.coverage, 2 / 3)
        self.assertEqual(result.get_document_info(topic_id=0)["text"].tolist(), [self.documents[0]])
        self.assertEqual(result.get_document_info(topic_id=-1)["text"].tolist(), [self.documents[2]])

    def test_unknown_scores_remain_missing_and_topic_details_are_copies(self):
        result = self.make_result()
        self.assertTrue(result.get_document_info()["score"].isna().all())
        details = result.get_topic(0)
        details["keywords"].append("changed")
        self.assertEqual(result.get_topic(0)["keywords"], ["电池", "续航"])
        with self.assertRaises(KeyError):
            result.get_topic(99)

    def test_csv_and_json_export_preserve_original_text_and_refuse_overwrite(self):
        result = self.make_result()
        with tempfile.TemporaryDirectory() as tmp:
            result.export(tmp)
            saved = json.loads((Path(tmp) / "result.json").read_text())
            self.assertEqual(saved["documents"][2]["topic_id"], -1)
            self.assertIsNone(saved["documents"][0]["score"])
            self.assertEqual(saved["topics"][0]["keywords"], ["电池", "续航"])
            self.assertEqual(pd.read_csv(Path(tmp) / "documents.csv")["text"].tolist(), self.documents)
            before = (Path(tmp) / "result.json").read_bytes()
            with self.assertRaises(FileExistsError):
                result.export(tmp)
            self.assertEqual((Path(tmp) / "result.json").read_bytes(), before)

    def test_bad_alignment_or_topic_indices_are_rejected(self):
        for assignments in ([0], [0, 1, 3], [0, 1, -2], [0, 1, 0.5]):
            with self.subTest(assignments=assignments), self.assertRaises(ValueError):
                TopicResult(self.documents, self.codebook, assignments)
        with self.assertRaises(ValueError):
            self.make_result(scores=[0.3, np.nan, 0.0])

    def test_duplicate_documents_are_not_silently_removed(self):
        result = TopicResult(["same", "same"], self.codebook, [0, 0])
        self.assertEqual(result.get_document_info()["document_id"].tolist(), [0, 1])
        self.assertEqual(result.get_topic(0)["count"], 2)

    def test_csv_input_preserves_order_text_and_ignores_label_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.csv"
            pd.DataFrame({"body": ["  原文\n保留", "NA", "123"], "label": [1, 2, 3]}).to_csv(p, index=False)
            self.assertEqual(read_documents(p, text_column="body"), ["  原文\n保留", "NA", "123"])
            with self.assertRaisesRegex(ValueError, "text_column"):
                read_documents(p)

    def test_invalid_text_input_fails_before_any_model_call(self):
        for documents in ([], "one string", ["ok", " "], ["ok", None], [42]):
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                TopicResult(documents, self.codebook, [])


if __name__ == "__main__":
    unittest.main()
