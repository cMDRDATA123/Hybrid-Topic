"""Offline regression evidence for AUD-06 (inventory) and AUD-07 (freeze)."""
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from hybrid_topic import generation_budget
from scripts.refresh_model_catalog import fetch_inventory_ids, merge_inventory


class CatalogInventoryAuditTests(unittest.TestCase):
    def test_anthropic_pagination_collects_every_page_and_preserves_query(self):
        calls = []

        def opener(request, timeout):
            calls.append(request.full_url)
            query = parse_qs(urlparse(request.full_url).query)
            self.assertEqual(query['limit'], ['2'])
            if 'after_id' not in query:
                payload = {'data': [{'id': 'claude-a'}, {'id': 'claude-b'}],
                           'has_more': True, 'last_id': 'claude-b'}
            else:
                self.assertEqual(query['after_id'], ['claude-b'])
                payload = {'data': [{'id': 'claude-c'}], 'has_more': False,
                           'last_id': 'claude-c'}
            return io.BytesIO(json.dumps(payload).encode())

        ids = fetch_inventory_ids('anthropic', 'https://example.invalid/models?limit=2',
                                  api_key='offline-key', opener=opener)
        self.assertEqual(ids, ['claude-a', 'claude-b', 'claude-c'])
        self.assertEqual(len(calls), 2)

    def test_has_more_without_valid_cursor_rejects_incomplete_inventory(self):
        for cursor in (None, '', 123):
            with self.subTest(cursor=cursor):
                payload = {'data': [{'id': 'claude-a'}], 'has_more': True,
                           'last_id': cursor}
                with self.assertRaises((ValueError, RuntimeError)):
                    fetch_inventory_ids('anthropic', 'https://example.invalid/models',
                                        api_key='offline-key',
                                        opener=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))

    def test_repeated_cursor_rejects_incomplete_inventory_without_looping(self):
        calls = []

        def opener(*args, **kwargs):
            calls.append(1)
            if len(calls) > 3:
                raise AssertionError('repeated pagination cursor must stop promptly')
            return io.BytesIO(json.dumps({'data': [{'id': 'claude-a'}],
                                         'has_more': True, 'last_id': 'claude-a'}).encode())

        with self.assertRaises((ValueError, RuntimeError)):
            fetch_inventory_ids('anthropic', 'https://example.invalid/models',
                                api_key='offline-key', opener=opener)

    def test_malformed_model_list_is_not_accepted_as_complete_inventory(self):
        for payload in ({'data': 'bad'}, {'error': 'denied'}, [],
                        {'data': [{'id': 'claude-a'}], 'has_more': 'false'}):
            with self.subTest(payload=payload):
                with self.assertRaises((ValueError, RuntimeError)):
                    fetch_inventory_ids('anthropic', 'https://example.invalid/models',
                                        api_key='offline-key',
                                        opener=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))

    def test_inventory_refresh_preserves_capability_verification_date_and_values(self):
        verified = {'context_window': 200000, 'max_output_tokens': 16384,
                    'source': 'official-page-2026-09-06', 'tokenizer_encoding': 'provider_count',
                    'reasoning_efforts': ['low', 'high']}
        catalog = {'snapshot_date': '2026-09-06',
                   'providers': {'anthropic': {'models': {'claude-a': dict(verified),
                                                          'claude-old': {}}}}}
        merged = merge_inventory(catalog, 'anthropic', ['claude-a', 'claude-new'],
                                 refreshed_at='2026-09-07T10:00:00Z')
        self.assertEqual(merged['snapshot_date'], '2026-09-06')
        provider = merged['providers']['anthropic']
        self.assertEqual(provider['last_inventory_refresh'], '2026-09-07T10:00:00Z')
        self.assertTrue(provider['models']['claude-a']['account_visible'])
        self.assertFalse(provider['models']['claude-old']['account_visible'])
        for key, value in verified.items():
            self.assertEqual(provider['models']['claude-a'][key], value)

    def test_inventory_only_model_is_explicitly_unverified_and_unknown_modality(self):
        merged = merge_inventory({'providers': {'openai': {'models': {}}}}, 'openai',
                                 ['new-image-model'], refreshed_at='2026-09-07T10:00:00Z')
        entry = merged['providers']['openai']['models']['new-image-model']
        self.assertEqual(entry.get('capability_status'), 'unverified')
        self.assertEqual(entry.get('modality'), 'unknown')
        self.assertIsNone(entry.get('context_window'))
        self.assertIsNone(entry.get('tokenizer_encoding'))


class CatalogFreezeAuditTests(unittest.TestCase):
    def test_freeze_copies_environment_selected_catalog_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'configured.json'
            original = b'{"snapshot_date":"2026-09-07","providers":{}}\n'
            source.write_bytes(original)
            destination = root / 'snapshot' / 'model_catalog.json'
            with patch.dict(os.environ, {'HYBRID_TOPIC_MODEL_CATALOG': str(source)}):
                record = generation_budget.freeze_model_catalog(destination)
            self.assertEqual(Path(record['path']).resolve(), destination.resolve())
            self.assertEqual(Path(record['source_path']).resolve(), source.resolve())
            self.assertEqual(record['sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(destination.read_bytes(), original)
            generation_budget.verify_frozen_model_catalog(record)

    def test_source_changes_cannot_change_frozen_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.json'
            original = b'{"providers":{"openai":{"models":{}}}}'
            source.write_bytes(original)
            record = generation_budget.freeze_model_catalog(root / 'frozen.json', catalog_path=source)
            source.write_text('{"providers":{}}')
            generation_budget.verify_frozen_model_catalog(record)
            self.assertEqual(Path(record['path']).read_bytes(), original)

    def test_frozen_copy_changes_are_detected_even_if_json_remains_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.json'
            source.write_text('{"providers":{}}')
            record = generation_budget.freeze_model_catalog(root / 'frozen.json', catalog_path=source)
            Path(record['path']).write_text('{"providers":{},"changed":true}')
            with self.assertRaises((ValueError, RuntimeError)):
                generation_budget.verify_frozen_model_catalog(record)

    def test_invalid_catalog_is_not_frozen_as_valid_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'invalid.json'
            source.write_text('{"providers":[]}')
            with self.assertRaises(ValueError):
                generation_budget.freeze_model_catalog(root / 'frozen.json', catalog_path=source)


if __name__ == '__main__':
    unittest.main()
