import io
import json
import unittest

from scripts.refresh_model_catalog import fetch_inventory_ids, merge_inventory


class ModelCatalogTests(unittest.TestCase):
    def test_inventory_merge_preserves_capabilities_and_marks_account_visibility(self):
        catalog = {
            'providers': {
                'openai': {
                    'models': {
                        'kept-model': {'context_window': 1234, 'source': 'official'},
                        'old-model': {'context_window': 5678},
                    }
                }
            }
        }
        merged = merge_inventory(catalog, 'openai', ['kept-model', 'new-model'], refreshed_at='2099-01-01T00:00:00Z')
        self.assertEqual(merged['providers']['openai']['models']['kept-model']['context_window'], 1234)
        self.assertTrue(merged['providers']['openai']['models']['kept-model']['account_visible'])
        self.assertFalse(merged['providers']['openai']['models']['old-model']['account_visible'])
        self.assertTrue(merged['providers']['openai']['models']['new-model']['account_visible'])
        self.assertNotIn('snapshot_date', merged)
        self.assertEqual(merged['providers']['openai']['last_inventory_refresh'], '2099-01-01T00:00:00Z')

    def test_inventory_parser_accepts_openai_style_data(self):
        payload = json.dumps({'data': [{'id': 'b'}, {'id': 'a'}, {'name': 'ignored-name'}]}).encode()

        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return payload

        ids = fetch_inventory_ids('openai', 'https://example.invalid/models', api_key='secret', opener=lambda *args, **kwargs: Response())
        self.assertEqual(ids, ['a', 'b', 'ignored-name'])


if __name__ == '__main__':
    unittest.main()
