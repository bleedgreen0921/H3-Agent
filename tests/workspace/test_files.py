import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ai_assistant.workspace import ReadOnlyWorkspace, WorkspaceError


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'a.txt').write_text('first\nneedle\nlast', encoding='utf-8')
        (self.root / 'sub').mkdir()
        (self.root / 'sub/b.txt').write_text('needle\nNeedle', encoding='utf-8')
        self.workspace = ReadOnlyWorkspace(self.root)

    def test_list_read_search(self):
        result = self.workspace.list_directory(limit=1)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['entries'][0]['path'], 'a.txt')
        result = self.workspace.read_text('a.txt', 2, 1)
        self.assertEqual(result['content'], '2: needle')
        self.assertTrue(result['truncated'])
        result = self.workspace.search_text('needle')
        self.assertEqual(len(result['matches']), 2)
        self.assertEqual({item['line'] for item in result['matches']}, {1, 2})
        self.assertTrue(self.workspace.search_text('needle', max_results=1)['truncated'])
        self.assertEqual(self.workspace.read_text('a.txt', 20)['content'], '')

    def test_exclusions_apply_to_every_operation(self):
        for name in ('assistant.toml', 'models.toml', '.env', 'private.key', 'credentials.json', 'x.pem'):
            (self.root / name).write_text('needle SECRET')
            with self.subTest(name=name), self.assertRaises(WorkspaceError):
                self.workspace.read_text(name)
        for name in ('.hidden', '.venv', 'node_modules', '__pycache__'):
            (self.root / name).mkdir()
            (self.root / name / 'data').write_text('needle SECRET')
        custom = self.root / 'custom.toml'
        custom.write_text('needle SECRET')
        workspace = ReadOnlyWorkspace(self.root, config_path=custom, exclude_patterns=['sub'])
        self.assertEqual([item['path'] for item in workspace.list_directory()['entries']], ['a.txt'])
        self.assertEqual(len(workspace.search_text('needle')['matches']), 1)
        with self.assertRaises(WorkspaceError):
            workspace.read_text('custom.toml')

    def test_paths_symlinks_and_special_files(self):
        (self.root / 'link').symlink_to(self.root / 'a.txt')
        (self.root / 'linked_dir').symlink_to(self.root / 'sub', target_is_directory=True)
        os.mkfifo(self.root / 'pipe')
        for path in ('../other', str(self.root / 'a.txt'), 'link', 'linked_dir/b.txt', 'pipe', 'missing'):
            with self.subTest(path=path), self.assertRaises(WorkspaceError):
                self.workspace.read_text(path)
        self.assertEqual(len(self.workspace.search_text('needle')['matches']), 2)

    def test_encoding_size_and_output_bounds(self):
        (self.root / 'bom').write_bytes(b'\xef\xbb\xbfhello')
        self.assertEqual(self.workspace.read_text('bom')['content'], '1: hello')
        for name, content in [('binary', b'abc\x00'), ('invalid', b'\xff'),
                              ('large', b'a' * (1024 * 1024 + 1))]:
            (self.root / name).write_bytes(content)
            with self.assertRaises(WorkspaceError):
                self.workspace.read_text(name)
        (self.root / 'long').write_text('needle' * 6000)
        self.assertTrue(self.workspace.read_text('long')['truncated'])
        match = self.workspace.search_text('needle', 'long')['matches'][0]
        self.assertEqual(len(match['text']), 500)
        self.assertTrue(match['line_truncated'])
        self.assertGreaterEqual(self.workspace.search_text('needle')['skipped'], 3)

    def test_scan_and_byte_limits(self):
        for number in range(21):
            (self.root / f'{number}.txt').write_bytes(b'a' * (1024 * 1024))
        self.assertTrue(self.workspace.search_text('not present')['truncated'])
        with TemporaryDirectory() as name:
            root = Path(name)
            for number in range(5001):
                (root / f'.hidden{number}').touch()
            workspace = ReadOnlyWorkspace(root)
            self.assertTrue(workspace.search_text('x')['truncated'])
            self.assertTrue(workspace.list_directory()['truncated'])


def test_config_aliases_and_unicode_files_on_actual_filesystem(tmp_path):
    from ai_assistant.workspace.files import ReadOnlyWorkspace, WorkspaceError
    import pytest
    config=tmp_path/'PrivateConfig.toml';config.write_text('PRIVATE_MARKER')
    workspace=ReadOnlyWorkspace(tmp_path,config_path=config)
    alias=tmp_path/'privateconfig.toml'
    if alias.exists():
        with pytest.raises(WorkspaceError):workspace.read_text(alias.name)
    source=tmp_path/'中文-é.txt';source.write_text('正文')
    assert workspace.read_text(source.name)['content']=='1: 正文'
