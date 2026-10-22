from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ai_assistant.workspace import ReadOnlyWorkspace, WorkspaceError
from ai_assistant.workspace.writes import WritableWorkspace


class WriteTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = ReadOnlyWorkspace(self.root, config_path=self.root / 'custom.toml')
        self.writes = WritableWorkspace(self.workspace)

    def test_create_without_overwrite_and_digest(self):
        result = self.writes.create_file('sub/code.py', 'print("hello")\n')
        path = self.root / 'sub/code.py'
        self.assertEqual(result['sha256'], sha256(path.read_bytes()).hexdigest())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(WorkspaceError):
            self.writes.create_file('sub/code.py', 'changed')
        self.assertEqual(path.read_text(), 'print("hello")\n')
        self.assertEqual(self.workspace.read_text('sub/code.py')['sha256'], result['sha256'])
        self.assertFalse(list(path.parent.glob('.assistant-write-*')))

    def test_edit_snapshot_bom_crlf_permissions(self):
        path = self.root / 'file'
        path.write_bytes(b'\xef\xbb\xbfalpha=1\r\nbeta=2\r\n')
        path.chmod(0o750)
        digest = self.workspace.read_text('file')['sha256']
        result = self.writes.edit_file('file', [
            {'old_text': 'alpha=1', 'new_text': 'alpha=3'},
            {'old_text': 'beta=2', 'new_text': 'beta=4'},
        ], digest)
        self.assertEqual(path.read_bytes(), b'\xef\xbb\xbfalpha=3\r\nbeta=4\r\n')
        self.assertEqual(path.stat().st_mode & 0o777, 0o750)
        self.assertEqual(result['replacements'], 2)
        self.assertIn('-alpha=1', result['diff'])
        with self.assertRaisesRegex(WorkspaceError, '摘要'):
            self.writes.edit_file('file', [{'old_text': 'alpha=3', 'new_text': 'oops'}], digest)

    def test_failed_matching_is_all_or_nothing(self):
        for edits in ([{'old_text': '', 'new_text': 'x'}],
                      [{'old_text': 'aa', 'new_text': 'x'}],
                      [{'old_text': 'missing', 'new_text': 'x'}],
                      [{'old_text': 'aaa', 'new_text': 'x'}, {'old_text': 'aab', 'new_text': 'y'}],
                      [{'old_text': 'aaa', 'new_text': 'x'}, {'old_text': 'missing', 'new_text': 'y'}]):
            path = self.root / 'file'
            path.write_text('aaab')
            with self.assertRaises(WorkspaceError):
                self.writes.edit_file('file', edits, sha256(path.read_bytes()).hexdigest())
            self.assertEqual(path.read_text(), 'aaab')

    def test_path_credential_symlink_hardlink_special_limits(self):
        (self.root / 'link').symlink_to(self.root / 'elsewhere', target_is_directory=True)
        (self.root / 'file').write_text('hello')
        os.link(self.root / 'file', self.root / 'hardlink')
        os.mkfifo(self.root / 'pipe')
        for path in ('../outside', '/tmp/outside', 'assistant.toml', 'custom.toml', '.env', 'link/file', '.'):
            with self.subTest(path=path), self.assertRaises(WorkspaceError):
                self.writes.create_file(path, 'data')
        for path in ('file', 'hardlink', 'pipe'):
            with self.assertRaises(WorkspaceError):
                self.writes.edit_file(path, [{'old_text': 'hello', 'new_text': 'bye'}], sha256(b'hello').hexdigest())
        for content in ('a\0b', 'a' * (1024 * 1024 + 1), '\ud800'):
            with self.assertRaises(WorkspaceError):
                self.writes.create_file('oversize', content)

    def test_write_failure_and_precommit_conflict_preserve_file(self):
        self.writes.create_file('file', 'before')
        digest = self.workspace.read_text('file')['sha256']
        edits = [{'old_text': 'before', 'new_text': 'after'}]
        with patch('ai_assistant.workspace.writes.os.replace', side_effect=PermissionError):
            with self.assertRaises(WorkspaceError):
                self.writes.edit_file('file', edits, digest)
        self.assertEqual((self.root / 'file').read_text(), 'before')
        original = self.writes._stage
        def stage(*args):
            result = original(*args)
            (self.root / 'file').write_text('external')
            return result
        with patch.object(self.writes, '_stage', side_effect=stage):
            with self.assertRaisesRegex(WorkspaceError, '发生变化'):
                self.writes.edit_file('file', edits, digest)
        self.assertEqual((self.root / 'file').read_text(), 'external')
        self.assertFalse(list(self.root.glob('.assistant-write-*')))

    def test_large_diff_and_custom_exclusion(self):
        before = 'a' * 30000
        self.writes.create_file('file', before)
        result = self.writes.edit_file('file', [{'old_text': before, 'new_text': 'b' * 30000}],
                                      sha256(before.encode()).hexdigest())
        self.assertEqual(len(result['diff']), 20000)
        self.assertTrue(result['truncated'])
        writes = WritableWorkspace(ReadOnlyWorkspace(self.root, exclude_patterns=['private']))
        with self.assertRaises(WorkspaceError):
            writes.create_file('private/file', 'x')
