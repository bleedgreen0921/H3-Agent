"""中文消息提取与自动登记的权限边界。"""
import unittest
from unittest.mock import AsyncMock, patch

from ai_assistant.knowledge.admission import admit_message, extract_document_links


class AdmissionLinksTests(unittest.IsolatedAsyncioTestCase):
    async def test_chinese_markdown_query_and_duplicates(self):
        url = 'https://team.feishu.cn/docx/Ab123'
        text = f'请读（{url}）。说明「{url}?from=chat#section」以及[表格](https://team.larksuite.com/sheets/X123)后续'
        self.assertEqual(list(extract_document_links(None, text)), [url, 'https://team.larksuite.com/sheets/X123'])

    async def test_invalid_paths_are_not_truncated(self):
        text = ' '.join('https://team.feishu.cn/docx/' + tail for tail in ('Ab12/more', 'Ab12_bad', 'Ab12%20bad'))
        self.assertEqual(list(extract_document_links(None, text)), [])

    async def test_removed_source_is_never_restored(self):
        url = 'https://team.feishu.cn/wiki/Ab123'
        with patch('ai_assistant.knowledge.admission.register', new_callable=AsyncMock) as register:
            register.return_value = {'status': 'removed'}
            self.assertEqual(await admit_message(None, f'阅读{url}。这是资料'), [{'status': 'removed'}])
            register.assert_awaited_once_with(None, url, restore=False)
