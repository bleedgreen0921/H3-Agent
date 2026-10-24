import asyncio
from contextlib import contextmanager
from email.message import EmailMessage
from email import policy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from ai_assistant.integrations.config import MailAccount
from ai_assistant.integrations.netease import MailService, MailError, html_text
from ai_assistant.workspace.files import ReadOnlyWorkspace, WorkspaceError


def service(tmp_path):
    return MailService('main', MailAccount(address='test@163.com', authorization_code='client-code',
        folders={'sent': 'Sent', 'drafts': 'Drafts'}), ReadOnlyWorkspace(tmp_path))


def fixture_mail():
    msg = EmailMessage(policy=policy.SMTP)
    msg['From'] = 'Alice <alice@example.org>'; msg['To'] = 'test@163.com, bob@example.org'
    msg['Cc'] = 'carol@example.org'; msg['Subject'] = '中文邮件'; msg['Message-ID'] = '<original@example.org>'
    msg.set_content('正文内容')
    msg.add_attachment(b'attachment', maintype='application', subtype='octet-stream', filename='sample.bin')
    return msg


REF = {'account': 'main', 'folder': 'INBOX', 'uidvalidity': '123', 'uid': '7'}


def fake_connection(service, conn):
    @contextmanager
    def connection(*args, **kwargs):
        yield conn
    service.connection = connection


def test_read_uses_peek_and_uidvalidity(tmp_path):
    svc = service(tmp_path); raw = fixture_mail().as_bytes()
    conn = Mock(); conn.response.return_value = ('UIDVALIDITY', [b'123'])
    conn.uid.side_effect = [('OK', [f'7 (RFC822.SIZE {len(raw)})'.encode()]), ('OK', [(b'7', raw)])]
    fake_connection(svc, conn)
    result = svc.read(REF)
    assert result['headers']['Subject'] == '中文邮件'
    assert '正文内容' in result['body']
    assert conn.uid.call_args_list[1].args[-1] == '(BODY.PEEK[])'
    conn.response.return_value = ('UIDVALIDITY', [b'999'])
    with pytest.raises(MailError, match='UIDVALIDITY'):
        svc.read(REF)


def test_unicode_structured_search_uses_literal_and_intersection(tmp_path):
    svc = service(tmp_path); conn = Mock()
    conn.response.return_value = ('UIDVALIDITY', [b'123'])
    header = b'From: alice@example.org\r\nSubject: sample\r\n\r\n'
    conn.uid.side_effect = [('OK', [b'7 9']), ('OK', [b'7']), ('OK', [(b'7', header)])]
    fake_connection(svc, conn)
    result = svc.search(sender='alice@example.org', subject='中文')
    assert result['total'] == 1 and result['items'][0]['ref']['uid'] == '7'
    assert conn.literal == '中文'.encode()
    assert conn.uid.call_args_list[1].args == ('SEARCH', 'CHARSET', 'UTF-8', 'SUBJECT')
    with pytest.raises(MailError):
        svc.search(subject='bad\r\nLOGOUT')


def test_attachment_paths_and_reply_references(tmp_path):
    svc = service(tmp_path); svc.fetch = lambda ref: fixture_mail()
    assert (tmp_path / svc.download(REF, 2)['path']).read_bytes() == b'attachment'
    malicious = fixture_mail(); list(malicious.walk())[2].replace_header('Content-Disposition', 'attachment; filename="../escape"')
    svc.fetch = lambda ref: malicious
    with pytest.raises(MailError, match='Unsafe'):
        svc.download(REF, 2)
    svc.fetch = lambda ref: fixture_mail()
    msg, recipients = svc.compose([], 'Re: subject', 'Reply', reply_ref=REF, reply_all=True)
    assert set(recipients) == {'alice@example.org', 'bob@example.org', 'carol@example.org'}
    assert msg['In-Reply-To'] == '<original@example.org>'
    assert '<original@example.org>' in msg['References']
    (tmp_path / 'ordinary.bin').write_bytes(b'bytes')
    assert list(svc.compose(['a@example.org'], 'test', 'body', attachments=['ordinary.bin'])[0].iter_attachments())
    (tmp_path / 'link.bin').symlink_to(tmp_path / 'ordinary.bin')
    with pytest.raises((OSError, WorkspaceError)):
        svc.compose(['a@example.org'], 'test', 'body', attachments=['link.bin'])


def test_precise_move_and_unread(tmp_path):
    svc = service(tmp_path); conn = Mock(); conn.response.return_value = ('UIDVALIDITY', [b'123'])
    conn.capabilities = (b'UIDPLUS',); conn.uid.return_value = ('OK', [])
    fake_connection(svc, conn)
    svc.move(REF, 'Archive')
    assert [c.args[:2] for c in conn.uid.call_args_list] == [('COPY','7'), ('STORE','7'), ('EXPUNGE','7')]
    conn.expunge.assert_not_called()
    conn.capabilities = ()
    with pytest.raises(MailError, match='Safe move'):
        svc.move(REF, 'Archive')
    svc.mark(REF, False)
    assert conn.uid.call_args.args == ('STORE', '7', '-FLAGS.SILENT', '(\\Seen)')


def test_smtp_partial_acceptance_and_copy_failure(tmp_path):
    svc = service(tmp_path)
    smtp = Mock(); smtp.send_message.return_value = {'bad@example.org': (550, b'rejected')}
    @contextmanager
    def broken(*args, **kwargs):
        raise OSError('copy failed')
        yield
    svc.connection = broken
    with patch('smtplib.SMTP_SSL') as factory:
        factory.return_value.__enter__.return_value = smtp
        result = svc.send(['ok@example.org','bad@example.org'], 'Subject', 'body')
    assert result['status'] == 'partially_accepted' and result['sent_copy'] == 'failed'
    assert result['refused'] == ['bad@example.org']


def test_html_and_draft_mapping(tmp_path):
    assert 'remote' not in html_text('<p>Text<img src="https://remote/"></p><script>bad()</script>')
    assert 'bad' not in html_text('<script>bad()</script>Visible')
    svc = service(tmp_path); svc.account = svc.account.model_copy(update={'folders': {}})
    with pytest.raises(MailError, match='drafts'):
        svc.draft(['a@example.org'], 'subject', 'body')
