"""TLS-only NetEase IMAP/SMTP tools. No permanent-delete operation is exposed."""
from contextlib import contextmanager
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses, make_msgid, formatdate
from html.parser import HTMLParser
import imaplib
import base64
import mimetypes
import os
from pathlib import PurePosixPath
import re
import smtplib
import stat
import ssl
from uuid import UUID, uuid4
from typing_extensions import TypedDict


class MailRef(TypedDict):
    account: str
    folder: str
    uidvalidity: str
    uid: str

from ai_assistant.workspace.files import WorkspaceError
from ai_assistant.workspace.writes import WritableWorkspace


class MailError(ValueError):
    pass


class TextHTML(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.hidden += 1
        if tag in {'p', 'br', 'div', 'li'}:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in {'script', 'style'} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def html_text(value):
    parser = TextHTML(); parser.feed(value)
    return ''.join(parser.parts)


def checked(result):
    status, data = result
    if status != 'OK':
        raise MailError('Mailbox rejected the operation; check authorization and folder mapping.')
    return data


def quoted(value):
    if any(c in value for c in '\r\n\x00'):
        raise MailError('Invalid mailbox search value.')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def mailbox(value):
    """RFC 3501 modified UTF-7 for folder names (search values remain UTF-8 literals)."""
    quoted(value)
    output, pending = [], []
    def flush():
        if pending:
            encoded = base64.b64encode(''.join(pending).encode('utf-16-be')).decode().rstrip('=').replace('/', ',')
            output.append('&' + encoded + '-'); pending.clear()
    for char in value:
        if 0x20 <= ord(char) <= 0x7e:
            flush(); output.append('&-' if char == '&' else char)
        else:
            pending.append(char)
    flush()
    return quoted(''.join(output))


class MailService:
    def __init__(self, name, account, workspace):
        self.name, self.account, self.workspace = name, account, workspace
        domain = account.address.rsplit('@', 1)[1].lower()
        self.imap_host = account.imap_host or 'imap.' + domain
        self.smtp_host = account.smtp_host or 'smtp.' + domain

    @contextmanager
    def connection(self, folder=None, *, readonly=True):
        conn = imaplib.IMAP4_SSL(self.imap_host, self.account.imap_port,
                                ssl_context=ssl.create_default_context(), timeout=30)
        try:
            conn.login(self.account.address, self.account.authorization_code.get_secret_value())
            # NetEase may require RFC 2971 ID before SELECT.
            if b'ID' in conn.capabilities or 'ID' in conn.capabilities:
                imaplib.Commands.setdefault('ID', ('AUTH', 'SELECTED'))
                conn._simple_command('ID', '("name" "ai-assistant" "version" "1")')
            if folder is not None:
                checked(conn.select(mailbox(folder), readonly=readonly))
            yield conn
        finally:
            try:
                conn.logout()
            except (OSError, imaplib.IMAP4.error):
                pass

    def validate_ref(self, conn, ref):
        if ref.get('account') != self.name or not str(ref.get('uid', '')).isdigit():
            raise MailError('Invalid mailbox reference.')
        validity = conn.response('UIDVALIDITY')[1]
        if not validity or str(ref.get('uidvalidity')) != validity[0].decode():
            raise MailError('UIDVALIDITY changed; search again before accessing this message.')
        return str(ref['uid'])

    def folders(self) -> dict:
        with self.connection() as conn:
            return {'folders': [v.decode('utf-8', errors='replace') for v in checked(conn.list())]}

    def search(self, folder: str = 'INBOX', sender: str | None = None, subject: str | None = None, since: str | None = None, unseen: bool = False, offset: int = 0, limit: int = 20, message_id: str | None = None) -> dict:
        if offset < 0 or not 1 <= limit <= 100:
            raise MailError('Invalid pagination.')
        criteria = []
        if sender:
            criteria.append(('FROM', sender))
        if subject:
            criteria.append(('SUBJECT', subject))
        if message_id:
            criteria.append(('HEADER Message-ID', message_id))
        if since:
            if not re.fullmatch(r'\d{1,2}-[A-Za-z]{3}-\d{4}', since):
                raise MailError('Use IMAP date DD-Mon-YYYY.')
            criteria.append(('SINCE', since))
        if unseen:
            criteria.append(('UNSEEN', None))
        with self.connection(folder) as conn:
            matches = None
            # Intersect structured criteria. A UTF-8 value is one literal, never an IMAP command.
            for key, value in criteria or [('ALL', None)]:
                if value is not None and not value.isascii():
                    quoted(value)  # reject control characters before encoding
                    conn.literal = value.encode('utf-8')
                    data = checked(conn.uid('SEARCH', 'CHARSET', 'UTF-8', key))
                else:
                    args = [key] + ([quoted(value)] if value is not None else [])
                    data = checked(conn.uid('SEARCH', None, *args))
                found = set(data[0].split()) if data and data[0] else set()
                matches = found if matches is None else matches & found
            ids = sorted(matches or (), key=int, reverse=True)
            validity = conn.response('UIDVALIDITY')[1][0].decode()
            items = []
            for uid in ids[offset:offset+limit]:
                raw = checked(conn.uid('FETCH', uid, '(BODY.PEEK[HEADER.FIELDS (FROM TO SUBJECT DATE MESSAGE-ID)] RFC822.SIZE)'))
                chunk = next((v[1] for v in raw if isinstance(v, tuple)), b'')
                message = BytesParser(policy=policy.default).parsebytes(chunk)
                items.append({'ref': dict(account=self.name, folder=folder, uidvalidity=validity, uid=uid.decode()),
                              **{key: str(message.get(key, '')) for key in ['From', 'To', 'Subject', 'Date']}})
            return dict(items=items, total=len(ids), next_offset=offset+limit if offset+limit < len(ids) else None)

    def fetch(self, ref):
        with self.connection(ref['folder']) as conn:
            uid = self.validate_ref(conn, ref)
            sizes = checked(conn.uid('FETCH', uid, '(RFC822.SIZE)'))
            match = re.search(rb'RFC822.SIZE (\d+)', b' '.join(v for v in sizes if isinstance(v, bytes)))
            if not match or int(match[1]) > self.account.message_limit:
                raise MailError('Message missing or larger than configured limit.')
            data = checked(conn.uid('FETCH', uid, '(BODY.PEEK[])'))
            raw = next((v[1] for v in data if isinstance(v, tuple)), None)
            if raw is None or len(raw) > self.account.message_limit:
                raise MailError('Message missing or too large.')
            return BytesParser(policy=policy.default).parsebytes(raw)

    def read(self, ref: MailRef, offset: int = 0, limit: int = 12000) -> dict:
        if offset < 0 or not 1 <= limit <= 20000:
            raise MailError('Invalid body pagination.')
        message = self.fetch(ref)
        body = message.get_body(preferencelist=('plain', 'html'))
        text = body.get_content() if body else ''
        if body and body.get_content_type() == 'text/html':
            text = html_text(text)
        attachments = [dict(index=i, filename=p.get_filename(), content_type=p.get_content_type())
                       for i, p in enumerate(message.walk()) if p.get_filename() or p.get_content_disposition() == 'attachment']
        return dict(headers={k: str(message.get(k, '')) for k in ['From', 'To', 'Cc', 'Subject', 'Date', 'Message-ID']},
                    body=text[offset:offset+limit], total=len(text), attachments=attachments,
                    next_offset=offset+limit if offset+limit < len(text) else None)

    def download(self, ref: MailRef, index: int) -> dict:
        message = self.fetch(ref)
        parts = list(message.walk())
        if index < 0 or index >= len(parts) or not (parts[index].get_filename() or parts[index].get_content_disposition() == 'attachment'):
            raise MailError('Unknown attachment index.')
        part = parts[index]
        name = part.get_filename() or 'attachment.bin'
        if '\\' in name or PurePosixPath(name).name != name or name in {'.', '..'}:
            raise MailError('Unsafe attachment filename.')
        data = part.get_payload(decode=True) or b''
        if len(data) > self.account.attachment_limit:
            raise MailError('Attachment exceeds configured limit.')
        path = f'outputs/mail/{uuid4().hex}/{name}'
        writer = WritableWorkspace(self.workspace)
        with writer._parent(path, create=True) as (parent, relative):
            fd = os.open(relative.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
        return dict(path=path, bytes=len(data))

    def compose(self, to, subject, body, cc=None, attachments=None, reply_ref=None, reply_all=False, forward_ref=None):
        message = EmailMessage(policy=policy.SMTP)
        original = self.fetch(reply_ref or forward_ref) if reply_ref or forward_ref else None
        if reply_ref:
            sender = str(original.get('Reply-To') or original.get('From', ''))
            to = [address for _, address in getaddresses([sender])]
            if reply_all:
                to += [address for _, address in getaddresses(original.get_all('To', []) + original.get_all('Cc', []))
                       if address.casefold() != self.account.address.casefold()]
            if original.get('Message-ID'):
                message['In-Reply-To'] = str(original['Message-ID'])
                message['References'] = (str(original.get('References', '')) + ' ' + str(original['Message-ID'])).strip()
        if not to:
            raise MailError('Recipients are required.')
        recipients = list(dict.fromkeys(to + (cc or [])))
        for address in recipients:
            if '\r' in address or '\n' in address or not re.fullmatch(r'[^\s<>@,;]+@[^\s<>@,;]+', address):
                raise MailError('Invalid recipient address.')
        message['From'] = self.account.address
        message['To'] = ', '.join(dict.fromkeys(to))
        if cc:
            message['Cc'] = ', '.join(cc)
        message['Subject'] = subject
        message['Message-ID'] = make_msgid(domain=self.account.address.rsplit('@', 1)[1])
        message['Date'] = formatdate(localtime=False)
        message.set_content(body)
        if forward_ref:
            message.add_attachment(original)
        for path in attachments or []:
            # Reuse the workspace descriptor walk and exclusions, never resolve then open.
            with WritableWorkspace(self.workspace)._parent(path) as (parent, relative):
                fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                with os.fdopen(fd, 'rb') as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise MailError('Attachments require ordinary files without hard links.')
                    data = stream.read(self.account.attachment_limit + 1)
            if len(data) > self.account.attachment_limit:
                raise MailError('Attachment exceeds configured limit.')
            mime = mimetypes.guess_type(path)[0] or 'application/octet-stream'
            main, sub = mime.split('/', 1)
            message.add_attachment(data, maintype=main, subtype=sub, filename=PurePosixPath(path).name)
        if len(message.as_bytes()) > self.account.message_limit:
            raise MailError('Encoded message exceeds configured limit.')
        return message, recipients

    def send(self, to: list[str], subject: str, body: str, cc: list[str] | None = None, attachments: list[str] | None = None, reply_ref: MailRef | None = None, reply_all: bool = False, forward_ref: MailRef | None = None, operation_id: str | None = None) -> dict:
        message, recipients = self.compose(to, subject, body, cc, attachments, reply_ref, reply_all, forward_ref)
        if operation_id:
            message.replace_header('Message-ID', '<' + str(UUID(operation_id)) + '@' + self.account.address.rsplit('@', 1)[1] + '>')
        with smtplib.SMTP_SSL(self.smtp_host, self.account.smtp_port,
                             context=ssl.create_default_context(), timeout=30) as smtp:
            smtp.login(self.account.address, self.account.authorization_code.get_secret_value())
            try:
                refused = smtp.send_message(message, to_addrs=recipients)
            except smtplib.SMTPRecipientsRefused as exc:
                return dict(status='rejected', refused=list(exc.recipients), message_id=message['Message-ID'])
            except (OSError, smtplib.SMTPException):
                return dict(status='unknown', message_id=message['Message-ID'], detail='SMTP acceptance could not be confirmed; do not automatically resend.')
        result = dict(status='partially_accepted' if refused else 'server_accepted',
                      refused=list(refused), message_id=message['Message-ID'], sent_copy='not_configured')
        folder = self.account.folders.get('sent')
        if folder:
            try:
                with self.connection() as conn:
                    checked(conn.append(mailbox(folder), '(\\Seen)', None, message.as_bytes()))
                result['sent_copy'] = 'saved'
            except Exception:
                result['sent_copy'] = 'failed'
        return result

    def draft(self, to: list[str], subject: str, body: str, cc: list[str] | None = None, attachments: list[str] | None = None) -> dict:
        folder = self.account.folders.get('drafts')
        if not folder:
            raise MailError('Configure the drafts folder mapping first.')
        message, _ = self.compose(to, subject, body, cc, attachments)
        with self.connection() as conn:
            response = checked(conn.append(mailbox(folder), '(\\Draft)', None, message.as_bytes()))
        return dict(status='saved', message_id=message['Message-ID'], response=[str(v) for v in response])

    def update_draft(self, ref: MailRef, to: list[str], subject: str, body: str, cc: list[str] | None = None, attachments: list[str] | None = None) -> dict:
        folder = self.account.folders.get('drafts')
        if not folder or ref.get('folder') != folder:
            raise MailError('Draft updates require the configured drafts folder.')
        message, _ = self.compose(to, subject, body, cc, attachments)
        with self.connection(folder, readonly=False) as conn:
            uid = self.validate_ref(conn, ref)
            data = checked(conn.uid('FETCH', uid, '(FLAGS)'))
            if b'\\Draft' not in b' '.join(v for v in data if isinstance(v, bytes)):
                raise MailError('The referenced message is not a draft.')
            checked(conn.append(mailbox(folder), '(\\Draft)', None, message.as_bytes()))
            try:
                checked(conn.uid('STORE', uid, '+FLAGS.SILENT', '(\\Deleted)'))
            except Exception:
                return dict(status='replacement_saved_old_retained', message_id=message['Message-ID'])
        return dict(status='updated_old_marked_deleted', message_id=message['Message-ID'])

    def mark(self, ref: MailRef, seen: bool = True) -> dict:
        with self.connection(ref['folder'], readonly=False) as conn:
            uid = self.validate_ref(conn, ref)
            checked(conn.uid('STORE', uid, '+FLAGS.SILENT' if seen else '-FLAGS.SILENT', '(\\Seen)'))
        return dict(status='updated', seen=seen)

    def move(self, ref: MailRef, destination: str) -> dict:
        with self.connection(ref['folder'], readonly=False) as conn:
            uid = self.validate_ref(conn, ref)
            caps = {v.decode() if isinstance(v, bytes) else v for v in conn.capabilities}
            if 'MOVE' in caps:
                checked(conn.uid('MOVE', uid, mailbox(destination)))
            else:
                # UIDPLUS is mandatory: never expunge unrelated deleted messages.
                if 'UIDPLUS' not in caps:
                    raise MailError('Safe move requires MOVE or UIDPLUS support.')
                checked(conn.uid('COPY', uid, mailbox(destination)))
                checked(conn.uid('STORE', uid, '+FLAGS.SILENT', '(\\Deleted)'))
                checked(conn.uid('EXPUNGE', uid))
        return dict(status='moved', destination=destination)


def create_server(name, account, workspace):
    from fastmcp import FastMCP
    service = MailService(name, account, workspace)
    server = FastMCP('netease-' + name)
    # Sync handlers execute in the server's worker pool; credentials remain in this closure.
    descriptions = {
        'folders': 'List mailbox folders without modifying messages.',
        'search': 'Search messages using structured conditions and UIDVALIDITY references.',
        'read': 'Read paginated text and attachment metadata without marking read.',
        'download': 'Download one attachment into a new controlled workspace output file.',
        'send': 'Send, reply or forward a message; server acceptance does not imply delivery.',
        'draft': 'Save a new draft in the explicitly configured drafts folder.',
        'update_draft': 'Save a replacement draft and mark the exact old draft deleted without expunging.',
        'mark': 'Explicitly set read or unread status.',
        'move': 'Move exactly one UID without expunging unrelated messages.',
    }
    for method, description in descriptions.items():
        server.tool(getattr(service, method), name=method, description=description)
    return server
