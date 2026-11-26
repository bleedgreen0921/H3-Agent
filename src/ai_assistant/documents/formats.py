"""附件读取与知识库入库共用的文档格式注册表。"""

PDF_EXTENSIONS = frozenset({".pdf"})
DOCX_EXTENSIONS = frozenset({".docx"})
CSV_EXTENSIONS = frozenset({".csv", ".tsv"})
OFFICE_EXTENSIONS = frozenset({".xlsx", ".pptx"})

TEXT_EXTENSIONS = frozenset({
    ".md", ".markdown", ".txt", ".rst", ".log",
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".vue", ".svelte",
    ".java", ".kt", ".kts", ".scala", ".groovy",
    ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cs",
    ".rb", ".php", ".pl", ".lua", ".r", ".swift", ".dart",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".json", ".jsonl", ".ndjson",
})

# 来源扫描和 RAG 入库能力；OFFICE 仅用于知识库，不默认进入附件白名单。
RAG_SCAN_EXTENSIONS = (
    PDF_EXTENSIONS | DOCX_EXTENSIONS | CSV_EXTENSIONS |
    OFFICE_EXTENSIONS | TEXT_EXTENSIONS
)

# 附件上传白名单：当前不包含 xlsx/pptx，保持附件读取范围更克制。
ATTACHMENT_EXTENSIONS = (
    PDF_EXTENSIONS | DOCX_EXTENSIONS | CSV_EXTENSIONS | TEXT_EXTENSIONS
)

SUPPORTED_EXTENSIONS = RAG_SCAN_EXTENSIONS


def is_text_suffix(suffix):
    return suffix.lower() in TEXT_EXTENSIONS


def is_csv_suffix(suffix):
    return suffix.lower() in CSV_EXTENSIONS
