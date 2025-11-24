from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

try:  # pragma: no cover - optional dependency
    import httpx
except ImportError:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ocr_benchmark")


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = PROJECT_ROOT / ".artifacts" 
DEFAULT_PDF = PROJECT_ROOT / "data" / "input.pdf"
DEFAULT_EXPECTED = PROJECT_ROOT / "data" / "expected_output.json"
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "extraction_prompt.md"
RESULTS_PATH = ARTIFACT_ROOT / "results.json"
RESULTS_HTML_PATH = ARTIFACT_ROOT / "results.html"
LLM_NAMES = ("gpt-4.1-mini", "gpt-5-mini")
DEFAULT_JUDGE = "gpt-4.1-mini"


def load_env(env_path: Path = PROJECT_ROOT / ".env") -> None:
    """Populate os.environ with variables defined in .env if present."""
    if not env_path.exists():
        logger.warning(f"No .env file found at {env_path}")
        return
    logger.info(f"Loading environment from {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env()


def load_persisted_results() -> Dict[str, Any]:
    if RESULTS_PATH.exists():
        try:
            data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    html = data.get("leaderboard_html") or ""
    if not html and RESULTS_HTML_PATH.exists():
        html = RESULTS_HTML_PATH.read_text(encoding="utf-8")
        data["leaderboard_html"] = html
    candidate_details = data.get("candidate_details")
    if not isinstance(candidate_details, list):
        candidate_details = []
    data["candidate_details"] = candidate_details
    data.setdefault("generated_at", None)
    return data


def _build_candidate_map(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    details = data.get("candidate_details")
    if not isinstance(details, list):
        return {}
    mapping: Dict[str, Dict[str, Any]] = {}
    for entry in details:
        if isinstance(entry, dict):
            label = entry.get("label")
            if isinstance(label, str):
                normalized = dict(entry)
                if "content" not in normalized and "output" in normalized:
                    normalized["content"] = normalized.pop("output")
                if "status" not in normalized:
                    normalized["status"] = "error" if normalized.get("error") else "ok"
                mapping[label] = normalized
    return mapping


def save_persisted_results(data: Dict[str, Any]) -> None:
    logger.info(f"Saving results to {RESULTS_PATH}")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "leaderboard_html": data.get("leaderboard_html", ""),
        "candidate_details": data.get("candidate_details", []),
        "generated_at": data.get("generated_at"),
    }
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html = payload.get("leaderboard_html")
    if isinstance(html, str):
        RESULTS_HTML_PATH.write_text(html, encoding="utf-8")
    global PERSISTED_RESULTS
    PERSISTED_RESULTS = payload
    logger.info("Results saved successfully.")


def get_persisted_results() -> Dict[str, Any]:
    return {
        "leaderboard_html": PERSISTED_RESULTS.get("leaderboard_html", ""),
        "candidate_details": list(PERSISTED_RESULTS.get("candidate_details", [])),
        "generated_at": PERSISTED_RESULTS.get("generated_at"),
    }


PERSISTED_RESULTS: Dict[str, Any] = load_persisted_results()


@dataclass
class OCRResult:
    """Represents the output of running a single OCR engine."""

    engine: str
    markdown: str
    page_images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateOutput:
    """Structured record describing a single LLM extraction attempt."""

    llm: str
    ocr: str
    mode: str
    content: str = ""
    error: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.llm}/{self.ocr}/{self.mode}"

    @property
    def status(self) -> str:
        return "error" if self.error else "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "display_label": self.label.replace("/", " · "),
            "llm": self.llm,
            "ocr": self.ocr,
            "mode": self.mode,
            "status": self.status,
            "content": self.content,
            "error": self.error,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class Scenario:
    """Describes a single OCR + LLM + context combination."""

    llm: str
    ocr: str
    mode: str
    label: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", f"{self.llm}/{self.ocr}/{self.mode}")


@dataclass(frozen=True)
class LLMConfig:
    """Configuration required to call an Azure OpenAI deployment."""

    name: str
    endpoint: str
    deployment: str
    api_version: str
    api_key: str


class AzureChatClient:
    """Tiny wrapper around Azure OpenAI chat completions with light retry logic."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        default_temperature: float = 0.0,
        default_top_p: float = 1.0,
    ) -> None:
        try:
            from openai import (  # type: ignore
                APIError,
                APITimeoutError,
                AzureOpenAI,
                BadRequestError,
                RateLimitError,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install the 'openai' package to call Azure OpenAI endpoints."
            ) from exc

        self._client = AzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
        )
        self._deployment = config.deployment
        self._default_params: Dict[str, Any] = {
            "temperature": default_temperature,
            "top_p": default_top_p,
        }
        self._BadRequestError = BadRequestError
        self._RateLimitError = RateLimitError
        self._APIError = APIError
        self._APITimeoutError = APITimeoutError
        self._max_attempts = 4
        self._retry_status_codes = {408, 409, 429, 500, 502, 503, 504}

    def generate(
        self,
        messages: Sequence[Dict[str, Any]],
        **overrides: Any,
    ) -> str:
        params = {**self._default_params, **overrides}
        last_error: Optional[Exception] = None

        logger.info(f"AzureChatClient: Generating with model={self._deployment}")
        
        for attempt in range(1, self._max_attempts + 1):
            try:
                logger.info(f"AzureChatClient: Attempt {attempt}/{self._max_attempts}")
                start_time = time.time()
                response = self._client.chat.completions.create(
                    model=self._deployment,
                    messages=list(messages),
                    **params,
                )
                duration = time.time() - start_time
                logger.info(f"AzureChatClient: Success in {duration:.2f}s")
                return _extract_choice_text(response)
            except self._BadRequestError as exc:  # type: ignore[attr-defined]
                logger.error(f"AzureChatClient: BadRequestError: {exc}")
                # Some deployments reject temperature entirely; retry once without it.
                if "temperature" in params:
                    logger.info("AzureChatClient: Retrying without temperature parameter.")
                    params = dict(params)
                    params.pop("temperature", None)
                    continue
                raise
            except (self._RateLimitError, self._APIError, self._APITimeoutError) as exc:  # type: ignore[misc]
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code is None:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                
                logger.warning(f"AzureChatClient: Error (status={status_code}): {exc}")

                retryable = isinstance(exc, self._RateLimitError) or (
                    isinstance(status_code, int) and status_code in self._retry_status_codes
                )
                if not retryable or attempt == self._max_attempts:
                    break
                wait_time = min(2 ** attempt, 15)
                logger.info(f"AzureChatClient: Sleeping {wait_time}s before retry.")
                time.sleep(wait_time)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"AzureChatClient: Unexpected error: {exc}")
                last_error = exc
                break

        message = f"Azure OpenAI request failed after retries: {last_error}"
        logger.error(message)
        raise RuntimeError(message)


def _extract_choice_text(response: Any) -> str:
    """Normalize Azure chat completion responses into a plain string."""
    choice = response.choices[0].message
    content = getattr(choice, "content", None)
    parts: List[str] = []

    if isinstance(content, list):
        for entry in content:
            if isinstance(entry, dict):
                text = entry.get("text")
                if text:
                    parts.append(str(text))
    elif isinstance(content, str):
        parts.append(content)

    refusal = getattr(choice, "refusal", None)
    if refusal:
        parts.append(str(refusal))

    if not parts:
        try:
            parts.append(json.dumps(response.model_dump(), ensure_ascii=False))
        except Exception:  # pragma: no cover - defensive
            parts.append(str(response))
    return "\n".join(parts).strip()


def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


def _build_llm_configs() -> Dict[str, LLMConfig]:
    configs: Dict[str, LLMConfig] = {}
    prefix_map = {
        "gpt-4.1-mini": "AZURE_OPENAI_GPT41",
        "gpt-5-mini": "AZURE_OPENAI_GPT5",
    }
    for name in LLM_NAMES:
        prefix = prefix_map[name]
        endpoint = _require_env(f"{prefix}_ENDPOINT")
        deployment = _require_env(f"{prefix}_DEPLOYMENT")
        api_version = _require_env(f"{prefix}_API_VERSION")
        api_key = _require_env(f"{prefix}_API_KEY")
        configs[name] = LLMConfig(
            name=name,
            endpoint=endpoint,
            deployment=deployment,
            api_version=api_version,
            api_key=api_key,
        )
    return configs


LLM_CONFIGS = _build_llm_configs()
_LLM_CLIENTS: Dict[str, AzureChatClient] = {}

def get_llm_client(name: str) -> AzureChatClient:
    if name not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM '{name}'. Available: {sorted(LLM_CONFIGS)}")
    if name not in _LLM_CLIENTS:
        _LLM_CLIENTS[name] = AzureChatClient(LLM_CONFIGS[name])
    return _LLM_CLIENTS[name]


def render_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("page-*.png"))
    if existing:
        return [str(path) for path in existing]

    from pdf2image import convert_from_path

    images = convert_from_path(str(pdf_path), dpi=dpi)
    page_paths: List[str] = []
    for index, image in enumerate(images, start=1):
        page_path = output_dir / f"page-{index:03d}.png"
        image.save(page_path, format="PNG")
        page_paths.append(str(page_path))
    if page_paths:
        return page_paths

    import fitz

    doc = fitz.open(str(pdf_path))
    page_paths: List[str] = []
    for index, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        page_path = output_dir / f"page-{index:03d}.png"
        pix.save(str(page_path))
        page_paths.append(str(page_path))
    return page_paths


def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".gif":
        return "image/gif"
    return "application/octet-stream"


def deepseek_ocr(pdf_path: Path, workspace: Path) -> OCRResult:
    endpoint = os.environ.get("DEEPSEEK_OCR_ENDPOINT")
    api_key = os.environ.get("DEEPSEEK_OCR_API_KEY")
    model = os.environ.get("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")
    if not endpoint or not api_key:
        raise EnvironmentError("DeepSeek OCR requires DEEPSEEK_OCR_ENDPOINT and DEEPSEEK_OCR_API_KEY.")

    logger.info(f"DeepSeek OCR: Processing {pdf_path.name} using {model}")
    pages_dir = workspace / "pages"
    page_images = render_pdf_to_images(pdf_path, pages_dir)
    markdown_pages: List[str] = []
    for index, image_path in enumerate(page_images, start=1):
        logger.info(f"DeepSeek OCR: Processing page {index}/{len(page_images)}")
        markdown = _deepseek_process_page(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            image_path=Path(image_path),
            index=index,
            total=len(page_images),
        )
        markdown_pages.append(f"<!-- Page {index} -->\n{markdown.strip()}")

    combined = "\n\n<!-- PageBreak -->\n\n".join(markdown_pages)
    return OCRResult(engine="deepseek", markdown=combined, page_images=page_images)


def _deepseek_process_page(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    image_path: Path,
    index: int,
    total: int,
) -> str:
    encoded_image = encode_file(image_path)
   
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    last_error: Optional[Exception] = None

    for attempt in range(1, 7):
        logger.info(f"DeepSeek OCR (Page {index}): Attempt {attempt}/6")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                        },
                    ],
                },
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
            "stream": False,
        }

        # Log request payload (sanitized)
        display_messages = []
        for msg in payload["messages"]:
            display_content = []
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        preview = encoded_image[:10] + "..."
                        display_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{preview}"}})
                    else:
                        display_content.append(item)
            else:
                display_content = msg["content"]
            display_messages.append({"role": msg["role"], "content": display_content})
            
        display_payload = {k: v for k, v in payload.items() if k != "messages"}
        display_payload["messages"] = display_messages
        logger.info(f"DeepSeek OCR (Page {index}): Request payload: {json.dumps(display_payload)}")

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            duration = time.time() - start_time
            logger.info(f"DeepSeek OCR (Page {index}): Response status {response.status_code} in {duration:.2f}s")

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                
                logger.info(f"DeepSeek OCR (Page {index}): Content preview: {str(content)[:200]}...")

                content = _clean_deepseek_markdown(str(content))
                if _looks_like_placeholder(content):
                    logger.warning(f"DeepSeek OCR (Page {index}): Placeholder content detected.")
                    last_error = RuntimeError("DeepSeek returned placeholder content")
                    if attempt < 7:
                        time.sleep(10)
                        continue
                elif content:
                    return content
                else:
                    logger.warning(f"DeepSeek OCR (Page {index}): Empty content returned.")
                    last_error = RuntimeError("DeepSeek returned empty content")
                    if attempt < 7:
                        time.sleep(10)
                        continue
            elif response.status_code in {502, 503, 504} and attempt < 7:
                logger.warning(f"DeepSeek OCR (Page {index}): Server error {response.status_code}, retrying...")
                time.sleep(10)
                continue
            else:
                logger.error(f"DeepSeek OCR (Page {index}): Failed with {response.status_code}: {response.text}")
                raise RuntimeError(f"DeepSeek OCR error {response.status_code}: {response.text}")
        except Exception as exc:  # pragma: no cover - network operations
            logger.error(f"DeepSeek OCR (Page {index}): Exception: {exc}")
            last_error = exc
            if attempt < 7:
                time.sleep(10)
                continue
    raise RuntimeError(f"DeepSeek OCR failed for page {index}: {last_error}")


def _clean_deepseek_markdown(markdown: str) -> str:
    cleaned = re.sub(r"image\[\[[^\]]*\]\]\s*", "", markdown)
    cleaned = re.sub(r"text\[\[[^\]]*\]\]\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


PLACEHOLDER_PATTERNS = [
    re.compile(r"use the markdown table syntax", re.IGNORECASE),
    re.compile(r"use the `?caption`? attribute", re.IGNORECASE),
    re.compile(r"convert the document to markdown", re.IGNORECASE),
]


def _looks_like_placeholder(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    if normalized.strip().upper().startswith("ERROR:"):
        return True
    collapsed = re.sub(r"\s+", " ", normalized).lower()
    repetitions = 0
    for pattern in PLACEHOLDER_PATTERNS:
        repetitions += len(pattern.findall(collapsed))
    if repetitions >= 5:
        return True
    unique_tokens = set(re.findall(r"[a-z]{3,}", collapsed))
    if len(unique_tokens) <= 4 and len(collapsed) > 40:
        return True
    return False


def _markdown_contains_only_placeholders(markdown: str) -> bool:
    sections = re.split(r"<!--\s*PageBreak\s*-->", markdown)
    filtered = [section.strip() for section in sections if section.strip()]
    if not filtered:
        return True
    return all(_looks_like_placeholder(section) for section in filtered)


def mistral_document_ai(pdf_path: Path, workspace: Path) -> OCRResult:
    if httpx is None:
        raise ImportError("Install 'httpx' to call the Mistral Document AI endpoint.")
    endpoint = os.environ.get("MISTRAL_DOC_AI_ENDPOINT")
    api_key = os.environ.get("MISTRAL_DOC_AI_KEY")
    model = os.environ.get("MISTRAL_DOC_AI_MODEL", "mistral-document-ai")
    if not endpoint or not api_key:
        raise EnvironmentError(
            "Mistral Document AI requires MISTRAL_DOC_AI_ENDPOINT and MISTRAL_DOC_AI_KEY."
        )

    logger.info(f"Mistral Doc AI: Processing {pdf_path.name} using {model}")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    pages_dir = workspace / "pages"
    page_images = render_pdf_to_images(pdf_path, pages_dir)
    markdown_pages: List[str] = []

    with httpx.Client(timeout=300.0) as client:
        for index, image_path in enumerate(page_images, start=1):
            logger.info(f"Mistral Doc AI: Processing page {index}/{len(page_images)}")
            encoded_page = encode_file(Path(image_path))
            last_error: Optional[Exception] = None
            for attempt in range(1, 4):
                logger.info(f"Mistral Doc AI (Page {index}): Attempt {attempt}/3")
                payload = {
                    "model": model,
                    "document": {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{encoded_page}",
                    },
                    "include_image_base64": False,
                }
                try:
                    start_time = time.time()
                    response = client.post(endpoint, json=payload, headers=headers)
                    duration = time.time() - start_time
                    logger.info(f"Mistral Doc AI (Page {index}): Response status {response.status_code} in {duration:.2f}s")
                    
                    response.raise_for_status()
                    result = response.json()
                    page_sections = _mistral_extract_markdown_pages(result)
                    page_markdown = ""
                    if page_sections:
                        page_markdown = "\n\n".join(page_sections)
                    else:
                        fallback = (
                            result.get("markdown")
                            or result.get("content")
                            or result.get("text")
                            or ""
                        )
                        page_markdown = str(fallback)
                    page_markdown = _normalize_mistral_page_markdown(page_markdown, index)
                    if _looks_like_placeholder(page_markdown):
                        logger.warning(f"Mistral Doc AI (Page {index}): Placeholder content detected.")
                        last_error = RuntimeError("Mistral Document AI returned placeholder content")
                        if attempt < 3:
                            time.sleep(5)
                            continue
                    markdown_pages.append(page_markdown)
                    break
                except Exception as exc:  # pragma: no cover - network operations
                    logger.error(f"Mistral Doc AI (Page {index}): Exception: {exc}")
                    last_error = exc
                    if attempt < 3:
                        time.sleep(5)
                        continue
                    raise RuntimeError(f"Mistral Document AI failed for page {index}: {exc}") from exc
            else:
                raise RuntimeError(f"Mistral Document AI failed for page {index}: {last_error}")

    combined = "\n\n<!-- PageBreak -->\n\n".join(markdown_pages)
    metadata = {"page_count": len(page_images)}
    return OCRResult(engine="mistral", markdown=combined, page_images=page_images, metadata=metadata)


def _mistral_extract_markdown_pages(result: Dict[str, Any]) -> List[str]:
    pages = result.get("pages")
    markdown_pages: List[str] = []

    def collect_text(value: Any) -> List[str]:
        collected: List[str] = []
        if isinstance(value, str) and value.strip():
            collected.append(value.strip())
        elif isinstance(value, dict):
            for key in ("markdown", "text", "content"):
                inner = value.get(key)
                if isinstance(inner, str) and inner.strip():
                    collected.append(inner.strip())
        return collected

    if isinstance(pages, list):
        for index, page in enumerate(pages, start=1):
            if not isinstance(page, dict):
                continue
            parts: List[str] = []
            for header_key in ("page_header", "header", "pageHeader"):
                parts.extend(collect_text(page.get(header_key)))
            primary = page.get("markdown")
            if isinstance(primary, str) and primary.strip():
                parts.append(primary.strip())
            for table in page.get("tables", []) or []:
                parts.extend(collect_text(table))
            for figure in page.get("figures", []) or []:
                parts.extend(collect_text(figure))
            for footer_key in ("page_footer", "footer", "pageFooter"):
                parts.extend(collect_text(page.get(footer_key)))
            if not parts:
                for fallback_key in ("content", "text"):
                    parts.extend(collect_text(page.get(fallback_key)))
            combined = "\n\n".join(part for part in parts if part).strip()
            if combined:
                prefix = f"<!-- Page {index} -->"
                if not combined.startswith(prefix):
                    combined = prefix + "\n" + combined
                markdown_pages.append(combined)

    if not markdown_pages:
        global_markdown = result.get("markdown") or result.get("content") or result.get("text")
        if isinstance(global_markdown, str) and global_markdown.strip():
            sections = re.split(r"\n\s*<!--\s*PageBreak\s*-->\s*\n", global_markdown)
            markdown_pages = [section.strip() for section in sections if section.strip()]
    return markdown_pages


def _normalize_mistral_page_markdown(markdown: str, index: int) -> str:
    stripped = markdown.strip()
    stripped = re.sub(r"^<!--\s*Page\s+\d+\s*-->[\s\n]*", "", stripped, flags=re.IGNORECASE)
    header = f"<!-- Page {index} -->"
    if not stripped:
        return header
    return f"{header}\n{stripped}"


def document_intelligence(pdf_path: Path, workspace: Path) -> OCRResult:
    endpoint = os.environ.get("DOCUMENTINTELLIGENCE_ENDPOINT")
    api_key = os.environ.get("DOCUMENTINTELLIGENCE_KEY")
    if not endpoint or not api_key:
        raise EnvironmentError(
            "Azure Document Intelligence requires DOCUMENTINTELLIGENCE_ENDPOINT and DOCUMENTINTELLIGENCE_KEY."
        )
    
    logger.info(f"Document Intelligence: Processing {pdf_path.name}")
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient  # type: ignore
        from azure.ai.documentintelligence.models import DocumentContentFormat  # type: ignore
        from azure.core.credentials import AzureKeyCredential  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Install azure-ai-documentintelligence to call the Layout model."
        ) from exc

    credential = AzureKeyCredential(api_key)
    client = DocumentIntelligenceClient(endpoint, credential)

    with open(pdf_path, "rb") as fh:
        logger.info("Document Intelligence: Sending analyze request...")
        start_time = time.time()
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=fh,
            output_content_format=DocumentContentFormat.MARKDOWN,
            content_type="application/pdf",
        )
        logger.info("Document Intelligence: Polling for result...")
        result = poller.result()
        duration = time.time() - start_time
        logger.info(f"Document Intelligence: Completed in {duration:.2f}s")

    markdown = getattr(result, "content", "") or ""
    page_markdowns = _split_markdown_pages(markdown)
    combined = _combine_pages_with_headers(page_markdowns)

    pages_dir = workspace / "pages"
    page_images = render_pdf_to_images(pdf_path, pages_dir)
    metadata = {
        "model_id": getattr(result, "model_id", None),
        "model_version": getattr(result, "model_version", None),
    }
    return OCRResult(
        engine="document-intelligence",
        markdown=combined,
        page_images=page_images,
        metadata=metadata,
    )


def _split_markdown_pages(markdown: str) -> List[str]:
    if not markdown:
        return []
    marker = "<!-- PageBreak -->"
    if marker in markdown:
        parts = [part.strip() for part in markdown.split(marker)]
        return [part for part in parts if part]
    return [markdown.strip()]


def _combine_pages_with_headers(page_markdowns: List[str]) -> str:
    enhanced = []
    for index, page_md in enumerate(page_markdowns, start=1):
        header = f"<!-- Page {index} -->"
        enhanced.append(f"{header}\n{page_md.strip()}")
    return "\n\n<!-- PageBreak -->\n\n".join(enhanced)


ENGINE_IMPLEMENTATIONS = {
    "deepseek": deepseek_ocr,
    "mistral": mistral_document_ai,
    "document-intelligence": document_intelligence,
}

SCENARIOS: List[Scenario] = [
    Scenario(llm=llm, ocr=engine, mode=mode)
    for llm in LLM_NAMES
    for engine in ENGINE_IMPLEMENTATIONS.keys()
    for mode in ("text-only", "multi-modal")
]
SCENARIO_LABELS: List[str] = [scenario.label for scenario in SCENARIOS]
SCENARIO_BY_LABEL: Dict[str, Scenario] = {scenario.label: scenario for scenario in SCENARIOS}


def load_cached_result(base_dir: Path) -> Optional[OCRResult]:
    md_path = base_dir / "output.md"
    meta_path = base_dir / "metadata.json"

    if not md_path.exists():
        return None

    markdown = md_path.read_text(encoding="utf-8")
    metadata: Dict[str, Any] = {}
    page_images: List[str] = []

    if meta_path.exists():
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata = raw.get("metadata", {})
        for rel in raw.get("page_images", []):
            page_images.append(str((base_dir / rel).resolve()))
    else:
        pages_dir = base_dir / "pages"
        if pages_dir.exists():
            page_images = [str(path.resolve()) for path in sorted(pages_dir.glob("*.png"))]

    engine = base_dir.parent.name
    return OCRResult(engine=engine, markdown=markdown, page_images=page_images, metadata=metadata)


def cache_result(base_dir: Path, result: OCRResult) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "output.md").write_text(result.markdown, encoding="utf-8")

    rel_images: List[str] = []
    for image in result.page_images:
        path = Path(image)
        try:
            rel_images.append(str(path.relative_to(base_dir)))
        except ValueError:
            # File is outside cache dir; copy it into cache for reproducibility.
            target_dir = base_dir / "pages"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / path.name
            if not target_path.exists():
                target_path.write_bytes(path.read_bytes())
            rel_images.append(str(target_path.relative_to(base_dir)))
    payload = {
        "page_images": rel_images,
        "metadata": result.metadata,
    }
    (base_dir / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def assemble_messages(markdown: str, images: Sequence[str], prompt_text: str, mode: str) -> List[Dict[str, Any]]:
    base_instruction = prompt_text.strip()
    body = markdown.strip()
    payload_text = base_instruction + ("\n\n" + body if body else "")

    if mode == "multi-modal" and images:
        content = [{"type": "text", "text": payload_text}]
        for image_path in images:
            path = Path(image_path)
            if not path.exists():
                continue
            encoded = encode_file(path)
            mime = guess_mime_type(path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )
        return [
            {"role": "system", "content": "You are an information extraction engine."},
            {"role": "user", "content": content},
        ]

    return [
        {"role": "system", "content": "You are an information extraction engine."},
        {"role": "user", "content": payload_text},
    ]


def run_ocrs(
    pdf_path: Path,
    *,
    use_cache: bool = True,
    engines: Optional[Sequence[str]] = None,
) -> Dict[str, OCRResult]:
    results: Dict[str, OCRResult] = {}
    selected = {code for code in engines} if engines is not None else None
    for engine, impl in ENGINE_IMPLEMENTATIONS.items():
        if selected is not None and engine not in selected:
            continue
        engine_dir = ARTIFACT_ROOT / "ocr" / engine / pdf_path.stem
        cached = load_cached_result(engine_dir) if use_cache else None
        if cached is not None:
            if engine in {"deepseek", "mistral"} and _markdown_contains_only_placeholders(cached.markdown):
                logger.info(f"Cache hit for {engine}, but content looks invalid. Re-running.")
                cached = None
            else:
                logger.info(f"Cache hit for {engine}. Skipping OCR.")
                results[engine] = cached
                continue
        
        logger.info(f"Running OCR engine: {engine}")
        result = impl(pdf_path, engine_dir)
        if engine in {"deepseek", "mistral"} and _markdown_contains_only_placeholders(result.markdown):
            logger.error(f"{engine} OCR returned placeholder content; aborting.")
            raise RuntimeError(f"{engine} OCR returned placeholder content; aborting.")
        cache_result(engine_dir, result)
        results[engine] = result
    return results


def run_extractions(
    pdf_path: Path,
    *,
    prompt_text: str,
    expected_text: str,
    scenario_labels: Sequence[str],
    existing_candidates: Optional[Dict[str, Dict[str, Any]]] = None,
    judge_model: str = DEFAULT_JUDGE,
    use_cache: bool = True,
) -> Dict[str, Any]:
    if not scenario_labels:
        raise ValueError("No scenarios provided for execution.")

    logger.info(f"Starting extraction pipeline for {len(scenario_labels)} scenarios.")
    selected_scenarios: List[Scenario] = []
    for label in scenario_labels:
        scenario = SCENARIO_BY_LABEL.get(label)
        if scenario is not None:
            selected_scenarios.append(scenario)
    if not selected_scenarios:
        raise ValueError("None of the requested scenarios are recognized.")

    engines_needed = {scenario.ocr for scenario in selected_scenarios}
    ocr_results = run_ocrs(pdf_path, use_cache=use_cache, engines=engines_needed)
    llm_clients = {name: get_llm_client(name) for name in LLM_NAMES}
    candidate_map: Dict[str, Dict[str, Any]] = {
        label: dict(data) for label, data in (existing_candidates or {}).items()
    }

    for i, scenario in enumerate(selected_scenarios, 1):
        logger.info(f"Processing scenario {i}/{len(selected_scenarios)}: {scenario.label}")
        ocr_result = ocr_results.get(scenario.ocr)
        if ocr_result is None:
            raise ValueError(f"OCR result missing for engine '{scenario.ocr}'.")
        client = llm_clients[scenario.llm]
        timestamp = datetime.now(timezone.utc).isoformat()
        candidate = CandidateOutput(
            llm=scenario.llm,
            ocr=scenario.ocr,
            mode=scenario.mode,
            updated_at=timestamp,
        )
        try:
            messages = assemble_messages(
                ocr_result.markdown,
                ocr_result.page_images,
                prompt_text,
                scenario.mode,
            )
            response = client.generate(messages, temperature=0.0)
            candidate.content = response
            candidate.error = None
        except Exception as exc:  # pragma: no cover - external calls
            logger.error(f"Scenario {scenario.label} failed: {exc}")
            candidate.content = ""
            candidate.error = str(exc)
        candidate_map[candidate.label] = candidate.to_dict()

    ordered_labels = [scenario.label for scenario in SCENARIOS if scenario.label in candidate_map]
    candidate_list = [candidate_map[label] for label in ordered_labels]

    logger.info(f"Running judge model: {judge_model}")
    judge_client = get_llm_client(judge_model)
    leaderboard_html = run_judging(expected_text, candidate_list, judge_client)
    generated_at = datetime.now(timezone.utc).isoformat()

    return {
        "leaderboard_html": leaderboard_html,
        "candidate_details": candidate_list,
        "generated_at": generated_at,
    }


def run_judging(
    expected_text: str,
    candidates: Sequence[Dict[str, Any]],
    judge_client: AzureChatClient,
) -> str:
    if not candidates:
        return "<div class=\"benchmark-report\"><p>No candidate outputs were produced.</p></div>"

    instructions = [
        "You are an impartial judge comparing extraction outputs from multiple OCR + LLM combinations.",
        "Generate a well-formed HTML fragment following these rules:",
        "1. Wrap everything in <div class=\"benchmark-report\"> ... </div>.",
        "2. Include a <h1> element titled 'OCR Benchmark Leaderboard'.",
        "3. Add a <table id=\"leaderboard\"> with columns Rank, Score, LLM, OCR, Mode, Notes.",
        "4. For each candidate, include a <section> with a <h2> heading and summarize its performance.",
        "5. Inside each candidate section, add a <details> block containing a two-column HTML table that shows a diff between the expected output and the candidate output. Use <pre> elements to preserve formatting.",
        "6. Always include every candidate even if its output failed; display the error text and note the failure.",
        "7. Return strictly HTML without Markdown or code fences.",
    ]

    sections: List[str] = [
        "\n".join(instructions),
        "",
        "Expected Output:",
        expected_text.strip(),
        "",
        "Candidate Outputs:",
    ]

    for index, candidate in enumerate(candidates, start=1):
        label = candidate.get("display_label") or candidate.get("label", f"Candidate {index}")
        status = candidate.get("status", "unknown")
        actual_text = candidate.get("error") or candidate.get("content") or ""
        sections.append(f"{label} :: status={status}")
        sections.append(f"LLM={candidate.get('llm')} | OCR={candidate.get('ocr')} | Mode={candidate.get('mode')}")
        sections.append("Actual Output:")
        sections.append(f"```text\n{actual_text}\n```")
        sections.append("")

    prompt = "\n".join(sections).strip()
    messages = [
        {"role": "system", "content": "You are a meticulous evaluation assistant."},
        {"role": "user", "content": prompt},
    ]
    return judge_client.generate(messages, temperature=0.0)


def load_prompt_text(path: Optional[Path] = None) -> str:
    source = path or DEFAULT_PROMPT
    content = Path(source).read_text(encoding="utf-8")
    return content.strip()


def load_expected_text(path: Optional[Path] = None) -> str:
    if path is None:
        path = DEFAULT_EXPECTED
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return raw.strip()


def resolve_pdf_path(path: Optional[Path] = None) -> Path:
    if path is None:
        return DEFAULT_PDF
    return Path(path)


def run_pipeline(
    scenario_label: Optional[str] = None,
    *,
    pdf_path: Optional[Path] = None,
    prompt_path: Optional[Path] = None,
    expected_path: Optional[Path] = None,
    force: bool = False,
    use_cache: bool = True,
) -> Dict[str, Any]:
    if scenario_label is not None and scenario_label not in SCENARIO_BY_LABEL:
        raise ValueError(f"Unknown scenario '{scenario_label}'.")

    if scenario_label is None and not force and PERSISTED_RESULTS.get("leaderboard_html"):
        return get_persisted_results()

    document_path = resolve_pdf_path(pdf_path)
    prompt_text = load_prompt_text(prompt_path)
    expected_text = load_expected_text(expected_path)

    existing_map = _build_candidate_map(PERSISTED_RESULTS)
    selected_labels = [scenario_label] if scenario_label else [scenario.label for scenario in SCENARIOS]

    result = run_extractions(
        document_path,
        prompt_text=prompt_text,
        expected_text=expected_text,
        scenario_labels=selected_labels,
        existing_candidates=existing_map,
        use_cache=use_cache,
    )
    save_persisted_results(result)
    return result


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    if scenario and scenario.lower() == "all":
        scenario = None
    try:
        result = run_pipeline(scenario_label=scenario, force=True)
    except ValueError as exc:
        print(f"Error: {exc}")
        print("Available scenarios:")
        print("- all")
        for label in SCENARIO_LABELS:
            print(f"- {label}")
        return

    html = result.get("leaderboard_html") or "<p>No leaderboard produced.</p>"
    print(html)
    print(f"\nSaved report to: {RESULTS_HTML_PATH}")


if __name__ == "__main__":
    main()


__all__ = [
    "ARTIFACT_ROOT",
    "DEFAULT_EXPECTED",
    "DEFAULT_PDF",
    "DEFAULT_PROMPT",
    "SCENARIO_LABELS",
    "run_pipeline",
    "run_extractions",
    "get_persisted_results",
    "load_prompt_text",
    "load_expected_text",
]
