import io
import logging
from datetime import date
from typing import Annotated, Literal

import httpx
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.file_query_params import Filter
from PIL import Image
from pydantic import BaseModel
from pypdf import PdfReader, PdfWriter
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import get_llama_cloud_client, project_id
from .config import DOCUMENT_TYPE_NAMES, SplitConfig

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"}


class FileEvent(StartEvent):
    """Input event with file ID(s) for processing.

    Use file_id for single file, or file_ids for multiple files to merge.
    """

    file_id: str | None = None
    file_ids: list[str] | None = None
    file_hash: str | None = None


class Status(Event):
    """Status update event for client notifications."""

    level: Literal["info", "warning", "error"]
    message: str


class SplitJobStartedEvent(Event):
    """Internal event after split job is started."""

    pass


class ProcessedSegment(BaseModel):
    """A processed document segment with its new name and file ID."""

    original_file_id: str
    new_file_id: str
    category: str
    pages: list[int]
    page_count: int
    filename: str
    confidence: str


class ProcessingResult(StopEvent):
    """Final result with all processed and split segments."""

    segments: list[ProcessedSegment]
    total_pages: int
    original_filename: str
    merged_from_files: int


class ProcessingState(BaseModel):
    """Workflow state for document processing."""

    file_id: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    split_job_id: str | None = None
    source_file_ids: list[str] = []
    merged_file_count: int = 0


def generate_document_name(category: str, page_count: int, index: int = 0) -> str:
    """Generate document name in format: type_DD_MM_YYYY_Npages.pdf."""
    today = date.today()
    date_str = today.strftime("%d_%m_%Y")
    type_name = DOCUMENT_TYPE_NAMES.get(category, "document")
    suffix = f"_{index + 1}" if index > 0 else ""
    return f"{type_name}_{date_str}_{page_count}pages{suffix}.pdf"


async def download_file(client: AsyncLlamaCloud, file_id: str) -> bytes:
    """Download file content from LlamaCloud."""
    content_info = await client.files.get(file_id)
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(content_info.url)
        response.raise_for_status()
        return response.content


def is_image_file(filename: str) -> bool:
    """Check if file is an image based on extension."""
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_pdf_file(filename: str) -> bool:
    """Check if file is a PDF based on extension."""
    return filename.lower().endswith(".pdf")


def image_to_pdf(image_bytes: bytes) -> bytes:
    """Convert image bytes to PDF bytes."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    output = io.BytesIO()
    image.save(output, format="PDF", resolution=100.0)
    return output.getvalue()


def merge_pdfs(pdf_contents: list[bytes]) -> bytes:
    """Merge multiple PDF byte contents into single PDF."""
    writer = PdfWriter()
    for pdf_bytes in pdf_contents:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            writer.add_page(page)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def extract_pages(pdf_content: bytes, pages: list[int]) -> tuple[bytes, list[int]]:
    """Extract specific pages from PDF and return as new PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_content))
    writer = PdfWriter()
    total_pages = len(reader.pages)
    extracted_pages: list[int] = []

    for page_num in pages:
        if 1 <= page_num <= total_pages:
            writer.add_page(reader.pages[page_num - 1])
            extracted_pages.append(page_num)

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue(), extracted_pages


class ProcessFileWorkflow(Workflow):
    """Process legal documents: merge images, split concatenated files.

    Supports two modes:
    - Single file (file_id): split into separate documents by type
    - Multiple files (file_ids): merge into one PDF, then split by type
    """

    @step()
    async def start_processing(
        self,
        event: FileEvent,
        ctx: Context[ProcessingState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        split_config: Annotated[
            SplitConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="split",
                label="Document Categories",
                description="Categories for identifying different types of legal documents",
            ),
        ],
    ) -> SplitJobStartedEvent:
        """Process input files: merge if multiple, then start splitting."""
        # Determine which mode we're in
        if event.file_ids and len(event.file_ids) > 1:
            # Multiple files mode: merge first
            file_ids = event.file_ids
            logger.info(f"Merging {len(file_ids)} files")

            ctx.write_event_to_stream(
                Status(level="info", message=f"Merging {len(file_ids)} files...")
            )

            files_response = await llama_cloud_client.files.query(
                filter=Filter(file_ids=file_ids)
            )
            files_by_id = {f.id: f for f in files_response.items}

            pdf_contents: list[bytes] = []
            for fid in file_ids:
                file_meta = files_by_id.get(fid)
                if not file_meta:
                    logger.warning(f"File {fid} not found, skipping")
                    continue

                filename = file_meta.name
                content = await download_file(llama_cloud_client, fid)

                if is_image_file(filename):
                    logger.info(f"Converting image to PDF: {filename}")
                    pdf_content = image_to_pdf(content)
                elif is_pdf_file(filename):
                    pdf_content = content
                else:
                    logger.info(f"Attempting to convert as image: {filename}")
                    pdf_content = image_to_pdf(content)

                pdf_contents.append(pdf_content)

            merged_pdf = merge_pdfs(pdf_contents)
            today = date.today().strftime("%d_%m_%Y")
            merged_filename = f"merged_document_{today}.pdf"

            merged_file = await llama_cloud_client.files.create(
                file=(merged_filename, merged_pdf, "application/pdf"),
                purpose="split",
                project_id=project_id,
            )

            logger.info(f"Created merged PDF: {merged_filename} ({merged_file.id})")
            ctx.write_event_to_stream(
                Status(level="info", message=f"Created merged document ({len(pdf_contents)} files)")
            )

            file_id = merged_file.id
            filename = merged_filename
            merged_count = len(file_ids)
            file_hash = None

        else:
            # Single file mode
            file_id = event.file_id or (event.file_ids[0] if event.file_ids else None)
            if not file_id:
                raise ValueError("Either file_id or file_ids must be provided")

            logger.info(f"Processing single file {file_id}")

            files_response = await llama_cloud_client.files.query(
                filter=Filter(file_ids=[file_id])
            )
            file_metadata = files_response.items[0]
            filename = file_metadata.name
            merged_count = 1
            file_hash = event.file_hash or file_metadata.external_file_id

            logger.info(f"Processing document: {filename}")

        ctx.write_event_to_stream(
            Status(level="info", message=f"Analyzing document structure: {filename}")
        )

        categories = [
            {"name": cat.name, "description": cat.description}
            for cat in split_config.categories
        ]

        split_job = await llama_cloud_client.beta.split.create(
            categories=categories,
            document_input={"type": "file_id", "value": file_id},
            splitting_strategy={
                "allow_uncategorized": split_config.settings.splitting_strategy.allow_uncategorized
            },
            project_id=project_id,
        )

        logger.info(f"Started split job: {split_job.id}")

        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.filename = filename
            state.file_hash = file_hash
            state.split_job_id = split_job.id
            state.merged_file_count = merged_count

        return SplitJobStartedEvent()

    @step()
    async def complete_processing(
        self,
        event: SplitJobStartedEvent,
        ctx: Context[ProcessingState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
    ) -> ProcessingResult:
        """Wait for split, extract pages, and upload separate PDF files."""
        state = await ctx.store.get_state()
        if state.split_job_id is None or state.file_id is None:
            raise ValueError("Split job ID and file ID cannot be null")

        ctx.write_event_to_stream(
            Status(level="info", message="Identifying document boundaries...")
        )

        completed_job = await llama_cloud_client.beta.split.wait_for_completion(
            state.split_job_id,
            polling_interval=1.0,
        )

        if completed_job.status != "completed" or completed_job.result is None:
            error_msg = f"Split job failed: {completed_job.status}"
            logger.error(error_msg)
            ctx.write_event_to_stream(Status(level="error", message=error_msg))
            raise RuntimeError(error_msg)

        segments = completed_job.result.segments
        logger.info(f"Found {len(segments)} document segments")

        ctx.write_event_to_stream(
            Status(level="info", message="Downloading file for splitting...")
        )
        pdf_content = await download_file(llama_cloud_client, state.file_id)

        category_counts: dict[str, int] = {}
        processed_segments: list[ProcessedSegment] = []
        total_pages = 0

        ctx.write_event_to_stream(
            Status(level="info", message=f"Splitting into {len(segments)} documents...")
        )

        for segment in segments:
            category = segment.category
            pages = segment.pages
            page_count = len(pages)
            total_pages = max(total_pages, max(pages) if pages else 0)

            index = category_counts.get(category, 0)
            category_counts[category] = index + 1

            segment_pdf, extracted_pages = extract_pages(pdf_content, pages)

            if not extracted_pages:
                logger.warning(f"Skipping segment {category}: no valid pages")
                continue

            actual_page_count = len(extracted_pages)
            new_filename = generate_document_name(category, actual_page_count, index)

            new_file = await llama_cloud_client.files.create(
                file=(new_filename, segment_pdf, "application/pdf"),
                purpose="extract",
                project_id=project_id,
            )

            processed_segment = ProcessedSegment(
                original_file_id=state.file_id,
                new_file_id=new_file.id,
                category=category,
                pages=extracted_pages,
                page_count=actual_page_count,
                filename=new_filename,
                confidence=segment.confidence_category or "medium",
            )
            processed_segments.append(processed_segment)

            logger.info(f"Created: {new_filename} (file_id: {new_file.id})")

            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Created: {new_filename} (pages {extracted_pages})",
                )
            )

        return ProcessingResult(
            segments=processed_segments,
            total_pages=total_pages,
            original_filename=state.filename or "",
            merged_from_files=state.merged_file_count,
        )


workflow = ProcessFileWorkflow(timeout=None)
