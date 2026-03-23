import io
import logging
from datetime import date
from typing import Annotated, Literal

import httpx
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.file_query_params import Filter
from pydantic import BaseModel
from pypdf import PdfReader, PdfWriter
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import get_llama_cloud_client, project_id
from .config import DOCUMENT_TYPE_NAMES, SplitConfig

logger = logging.getLogger(__name__)


class FileEvent(StartEvent):
    """Input event with file ID for processing."""

    file_id: str
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


class ProcessingState(BaseModel):
    """Workflow state for document processing."""

    file_id: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    split_job_id: str | None = None


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


def extract_pages(pdf_content: bytes, pages: list[int]) -> tuple[bytes, list[int]]:
    """Extract specific pages from PDF and return as new PDF bytes.

    Args:
        pdf_content: Original PDF file content
        pages: List of 1-indexed page numbers to extract

    Returns:
        Tuple of (PDF bytes, list of actually extracted pages)
    """
    reader = PdfReader(io.BytesIO(pdf_content))
    writer = PdfWriter()
    total_pages = len(reader.pages)
    extracted_pages: list[int] = []

    for page_num in pages:
        # Skip pages that don't exist in the PDF
        if 1 <= page_num <= total_pages:
            writer.add_page(reader.pages[page_num - 1])
            extracted_pages.append(page_num)

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue(), extracted_pages


class ProcessFileWorkflow(Workflow):
    """Process legal documents: split concatenated files into separate PDFs.

    Accepts documents in various formats, identifies logical document
    boundaries, splits into separate PDF files, and renames them
    by document type.
    """

    @step()
    async def start_split(
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
        """Upload document and start splitting by document type."""
        file_id = event.file_id
        logger.info(f"Processing file {file_id}")

        # Get file metadata via query
        files_response = await llama_cloud_client.files.query(
            filter=Filter(file_ids=[file_id])
        )
        file_metadata = files_response.items[0]
        filename = file_metadata.name
        logger.info(f"Processing document: {filename}")

        ctx.write_event_to_stream(
            Status(level="info", message=f"Analyzing document structure: {filename}")
        )

        # Prepare split categories from config
        categories = [
            {"name": cat.name, "description": cat.description}
            for cat in split_config.categories
        ]

        # Start split job to identify document boundaries
        split_job = await llama_cloud_client.beta.split.create(
            categories=categories,
            document_input={"type": "file_id", "value": file_id},
            splitting_strategy={
                "allow_uncategorized": split_config.settings.splitting_strategy.allow_uncategorized
            },
            project_id=project_id,
        )

        logger.info(f"Started split job: {split_job.id}")

        # Save state
        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.filename = filename
            state.file_hash = event.file_hash or file_metadata.external_file_id
            state.split_job_id = split_job.id

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

        # Wait for split job completion
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

        # Download original PDF for splitting
        ctx.write_event_to_stream(
            Status(level="info", message="Downloading original file...")
        )
        pdf_content = await download_file(llama_cloud_client, state.file_id)

        # Track category counts for unique naming
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

            # Get index for this category (for unique naming)
            index = category_counts.get(category, 0)
            category_counts[category] = index + 1

            # Generate standardized filename
            new_filename = generate_document_name(category, page_count, index)

            # Extract pages and create new PDF
            segment_pdf, extracted_pages = extract_pages(pdf_content, pages)

            # Skip if no pages were extracted
            if not extracted_pages:
                logger.warning(f"Skipping segment {category}: no valid pages")
                continue

            # Update filename with actual page count
            actual_page_count = len(extracted_pages)
            new_filename = generate_document_name(category, actual_page_count, index)

            # Upload new PDF file
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
                    message=f"Created: {new_filename} (pages {pages})",
                )
            )

        return ProcessingResult(
            segments=processed_segments,
            total_pages=total_pages,
            original_filename=state.filename or "",
        )


workflow = ProcessFileWorkflow(timeout=None)
