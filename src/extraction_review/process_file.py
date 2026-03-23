import logging
from datetime import date
from typing import Annotated, Literal

from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.file_query_params import Filter
from pydantic import BaseModel
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
    """A processed document segment with its new name."""

    original_file_id: str
    category: str
    pages: list[int]
    page_count: int
    suggested_name: str
    confidence: str


class ProcessingResult(StopEvent):
    """Final result with all processed segments."""

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
    """Generate document name in format: type_DD_MM_YYYY_Npages."""
    today = date.today()
    date_str = today.strftime("%d_%m_%Y")
    type_name = DOCUMENT_TYPE_NAMES.get(category, "document")
    suffix = f"_{index + 1}" if index > 0 else ""
    return f"{type_name}_{date_str}_{page_count}pages{suffix}"


class ProcessFileWorkflow(Workflow):
    """Process legal documents: split concatenated files and rename by type.

    Accepts documents in various formats (PDF, Word, JPG, etc.),
    identifies logical document boundaries, and generates
    standardized names based on document type.
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
        """Wait for split to complete and generate renamed segments."""
        state = await ctx.store.get_state()
        if state.split_job_id is None:
            raise ValueError("Split job ID cannot be null")

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

        # Track category counts for unique naming
        category_counts: dict[str, int] = {}
        processed_segments: list[ProcessedSegment] = []
        total_pages = 0

        for segment in segments:
            category = segment.category
            pages = segment.pages
            page_count = len(pages)
            total_pages = max(total_pages, max(pages) if pages else 0)

            # Get index for this category (for unique naming)
            index = category_counts.get(category, 0)
            category_counts[category] = index + 1

            # Generate standardized name
            suggested_name = generate_document_name(category, page_count, index)

            processed_segment = ProcessedSegment(
                original_file_id=state.file_id,
                category=category,
                pages=pages,
                page_count=page_count,
                suggested_name=suggested_name,
                confidence=segment.confidence_category or "medium",
            )
            processed_segments.append(processed_segment)

            logger.info(
                f"Segment: {category} (pages {pages}) -> {suggested_name}"
            )

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Found {len(processed_segments)} documents in file",
            )
        )

        # Stream each segment for client
        for seg in processed_segments:
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Document: {seg.suggested_name} (pages {seg.pages})",
                )
            )

        return ProcessingResult(
            segments=processed_segments,
            total_pages=total_pages,
            original_filename=state.filename or "",
        )


workflow = ProcessFileWorkflow(timeout=None)
