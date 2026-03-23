"""
Tests for the legal document processing workflow.
"""

import json
from pathlib import Path

import pytest
from extraction_review.clients import fake

from extraction_review.config import EXTRACTED_DATA_COLLECTION
from extraction_review.metadata_workflow import MetadataResponse
from extraction_review.metadata_workflow import workflow as metadata_workflow
from extraction_review.process_file import FileEvent, ProcessingResult
from extraction_review.process_file import workflow as process_file_workflow
from workflows.events import StartEvent


def get_extraction_schema() -> dict:
    """Load the extraction schema from the unified config file."""
    config_path = Path(__file__).parent.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text())
    return config["extract"]["json_schema"]


def get_split_categories() -> list[dict[str, str]]:
    """Load split categories from the config file."""
    config_path = Path(__file__).parent.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text())
    return [
        {"name": cat["name"], "description": cat["description"]}
        for cat in config["split"]["categories"]
    ]


@pytest.mark.asyncio
async def test_process_file_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")
    if fake is None:
        pytest.skip("Set FAKE_LLAMA_CLOUD=true to enable this test")

    # Load a test file
    file_id = fake.files.preload(path="tests/files/test.pdf")

    # Run the workflow
    result = await process_file_workflow.run(start_event=FileEvent(file_id=file_id))

    # Verify the result structure
    assert result is not None
    assert isinstance(result, ProcessingResult)
    assert isinstance(result.segments, list)
    assert result.original_filename == "test.pdf"

    # Verify each segment has required fields
    for segment in result.segments:
        assert segment.original_file_id == file_id
        assert segment.new_file_id is not None
        assert segment.filename.endswith(".pdf")
        assert segment.page_count == len(segment.pages)


@pytest.mark.asyncio
async def test_metadata_workflow() -> None:
    result = await metadata_workflow.run(start_event=StartEvent())

    assert isinstance(result, MetadataResponse)
    assert result.extracted_data_collection == EXTRACTED_DATA_COLLECTION
    assert result.json_schema == get_extraction_schema()
    assert result.split_categories == get_split_categories()
