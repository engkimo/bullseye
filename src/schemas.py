from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, model_validator
from typing import List, Union


class ProcessOptions(BaseModel):
    detect_layout: bool = False
    detect_tables: bool = False
    extract_reading_order: bool = False
    enable_llm: bool = False
    lite: bool = False
    output_format: Literal['json', 'md', 'html', 'csv', 'pdf'] = 'json'
    vis: bool = False
    max_pages: Optional[int] = Field(default=None, ge=1)
    # LLM task options (optional)
    llm_task: Optional[Literal['summary', 'qa', 'json']] = 'summary'
    llm_question: Optional[str] = None
    llm_schema: Optional[Dict[str, Any]] = None


class ProcessJSONRequest(BaseModel):
    file_base64: Optional[str] = Field(default=None, description="Base64 encoded file content")
    file_url: Optional[HttpUrl] = Field(default=None, description="Public URL to fetch the file")
    filename: Optional[str] = Field(default=None, description="Original filename (optional)")
    options: Optional[ProcessOptions] = Field(default_factory=ProcessOptions)

    @model_validator(mode='after')
    def validate_source(self):
        if not self.file_base64 and not self.file_url:
            raise ValueError('Either file_base64 or file_url must be provided')
        return self


class DocumentResponse(BaseModel):
    filename: str
    pages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    vis_previews: Optional[List[str]] = None


class ProcessArtifactResponse(BaseModel):
    content_type: str
    file_name: Optional[str] = None
    size_bytes: Optional[int] = None
    artifact_text: Optional[str] = None
    artifact_base64: Optional[str] = None
    vis_previews: Optional[List[str]] = None
