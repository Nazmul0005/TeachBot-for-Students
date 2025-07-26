from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ExtractedTextMetadata(BaseModel):
    """Metadata for extracted text"""
    file_size: Optional[int] = Field(None, description="Size of the file in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    page_count: Optional[int] = Field(None, description="Number of pages in the document")
    extraction_method: Optional[str] = Field(None, description="Method used for text extraction")
    processing_time: Optional[float] = Field(None, description="Time taken to process the document")
    chunks_processed: Optional[int] = Field(None, description="Number of chunks processed")

class SuccessfulFileResult(BaseModel):
    """Result for a successfully processed file"""
    file_name: str = Field(..., description="Name of the processed file")
    text: str = Field(..., description="Extracted text content")
    text_length: int = Field(..., description="Length of extracted text")
    saved_to: Optional[str] = Field(None, description="Path where the text was saved")

class FailedFileResult(BaseModel):
    """Result for a failed file"""
    file_name: str = Field(..., description="Name of the failed file")
    error: str = Field(..., description="Error message")

class SavedFileInfo(BaseModel):
    """Information about saved text files"""
    original_file: str = Field(..., description="Original filename")
    saved_path: str = Field(..., description="Path where text was saved")
    text_length: int = Field(..., description="Length of extracted text")

class ExtractionSummary(BaseModel):
    """Summary of extraction results"""
    total_files_processed: int = Field(..., description="Total number of files processed")
    successful_extractions: int = Field(..., description="Number of successful extractions")
    failed_extractions: int = Field(..., description="Number of failed extractions")
    files_saved_to_disk: int = Field(..., description="Number of files saved to disk")
    total_characters_extracted: int = Field(..., description="Total characters extracted")

class ExtractExtractionData(BaseModel):
    """Data structure for extraction response"""
    total_files: int = Field(..., description="Total number of files processed")
    successful_files: List[SuccessfulFileResult] = Field(default_factory=list, description="List of successfully processed files")
    failed_files: List[FailedFileResult] = Field(default_factory=list, description="List of failed files")
    saved_files: List[SavedFileInfo] = Field(default_factory=list, description="List of files saved to disk")
    summary: Optional[ExtractionSummary] = Field(None, description="Summary of extraction results")

class DocumentExtractionResponse(BaseModel):
    """Complete response for document extraction request matching NetworkResponse structure"""
    success: bool = Field(..., description="Overall success status")
    message: str = Field(..., description="Status message")
    data: ExtractExtractionData = Field(..., description="Extraction results")
    resource: str = Field(..., description="API endpoint resource path")
    duration: str = Field(..., description="Processing duration")

class SavedFileListItem(BaseModel):
    """Information about a saved file in the list"""
    filename: str = Field(..., description="Name of the saved file")
    path: str = Field(..., description="Full path to the saved file")
    size_bytes: int = Field(..., description="File size in bytes")
    created: str = Field(..., description="Creation timestamp")
    modified: str = Field(..., description="Last modification timestamp")

class SavedFilesListResponse(BaseModel):
    """Response for listing saved files"""
    message: str = Field(..., description="Response message")
    total_files: int = Field(..., description="Total number of saved files")
    files: List[SavedFileListItem] = Field(..., description="List of saved files")

class DeleteFileResponse(BaseModel):
    """Response for deleting a file"""
    message: str = Field(..., description="Response message")
    deleted_file: str = Field(..., description="Name of the deleted file")