"""
Slack Schemas for Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class SlackUploadRequest(BaseModel):
    """Request model for Slack file upload (used for form validation)"""
    channel_name: str = Field(..., description="Slack channel name (e.g., '#new-channel')")
    message: Optional[str] = Field(None, description="Optional message to post with the file")


class SlackUploadResponse(BaseModel):
    """Response model for Slack file upload"""
    success: bool = Field(..., description="Whether the upload was successful")
    message: str = Field(..., description="Status message or error description")
    file_id: Optional[str] = Field(None, description="Slack file ID if upload was successful")
    channel_id: Optional[str] = Field(None, description="Slack channel ID where file was uploaded")
    timestamp: str = Field(..., description="ISO timestamp of the operation")


class ChannelMappingRequest(BaseModel):
    """Request model for adding a new channel mapping"""
    channel_name: str = Field(..., description="Channel name (e.g., '#new-channel')")
    channel_id: str = Field(..., description="Slack channel ID (e.g., 'C09R6GD20T1')")


class ChannelMappingResponse(BaseModel):
    """Response model for channel mapping information"""
    channel_name: str = Field(..., description="Channel name")
    channel_id: str = Field(..., description="Slack channel ID")


class ChannelListResponse(BaseModel):
    """Response model for listing all channel mappings"""
    channels: List[ChannelMappingResponse] = Field(..., description="List of channel mappings")
    count: int = Field(..., description="Total number of channel mappings")


