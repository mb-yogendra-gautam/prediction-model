"""
API Routes for Slack Notifications
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import logging

from src.api.schemas.slack_schemas import (
    SlackUploadResponse,
    ChannelMappingRequest,
    ChannelMappingResponse,
    ChannelListResponse
)
from src.utils.response_formatter import round_numeric_values

logger = logging.getLogger(__name__)
router = APIRouter()

# Global state for service (will be injected)
_slack_service = None


def set_slack_service(service):
    """Set the Slack service instance"""
    global _slack_service
    _slack_service = service


def get_slack_service():
    """Dependency to get Slack service"""
    if _slack_service is None:
        raise HTTPException(status_code=500, detail="Slack service not initialized")
    return _slack_service


@router.post("/upload", response_model=SlackUploadResponse)
async def upload_file_to_slack(
    file: UploadFile = File(..., description="PDF file to upload"),
    channel_name: str = Form(..., description="Slack channel name (e.g., '#new-channel')"),
    message: Optional[str] = Form(None, description="Optional message to post with the file")
):
    """
    Upload a PDF file to a Slack channel
    
    **Parameters:**
    - **file**: PDF file to upload (multipart/form-data)
    - **channel_name**: Name of the Slack channel (e.g., '#new-channel')
    - **message**: Optional message to post with the file
    
    **Returns:**
    - Upload status with file ID and channel information
    
    **Example using curl:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/notifications/slack/upload" \\
      -F "file=@report.pdf" \\
      -F "channel_name=#new-channel" \\
      -F "message=📄 Daily report attached!"
    ```
    """
    try:
        slack_service = get_slack_service()
        
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name is required")
        
        # Check if file is PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported. Please upload a file with .pdf extension"
            )
        
        # Check content type if provided
        if file.content_type and file.content_type not in ['application/pdf', 'application/octet-stream']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type: {file.content_type}. Expected application/pdf"
            )
        
        logger.info(f"Received upload request for file: {file.filename} to channel: {channel_name}")
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Upload to Slack
        result = await slack_service.upload_file(
            file_bytes=file_content,
            filename=file.filename,
            channel_name=channel_name,
            message=message
        )
        
        # Return response
        response = SlackUploadResponse(
            success=result["success"],
            message=result["message"],
            file_id=result.get("file_id"),
            channel_id=result.get("channel_id"),
            timestamp=result["timestamp"]
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return round_numeric_values(response.model_dump())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file to Slack: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/channels", response_model=ChannelListResponse)
async def list_channels():
    """
    Get all available Slack channel mappings
    
    **Returns:**
    - List of channel names and their corresponding Slack channel IDs
    
    **Example:**
    ```json
    {
      "channels": [
        {
          "channel_name": "#new-channel",
          "channel_id": "C09R6GD20T1"
        }
      ],
      "count": 1
    }
    ```
    """
    try:
        slack_service = get_slack_service()
        channels = slack_service.get_all_channels()
        
        return round_numeric_values({
            "channels": channels,
            "count": len(channels)
        })
        
    except Exception as e:
        logger.error(f"Error listing channels: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/channels", response_model=ChannelMappingResponse)
async def add_channel_mapping(request: ChannelMappingRequest):
    """
    Add a new channel name to ID mapping
    
    **Parameters:**
    - **channel_name**: Channel name (e.g., '#new-channel')
    - **channel_id**: Slack channel ID (e.g., 'C09R6GD20T1')
    
    **Returns:**
    - Confirmation of the added channel mapping
    
    **Example:**
    ```json
    {
      "channel_name": "#new-channel",
      "channel_id": "C09R6GD20T1"
    }
    ```
    """
    try:
        slack_service = get_slack_service()
        
        # Validate channel ID format (basic validation)
        if not request.channel_id or len(request.channel_id) < 5:
            raise HTTPException(
                status_code=400,
                detail="Invalid channel ID format. Slack channel IDs are typically 9-11 characters starting with 'C'"
            )
        
        # Add the mapping
        slack_service.add_channel_mapping(
            channel_name=request.channel_name,
            channel_id=request.channel_id
        )
        
        return round_numeric_values({
            "channel_name": request.channel_name if request.channel_name.startswith("#") else f"#{request.channel_name}",
            "channel_id": request.channel_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding channel mapping: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


