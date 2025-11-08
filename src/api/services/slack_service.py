"""
Slack Service for uploading files to Slack channels using Slack Python SDK
"""

import logging
import os
from typing import Dict, Optional, List
from datetime import datetime
from dotenv import load_dotenv
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

load_dotenv()

logger = logging.getLogger(__name__)


class SlackService:
    """Service for managing Slack file uploads and channel mappings using Slack SDK"""
    
    def __init__(self, slack_token: Optional[str] = None):
        """
        Initialize Slack service with Slack Python SDK
        
        Args:
            slack_token: Slack Bot token (xoxb-...). If not provided, reads from SLACK_BOT_TOKEN env var
        """
        self.slack_token = slack_token or os.getenv("SLACK_BOT_TOKEN")
        logger.info(f"Slack token: {'found' if self.slack_token else 'not found'}")
        
        if not self.slack_token:
            logger.warning("SLACK_BOT_TOKEN not found in environment variables")
            raise ValueError("Slack Bot token must be provided via SLACK_BOT_TOKEN environment variable")
        
        # Initialize Slack AsyncWebClient
        self.client = AsyncWebClient(token=self.slack_token)
        
        # In-memory channel name to ID mapping
        self.channel_mappings: Dict[str, str] = {
            "#new-channel": "C09R6GD20T1"
        }
        
        logger.info(f"SlackService initialized with Slack SDK and {len(self.channel_mappings)} channel mappings")
    
    def add_channel_mapping(self, channel_name: str, channel_id: str) -> None:
        """
        Add or update a channel name to ID mapping
        
        Args:
            channel_name: Channel name (e.g., "#new-channel")
            channel_id: Slack channel ID (e.g., "C09R6GD20T1")
        """
        # Ensure channel name starts with #
        if not channel_name.startswith("#"):
            channel_name = f"#{channel_name}"
        
        self.channel_mappings[channel_name] = channel_id
        logger.info(f"Added channel mapping: {channel_name} -> {channel_id}")
    
    def get_channel_id(self, channel_name: str) -> Optional[str]:
        """
        Get channel ID from channel name
        
        Args:
            channel_name: Channel name (e.g., "#new-channel")
            
        Returns:
            Channel ID if found, None otherwise
        """
        # Ensure channel name starts with #
        if not channel_name.startswith("#"):
            channel_name = f"#{channel_name}"
        
        return self.channel_mappings.get(channel_name)
    
    def get_all_channels(self) -> List[Dict[str, str]]:
        """
        Get all channel mappings
        
        Returns:
            List of dictionaries with channel_name and channel_id
        """
        return [
            {"channel_name": name, "channel_id": channel_id}
            for name, channel_id in self.channel_mappings.items()
        ]
    
    async def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        channel_name: str,
        message: Optional[str] = None
    ) -> Dict:
        """
        Upload a file to a Slack channel using Slack SDK's files_upload_v2 method
        (files.upload was deprecated in March 2025)
        
        This method uses the simplified files_upload_v2 which handles all the complexity
        of the 3-step upload process internally.
        
        Args:
            file_bytes: File content as bytes
            filename: Name of the file
            channel_name: Channel name (e.g., "#new-channel")
            message: Optional initial comment/message to post with the file
            
        Returns:
            Dictionary with upload response containing:
            - success: bool
            - message: str
            - file_id: Optional[str]
            - channel_id: Optional[str]
            - slack_response: Optional[Dict] (full Slack API response)
            
        Raises:
            ValueError: If channel name is not found in mappings
            SlackApiError: If the Slack API request fails
        """
        # Get channel ID from mapping
        channel_id = self.get_channel_id(channel_name)
        
        if not channel_id:
            available_channels = ", ".join(self.channel_mappings.keys())
            error_msg = f"Channel '{channel_name}' not found in mappings. Available channels: {available_channels}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "file_id": None,
                "channel_id": None,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            file_size = len(file_bytes)
            logger.info(f"Uploading file '{filename}' ({file_size} bytes) to channel {channel_name} ({channel_id}) using SDK")
            
            # Use files_upload_v2 which handles the 3-step process internally
            # This is the recommended approach per Slack SDK documentation
            if message:
                response = await self.client.files_upload_v2(
                    file=file_bytes,
                    filename=filename,
                    title=filename,
                    channel=channel_id,
                    initial_comment=message
                )
            else:
                response = await self.client.files_upload_v2(
                    file=file_bytes,
                    filename=filename,
                    title=filename,
                    channel=channel_id
                )
            
            # Check if upload was successful
            if not response.get("ok"):
                error = response.get("error", "Unknown error")
                logger.error(f"File upload failed - Slack API error: {error}")
                
                return {
                    "success": False,
                    "message": f"Failed to upload file: {error}",
                    "file_id": None,
                    "channel_id": channel_id,
                    "timestamp": datetime.now().isoformat(),
                    "slack_response": response.data
                }
            
            # Extract file ID from response
            file_id = response.get("file", {}).get("id") if response.get("file") else None
            
            logger.info(f"File '{filename}' uploaded successfully to channel {channel_name}. File ID: {file_id}")
            
            return {
                "success": True,
                "message": f"File '{filename}' uploaded and shared successfully to {channel_name}",
                "file_id": file_id,
                "channel_id": channel_id,
                "timestamp": datetime.now().isoformat(),
                "slack_response": response.data
            }
                    
        except SlackApiError as e:
            error_msg = f"Slack API error during file upload: {e.response['error']}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "message": error_msg,
                "file_id": None,
                "channel_id": channel_id,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            error_msg = f"Unexpected error during file upload: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "message": error_msg,
                "file_id": None,
                "channel_id": channel_id,
                "timestamp": datetime.now().isoformat()
            }

