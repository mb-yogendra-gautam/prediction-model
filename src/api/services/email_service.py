"""
Email Service using SendGrid API
"""

import os
import base64
import logging
from typing import Optional, List, Dict, Any
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail, 
    Email, 
    To, 
    Content, 
    Attachment, 
    FileContent, 
    FileName, 
    FileType, 
    Disposition
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class EmailService:
    """
    Service for sending emails with attachments using SendGrid
    """
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    
    def __init__(self, api_key: Optional[str] = None, default_from_email: Optional[str] = None):
        """
        Initialize EmailService with SendGrid API key
        
        Args:
            api_key: SendGrid API key (reads from SENDGRID_API_KEY env var if not provided)
            default_from_email: Default sender email (reads from SENDGRID_FROM_EMAIL env var if not provided)
        
        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        self.api_key = api_key or os.getenv("SENDGRID_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SendGrid API key not found. Please set SENDGRID_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.default_from_email = default_from_email or os.getenv("SENDGRID_FROM_EMAIL", "noreply@example.com")
        self.client = SendGridAPIClient(self.api_key)
        logger.info("EmailService initialized successfully")
    
    def validate_file_size(self, file_content: bytes, filename: str) -> None:
        """
        Validate that file size is within limits
        
        Args:
            file_content: File content as bytes
            filename: Name of the file
            
        Raises:
            ValueError: If file size exceeds limit
        """
        file_size = len(file_content)
        if file_size > self.MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            max_size_mb = self.MAX_FILE_SIZE / (1024 * 1024)
            raise ValueError(
                f"File '{filename}' size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
            )
        logger.info(f"File '{filename}' size validated: {file_size / 1024:.2f}KB")
    
    def create_attachment(
        self, 
        file_content: bytes, 
        filename: str, 
        content_type: str = "application/octet-stream"
    ) -> Attachment:
        """
        Create a SendGrid attachment from file content
        
        Args:
            file_content: File content as bytes
            filename: Name of the file
            content_type: MIME type of the file
            
        Returns:
            SendGrid Attachment object
        """
        # Encode file content to base64
        encoded_content = base64.b64encode(file_content).decode()
        
        # Create attachment
        attachment = Attachment()
        attachment.file_content = FileContent(encoded_content)
        attachment.file_name = FileName(filename)
        attachment.file_type = FileType(content_type)
        attachment.disposition = Disposition("attachment")
        
        logger.info(f"Created attachment: {filename} ({content_type})")
        return attachment
    
    async def send_email(
        self,
        to: List[str],
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        is_html: bool = False,
        attachment_content: Optional[bytes] = None,
        attachment_filename: Optional[str] = None,
        attachment_content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send an email via SendGrid
        
        Args:
            to: List of recipient email addresses
            subject: Email subject
            body: Email body content
            from_email: Sender email (uses default if not provided)
            from_name: Sender name
            cc: List of CC recipients
            bcc: List of BCC recipients
            is_html: Whether body is HTML content
            attachment_content: File content as bytes
            attachment_filename: Name of the attachment file
            attachment_content_type: MIME type of attachment
            
        Returns:
            Dict with success status, message, and message_id
            
        Raises:
            ValueError: If file size exceeds limit
            Exception: If SendGrid API call fails
        """
        try:
            # Validate attachment if provided
            if attachment_content and attachment_filename:
                self.validate_file_size(attachment_content, attachment_filename)
            
            # Set sender
            sender_email = from_email or self.default_from_email
            from_email_obj = Email(sender_email, from_name)
            
            # Set recipients
            to_emails = [To(email) for email in to]
            
            # Set content type
            content_type = "text/html" if is_html else "text/plain"
            content = Content(content_type, body)
            
            # Create mail object
            message = Mail(
                from_email=from_email_obj,
                to_emails=to_emails,
                subject=subject,
                plain_text_content=content if not is_html else None,
                html_content=content if is_html else None
            )
            
            # Add CC recipients
            if cc:
                for cc_email in cc:
                    message.add_cc(cc_email)
            
            # Add BCC recipients
            if bcc:
                for bcc_email in bcc:
                    message.add_bcc(bcc_email)
            
            # Add attachment if provided
            if attachment_content and attachment_filename:
                content_type = attachment_content_type or "application/octet-stream"
                attachment = self.create_attachment(
                    attachment_content, 
                    attachment_filename, 
                    content_type
                )
                message.add_attachment(attachment)
            
            # Send email
            logger.info(f"Sending email to {len(to)} recipient(s): {to}")
            response = self.client.send(message)
            
            # Check response
            if response.status_code in [200, 201, 202]:
                logger.info(f"Email sent successfully. Status: {response.status_code}")
                return {
                    "success": True,
                    "message": "Email sent successfully",
                    "message_id": response.headers.get("X-Message-Id"),
                    "status_code": response.status_code
                }
            else:
                logger.error(f"SendGrid returned unexpected status: {response.status_code}")
                return {
                    "success": False,
                    "message": f"Failed to send email. Status: {response.status_code}",
                    "status_code": response.status_code
                }
                
        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error sending email: {e}", exc_info=True)
            raise Exception(f"Failed to send email: {str(e)}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get email service status
        
        Returns:
            Dict with service configuration and status
        """
        return {
            "service": "SendGrid",
            "configured": bool(self.api_key),
            "default_from_email": self.default_from_email,
            "max_file_size_mb": self.MAX_FILE_SIZE / (1024 * 1024)
        }

