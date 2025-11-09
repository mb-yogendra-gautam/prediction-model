"""
Email API Request and Response Schemas
"""

from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


class EmailRequest(BaseModel):
    """Request schema for sending emails"""
    
    to: List[EmailStr] = Field(
        ..., 
        description="List of recipient email addresses",
        min_items=1
    )
    subject: str = Field(
        ..., 
        description="Email subject line",
        min_length=1,
        max_length=200
    )
    body: str = Field(
        ..., 
        description="Email body content (plain text or HTML)",
        min_length=1
    )
    from_email: Optional[EmailStr] = Field(
        None,
        description="Sender email address (uses SendGrid default if not provided)"
    )
    from_name: Optional[str] = Field(
        None,
        description="Sender name"
    )
    cc: Optional[List[EmailStr]] = Field(
        None,
        description="List of CC email addresses"
    )
    bcc: Optional[List[EmailStr]] = Field(
        None,
        description="List of BCC email addresses"
    )
    is_html: bool = Field(
        False,
        description="Whether the body content is HTML"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "to": ["recipient@example.com"],
                "subject": "Test Email",
                "body": "This is a test email with an attachment.",
                "from_email": "sender@example.com",
                "from_name": "Sender Name",
                "is_html": False
            }
        }


class EmailResponse(BaseModel):
    """Response schema for email sending"""
    
    success: bool = Field(
        ...,
        description="Whether the email was sent successfully"
    )
    message: str = Field(
        ...,
        description="Status message"
    )
    message_id: Optional[str] = Field(
        None,
        description="SendGrid message ID (if successful)"
    )
    recipients: List[str] = Field(
        ...,
        description="List of recipient email addresses"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Email sent successfully",
                "message_id": "abc123xyz",
                "recipients": ["recipient@example.com"]
            }
        }


class EmailError(BaseModel):
    """Error response schema"""
    
    success: bool = Field(
        False,
        description="Always false for errors"
    )
    error: str = Field(
        ...,
        description="Error message"
    )
    details: Optional[str] = Field(
        None,
        description="Additional error details"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Failed to send email",
                "details": "Invalid API key"
            }
        }

