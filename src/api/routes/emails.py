"""
Email Routes for sending emails with attachments
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from src.api.schemas.email_schemas import EmailResponse, EmailError
from src.api.services.email_service import EmailService

logger = logging.getLogger(__name__)

router = APIRouter()

# Global email service instance (set during app startup)
_email_service: Optional[EmailService] = None


def set_email_service(service: EmailService):
    """Set the email service instance"""
    global _email_service
    _email_service = service
    logger.info("Email service set in router")


def get_email_service() -> EmailService:
    """Get the email service instance"""
    if _email_service is None:
        raise HTTPException(
            status_code=503,
            detail="Email service not initialized. Please check server configuration."
        )
    return _email_service


@router.post(
    "/send",
    response_model=EmailResponse,
    responses={
        200: {"model": EmailResponse, "description": "Email sent successfully"},
        400: {"model": EmailError, "description": "Bad request - validation error"},
        413: {"model": EmailError, "description": "File too large"},
        500: {"model": EmailError, "description": "Server error"},
        503: {"model": EmailError, "description": "Service unavailable"}
    },
    summary="Send email with optional attachment",
    description="""
    Send an email with an optional file attachment using SendGrid.
    
    - Maximum file size: 10MB
    - Supports multiple recipients, CC, and BCC
    - Supports both plain text and HTML content
    - File attachment is optional
    """
)
async def send_email(
    to: str = Form(..., description="Comma-separated list of recipient email addresses"),
    subject: str = Form(..., description="Email subject line"),
    body: str = Form(..., description="Email body content"),
    from_email: Optional[str] = Form(None, description="Sender email address"),
    from_name: Optional[str] = Form(None, description="Sender name"),
    cc: Optional[str] = Form(None, description="Comma-separated list of CC email addresses"),
    bcc: Optional[str] = Form(None, description="Comma-separated list of BCC email addresses"),
    is_html: bool = Form(False, description="Whether the body content is HTML"),
    attachment: Optional[UploadFile] = File(None, description="File attachment (max 10MB)"),
    email_service: EmailService = Depends(get_email_service)
):
    """
    Send an email with optional attachment
    
    Args:
        to: Comma-separated recipient emails (e.g., "user1@example.com,user2@example.com")
        subject: Email subject
        body: Email body (plain text or HTML)
        from_email: Sender email (optional, uses default if not provided)
        from_name: Sender name (optional)
        cc: Comma-separated CC emails (optional)
        bcc: Comma-separated BCC emails (optional)
        is_html: Whether body is HTML (default: False)
        attachment: File attachment (optional, max 10MB)
        email_service: Email service dependency (injected)
    
    Returns:
        EmailResponse with success status and details
    """
    try:
        # Parse email addresses
        to_list = [email.strip() for email in to.split(",") if email.strip()]
        if not to_list:
            raise HTTPException(
                status_code=400,
                detail="At least one recipient email address is required"
            )
        
        cc_list = [email.strip() for email in cc.split(",") if email.strip()] if cc else None
        bcc_list = [email.strip() for email in bcc.split(",") if email.strip()] if bcc else None
        
        # Validate email addresses (basic validation)
        for email_list, field_name in [(to_list, "to"), (cc_list, "cc"), (bcc_list, "bcc")]:
            if email_list:
                for email in email_list:
                    if "@" not in email or "." not in email:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid email address in {field_name}: {email}"
                        )
        
        # Validate subject and body
        if not subject or len(subject.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Subject cannot be empty"
            )
        
        if not body or len(body.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Body cannot be empty"
            )
        
        # Read attachment if provided
        attachment_content = None
        attachment_filename = None
        attachment_content_type = None
        
        if attachment:
            logger.info(f"Processing attachment: {attachment.filename} ({attachment.content_type})")
            
            # Read file content
            attachment_content = await attachment.read()
            attachment_filename = attachment.filename
            attachment_content_type = attachment.content_type
            
            # Check file size before processing
            file_size_mb = len(attachment_content) / (1024 * 1024)
            if len(attachment_content) > email_service.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size (10MB)"
                )
        
        # Send email
        logger.info(f"Sending email to {len(to_list)} recipient(s)")
        result = await email_service.send_email(
            to=to_list,
            subject=subject,
            body=body,
            from_email=from_email,
            from_name=from_name,
            cc=cc_list,
            bcc=bcc_list,
            is_html=is_html,
            attachment_content=attachment_content,
            attachment_filename=attachment_filename,
            attachment_content_type=attachment_content_type
        )
        
        if result["success"]:
            return EmailResponse(
                success=True,
                message=result["message"],
                message_id=result.get("message_id"),
                recipients=to_list
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
            
    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error sending email: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email: {str(e)}"
        )


@router.get(
    "/status",
    summary="Get email service status",
    description="Check the status and configuration of the email service"
)
async def get_email_status(
    email_service: EmailService = Depends(get_email_service)
):
    """
    Get email service status and configuration
    
    Returns:
        Service configuration details
    """
    try:
        status = email_service.get_service_status()
        return JSONResponse(
            status_code=200,
            content=status
        )
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get service status: {str(e)}"
        )

