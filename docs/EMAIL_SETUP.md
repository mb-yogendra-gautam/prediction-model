# Email API Setup Guide

This guide explains how to set up and use the email API with SendGrid integration.

## Prerequisites

1. **SendGrid Account**: Sign up for a free account at [SendGrid](https://sendgrid.com/)
2. **SendGrid API Key**: Generate an API key from the [SendGrid Dashboard](https://app.sendgrid.com/settings/api_keys)

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required: SendGrid API Key
SENDGRID_API_KEY=your_sendgrid_api_key_here

# Optional: Default sender email address
# If not provided, defaults to noreply@example.com
SENDGRID_FROM_EMAIL=your_email@yourdomain.com
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The email functionality requires the `sendgrid` package, which is included in `requirements.txt`.

## API Endpoint

### Send Email with Attachment

**Endpoint**: `POST /api/v1/email/send`

**Content-Type**: `multipart/form-data`

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | Yes | Comma-separated list of recipient email addresses |
| `subject` | string | Yes | Email subject line |
| `body` | string | Yes | Email body content (plain text or HTML) |
| `from_email` | string | No | Sender email address (uses default if not provided) |
| `from_name` | string | No | Sender name |
| `cc` | string | No | Comma-separated list of CC email addresses |
| `bcc` | string | No | Comma-separated list of BCC email addresses |
| `is_html` | boolean | No | Whether the body content is HTML (default: false) |
| `attachment` | file | No | File attachment (max 10MB) |

**Example Request using cURL**:

```bash
curl -X POST "http://localhost:8000/api/v1/email/send" \
  -F "to=recipient@example.com" \
  -F "subject=Test Email" \
  -F "body=This is a test email with an attachment." \
  -F "from_email=sender@example.com" \
  -F "from_name=John Doe" \
  -F "is_html=false" \
  -F "attachment=@/path/to/file.pdf"
```

**Example Request using Python**:

```python
import requests

url = "http://localhost:8000/api/v1/email/send"

# Email data
data = {
    "to": "recipient@example.com",
    "subject": "Test Email",
    "body": "This is a test email with an attachment.",
    "from_email": "sender@example.com",
    "from_name": "John Doe",
    "is_html": False
}

# File attachment
files = {
    "attachment": open("document.pdf", "rb")
}

response = requests.post(url, data=data, files=files)
print(response.json())
```

**Example Response**:

```json
{
  "success": true,
  "message": "Email sent successfully",
  "message_id": "abc123xyz",
  "recipients": ["recipient@example.com"]
}
```

### Check Email Service Status

**Endpoint**: `GET /api/v1/email/status`

**Example Request**:

```bash
curl -X GET "http://localhost:8000/api/v1/email/status"
```

**Example Response**:

```json
{
  "service": "SendGrid",
  "configured": true,
  "default_from_email": "sender@example.com",
  "max_file_size_mb": 10.0
}
```

## Features

- ✅ Send emails to multiple recipients
- ✅ CC and BCC support
- ✅ File attachments up to 10MB
- ✅ Both plain text and HTML email support
- ✅ Custom sender email and name
- ✅ Automatic file size validation
- ✅ Comprehensive error handling

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Email sent successfully
- **400**: Bad request (validation error)
- **413**: File size exceeds 10MB limit
- **500**: Server error (SendGrid API failure)
- **503**: Service unavailable (SendGrid not configured)

## Common Error Messages

### "SendGrid API key not found"

**Solution**: Set the `SENDGRID_API_KEY` environment variable in your `.env` file.

### "File size exceeds maximum allowed size"

**Solution**: Ensure your attachment is smaller than 10MB.

### "Email service not initialized"

**Solution**: Check that the API server started successfully and the SendGrid API key is valid.

## Testing

You can test the email API using the interactive API documentation:

1. Start the API server: `python run_api.py`
2. Open your browser to `http://localhost:8000/docs`
3. Navigate to the "Email" section
4. Try the `/api/v1/email/send` endpoint

## Security Notes

- Never commit your `.env` file or expose your SendGrid API key
- In production, use environment-specific API keys
- Consider adding rate limiting to prevent abuse
- Implement authentication for production use

## Troubleshooting

### Email not received

1. Check SendGrid activity logs in the [SendGrid Dashboard](https://app.sendgrid.com/email_activity)
2. Verify the sender email is verified in SendGrid
3. Check recipient spam folders
4. Ensure SendGrid account is not suspended

### API returns 503 error

1. Verify `SENDGRID_API_KEY` is set correctly
2. Check API server logs for initialization errors
3. Validate the API key in SendGrid dashboard

## Support

For SendGrid-specific issues, refer to:
- [SendGrid Documentation](https://docs.sendgrid.com/)
- [SendGrid API Reference](https://docs.sendgrid.com/api-reference)

