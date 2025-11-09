# Email API - Complete Guide

## 📧 Overview

Complete email API with file attachment support using SendGrid integration for the Studio Revenue Simulator API.

**Key Features:**
- ✅ Send emails to multiple recipients (to, cc, bcc)
- ✅ File attachments up to 10MB
- ✅ Plain text and HTML email support
- ✅ SendGrid integration with professional delivery
- ✅ No authentication required (configurable)
- ✅ Comprehensive error handling and logging
- ✅ Interactive API documentation

---

## 🚀 Quick Start

### 1. Prerequisites

- **SendGrid Account**: [Sign up for free](https://sendgrid.com/)
- **SendGrid API Key**: [Generate from dashboard](https://app.sendgrid.com/settings/api_keys)
- **Verified Sender Email**: [Verify your email](https://app.sendgrid.com/settings/sender_auth/senders)

### 2. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The sendgrid package is already included in requirements.txt
```

### 3. Configuration

Create a `.env` file in the project root:

```bash
# Required: SendGrid API Key
SENDGRID_API_KEY=SG.your_sendgrid_api_key_here

# Optional: Default sender email (must be verified in SendGrid)
SENDGRID_FROM_EMAIL=your_verified_email@domain.com
```

### 4. Start the API

```bash
python run_api.py
```

The API will be available at: `http://localhost:8000`

### 5. Test the Setup

```bash
# Run the test script
python test_sendgrid_config.py
```

---

## 📋 API Endpoints

### Send Email with Attachment

**Endpoint**: `POST /api/v1/email/send`

**Content-Type**: `multipart/form-data`

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | Yes | Comma-separated recipient email addresses |
| `subject` | string | Yes | Email subject line |
| `body` | string | Yes | Email body content (plain text or HTML) |
| `from_email` | string | No | Sender email (must be verified in SendGrid) |
| `from_name` | string | No | Sender name |
| `cc` | string | No | Comma-separated CC email addresses |
| `bcc` | string | No | Comma-separated BCC email addresses |
| `is_html` | boolean | No | Whether body is HTML (default: false) |
| `attachment` | file | No | File attachment (max 10MB) |

**Example - cURL**:

```bash
curl -X POST "http://localhost:8000/api/v1/email/send" \
  -F "to=recipient@example.com" \
  -F "subject=Test Email" \
  -F "body=This is a test email with an attachment." \
  -F "from_email=your_verified_email@domain.com" \
  -F "from_name=John Doe" \
  -F "is_html=false" \
  -F "attachment=@/path/to/file.pdf"
```

**Example - Python**:

```python
import requests

url = "http://localhost:8000/api/v1/email/send"

data = {
    "to": "recipient@example.com",
    "subject": "Revenue Report",
    "body": "Please find attached the monthly report.",
    "from_email": "your_verified_email@domain.com",
    "from_name": "Analytics Team",
    "is_html": False
}

files = {
    "attachment": open("report.pdf", "rb")
}

response = requests.post(url, data=data, files=files)
print(response.json())
```

**Success Response**:

```json
{
  "success": true,
  "message": "Email sent successfully",
  "message_id": "abc123xyz",
  "recipients": ["recipient@example.com"]
}
```

**Error Response**:

```json
{
  "detail": "File size exceeds 10MB limit"
}
```

### Check Email Service Status

**Endpoint**: `GET /api/v1/email/status`

**Example Request**:

```bash
curl -X GET "http://localhost:8000/api/v1/email/status"
```

**Response**:

```json
{
  "service": "SendGrid",
  "configured": true,
  "default_from_email": "your_verified_email@domain.com",
  "max_file_size_mb": 10.0
}
```

---

## 🌐 Integration with Next.js (TypeScript)

### TypeScript Types

Create `types/email.ts`:

```typescript
export interface EmailRequest {
  to: string;
  subject: string;
  body: string;
  from_email?: string;
  from_name?: string;
  cc?: string;
  bcc?: string;
  is_html?: boolean;
  attachment?: File;
}

export interface EmailResponse {
  success: boolean;
  message: string;
  message_id?: string;
  recipients: string[];
}
```

### Email Service

Create `services/emailService.ts`:

```typescript
import { EmailRequest, EmailResponse } from '@/types/email';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class EmailService {
  static async sendEmail(data: EmailRequest): Promise<EmailResponse> {
    const formData = new FormData();
    
    formData.append('to', data.to);
    formData.append('subject', data.subject);
    formData.append('body', data.body);
    
    if (data.from_email) formData.append('from_email', data.from_email);
    if (data.from_name) formData.append('from_name', data.from_name);
    if (data.cc) formData.append('cc', data.cc);
    if (data.bcc) formData.append('bcc', data.bcc);
    if (data.is_html !== undefined) formData.append('is_html', String(data.is_html));
    if (data.attachment) formData.append('attachment', data.attachment);

    const response = await fetch(`${API_BASE_URL}/api/v1/email/send`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to send email');
    }

    return response.json();
  }
}
```

### React Component Example

Create `components/EmailForm.tsx`:

```typescript
'use client';

import React, { useState } from 'react';
import { EmailService } from '@/services/emailService';

export default function EmailForm() {
  const [formData, setFormData] = useState({
    to: '',
    subject: '',
    body: '',
    from_email: 'your_verified_email@domain.com',
    from_name: 'Your Name',
  });
  const [attachment, setAttachment] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await EmailService.sendEmail({
        ...formData,
        attachment: attachment || undefined,
      });

      setResult({
        type: 'success',
        message: `Email sent successfully! Message ID: ${response.message_id}`,
      });

      // Reset form
      setFormData({
        to: '',
        subject: '',
        body: '',
        from_email: 'your_verified_email@domain.com',
        from_name: 'Your Name',
      });
      setAttachment(null);
    } catch (error) {
      setResult({
        type: 'error',
        message: error instanceof Error ? error.message : 'Failed to send email',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        setResult({ type: 'error', message: 'File size exceeds 10MB limit' });
        return;
      }
      setAttachment(file);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Send Email</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">To</label>
          <input
            type="text"
            value={formData.to}
            onChange={(e) => setFormData({ ...formData, to: e.target.value })}
            className="w-full p-2 border rounded"
            placeholder="recipient@example.com"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Subject</label>
          <input
            type="text"
            value={formData.subject}
            onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
            className="w-full p-2 border rounded"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Body</label>
          <textarea
            value={formData.body}
            onChange={(e) => setFormData({ ...formData, body: e.target.value })}
            className="w-full p-2 border rounded h-32"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Attachment (max 10MB)</label>
          <input
            type="file"
            onChange={handleFileChange}
            className="w-full p-2 border rounded"
          />
          {attachment && (
            <p className="text-sm text-gray-600 mt-1">
              Selected: {attachment.name} ({(attachment.size / 1024).toFixed(2)} KB)
            </p>
          )}
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-gray-400"
        >
          {loading ? 'Sending...' : 'Send Email'}
        </button>
      </form>

      {result && (
        <div className={`mt-4 p-4 rounded ${
          result.type === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
        }`}>
          {result.message}
        </div>
      )}
    </div>
  );
}
```

### Environment Variables

Create `.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🔧 Implementation Details

### Architecture

```
src/
├── api/
│   ├── main.py                    # FastAPI app with email router
│   ├── routes/
│   │   └── emails.py              # Email endpoints
│   ├── services/
│   │   └── email_service.py       # SendGrid integration
│   └── schemas/
│       └── email_schemas.py       # Pydantic models
```

### Components Created

**1. Email Schemas** (`src/api/schemas/email_schemas.py`)
- `EmailRequest`: Request validation model
- `EmailResponse`: Success response model
- `EmailError`: Error response model

**2. Email Service** (`src/api/services/email_service.py`)
- SendGrid client initialization
- File size validation (10MB limit)
- Base64 encoding for attachments
- Support for HTML/plain text
- Comprehensive error handling

**3. Email Routes** (`src/api/routes/emails.py`)
- `POST /api/v1/email/send` - Send email endpoint
- `GET /api/v1/email/status` - Service status endpoint
- Multipart form data handling
- File upload validation

**4. Main App Integration** (`src/api/main.py`)
- Email service initialization on startup
- Graceful degradation if SendGrid not configured
- Email router included at `/api/v1/email`

---

## 🐛 Troubleshooting

### Email Not Received

**Check SendGrid Activity Dashboard** (Most Important):
1. Go to: https://app.sendgrid.com/email_activity
2. Search for recipient email
3. Check status: Delivered, Processed, Dropped, Bounced, or Blocked

**Common Causes:**
- ✅ Check **Spam/Junk folder**
- ✅ Check **Promotions** or **Updates** tab in Gmail
- ⚠️ **Sandbox Mode** - New accounts can only send to verified emails
- ⚠️ **Unverified sender** - Must verify sender email in SendGrid

### HTTP Error 403: Forbidden

**Causes:**
1. **Unverified sender email** (most common)
2. API key missing "Mail Send" permission
3. Invalid or expired API key
4. SendGrid account suspended

**Solutions:**
1. Verify sender email: https://app.sendgrid.com/settings/sender_auth/senders
2. Check API key permissions: https://app.sendgrid.com/settings/api_keys
3. Use only verified emails in `from_email` field

### SendGrid Sandbox Mode

New SendGrid accounts are in **Sandbox Mode** and can ONLY send to verified email addresses.

**Solution:**
- Verify recipient emails in SendGrid, OR
- Request full access (may require business verification)
- Go to: https://app.sendgrid.com/settings/sender_auth

### File Size Errors

Error: "File size exceeds maximum allowed size"

**Solution:**
- Ensure files are under 10MB
- Compress large files before sending
- Use cloud storage links for larger files

### API Returns 503 Service Unavailable

Error: "Email service not initialized"

**Causes:**
1. `SENDGRID_API_KEY` not set in environment
2. Invalid API key
3. SendGrid service initialization failed

**Solution:**
1. Check `.env` file has correct `SENDGRID_API_KEY`
2. Verify API key starts with `SG.`
3. Check API server startup logs for errors

---

## 🧪 Testing

### Interactive API Documentation

1. Start API: `python run_api.py`
2. Open: http://localhost:8000/docs
3. Navigate to "Email" section
4. Test endpoints interactively

### Test Script

Run the included test script:

```bash
python test_sendgrid_config.py
```

This will:
- Verify API key configuration
- Check SendGrid connectivity
- Send a test email
- Report any issues

### Manual Testing Checklist

- [ ] Send email without attachment
- [ ] Send email with PDF attachment
- [ ] Send to multiple recipients (comma-separated)
- [ ] Test CC and BCC
- [ ] Send HTML email
- [ ] Test file size limit (try >10MB)
- [ ] Check error handling (invalid email)
- [ ] Verify email delivery in inbox
- [ ] Check SendGrid activity logs

---

## 🔒 Security & Best Practices

### Security

- ✅ Never commit `.env` file to version control
- ✅ Use environment-specific API keys
- ✅ Rotate API keys periodically
- ✅ Use restricted API keys with only "Mail Send" permission
- ⚠️ Add authentication for production use
- ⚠️ Implement rate limiting to prevent abuse
- ⚠️ Validate and sanitize all user inputs

### Best Practices

1. **Always use verified sender emails**
2. **Set appropriate from_email** for each use case
3. **Log email sends** for audit trail
4. **Monitor SendGrid activity** regularly
5. **Handle errors gracefully** with user-friendly messages
6. **Test in staging** before production
7. **Set up SPF/DKIM** records for better deliverability
8. **Monitor bounce rates** and unsubscribes

---

## 📊 Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 200 | Success | Email sent successfully |
| 400 | Bad Request | Check request parameters and validation |
| 413 | Payload Too Large | File exceeds 10MB limit |
| 500 | Internal Server Error | Check SendGrid API key and logs |
| 503 | Service Unavailable | SendGrid not configured properly |

---

## 🔗 Useful Links

- **SendGrid Dashboard**: https://app.sendgrid.com/
- **Email Activity**: https://app.sendgrid.com/email_activity
- **API Keys**: https://app.sendgrid.com/settings/api_keys
- **Sender Authentication**: https://app.sendgrid.com/settings/sender_auth
- **SendGrid Documentation**: https://docs.sendgrid.com/
- **SendGrid API Reference**: https://docs.sendgrid.com/api-reference

---

## 📦 Files Reference

**Created Files:**
- `src/api/schemas/email_schemas.py` - Pydantic models
- `src/api/services/email_service.py` - SendGrid integration
- `src/api/routes/emails.py` - API endpoints
- `test_sendgrid_config.py` - Configuration test script
- `EMAIL_README.md` - This comprehensive guide

**Modified Files:**
- `requirements.txt` - Added sendgrid package
- `src/api/main.py` - Integrated email service
- `docs/API_DOCUMENTATION.md` - Added email API section

---

## 🎯 Next Steps

### Optional Enhancements

- [ ] Add authentication/authorization
- [ ] Implement rate limiting (e.g., 10 emails/minute)
- [ ] Add email templates support
- [ ] Queue emails for background processing (Celery/Redis)
- [ ] Add retry logic for failed sends
- [ ] Implement email scheduling
- [ ] Add email tracking/analytics
- [ ] Support multiple attachments
- [ ] Add email preview before sending
- [ ] Implement email bounce handling

---

## 💡 Support

For issues or questions:

1. **Check SendGrid Activity Dashboard** first
2. **Review this guide** for common solutions
3. **Check API logs** for detailed error messages
4. **Verify SendGrid configuration** (API key, verified emails)
5. **Contact SendGrid Support** for service-specific issues

---

## 📝 License

This email API integration is part of the Studio Revenue Simulator project.

---

**Last Updated**: November 2025
**API Version**: 1.0.0
**SendGrid SDK Version**: 6.11.0

