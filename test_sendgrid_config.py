"""
Test SendGrid Configuration
This script tests if your SendGrid API key and sender email are properly configured.
"""

import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

# Load environment variables
load_dotenv()

def test_sendgrid_config():
    """Test SendGrid configuration"""
    
    print("=" * 60)
    print("Testing SendGrid Configuration")
    print("=" * 60)
    
    # Get configuration from environment
    api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL", "noreply@example.com")
    to_email = "devshrishticomplex2@gmail.com"
    
    # Validate configuration
    print("\n1. Checking configuration...")
    if not api_key:
        print("ERROR: SENDGRID_API_KEY not found in environment variables")
        print("   Please set SENDGRID_API_KEY in your .env file")
        return False
    
    print(f"[OK] API Key found: {api_key[:10]}...{api_key[-4:]}")
    print(f"[OK] From Email: {from_email}")
    print(f"[OK] To Email: {to_email}")
    
    # Create test email
    print("\n2. Creating test email...")
    try:
        message = Mail(
            from_email=Email(from_email, "SendGrid Test"),
            to_emails=To(to_email),
            subject="SendGrid Configuration Test",
            plain_text_content="This is a test email to verify your SendGrid configuration is working correctly."
        )
        print("[OK] Email message created")
    except Exception as e:
        print(f"ERROR creating message: {e}")
        return False
    
    # Initialize SendGrid client
    print("\n3. Initializing SendGrid client...")
    try:
        sg = SendGridAPIClient(api_key)
        print("[OK] SendGrid client initialized")
    except Exception as e:
        print(f"ERROR initializing client: {e}")
        return False
    
    # Send test email
    print("\n4. Sending test email...")
    try:
        response = sg.send(message)
        
        if response.status_code in [200, 201, 202]:
            print(f"[OK] Email sent successfully!")
            print(f"  Status Code: {response.status_code}")
            print(f"  Message ID: {response.headers.get('X-Message-Id', 'N/A')}")
            print(f"\nSUCCESS! Check {to_email} inbox for the test email.")
            return True
        else:
            print(f"WARNING: Unexpected status code: {response.status_code}")
            print(f"  Response: {response.body}")
            return False
            
    except Exception as e:
        print(f"ERROR sending email: {e}")
        print("\nCommon issues:")
        print("  1. Sender email not verified in SendGrid")
        print("     → Go to: https://app.sendgrid.com/settings/sender_auth/senders")
        print("  2. API key doesn't have 'Mail Send' permission")
        print("     → Go to: https://app.sendgrid.com/settings/api_keys")
        print("  3. Invalid API key")
        print("  4. SendGrid account issue")
        return False


if __name__ == "__main__":
    print("\nSendGrid Configuration Test\n")
    
    success = test_sendgrid_config()
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: All tests passed! Your SendGrid is configured correctly.")
        print("   You can now use the email API with confidence.")
    else:
        print("FAILED: Test failed. Please fix the issues above and try again.")
    print("=" * 60)
    print()

