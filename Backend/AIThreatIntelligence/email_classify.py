import imaplib
import email
from email.header import decode_header
import re
from .classifier import classifier  # Your ML classifier function

def email_extract():
    # --- Email server configuration ---
    IMAP_SERVER = "imap.gmail.com"
    EMAIL_ACCOUNT = "shaikhusama541@gmail.com"
    EMAIL_PASSWORD = "pofp rdpe dekk evzl"  # Gmail App Password

    # --- Connect to mailbox ---
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
    mail.select("inbox")

    # --- Search all unseen emails ---
    status, messages = mail.search(None, "UNSEEN")
    email_ids = messages[0].split()

    results = []

    for eid in email_ids:
        status, msg_data = mail.fetch(eid, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])

                # --- Decode subject ---
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")

                # --- Extract sender (name + email) ---
                sender = msg.get("From")
                match = re.match(r'\"?([^\"<]*)\"?\s*<(.+)>', sender)
                sender_name = match.group(1).strip() if match else sender
                sender_email = match.group(2).strip() if match else ""

                # --- Extract body ---
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                            body = part.get_payload(decode=True).decode(errors="ignore")
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")

                # --- Combine subject + body and classify ---
                text_to_classify = f"{subject} {body}"
                prediction = classifier(text_to_classify)  # returns "Phishing" or "Safe"

                results.append({
                    "sender_name": sender_name,
                    "sender_email": sender_email,
                    "status": prediction
                })

    return results