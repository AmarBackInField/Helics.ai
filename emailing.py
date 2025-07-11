import os
import pandas as pd
import smtplib
from typing import Dict, Any, Optional
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from dotenv import load_dotenv
load_dotenv()

# ========== Configuration ==========
SMTP_SERVER = "smtp.gmail.com"  # change as needed
SMTP_PORT = 465
WORKERS = 5

# ========== Logger ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== EmailService Class ==========
class EmailService:
    def __init__(self, user_info: Dict[str, Any], smtp_server: str = SMTP_SERVER, smtp_port: int = SMTP_PORT, max_workers: int = WORKERS):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.user_info = user_info
        self.max_workers = max_workers

        self.email = os.environ.get("EMAIL_USERNAME")
        if not self.email:
            raise ValueError("EMAIL_USERNAME environment variable must be set")

        self.password = os.environ.get("EMAIL_PASSWORD")
        if not self.password:
            raise ValueError("EMAIL_PASSWORD environment variable must be set")

    def generate_email_content(self) -> Dict[str, str]:
        subject = "What If Your Brand Could Think? (Founder Chat Invite)"
        body = """
            <p><strong>Hi [Name],</strong></p>

            <p>I'm building Helics.ai — an AI-powered creative workspace that helps teams generate brand-consistent content effortlessly.</p>

            <p>I’d love to get your input as the founder of <strong>[Company]</strong>. Your feedback will help us shape the product to better serve startups like yours.</p>

            <p>As a thank you, <strong>[Company]</strong> will receive <em>exclusive early access and special launch discounts</em> when we go live.</p>

            <p>Would you be open to a quick 15–20 min chat this week?</p>

            <p>Best,<br>[Your Name]</p>
        """
        return {"subject": subject, "body": body}

    def send_email(self, recipient_email: str, subject: str, body: str, sender_name: Optional[str] = None) -> bool:
        try:
            msg = MIMEMultipart()
            from_address = f"{sender_name} <{self.email}>" if sender_name else self.email
            msg["From"] = from_address
            msg["To"] = recipient_email
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "html"))

            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.email, self.password)
                server.send_message(msg)

            return True
        except Exception as e:
            logger.error(f"Error sending email to {recipient_email}: {str(e)}")
            return False

    def personalize_and_send_bulk_emails(self, csv_file_path: str, sender_name: str):
        try:
            df = pd.read_excel(csv_file_path)
            df=df[124:]
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Sending Emails"):
                email = row.get("Email", "").strip()
                firstname = row.get("Firstname", "").strip()
                lastname = row.get("Lastname", "").strip()
                company = row.get("Company", "").strip()

                if not email or not firstname or not company:
                    logger.warning(f"Skipping row due to missing data: {row}")
                    continue

                content = self.generate_email_content()
                body = (
                    content["body"]
                    .replace("[Name]", f"{firstname} {lastname}".strip())
                    .replace("[Company]", company)
                    .replace("[Your Name]", sender_name)
                )
                subject = content["subject"].replace("[Name]", f"{firstname} {lastname}".strip())

                futures.append(executor.submit(self.send_email, email, subject, body, sender_name))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Threaded email send error: {e}")
                    results.append(False)

        logger.info(f"Emails sent successfully: {sum(results)} / {len(results)}")

# ========== Usage Example ==========
if __name__ == "__main__":
    user_info = {
        "role": "founder",
        "company": "Helics.ai"
    }

    email_service = EmailService(user_info=user_info)
    email_service.personalize_and_send_bulk_emails("data.xlsx", sender_name="Helics")
