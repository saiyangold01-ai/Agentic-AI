"""Generate a customer reply and summary from a document using OpenRouter."""

# json reads the generic service-guidance dataset.
import json
# os reads the API key from the computer's environment.
import os
# sys lets us print errors clearly before stopping the program.
import sys
# Path makes file and folder paths easier to work with.
from pathlib import Path

# requests sends the prompt to the OpenRouter web API.
import requests
# Document reads text from Microsoft Word files.
from docx import Document
# PdfReader reads text from PDF files.
from pypdf import PdfReader


DATASET_PATH = Path(__file__).with_name("customer_document_dataset.json")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# The model is fixed in the code as requested. The API key remains external.
# This must be a valid model identifier available through the OpenRouter account.
MODEL = "openai/gpt-5.6-luna"
# Avoid waiting forever if the API or network is unavailable.
REQUEST_TIMEOUT_SECONDS = 60
# These are the three service areas the user can choose from.
SERVICE_AREAS = (
    "Branch Agent",
    "Contact Center Agent",
    "Service Request Fulfillment Agent",
)
SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx"}


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, str]]:
    """Load generic guidance; this file contains no customer information."""
    # Stop early if the guidance file is missing.
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Read the permanent knowledge file. It contains only general instructions.
    with path.open("r", encoding="utf-8") as dataset_file:
        records = json.load(dataset_file)

    # Make sure the file has the expected list format and is not empty.
    if not isinstance(records, list) or not records:
        raise ValueError("The dataset must be a non-empty JSON list.")

    # Every guidance record must have these five fields.
    required_fields = {
        "id",
        "service_area",
        "scenario",
        "customer_enquiry_type",
        "guidance",
    }
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {index} must be an object.")
        missing = required_fields - record.keys()
        if missing:
            raise ValueError(
                f"Dataset record {index} is missing: {', '.join(sorted(missing))}"
            )
        if any(not isinstance(record[field], str) for field in required_fields):
            raise ValueError(f"Dataset record {index} contains an invalid field.")

    return records


def extract_document_text(file_path: Path) -> str:
    """Read text from TXT, PDF, or DOCX without creating a copy."""
    # The extension tells us which reader should be used.
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            "ERROR: Unsupported document format.\n"
            "Supported formats: .txt, .pdf, .docx"
        )
    # Confirm that the path points to an existing file.
    if not file_path.is_file():
        raise FileNotFoundError(f"Customer document not found: {file_path}")

    # Plain text files can be read directly with Python.
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")

    # PDF files may contain several pages, so combine their extracted text.
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    # For DOCX files, combine the text from all paragraphs.
    document = Document(str(file_path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def get_service_guidance(
    records: list[dict[str, str]],
    service_area: str,
    document_text: str,
) -> str:
    """Keep the chosen area and rank its guidance using temporary word matches."""
    # Never send guidance from unrelated service areas to the model.
    area_records = [
        record for record in records if record["service_area"] == service_area
    ]
    if not area_records:
        raise ValueError(f"No service guidance found for {service_area}.")

    # Use document words only temporarily to find the most relevant guidance.
    document_words = set(document_text.lower().split())

    def relevance(record: dict[str, str]) -> tuple[int, str]:
        guidance_words = set(
            (
                record["scenario"]
                + " "
                + record["customer_enquiry_type"]
                + " "
                + record["guidance"]
            ).lower().split()
        )
        return len(document_words & guidance_words), record["id"]

    # Put the most relevant scenarios first. Nothing is saved to disk.
    selected = sorted(area_records, key=relevance, reverse=True)
    return "\n\n".join(
        f"Scenario: {record['scenario']}\n"
        f"Enquiry type: {record['customer_enquiry_type']}\n"
        f"Guidance: {record['guidance']}"
        for record in selected
    )


def build_prompt(
    service_area: str,
    document_text: str,
    service_guidance: str,
) -> str:
    """Build the instructions sent to the LLM."""
    # The model receives the selected area, document text, and generic guidance.
    # It must identify the enquiry directly from the uploaded document.
    return f"""You are a professional customer-service communication agent.

Service Area:
{service_area}

Customer Document:
{document_text}

Relevant Service Guidance:
{service_guidance}

Identify the customer's enquiry, issue, requested action, supporting details,
urgency if explicitly stated, and missing information from the document.

Then perform exactly two tasks:

Task A - CUSTOMER_REPLY:
Write a concise, professional, polite, and empathetic customer-facing response.
Address the actual issue, explain the appropriate next step, and ask for missing
information when necessary. Do not invent customer information, transaction or
reference numbers, dates, timelines, resolutions, or promises. Do not mention AI,
LLM, OpenRouter, Python, prompts, or internal instructions.

Task B - EMAIL_SUMMARY:
Write a concise factual internal summary based only on the customer document.
Include the main enquiry, issue, relevant details, requested action, explicit
urgency, and missing information when relevant. Do not invent information.

Return exactly this format and nothing else:
CUSTOMER_REPLY:
<professional customer-facing response>

EMAIL_SUMMARY:
<concise internal summary>"""


def call_openrouter(prompt: str, api_key: str) -> str:
    """Send the prompt to OpenRouter and return the model's text response."""
    # The API key is used only for this request and is never printed or saved.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost/customer-document-ai",
        "X-Title": "Customer Document Communication Assistant",
    }
    # This is the request body expected by the OpenRouter chat API.
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Follow the required output format exactly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        # Send the request and wait only up to the configured timeout.
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.Timeout as error:
        raise RuntimeError("The OpenRouter request timed out.") from error
    except requests.RequestException as error:
        detail = error.response.text[:300] if error.response is not None else str(error)
        raise RuntimeError(f"OpenRouter request failed: {detail}") from error
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError("OpenRouter returned an invalid response.") from error


def parse_response(response: str) -> tuple[str, str]:
    """Separate the model response into a reply and an internal summary."""
    # The prompt requires these two labels so the output can be split reliably.
    reply_marker = "CUSTOMER_REPLY:"
    summary_marker = "EMAIL_SUMMARY:"
    if reply_marker not in response or summary_marker not in response:
        raise ValueError("The LLM response is missing a required section.")

    reply_start = response.index(reply_marker) + len(reply_marker)
    summary_start = response.index(summary_marker)
    reply = response[reply_start:summary_start].strip()
    summary = response[summary_start + len(summary_marker):].strip()
    if not reply or not summary:
        raise ValueError("The LLM returned an empty reply or summary.")
    return reply, summary


def choose_service_area() -> str:
    """Show the menu and return the selected service area."""
    print("=" * 40)
    print("Customer Service AI Assistant")
    print("=" * 40)
    print("\nSelect Service Area:\n")
    for number, service_area in enumerate(SERVICE_AREAS, start=1):
        print(f"{number}. {service_area}")

    # Validate the user's menu choice before continuing.
    choice = input("\nEnter choice: ").strip()
    if choice not in {"1", "2", "3"}:
        raise ValueError("Service area selection is invalid.")
    return SERVICE_AREAS[int(choice) - 1]


def display_result(reply: str, summary: str) -> None:
    """Display the reply and summary; neither is written to a file."""
    print("\n" + "=" * 40)
    print("CUSTOMER REPLY")
    print("=" * 40 + "\n")
    print(reply)
    print("\n" + "=" * 40)
    print("EMAIL SUMMARY")
    print("=" * 40 + "\n")
    print(summary)


def main() -> None:
    """Process one document in memory and then finish without storing it."""
    try:
        # Require the secret API key before reading or sending customer data.
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("ERROR: OPENROUTER_API_KEY is not configured.")

        service_area = choose_service_area()
        # The user supplies a path; the program does not copy or modify the file.
        document_path = Path(input("\nEnter path to customer document: ").strip())

        print("\nReading document...")
        # Extracted text exists only in memory during this one request.
        document_text = extract_document_text(document_path)
        if not document_text.strip():
            raise ValueError("ERROR: The document contains no readable text.")

        print("Extracting customer enquiry and email content...")
        # Load generic guidance and select records for the chosen service area.
        records = load_dataset()
        guidance = get_service_guidance(records, service_area, document_text)
        prompt = build_prompt(service_area, document_text, guidance)

        print("Loading service guidance...")
        print("Calling OpenRouter...")
        # Ask the model for both outputs, then display them separately.
        reply, summary = parse_response(call_openrouter(prompt, api_key))
        display_result(reply, summary)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n{error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    # Run the application only when this file is executed directly.
    main()
