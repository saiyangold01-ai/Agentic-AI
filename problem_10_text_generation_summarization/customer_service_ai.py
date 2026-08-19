"""Privacy-safe customer enquiry response and summarization application."""

# json reads the service-guidance file.
import json
# os reads the secret API key from the environment.
import os
# sys is used to print errors clearly.
import sys
from pathlib import Path
from typing import Any

# requests sends the prompt to the OpenRouter web service.
import requests


# Find the guidance file in the same folder as this Python file.
DATASET_PATH = Path(__file__).with_name("customer_service_dataset.json")
# This is the web address used for OpenRouter chat requests.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Keep the model selection in code instead of reading OPENROUTER_MODEL.
# Confirm this exact model slug is available in the OpenRouter account.
# This name tells OpenRouter which language model should process the request.
MODEL = "openai/gpt-5.6-luna"
# Stop waiting if the internet request takes too long.
REQUEST_TIMEOUT_SECONDS = 60
# These are the service areas shown to the user.
SERVICE_AREAS = (
    "Branch Agent",
    "Contact Center Agent",
    "Service Request Fulfillment Agent",
)


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, str]]:
    """Load general guidance; customer information is never read from it."""
    # Check that the permanent guidance file exists.
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Read the JSON file into memory. It contains only general instructions.
    with path.open("r", encoding="utf-8") as dataset_file:
        data = json.load(dataset_file)

    # Make sure the file contains a non-empty list of records.
    if not isinstance(data, list) or not data:
        raise ValueError("The service guidance dataset must be a non-empty list.")

    # Every guidance record must contain these fields.
    required_fields = {"id", "service_area", "scenario", "guidance"}
    for index, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {index} must be an object.")
        missing = required_fields - record.keys()
        if missing:
            raise ValueError(
                f"Dataset record {index} is missing: {', '.join(sorted(missing))}"
            )
        if any(not isinstance(record[field], str) for field in required_fields):
            raise ValueError(f"Dataset record {index} contains a non-text field.")

    return data


def get_service_guidance(
    records: list[dict[str, str]],
    service_area: str,
    customer_enquiry: str,
    customer_email_body: str,
) -> str:
    """Keep the selected area and rank its guidance by matching words."""
    # Do not send guidance from unrelated service areas to the AI model.
    area_records = [
        record for record in records if record["service_area"] == service_area
    ]
    if not area_records:
        raise ValueError(f"No guidance exists for {service_area}.")

    # Use the enquiry and email body temporarily to find relevant guidance.
    query_words = set((customer_enquiry + " " + customer_email_body).lower().split())

    def relevance(record: dict[str, str]) -> tuple[int, str]:
        guidance_words = set(
            (record["scenario"] + " " + record["guidance"]).lower().split()
        )
        return len(query_words & guidance_words), record["id"]

    # Put the guidance with the most matching words first.
    selected = sorted(area_records, key=relevance, reverse=True)
    return "\n\n".join(
        f"Scenario: {record['scenario']}\nGuidance: {record['guidance']}"
        for record in selected
    )


def build_prompt(
    service_area: str,
    customer_enquiry: str,
    customer_email_body: str,
    service_guidance: str,
) -> str:
    """Build the instructions that will be sent to the AI model."""
    # The customer's enquiry and email body exist only during this request.
    return f"""You are a professional financial-services customer-support assistant.

Service Area:
{service_area}

Customer Enquiry:
{customer_enquiry}

Customer Email Body:
{customer_email_body}

Relevant Service Guidance:
{service_guidance}

Perform exactly two tasks.

Task A - CUSTOMER_REPLY:
Write a concise, polite, empathetic customer-facing reply. Address the enquiry
and email body, explain the appropriate next step, and ask for missing information
when necessary. Never invent customer information, transaction or reference
numbers, dates, timelines, resolutions, or unsupported promises. Do not mention
AI, LLM, OpenRouter, Python, prompts, or internal instructions.

Task B - EMAIL_SUMMARY:
Write a concise factual internal summary containing the main enquiry, main issue,
important details, requested action, explicitly stated urgency, and relevant
missing information. Use only the supplied enquiry and email body.

Return exactly this format and nothing else:
CUSTOMER_REPLY:
<professional customer-facing response>

EMAIL_SUMMARY:
<concise internal summary>"""


def call_openrouter(prompt: str, api_key: str) -> str:
    """Send the prompt to OpenRouter and return the generated answer."""
    # The API key is used for authentication but is never printed or saved.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost/customer-service-ai",
        "X-Title": "Customer Service AI Assistant",
    }
    # This dictionary is the message sent to the chat-completions API.
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Follow the required output format exactly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        # Send the request and wait only for the configured amount of time.
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
    """Split the AI answer into the customer reply and internal summary."""
    # The prompt asks the model to use these two labels.
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


def get_console_input() -> tuple[str, str, str]:
    """Collect the three inputs and keep customer text in memory only."""
    print("=" * 40)
    print("Customer Service AI Assistant")
    print("=" * 40)
    print("\nSelect Service Area:\n")
    for number, service_area in enumerate(SERVICE_AREAS, start=1):
        print(f"{number}. {service_area}")

    # Ask the user to choose one of the three service areas.
    choice = input("\nEnter choice: ").strip()
    if choice not in {"1", "2", "3"}:
        raise ValueError("Service area selection is invalid.")
    service_area = SERVICE_AREAS[int(choice) - 1]

    # Ask for a short description of what the customer needs.
    customer_enquiry = input("\nEnter Customer Enquiry:\n\n").strip()
    if not customer_enquiry:
        raise ValueError("ERROR: Customer enquiry cannot be empty.")

    # Read the email body one line at a time until the user types END.
    print("\nEnter Customer Email Body.")
    print("Type END on a new line when finished:\n")
    body_lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        body_lines.append(line)

    # Join the lines back together into one temporary piece of text.
    customer_email_body = "\n".join(body_lines).strip()
    if not customer_email_body:
        raise ValueError("ERROR: Customer email body cannot be empty.")

    return service_area, customer_enquiry, customer_email_body


def display_result(reply: str, summary: str) -> None:
    """Display the generated reply and summary without saving them."""
    print("\n" + "=" * 40)
    print("CUSTOMER REPLY")
    print("=" * 40 + "\n")
    print(reply)
    print("\n" + "=" * 40)
    print("EMAIL SUMMARY")
    print("=" * 40 + "\n")
    print(summary)


def main() -> None:
    """Run one request and discard customer inputs when the program ends."""
    try:
        # Get the secret API key without placing it in the source code.
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("ERROR: OPENROUTER_API_KEY is not configured.")

        # Collect the user's service area, enquiry, and email body.
        service_area, customer_enquiry, customer_email_body = get_console_input()
        # Load and select only the relevant general service guidance.
        records = load_dataset()
        guidance = get_service_guidance(
            records,
            service_area,
            customer_enquiry,
            customer_email_body,
        )
        prompt = build_prompt(
            service_area,
            customer_enquiry,
            customer_email_body,
            guidance,
        )
        # Send the temporary request data to the AI model.
        print("\nProcessing request...")
        reply, summary = parse_response(call_openrouter(prompt, api_key))
        display_result(reply, summary)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n{error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    # Start the application only when this file is run directly.
    main()
