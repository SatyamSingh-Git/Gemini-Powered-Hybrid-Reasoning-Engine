import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Gemini Reasoning Engine",
    page_icon="🤖",
    layout="wide",
)

# --- Header and Introduction ---
st.title("HackRx: Gemini-Powered Hybrid Reasoning Engine 🤖")
st.markdown(
    """
    This application demonstrates an advanced query-retrieval system built for the HackRx hackathon.
    Enter document URLs and your questions below, and the engine will provide structured, evidence-backed answers.
    """
)

# --- Constants ---
API_URL = "http://localhost:8000/hackrx/run"
# This is the bearer token from the problem description, hardcoded for the demo.
BEARER_TOKEN = "Bearer bc916bc507a9b3b680e613c91243b99771a30be1587ca8d9eb8cc4b9dfab5a55"

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Inputs")
    st.markdown("Enter the URLs of the policy documents and your questions.")

    # Input for Document URLs
    default_doc_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    doc_urls_input = st.text_area(
        "Document URLs (one per line)",
        value=default_doc_url,
        height=100,
    )

    # Input for Questions
    default_questions = """What is the grace period for premium payment?
What is the waiting period for pre-existing diseases (PED) to be covered?
Does this policy cover maternity expenses, and what are the conditions?
What is the waiting period for cataract surgery?
Are there any sub-limits on room rent and ICU charges for Plan A?"""
    questions_input = st.text_area(
        "Questions (one per line)",
        value=default_questions,
        height=200,
    )

    submit_button = st.button("Run Engine", type="primary", use_container_width=True)

# --- Main Content Area for Outputs ---
st.header("📝 Results")

if submit_button:
    # Validate inputs
    doc_urls = [url.strip() for url in doc_urls_input.split('\n') if url.strip()]
    questions = [q.strip() for q in questions_input.split('\n') if q.strip()]

    if not doc_urls or not questions:
        st.error("Please provide at least one document URL and one question.")
    else:
        # Prepare the request payload
        payload = {"documents": doc_urls, "questions": questions}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": BEARER_TOKEN,
        }

        # Call the API
        with st.spinner("Processing documents and reasoning on your questions... This may take a moment."):
            try:
                response = requests.post(API_URL, data=json.dumps(payload), headers=headers, timeout=300)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                # Display the results
                results = response.json()
                st.success("Successfully received a response from the engine!")

                st.subheader("Final Answers")
                # According to the spec, the final response is a simple list of strings.
                for i, answer_str in enumerate(results.get("answers", [])):
                    st.markdown(f"**Q{i+1}:** `{questions[i]}`")
                    st.info(f"**A:** {answer_str}")
                    st.divider()

                # For the demo, we can also show the raw JSON for the judges to inspect
                with st.expander(" görmek the raw JSON response"):
                    st.json(results)

            except requests.exceptions.HTTPError as http_err:
                st.error(f"An HTTP error occurred: {http_err}")
                try:
                    # Try to show the detailed error from the API response body
                    st.json(http_err.response.json())
                except json.JSONDecodeError:
                    st.text(http_err.response.text)
            except requests.exceptions.RequestException as req_err:
                st.error(f"A network error occurred: {req_err}. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

else:
    st.info("Please provide document URLs and questions in the sidebar and click 'Run Engine'.")