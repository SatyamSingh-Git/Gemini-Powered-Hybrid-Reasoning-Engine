from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

# ==============================================================================
# 1. API Request/Response Models (External Contract)
# ==============================================================================

class APIRequest(BaseModel):
    """
    Defines the structure of the incoming request to the /hackrx/run endpoint.
    """
    documents: List[HttpUrl] = Field(
        ...,
        description="A list of URLs pointing to the PDF documents to be processed.",
        examples=[
            "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=..."
        ],
    )
    questions: List[str] = Field(
        ...,
        description="A list of natural language questions to be answered based on the documents.",
        examples=[
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
        ],
    )


class Evidence(BaseModel):
    """
    Represents a piece of evidence extracted from the document,
    providing full traceability for a decision. This is vital for Explainability.
    """
    source_document: str = Field(
        ..., description="The URL of the source document for the evidence."
    )
    page_label: str = Field(
        ..., description="The page number or label where the evidence was found."
    )
    text: str = Field(
        ..., description="The exact text snippet from the document used as evidence."
    )


class Justification(BaseModel):
    """
    Provides a detailed, step-by-step justification for the final answer,
    making the agent's reasoning process transparent.
    """
    step: str = Field(
        ..., description="The reasoning step performed (e.g., 'Waiting Period Check')."
    )
    reasoning: str = Field(
        ..., description="A natural language explanation of how the conclusion was reached for this step."
    )
    evidence: Evidence


class Answer(BaseModel):
    """
    A structured answer for a single question, designed for clarity and auditability.
    """
    query: str = Field(
        ..., description="The original question that was asked."
    )
    decision: str = Field(
        ..., description="The final decision or answer (e.g., 'Approved', 'Rejected', 'Information Found')."
    )
    payout: Optional[str] = Field(
        None, description="The monetary amount, if applicable (e.g., '₹50,000')."
    )
    justification: List[Justification] = Field(
        ..., description="A list of reasoning steps and evidence that support the decision."
    )
    # A simplified text-only answer to match the sample response format if needed
    simple_answer: str = Field(
        ..., description="A concise, text-only summary of the decision."
    )


class APIResponse(BaseModel):
    """
    The final, structured response returned by the API. The top-level key 'answers'
    matches the sample, but the content is structured for explainability.
    """
    # For the final submission, we can choose to return List[Answer] or List[str]
    # by selecting the appropriate field in the response model.
    answers: List[str]


# ==============================================================================
# 2. Internal Data Models (Used within the engine)
# ==============================================================================

class DocumentChunk(BaseModel):
    """
    Represents a single chunk of text extracted from a document,
    enriched with metadata for retrieval and evidence gathering.
    """
    chunk_id: str
    text: str
    source_document: str
    page_label: str


class DeconstructedQuery(BaseModel):
    """
    Represents the output of the initial LLM call that parses
    the user's natural language query into a structured format.
    """
    intent: str = Field(
        ..., description="The primary goal of the user's query (e.g., 'find_monetary_limit')."
    )
    procedure: Optional[str] = Field(
        None, description="The main subject or procedure (e.g., 'knee surgery', 'maternity')."
    )
    entities: List[str] = Field(
        default_factory=list, description="Other key entities mentioned (e.g., 'waiting period', 'sub-limit')."
    )