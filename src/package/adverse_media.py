
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from data.news import SNIPPETS

load_dotenv()

client = OpenAI()

# Schema for a single snippet classification
class AdverseMediaResult(BaseModel):
    adverse: bool
    rationale: str
    confidence: float

# System prompt
SYSTEM_PROMPT = """You are an expert compliance analyst specialising in financial crime and AML/CFT screening.
Your task is to classify short news snippets as either ADVERSE or BENIGN for the purposes of PEP and sanctions screening.

ADVERSE media includes: sanctions designations, criminal indictments or convictions, money laundering allegations,
bribery or corruption, human rights abuses, terrorist financing, fraud, asset seizures, regulatory enforcement actions,
or any content that would materially affect a compliance officer's risk assessment of a named individual or entity.

BENIGN media includes: neutral business news, sports, weather, local community events, academic publications,
or any content with no relevance to financial crime or regulatory risk.

You must respond ONLY with a valid JSON object in this exact format:
{
  "adverse": true or false,
  "confidence": 0.0 to 1.0,
  "rationale": "one sentence explaining the classification decision"
}
Do not include any text outside the JSON object."""


def classify_snippet(snippet_id:str, text:str) -> AdverseMediaResult:
    """Classify a single new snippet."""
    user_content = json.dumps({"snippet_id": snippet_id, "text": text})
    response = client.responses.parse(
        model = "gpt-4o-mini",
        input = [
            {"role": "system", "content" : SYSTEM_PROMPT},
            {"role" : "user", "content": user_content }
        ],
        text_format=AdverseMediaResult
    )
    result = response.output_parsed
    return {
        "snippet_id": snippet_id,  
        "adverse": result.adverse,
        "rationale": result.rationale,
        "confidence": result.confidence,
    }

# Batch classifierr + output assembler
def run_classifier(snippets: list[dict] = SNIPPETS) -> dict:
    """
    Runs the classifier across all snippets and returns the full
    structured output for a given query entity.
    """
    results = []
    for snippet in snippets:
        result = classify_snippet(snippet["id"], snippet["text"])
        results.append(result)
    return results

# Standalone entry point
if __name__ == "__main__":
    print(f"Classifying {len(SNIPPETS)} snippets...\n")
 
    results = run_classifier()
 
    for r in results:
        label = "ADVERSE ⚠️ " if r["adverse"] else "benign  ✓"
        print(f"  [{r['snippet_id']}] {label}  conf={r['confidence']:.2f}  — {r['rationale']}")
 
    adverse_count = sum(1 for r in results if r["adverse"])
    print(f"\nSummary: {adverse_count}/{len(results)} snippets flagged as adverse.")
    print("\nRaw output (for pipeline consumption):")
    print(json.dumps(results, indent=2, ensure_ascii=False))