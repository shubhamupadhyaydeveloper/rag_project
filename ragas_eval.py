import os
import re
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from answer import answer_question
from dotenv import load_dotenv


class GeminiNoFences(ChatGoogleGenerativeAI):
    """Strip markdown code fences from Gemini responses so RAGAS JSON parsers don't choke."""

    def _strip(self, text: str) -> str:
        text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
        text = re.sub(r'\n?```\s*$', '', text)
        return text.strip()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        cleaned = []
        for gen in result.generations:
            msg = gen.message
            if isinstance(msg.content, str):
                msg = AIMessage(content=self._strip(msg.content),
                                response_metadata=getattr(msg, 'response_metadata', {}))
            cleaned.append(ChatGeneration(message=msg))
        return ChatResult(generations=cleaned, llm_output=result.llm_output)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        cleaned = []
        for gen in result.generations:
            msg = gen.message
            if isinstance(msg.content, str):
                msg = AIMessage(content=self._strip(msg.content),
                                response_metadata=getattr(msg, 'response_metadata', {}))
            cleaned.append(ChatGeneration(message=msg))
        return ChatResult(generations=cleaned, llm_output=result.llm_output)


load_dotenv(override=True)

_llm = GeminiNoFences(model="gemini-2.5-flash-lite", temperature=0)
_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=768,
)
ragas_llm = LangchainLLMWrapper(_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(_embeddings)

test_cases = [
    {
        "question": "What is the LG customer support phone number for warranty service in India?",
        "ground_truth": (
            "LG customer support in India can be reached at 08069379999 or 9711709999 (WhatsApp). "
            "You can also email serviceindia@lge.com. When contacting support, provide your complaint "
            "number or registered contact number."
        ),
    },
    {
        "question": "Where and by when should I pick up my repaired MacBook Air from Apple?",
        "ground_truth": (
            "The repaired MacBook Air (M2, 2022) with Repair ID R705453717 is ready for pickup at "
            "Apple Saket. It must be picked up before Tuesday, March 24, 2026. For alternate pickup "
            "arrangements, contact 0008000404503."
        ),
    },
    {
        "question": "What promo code can I use for Cambly's spring discount offer?",
        "ground_truth": (
            "The promo code is spring26e. It gives up to 50% off on Private+, Pro, and Groups "
            "12-month plans, and 30% off on 3-month plans. The offer is valid until March 31, 2026 "
            "at 11:59 PM HST."
        ),
    },
    {
        "question": "What is the starting price of the new iPad Air with M4 chip in India?",
        "ground_truth": (
            "The new iPad Air with M4 chip starts at ₹64,900 (outright purchase) or ₹10,150 per "
            "month for 6 months on No Cost EMI. Additional instant cashback of up to ₹4,000 is "
            "available with eligible cards."
        ),
    },
    {
        "question": "What screen sizes are available for the iPad Air with M4 chip?",
        "ground_truth": (
            "The iPad Air with M4 chip is available in two sizes: 27.59 cm (11-inch) and "
            "32.78 cm (13-inch)."
        ),
    },
    {
        "question": "How can I contact Jupiter Money support if my account is frozen due to eKYC?",
        "ground_truth": (
            "To resolve a frozen Jupiter Money account due to eKYC, email support@jupiter.money "
            "using your registered email ID (not an unregistered one). You can also use the "
            "in-app chat by tapping the headphone icon inside the Jupiter app."
        ),
    },
    {
        "question": "What was the Sensex value reported in the Groww Digest on March 11, 2026?",
        "ground_truth": (
            "The Sensex was at 76,863.71 on March 11, 2026, down 1.72% for the day. The market "
            "fell throughout the day due to concerns about US-Iran war tensions."
        ),
    },
    {
        "question": "What was the 'Word of the Day' in the Groww Digest for March 11, 2026?",
        "ground_truth": (
            "The Word of the Day in the Groww Digest for March 11, 2026 was 'Rebalancing', which "
            "refers to adjusting your portfolio allocation back to your target percentages."
        ),
    },
    {
        "question": "What happened with the boAt warranty replacement for ticket #50004600?",
        "ground_truth": (
            "In ticket #50004600, the customer had submitted a Redgear keyboard for warranty "
            "replacement but received an RG MP 35 Speed Gaming Mouse Pad instead. "
            "The case was being escalated to the concern team for resolution."
        ),
    },
    {
        "question": "What is the last date to apply for VITEEE 2026?",
        "ground_truth": (
            "The last date to submit the VITEEE 2026 application form is March 31, 2026."
        ),
    },
    {
        "question": "What problem does Amazon Nova Act solve in browser automation?",
        "ground_truth": (
            "Amazon Nova Act solves the 'Selector Tax' problem — automation scripts break "
            "whenever UI changes class names. Nova Act uses visual understanding to click "
            "buttons by description regardless of CSS changes."
        ),
    },
    {
        "question": "What is the new AI feature introduced in Android Studio Panda 2?",
        "ground_truth": (
            "Android Studio Panda 2 introduced the 'Create with AI' feature, which lets "
            "developers describe their app in natural language. Gemini then generates the "
            "project foundation including Kotlin and Compose code."
        ),
    },
]


def run_ragas_eval():
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Generating answers...")
    for case in test_cases:
        question = case["question"]
        ground_truth = case["ground_truth"]

        try:
            answer, docs = answer_question(question)
            context = [doc.page_content for doc in docs]
        except Exception as e:
            print(f"❌ Failed: {question[:50]} — {e}")
            continue

        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(ground_truth)
        print(f"✅ {question[:50]}...")

    if not questions:
        print("No answers generated — check answer.py")
        return

    dataset = Dataset.from_dict({
        "user_input":          questions,
        "response":            answers,
        "retrieved_contexts":  contexts,
        "reference":           ground_truths,
    })

    print(f"\nEvaluating {len(questions)} questions...")

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ],
    )

    faithfulness      = sum(v for v in result['faithfulness']      if v is not None) / sum(1 for v in result['faithfulness']      if v is not None)
    answer_relevancy  = sum(v for v in result['answer_relevancy']  if v is not None) / sum(1 for v in result['answer_relevancy']  if v is not None)
    context_precision = sum(v for v in result['context_precision'] if v is not None) / sum(1 for v in result['context_precision'] if v is not None)
    context_recall    = sum(v for v in result['context_recall']    if v is not None) / sum(1 for v in result['context_recall']    if v is not None)

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    print(f"Faithfulness:      {faithfulness:.4f}")
    print(f"Answer Relevancy:  {answer_relevancy:.4f}")
    print(f"Context Precision: {context_precision:.4f}")
    print(f"Context Recall:    {context_recall:.4f}")
    print("=" * 50)

    avg = (faithfulness + answer_relevancy + context_precision + context_recall) / 4
    print(f"Average Score:     {avg:.4f}")
    print("=" * 50)

    return result


if __name__ == "__main__":
    run_ragas_eval()
