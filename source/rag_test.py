from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from datasets import Dataset
from rag_pipeline import query_rag

# ── Your chroma path ──────────────────────────────────────────────────────────
# After uploading your PDF through rag.html, check your chat_history folder
# Open the rag_XXXXXXXX.json file and copy the chroma_path value
CHROMA_PATH = "./chroma_db/af1b4dae"
MODEL       = "llama3"

# ── Paste the 10 questions + ground truths here ───────────────────────────────
test_cases = [
    {
        "question": "What are the three core characteristics of AI Agents?",
        "ground_truth": "Autonomy, Task-Specificity, and Reactivity with Adaptation."
    },
    {
        "question": "How does the paper define the difference between AI Agents and Agentic AI in terms of collaboration?",
        "ground_truth": "AI Agents operate independently as single-entity systems, while Agentic AI involves multiple specialized agents that coordinate and dynamically allocate sub-tasks within a broader workflow."
    },
    {
        "question": "What role does RAG play in addressing AI Agent limitations?",
        "ground_truth": "RAG mitigates hallucinations and expands static LLM knowledge by grounding outputs in real-time data through retrieval from vector databases."
    },
    {
        "question": "What are the five core principles of the Agent-to-Agent A2A protocol?",
        "ground_truth": "Embracing agentic capabilities, building on existing standards, securing interactions by default, supporting long-running tasks, and ensuring modality agnosticism."
    },
    {
        "question": "What are the four primary architectural components of traditional AI Agents?",
        "ground_truth": "Perception Module, Knowledge Representation and Reasoning Module, Action Selection and Execution Module, and Basic Learning and Adaptation."
    },
    {
        "question": "What is the AZR framework and why is it significant?",
        "ground_truth": "AZR removes dependency on external datasets by enabling agents to autonomously generate, validate, and solve their own tasks using verifiable feedback mechanisms like code execution."
    },
    {
        "question": "What are the four key challenges specific to Agentic AI systems?",
        "ground_truth": "Amplified causality challenges, communication and coordination bottlenecks, emergent behavior and unpredictability, and scalability and debugging complexity."
    },
    {
        "question": "What distinguishes Generative AI from AI Agents?",
        "ground_truth": "Generative AI is stateless and lacks memory and goal-following mechanisms, while AI Agents add memory buffers, tool-calling APIs, and planning routines enabling active task completion."
    },
    {
        "question": "What is the ReAct framework?",
        "ground_truth": "ReAct combines reasoning and action in an iterative loop where LLMs alternate between internal cognition and external tool interaction to reduce hallucination and improve decision making."
    },
    {
        "question": "What are the five future roadmap directions for AI Agents?",
        "ground_truth": "Proactive Intelligence, Tool Integration, Causal Reasoning, Continuous Learning, and Trust and Safety."
    },
]

# ── Run RAG pipeline on each question ────────────────────────────────────────
print("Running RAG pipeline on test questions...")
results = []
for tc in test_cases:
    print(f"\n  Q: {tc['question'][:60]}...")
    output = query_rag(tc["question"], CHROMA_PATH, model=MODEL)
    results.append({
        "question":     tc["question"],
        "answer":       output["answer"],
        "contexts":     output["sources"],
        "ground_truth": tc["ground_truth"],
    })
    print(f"  A: {output['answer'][:80]}...")

# ── Run RAGAS evaluation ──────────────────────────────────────────────────────
print("\nRunning RAGAS evaluation...")
dataset   = Dataset.from_list(results)
local_llm = LangchainLLMWrapper(ChatOllama(model=MODEL, temperature=0))

score = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=local_llm,
)

print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)
print(score)
print()
print(score.to_pandas().to_string())