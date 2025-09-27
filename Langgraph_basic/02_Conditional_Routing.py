from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class ConditionalState(TypedDict):
    input_text: str
    classification: str
    response: str

def classify_input(state):
    # Simple classification logic
    text = state["input_text"].lower()
    if "question" in text or "?" in text:
        classification = "question"
    elif "help" in text:
        classification = "help"
    else:
        classification = "statement"
    
    return {"classification": classification}

def handle_question(state):
    return {"response": f"Answering your question about: {state['input_text']}"}

def handle_help(state):
    return {"response": "Here's some help information"}

def handle_statement(state):
    return {"response": f"I understand your statement: {state['input_text']}"}

def route_based_on_classification(state):
    classification = state.get("classification", "")
    if classification == "question":
        return "handle_question"
    elif classification == "help":
        return "handle_help"
    else:
        return "handle_statement"

# Build the graph
workflow = StateGraph(ConditionalState)

workflow.add_node("classify", classify_input)
workflow.add_node("handle_question", handle_question)
workflow.add_node("handle_help", handle_help)
workflow.add_node("handle_statement", handle_statement)

# Add conditional edges
workflow.add_conditional_edges(
    "classify",
    route_based_on_classification,
    {
        "handle_question": "handle_question",
        "handle_help": "handle_help",
        "handle_statement": "handle_statement"
    }
)

# End edges
workflow.add_edge("handle_question", END)
workflow.add_edge("handle_help", END)
workflow.add_edge("handle_statement", END)

workflow.set_entry_point("classify")
app = workflow.compile()

# Test with different inputs
test_cases = [
    {"input_text": "What is the weather?", "classification": "", "response": ""},
    {"input_text": "I need help", "classification": "", "response": ""},
    {"input_text": "This is a statement", "classification": "", "response": ""}
]

for test in test_cases:
    result = app.invoke(test)
    print(f"Input: {test['input_text']} -> {result['response']}")