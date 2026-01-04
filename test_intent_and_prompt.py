import pytest
from main import get_adaptive_prompt

# Mock data for tests
MOCK_CONTEXT = [{"text": "Arjuna was a great archer.", "source": "Mahabharata"}]
MOCK_HISTORY = [
    {"role": "user", "content": "Who is Arjuna?"},
    {"role": "model", "content": "He is a Pandava prince."}
]

def test_factual_intent_generates_expert_prompt():
    """
    Verifies that 'factual' intent triggers the Expert persona.
    Checks for:
    1. Presence of Expert rules (conciseness, factual accuracy).
    2. Absence of Therapist language (emotional safety, etc).
    """
    prompt = get_adaptive_prompt(
        question="Who is Arjuna?",
        context=MOCK_CONTEXT,
        history=MOCK_HISTORY,
        intent="factual"
    )

    # Expert Persona Markers
    assert "Answer the question with factual accuracy." in prompt
    assert "2–3 sentences ONLY." in prompt
    assert "Stop immediately after the answer." in prompt
    
    # Therapist Persona Markers (Should be ABSENT)
    assert "Respond with emotional clarity" not in prompt
    assert "Emotional safety > sounding wise" not in prompt
    assert "restraint" not in prompt

def test_guidance_intent_generates_therapist_prompt():
    """
    Verifies that 'guidance' intent triggers the Therapist persona.
    Checks for:
    1. Presence of Therapist rules (emotional clarity, conflict resolution).
    2. Absence of Expert language (strict sentence limits).
    """
    prompt = get_adaptive_prompt(
        question="I feel conflicted.",
        context=MOCK_CONTEXT,
        history=MOCK_HISTORY,
        intent="guidance"
    )

    # Therapist Persona Markers
    assert "Respond with emotional clarity and restraint." in prompt
    assert "Emotional safety > sounding wise." in prompt
    assert "always choose:\nclarity,\nemotional safety,\nand restraint." in prompt
    
    # Expert Persona Markers (Should be ABSENT)
    assert "Answer the question with factual accuracy." not in prompt
    assert "2–3 sentences ONLY." not in prompt

def test_history_injection_correctness():
    """
    Verifies that the conversation history is correctly formatted and injected.
    """
    prompt = get_adaptive_prompt(
        question="Next question",
        context=[],
        history=MOCK_HISTORY,
        intent="general" # Should default to guidance prompts often, or handle generally
    )
    
    # Check for history formatting
    assert "USER: Who is Arjuna?" in prompt
    assert "MODEL: He is a Pandava prince." in prompt

def test_default_intent_behavior():
    """
    Verifies that if no intent is passed, it defaults to 'guidance' (Therapist).
    """
    prompt = get_adaptive_prompt(
        question="Default test",
        context=[],
        history=[]
    )
    
    # Should contain Therapist rules by default
    assert "Respond with emotional clarity" in prompt

def test_safety_language_presence():
    """
    Verifies that safety and conflict resolution rules are present in guidance.
    """
    prompt = get_adaptive_prompt(
        question="Hard choice",
        context=[],
        history=[],
        intent="guidance"
    )
    
    assert "emotional safety" in prompt
    assert "helpfulness and impressiveness" in prompt
