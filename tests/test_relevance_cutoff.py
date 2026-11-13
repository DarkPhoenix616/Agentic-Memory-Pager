"""
tests/test_relevance_cutoff.py — page_context leaves out memories that are not
about the question (the relevance cut-off, report section 5o).

In-process server on the temporary test vault from conftest.py.

Run with:
    pytest tests/test_relevance_cutoff.py -v
"""
import pytest

pytest.importorskip("chromadb", reason="the memory server needs chromadb")

from harness.local_keys import session_key  # noqa: E402
from tests.test_server_guards import store, text_memory  # noqa: E402

CAT = "The user said their cat is called Pixel."
FAR = ["The team explained how TCP congestion control works with slow start.",
       "The assistant described the CAP theorem with examples of databases.",
       "The user asked how B-tree indexes keep their pages balanced."]


@pytest.fixture
def server():
    import server.amp_memory_server as server
    return server


def _page(server, sid, **kw):
    return server.page_context("What is my cat called?", 1000, sid, session_key=session_key(sid), **kw)


def _plant(server, sid):
    for text in [CAT, *FAR]:
        store(server, text_memory(sid, text))


def test_without_a_cut_off_every_match_that_fits_is_paged_in(server, monkeypatch):
    monkeypatch.setenv("AMP_PAGE_MAX_DISTANCE", "off")
    _plant(server, "rc-off")
    result = _page(server, "rc-off")
    assert len(result.memories) == 4 and result.too_far == 0


def test_far_memories_are_left_out_and_counted(server, monkeypatch):
    monkeypatch.delenv("AMP_PAGE_MAX_DISTANCE", raising=False)
    _plant(server, "rc-on")
    result = _page(server, "rc-on", max_distance=1.2)
    assert [m.memory.text_payload.factual_summary for m in result.memories] == [CAT]
    assert result.too_far == 3 and result.tokens_used == result.memories[0].token_cost
    assert all(m.distance <= 1.2 for m in result.memories)


def test_the_cut_off_comes_from_the_caller_then_the_environment_then_the_default(server, monkeypatch):
    general = "Explain the CAP theorem."
    monkeypatch.setenv("AMP_PAGE_MAX_DISTANCE", "1.3")
    assert server._page_max_distance(None, general) == 1.3
    assert server._page_max_distance(0.9, general) == 0.9
    assert server._page_max_distance(0, general) is None             # 0 from the caller: no cut-off
    monkeypatch.setenv("AMP_PAGE_MAX_DISTANCE", "off")
    assert server._page_max_distance(None, general) is None
    monkeypatch.setenv("AMP_PAGE_MAX_DISTANCE", "far")               # not a number: the default, with a warning
    assert server._page_max_distance(None, general) == server.DEFAULT_PAGE_MAX_DISTANCE == 1.4
    monkeypatch.delenv("AMP_PAGE_MAX_DISTANCE")
    assert server._page_max_distance(None, general) == 1.4


@pytest.mark.parametrize("question", ["What do you remember about me?", "What did we talk about earlier?",
                                      "Which database did I say we must never use?", "Remind me of the plan.",
                                      "Tell me everything you know about our setup."])
def test_questions_about_the_user_or_the_chat_have_no_cut_off(server, monkeypatch, question):
    monkeypatch.delenv("AMP_PAGE_MAX_DISTANCE", raising=False)
    assert server._page_max_distance(None, question) is None


@pytest.mark.parametrize("question", ["Explain the CAP theorem with examples.", "Which port does the memory server "
                                      "run on?", "What was the bug in pager_v3.py?", "How do hash tables work?"])
def test_other_questions_get_the_cut_off(server, monkeypatch, question):
    monkeypatch.delenv("AMP_PAGE_MAX_DISTANCE", raising=False)
    assert server._page_max_distance(None, question) == 1.4


def test_a_general_question_pages_in_nothing_unrelated_by_default(server, monkeypatch):
    monkeypatch.delenv("AMP_PAGE_MAX_DISTANCE", raising=False)
    _plant(server, "rc-general")
    result = server.page_context("Explain how a hash map handles collisions.", 1000, "rc-general",
                                 session_key=session_key("rc-general"))
    assert result.memories == [] and result.too_far == 4
    personal = _page(server, "rc-general")                           # "What is my cat called?": as before
    assert len(personal.memories) == 4 and personal.too_far == 0


def test_nothing_relevant_means_nothing_is_paged_in(server, monkeypatch):
    monkeypatch.delenv("AMP_PAGE_MAX_DISTANCE", raising=False)
    for text in FAR:
        store(server, text_memory("rc-none", text))
    result = _page(server, "rc-none", max_distance=1.2)
    assert result.memories == [] and result.too_far == 3 and result.tokens_used == 0
