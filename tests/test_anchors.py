from src.anchors import contains_anchor, count_bullets, count_words, extract_anchors, numeric_values
from src.schemas import Anchor, AnchorKind


def by_kind(anchors, kind):
    return [a for a in anchors if a.kind is kind]


def test_extracts_numbers_with_units_and_currency():
    anchors = extract_anchors("Latency fell 36% from 2.5 s to 1.6 s, saving $4.2M across 10,000 RPS in Q3 2025.")
    numbers = [a.text for a in by_kind(anchors, AnchorKind.NUMBER)]
    assert numbers == ["36%", "2.5 s", "1.6 s", "$4.2M", "10,000 RPS", "Q3 2025"]


def test_extracts_entities_and_ignores_instruction_verbs():
    anchors = extract_anchors("Give a summary of how General Motors used PostgreSQL and FAISS for p99 tracking.")
    entities = [a.text for a in by_kind(anchors, AnchorKind.ENTITY)]
    assert "General Motors" in entities
    assert "PostgreSQL" in entities and "FAISS" in entities and "p99" in entities
    assert "Give" not in entities


def test_constraints_have_rules_and_do_not_leak_into_numbers():
    anchors = extract_anchors("Compare Redis and Kafka in under 80 words, in exactly 3 bullet points, and do not mention Pulsar.")
    rules = {a.rule for a in by_kind(anchors, AnchorKind.CONSTRAINT)}
    assert rules == {("max_words", 80), ("bullets", 3), ("forbid", "Pulsar")}
    assert by_kind(anchors, AnchorKind.NUMBER) == []
    assert "Pulsar" not in [a.text for a in by_kind(anchors, AnchorKind.ENTITY)]


def test_forbid_term_strips_trailing_punctuation():
    anchors = extract_anchors("Summarise the incident without mentioning Kafka.")
    assert [a.rule for a in by_kind(anchors, AnchorKind.CONSTRAINT)] == [("forbid", "Kafka")]


def test_extraction_is_deterministic_with_stable_ids_and_no_duplicates():
    query = "RaftKV kept 5 nodes healthy; node 2 failed, then node 2 recovered in 180 ms."
    first, second = extract_anchors(query), extract_anchors(query)
    assert first == second
    assert len({a.id for a in first}) == len(first)
    assert [a.text for a in by_kind(first, AnchorKind.NUMBER)].count("2") == 1


def test_contains_anchor_is_verbatim_but_whitespace_and_case_tolerant():
    anchor = Anchor(id="a1", kind=AnchorKind.NUMBER, text="2.5 s")
    assert contains_anchor("Down from 2.5  S before.", anchor)
    assert not contains_anchor("Down from 2.7 s before.", anchor)
    assert not contains_anchor("Down from 12.5 s before.", anchor)


def test_numeric_values_and_counters():
    assert numeric_values("36% of 10,000 requests took 2.5 s") == {"36", "10000", "2.5"}
    assert count_words("one two  three\nfour") == 4
    assert count_bullets("- a\n* b\n1. c\nplain") == 3
