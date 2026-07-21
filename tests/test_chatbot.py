from __future__ import annotations

from proteomics_meta.chatbot import parse_gene_query


def test_parses_standard_symbol():
    assert parse_gene_query("tell me about TP53") == "TP53"


def test_parses_after_gene_keyword():
    assert parse_gene_query("what does the gene EGFR do") == "EGFR"


def test_returns_none_without_symbol():
    assert parse_gene_query("what is the weather today") is None
