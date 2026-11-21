"""Offline test: real LangGraph structure with fake web/AI dependencies."""
import unittest
from types import SimpleNamespace
from research_workflow import build_graph

class FakeModel:
    def invoke(self, prompt):
        if "Create a Markdown research report" in prompt:
            return SimpleNamespace(content="# Research report\n\nEvidence [1] and [2].")
        return SimpleNamespace(content="Two sources suggest useful developments [1][2].")

class FakeSearch:
    def text(self, query, max_results=3):
        return [
            {"title": "Source One", "href": "https://example.org/one", "body": "Some evidence one."},
            {"title": "Source Two", "href": "https://example.org/two", "body": "Some evidence two."},
        ]

class GraphTests(unittest.TestCase):
    def test_finite_graph_with_real_source_urls(self):
        seen = []
        result = build_graph(FakeModel(), FakeSearch, progress=seen.append).invoke({"topic":"AI agents"}, config={"recursion_limit": 10})
        self.assertEqual(len(seen), 5)
        self.assertEqual(result["searches_done"], 1)
        self.assertEqual(len(result["sources"]), 2)
        self.assertIn("https://example.org/one", result["report"])
        self.assertIn("Evidence warning", result["report"])

if __name__ == "__main__":
    unittest.main()
