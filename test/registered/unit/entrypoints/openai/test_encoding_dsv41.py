"""Unit tests for the DeepSeek-V4.1 chat encoder -- no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.entrypoints.openai import encoding_dsv41
from sglang.srt.entrypoints.openai.encoding_dsv41 import (
    encode_messages,
)
from sglang.test.test_utils import CustomTestCase


def _tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }


class TestDsmlTags(CustomTestCase):
    def test_multiturn_prompt_preserves_reasoning_and_tool_results(self):
        messages = [
            {"role": "system", "content": "system", "tools": [_tool()]},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "reason",
                "content": "summary",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"query": query}},
                    }
                    for call_id, query in (("a", "first"), ("b", "second"))
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "second result"},
            {"role": "tool", "tool_call_id": "a", "content": "first result"},
        ]
        prompt = encode_messages(messages, thinking_mode="thinking")
        expected = (
            "<｜User｜>question<｜Assistant｜><think>reason</think>summary\n\n"
            "<｜DSML｜ calls>\n"
            '<｜DSML｜ invoke name="lookup">\n'
            '<｜DSML｜ parameter name="query" string="true">first</｜DSML｜ parameter>\n'
            "</｜DSML｜ invoke>\n"
            '<｜DSML｜ invoke name="lookup">\n'
            '<｜DSML｜ parameter name="query" string="true">second</｜DSML｜ parameter>\n'
            "</｜DSML｜ invoke>\n"
            "</｜DSML｜ calls><｜end▁of▁sentence｜>"
            "<｜User｜><tool_result>first result</tool_result>\n\n"
            "<tool_result>second result</tool_result><｜Assistant｜><think>"
        )
        self.assertEqual(prompt[prompt.index("<｜User｜>") :], expected)


class TestDeepSeekV41Encoding(CustomTestCase):
    def test_release_reasoning_effort_budgets(self):
        expected_budgets = {
            "low": 50,
            "high": 75,
            "xhigh": 75,
            "max": 100,
        }

        for effort, budget in expected_budgets.items():
            with self.subTest(effort=effort):
                self.assertEqual(
                    encoding_dsv41.render_reasoning_effort(
                        index=0,
                        thinking_mode="thinking",
                        effort=effort,
                    ),
                    encoding_dsv41.REASONING_EFFORT_TEMPLATE.format(budget=budget),
                )

        self.assertEqual(
            encoding_dsv41.render_reasoning_effort(
                index=0,
                thinking_mode="thinking",
                effort=None,
            ),
            encoding_dsv41.REASONING_EFFORT_TEMPLATE.format(budget=75),
        )


if __name__ == "__main__":
    unittest.main()
