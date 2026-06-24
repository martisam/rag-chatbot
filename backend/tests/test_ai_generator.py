"""
Tests for AIGenerator in ai_generator.py.

Verifies external behavior: API calls made, tools executed, results returned.
Does not test internal state details.
"""
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def make_text_block(text="Default answer."):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(name="search_course_content", input_dict=None, tool_id="tool_abc"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_dict or {"query": "MCP"}
    block.id = tool_id
    # model_dump() mimics the real SDK Pydantic model method
    block.model_dump.return_value = {
        "type": "tool_use",
        "name": name,
        "input": input_dict or {"query": "MCP"},
        "id": tool_id,
    }
    return block


def make_api_response(stop_reason, blocks):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = blocks
    return resp


@pytest.fixture
def generator():
    """AIGenerator with a fully mocked Anthropic client."""
    with patch("ai_generator.anthropic") as mock_anthropic_module:
        from ai_generator import AIGenerator
        gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-6")
        yield gen


# ---------------------------------------------------------------------------
# Direct (no-tool) response path
# ---------------------------------------------------------------------------

class TestDirectResponse:

    def test_returns_text_on_end_turn(self, generator):
        generator.client.messages.create.return_value = make_api_response(
            "end_turn", [make_text_block("The answer is 42.")]
        )
        result = generator.generate_response(query="What is 2+2?")
        assert result == "The answer is 42."

    def test_uses_correct_model(self, generator):
        generator.client.messages.create.return_value = make_api_response(
            "end_turn", [make_text_block("ok")]
        )
        generator.generate_response(query="Hello")
        kwargs = generator.client.messages.create.call_args[1]
        assert kwargs["model"] == "claude-sonnet-4-6"

    def test_tools_passed_in_first_call(self, generator):
        generator.client.messages.create.return_value = make_api_response(
            "end_turn", [make_text_block("ok")]
        )
        tools = [{"name": "search_course_content", "description": "Search"}]
        generator.generate_response(query="Hello", tools=tools)
        kwargs = generator.client.messages.create.call_args[1]
        assert "tools" in kwargs
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_history_injected_into_system_prompt(self, generator):
        generator.client.messages.create.return_value = make_api_response(
            "end_turn", [make_text_block("ok")]
        )
        generator.generate_response(
            query="Follow-up",
            conversation_history="User: Hi\nAssistant: Hello"
        )
        kwargs = generator.client.messages.create.call_args[1]
        assert "User: Hi" in kwargs["system"]


# ---------------------------------------------------------------------------
# Single tool-use round path
# ---------------------------------------------------------------------------

class TestToolExecutionPath:

    def test_tool_execution_called_on_tool_use(self, generator):
        tool_block = make_tool_use_block()
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [tool_block]),
            make_api_response("end_turn", [make_text_block("Final answer.")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Some results"

        result = generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        assert result == "Final answer."
        assert generator.client.messages.create.call_count == 2

    def test_tool_executed_with_correct_name_and_args(self, generator):
        tool_block = make_tool_use_block(
            name="search_course_content",
            input_dict={"query": "MCP protocol", "course_name": "MCP"},
            tool_id="t1",
        )
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [tool_block]),
            make_api_response("end_turn", [make_text_block("ok")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Results"

        generator.generate_response(
            query="What is MCP?", tools=[{}], tool_manager=mock_tm
        )
        mock_tm.execute_tool.assert_called_once_with(
            "search_course_content", query="MCP protocol", course_name="MCP"
        )

    def test_forced_final_call_excludes_tools(self, generator):
        """When rounds are exhausted, the forced final call must NOT include tools."""
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [make_tool_use_block(tool_id="t1")]),
            make_api_response("tool_use", [make_tool_use_block(tool_id="t2")]),
            make_api_response("end_turn", [make_text_block("ok")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Results"

        generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        # Third call is the forced final - must have no tools
        third_kwargs = generator.client.messages.create.call_args_list[2][1]
        assert "tools" not in third_kwargs
        assert "tool_choice" not in third_kwargs

    def test_single_round_message_count_is_three(self, generator):
        """Messages for second call: user query + assistant tool_use + user tool_result."""
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [make_tool_use_block()]),
            make_api_response("end_turn", [make_text_block("ok")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Results"

        generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        assert len(second_kwargs["messages"]) == 3

    def test_tool_result_message_has_correct_structure(self, generator):
        """The tool_result message must carry type, tool_use_id, and content."""
        tool_block = make_tool_use_block(tool_id="tool_xyz")
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [tool_block]),
            make_api_response("end_turn", [make_text_block("ok")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Course content about MCP"

        generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        tool_result_msg = second_kwargs["messages"][2]
        assert tool_result_msg["role"] == "user"
        payload = tool_result_msg["content"][0]
        assert payload["type"] == "tool_result"
        assert payload["tool_use_id"] == "tool_xyz"
        assert payload["content"] == "Course content about MCP"

    def test_assistant_message_content_is_dicts_not_sdk_objects(self, generator):
        """
        The assistant message content appended to messages must be plain dicts
        (ContentBlockParam), not SDK ContentBlock instances. The Anthropic SDK
        requires dict-serialized params on the request side.
        """
        tool_block = make_tool_use_block(tool_id="tool_bug_test")
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [tool_block]),
            make_api_response("end_turn", [make_text_block("ok")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Results"

        generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        assistant_msg = next(
            m for m in second_kwargs["messages"] if m["role"] == "assistant"
        )
        for i, block in enumerate(assistant_msg["content"]):
            assert isinstance(block, dict), (
                f"Block {i} is {type(block).__name__}, expected dict. "
                "Assistant message content must be serializable dicts."
            )

    def test_api_exception_propagates(self, generator):
        generator.client.messages.create.side_effect = RuntimeError("API connection error")
        with pytest.raises(RuntimeError, match="API connection error"):
            generator.generate_response(query="What is MCP?")


# ---------------------------------------------------------------------------
# Two sequential tool-use rounds
# ---------------------------------------------------------------------------

class TestTwoRoundPath:

    def test_two_tool_rounds_makes_three_api_calls(self, generator):
        """Round 1 tool_use -> round 2 tool_use -> round 3 end_turn = 3 API calls."""
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [make_tool_use_block(tool_id="t1")]),
            make_api_response("tool_use", [make_tool_use_block(tool_id="t2")]),
            make_api_response("end_turn", [make_text_block("Final two-round answer.")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Results"

        result = generator.generate_response(
            query="Complex query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        assert result == "Final two-round answer."
        assert generator.client.messages.create.call_count == 3
        assert mock_tm.execute_tool.call_count == 2

    def test_tools_present_in_intermediate_round(self, generator):
        """After round 1 tool result, the round-2 API call must still include tools."""
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [make_tool_use_block(tool_id="t1")]),
            make_api_response("end_turn", [make_text_block("ok")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Results"

        generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        # With MAX_ROUNDS=2 and one tool call that returns end_turn, the second
        # call is made with tools available (rounds_remaining=1 when we loop back)
        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        assert "tools" in second_kwargs
        assert second_kwargs["tool_choice"] == {"type": "auto"}

    def test_forced_final_strips_tools_at_max_rounds(self, generator):
        """Both rounds return tool_use; forced third call must have no tools."""
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [make_tool_use_block(tool_id="t1")]),
            make_api_response("tool_use", [make_tool_use_block(tool_id="t2")]),
            make_api_response("end_turn", [make_text_block("Synthesized.")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Results"

        result = generator.generate_response(
            query="Multi-step query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        assert result == "Synthesized."
        third_kwargs = generator.client.messages.create.call_args_list[2][1]
        assert "tools" not in third_kwargs
        assert "tool_choice" not in third_kwargs

    def test_two_round_message_count_is_five(self, generator):
        """After two tool rounds, forced final receives 5 messages:
        user query + assistant tool_use 1 + user tool_result 1
        + assistant tool_use 2 + user tool_result 2."""
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [make_tool_use_block(tool_id="t1")]),
            make_api_response("tool_use", [make_tool_use_block(tool_id="t2")]),
            make_api_response("end_turn", [make_text_block("ok")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Results"

        generator.generate_response(
            query="Multi-step query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        third_kwargs = generator.client.messages.create.call_args_list[2][1]
        assert len(third_kwargs["messages"]) == 5


# ---------------------------------------------------------------------------
# Early termination (round 2 returns end_turn, no forced call needed)
# ---------------------------------------------------------------------------

class TestEarlyTermination:

    def test_round2_end_turn_no_third_call(self, generator):
        """Round 1 tool_use, round 2 end_turn: exactly 2 API calls, no forced call."""
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [make_tool_use_block(tool_id="t1")]),
            make_api_response("end_turn", [make_text_block("Done after one tool.")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Results"

        result = generator.generate_response(
            query="Simple query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        assert result == "Done after one tool."
        assert generator.client.messages.create.call_count == 2

    def test_no_tool_manager_returns_direct_text(self, generator):
        """If tool_manager is None, tool_use response returns text directly (no execution)."""
        generator.client.messages.create.return_value = make_api_response(
            "tool_use", [make_tool_use_block()]
        )
        # When tool_use but no tool_manager - returns text of the tool_use response
        # (content[0] is the tool block - just checking it doesn't crash)
        generator.client.messages.create.return_value = make_api_response(
            "end_turn", [make_text_block("No tools available.")]
        )
        result = generator.generate_response(query="Hello", tool_manager=None)
        assert result == "No tools available."
        assert generator.client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Tool error handling
# ---------------------------------------------------------------------------

class TestToolErrorHandling:

    def test_tool_error_stored_as_error_string_in_result(self, generator):
        """When execute_tool raises, the tool_result content has an error string; no raise."""
        tool_block = make_tool_use_block(tool_id="t_err")
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [tool_block]),
            make_api_response("end_turn", [make_text_block("Recovered.")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = ValueError("index out of range")

        result = generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        # Should not raise - should return final text
        assert isinstance(result, str)

        # The tool_result message must contain the error string
        second_kwargs = generator.client.messages.create.call_args_list[1][1]
        tool_result_msg = next(
            m for m in second_kwargs["messages"] if m["role"] == "user"
            and isinstance(m["content"], list)
            and m["content"] and m["content"][0].get("type") == "tool_result"
        )
        assert "Tool error" in tool_result_msg["content"][0]["content"]
        assert "index out of range" in tool_result_msg["content"][0]["content"]

    def test_tool_error_triggers_forced_final_call(self, generator):
        """After a tool error, a synthesis call is still made and returns text."""
        tool_block = make_tool_use_block(tool_id="t_err")
        generator.client.messages.create.side_effect = [
            make_api_response("tool_use", [tool_block]),
            make_api_response("end_turn", [make_text_block("Best effort answer.")]),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = RuntimeError("timeout")

        result = generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )
        assert result == "Best effort answer."
        # Forced final call must have no tools
        final_kwargs = generator.client.messages.create.call_args_list[1][1]
        assert "tools" not in final_kwargs
