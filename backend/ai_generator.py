import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive set of tools for course information.

Tool Usage:
- **`search_course_content`**: Use for questions about specific course content or detailed educational materials
- **`get_course_outline`**: Use for questions about course structure, what lessons a course contains, or requests for a course overview/outline
- **Up to two sequential tool calls**: Use a second call only when the first result is insufficient to answer the question. Stop as soon as you have enough to answer.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Outline Response Format:
- When responding to an outline query, present: course title, course link (if available), then a numbered list of lessons with their lesson number and title

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use `search_course_content` first, then answer
- **Outline/structure questions**: Use `get_course_outline` first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_ROUNDS sequential tool-calling rounds.
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        rounds_remaining = self.MAX_ROUNDS

        while rounds_remaining > 0:
            response = self.client.messages.create(**api_params)

            if response.stop_reason != "tool_use" or tool_manager is None:
                return response.content[0].text

            success = self._run_tool_round(messages, response, tool_manager)
            rounds_remaining -= 1

            if not success or rounds_remaining == 0:
                final_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": system_content,
                }
                return self.client.messages.create(**final_params).content[0].text

            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
                "tools": tools,
                "tool_choice": {"type": "auto"},
            }

        return ""  # unreachable, satisfies type checker

    def _run_tool_round(self, messages: List[Dict], response, tool_manager) -> bool:
        """
        Serialize the assistant response and execute all tool calls onto messages.
        Returns True if all tools succeeded, False if any failed.
        """
        messages.append(
            {
                "role": "assistant",
                "content": [block.model_dump() for block in response.content],
            }
        )

        tool_results = []
        all_succeeded = True

        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                except Exception as e:
                    result = f"Tool error: {str(e)}"
                    all_succeeded = False

                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return all_succeeded
