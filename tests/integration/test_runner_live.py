"""End-to-end through ADK's Runner with live TopK.

A scripted model stands in for Gemini so the test needs no LLM key: it issues the
tool calls a model would make, and the test checks what ADK fed back to it.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory, preload_memory
from google.genai import types

from adk_topk import TopKMemoryService, TopKMemoryToolset, TopKSearchToolset

APP = "runner_test"


class ScriptedLlm(BaseLlm):
    """Calls the next scripted tool, then answers with the tool's result as text."""

    model: str = "scripted"
    script: list[dict[str, Any]] = []
    requests: list[LlmRequest] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        responses = [p.function_response for c in llm_request.contents for p in c.parts or [] if p.function_response]
        if not self.script:
            # The scripted call has run; answer with its result.
            text = json.dumps(responses[-1].response, default=str)
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
            return
        call = self.script.pop(0)
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name=call["name"], args=call["args"]))],
            )
        )


async def _ask(runner: Runner, user_id: str, session_id: str, text: str) -> str:
    sessions = runner.session_service
    if not await sessions.get_session(app_name=APP, user_id=user_id, session_id=session_id):
        await sessions.create_session(app_name=APP, user_id=user_id, session_id=session_id)
    final = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final = event.content.parts[0].text or ""
    return final


async def test_agent_remembers_recalls_searches_and_forgets(client, collection_name):
    memory = TopKMemoryService(collection_name, client=client)
    docs = TopKSearchToolset(
        collection_name + "_docs", client=client, corpus_description="support runbooks", metadata_fields={"source": "x"}
    )
    try:
        await docs.add_texts(
            ["Error E-4021 means the card was declined by the bank.", "Refunds take 5 to 7 business days."],
            metadatas=[{"source": "runbook"}, {"source": "faq"}],
        )
        llm = ScriptedLlm()
        agent = Agent(
            name="assistant",
            model=llm,
            instruction="Help the user. Remember lasting facts about them.",
            tools=[TopKMemoryToolset(memory), docs, load_memory, preload_memory],
        )
        runner = Runner(agent=agent, app_name=APP, session_service=InMemorySessionService(), memory_service=memory)

        # Session 1: the model saves a fact with topk_remember.
        llm.script = [
            {"name": "topk_remember", "args": {"memory": "The user is allergic to peanuts.", "category": "health"}}
        ]
        saved = json.loads(await _ask(runner, "alice", "s1", "Heads up: I'm allergic to peanuts."))
        assert saved["status"] == "success" and saved["memory_id"].startswith("mem-")

        # Tool declarations the model saw carry the corpus and filter hints.
        decls = {d.name: d for t in llm.requests[0].config.tools for d in t.function_declarations}
        assert {"topk_remember", "topk_recall", "topk_forget", "topk_hybrid_search", "load_memory"} <= set(decls)
        assert "support runbooks" in decls["topk_keyword_search"].description

        # Save the whole session too, as an after-agent callback would.
        s1 = await runner.session_service.get_session(app_name=APP, user_id="alice", session_id="s1")
        await memory.add_session_to_memory(s1)

        # Session 2: preload_memory injects past context, and topk_recall finds the fact.
        llm.requests.clear()
        llm.script = [{"name": "topk_recall", "args": {"query": "food allergies"}}]
        recalled = json.loads(await _ask(runner, "alice", "s2", "Suggest a snack for me."))
        texts = [m["text"] for m in recalled["memories"]]
        assert texts[0] == "The user is allergic to peanuts."
        assert recalled["memories"][0]["category"] == "health"
        # Newer ADK adds preloaded memories to the contents, older ADK to the system instruction.
        prompt = json.dumps([c.model_dump(mode="json") for c in llm.requests[0].contents])
        prompt += str(llm.requests[0].config.system_instruction)
        assert "PAST_CONVERSATIONS" in prompt and "peanuts" in prompt

        # ADK's own load_memory tool reads the same TopK partition.
        llm.script = [{"name": "load_memory", "args": {"query": "peanut allergy"}}]
        loaded = json.loads(await _ask(runner, "alice", "s2", "What do you remember about my allergies?"))
        # ADK passes load_memory's pydantic result to the model as a string.
        assert "I'm allergic to peanuts." in loaded["result"] and "The user is allergic to peanuts." in loaded["result"]

        # Another user sees nothing.
        llm.script = [{"name": "topk_recall", "args": {"query": "allergies"}}]
        assert json.loads(await _ask(runner, "bob", "s3", "What am I allergic to?"))["memories"] == []

        # Document search with an exact code and a metadata filter.
        llm.script = [{"name": "topk_keyword_search", "args": {"query": "E-4021", "filters": {"source": "runbook"}}}]
        found = json.loads(await _ask(runner, "alice", "s2", "What does E-4021 mean?"))
        assert [r["text"] for r in found["results"]] == ["Error E-4021 means the card was declined by the bank."]

        # Forget, then recall again.
        memory_id = recalled["memories"][0]["id"]
        llm.script = [{"name": "topk_forget", "args": {"memory_id": memory_id}}]
        assert json.loads(await _ask(runner, "alice", "s2", "Forget my allergy."))["deleted"] == memory_id
        llm.script = [{"name": "topk_recall", "args": {"query": "peanut allergy"}}]
        after = json.loads(await _ask(runner, "alice", "s2", "Any allergies?"))
        assert memory_id not in [m["id"] for m in after["memories"]]
    finally:
        await client.collections().delete(collection_name + "_docs")
