from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .llm import chat as llm_chat


def _to_langchain_message(message: dict) -> BaseMessage:
    role = message.get("role")
    content = message.get("content") or ""
    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def _from_langchain_messages(messages: List[BaseMessage]) -> List[dict]:
    converted: List[dict] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            converted.append({"role": "system", "content": message.content})
        elif isinstance(message, AIMessage):
            converted.append({"role": "assistant", "content": message.content})
        else:
            converted.append({"role": "user", "content": message.content})
    return converted


class AppChatModel(BaseChatModel):
    provider: str
    base_url: str
    model: str
    api_key: Optional[str] = None
    temperature: Optional[float] = None

    @property
    def _llm_type(self) -> str:
        return "app-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        payload = _from_langchain_messages(messages)
        content = llm_chat(
            provider=self.provider,
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            messages=payload,
            temperature=self.temperature,
        )
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])
