from openai import OpenAI

from config import KIMI_BASE_URL


class KimiClient:
    """Client for Kimi's OpenAI-compatible chat completion API."""

    def __init__(
        self,
        api_key: str,
        model: str = "kimi-k2.6",
        base_url: str = KIMI_BASE_URL,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Kimi API key is empty. Set KIMI_API_KEY or MOONSHOT_API_KEY."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def inquire_LLMs(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.5,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        request_args = {
            "model": self.model,
            "messages": messages,
        }

        if self.model in {"kimi-k2.5", "kimi-k2.6"}:
            # These models only accept provider-defined temperature values.
            # MATMCD requests 0.5/0.8, so omit temperature and use the cheaper,
            # parser-friendly non-thinking mode.
            request_args["extra_body"] = {"thinking": {"type": "disabled"}}

        response = self.client.chat.completions.create(**request_args)
        return response.choices[0].message.content
