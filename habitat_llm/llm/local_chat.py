import json
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from habitat_llm.llm.base_llm import BaseLLM


class LocalChat(BaseLLM):
    def __init__(self, conf):
        super().__init__(conf)
        self.url = str(conf.base_url).rstrip("/") + "/chat/completions"
        self.timeout = float(conf.request_timeout)
        # Local requests should not pass through an HTTP proxy.
        self.http = build_opener(ProxyHandler({}))

    def generate(
        self, prompt, stop=None, max_length=None, generation_args=None
    ):
        if not isinstance(prompt, str):
            raise TypeError("This backend currently accepts text prompts only.")
        if generation_args:
            raise ValueError(
                "Grammar-constrained generation is not implemented. "
                "Set constrained_generation=False."
            )

        params = self.generation_params
        payload = {
            "model": str(params.model),
            "messages": [
                {
                    "role": "system",
                    "content": str(self.llm_conf.system_message),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(params.temperature),
            "top_p": float(params.top_p),
            "max_tokens": int(
                max_length if max_length is not None else params.max_tokens
            ),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if stop:
            payload["stop"] = stop

        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.http.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc

        self.response = result
        choice = result["choices"][0]
        text = choice["message"].get("content") or ""
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "LLM output reached max_tokens. Inspect the prompt/output "
                "and increase the token limit if necessary."
            )
        if not text.strip():
            raise RuntimeError(f"Empty LLM response: {result}")
        return text
