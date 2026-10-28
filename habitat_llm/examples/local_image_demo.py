"""Send actual image pixels to the local multimodal endpoint."""

import argparse
from habitat_llm.llm import instantiate_llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Local PNG/JPEG files in time order")
    parser.add_argument(
        "--prompt",
        default="Describe what is visible in these images in order. Do not infer objects that are not visible.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    llm = instantiate_llm("local_chat")
    print(llm.generate(args.prompt, images=args.images, max_length=args.max_tokens))
    print("usage:", llm.response.get("usage", {}))


if __name__ == "__main__":
    main()
