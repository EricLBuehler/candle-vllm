import openai
import asyncio
from openai import Stream
from openai.types.chat import ChatCompletionChunk
from typing import List
import argparse
# Run candle-vllm service: cargo run --release -- --port 2000 --model-id <MODEL_ID> <MODEL_TYPE> --repeat-last-n 64
# MODEL_ID is the huggingface model id or local weight path
# MODEL_TYPE is one of ["llama", "llama3", "mistral", "phi2", "phi3", "qwen2", "gemma", "yi", "stable-lm"]
# Then run this file: python3 examples/benchmark.py --batch 16

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:2000/v1/"

# You may add your custom prompts here
PROMPT_CANDIDATES = ["Explain how to best learn Rust.", 
            "Please talk about deep learning.", 
            "Do you know the capital city of China? Talk the details of you known.", 
            "Who is the best female actor in the world? Explain why.",
            "Let me know how to deal with depression?",
            "How to make money in short time?",
            "What is the future trend of large language model?",
            "The famous tech companies in the world."]

async def chat_completion(model, max_tokens, prompt):
    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens = max_tokens,
        stream=True,
    )
    return completion

async def stream_response(response_idx, stream: Stream[ChatCompletionChunk]):
    result = ""
    for o in stream:
        r = o.choices[0].delta.content
        if r != None:
            result += r
    return (response_idx, result)

async def benchmark(batch, max_tokens=1024):
    model = "any" # model used dependent on the server side
    # candidate requests
    prompts = []
    for i in range(batch):
        prompts.append(PROMPT_CANDIDATES[i % len(PROMPT_CANDIDATES)])

    # avoid generating very short answers
    for i in range(len(prompts)):
        prompts[i] = prompts[i] + " Respond in more than {} words.".format(int(max_tokens / 10) * 10)

    # send 16 chat requests at the same time
    tasks: List[asyncio.Task] = []
    for i in range(len(prompts)):
        tasks.append(
            asyncio.create_task(
                chat_completion(model, max_tokens, prompts[i]))
        )

    # obtain the correspond stream object for each request
    outputs: List[Stream[ChatCompletionChunk]] = await asyncio.gather(*tasks)

    # tasks for streaming chat responses
    tasks_stream: List[asyncio.Task] = []
    for i in range(len(outputs)):
        tasks_stream.append(
            asyncio.create_task(
                stream_response(i, outputs[i]))
        )

    # gathering the response texts
    outputs: List[(int, str)] = await asyncio.gather(*tasks_stream)

    # print the results, you may find chat completion statistics in the backend server (i.e., candle-vllm)
    for idx, output in outputs:
        print("\n\n Response {}: \n\n {}".format(idx, output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using 'batch' and 'max_tokens' parameters for candle-vllm benchmark.")
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--max_tokens', default=1024, type=int)
    args = parser.parse_args()
    asyncio.run(benchmark(args.batch, args.max_tokens))