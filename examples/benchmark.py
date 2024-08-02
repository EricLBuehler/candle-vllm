import openai
import asyncio
from openai import Stream
from openai.types.chat import ChatCompletionChunk
from typing import List
# Run: cargo run --release -- --port 2000 --model-id <MODEL_ID> <MODEL_TYPE> --repeat-last-n 64
# MODEL_ID is the huggingface model id or local weight path
# MODEL_TYPE is one of ["llama", "llama3", "mistral", "phi2", "phi3", "qwen2", "gemma", "yi", "stable-lm"]


openai.api_key = "EMPTY"

openai.base_url = "http://localhost:2000/v1/"

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

async def benchmark():
    model = "mistral7b"
    max_tokens = 1024
    # 16 requests
    prompts = ["Explain how to best learn Rust.", 
               "Please talk about deep learning.", 
               "Do you know the capital city of China? Talk the details of you known.", 
               "Who is the best female actor in the world? Explain why.",
               "Let me know how to deal with depression?",
               "How to make money in short time?",
               "What is the future trend of large language model?",
               "The famous tech companies in the world.",
               "Explain how to best learn Rust.", 
               "Please talk about deep learning.", 
               "Do you know the capital city of China? Talk the details of you known.", 
               "Who is the best female actor in the world? Explain why.",
               "Let me know how to deal with depression?",
               "How to make money in short time?",
               "What is the future trend of large language model?",
               "The famous tech companies in the world."]

    # avoid generating very short answers
    for i in range(len(prompts)):
        prompts[i] = prompts[i] + " Describe in about {} words.".format((int(max_tokens / 1.3 / 10) + 1) * 10)

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


asyncio.run(benchmark())