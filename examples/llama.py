import openai

# Run: HF_TOKEN=... cargo run --release -- --hf-token HF_TOKEN --port 2000 llama --repeat-last-n 64

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:2000/v1/"

completion = openai.chat.completions.create(
    model="llama",
    messages=[
        {
            "role": "user",
            "content": "How should I learn to type?",
        },
    ],
    max_tokens = 32,
)
print(completion.choices[0].message.content)