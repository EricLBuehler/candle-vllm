import openai

# optional; defaults to `os.environ['OPENAI_API_KEY']`
openai.api_key = "EMPTY"

# all client options can be configured just like the `OpenAI` instantiation counterpart
openai.base_url = "http://localhost:2000/v1/"

completion = openai.chat.completions.create(
    model="llama",
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
)
print(completion.choices[0].message.content)