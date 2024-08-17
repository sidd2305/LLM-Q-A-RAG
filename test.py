from langchain.llms import HuggingFaceEndpoint

# Initialize HuggingFace LLM endpoint
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token="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",
    max_new_tokens=6096
)

# Example usage with parameters like max_new_tokens
response = llm("What is the capital of France?")
print(response)
