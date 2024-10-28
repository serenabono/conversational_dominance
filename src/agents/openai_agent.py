### Class for OpenAI API to run GPT-3-turbo model

from openai import OpenAI
import json
import os

from retrying import retry

class OpenAIAgent:
    def __init__(self, engine="davinci", model_name="gpt-4-turbo", temperature=0.9, max_tokens=100):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.api_key = ""
        print(self.api_key)
        self.model_name = model_name

        assert self.api_key is not None, "OpenAI API key not found in environment variables"

        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=self.api_key,
        )

    @retry(stop_max_attempt_number=2, wait_fixed=1000)
    def get_response(self, prompt, model_name="gpt-3.5-turbo", temperature=0.1, max_tokens=150):
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                # request_timeout=30,
            )
        except Exception as e:
            print(e)
            print("\n----Error in get_response-----\n")
            
        return response
    
    @staticmethod
    def response_text_content(response):
        return response.choices[0].message.content
