import os
import pandas as pd
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


class LlamaModel():

    def __init__(self):
        # GPU llama-cpp-python
        # Set environment variables
        os.environ['CMAKE_ARGS'] = "-DLLAMA_CUBLAS=on"
        os.environ['FORCE_CMAKE'] = "1"


        model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
        model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format

        # Download the model
        self.model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    
        self.lcpp_llm = Llama(
            model_path=self.model_path,
            n_threads=2, # CPU cores
            n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_gpu_layers=43, # Change this value based on your model and your GPU VRAM pool.
            n_ctx=4096, # Context window
            logits_all=True
        )

        self.default_prompt_template = '''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

        USER: {}

        ASSISTANT: '''
    
    def get_response(
            self,
            prompt,
            prompt_template=None,
            max_tokens=256,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            logprobs=10,
            stop=['USER:']):
        if prompt_template is None:
            prompt_template = self.default_prompt_template

        context = prompt_template.format(prompt)
        
        response = self.lcpp_llm(
            prompt=context,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.2,
            top_k=top_k,
            logprobs=logprobs,
            stop = stop, # Dynamic stopping when such token is detected.
            echo=True # return the prompt
        )

        logprobs_df = pd.DataFrame(response["choices"][0]["logprobs"]["top_logprobs"])

        return response["choices"][0]["text"], logprobs_df
    