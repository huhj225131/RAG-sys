from typing import  List, Any,Dict
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.embeddings import BaseEmbedding
import json, requests


with open("api-keys.json", "r", encoding="utf-8") as f:
    keys = json.load(f)

token_ids = {}
authors = {}
token_keys = {}
for key in keys:
    token_ids[key["llmApiName"]] = key["tokenId"]
    authors[key["llmApiName"]] = key["authorization"]
    token_keys[key["llmApiName"]]= key["tokenKey"]

models = {}
models["LLM small"]= 'vnptai_hackathon_small'
models["LLM embedings"]= 'vnptai_hackathon_embedding'
models["LLM large"]='vnptai_hackathon_large'

def llm_req(author:str,
            token_id:str,
            token_key:str,
            model:str,
            prompt:List[Dict],
            temperature:float,
            top_q:float,
            top_k:int,
            n:int,
            max_completion_tokens:int,
            api_url:str,
            **kwargs):
        headers = {
                'Authorization': author,
                'Token-id': token_id,
                'Token-key': token_key,
                'Content-Type': 'application/json',
                }
        json_data = {
            'model': model,
            'messages': prompt,
            'temperature': temperature,
            'top_p': top_q,
            'top_k': top_k,
            'n': n,
            'max_completion_tokens': max_completion_tokens,
            **kwargs
            }
        try:
            resp = requests.post(api_url, headers=headers, json=json_data)
            resp.raise_for_status()  # bắt lỗi HTTP 4xx / 5xx
            data = resp.json()

            # bắt lỗi API trả về JSON không có "choices"
            if "choices" not in data:
                raise ValueError(f"API response malformed: {data}")

            return data

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request failed: {str(e)}")

        except ValueError as e:
            raise RuntimeError(f"API returned invalid format: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Unknown error: {str(e)}")
            
        
class LLM_Small(CustomLLM):
    temperature:float = 1.0 
    top_q:float = 0.6
    top_k:int = 10
    n:int = 1
    model_name:str = "LLM small"
    model:str = models[model_name]
    max_completion_tokens:int = 20
        
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name = self.model,
            temperature= self.temperature,
            num_output= self.max_completion_tokens
        )

    @llm_completion_callback()
    def complete(self, prompt: List[Dict], **kwargs: Any) -> CompletionResponse:
        response = llm_req(authors[self.model_name], token_ids[self.model_name],
                           token_keys[self.model_name],self.model,
                           prompt, self.temperature,self.top_q,self.top_k,self.n,self.max_completion_tokens,
                           api_url='https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small',
                           **kwargs)
        
        return CompletionResponse(text=response['choices'][-1]["message"]['content'],raw=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: List[Dict], **kwargs: Any
    ) -> CompletionResponseGen:
        response = llm_req(authors[self.model_name], token_ids[self.model_name],
                           token_keys[self.model_name],self.model,
                           prompt, self.temperature,self.top_q,self.top_k,self.n,self.max_completion_tokens,
                           api_url='https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small',
                           **kwargs)
        full_text = response['choices'][-1]["message"]['content']
        words = full_text.split()
        cum = ""
        for w in words:
            if cum == "":
                cum = w
            else:
                cum += " " + w

            yield CompletionResponse(
                text=cum,
                delta=w,
                raw=response if cum == full_text else None
            )

class LLM_Large(CustomLLM):
    temperature:float = 1.0 
    top_q:float = 0.6
    top_k:int = 10
    n:int = 1
    max_completion_tokens:int = 20
    model_name:str = "LLM large"
    model:str = models[model_name]
    
        
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name = self.model,
            temperature= self.temperature,
            num_output= self.max_completion_tokens
        )

    @llm_completion_callback()
    def complete(self, prompt: List[Dict], **kwargs: Any) -> CompletionResponse:
        response = llm_req(authors[self.model_name], token_ids[self.model_name],
                           token_keys[self.model_name],self.model,
                           prompt, self.temperature,self.top_q,self.top_k,self.n,self.max_completion_tokens,
                           api_url='https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large',
                           **kwargs)
        
        return CompletionResponse(text=response['choices'][-1]["message"]['content'],raw=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: List[Dict], **kwargs: Any
    ) -> CompletionResponseGen:
        response = llm_req(authors[self.model_name], token_ids[self.model_name],
                           token_keys[self.model_name],self.model,
                           prompt, self.temperature,self.top_q,self.top_k,self.n,self.max_completion_tokens,
                           api_url='https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large',
                           **kwargs)
        full_text = response['choices'][-1]["message"]['content']
        words = full_text.split()
        cum = ""
        for w in words:
            if cum == "":
                cum = w
            else:
                cum += " " + w

            yield CompletionResponse(
                text=cum,
                delta=w,
                raw=response if cum == full_text else None
            )


def emb_req(api_url:str, **kwargs):
    headers = {
        'Authorization': authors["LLM embedings"],
        'Token-id': token_ids["LLM embedings"],
        'Token-key': token_keys["LLM embedings"],
        'Content-Type': 'application/json',
        }
    json_data = {
        'model': models["LLM embedings"],
        **kwargs
        }
    try:
        resp = requests.post(api_url, headers=headers, json=json_data)
        resp.raise_for_status()  
        data = resp.json()
        if "data" not in data:
            raise ValueError(f"API response malformed: {data}")

        return data

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP request failed: {str(e)}")

    except ValueError as e:
        raise RuntimeError(f"API returned invalid format: {str(e)}")

    except Exception as e:
        raise RuntimeError(f"Unknown error: {str(e)}")
class Embedding(BaseEmbedding):
    model_name:str = "LLM embedings"
    model:str = models[model_name]
    encoding_format:str = "float"

    def _get_query_embedding(self, query: str) -> List[float]:
        resp = emb_req(api_url='https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding',
                       input=query, encoding_format=self.encoding_format)
        return resp["data"][0]["embedding"]
    def _get_text_embedding(self, text: str) -> List[float]:
        resp = emb_req(api_url='https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding',
                       input=text, encoding_format=self.encoding_format)
        return resp["data"][0]["embedding"]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        resp = emb_req(api_url='https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding',
                       input=texts, encoding_format=self.encoding_format)
        return [item["embedding"] for item in resp["data"]]
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

# test
# respone = LLM_Large().complete([
# {
# "role": "user",
# "content": "Chào bạn!"
# }
# ])
# print(respone)
# print(Embedding()._get_text_embeddings(texts=["hehe", "hehehe"])[0])