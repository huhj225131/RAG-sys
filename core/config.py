
## Set up opik để check log, embedding model và llm 
import os
import opik
import json
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from opik.integrations.llama_index import LlamaIndexCallbackHandler
from .model import LLM_Large, LLM_Small, Embedding # Import model của bạn

def setup_opik():
    enable_opik = os.getenv("ENABLE_OPIK", "False").lower() == "true"

    if enable_opik:
        import opik
        from llama_index.core.callbacks import CallbackManager
        from opik.integrations.llama_index import LlamaIndexCallbackHandler
        
        prj_name = os.getenv("OPIK_PROJECT_NAME", "Default Project")
        
        # Cấu hình Opik
        opik.configure(use_local=False)
        opik_callback = LlamaIndexCallbackHandler(
            project_name=prj_name,
            skip_index_construction_trace=True 
        )
        Settings.callback_manager = CallbackManager([opik_callback])
        # print(f"✅ Opik is ENABLED (Project: {prj_name})")
    
    
    