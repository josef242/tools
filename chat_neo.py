# chat_neo_unified_v2.py
# Unified chat interface with universal YAML format support

import time
from prompt_toolkit import PromptSession
import yaml
import numpy as np
import sys
import os
import argparse
import re
from typing import Optional, Union, List, Tuple, Dict

# Try to import llama_cpp for GGUF support
try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

# Import logger from common_fsdp2 (pure Python, no torch dependency)
_common_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'common_fsdp2'))
if _common_path not in sys.path:
    sys.path.insert(0, _common_path)
try:
    import logger
    HAS_LOGGER = True
except ImportError:
    HAS_LOGGER = False

def log(msg):
    """Log a message via logger if available, otherwise print."""
    if HAS_LOGGER:
        logger.print_and_log(msg)
    else:
        print(msg)

def _ls_listing(directory, args, default_suffix=None):
    """Mimic basic `ls`.

    args      : sequence of tokens after the command (e.g. ["-l"], ["*.txt"]).
    -l        : long format (size + mtime, one entry per line).
    pattern   : a glob like *.txt filters the listing and overrides default_suffix.
    default_suffix : when no glob pattern is given, show only files ending with this
                     (e.g. ".yaml"); None shows everything.
    Returns a formatted string ready to print.
    """
    import fnmatch, shutil
    from datetime import datetime

    long_format = False
    pattern = None
    for tok in args:
        if tok.startswith("-"):
            if "l" in tok:
                long_format = True
        elif pattern is None:
            pattern = tok

    try:
        entries = sorted(os.listdir(directory))
    except OSError as e:
        return f"   (cannot list directory: {e})"

    if pattern is not None:
        entries = [e for e in entries if fnmatch.fnmatch(e, pattern)]
    elif default_suffix:
        entries = [e for e in entries if e.endswith(default_suffix)]

    if not entries:
        return "   (no matching files)"

    if long_format:
        lines = []
        for name in entries:
            path = os.path.join(directory, name)
            try:
                st = os.stat(path)
                size = st.st_size
                mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                flag = "/" if os.path.isdir(path) else " "
            except OSError:
                size, mtime, flag = 0, "?", " "
            lines.append(f"   {size:>10}  {mtime}  {name}{flag}")
        return "\n".join(lines)

    # Column layout (fill rows left-to-right within the terminal width).
    width = shutil.get_terminal_size((80, 24)).columns
    col_w = max(len(e) for e in entries) + 2
    ncols = max(1, width // col_w)
    rows = []
    for i in range(0, len(entries), ncols):
        rows.append("".join(e.ljust(col_w) for e in entries[i:i + ncols]).rstrip())
    return "\n".join(rows)

# Lazy loading for PyTorch - only import when needed
TORCH_LOADED = False
torch = None
F = None

def load_torch_if_needed():
    """Lazy load PyTorch only when actually needed"""
    global TORCH_LOADED, torch, F
    if not TORCH_LOADED:
        import torch as torch_module
        from torch.nn import functional as F_module
        torch = torch_module
        F = F_module
        TORCH_LOADED = True
        return True
    return False


class ChatTemplateBuilder:
    """Builds prompts for different chat formats, with support for continuation."""
    
    @staticmethod
    def build_prompt(messages: List[Dict], format_name: str, continue_last: bool = False) -> str:
        """
        Build a prompt from messages in the specified format.

        Args:
            messages: List of message dicts with 'role' and 'content'
            format_name: Name of the chat format ('llama-2', 'llama-3', 'mara', etc.)
            continue_last: If True, omit closing tags to allow continuation

        Returns:
            Formatted prompt string
        """
        if format_name == 'llama-3':
            return ChatTemplateBuilder._build_llama3_prompt(messages, continue_last)
        elif format_name == 'chatml':
            return ChatTemplateBuilder._build_chatml_prompt(messages, continue_last)
        elif format_name == 'llama-2':
            return ChatTemplateBuilder._build_llama2_prompt(messages, continue_last)
        elif format_name == 'mara':
            return ChatTemplateBuilder._build_mara_prompt(messages, continue_last)
        else:
            # Fallback - just return a basic format
            # This allows unsupported formats to at least attempt continuation
            return ChatTemplateBuilder._build_basic_prompt(messages, continue_last)

    @staticmethod
    def _build_mara_prompt(messages: List[Dict], continue_last: bool) -> str:
        """
        Build Mara format prompt using special tokens.

        Matches training format (pre_tokenize_conversations.py):
        - Format: <|bos|><|system_start|>{system}<|system_end|>
                  <|user_start|>{user}<|user_end|>
                  <|assistant_start|>{assistant}<|assistant_end|>...
        """
        prompt = "<|bos|>"

        for i, msg in enumerate(messages):
            role = msg['role']
            content = msg['content']
            is_last = (i == len(messages) - 1)

            if role == 'system':
                prompt += f"<|system_start|>{content}<|system_end|>"
            elif role == 'user':
                prompt += f"<|user_start|>{content}<|user_end|>"
            elif role == 'assistant':
                prompt += f"<|assistant_start|>{content}"
                if not (is_last and continue_last):
                    prompt += "<|assistant_end|>"

        # Add generation prompt if last message was from user (and not continuing)
        if messages and messages[-1]['role'] in ('user', 'system') and not continue_last:
            prompt += "<|assistant_start|>"

        return prompt
    
    @staticmethod
    def _build_llama3_prompt(messages: List[Dict], continue_last: bool) -> str:
        """
        Build Llama-3 format prompt.
        
        Format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_message}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_message}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {assistant_message}<|eot_id|>  # Omit this for continuation
        """
        # In continuation mode, llama_cpp may add begin_of_text automatically
        # so we skip it to avoid duplicates
        if continue_last:
            prompt = ""
        else:
            prompt = "<|begin_of_text|>"
        
        
        for i, msg in enumerate(messages):
            role = msg['role']
            content = msg['content']
            
            # Add header
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n"
            prompt += content
            
            # Add end-of-turn marker, unless it's the last message and we're continuing
            is_last = (i == len(messages) - 1)
            if not (is_last and continue_last and role == 'assistant'):
                prompt += "<|eot_id|>"
        
        return prompt
    
    @staticmethod
    @staticmethod
    def _build_chatml_prompt(messages: List[Dict], continue_last: bool) -> str:
        """
        Build ChatML format prompt.
        
        Format:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>  # Omit this for continuation
        """
        prompt = ""
        
        for i, msg in enumerate(messages):
            role = msg['role']
            content = msg['content']
            
            # Add message with ChatML format
            prompt += f"<|im_start|>{role}\n"
            prompt += content
            
            # Add end marker, unless it's the last assistant message and we're continuing
            is_last = (i == len(messages) - 1)
            if not (is_last and continue_last and role == 'assistant'):
                prompt += "<|im_end|>\n"
            # For continuation, we leave it open without <|im_end|>
        
        return prompt

    @staticmethod
    def _build_llama2_prompt(messages: List[Dict], continue_last: bool) -> str:
        """
        Build Llama-2 format prompt.
        
        Format varies based on whether there's a system message:
        With system:
        <s>[INST] <<SYS>>
        {system_message}
        <</SYS>>
        
        {user_message} [/INST] {assistant_message}</s>
        
        Without system:
        <s>[INST] {user_message} [/INST] {assistant_message}</s>
        """
        prompt = ""
        system_message = None
        
        # Extract system message if present
        if messages and messages[0]['role'] == 'system':
            system_message = messages[0]['content']
            messages = messages[1:]  # Process remaining messages
        
        # Process conversation
        i = 0
        while i < len(messages):
            if messages[i]['role'] == 'user':
                # Start a new turn
                prompt += "<s>[INST] "
                
                # Include system message in first user turn
                if system_message and i == 0:
                    prompt += f"<<SYS>>\n{system_message}\n<</SYS>>\n\n"
                
                prompt += messages[i]['content'] + " [/INST]"
                
                # Check if there's an assistant response
                if i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                    assistant_content = messages[i + 1]['content']
                    prompt += " " + assistant_content
                    
                    # Add closing tag unless we're continuing the last assistant message
                    is_last_assistant = (i + 1 == len(messages) - 1)
                    if not (is_last_assistant and continue_last):
                        prompt += "</s>"
                    
                    i += 2  # Skip both user and assistant
                else:
                    # No assistant response yet
                    prompt += " "  # Ready for assistant response
                    i += 1
            elif messages[i]['role'] == 'assistant':
                # Standalone assistant message (shouldn't normally happen in Llama-2)
                # But handle it gracefully
                prompt += " " + messages[i]['content']
                
                is_last = (i == len(messages) - 1)
                if not (is_last and continue_last):
                    prompt += "</s>"
                i += 1
            else:
                # Skip other roles (like additional system messages)
                i += 1
        
        return prompt
    
    @staticmethod
    def _build_basic_prompt(messages: List[Dict], continue_last: bool) -> str:
        """
        Build a basic prompt format as fallback.
        Simply concatenates messages with role labels.
        """
        prompt = ""
        
        for i, msg in enumerate(messages):
            role = msg['role'].capitalize()
            content = msg['content']
            
            # Add role prefix if not continuing the last message
            is_last = (i == len(messages) - 1)
            if not (is_last and continue_last and msg['role'] == 'assistant'):
                if prompt:
                    prompt += "\n"
                prompt += f"{role}: {content}"
            else:
                # For continuation, just add the content without role prefix
                if prompt:
                    prompt += "\n"
                prompt += f"{role}: {content}"
        
        return prompt
    
    @staticmethod
    def is_supported(format_name: str) -> bool:
        """Check if a format is explicitly supported."""
        return format_name in ['llama-2', 'llama-3', 'chatml', 'mara']


prompt_session = PromptSession()

# ============================================================================
# YAML Converter for Universal Format
# ============================================================================

class ConversationConverter:
    """Converts universal YAML format to model-specific formats"""
    
    def __init__(self, user_names: List[str]):
        self.user_names = user_names
        self.primary_user = user_names[0] if user_names else "User"
        self.ai_name = "Assistant"
    
    def load_yaml(self, path: str) -> Dict:
        """Load YAML file and return parsed data"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data
    
    def parse_universal_yaml(self, path: str) -> Tuple[str, str, int, List[Dict]]:
        """Parse universal format YAML"""
        data = self.load_yaml(path)
        
        # For compatibility this can be either "char" or "ai_name"
        ai_name = data.get("char", data.get("ai_name", "Assistant"))
        prompt = data.get("prompt", "")
        seed = data.get("seed", -1)
        conversations = data.get("conversations", [])
        
        return ai_name, prompt, seed, conversations
    
    def var_replace(self, text: str) -> str:
        """Replace template variables in text"""
        text = text.replace("{{char}}", self.ai_name)
        text = text.replace("{{ai_name}}", self.ai_name)
        text = text.replace("{{user}}", self.primary_user)
        text = text.replace("{{user_name}}", self.primary_user)
        text = text.replace("{{nl}}", "\n")
        return text
    
    def to_chat_messages(self, yaml_path: str) -> Tuple[List[Dict], str, int]:
        """Convert YAML to chat completion format"""
        ai_name, prompt, seed, conversations = self.parse_universal_yaml(yaml_path)
        self.ai_name = ai_name

        messages = []

        if prompt and not conversations:
            # No separate conversations list - check for inline conversations in prompt
            # Lines starting with {{char}}: or {{user}}: are conversation turns
            system_lines = []
            char_prefix = "{{char}}:"
            user_prefix = "{{user}}:"
            # Also check for already-substituted names
            ai_prefix = f"{ai_name}:"
            user_name_prefix = f"{self.primary_user}:"

            for line in prompt.split("\n"):
                stripped = line.strip()
                if stripped.startswith(char_prefix):
                    content = stripped[len(char_prefix):].strip()
                    content = self.var_replace(content)
                    conversations.append({"role": "{{char}}", "content": content})
                elif stripped.startswith(user_prefix):
                    content = stripped[len(user_prefix):].strip()
                    content = self.var_replace(content)
                    conversations.append({"role": "{{user}}", "content": content})
                elif stripped.startswith(ai_prefix) and not conversations:
                    # After var_replace, ai_name might already be substituted
                    content = stripped[len(ai_prefix):].strip()
                    conversations.append({"role": "{{char}}", "content": content})
                elif stripped.startswith(user_name_prefix) and not conversations:
                    content = stripped[len(user_name_prefix):].strip()
                    conversations.append({"role": "{{user}}", "content": content})
                else:
                    if not conversations:
                        system_lines.append(line)
                    # After conversations start, non-prefixed lines are ignored

            system_prompt = self.var_replace("\n".join(system_lines))
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
        elif prompt:
            # Has separate conversations list - prompt is just the system message
            prompt = self.var_replace(prompt)
            messages.append({"role": "system", "content": prompt})

        # Convert conversations with variable substitution
        for conv in conversations:
            role = conv["role"]
            content = conv["content"]

            # Replace template variables in content
            content = self.var_replace(content)

            # Replace template variables with standard roles
            if "{{char}}" in role:
                role = "assistant"
            elif "{{user_name}}" in role or "{{user}}" in role:
                role = "user"
            # else keep the role as-is

            messages.append({"role": role, "content": content})

        return messages, ai_name, seed
    
    def to_raw_text(self, yaml_path: str) -> Tuple[str, str, int]:
        """Convert YAML to raw text format"""
        ai_name, prompt, seed, conversations = self.parse_universal_yaml(yaml_path)
        self.ai_name = ai_name
        
        # Replace template variables in the prompt
        prompt = self.var_replace(prompt)

        text_parts = []
        if prompt:
            text_parts.append(prompt)
            if conversations:
                text_parts.append("")
        
        for conv in conversations:
            role = conv["role"]
            content = conv["content"]
            
            # Replace template variables with actual names
            if "{{char}}" in role:
                role = ai_name
            elif "{{user_name}}" in role or "{{user}}" in role:
                role = self.primary_user

            # Replace template variables in content
            content = self.var_replace(content)

            text_parts.append(f"{role}: {content}")
        
        full_text = "\n".join(text_parts)
        # Ensure text ends with newline so subsequent messages are properly separated
        if full_text and not full_text.endswith("\n"):
            full_text += "\n"
        return full_text, ai_name, seed

# ============================================================================
# Model Wrappers
# ============================================================================

class ModelWrapper:
    """Base class for model wrappers"""
    def generate(self, prompt: str, max_new_tokens: int, temperature: float,
                top_p: float, stop_sequences: List[str], stream_output: bool = True) -> str:
        raise NotImplementedError
    
    def get_context_length(self) -> int:
        raise NotImplementedError
    
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

class CustomModelWrapper(ModelWrapper):
    """Wrapper for custom transformer checkpoints"""
    def __init__(self, model, tokenizer, context_len):
        load_torch_if_needed()
        self.model = model
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.device = next(model.parameters()).device

    def generate(self, prompt: str, max_new_tokens: int, temperature: float,
                top_p: float, stop_sequences: List[str], stream_output: bool = True,
                return_stop_info: bool = False,
                pretty_print: bool = False, role_names: dict = None):
        # Import neo_common here if needed
        if 'nc' not in globals():
            # Primary common path (FSDP2)
            common_path = '../common_fsdp2'
            if common_path not in sys.path:
                sys.path.insert(0, common_path)
            # Also add saved_code for FSDP1 checkpoint support
            saved_code_path = '../saved_code'
            if saved_code_path not in sys.path:
                sys.path.insert(0, saved_code_path)
            import neo_common as nc

        return nc.stream_generate_kv(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens,
            self.context_len,
            temperature,
            top_p,
            display=stream_output,
            stop_sequences=stop_sequences,
            print_prompt=False,
            return_stop_info=return_stop_info,
            pretty_print=pretty_print,
            role_names=role_names
        )

    def get_context_length(self) -> int:
        return self.context_len

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, bos=True, eos=False)

class GGUFModelWrapper(ModelWrapper):
    """Wrapper for GGUF models via llama_cpp"""
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = None, 
                 chat_format: str = None, tensor_split: Optional[List[float]] = None,
                 use_chat_completion: bool = False):
        if not HAS_LLAMA_CPP:
            raise ImportError("llama_cpp is required for GGUF models")
        
        log(f"Loading GGUF model: {model_path}")
        self.use_chat_completion = use_chat_completion
        # Messages now passed directly from session
        
        if tensor_split == 'auto':
            tensor_split = self._auto_detect_gpu_split()
        
        if n_ctx is None:
            n_ctx = self._detect_context_length(model_path)
        
        model_params = {
            "model_path": model_path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "verbose": False,
            "n_threads": 8,
            "n_batch": 512,
        }
        
        # Only add chat_format if specified and using chat completion
        if chat_format and use_chat_completion:
            model_params["chat_format"] = chat_format
        
        if tensor_split:
            model_params["tensor_split"] = tensor_split
            log(f"Using {len(tensor_split)} GPU(s) with split: {[f'{x:.1%}' for x in tensor_split]}")
        
        self.model = llama_cpp.Llama(**model_params)
        self.context_len = n_ctx
        self.chat_format = chat_format
        log(f"Context length: {self.context_len} tokens")

        if use_chat_completion and chat_format:
            log(f"Chat completion mode with format: {chat_format}")
        else:
            log("Raw completion mode")
    
    
    
    def generate(self, prompt: str = None, max_new_tokens: int = 128,
                temperature: float = 0.7, top_p: float = 0.95,
                stop_sequences: List[str] = None, stream_output: bool = True,
                messages: List[Dict] = None, continue_mode: bool = False) -> str:
        """Generate based on mode with streaming support"""
        if self.use_chat_completion:
            # Check if we're in continuation mode
            if continue_mode and messages:
                # Build prompt manually for continuation
                builder = ChatTemplateBuilder()
                prompt = builder.build_prompt(messages, self.chat_format, continue_last=True)
                
                # Use raw completion for continuation
                output = self.model.create_completion(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_sequences if stop_sequences else None,
                    stream=stream_output
                )
                
                if stream_output:
                    # Stream tokens for continuation
                    response = ""
                    for chunk in output:
                        if 'choices' in chunk:
                            content = chunk['choices'][0].get('text', '')
                            if content:
                                print(content, end='', flush=True)
                                response += content
                    return response
                else:
                    return output['choices'][0]['text']
            else:
                # Normal chat completion mode
                output = self.model.create_chat_completion(
                    messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_sequences if stop_sequences else None,
                    stream=stream_output  # Enable streaming
                )
                
                if stream_output:
                    # Stream and print tokens as they arrive
                    response = ""
                    for chunk in output:
                        delta = chunk['choices'][0]['delta']
                        if 'content' in delta:
                            content = delta['content']
                            print(content, end='', flush=True)
                            response += content
                    return response
                else:
                    # Non-streaming mode
                    return output['choices'][0]['message']['content']
        else:
            # Raw completion mode - keep as is for now
            # Could add streaming here too if desired
            completion = self.model.create_completion(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences if stop_sequences else None,
                stream=False,
                echo=False
            )
            return completion['choices'][0]['text']

    
    def _auto_detect_gpu_split(self) -> Optional[List[float]]:
        """Auto-detect GPUs and create even split"""
        try:
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
            except:
                load_torch_if_needed()
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                else:
                    return None
            
            if gpu_count > 1:
                gpu0_reduction = 0.05
                gpu0_share = (1.0 / gpu_count) * (1 - gpu0_reduction)
                remaining = 1.0 - gpu0_share
                other_gpu_share = remaining / (gpu_count - 1)
                tensor_split = [gpu0_share] + [other_gpu_share] * (gpu_count - 1)
                log(f"Auto-detected {gpu_count} GPUs - using adjusted split")
                return tensor_split
        except:
            pass
        return None
    
    def _detect_context_length(self, model_path: str) -> int:
        """Try to detect context length from GGUF model metadata"""
        try:
            temp_model = llama_cpp.Llama(
                model_path=model_path,
                n_ctx=512,
                n_gpu_layers=0,
                verbose=False
            )
            
            possible_keys = [
                'llama.context_length',
                'context_length', 
                'max_position_embeddings',
                'n_ctx',
                'max_seq_len'
            ]
            
            if hasattr(temp_model, 'metadata'):
                metadata = temp_model.metadata
                for key in possible_keys:
                    if key in metadata:
                        detected = int(metadata[key])
                        log(f"Auto-detected context length from model: {detected} tokens")
                        del temp_model
                        return detected
            
            if hasattr(temp_model, 'n_ctx_train'):
                detected = temp_model.n_ctx_train
                log(f"Auto-detected training context length: {detected} tokens")
                del temp_model
                return detected
            
            del temp_model
        except Exception as e:
            log(f"Could not auto-detect context length: {e}")

        log("Using default context length: 4096 tokens (use --context_len to override)")
        return 4096
    
    def get_context_length(self) -> int:
        return self.context_len
    
    def encode(self, text: str) -> List[int]:
        return self.model.tokenize(text.encode('utf-8'))

# ============================================================================
# Chat Mode Handling
# ============================================================================

class ChatSession:
    """Manages a chat session with either format"""
    def __init__(self, model_wrapper: ModelWrapper, converter: ConversationConverter,
                 config: dict, use_chat_mode: bool = False, chat_format: str = None):
        self.model_wrapper = model_wrapper
        self.converter = converter
        self.config = config
        self.use_chat_mode = use_chat_mode
        self.chat_format = chat_format  # 'mara', 'llama-3', 'chatml', etc.
        self.messages = []  # For chat mode
        self.raw_conversation = []  # For raw mode
        self.ai_name = "Assistant"
        self.seed = -1
        self.user_name = converter.primary_user if converter else "User"
    
    def load_prompt(self, yaml_path: str):
        """Load initial prompt from YAML"""
        if self.use_chat_mode:
            messages, ai_name, seed = self.converter.to_chat_messages(yaml_path)
            self.messages = messages
            self.ai_name = ai_name
            self.seed = seed
            # Format messages with actual names instead of role names
            def format_msg(m):
                role = m['role']
                if role == 'assistant':
                    return f"{self.ai_name}: {m['content']}"
                elif role == 'user':
                    return f"{self.user_name}: {m['content']}"
                else:
                    return f"{role}: {m['content']}"
            return "\n".join([format_msg(m) for m in messages])
        else:
            text, ai_name, seed = self.converter.to_raw_text(yaml_path)
            self.raw_conversation = [text]
            self.ai_name = ai_name
            self.seed = seed
            return text
        
    
    def add_user_message(self, content: str):
        """Add a user message"""
        if self.use_chat_mode:
            self.messages.append({"role": "user", "content": content})
            # No longer need to add to wrapper - it gets messages when generating
        else:
            self.raw_conversation.append(f"{self.converter.primary_user}: {content}")
    
    def get_last_message(self) -> str:
        """Get the last message in the conversation"""
        if self.use_chat_mode:
            if self.messages:
                return self.messages[-1]['content']
            return ""
        else:
            if self.raw_conversation:
                return self.raw_conversation[-1]
            return ""
    
    def replace_last_message(self, new_content: str):
        """Replace the last message with new content"""
        if self.use_chat_mode:
            if self.messages:
                self.messages[-1]['content'] = new_content
                # No longer need to reset messages in wrapper - it gets them when generating
        else:
            if self.raw_conversation:
                self.raw_conversation[-1] = new_content
    
    def add_raw_text(self, text: str):
        """Add raw text to conversation (for // command and emotes)"""
        if self.use_chat_mode:
            # In chat mode, append to last message or create new one
            if self.messages and self.messages[-1]['role'] == 'assistant':
                self.messages[-1]['content'] += "\n" + text
            else:
                self.messages.append({"role": "assistant", "content": text})
        else:
            self.raw_conversation.append(text)
    
    def append_newline_to_prior(self):
        """Add newline to prior message if it doesn't have one"""
        if self.use_chat_mode:
            if self.messages and not self.messages[-1]['content'].endswith("\n"):
                self.messages[-1]['content'] += "\n"
        else:
            if self.raw_conversation and not self.raw_conversation[-1].endswith("\n"):
                self.raw_conversation[-1] += "\n"
    
    def process_user_input(self, text: str) -> str:
        """Process user input - add quotes intelligently"""
        # Add quotes if not already present and not an action
        if '"' in text:
            return text
        
        if '*' not in text:
            return f'"{text}"'
        
        # Split by asterisk pairs
        parts = re.split(r'(\*[^*]+\*)', text)
        
        result = []
        for part in parts:
            if part.startswith('*') and part.endswith('*'):
                result.append(part)
            elif part.strip():
                result.append(f'"{part.strip()}"')
        
        return ' '.join(result)
    
    def continue_response(self) -> str:
        """Continue the last AI response (for / command)"""
        if self.use_chat_mode:
            if self.chat_format == 'mara' and isinstance(self.model_wrapper, CustomModelWrapper):
                # Mara format continuation - build prompt with continue_last=True
                prompt = ChatTemplateBuilder.build_prompt(self.messages, 'mara', continue_last=True)
                role_names = {"assistant": self.ai_name, "user": self.user_name}

                response = self.model_wrapper.generate(
                    prompt=prompt,
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    stop_sequences=self._get_stop_sequences(),
                    stream_output=True,
                    return_stop_info=False,
                    pretty_print=True,
                    role_names=role_names
                )
                # Clean up response
                for tag in ["<|assistant_end|>", "<|user_start|>", "<|user_end|>", "<|system_start|>", "<|system_end|>"]:
                    if tag in response:
                        response = response.split(tag)[0]
                # Append to last assistant message
                if self.messages and self.messages[-1]['role'] == 'assistant':
                    self.messages[-1]['content'] += response
                else:
                    self.messages.append({"role": "assistant", "content": response})
                return response
            else:
                # GGUF chat completion continuation
                response = self.model_wrapper.generate(
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    stop_sequences=self._get_stop_sequences()
                )
                # Append to last assistant message
                if self.messages and self.messages[-1]['role'] == 'assistant':
                    self.messages[-1]['content'] += response
                else:
                    self.messages.append({"role": "assistant", "content": response})
                return response
        else:
            # Continue from current conversation (Custom model)
            prompt = "\n".join(self.raw_conversation)
            response = self.model_wrapper.generate(
                prompt=prompt,
                max_new_tokens=self.config["max_new_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                stop_sequences=self._get_stop_sequences(),
                stream_output=True,
                return_stop_info=False
            )
            self.raw_conversation[-1] += response  # Append to last entry
            return response
    
    def generate_response(self) -> str:
        """Generate AI response"""
        if self.use_chat_mode:
            if self.chat_format == 'mara' and isinstance(self.model_wrapper, CustomModelWrapper):
                # Mara format for custom models - build prompt with special tokens
                prompt = ChatTemplateBuilder.build_prompt(self.messages, 'mara', continue_last=False)
                role_names = {"assistant": self.ai_name, "user": self.user_name}

                # Print AI name prefix (the <|assistant_start|> is in the prompt, not generated)
                print(f"{self.ai_name}: ", end="", flush=True)

                response = self.model_wrapper.generate(
                    prompt=prompt,
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    stop_sequences=self._get_stop_sequences(),
                    stream_output=True,
                    return_stop_info=False,
                    pretty_print=True,
                    role_names=role_names
                )
                # Clean up response - remove any trailing special tokens
                for tag in ["<|assistant_end|>", "<|user_start|>", "<|user_end|>", "<|system_start|>", "<|system_end|>"]:
                    if tag in response:
                        response = response.split(tag)[0]
                self.messages.append({"role": "assistant", "content": response})
                return response
            else:
                # GGUF chat completion mode
                response = self.model_wrapper.generate(
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    stop_sequences=self._get_stop_sequences()
                )
                self.messages.append({"role": "assistant", "content": response})
                return response
        else:
            # Raw completion mode
            prompt = "\n".join(self.raw_conversation)
            if self.config.get("force_response", False):
                if not prompt.endswith(f"{self.ai_name}:") and not prompt.endswith(f"{self.ai_name}: ") and not prompt.endswith(f"{self.ai_name}: \""):
                    prompt += f"\n{self.ai_name}:"

            response = self.model_wrapper.generate(
                prompt=prompt,
                max_new_tokens=self.config["max_new_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                stop_sequences=self._get_stop_sequences(),
                stream_output=True,
                return_stop_info=False
            )

            # Clean up response like original chat_neo
            while response.endswith("\n\n"):
                response = response[:-1]
            while response.startswith("\n"):
                response = response[1:]

            if self.config.get("force_response", False):
                response = f"{self.ai_name}: {response}"

            self.raw_conversation.append(response)
            return response
    
    
    def _check_and_trim_stops(self, text: str) -> Tuple[bool, str]:
        """Check for stop sequences and trim text if found"""
        stops = []
        
        # Build stop sequences for user names
        for name in self.converter.user_names:
            stops.extend([
                f"\n{name}:",
                f"{name}:",
                f"{name} :",
                f"{name} says",
                f"{name} said",
                f"{name} smiles and says",
                f"{name} frowns and mumbles",
                f"{name} grins and says",
                f"{name} laughs and says",
                f"{name} sighs and says",
                f"{name} whispers",
                f"{name} exclaims",
                f"{name} shouts",
                f"{name} yells"
            ])
        
        # Check each stop sequence
        for stop in stops:
            if stop in text:
                # Find the LAST occurrence and trim there
                index = text.rfind(stop)
                text = text[:index]
                return True, text
        
        return False, text
    
    def _get_stop_sequences(self) -> List[str]:
        """Get stop sequences based on mode and user names"""
        stops = []
        # For mara format, use special token stop sequences
        if self.chat_format == 'mara':
            stops.append("<|assistant_end|>")
            stops.append("<|user_start|>")  # Also stop if model tries to start a new user turn
            stops.append("<|system_start|>")  # Also stop if model tries to start a system block
        else:
            # Raw mode - stop on user name patterns
            for name in self.converter.user_names:
                stops.extend([
                    f"\n{name}:", f"{name}:", f"{name} says", f"{name} said"
                ])
        return stops
    
    def _check_stops(self, text: str) -> bool:
        """Check if we should stop generation"""
        for stop in self._get_stop_sequences():
            if stop in text:
                return True
        return False
    
    def get_full_conversation(self) -> str:
        """Get full conversation as text"""
        if self.use_chat_mode:
            parts = []
            for msg in self.messages:
                parts.append(f"{msg['role']}: {msg['content']}")
            return "".join(parts)
        else:
            return "".join(self.raw_conversation)
    
    def trim_context(self):
        """Trim conversation to fit context"""
        max_tokens = self.model_wrapper.get_context_length() - self.config["max_new_tokens"]
        
        if self.use_chat_mode:
            # Keep system message, trim from oldest user/assistant pairs
            def _chat_token_count():
                if self.chat_format and ChatTemplateBuilder.is_supported(self.chat_format):
                    prompt = ChatTemplateBuilder.build_prompt(self.messages, self.chat_format)
                    return len(self.model_wrapper.encode(prompt))
                return len(self.model_wrapper.encode(str(self.messages)))
            while _chat_token_count() > max_tokens and len(self.messages) > 1:
                # Find first non-system message
                for i, msg in enumerate(self.messages):
                    if msg['role'] != 'system':
                        self.messages.pop(i)
                        break
        else:
            # Trim from beginning, keeping some context
            while len(self.model_wrapper.encode("\n".join(self.raw_conversation))) > max_tokens and len(self.raw_conversation) > 1:
                self.raw_conversation.pop(1)  # Keep initial prompt

# ============================================================================
# Utilities
# ============================================================================

def resolve_model_path(model_path: str) -> str:
    """
    Resolve model path to a specific model file.

    If model_path points to a directory, finds the most recent .pt file.
    If model_path points to a file, returns it as-is.

    Args:
        model_path: Path to model file or directory

    Returns:
        str: Path to the resolved model file

    Raises:
        FileNotFoundError: If no valid model file is found
    """
    # If it's a file, return it directly
    if os.path.isfile(model_path):
        return model_path

    # If it's a directory, find the most recent model checkpoint .pt file
    if os.path.isdir(model_path):
        pt_files = []
        for file in os.listdir(model_path):
            # Only match model checkpoint files, not auxiliary files like
            # ep_experts_step_*.pt, moe_bias_step_*.pt, optim_*.pt, rng_*.pt, awd_*.pt
            if file.startswith("model_") and file.endswith('.pt'):
                full_path = os.path.join(model_path, file)
                step_match = re.search(r'_(\d+)\.pt', file)
                if step_match:
                    step_number = int(step_match.group(1))
                    pt_files.append((step_number, full_path))

        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in directory: {model_path}")

        # Sort by step number and get the highest one
        pt_files.sort(reverse=True)
        selected_step, selected_path = pt_files[0]

        log(f"Auto-selected checkpoint: {os.path.basename(selected_path)} (step {selected_step})")
        return selected_path

    # Path doesn't exist
    raise FileNotFoundError(f"Model path not found: {model_path}")

def detect_model_type(model_path: str) -> str:
    """Detect whether the model is a custom checkpoint or GGUF"""
    if model_path.endswith('.gguf'):
        return 'gguf'
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        return 'custom'
    else:
        if os.path.exists(model_path):
            return 'custom'
        raise ValueError(f"Cannot determine model type for: {model_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified chat interface")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint (.pt), GGUF file (.gguf), or directory (auto-selects most recent .pt file by step number)")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.98, help="Top-p sampling")
    parser.add_argument("--full", action="store_true", help="Use full precision (fp32) instead of half precision.")
    parser.add_argument("--force", action="store_true", help="Force response mode")
    
    
    # Custom model arguments
    parser.add_argument("--tok_kind", type=str, default=None, help="Tokenizer kind (auto-detected from checkpoint if not specified)")
    parser.add_argument("--tok_path", type=str, default=None, help="Path to tokenizer files (auto-detected from checkpoint if not specified)")
    parser.add_argument("--special_tokens", type=str, default=None, help="Path to special tokens JSON file (auto-detected from checkpoint if not specified)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index")
    parser.add_argument("--max_memory", type=str, default=None, help="Max memory per GPU")
    parser.add_argument("--shard_strategy", type=str, default="balanced", choices=["auto", "balanced", "none"])
    parser.add_argument("--use_keel", action="store_true",
                       help="Enable KEEL (Highway-style Post-LN) - use when checkpoint was trained with use_keel but config doesn't include it")

    # GGUF model arguments
    parser.add_argument("--n_gpu_layers", type=int, default=-1, help="GPU layers for GGUF")
    parser.add_argument("--chat_format", type=str, default=None, 
                       help="Chat format (llama-3, chatml, etc.) - enables chat completion mode")
    parser.add_argument("--tensor_split", type=str, default="auto", help="Tensor split for GGUF")
    
    # Chat arguments
    parser.add_argument("--user", type=str, default="User", help="User name(s)")
    parser.add_argument("--context_len", type=int, default=4096, help="Context length")
    parser.add_argument("--gen_size", type=int, default=128, help="Max new tokens")

    args = parser.parse_args()

    if args.tensor_split and args.tensor_split != 'auto':
        try:
            # Parse comma-delimited values
            splits = [float(x) for x in args.tensor_split.split(',')]
            
            # Convert percentages to proportions if needed
            if sum(splits) > 1.1:  # Likely percentages
                splits = [x/100.0 for x in splits]
            
            # Normalize to ensure they sum to 1.0
            total = sum(splits)
            args.tensor_split = [x/total for x in splits]
        except:
            args.tensor_split = 'auto'  # Fallback to auto
    
    return args

def load_model(config: dict, args: argparse.Namespace) -> ModelWrapper:
    """Load model based on type"""
    model_type = detect_model_type(args.model_path)
    
    if model_type == 'gguf':
        if not HAS_LLAMA_CPP:
            raise ImportError("llama_cpp is required for GGUF models")
        
        log("Loading GGUF model (PyTorch not loaded - saving VRAM)")
        
        n_gpu_layers = args.n_gpu_layers
        chat_format = args.chat_format
        use_chat = chat_format is not None  # Use chat mode if chat_format is specified
        tensor_split = args.tensor_split
        
        n_ctx = config["context_len"] if config["context_len"] != 4096 else None
        
        wrapper = GGUFModelWrapper(
            model_path=args.model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            chat_format=chat_format,
            tensor_split=tensor_split,
            use_chat_completion=use_chat
        )
        
        config["context_len"] = wrapper.get_context_length()
        
    else:  # custom checkpoint
        log("Loading custom checkpoint (importing PyTorch...)")
        load_torch_if_needed()
        
        # Also add saved_code for FSDP1 checkpoint support
        saved_code_path = '../saved_code'
        if saved_code_path not in sys.path:
            sys.path.insert(0, saved_code_path)
        # Note: Transformer/ModelArgs imported dynamically in neo_common based on checkpoint version
        from tokenizer_abstraction import get_tokenizer
        import neo_common as nc
        
        device = nc.detect_device(preferred_gpu=args.gpu)
        config["device"] = device
        
        model, enc, model_cfg = nc.load_model_and_tokenizer(
            config["checkpoint"],
            device=device,
            half_precision=config["half_precision"],
            tok_kind=args.tok_kind,
            tok_path=args.tok_path,
            special_tokens=args.special_tokens,
            shard_strategy=args.shard_strategy,
            preferred_gpu=args.gpu,
            max_memory_per_gpu=args.max_memory,
            use_keel=args.use_keel or None
        )
        
        if hasattr(model_cfg, "max_seq_len"):
            config["context_len"] = model_cfg.max_seq_len
            log(f"Context length from model config: {config['context_len']} tokens")
        
        wrapper = CustomModelWrapper(model, enc, config["context_len"])
    
    return wrapper

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    config = {
        "checkpoint": None,
        "max_new_tokens": 128,
        "temperature": 0.65,
        "top_p": 0.99,
        "half_precision": True,
        "seed": time.time(),
        "device": 'cpu',
        "prompt": 'ev4.yaml',
        "prompt_dir": "../xn/mpd/",
        "force_response": False,
        "context_len": 4096,
        "debug": False,
        "compact": True,
    }
    
    args = parse_args()

    # Initialize logger
    if HAS_LOGGER:
        logger._instance.set_logdir("./logs")
        logger._instance.set_default_logfile("chat_log.txt")
        logger._instance.set_rank(0)

    if not HAS_LLAMA_CPP:
        log("Warning: llama_cpp not installed. GGUF support disabled.")

    # Resolve model path (handles both files and directories)
    try:
        resolved_path = resolve_model_path(args.model_path)
        args.model_path = resolved_path  # Update args with resolved path
    except FileNotFoundError as e:
        log(f"Error: {e}")
        exit(1)

    config["checkpoint"] = args.model_path
    config["temperature"] = args.temp
    config["top_p"] = args.top_p
    config["half_precision"] = not args.full
    config["force_response"] = args.force
    config["context_len"] = args.context_len
    config["max_new_tokens"] = args.gen_size

    usr_names = [name.strip() for name in args.user.split(",") if name.strip()]

    # Load model
    model_wrapper = load_model(config, args)
    
    # Create converter
    converter = ConversationConverter(usr_names)
    
    # Determine if we're using chat mode
    # Chat mode is enabled for:
    # 1. GGUF models with any chat_format specified
    # 2. Custom models with chat_format='mara'
    if args.chat_format == 'mara':
        use_chat_mode = True
        use_mara_format = True
    elif args.chat_format is not None and isinstance(model_wrapper, GGUFModelWrapper):
        use_chat_mode = True
        use_mara_format = False
    else:
        use_chat_mode = False
        use_mara_format = False

    log(f"Using context length: {model_wrapper.get_context_length()}")
    if use_mara_format:
        log(f"Mode: Chat Completion (mara format with special tokens)")
    elif use_chat_mode:
        log(f"Mode: Chat Completion ({args.chat_format})")
    else:
        log(f"Mode: Raw Completion")
    log("/help for commands")

    # Drain logger queue so boot messages don't overlap with input prompts
    if HAS_LOGGER:
        logger.flush()

    while True:
        # Ask for prompt file
        config["prompt"] = input(f"Enter prompt file [{config['prompt']}]: ") or config["prompt"]
        
        try:
            prompt_path = os.path.join(config["prompt_dir"], config["prompt"])
            
            # Create chat session
            session = ChatSession(model_wrapper, converter, config, use_chat_mode, chat_format=args.chat_format)
            initial_text = session.load_prompt(prompt_path)

            log(f"Loaded: {session.ai_name} (seed: {session.seed})")
            if HAS_LOGGER:
                logger.flush()  # Flush after loading prompt to keep things tidy
                
            print(f"\n{initial_text}")
            
            # Set seed (numpy for general use, torch for model sampling via torch.multinomial)
            if session.seed == -1:
                session.seed = np.random.randint(0, 2**31 - 1)
            np.random.seed(session.seed)
            if TORCH_LOADED:
                torch.manual_seed(session.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(session.seed)
        except Exception as e:
            log(f"Error loading prompt: {e}")
            continue
                
        # Chat loop
        done = False
        concat = False
        # skip_ai_turn is set to true if the last message in the loaded conversation is from the assistant
        # First handle for chat mode
        if use_chat_mode:
            skip_ai_turn = len(session.messages) > 0 and session.messages[-1]['role'] == 'assistant'
        else:
            # skip_ai_turn = len(session.raw_conversation) > 0 and session.raw_conversation[-1].startswith(f"{session.ai_name}:")
            # The correct way to do this is to scan from the end of the file for the first occurrence of either user: or ai_name:
            skip_ai_turn = False
            # raw conversation is a single string, need to split by lines and scan backwards
            last_message = session.get_last_message()
            # Split on newlines and scan backwards
            lines = last_message.strip().split("\n")
            for line in reversed(lines):
                #print(f"Debug: Scanning line: {line}")
                if line.startswith(f"{session.ai_name}:"):
                    skip_ai_turn = True
                    break
                elif any(line.startswith(f"{name}:") for name in usr_names):
                    break  # Found a user message first, so no skip

        while not done:
            if not skip_ai_turn:
                session.trim_context()  # Using session's method instead of conversation.trim_messages
                
                # Different generation logic for chat mode vs raw mode
                if session.use_chat_mode:
                    # Chat completion mode - use session methods which handle mara/GGUF correctly
                    if concat:
                        # For "/" command - continue generation
                        answer = session.continue_response()
                        concat = False
                    else:
                        # Normal generation
                        answer = session.generate_response()
                else:
                    # Raw completion mode - use chunk-based generation as before
                    prompt = session.get_full_conversation()  # Get the current full prompt
                    force = config["force_response"] and not concat  # force_response mode is only active if we are not concatenating

                    if force:
                        # Check to see if the prompt already ends with "{ai_name}:"
                        if not prompt.endswith(f"{session.ai_name}:") and not prompt.endswith(f"{session.ai_name}: ") and not prompt.endswith(f"{session.ai_name}: \""):
                            saved_prompt = prompt
                            prompt += f"\n{session.ai_name}:"
                            # Print the AI name prefix before streaming starts
                            print(f"{session.ai_name}: ", end="", flush=True)

                    # Generate the raw response
                    if config.get("debug", False):
                        answer, stop_info = session.model_wrapper.generate(
                            prompt=prompt,
                            max_new_tokens=session.config["max_new_tokens"],
                            temperature=session.config["temperature"],
                            top_p=session.config["top_p"],
                            stop_sequences=session._get_stop_sequences(),
                            stream_output=True,
                            return_stop_info=True
                        )
                        print(f"\n[DEBUG] Stop reason: {stop_info['reason']}, detail: {stop_info['detail']}, tokens: {stop_info['tokens_generated']}")
                    else:
                        answer = session.model_wrapper.generate(
                            prompt=prompt,
                            max_new_tokens=session.config["max_new_tokens"],
                            temperature=session.config["temperature"],
                            top_p=session.config["top_p"],
                            stop_sequences=session._get_stop_sequences(),
                            stream_output=True,
                            return_stop_info=False
                        )

                    # Handle adding to conversation for raw mode
                    # Strip leading/trailing whitespace from response to keep conversation compact
                    if config.get("compact", True):
                        answer = answer.strip()

                    if concat:
                        session.raw_conversation[-1] += answer  # Append to last message
                        concat = False
                    else:
                        if force:
                            # make sure that the message starts with f"{ai_name}:" if we are in force_response mode
                            answer = f"{session.ai_name}: {answer}"
                        session.raw_conversation.append(answer)
                
                # Clean up and print the answer (for both modes)
                # If the answer ends in multiple newlines, we need to trim it down to just one
                while answer.endswith("\n\n"):
                    answer = answer[:-1]
                # If the answer starts with newlines, remove them
                while answer.startswith("\n"):
                    answer = answer[1:]
                
                # Note: All generation paths now stream output directly, so no extra print needed
                # - GGUF chat mode: streams via llama_cpp
                # - Custom model chat mode (mara): streams via stream_generate_kv
                # - Raw mode: streams via stream_generate_kv
                
                if not answer.endswith("\n"):
                    print("")

            skip_ai_turn = False  # Reset skip_ai_turn after first iteration
            # Calculate prompt token size
            tok_size = len(model_wrapper.encode(session.get_full_conversation()))
            
            # Get the user response
            getting_input = True
            while getting_input:
                getting_input = False
                user_response = ""
                while user_response == "":
                    user_response = input(f"[{tok_size}] You: ")
                
                # Check for commands starting with /
                if user_response.startswith("/"):
                    if len(user_response) > 1:
                        if user_response.startswith("/rep"):
                            # Try multiline edit - should work in most terminals
                            try:
                                user_input = prompt_session.prompt(">", default=session.get_last_message(), multiline=True)
                                session.replace_last_message(user_input)
                            except Exception as e:
                                # Check for specific console errors that require fallback
                                error_name = type(e).__name__
                                
                                if "NoConsoleScreenBufferError" in error_name:
                                    # This specific error means we're in Git Bash or similar
                                    print("Note: Multiline edit not supported in this terminal (Git Bash/MinGW detected).")
                                    print("Try using PowerShell or Windows Terminal for full features.")
                                    print("\nCurrent message:", session.get_last_message()[:100] + "..." if len(session.get_last_message()) > 100 else session.get_last_message())
                                    user_input = input("Replace with: ")
                                    session.replace_last_message(user_input)
                                elif "EOFError" in error_name or "KeyboardInterrupt" in error_name:
                                    # User cancelled the edit (Ctrl+C or Ctrl+D)
                                    print("Edit cancelled.")
                                    user_input = None
                                else:
                                    # Unknown error - show it but try fallback
                                    print(f"Unexpected error with multiline editor: {error_name}: {e}")
                                    print("Falling back to single-line input.")
                                    print("\nCurrent message:", session.get_last_message()[:100] + "..." if len(session.get_last_message()) > 100 else session.get_last_message())
                                    user_input = input("Replace with: ")
                                    session.replace_last_message(user_input)
                            
                            getting_input = True  # We need to get another input
                        elif user_response.startswith("//"):
                            user_input = user_response[2:].rstrip()  # Remove the leading "//" and trailing spaces
                            if session.use_chat_mode:
                                # In chat mode, add as a system message for narrative/OOC content
                                session.messages.append({"role": "system", "content": user_input})
                                concat = False  # No concatenation needed in chat mode
                            else:
                                # In raw mode, keep the original behavior
                                session.raw_conversation.append(user_input)  # Add directly to conversation
                                concat = True  # Concat true means that the next AI response will be appended to this message
                        elif user_response.startswith("/cls"):
                            os.system('cls' if os.name == 'nt' else 'clear')
                            getting_input = True
                        elif user_response.startswith("/new"):
                            done = True
                            break
                        elif user_response.startswith("/exit"):
                            exit(0)
                        elif user_response.startswith("/temp"):
                            cur_temp = config["temperature"]
                            temp = input(f"Enter temperature[current: {cur_temp}]: ")
                            config["temperature"] = float(temp)
                            getting_input = True
                        elif user_response.startswith("/top"):
                            cur_top_p = config["top_p"]
                            top_p = input(f"Enter top_p[current: {cur_top_p}]: ")
                            config["top_p"] = float(top_p)
                            getting_input = True
                        elif user_response.startswith("/ls"):
                            # ls [-l] [pattern]; defaults to *.yaml when no pattern given
                            ls_args = user_response.split()[1:]
                            print(f"Files in {config['prompt_dir']}:")
                            print(_ls_listing(config['prompt_dir'], ls_args, default_suffix=".yaml"))
                            getting_input = True
                        elif user_response.startswith("/cd"):
                            val = input(f"Prompt directory [{config['prompt_dir']}]: ")
                            if val:
                                val = val.replace("\\", "/")
                                if not val.endswith("/"):
                                    val += "/"
                                if os.path.isdir(val):
                                    config["prompt_dir"] = val
                                    log(f"Prompt directory: {config['prompt_dir']}")
                                else:
                                    log(f"Directory not found: {val}")
                            else:
                                log(f"Prompt directory: {config['prompt_dir']}")
                            getting_input = True
                        elif user_response.startswith("/prompt"):
                            # Display the full prompt
                            print(f"\n{session.get_full_conversation()}")
                            getting_input = True
                        elif user_response.startswith("/name"):
                            # Change user names
                            new_names = input(f"Enter new user names (comma-separated) [{', '.join(usr_names)}]: ")
                            usr_names = [name.strip() for name in new_names.split(",")]
                            converter.user_names = usr_names  # Update converter too
                            converter.primary_user = usr_names[0]
                            getting_input = True
                        elif user_response.startswith("/force"):
                            # Toggle force response mode
                            config["force_response"] = not config["force_response"]
                            log(f"Force response mode: {config['force_response']}")
                            getting_input = True
                        elif user_response.startswith("/gen"):
                            # Generate tokens
                            log(f"Response tokens: {config['max_new_tokens']}")
                            new_response_tokens = input(f"Enter new response tokens: ")
                            config["max_new_tokens"] = int(new_response_tokens)
                            getting_input = True
                        elif user_response.startswith("/help"):
                            print("Commands:")
                            print(f"/help   - Show this help")
                            print(f"/       - Continue last message")
                            print(f"/rep    - Replace last message")
                            print(f"/exit   - Exit Program")
                            print(f"/new    - Start a new chat session")
                            print(f"/temp   - Set temperature [{config['temperature']}]")
                            print(f"/top    - Set top_p [{config['top_p']}]")
                            print(f"/name   - Change user names [{', '.join(usr_names)}]")
                            print(f"/ls     - List prompt files (/ls, /ls -l, /ls *.txt)")
                            print(f"/cd     - Change prompt directory [{config['prompt_dir']}]")
                            print(f"/prompt - Display the full prompt")
                            print(f"/force  - Toggle force response mode [{config['force_response']}]")
                            print(f"/size   - Response Tokens [{config['max_new_tokens']}]")
                            print(f"/cls    - Clear the screen")
                            print(f"/debug  - Toggle debug mode (show stop reasons) [{config['debug']}]")
                            print(f"/compact - Toggle compact mode (strip newlines) [{config['compact']}]")
                            print(f"/raw    - Toggle mara chat format / raw completion mode")
                            print(f"//<str> - Add narrative/OOC text (system message in chat mode)")
                            getting_input = True
                        elif user_response.startswith("/debug"):
                            config["debug"] = not config["debug"]
                            log(f"Debug mode: {config['debug']}")
                            getting_input = True
                        elif user_response.startswith("/compact"):
                            config["compact"] = not config["compact"]
                            log(f"Compact mode: {config['compact']}")
                            getting_input = True
                        elif user_response.startswith("/raw"):
                            # Toggle between mara chat format and raw completion mode
                            if session.chat_format == 'mara':
                                session.use_chat_mode = not session.use_chat_mode
                                if session.use_chat_mode:
                                    log("Switched to: Mara chat format")
                                else:
                                    log("Switched to: Raw completion mode")
                            else:
                                log("This command only works with mara chat format")
                            getting_input = True
                        else:
                            # Unknown command - show error and continue
                            log(f"Unknown command: {user_response}")
                            log("Use /help to see available commands")
                            getting_input = True
                    else:
                        # NOTE: IF this is JUST a / then we don't add any response - Just let the AI continue
                        concat = True
                else:
                    # Standard response with no "/"
                    if session.use_chat_mode:
                        # In chat mode, just add the message
                        session.add_user_message(user_response)
                    else:
                        # In raw mode, process and add with formatting
                        user_response = session.process_user_input(user_response)
                        session.append_newline_to_prior()
                        session.raw_conversation.append(usr_names[0] + ": " + user_response + "\n")
