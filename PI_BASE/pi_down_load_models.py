# -*-coding: utf-8 -*-
# LLM/embedding model download, load and run 

from modelscope import snapshot_download as ms_snapshot_download
from modelscope.hub.constants import MODEL_ID_SEPARATOR
from huggingface_hub import snapshot_download as hf_snapshot_download
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig

import torch
import os

# 导入PI项目日志模块
from pi_log import *

class PiLLM:
    def __init__(self):# 是否本地模型, 默认使用远程模型
        self.model_id = None        # 模型ID，用于标识模型
        self.tokenizer = None       # 本地llm的分词器
        self.llm_model = None       # 本地llm的模型
        self.embeddings = None      # embedding 模型

    def download_model(self, model_tag, model_dir, model_repo):
        """下载并加载模型的一站式方法"""
        model_path = download_model_to_local(model_tag, model_dir, model_repo)
        return model_path

    def load_llm_model(self, model_id, isLocal = False):
        if isLocal:
            self.tokenizer, self.llm_model = load_llm_from_local(model_id)
        else:
            self.llm_model = load_llm_from_ollama(model_id)

    def load_embeddings_model(self, model_id, isLocal = False):
        if isLocal:
            self.embeddings = load_embeddings_model_from_local(model_id)
        else:
            self.embeddings = load_embeddings_model_from_ollama(model_id)

# 获取logger实例（启用文件日志）
logger = get_logger("PI_LLM", log_to_file=True)

# 下载整个模型仓库到本地， 这里使用modelscope 的snapshot_download 方法
# model_repo, 如果使用huggingface_hub 下载，则值为1，modelscope 下载，设置MODEL_SOURCE 为2
def download_model_to_local(model_id, model_dir, model_repo):
    """下载模型到本地目录"""
    model_path = os.path.join(model_dir, model_id.replace(MODEL_ID_SEPARATOR, "/"))
    
    log_download_start(model_id, model_dir, logger)
    
    if os.path.exists(model_path):
        log_file_exists(model_path, logger)
        return model_path
    else:
        if model_repo == 2:
            try:
                local_dir = ms_snapshot_download(
                    model_id = model_id,
                    revision = "master",
                    cache_dir = model_dir,
                    max_workers = 4
                )
                log_download_complete(local_dir, logger)
                return local_dir
            except Exception as e:
                log_error(f"模型下载失败: {model_id} - 错误: {str(e)}", logger)
                raise
        elif model_repo == 1:
            try:
                local_dir = hf_snapshot_download(
                    repo_id = model_id,
                    revision = "main",
                    cache_dir = model_dir,
                    max_workers = 4
                )
                log_download_complete(local_dir, logger)
                return local_dir
            except Exception as e:
                log_error(f"模型下载失败: {model_id} - 错误: {str(e)}", logger)
                raise
        try:
            local_dir = ms_snapshot_download(
                model_id = model_id,
                revision = "master",
                cache_dir = model_dir,
                max_workers = 4
            )
            log_download_complete(local_dir, logger)
            return local_dir
        except Exception as e:
            log_error(f"模型下载失败: {model_id} - 错误: {str(e)}", logger)
            raise

# 加载本地的embedding 模型, 不通过ollama加载
def load_embeddings_model_from_local(model_id):
    """从本地加载嵌入模型"""
    log_info(f"📦 加载嵌入模型: {model_id}", logger)
    
    try:
        embeddings = SentenceTransformerEmbeddings(model_name = model_id)
        log_info(f"✅ 嵌入模型加载成功: {model_id}", logger)
        return embeddings
    except Exception as e:
        log_error(f"嵌入模型加载失败: {model_id} - 错误: {str(e)}", logger)
        raise

# 从本地的ollama 加载 embeddings 模型
def load_embeddings_model_from_ollama(model_id):
    """从本地加载嵌入模型"""
    log_info(f"📦 加载嵌入模型: {model_id}", logger)

    try:
        embeddings = OllamaEmbeddings(
            model = model_id,
            base_url = "http://localhost:11434"  # 默认可省略
        )
        log_info(f"✅ 嵌入模型加载成功: {model_id}", logger)
        return embeddings
    except Exception as e:
        log_error(f"嵌入模型加载失败: {model_id} - 错误: {str(e)}", logger)
        raise

# 从本地的ollama 加载 llm 模型
def load_llm_from_ollama(model_id):
    """从本地加载 llm 模型"""
    try:
        llm = OllamaLLM(
            model = model_id,
            base_url = "http://localhost:11434"  # 默认可省略
        )
        log_info(f"✅ llm 模型加载成功: {model_id}", logger)
        return llm
    except Exception as e:
        log_error(f"llm 模型加载失败: {model_id} - 错误: {str(e)}", logger)
        raise   

# 加载本地 LLM，在mac M1芯片上使用 MPS 加速
def load_llm_from_local(model_path):
    """加载本地LLM，支持缓存避免重复加载"""
    log_model_loading_start(model_path, logger)
    if not os.path.exists(model_path):
        log_error(f"模型不存在: {model_path}", logger)
        raise FileNotFoundError(f"模型不存在: {model_path}")
    else:
        # 设备检测
        if torch.backends.mps.is_available():
            device = "mps"
            data_type = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            data_type = torch.float16
        else:
            device = "cpu"
            data_type = torch.float32
            
    log_device_detection(device, str(data_type), logger)

    # 加载分词器tokenizer 和模型 model
    try:
        # 加载 tokenizer
        log_info(f"🔤 加载分词器: {model_path}", logger)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code = True
        )

        # 加载模型
        log_info(f"🤖 加载模型: {model_path}", logger)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map = device, # "auto"
            trust_remote_code = True,
            dtype = data_type,
        )

        # 返回 tokenizer 和 model
        return tokenizer, model
        
    except Exception as e:
        log_model_loading_failed(model_path, str(e), logger)
        raise

# 测试LLM模型功能, 直接使用tokenizer 和 model, 使用generate 方法
def run_local_llm(tokenizer, model, messages):
    """测试LLM模型功能"""
    log_info("🧪 开始LLM模型功能测试", logger)

    # Prompt生成：使用 tokenizer 内置模板生成 prompt（关键！）
    # 使用 tokenizer.apply_chat_template()，控制输入格式，确保符合官方规范
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True  # 添加 <|im_start|>assistant 标记
    )

    # 编码为 input_ids
    model_inputs = tokenizer([text], return_tensors = "pt").to(model.device)

    # 生成回复
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 只取新生成的部分（去掉输入 prompt）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]

    # 解码
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    log_info(f"💬 模型回答: {response.strip()}", logger)
    log_info("✅ LLM模型功能测试完成", logger)
    return response

def run_ollama_llm(ollama_llm, prompt):
    """测试 Ollama LLM 模型功能"""
    log_info("🧪 开始 Ollama LLM 模型功能测试", logger)
    
    try:
        # 方式一：使用 invoke 方法（推荐）
        log_info(f"📝 发送提示: {prompt}", logger)
        response = ollama_llm.invoke(prompt)
        log_info(f"💬 模型回答: {response.strip()}", logger)
        
        # 方式二：使用 generate 方法（批量处理）
        '''
        results = ollama_llm.generate([prompt])
        response = results.generations[0][0].text
        log_info(f"💬 模型回答: {response.strip()}", logger)
        '''
        
        log_info("✅ Ollama LLM 模型功能测试完成", logger)
        return response
        
    except Exception as e:
        log_error(f"Ollama 测试失败: {str(e)}", logger)
        raise

if __name__ == "__main__":
    MODEL_REPO = 2 # 1 for huggingface_hub, 2 for modelscope repo
    MODEL_DIR = "/Users/carlos/Desktop/PileGo.Ai/ahum_llm/llms"
    LLM_MODEL_LOCAL_TAG = "Qwen/Qwen3-0.6B"
    EMBEDDING_MODEL_LOCAL_TAG = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_MODEL_OLLAMA_TAG = "bge-m3"
    LLM_MODEL_OLLAMA_TAG = "qwen3:8b"

    # 下载embedding model to local
    embedding_model_dir = download_model_to_local(EMBEDDING_MODEL_LOCAL_TAG, MODEL_DIR, MODEL_REPO)
    print(f"Embedding model downloaded to: {embedding_model_dir}")
    ################################################
    ################ local model ###################
    ################################################
    
    '''# 构造对话消息（Qwen3 标准格式）
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "嗨！我是谁？"}
    ]
    prompt = "你好！请简单介绍一下你自己。"

    llm = PI_LLM(isLocal=True)
    local_model_dir = llm.download_model(LLM_MODEL_LOCAL_TAG, MODEL_DIR, MODEL_REPO)
    llm.load_llm_model(local_model_dir)
    # print(llm.tokenizer, llm.llm_model)
    run_local_llm(llm.tokenizer, llm.llm_model, messages)'''

    ################################################
    ################ ollama model ##################
    ################################################
    
    # Ollama 使用简单的文本提示，不需要复杂的 tokenizer 处理
    '''
    prompt = "你好！请简单介绍一下你自己。"
    
    llm = PI_LLM(isLocal=False)
    llm.load_llm_model(LLM_MODEL_OLLAMA_TAG)
    print(llm.ollama_llm)
    llm.load_embeddings_model(EMBEDDING_MODEL_OLLAMA_TAG)
    print(llm.embeddings)
    # print(local_model_dir)
    # print(llm.tokenizer, llm.model)
    
     # 测试 Ollama 模型
    response = run_ollama_llm(llm.ollama_llm, prompt)
    print(f"最终响应: {response}")
    '''