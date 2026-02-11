from modelscope import snapshot_download as ms_snapshot_download
from modelscope.hub.constants import MODEL_ID_SEPARATOR
# from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub import snapshot_download as hf_snapshot_download 

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 导入PI项目日志模块
from pi_log import *

# 全局变量存储已加载的模型
_LOADED_MODELS = {}

# 获取logger实例（启用文件日志）
logger = get_logger("PI_LLM", log_to_file=True)

# 下载整个模型仓库到本地， 这里使用modelscope 的snapshot_download 方法
def download_model_to_local(model_id, model_dir):
    """下载模型到本地目录"""
    model_path = os.path.join(model_dir, model_id.replace(MODEL_ID_SEPARATOR, "/"))
    
    log_download_start(model_id, model_dir, logger)
    
    if os.path.exists(model_path):
        log_file_exists(model_path, logger)
        return model_path
    else:
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

def load_embedding_model_from_local(model_id):
    """从本地加载嵌入模型"""
    log_info(f"📦 加载嵌入模型: {model_id}", logger)
    try:
        model = SentenceTransformer(model_id)
        log_info(f"✅ 嵌入模型加载成功: {model_id}", logger)
        return model
    except Exception as e:
        log_error(f"❌ 嵌入模型加载失败: {model_id} - 错误: {str(e)}", logger)
        raise

# 加载本地 LLM，在mac M1芯片上使用 MPS 加速
def load_llm_from_local(model_path, use_cache=True):
    """加载本地LLM，支持缓存避免重复加载"""
    log_model_loading_start(model_path, logger)
    
    # 检查缓存
    if use_cache and model_path in _LOADED_MODELS:
        log_cache_hit(model_path, logger)
        return _LOADED_MODELS[model_path]
    else:
        log_cache_miss(model_path, logger)
        
        # 设备检测
        if torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32
            
        log_device_detection(device, str(torch_dtype), logger)

    try:
        # 加载 tokenizer
        log_info(f"🔤 加载分词器: {model_path}", logger)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )

        # 加载模型
        log_info(f"🤖 加载模型: {model_path}", logger)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # 缓存模型
        if use_cache:
            _LOADED_MODELS[model_path] = (tokenizer, model)
            log_cache_store(model_path, logger)

        log_model_loading_success(model_path, device, logger)
        return tokenizer, model
        
    except Exception as e:
        log_model_loading_failed(model_path, str(e), logger)
        raise

def test_embedding_model(model):
    """测试嵌入模型功能"""
    log_info("🧪 测试嵌入模型功能", logger)
    sentences = ["Hello, world!", "你好，世界！"]
    embeddings = model.encode(sentences)
    log_info(f"📊 Embedding shape: {embeddings.shape}", logger)
    log_info(f"📈 First 5 dims of first sentence: {embeddings[0][:5]}", logger)
    return embeddings

def test_llm(tokenizer, model):
    """测试LLM模型功能"""
    log_info("🧪 开始LLM模型功能测试", logger)
    
    # 构造对话消息（Qwen3 标准格式）
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "嗨！我是谁？"}
    ]

    # 使用 tokenizer 内置模板生成 prompt（关键！）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # 添加 <|im_start|>assistant 标记
    )

    # 编码为 input_ids
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

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

def clear_model_cache(model_path=None):
    """清理模型缓存"""
    if model_path is None:
        # 清理所有缓存
        _LOADED_MODELS.clear()
        log_cache_clear("ALL", logger)
    elif model_path in _LOADED_MODELS:
        # 清理指定模型
        del _LOADED_MODELS[model_path]
        log_cache_clear(model_path, logger)
    else:
        log_warning(f"未找到缓存: {model_path}", logger)

# 使用示例
# clear_model_cache()  # 清理全部
# clear_model_cache("/path/to/specific/model")  # 清理特定模型

if __name__ == "__main__":
    '''
    # 设置网络相关环境变量
    import os
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像站点
    '''

    model_dir = "/Users/carlos/Desktop/PileGo.Ai/ahum_llm/llms"
    
    embeddings_model_tag = "BAAI/bge-m3"
    llm_model_tag      = "Qwen/Qwen3-1.7B"
    llm_model_path = os.path.join(model_dir, llm_model_tag.replace(MODEL_ID_SEPARATOR, "/"))
    # embeddings_model_path = os.path.join(model_dir, embeddings_model_tag.replace(MODEL_ID_SEPARATOR, "/"))

    download_model_to_local(llm_model_tag, model_dir)
    llm_model = load_llm_from_local(llm_model_path)
   
    # embeddings_model = load_embedding_model_from_local(embeddings_model_path)
    # test_embedding_model(embeddings_model)
    test_llm(llm_model[0], llm_model[1])

    '''
    # 判断是否加载成功
    if is_model_loaded(llm_model[0], llm_model[1]):
        print("✅ Qwen3-1.7B 模型已成功加载！")
        # 执行推理...
    else:
        print("❌ 模型加载失败，请检查路径或环境")
        exit(1)'''