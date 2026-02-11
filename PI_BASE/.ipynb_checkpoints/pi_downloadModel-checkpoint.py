from modelscope import snapshot_download as ms_snapshot_download
from modelscope.hub.constants import MODEL_ID_SEPARATOR
from huggingface_hub import snapshot_download as hf_snapshot_download 
import os

# 导入PI项目日志模块
from pi_log import *

# 全局变量存储已加载的模型
_LOADED_MODELS = {}

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

if __name__ == "__main__":
    MODEL_REPO = 1
    MODEL_DIR = "/Users/carlos/Desktop/PileGo.Ai/ahum_llm/llms"
    LLM_MODEL_TAG = "Qwen/Qwen3-0.6B"
    
    download_model_to_local(LLM_MODEL_TAG, MODEL_DIR, MODEL_REPO)