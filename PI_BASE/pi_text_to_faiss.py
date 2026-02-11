# pi_text_to_faiss.py
"""
pi_text_to_faiss.py
Convert PDF to FAISS vector database
This script converts a PDF file into a FAISS vector database.
Usage:
    python pi_text_to_faiss.py
Process:
    1. Load the PDF file.
    2. Extract text and page numbers.
    3. Split text into chunks.
    4. Create a vector database using the chunks.
"""
from pypdf import PdfReader
# from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback
# from langchain_community.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from pi_down_load_models import PiLLM
from typing import List, Tuple
from pi_log import *
import os
import pickle
import inspect
from langchain_community.llms import Tongyi

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:
    """
    从PDF中提取文本并记录每行文本对应的页码
    
    参数:
        pdf: PDF文件对象
    
    返回:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
    """
    text = ""   # Initialize an empty string to store the extracted text
    page_numbers = []   # Initialize an empty list to store the corresponding page numbers

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:  
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))

    return text, page_numbers

## extract_text_with_page_numbers 将PDF文件提取为文本并记录每行文本对应的页码，每一页的文本与page 对应上了
## process_text_with_splitter 将文本切分成chunk,并与page_numbers 对应上了。
def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None, embeddings = None) -> FAISS:
    """
    处理文本并创建向量存储
    
    参数:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
        save_path: 可选，保存向量数据库的路径
    
    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本类型: {type(chunks)}")  # Print the type of chunks
    print(f"文本被分割成 {len(chunks)} 个块。")
    
    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")
    
    # 改进：存储每个文本块对应的页码信息
    # 创建原始文本的行列表和对应的页码列表
    # text是一个字符串，lines是一个字符串列表
    # text.find(chunk[:100]) 表示在text 找到chunk[:100]的匹配项，返回匹配项的起始位置
    ######################################################################################
    ###### text.find(chunk[:100]) 来查找 chunk 在 text 的匹配项，并返回位置，##################
    ###### 如果找不到，则通过在 chunk 中查找 line[:min(50), len(line)] 某一行前 50 #############
    ###### 个字是否匹配，如果匹配，则把改行的行号i 设置给start_idx ? 如果通过 text.find ###########
    ###### 找到了匹配位置，则把page_numbers列表中的对应行的page页码返回给 page_info[chunk]? ######
     ######################################################################################
    lines = text.split("\n")
    
    # 为每个chunk找到最匹配的页码
    page_info = {}  # {chunk: page_number}
    for chunk in chunks:
        # 查找chunk在原始文本中的开始位置
        start_idx = text.find(chunk[:100])  # 使用chunk的前100个字符作为定位点， 在text 中查找 chunk[:100]的匹配项【第一轮循环中，1号chunk的前100个字符能否在text中找到？】
        if start_idx == -1:
            # 如果找不到精确匹配，则使用模糊匹配
            for i, line in enumerate(lines):
                if chunk.startswith(line[:min(50, len(line))]):  # 检查chunk是否以某行的前50个字符或者“min方法的最小值”开头， 【min(50, len(line)) - 取较小值】
                    start_idx = i
                    break
            
            # 如果仍然找不到，尝试另一种匹配方式
            if start_idx == -1:
                for i, line in enumerate(lines):
                    if line and line in chunk:
                        start_idx = text.find(line)
                        break
        
        # 如果找到了起始位置，确定对应的页码
        if start_idx != -1:
            # 计算这个位置对应原文中的哪一行：# 【行号计算的核心思想：统计start_idx位置前的换行符数量】
            line_count = text[:start_idx].count("\n")
            # 确保不超出页码列表长度
            if line_count < len(page_numbers):
                page_info[chunk] = page_numbers[line_count]
            else:
                # 如果超出范围，使用最后一个页码
                page_info[chunk] = page_numbers[-1] if page_numbers else 1
        else:
            # 如果无法匹配，使用默认页码-1（这里应该根据实际情况设置一个合理的默认值）
            page_info[chunk] = -1
    
    knowledgeBase.page_info = page_info # 动态添加 page_info 属性， 本身knowledgeBase(FAISS 对象) 是没有这个属性的。
    
    # 如果提供了保存路径，则保存向量数据库和页码信息
    if save_path:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS向量数据库
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")
        
        # 保存页码信息到同一目录
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)
        print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")

    return knowledgeBase

def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息
    
    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例
    
    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    # 如果没有提供 embeddings 嵌入模型，则创建一个新的
    if embeddings is None:
        # log记录
        log_critical("没有提供 embeddings 嵌入模型，请配置 load_knowledge_base 的 embeddings arg")
        raise ValueError("没有提供 embeddings 嵌入模型，请配置 load_knowledge_base 的 embeddings arg")
    
    # 加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")
    
    # 加载页码信息
    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            page_info = pickle.load(f)
        knowledgeBase.page_info = page_info
        print("页码信息已加载。")
    else:
        print("警告: 未找到页码信息文件。")
    
    return knowledgeBase


## 导入 pi_down_load_models 中的 PI_LLM，维护一个class, 包含了 llm 和embedding 模型
ollama_EMBEDDING_MODEL = "bge-m3"
ollama_LLM_MODEL = "qwen3:8b"

MODEL_DIR = "/Users/carlos/Desktop/PileGo.Ai/ahum_llm/llms"
LLM_MODEL_LOCAL_TAG = "Qwen/Qwen3-0.6B"
EMBEDDING_MODEL_LOCAL_TAG = "Qwen/Qwen3-Embedding-0.6B"

cLLM = PiLLM()
cLLM.load_embeddings_model(ollama_EMBEDDING_MODEL, isLocal=False)
# cLLM.load_llm_model(os.path.join(MODEL_DIR, LLM_MODEL_LOCAL_TAG), isLocal = True)
# cLLM.load_llm_model(ollama_LLM_MODEL, isLocal=False)

# llm = cLLM.llm_model
embeddings = cLLM.embeddings
llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY) # qwen-turbo

# 读取PDF文件
pdf_reader = PdfReader('./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')
# 提取文本和页码信息
text, page_numbers = extract_text_with_page_numbers(pdf_reader)
text


print(f"提取的文本长度: {len(text)} 个字符。")
    
# 处理文本并创建知识库，同时保存到磁盘
save_dir = "./vector_db"
knowledgeBase = process_text_with_splitter(text, page_numbers, save_path=save_dir, embeddings = embeddings)

# 示例：如何加载已保存的向量数据库
# 注释掉以下代码以避免在当前运行中重复加载
"""
# 创建嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)
# 从磁盘加载向量数据库
loaded_knowledgeBase = load_knowledge_base("./vector_db", embeddings)
# 使用加载的知识库进行查询
docs = loaded_knowledgeBase.similarity_search("客户经理每年评聘申报时间是怎样的？")

# 直接使用FAISS.load_local方法加载（替代方法）
# loaded_knowledgeBase = FAISS.load_local("./vector_db", embeddings, allow_dangerous_deserialization=True)
# 注意：使用这种方法加载时，需要手动加载页码信息
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 定义 prompt 模板（可选，也可以用默认）
prompt = PromptTemplate.from_template(
    "使用以下上下文回答问题：\n\n{context}\n\n问题：{question}\n答案："
)

# 设置查询问题
# query = "客户经理被投诉了，投诉一次扣多少分"
query = "客户经理每年评聘申报时间是怎样的？"

if query:
    #（1）这是第一步，找到相关的文档块，返回最关联的2个文档块
    # 执行相似度搜索，找到与查询相关的文档
    # query: 用户的查询问题
    # k=2: 指定返回前2个最相关的文档块
    # docs: 返回值为包含2个document 对象的列表
    docs = knowledgeBase.similarity_search(query,k = 3)

    '''# 加载问答链,load_qa_chain 是以前的版本了
    # chain = load_qa_chain(llm, chain_type="stuff")'''

    #（2）这是第二步，构建问答链，将文档块和问题一起传递给LLM
    # 构建 chain
    # 这不是循环的概念，而是列表推导式：
    # [doc.page_content for doc in x["input_documents"]]
    # 等价于：
    # content_list = []
    # for doc in x["input_documents"]:  # 遍历所有文档对象
    #     content_list.append(doc.page_content)  # 提取每个文档的内容属性
    #
    # RunnablePassthrough() 是 LangChain 中的一个传递组件，它的作用是原样传递输入数据。

    rag_chain = (
        {"context": lambda x: "\n\n".join([doc.page_content for doc in x["input_documents"]]), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 准备输入数据, 包含(1)文档块和(2)问题
    input_data = {"input_documents": docs, "question": query}

    # 使用回调函数跟踪API调用成本
    with get_openai_callback() as cost:
        # 执行问答链
        response = rag_chain.invoke(input = input_data)
        print(f"查询已处理。成本: {cost}")
        # print(response["output_text"])
        print(response)
        print("来源:")

    # 记录唯一的页码
    unique_pages = set()  #set是创建一个集合

    # 显示每个文档块的来源页码
    # getattr 动态获取对象属性
    for doc in docs:
        text_content = getattr(doc, "page_content", "")  # 动态获取doc对象的page_content属性
        source_page = knowledgeBase.page_info.get(
            text_content.strip(), "未知"
        )

        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本块页码: {source_page}")