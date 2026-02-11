from pypdf import PdfReader
# from langchain.chains.question_answering import load_qa_chain
# from langchain_openai import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.retrievers import MultiQueryRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import Tongyi
from typing import List, Tuple
import os
import pickle
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

## 导入 pi_down_load_models 中的 PI_LLM，维护一个class, 包含了 llm 和embedding 模型
from pi_down_load_models import PiLLM
ollama_EMBEDDING_MODEL = "bge-m3"
ollama_LLM_MODEL = "qwen3:8b"

MODEL_DIR = "/Users/carlos/Desktop/PileGo.Ai/ahum_llm/llms"
LLM_MODEL_LOCAL_TAG = "Qwen/Qwen3-0.6B"
EMBEDDING_MODEL_LOCAL_TAG = "Qwen/Qwen3-Embedding-0.6B"

# 初始化 PiLLM：加载 LLM 和 Embedding 模型
cLLM = PiLLM()
cLLM.load_embeddings_model(ollama_EMBEDDING_MODEL, isLocal=False)
cLLM.load_llm_model(ollama_LLM_MODEL, isLocal=False)
llm = cLLM.llm_model
embeddings = cLLM.embeddings
if llm is None:
    raise RuntimeError("LLM 模型未成功加载！请检查 PiLLM.load_llm_model 的实现和模型名称。")
if embeddings is None:
    raise RuntimeError("嵌入模型未成功加载！请检查 PiLLM.load_embeddings_model 的实现和模型名称。")

# 获取环境变量中的 DASHSCOPE_API_KEY
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
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))
        else:
            print(f"No text found on page {page_number}.")

    return text, page_numbers

def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None) -> FAISS:
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
    print(f"文本被分割成 {len(chunks)} 个块。")

    embeddings = cLLM.embeddings
    
    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")
    
    # 改进：存储每个文本块对应的页码信息
    lines = text.split("\n")
    page_info = {}
    for chunk in chunks:
        # 查找chunk在原始文本中的开始位置
        start_idx = text.find(chunk[:100])  # 使用chunk的前100个字符作为定位点
        if start_idx == -1:
            # 如果找不到精确匹配，则使用模糊匹配
            for i, line in enumerate(lines):
                if chunk.startswith(line[:min(50, len(line))]):
                    start_idx = i
                    break
            if start_idx == -1:
                for i, line in enumerate(lines):
                    if line and line in chunk:
                        start_idx = text.find(line)
                        break
        if start_idx != -1:
            line_count = text[:start_idx].count("\n")
            if line_count < len(page_numbers):
                page_info[chunk] = page_numbers[line_count]
            else:
                page_info[chunk] = page_numbers[-1] if page_numbers else 1
        else:
            page_info[chunk] = -1
    knowledgeBase.page_info = page_info
    
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
    # 如果没有提供嵌入模型，则创建一个新的
    if embeddings is None:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
    
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


###### 使用MultiQueryRetriever 基本工作流程：######
###### 1. 用户输入单个查询
###### 2. LLM 生成多个相关查询变体, 这里使用MultiQueryRetriever
###### 3. 对每个查询变体执行检索
###### 4. 合并并去重所有检索结果
###### 5. 返回最相关的结果
################################################

############ 创建MultiQueryRetriever ############
## 返回MultiQueryRetriever 检索器
# 创建MultiQueryRetriever
def create_multi_query_retriever(vectorstore, llm):
    """
    创建MultiQueryRetriever
    
    参数:
        vectorstore: 向量数据库
        llm: 大语言模型，用于查询改写
    
    返回:
        retriever: MultiQueryRetriever对象
    """

    # 创建基础检索器：这个基础检索器是由向量数据库转化成的。
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 创建MultiQueryRetriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt = QUERY_PROMPT
    )
    
    return retriever

# 如果你想要类似 MultiQuery 的效果，可以手动实现：
def enhanced_retrieval(query, base_retriever, llm):
    """手动实现增强检索"""
    # 1. 使用基础检索
    docs = base_retriever.get_relevant_documents(query)
    
    # 2. 可选：使用 LLM 生成相关查询进行额外检索
    # additional_queries = llm.invoke(f"Related queries for: {query}")
    # for q in additional_queries:
    #     docs.extend(base_retriever.get_relevant_documents(q))
    
    return docs

# 使用MultiQueryRetriever处理查询
def process_query_with_multi_retriever(query: str, retriever, llm, knowledgeBase):
    """
    使用MultiQueryRetriever处理查询
    
    参数:
        query: 用户查询
        retriever: MultiQueryRetriever对象
        llm: 大语言模型
    
    返回:
        response: 回答
        unique_pages: 相关文档的页码集合
    """
    # 执行查询，获取相关文档
    docs = retriever.invoke(query)
    print(f"找到 {len(docs)} 个相关文档")

    # 构建 RAG 链（现代写法）
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_prompt = PromptTemplate.from_template(
        "使用以下上下文回答问题：\n\n{context}\n\n问题：{question}\n答案："
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    response_text = rag_chain.invoke(query)

    # 获取页码
    unique_pages = set()
    for doc in docs:
        page = knowledgeBase.page_info.get(doc.page_content.strip(), "未知")
        unique_pages.add(page)

    return {"output_text": response_text}, unique_pages

## from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    pdf_path = './浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf'
    vector_db_path = './vector_db'

    # 加载或创建向量数据库
    if os.path.exists(vector_db_path) and os.path.isdir(vector_db_path):
        print(f"发现现有向量数据库: {vector_db_path}")
        knowledgeBase = load_knowledge_base(vector_db_path, embeddings)
    else:
        print("未找到向量数据库，正在从 PDF 创建...")
        pdf_reader = PdfReader(pdf_path)
        text, page_numbers = extract_text_with_page_numbers(pdf_reader)
        print(f"提取文本长度: {len(text)} 字符")
        knowledgeBase = process_text_with_splitter(text, page_numbers, save_path=vector_db_path)

    # === Step 1: 创建 MultiQueryRetriever（使用专用 prompt）===
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=(
            "你是一个AI助手，任务是将用户的问题改写成3个语义相同但表述不同的搜索查询。\n"
            "这些查询将用于检索相关文档。请每行输出一个查询，不要编号，不要解释。\n\n"
            "原始问题: {question}\n\n"
            "改写后的查询："
        ),
    )

    base_retriever = knowledgeBase.as_retriever(search_kwargs={"k": 4})
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=QUERY_PROMPT
    )

    # === Step 2: 定义 RAG 回答链（使用 QA prompt）===
    QA_PROMPT = PromptTemplate.from_template(
        "使用以下上下文回答问题。如果不知道，请回答“根据提供的资料无法确定”。\n\n"
        "上下文:\n{context}\n\n"
        "问题: {question}\n"
        "答案:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": multi_query_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    # === Step 3: 执行查询 ===
    queries = [
        "客户经理被投诉了，投诉一次扣多少分",
        "客户经理每年评聘申报时间是怎样的？",
        "客户经理的考核标准是什么？"
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"🔍 查询: {query}")

        # 获取检索到的文档（用于页码）
        retrieved_docs = multi_query_retriever.invoke(query)
        print(f"📄 检索到 {len(retrieved_docs)} 个相关片段")

        # 获取回答
        answer = rag_chain.invoke(query)

        # 提取唯一来源页码
        unique_pages = set()
        for doc in retrieved_docs:
            content = doc.page_content.strip()
            page = knowledgeBase.page_info.get(content, "未知")
            unique_pages.add(page)

        # 输出结果
        print("\n💡 回答:")
        print(answer)
        print("\n📚 来源页码:")
        for p in sorted(unique_pages):
            print(f"  - 第 {p} 页")
        print("=" * 60)

if __name__ == "__main__":
    main()

