import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """初始化向量库管理器
        
        Args:
            persist_directory: 向量库持久化存储目录
        """
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        self.vectorstore = self._init_vectorstore()
    
    def _init_vectorstore(self) -> Chroma:
        """初始化向量数据库"""
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        
        # 如果不存在，创建新的向量数据库
        loader = DirectoryLoader(
            '../txts',
            glob="**/*.txt",
            show_progress=True,
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)
        
        return Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def add_file(self, file_path: str) -> bool:
        """添加单个文件到向量库
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否添加成功
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False
            
        try:
            # 尝试不同编码
            for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    documents = loader.load()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"无法读取文件，请确保文件是文本文件: {file_path}")
                return False
            
            splits = self.text_splitter.split_documents(documents)
            self.vectorstore.add_documents(splits)
            print(f"成功添加文件: {file_path}")
            return True
            
        except Exception as e:
            print(f"添加文件时出错: {str(e)}")
            return False
    
    def add_directory(self, dir_path: str) -> bool:
        """添加整个目录到向量库
        
        Args:
            dir_path: 目录路径
            
        Returns:
            bool: 是否添加成功
        """
        if not os.path.exists(dir_path):
            print(f"目录不存在: {dir_path}")
            return False
            
        try:
            # 尝试不同编码
            for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
                try:
                    loader = DirectoryLoader(
                        dir_path,
                        glob="**/*.txt",
                        show_progress=True,
                        loader_cls=TextLoader,
                        loader_kwargs={'encoding': encoding}
                    )
                    documents = loader.load()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"无法读取文件，请确保文件是文本文件: {dir_path}")
                return False
            
            splits = self.text_splitter.split_documents(documents)
            self.vectorstore.add_documents(splits)
            print(f"成功添加目录: {dir_path}")
            return True
            
        except Exception as e:
            print(f"添加目录时出错: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[dict]:
        """搜索相似文档
        
        Args:
            query: 搜索关键词
            k: 返回的文档数量
            
        Returns:
            List[dict]: 搜索结果列表
        """
        docs = self.vectorstore.similarity_search(query, k=k)
        results = []
        for doc in docs:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        return results
    
    def get_all_documents(self) -> List[dict]:
        """获取所有文档
        
        Returns:
            List[dict]: 所有文档列表
        """
        collection = self.vectorstore.get()
        results = []
        for doc, metadata in zip(collection['documents'], collection['metadatas']):
            results.append({
                'content': doc,
                'metadata': metadata
            })
        return results

def main():
    """命令行界面"""
    manager = VectorStoreManager()
    
    while True:
        print("\n=== 向量库管理系统 ===")
        print("1. 添加单个文件")
        print("2. 添加整个目录")
        print("3. 查看所有文档")
        print("4. 搜索相似文档")
        print("5. 退出")
        
        choice = input("\n请选择操作 (1-5): ")
        
        if choice == "1":
            file_path = input("\n请输入要添加的文件路径: ")
            manager.add_file(file_path)
            
        elif choice == "2":
            dir_path = input("\n请输入要添加的目录路径: ")
            manager.add_directory(dir_path)
            
        elif choice == "3":
            docs = manager.get_all_documents()
            print("\n=== 所有文档 ===")
            for i, doc in enumerate(docs, 1):
                print(f"\n文档 {i}:")
                print(f"内容: {doc['content']}")
                print(f"元数据: {doc['metadata']}")
            
        elif choice == "4":
            query = input("\n请输入搜索关键词: ")
            k = int(input("请输入要返回的文档数量: "))
            docs = manager.search(query, k)
            print(f"\n=== 与 '{query}' 相关的文档 ===")
            for i, doc in enumerate(docs, 1):
                print(f"\n文档 {i}:")
                print(f"内容: {doc['content']}")
                print(f"元数据: {doc['metadata']}")
            
        elif choice == "5":
            break
            
        else:
            print("无效的选择！")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main() 