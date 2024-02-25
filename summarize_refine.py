from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

ret = chain.invoke(split_docs)
print(ret)