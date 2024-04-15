# -*- coding:utf-8 -*-
# Author: illusion_Shen
# Time  : 2024/3/25 14:19

# from langchain.memory import ConversationSummaryBufferMemory
# from langchain.llms import OpenAI
#
# llm = OpenAI()
#
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)
# memory.save_context({"input": "hi"}, {"output": "whats up"})
# memory.save_context({"input": "not much you"}, {"output": "not much"})
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Milvus


import faiss

from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS


embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
)
# In actual usage, you would set `k` to be a higher value, but we use k=1 to show that
# the vector lookup still returns the semantically relevant information
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# When added to an agent, the memory object can save pertinent information from conversations or used tools
memory.save_context({"input": "My favorite food is pizza"}, {"output": "that's good to know"})
memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"})

print(memory.load_memory_variables({"prompt": "what sport should i watch?"})["history"])

docs = vector_db.similarity_search('hahahaha', search_kwargs={"filter": {'namespace':"ankush"}}, k=2)
