import psycopg2
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

class LlamaIndex:
    def __init__(self, db_name: str, db_user: str, db_password: str, db_host: str, db_port: str, embed_model: str = 'jinaai/jina-embeddings-v2-base-zh'):
        self.embed_model = self.load_hf_embedding_model(embed_model)
        self.llm = self.load_zephyr()
        self.vector_store = self.init_vector_store(db_name, db_user, db_password, db_host, db_port)

    def load_hf_embedding_model(self, model_name: str) -> HuggingFaceEmbedding:
        """Load embedding model."""
        ret = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)
        print(f'Embedding model: {model_name} loaded')
        return ret

    def load_zephyr(self) -> HuggingFaceLLM:
        """Load LLM."""
        def completion_to_prompt(completion):
            return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"
        def messages_to_prompt(messages):
            prompt = ""
            for message in messages:
                if message.role == "system":
                    prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == "user":
                    prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == "assistant":
                    prompt += f"<|assistant|>\n{message.content}</s>\n"
            if not prompt.startswith("<|system|>\n"):
                prompt = "<|system|>\n</s>\n" + prompt
            prompt = prompt + "<|assistant|>\n"
            return prompt
        llm = HuggingFaceLLM(
            model_name="TheBloke/zephyr-7B-beta-GPTQ",
            tokenizer_name="TheBloke/zephyr-7B-beta-GPTQ",
            context_window=3900,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            device_map="auto",
        )
        print('LLM quantized zephyr-beta loaded')
        return llm

    def init_vector_store(self, db_name:str,  user:str, password:str, host:str, port:str) -> None:
        """Initialize postgresql vector store."""
        conn = psycopg2.connect(
            dbname="postgres",
            host=host,
            password=password,
            port=port,
            user=user,
        )
        conn.autocommit = True

        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
            c.execute(f"CREATE DATABASE {db_name}")

        vector_store = PGVectorStore.from_params(
            database=db_name,
            host=host,
            password=password,
            port=port,
            user=user,
            table_name="rag",
            embed_dim=768,  # jinaai embedding dimension
        )
        print('PostgreSQL initialized')
        return vector_store

    def load_data(self, path: str, chunk_size: int = 256, overlap: int=64) -> None:
        import os
        files = ['/content/data/' + x for x in os.listdir('content/data')]
        reader = SimpleDirectoryReader(
            input_files=files
        )
        text_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        doc_n = 0
        chunk_n = 0
        for docs in reader.iter_data():
            for doc in docs:
                cur_text_chunks = text_parser.split_text(doc.text)
                for chunk in cur_text_chunks:
                    node = TextNode(
                        text=chunk,
                    )
                    node.metadata = doc.metadata
                    node.embedding = self.embed_model.get_text_embedding(
                        node.get_content(metadata_mode="all")
                    )
                    self.vector_store.add([node])
                    chunk_n += 1
                doc_n += 1

        print(f'{doc_n} documents loaded. Total {chunk_n} chunks.')