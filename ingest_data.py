from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter


def prepare_dataset():
    loader = PyPDFLoader("baza_wiedzy.pdf")
    pages = loader.load()
    texts= [page.page_content for page in pages]
    return texts

def main():
    data = prepare_dataset()
    texts = data[:5]

    # embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={"device": "cpu"})
    text_embeddings = embedding_model.embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))

    # vector db
    vector_db = FAISS.from_embeddings(text_embedding_pairs, embedding_model)
    vector_db.save_local("vector_db_faiss")


if __name__ == "__main__":
    main()
