from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming resposne
import gradio as gr
from similarity_search import similarity_search



MODEL_PATH = r"C:\Users\Lukasz\AppData\Local\nomic.ai\GPT4All\Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf" 


def prepare_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm_model = LlamaCpp(
        model_path=MODEL_PATH, callback_manager=callback_manager, verbose=True, n_ctx=1024
    )

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}"""

    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    llm_chain = LLMChain(prompt=prompt, llm=llm_model, verbose=True)
    return llm_chain

def generate_answer(question: str) -> str:
    context_docs = similarity_search(question)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    input = {
    'question': question,
    'context': context
    }

    llm_chain = prepare_llm()

    answer = llm_chain.invoke(input=input)
    return answer

def main():
    while True:
        question = input("Enter your question. If you want to exit, type 'exit': ")
        if question == "exit":
            break

        answer = generate_answer(question)
        print(answer)



if __name__ == "__main__":
    main()
