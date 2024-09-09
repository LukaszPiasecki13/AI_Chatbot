AI_Chatbot was created as a study aid for the university entrance exam. When I was preparing for exams in my studies I got an idea to prepare something like CHAT GPT, where I load our notes from my studies and based on this he will be answering my questions. 
The biggest problem is that the LLM model is on my computer, because OpenAI and Google models are paid and additionally all operation are doing on CPU not GPU, because I had problem with CUDA settings and this is the reason why this chat is so slow. 

![Projekt-bez-nazwy](https://github.com/user-attachments/assets/69f5cc59-59e5-4059-9691-e44adb7acdf6)


Firstly, I am downloading data from PDF file. It was planned to be organised such a way that one question and answer were on one separate page, and then I am reading this data and creating a list. If I had a larger file, I would use a generator. In next stage I am calculating embeddings using open source model to calculate and I save it in file. During using chat my model will using these vectors. In next step I am searching pages on which are most suitable data for my query. Next I combine my query, context which I got and I create promt, which I send to language model. 
