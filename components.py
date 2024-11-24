from groq import Groq

class mem_cell():
    def __init__(self, human_message, AI_message) -> None:
        self.mem = {"human":human_message,
                    "AI":AI_message}

class Memory():
    def __init__(self) -> None:
        self.memory = []
    
    def add_mem(self, human_message, AI_message):
        cell = mem_cell(human_message, AI_message)
        if len(self.memory) == 5:
            del self.memory[0]
        self.memory.append(cell.mem)

main_prompt = """
You are an assistant that needs to answer questions about me (Arav) to my future employers. Use the context to answer the user's questions. 
Note that all questions might not require you to use the context. If there is something you don't know about, say so and do not make up any information.
Make sure to give elaborate but concise answers, not longer than 40 words
context: {context}
"""

rephrase_prompt = """
rephrase the given question using previous conversation to be a standalone question that can be passed to another llm. Do not answer the question 
only rephrase it to give a new question. If there is no need to rephrase then just pass the question as it is.
memory: {memory}
"""

class ChatGroq():
    def __init__(self, api_key):
        self.client = Groq(
            api_key = api_key,
            )
        self.prompt = None

    def invoke(self, message):
        if self.prompt == None:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                model="llama3-70b-8192",
            )
        else:
            chat_completion = self.client.chat.completions.create(
                messages = [
                    {
                        "role": "system",
                        "content": self.prompt,
                    },
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                model="llama3-70b-8192",
            )
        return chat_completion.choices[0].message.content
    
    def pass_prompt(self, prompt):
        self.prompt = prompt