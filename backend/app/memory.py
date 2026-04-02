from collections import deque

class ConversationMemory:
    def __init__(self, max_len=5):
        self.history = deque(maxlen=max_len)

    def add(self, query, response):
        self.history.append({
            "query": query,
            "answer": response
        })

    def get_context(self):
        context = ""
        for item in self.history:
            context += f"\nUser: {item['query']}\nAI: {item['answer']}\n"
        return context