# conversation_store.py
from uuid import uuid4

class ConversationStore:
    def __init__(self):
        self.sessions = {}

    def new_session(self, agent):
        session_id = str(uuid4())
        self.sessions[session_id] = agent.new_conversation()
        return session_id

    def get(self, session_id):
        return self.sessions.get(session_id)