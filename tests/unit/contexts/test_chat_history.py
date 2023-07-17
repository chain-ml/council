import unittest

from council.contexts import ChatHistory


class TestChatHistory(unittest.TestCase):
    def test_last_user_message(self):
        history = ChatHistory()
        history.add_user_message("from user")
        history.add_agent_message("from agent")
        history.add_user_message("from user again")
        self.assertEqual("from user again", history.try_last_user_message.unwrap().message)

    def test_last_user_message_does_not_exist(self):
        history = ChatHistory()
        history.add_agent_message("from agent")
        self.assertTrue(history.try_last_user_message.is_none())

    def test_last_agent_message(self):
        history = ChatHistory()
        history.add_user_message("from user")
        history.add_agent_message("from agent")
        history.add_agent_message("from agent again")
        self.assertEqual("from agent again", history.try_last_agent_message.unwrap().message)
