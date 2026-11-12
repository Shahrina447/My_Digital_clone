import unittest

from src.engine import CloneEngine


class CloneEngineFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CloneEngine(
            data_dir='data',
            chunks_path='chunks.json',
            embeddings_path='embeddings.npy',
        )

    def test_full_name_question_returns_matching_faq_answer(self):
        result = self.engine.ask('What is your full name?', mode='Hybrid (RRF)', top_k=3)
        self.assertIn('Shahrina Khan', result['answer'])

    def test_location_question_returns_location_answer(self):
        result = self.engine.ask('Where are you from?', mode='Hybrid (RRF)', top_k=3)
        self.assertIn('Sialkot', result['answer'])

    def test_question_with_unrelated_keywords_uses_context_not_advice(self):
        result = self.engine.ask('Why did you choose computer science?', mode='Hybrid (RRF)', top_k=3)
        self.assertIn('Computer Science', result['answer'])
        self.assertNotIn('tutorial hell', result['answer'].lower())

    def test_background_question_returns_background_summary(self):
        result = self.engine.ask("What's your background?", mode='Hybrid (RRF)', top_k=3)
        self.assertIn('fourth-year BS Computer Science student', result['answer'])
        self.assertIn('UMT Sialkot', result['answer'])
        self.assertNotIn('###', result['answer'])

    def test_python_question_returns_profile_skills_not_junior_advice(self):
        result = self.engine.ask('What do you know about Python?', mode='Hybrid (RRF)', top_k=3)
        self.assertIn('FastAPI and Flask', result['answer'])
        self.assertNotIn('fundamentals before', result['answer'].lower())

    def test_informal_project_question_matches_projects_faq(self):
        result = self.engine.ask('What project you have dont yet?', mode='Hybrid (RRF)', top_k=3)
        self.assertIn('DeadlineOS', result['answer'])
        self.assertIn('Verdade', result['answer'])
        self.assertNotIn('###', result['answer'])

    def test_standalone_greeting_gets_welcome_without_retrieval(self):
        result = self.engine.ask('Hi!')
        self.assertIn("AI clone", result['answer'])
        self.assertEqual(result['chunks'], [])
        self.assertEqual(result['generated_by'], 'Greeting')

    def test_positive_acknowledgment_gets_conversational_reply(self):
        result = self.engine.ask('Ahnnn..... Thats great!')
        self.assertIn("glad that was helpful", result['answer'])
        self.assertEqual(result['chunks'], [])
        self.assertEqual(result['generated_by'], 'Conversation')

    def test_substantive_question_with_positive_word_still_uses_retrieval(self):
        result = self.engine.ask('What is a great project you have worked on?')
        self.assertIn('DeadlineOS', result['answer'])
        self.assertNotEqual(result['generated_by'], 'Conversation')


if __name__ == '__main__':
    unittest.main()
