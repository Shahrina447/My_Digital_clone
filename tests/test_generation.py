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

    def test_standalone_greeting_gets_welcome_without_retrieval(self):
        result = self.engine.ask('Hi!')
        self.assertIn("AI clone", result['answer'])
        self.assertEqual(result['chunks'], [])
        self.assertEqual(result['generated_by'], 'Greeting')


if __name__ == '__main__':
    unittest.main()
