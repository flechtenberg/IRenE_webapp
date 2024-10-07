# tests/test_app.py

import unittest
from app import app  # Import your Flask app

class TestApp(unittest.TestCase):

    def setUp(self):
        # Set up the test client
        app.testing = True  # Enable testing mode
        app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
        self.client = app.test_client()

        # Create a context for the tests
        self.ctx = app.app_context()
        self.ctx.push()

    def tearDown(self):
        # Pop the context
        self.ctx.pop()

    def test_index_route(self):
        # Test the index route
        response = self.client.get('/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'IRenE Webapp - Home', response.data)


if __name__ == '__main__':
    unittest.main()
