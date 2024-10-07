import unittest
from app import app

class TestApp(unittest.TestCase):

    def setUp(self):
        # Set up the test client
        app.testing = True  # Enable testing mode
        self.client = app.test_client()

    def test_index_route(self):
        # Send a GET request to the index route
        response = self.client.get('/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>IRenE Webapp - Home</title>', response.data)

if __name__ == '__main__':
    unittest.main()
