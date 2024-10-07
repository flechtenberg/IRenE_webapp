import unittest
from unittest.mock import patch
from backend.processing import preprocess_text, get_keywords, scopus_sampling_process, weighted_random_selection, construct_search_query, execute_search_scopus

class TestProcessing(unittest.TestCase):

    def test_preprocess_text(self):
        raw_text = (
            "This is a Sample Text with Numbers 123 and URLs https://example.com. "
            "Also, mentions of New York and Microsoft."
        )
        expected_output = "sample text number url mention"
        processed_text = preprocess_text(raw_text)
        self.assertEqual(processed_text, expected_output)

    def test_get_keywords(self):
        # Simulate multiple documents by concatenating sentences
        document1 = (
            "Machine learning is a method of data analysis that automates analytical model building. "
            "It is a branch of artificial intelligence based on the idea that systems can learn from data, "
            "identify patterns and make decisions with minimal human intervention."
        )

        document2 = (
            "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems "
            "to extract knowledge and insights from noisy, structured and unstructured data."
        )

        document3 = (
            "Artificial intelligence refers to the simulation of human intelligence in machines that are programmed "
            "to think like humans and mimic their actions."
        )

        # Combine the documents into one seed text
        seed_texts = [document1, document2, document3]

        num_keywords = 5
        keywords = get_keywords(seed_texts, num_keywords)
        extracted_keywords = [kw['word'] for kw in keywords]

        # Expected keywords based on term frequency and importance
        expected_keywords = ['data', 'intelligence', 'systems', 'artificial', 'human']

        self.assertEqual(len(keywords), num_keywords)
        self.assertListEqual(extracted_keywords, expected_keywords)

    def test_weighted_random_selection(self):
        keywords = ['keyword1', 'keyword2', 'keyword3']
        weights = [0.5, 0.3, 0.2]
        selected_keyword = weighted_random_selection(keywords, weights)
        self.assertIn(selected_keyword, keywords)

    def test_weighted_random_selection_zero_weights(self):
        keywords = ['keyword1', 'keyword2']
        weights = [0, 0]
        selected_keyword = weighted_random_selection(keywords, weights)
        self.assertIsNone(selected_keyword)

    def test_construct_search_query(self):
        selected_keywords = ['keyword1', 'keyword2', 'keyword3']
        expected_query = 'keyword1 AND keyword2 AND keyword3'
        query = construct_search_query(selected_keywords)
        self.assertEqual(query, expected_query)

    def test_construct_search_query_empty(self):
        selected_keywords = []
        expected_query = ''
        query = construct_search_query(selected_keywords)
        self.assertEqual(query, expected_query)

    @patch('backend.processing.ElsSearch')
    def test_execute_search_scopus_success(self, mock_els_search):
        # Mock the ElsSearch instance
        mock_search_instance = mock_els_search.return_value
        mock_search_instance.tot_num_res = 1
        mock_search_instance.results = [{
            'dc:identifier': 'SCOPUS_ID:1234567890',
            'dc:creator': 'John Doe',
            'prism:coverDate': '2021-01-01',
            'dc:title': 'Sample Paper Title',
            'prism:publicationName': 'Sample Journal',
            'citedby-count': '10',
            'openaccess': '1',
            'link': [{'@href': 'link1'}, {'@href': 'link2'}, {'@href': 'http://scopus.com/samplelink'}]
        }]

        # Call the function
        match_count, matched_papers = execute_search_scopus(
            query='test query',
            scopus_api_key={'apikey': 'dummy_key', 'insttoken': 'dummy_token'},
            threshold=10
        )

        # Assertions
        self.assertEqual(match_count, 1)
        self.assertEqual(len(matched_papers), 1)
        self.assertEqual(matched_papers[0]['scopus_id'], '1234567890')
        self.assertEqual(matched_papers[0]['first_author'], 'John Doe')

    @patch('backend.processing.ElsSearch.execute')
    def test_execute_search_scopus_exception(self, mock_execute):
        # Mock an exception during execute
        mock_execute.side_effect = Exception('API Error')

        # Call the function
        match_count, matched_papers = execute_search_scopus(
            query='test query',
            scopus_api_key={'apikey': 'dummy_key', 'insttoken': 'dummy_token'},
            threshold=10
        )

        # Assertions
        self.assertEqual(match_count, 0)
        self.assertEqual(matched_papers, set())

    @patch('backend.processing.execute_search_scopus')
    def test_scopus_sampling_process(self, mock_execute_search):
        # Mock execute_search_scopus to return consistent results
        mock_execute_search.return_value = (5, [
            {
                'scopus_id': '12345',
                'first_author': 'John Doe',
                'year': '2021',
                'title': 'Sample Paper Title',
                'journal': 'Sample Journal',
                'citations': '10',
                'open_access': '1',
                'link': 'http://scopus.com/samplelink'
            }
        ])
        weight_dict = {'keyword1': 1.0, 'keyword2': 2.0}
        threshold = 10
        outer_iterations = 2

        result = scopus_sampling_process(
            weight_dict=weight_dict,
            threshold=threshold,
            outer_iterations=outer_iterations,
            scopus_api_key={'apikey': 'dummy_key', 'insttoken': 'dummy_token'}
        )

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['scopus_id'], '12345')
        self.assertEqual(result[0]['occurrences'], outer_iterations)

if __name__ == '__main__':
    unittest.main()
