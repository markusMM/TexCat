import unittest
import pandas as pd
from src.preprocessing import detect_language, pp_detect_language, pp_numeric_label_with_missings


class TestDetectLanguage(unittest.TestCase):

    def test_detect_language(self):
        # Test case 1: English text
        text1 = "This is an English text."
        result1 = detect_language(text1)
        self.assertEqual(result1['lang'], 'en')
        self.assertEqual(result1['Lang_acc'], 100)

        # Test case 2: German text
        text2 = "Das ist ein deutscher Text."
        result2 = detect_language(text2)
        self.assertEqual(result2['lang'], 'de')
        self.assertEqual(result2['Lang_acc'], 100)

        # Test case 3: Mixed language text
        text3 = "Bonjour! This is an English and French text."
        result3 = detect_language(text3)
        self.assertIn(result3['lang'], ['en', 'fr'])  # Language could be either English or French
        self.assertEqual(result3['Lang_acc'], 100)

        # Test case 4: Empty text
        text4 = ""
        result4 = detect_language(text4)
        self.assertEqual(result4['lang'], '')  # Empty string indicating no detected language
        self.assertEqual(result4['Lang_acc'], 0)


def test_pp_detect_language(self):
        # Test case 1: DataFrame with one English text column
        df1 = pd.DataFrame({'text': ['This is an English text.']})
        result_df1 = pp_detect_language(df1)
        self.assertTrue('lang' in result_df1.columns)
        self.assertTrue('Lang_acc' in result_df1.columns)
        self.assertEqual(result_df1['lang'][0], 'en')
        self.assertEqual(result_df1['Lang_acc'][0], 100)

        # Test case 2: DataFrame with multiple mixed language text columns
        df2 = pd.DataFrame({'text': ['This is an English text.', 'Das ist ein deutscher Text.', 'Bonjour! This is an English and French text.']})
        result_df2 = pp_detect_language(df2)
        self.assertTrue('lang' in result_df2.columns)
        self.assertTrue('Lang_acc' in result_df2.columns)
        self.assertEqual(result_df2['lang'][0], 'en')
        self.assertEqual(result_df2['lang'][1], 'de')
        self.assertIn(result_df2['lang'][2], ['en', 'fr'])  # Language could be either English or French
        self.assertEqual(result_df2['Lang_acc'][0], 100)
        self.assertEqual(result_df2['Lang_acc'][1], 100)
        self.assertEqual(result_df2['Lang_acc'][2], 100)

        # Test case 3: DataFrame with empty text column
        df3 = pd.DataFrame({'text': ['']})
        result_df3 = pp_detect_language(df3)
        self.assertTrue('lang' in result_df3.columns)
        self.assertTrue('Lang_acc' in result_df3.columns)
        self.assertEqual(result_df3['lang'][0], '')
        self.assertEqual(result_df3['Lang_acc'][0], 0)


class TestPPNumericLabelWithMissings(unittest.TestCase):

    def test_pp_numeric_label_with_missings(self):
        # Test case 1: DataFrame with labels mapped according to default mapping
        df1 = pd.DataFrame({'label': ['ft', 'mr', 'ct', 'pkg', 'ch', 'cnc']})
        result_df1 = pp_numeric_label_with_missings(df1)
        self.assertTrue('y' in result_df1.columns)
        self.assertEqual(result_df1['y'].tolist(), [0, 1, 2, 3, 4, 5])

        # Test case 2: DataFrame with labels mapped according to custom mapping
        custom_mapping = {'ft': 1, 'mr': 2, 'ct': 3, 'pkg': 4, 'ch': 5, 'cnc': 6}
        df2 = pd.DataFrame({'label': ['ft', 'mr', 'ct', 'pkg', 'ch', 'cnc']})
        result_df2 = pp_numeric_label_with_missings(df2, mapping=custom_mapping)
        self.assertTrue('y' in result_df2.columns)
        self.assertEqual(result_df2['y'].tolist(), [1, 2, 3, 4, 5, 6])

        # Test case 3: DataFrame with missing values mapped to default missing_class
        df3 = pd.DataFrame({'label': ['ft', 'mr', 'ct', 'pkg', None]})
        result_df3 = pp_numeric_label_with_missings(df3)
        self.assertTrue('y' in result_df3.columns)
        self.assertEqual(result_df3['y'].tolist(), [0, 1, 2, 3, -1])

        # Test case 4: DataFrame with missing values mapped to custom missing_class
        df4 = pd.DataFrame({'label': ['ft', 'mr', 'ct', 'pkg', None]})
        result_df4 = pp_numeric_label_with_missings(df4, missing_class=999)
        self.assertTrue('y' in result_df4.columns)
        self.assertEqual(result_df4['y'].tolist(), [0, 1, 2, 3, 999])

if __name__ == '__main__':
    unittest.main()
