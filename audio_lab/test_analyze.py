import tempfile
import unittest
from pathlib import Path
import wave
import numpy as np
from analyze import analyze, word_error


class AnalysisTests(unittest.TestCase):
    def test_no_model_report_keeps_ratings_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('Input','A','B'):
                with wave.open(str(Path(directory)/f'{name}.wav'),'wb') as output:
                    output.setnchannels(2); output.setsampwidth(2); output.setframerate(16000)
                    output.writeframes(np.zeros((16000,2),dtype='<i2').tobytes())
            result = analyze(directory)
            self.assertEqual(result['clips']['A']['recording_checks']['duration_seconds'],1)
            self.assertEqual(result['clips']['B']['recording_checks']['rms_dbfs'],-240)
            self.assertTrue(all(v['choice'] is None for v in result['evaluation_draft'].values()))
            self.assertFalse(result['submitted'])
            self.assertTrue((Path(directory)/'report.html').exists())

    def test_target_text_difference_not_reference_speaker_text(self):
        self.assertEqual(word_error('Hello, WORLD!','hello world'),0)
        self.assertEqual(word_error('one two three','one three'),1/3)
        self.assertIsNone(word_error('','test'))
