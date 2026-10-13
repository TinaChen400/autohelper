from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from audio_lab.task_context import capture_context


class ContextReuseTests(unittest.TestCase):
    def test_clear_english_body_reuses_pixels_but_weak_text_gets_new_crop_ocr(self):
        root=Path(__file__).resolve().parents[1]
        if not (root/'audio_lab/models/english-ocr/en_PP-OCRv4_rec_mobile.onnx').exists():
            self.skipTest('Local English OCR model not installed')
        def line(text,y,confidence):return (np.array([[10,y],[390,y],[390,y+20],[10,y+20]]),text,confidence)
        image=np.full((200,500,3),255,np.uint8)
        for score in (.99,.84):
            with self.subTest(score=score),tempfile.TemporaryDirectory(dir=root/'artifacts') as folder:
                lines=[line('Here is the transcript:',10,score),line('A clear complete target sentence.',40,score)]
                engine=Mock(return_value=([line('Here is the transcript: A corrected target sentence.',10,.99)],None))
                detector=SimpleNamespace(text_ocr=engine,english_page_ocr=True,last_task_lines=lines)
                result=capture_context(detector,image,(0,0,500,100),folder)
                if score>=.9:
                    engine.assert_not_called()
                    self.assertEqual(result['target_text'],'A clear complete target sentence.')
                else:
                    engine.assert_called_once()
                    self.assertEqual(result['target_text'],'A corrected target sentence.')
