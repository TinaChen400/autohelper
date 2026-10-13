from pathlib import Path
import unittest
from unittest.mock import Mock
import cv2
import numpy as np
from rating_reader import question_marker
from vision import Detector


class MarkerRefinementTests(unittest.TestCase):
    def test_original_pixels_correct_reduced_page_digit_error(self):
        image=cv2.imread(str(Path(__file__).parent/'fixtures/question_marker_24062.png'))
        lines=[(np.array([[20,8],[248,8],[248,28],[20,28]]),'Prompt 10773526 Run 24962',.99)]
        result=Detector().refine_marker_lines(image,lines)
        self.assertEqual(dict(question_marker(result)),{'prompt':'10773526','run':'24062'})

    def test_failed_refinement_retains_existing_evidence(self):
        detector=object.__new__(Detector)
        detector.ui_lines=Mock(return_value=[])
        lines=[(np.array([[10,10],[90,10],[90,20],[10,20]]),'Prompt 10773526 Run 24062',.99)]
        result=detector.refine_marker_lines(np.zeros((40,100,3),np.uint8),lines)
        self.assertIs(result,lines)

    def test_misread_prompt_label_is_recovered_using_neighboring_run_field(self):
        image=cv2.imread(str(Path(__file__).parent/'fixtures/question_marker_670528.png'))
        lines=[(np.array([[58,8],[271,8],[271,30],[58,30]]),'Pzompt #670528Run #20782',.95)]
        result=Detector().refine_marker_lines(image,lines)
        self.assertEqual(dict(question_marker(result)),{'prompt':'670528','run':'20782'})

if __name__=='__main__':unittest.main()
