"""Real screenshot regressions; no Windows clicks or desktop capture."""
from pathlib import Path
import unittest
import cv2
from vision import Detector, glyph_state


class RemoteRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = Detector()

    def fixture(self, name):
        path = Path(__file__).parent/'fixtures'/name
        image = cv2.imread(str(path))
        self.assertIsNotNone(image)
        return image

    def test_half_resolution_input_and_pause(self):
        image = self.fixture('scaled_player.png')
        targets, issues, _ = self.detector.detect(image)
        self.assertEqual(set(targets), {'Input', 'A', 'B'}, issues)
        self.assertEqual(targets['Input'].state, 'idle')
        self.assertEqual(targets['A'].state, 'active')
        self.assertEqual(targets['B'].state, 'idle')

    def test_visible_b_heading_survives_ocr_downscale_loss(self):
        image = self.fixture('native_heading_retry.png')
        targets, issues, _ = self.detector.detect(image)
        self.assertEqual(set(targets), {'Input', 'A', 'B'}, issues)
        self.assertEqual(targets['B'].state, 'idle')

    def test_five_second_timer_uses_alternate_resampling(self):
        from playback import read_players
        image = self.fixture('timer_resampling_retry.png')
        targets, issues, _ = self.detector.detect(image)
        self.assertEqual(set(targets), {'Input', 'A', 'B'}, issues)
        readings = read_players(self.detector, image, targets)
        self.assertEqual((readings['A'].current, readings['A'].total), (5, 5))
        self.assertEqual(readings['Input'].total, 7)
        self.assertEqual(readings['B'].total, 7)

    def test_first_play_locked_responses_are_located_but_not_idle(self):
        image = self.fixture('locked_first_play.png')
        targets, issues, _ = self.detector.detect(image)
        self.assertEqual(set(targets), {'Input', 'A', 'B'}, issues)
        self.assertEqual(targets['Input'].state, 'idle')
        self.assertEqual(targets['A'].state, 'locked')
        self.assertEqual(targets['B'].state, 'locked')

    def test_lock_icon_read_as_digit_does_not_hide_b_on_new_question(self):
        image=self.fixture('locked_header_digit.png')
        targets,issues,_=self.detector.detect(image)
        self.assertEqual(set(targets),{'Input','A','B'},issues)
        self.assertEqual(targets['Input'].state,'idle')
        self.assertEqual(targets['A'].state,'locked')
        self.assertEqual(targets['B'].state,'locked')

    def test_prerequisite_first_read_as_tirst_preserves_locked_b(self):
        image=self.fixture('locked_header_tirst.png')
        targets,issues,_=self.detector.detect(image)
        self.assertEqual(set(targets),{'Input','A','B'},issues)
        self.assertEqual(targets['Input'].state,'idle')
        self.assertEqual(targets['A'].state,'locked')
        self.assertEqual(targets['B'].state,'locked')

    def test_reported_screenshot_input_recovered_but_pointer_obstruction_rejected(self):
        image = self.fixture('reported_remote.png')
        targets, issues, _ = self.detector.detect(image)
        self.assertIn('Input', targets, issues)
        self.assertEqual(targets['Input'].state, 'idle')
        self.assertEqual(targets['B'].state, 'idle')
        # User's pointer covers A's icon. A screenshot cannot recover hidden pixels.
        self.assertNotIn('A', targets)
        self.assertNotEqual(glyph_state(image, (105, 335, 15, 15)), 'active')

    def test_locked_a_prerequisite_firsl_is_located_on_new_question(self):
        image=self.fixture('locked_header_firsl.png')
        targets,issues,_=self.detector.detect(image)
        self.assertEqual(set(targets),{'Input','A','B'},issues)
        self.assertEqual(targets['Input'].state,'idle')
        self.assertEqual(targets['A'].state,'locked')
        self.assertEqual(targets['B'].state,'locked')


if __name__ == '__main__':
    unittest.main()
