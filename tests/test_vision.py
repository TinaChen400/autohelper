import unittest
import cv2
import numpy as np
from vision import Target, associate, glyph_state, label_name, scene_difference, stable_target


def line(text, x, y, confidence=.98):
    return ([[x, y], [x+100, y], [x+100, y+20], [x, y+20]], text, confidence)


class VisionTests(unittest.TestCase):
    def test_exclude_quick_replay_and_body_text(self):
        self.assertIsNone(label_name('2 Response A'))
        self.assertIsNone(label_name('The response A sounds better'))
        self.assertEqual(label_name('Generate speech using this voice:'), 'Input')
        self.assertEqual(label_name('Generate sptech using this voice:'), 'Input')
        self.assertEqual(label_name('Generate speech using thls voloe:'), 'Input')
        self.assertIsNone(label_name('Generate music using this voice:'))

    def test_text_growth_moves_headers_and_players_together(self):
        for offset in (0, 100, 250):
            lines = [line('Response A', 100, 100+offset), line('Response B', 600, 100+offset)]
            buttons = [((80, 270+offset, 32, 32), 'idle'), ((580, 270+offset, 32, 32), 'idle')]
            targets, _ = associate(lines, buttons, (1000, 1200, 3))
            self.assertEqual(targets['A'].button, buttons[0][0])
            self.assertEqual(targets['B'].button, buttons[1][0])

    def test_ambiguous_and_low_confidence_targets_not_returned(self):
        lines = [line('Response A', 100, 100), line('Response B', 600, 100, .5)]
        buttons = [((80, 270, 32, 32), 'idle'), ((110, 290, 32, 32), 'idle'), ((580, 270, 32, 32), 'idle')]
        targets, issues = associate(lines, buttons, (1000, 1200, 3))
        self.assertFalse(targets)
        self.assertTrue(issues)

    def test_does_not_cross_intervening_header(self):
        lines = [line('Input speech', 100, 100), line('Response A', 100, 200)]
        targets, _ = associate(lines, [((80, 300, 32, 32), 'idle')], (1000, 1200, 3))
        self.assertNotIn('Input', targets)
        self.assertIn('A', targets)

    def test_play_pause_and_unknown_glyphs(self):
        image = np.full((100, 100, 3), 255, np.uint8)
        cv2.circle(image, (50, 50), 16, (20, 20, 20), -1)
        cv2.fillConvexPoly(image, np.array([[45, 42], [45, 58], [57, 50]]), (255, 255, 255))
        self.assertEqual(glyph_state(image, (34, 34, 33, 33)), 'idle')
        cv2.circle(image, (50, 50), 16, (20, 20, 20), -1)
        cv2.rectangle(image, (44, 43), (47, 57), (255, 255, 255), -1)
        cv2.rectangle(image, (53, 43), (56, 57), (255, 255, 255), -1)
        self.assertEqual(glyph_state(image, (34, 34, 33, 33)), 'active')
        small = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
        self.assertEqual(glyph_state(small, (17, 17, 16, 16)), 'active')
        cv2.circle(image, (50, 50), 16, (20, 20, 20), -1)
        self.assertEqual(glyph_state(image, (34, 34, 33, 33)), 'unknown')

    def test_click_stability_checks_label_state_and_geometry(self):
        a = Target('A', (100, 100, 100, 20), (80, 270, 32, 32), 'idle', .99)
        b = Target('A', (100, 100, 100, 20), (80, 300, 32, 32), 'idle', .99)
        self.assertFalse(stable_target(a, b))
        self.assertTrue(stable_target(a, a))
        self.assertEqual(scene_difference(np.zeros((100, 100, 3), np.uint8), np.full((100, 100, 3), 20, np.uint8)), 20)


if __name__ == '__main__':
    unittest.main()
