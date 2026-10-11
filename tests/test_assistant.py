import unittest
from assistant import absolute_box, signature, App


class CalibrationTests(unittest.TestCase):
    def test_button_coordinates_relative_to_remote_viewport(self):
        config = {'viewport': [.1, .2, .8, .6], 'buttons': {'A': {'box': [.25, .5, .1, .2]}}}
        for actual, expected in zip(absolute_box(config, 'A'), [.3, .5, .08, .12]):
            self.assertAlmostEqual(actual, expected)

    def test_signature_is_resolution_independent_for_uniform_region(self):
        small = (30, 20, bytes([10, 20, 30]) * 600)
        large = (60, 40, bytes([10, 20, 30]) * 2400)
        self.assertEqual(signature(small, [.2, .2, .5, .5]), signature(large, [.2, .2, .5, .5]))

    def test_playback_classification_rejects_unrelated_and_identical_samples(self):
        app = App.__new__(App)
        app.config = {'viewport': [0, 0, 1, 1], 'buttons': {'A': {'box': [0, 0, 1, 1], 'idle': [0]*1728, 'active': [100]*1728}}}
        def frame(value): return (24, 24, bytes([value])*1728)
        self.assertEqual(app.classify('A', frame(0))[0], 'idle')
        self.assertEqual(app.classify('A', frame(100))[0], 'active')
        self.assertEqual(app.classify('A', frame(240))[0], 'unknown')
        app.config['buttons']['A']['active'] = [0]*1728
        self.assertEqual(app.classify('A', frame(0))[0], 'invalid')


if __name__ == '__main__':
    unittest.main()
