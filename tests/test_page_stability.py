import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np

from workflow import Workflow,Settings,PageUnconfirmed


class PageStabilityTests(unittest.TestCase):
    def setUp(self):
        self.flow=object.__new__(Workflow)
        self.flow.settings=Settings();self.flow.locked_start=False
        self.flow.targets={n:SimpleNamespace(label=(20+100*i,20,80,25)) for i,n in enumerate(('Input','A','B'))}
        self.flow.identity=(20,65,280,100)
        self.flow.reference=np.random.default_rng(4).integers(0,256,(200,340,3),dtype=np.uint8)

    def test_shared_one_pixel_remote_settling_is_accepted(self):
        for dx,dy in ((0,-1),(0,1),(-1,0),(1,1)):
            with self.subTest(dx=dx,dy=dy):
                self.flow.check_page(np.roll(self.flow.reference,(dy,dx),axis=(0,1)))

    def test_larger_scroll_is_rejected_against_original_reference(self):
        self.flow.check_page(np.roll(self.flow.reference,1,axis=0))
        with self.assertRaisesRegex(RuntimeError,'内容或布局变化'):
            self.flow.check_page(np.roll(self.flow.reference,2,axis=0))

    def test_different_region_offsets_are_not_independently_accepted(self):
        image=self.flow.reference.copy()
        image[:50]=np.roll(image[:50],1,axis=0)
        image[60:170]=np.roll(image[60:170],-1,axis=0)
        with self.assertRaisesRegex(RuntimeError,'内容或布局变化'):
            self.flow.check_page(image)

    def test_changed_transcript_and_dimensions_still_stop(self):
        image=self.flow.reference.copy();image[65:165,20:300]=255
        for changed in (image,image[:-1]):
            with self.assertRaises(RuntimeError):self.flow.check_page(changed)

    def test_transient_occlusion_is_reobserved_without_input_during_playback(self):
        changed=self.flow.reference.copy();changed[65:165,20:300]=255
        self.flow.snapshot=Mock(side_effect=[changed,changed,self.flow.reference])
        self.flow.mouse=Mock();self.flow.log=Mock();self.flow.status=Mock()
        self.flow.observing_playback=True
        image=self.flow.page_snapshot()
        self.assertIs(image,self.flow.reference)
        self.assertEqual(self.flow.snapshot.call_count,3)
        self.assertEqual(self.flow.mouse.wait_observation.call_count,2)
        self.flow.mouse.wait.assert_not_called();self.flow.mouse.move.assert_not_called()

    def test_persistent_changed_page_is_bounded_and_dimensions_are_not_retried(self):
        changed=self.flow.reference.copy();changed[65:165,20:300]=255
        self.flow.snapshot=Mock(return_value=changed)
        self.flow.mouse=Mock();self.flow.log=Mock();self.flow.status=Mock()
        with self.assertRaises(PageUnconfirmed):self.flow.page_snapshot()
        self.assertEqual(self.flow.snapshot.call_count,4)
        self.flow.snapshot=Mock(return_value=changed[:-1])
        with self.assertRaisesRegex(RuntimeError,'尺寸变化'):self.flow.page_snapshot()
        self.flow.snapshot.assert_called_once()


if __name__=='__main__':unittest.main()
