import json
from pathlib import Path
import tempfile
import unittest
import wave
from types import SimpleNamespace
import cv2
import numpy as np
from audio_lab.decision_engine import build_decisions,quality_candidate,pronunciation_candidate,align,tokens
from audio_lab.task_context import context_from_lines,waveform_profiles,response_identity_evidence
from audio_lab.comparison_record import model_judgments


class DecisionTests(unittest.TestCase):
    def test_played_waveform_color_change_does_not_block_same_question(self):
        fixtures=Path(__file__).parent/'fixtures'
        geometry=json.loads((fixtures/'waveform_pale_geometry.json').read_text())
        before=cv2.imread(str(fixtures/'waveform_pale_before.png'))
        after=cv2.imread(str(fixtures/'waveform_pale_after.png'))
        targets={side:{name:SimpleNamespace(button=tuple(box)) for name,box in values.items()}
                 for side,values in geometry.items()}
        stored=waveform_profiles(before,targets['before'],('A','B'))
        self.assertEqual(response_identity_evidence(after,targets['after'],stored),'same')
        other=cv2.imread(str(fixtures/'waveform_before.png'))
        other_targets={'A':SimpleNamespace(button=(209,437,30,30)),
                       'B':SimpleNamespace(button=(965,437,30,30))}
        self.assertNotEqual(response_identity_evidence(other,other_targets,stored),'same')

    def test_same_audio_waveform_survives_scrolling_and_ocr_label_offsets(self):
        profiles=[]
        for filename,y in [('waveform_before.png',437),('waveform_reopened.png',562)]:
            image=cv2.imread(str(Path(__file__).parent/'fixtures'/filename))
            targets={'A':SimpleNamespace(button=(209,y,30,30),label=(230,0,80,15)),
                     'B':SimpleNamespace(button=(965,y,30,30),label=(947,0,120,15))}
            profiles.append(waveform_profiles(image,targets,('A','B')))
        for name in ('A','B'):
            self.assertGreater(float(np.corrcoef(profiles[0][name],profiles[1][name])[0,1]),.85)

    def fixture(self,folder,reference=True):
        folder=Path(folder)
        for index,name in enumerate(('A','B')):
            signal=(np.sin(np.arange(16000)*2*np.pi*(200+index*17)/16000)*6000).astype('<i2')
            with wave.open(str(folder/f'{name}.wav'),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(signal.tobytes())
        def save(name,data): (folder/name).write_text(json.dumps(data),encoding='utf-8')
        save('session.json',{'player_order':['Input','A','B'] if reference else ['A','B']})
        save('task_context.json',{'target_text':'The little cat was sleeping beside the window.','supported_english_text':True})
        quality={name:{'MOS_OVRL':3.5,'MOS_SIG':4,'MOS_DISC':4,'MOS_NOISE':4} for name in ('A','B')}
        report={'transcripts':{name:'The little cat was sleeping beside the window.' for name in ('A','B')},
                'quality':quality,'speaker_similarity':{'A':.91,'B':.7}}
        save('model_comparison.json',report)
        save('naturalness.json',{'scores':{'A':{'mos':3},'B':{'mos':3.1}}})
        return report

    def test_context_reads_only_current_page_target_not_buttons(self):
        def line(text,y):return ([[0,y],[500,y],[500,y+15],[0,y+15]],text,.99)
        result=context_from_lines([line('You should adjust emotion. Here is the transcript: Today the rain stopped.',10),
            line('Translate to English',35),line('Response A',80)])
        self.assertEqual(result['target_text'],'Today the rain stopped.')
        self.assertTrue(result['supported_english_text'])
        self.assertFalse(context_from_lines([line('Response A Response B',10)])['supported_english_text'])
        delivery=context_from_lines([line('Here is the transcript: I love this idea. (Delivery: angry).',10)])
        self.assertNotIn('angry',delivery['target_text'])
        self.assertIn('angry',delivery['instructions'])
        repeated=context_from_lines([line('Adjust based on the transcript. Here is the transcript: The rain has stopped.',10)])
        self.assertEqual(repeated['target_text'],'The rain has stopped.')
        icon=context_from_lines([line('Here is the transcript: The rain has stopped.',10),line('*ATranslate to English',40)])
        self.assertEqual(icon['target_text'],'The rain has stopped.')
        glued=context_from_lines([line('Here is the transcript: Thelittlecatwaswalkingbesidethewindow. Thesunisrising.',10)])
        self.assertFalse(glued['supported_english_text'])

    def test_pronunciation_is_target_dependent_and_numeric_punctuation_normalized(self):
        self.assertEqual(tokens("He's ready in 2024."),tokens('He is ready in two thousand twenty four'))
        transcripts={'A':'The cat sat beside the window.','B':'The dog ran.'}
        first,_=pronunciation_candidate('The cat sat beside the window.',transcripts)
        second,_=pronunciation_candidate('The dog ran.',transcripts)
        self.assertEqual(first['judgment'],'A');self.assertEqual(second['judgment'],'B')
        with self.assertRaises(ValueError):pronunciation_candidate('',transcripts)

    def test_quality_does_not_treat_small_mos_gap_as_defect(self):
        scores={name:{'MOS_OVRL':3.4 if name=='A' else 3.7,'MOS_SIG':4,'MOS_DISC':4,'MOS_NOISE':4} for name in ('A','B')}
        metrics={name:{'clipped_run_fraction':0,'internal_digital_zero_gaps_seconds':[]} for name in ('A','B')}
        result,_=quality_candidate(scores,metrics)
        self.assertEqual(result['judgment'],'Both Good')
        metrics['B']['clipped_run_fraction']=.05
        result,_=quality_candidate(scores,metrics)
        self.assertEqual(result['judgment'],'A')

    def test_alphanumeric_spacing_is_not_a_content_error(self):
        self.assertEqual(tokens('CYP2D6'),tokens('CYP 2D6'))
        self.assertEqual(tokens('B12 and H2O'),tokens('B 12 and H 2 O'))

    def test_units_negative_numbers_and_joined_labels_preserve_spoken_content(self):
        self.assertEqual(tokens('isAccess point: 5 GHz. Signal strength:-61dBm.'),
                         tokens('is AccessPoint five gigahertz signal strength minus sixty one DBM'))
        self.assertEqual(tokens('2.5GHz'),tokens('two point five giga hertz'))
        self.assertNotEqual(tokens('-61dBm'),tokens('61dBm'))
        target='The Wi-Fi diagnostic for the Harborview Medical Center east wing isAccess point: Studio_5G.Band: 5 GHz.Signal strength:-61dBm.'
        primary={'A':'The Wi-Fi diagnostic for the Harborview Medical Center East Wing is AccessPoint. Studio underscore 5G. Band 5 gigahertz. Signal strength minus 61 DBM.',
                 'B':'The Wi-Fi diagnostic for the Harborview Medical Center East Wing is Access Point, Studio 5G, banned 5 gigahertz, signal strength, minus 61 DBM.'}
        candidate,details=pronunciation_candidate(target,primary)
        self.assertEqual(candidate['judgment'],'Both Good')
        self.assertLess(max(r['word_error_rate'] for r in details.values()),.1)

    def test_shared_special_term_asr_errors_do_not_vote_both_bad(self):
        result,_=pronunciation_candidate('No atenolol is not metabolized by CYP2D6.',{
            'A':'No, a tenelol is not metabolized by SIP 2D6.',
            'B':'No, a tenelol is not metabolized by CYP 2D6.'})
        self.assertEqual(result['judgment'],'需复核')
        self.assertTrue(result['evidence']['requires_listening_review'])
        self.assertTrue(result['evidence']['shared_edits'])

    def test_unresolved_asr_errors_do_not_drive_overall_preference(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);report=self.fixture(folder)
            (folder/'task_context.json').write_text(json.dumps({'target_text':'No atenolol is not metabolized by CYP2D6.',
                'supported_english_text':True}))
            report['transcripts']={'A':'No, a tenelol is not metabolized by SIP.',
                                   'B':'No, a tenelol is not metabolized by CYP 2D6.'}
            (folder/'model_comparison.json').write_text(json.dumps(report))
            result=build_decisions(folder)
            self.assertEqual(result['judgments']['pronunciation']['judgment'],'需复核')
            self.assertNotEqual(result['judgments']['overall']['dominant_factor'],'pronunciation')
            loaded=model_judgments(folder)
            self.assertEqual(loaded['pronunciation']['judgment'],'需复核')
            self.assertEqual(loaded['quality']['judgment'],result['judgments']['quality']['judgment'])
            self.assertEqual(loaded['overall']['judgment'],result['judgments']['overall']['judgment'])

    def test_complete_candidates_and_factor_flip_when_responses_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);report=self.fixture(folder)
            first=build_decisions(folder)
            self.assertTrue(all(value['judgment']!='需复核' for value in first['judgments'].values()))
            self.assertEqual(first['judgments']['overall']['judgment'],'A')
            self.assertEqual(first['judgments']['overall']['dominant_factor'],'speaker')
            report['speaker_similarity']={'A':.7,'B':.91}
            (folder/'model_comparison.json').write_text(json.dumps(report))
            second=build_decisions(folder)
            self.assertEqual(second['judgments']['overall']['judgment'],'B')
            self.assertEqual(model_judgments(folder)['overall']['dominant_factor'],'speaker')

    def test_ab_mode_excludes_reference_similarity(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,reference=False)
            result=build_decisions(directory)
            self.assertEqual(result['judgments']['speaker']['judgment'],'不适用')
            self.assertEqual(result['judgments']['overall']['dominant_factor'],'naturalness')

    def test_material_content_error_drives_overall_and_factor(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);report=self.fixture(folder)
            report['transcripts']['A']='A completely unrelated sentence.'
            (folder/'model_comparison.json').write_text(json.dumps(report))
            result=build_decisions(folder)
            self.assertEqual(result['judgments']['overall']['judgment'],'B')
            self.assertEqual(result['judgments']['overall']['dominant_factor'],'pronunciation')

    def test_missing_target_or_model_does_not_fabricate_complete_result(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);self.fixture(folder)
            (folder/'task_context.json').write_text(json.dumps({'supported_english_text':False}))
            with self.assertRaises(ValueError):build_decisions(folder)
            self.assertFalse((folder/'evaluation_decisions.json').exists())


if __name__=='__main__':unittest.main()
