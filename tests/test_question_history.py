import json
from pathlib import Path
import tempfile
import unittest
from audio_lab.question_history import find_history


class QuestionHistoryTests(unittest.TestCase):
    def test_same_text_different_audio_never_reuses_answer(self):
        with tempfile.TemporaryDirectory() as root:
            a,b=Path(root)/'a',Path(root)/'b'
            for folder in (a,b):
                folder.mkdir()
                (folder/'task_context.json').write_text(json.dumps({'target_text':'The same spoken sentence.'}))
                (folder/'session.json').write_text(json.dumps({'player_order':['A','B']}))
                for name in ('A','B'):(folder/f'{name}.wav').write_bytes(name.encode())
            (a/'comparison.json').write_text(json.dumps({'submission_observation':{'server_success_verified':True},
                'remote_ratings':{'quality':{'judgment':'A','source':'human_remote'}}}))
            self.assertEqual(find_history(b)['reused_judgments']['quality']['judgment'],'A')
            (b/'B.wav').write_bytes(b'other audio')
            result=find_history(b)
            self.assertTrue(result['matches'])
            self.assertEqual(result['reused_judgments'],{})
            (b/'B.wav').write_bytes(b'B')
            (a/'comparison.json').write_text(json.dumps({'remote_ratings':{'quality':{'judgment':'A','source':'model_auto'}}}))
            self.assertEqual(find_history(b)['reused_judgments'],{})
