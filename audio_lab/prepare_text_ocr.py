"""Download a pinned official RapidAI English recognizer and verify its hash."""
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

URL='https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx'
SHA256='e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318'
MODEL=Path(__file__).resolve().parent/'models'/'english-ocr'/'en_PP-OCRv4_rec_mobile.onnx'


def main():
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    temporary=MODEL.with_suffix('.download')
    urllib.request.urlretrieve(URL,temporary)
    digest=sha256(temporary.read_bytes()).hexdigest()
    if digest!=SHA256: raise RuntimeError('English OCR model SHA256 mismatch; not installed')
    temporary.replace(MODEL)
    MODEL.with_suffix('.source.json').write_text(json.dumps({'url':URL,'sha256':digest},indent=2),encoding='utf-8')
    print('Verified English OCR model installed.')


if __name__=='__main__':main()
