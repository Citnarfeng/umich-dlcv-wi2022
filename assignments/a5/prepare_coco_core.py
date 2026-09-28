# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
# Reconstructed COCO2014 subset differs from the unavailable original course data.
import collections
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import random
import re
import time
import urllib.request
import zipfile
import numpy as np
from PIL import Image
import torch
ANNOTATION_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'

def _sha(data):
    return hashlib.sha256(data).hexdigest()

def _save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)

def _tokens(caption):
    return re.findall("[a-z0-9]+(?:'[a-z0-9]+)?", caption.lower())[:15]

def _validate(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    for split, count in [('train', 10000), ('val', 500)]:
        images, captions = (data[split + '_images'], data[split + '_captions'])
        assert images.shape == (count, 3, 112, 112) and images.dtype == torch.uint8
        assert captions.shape == (count, 17) and captions.dtype == torch.int64
        assert int(captions.min()) >= 0 and int(captions.max()) < 864
    vocab = data['vocab']
    assert isinstance(vocab['idx_to_token'], list) and len(vocab['idx_to_token']) == 864
    assert len(vocab['token_to_idx']) == 864
    assert all((vocab['token_to_idx'][word] == i for i, word in enumerate(vocab['idx_to_token'])))
    assert all((word in vocab['token_to_idx'] for word in ['<NULL>', '<START>', '<END>', '<UNK>']))

def prepare_coco(path='./datasets/coco.pt'):
    path = Path(path)
    if path.exists():
        _validate(path)
        return path
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        _prepare(path)
    finally:
        torch.set_num_threads(previous_threads)
    return path

def _prepare(path):
    cache = path.parent / 'coco_source'
    resized = cache / 'resized'
    resized.mkdir(parents=True, exist_ok=True)
    archive = cache / 'annotations_trainval2014.zip'
    if not archive.exists():
        print('Downloading official COCO2014 annotations.', flush=True)
        temporary = archive.with_suffix('.partial')
        urllib.request.urlretrieve(ANNOTATION_URL, temporary)
        assert temporary.stat().st_size == 252872794, 'Incomplete annotations download'
        temporary.replace(archive)
    assert archive.stat().st_size == 252872794
    sources = {'annotation_url': ANNOTATION_URL, 'annotation_bytes': archive.stat().st_size, 'annotation_sha256': _sha(archive.read_bytes()), 'caption_files': {}}
    records = []
    with zipfile.ZipFile(archive) as z:
        for split, count in [('train', 10000), ('val', 500)]:
            name = 'annotations/captions_' + split + '2014.json'
            raw = z.read(name)
            sources['caption_files'][name] = {'sha256': _sha(raw), 'bytes': len(raw)}
            data = json.loads(raw)
            images = {im['id']: im for im in data['images']}
            captions = collections.defaultdict(list)
            for ann in data['annotations']:
                captions[ann['image_id']].append(ann)
            rng = random.Random(0)
            image_ids = rng.sample(sorted(images.keys() & captions.keys()), count)
            for row, image_id in enumerate(image_ids):
                image = images[image_id]
                ann = rng.choice(sorted(captions[image_id], key=lambda item: item['id']))
                records.append({'split': split, 'row': row, 'image_id': image_id, 'caption_id': ann['id'], 'caption': ann['caption'], 'tokens': _tokens(ann['caption']), 'image_url': 'http://images.cocodataset.org/' + split + '2014/' + image['file_name'], 'original_size': [image['height'], image['width']]})
    freq = collections.Counter((t for r in records if r['split'] == 'train' for t in r['tokens']))
    words = ['<NULL>', '<START>', '<END>', '<UNK>'] + sorted(freq, key=lambda t: (-freq[t], t))[:860]
    vocab = {'idx_to_token': words, 'token_to_idx': {word: index for index, word in enumerate(words)}}
    assert len(words) == 864 and len({r['image_id'] for r in records}) == 10500
    _save_json(cache / 'sources.json', sources)
    _save_json(cache / 'selection_manifest.json', records)
    _save_json(cache / 'vocab.json', vocab)

    def get_image(record):
        stem = record['split'] + '_' + str(record['image_id'])
        pixels_path, meta_path = (resized / (stem + '.bin'), resized / (stem + '.json'))
        if pixels_path.exists() and pixels_path.stat().st_size == 3 * 112 * 112 and meta_path.exists():
            pixels, details = (pixels_path.read_bytes(), json.loads(meta_path.read_text()))
            if _sha(pixels) == details['resized_sha256']:
                return (record, pixels, details)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(record['image_url'], timeout=45) as response:
                    raw = response.read()
                with Image.open(io.BytesIO(raw)) as image:
                    rgb = image.convert('RGB').resize((112, 112), Image.Resampling.BILINEAR)
                    pixels = np.asarray(rgb, dtype=np.uint8).transpose(2, 0, 1).copy().tobytes()
                details = {'source_sha256': _sha(raw), 'source_bytes': len(raw), 'resized_sha256': _sha(pixels)}
                pixels_path.write_bytes(pixels)
                _save_json(meta_path, details)
                return (record, pixels, details)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))
    image_arrays = {s: np.empty((n, 3, 112, 112), dtype=np.uint8) for s, n in [('train', 10000), ('val', 500)]}
    caption_arrays = {s: np.zeros((n, 17), dtype=np.int64) for s, n in [('train', 10000), ('val', 500)]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        pending = [pool.submit(get_image, record) for record in records]
        for done, future in enumerate(concurrent.futures.as_completed(pending), 1):
            record, pixels, details = future.result()
            record.update(details)
            split, row = (record['split'], record['row'])
            image_arrays[split][row] = np.frombuffer(pixels, dtype=np.uint8).reshape(3, 112, 112)
            ids = [1] + [vocab['token_to_idx'].get(token, 3) for token in record['tokens']] + [2]
            caption_arrays[split][row, :len(ids)] = ids
            if done % 250 == 0:
                print(f'Prepared {done}/10500 COCO images.', flush=True)
    data = {'vocab': vocab}
    for split in ['train', 'val']:
        data[split + '_images'] = torch.from_numpy(image_arrays[split])
        data[split + '_captions'] = torch.from_numpy(caption_arrays[split])
    temporary = path.with_suffix('.partial')
    torch.save(data, temporary)
    _validate(temporary)
    temporary.replace(path)
    _save_json(cache / 'selection_manifest.json', records)
    _save_json(cache / 'result.json', {'bytes': path.stat().st_size, 'sha256': _sha(path.read_bytes()), 'complete': True})
