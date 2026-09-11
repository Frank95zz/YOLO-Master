"""Offline data/export/partition tests; no downloads or training."""


import pytest
from PIL import Image

from scripts.d1.prepare_visdrone import convert_annotation, prepare
from scripts.d1.evaluate_visdrone import METRICS, TOOLKIT_COMMIT, export_predictions, validate_official_report
from scripts.d1.cache_visdrone import batches


def test_annotation_flags_and_clipping():
    text = '-5,0,20,10,1,1,0,2\n0,0,20,20,0,4,0,0\n0,0,9,9,1,0,0,0\n0,0,9,9,0,11,0,0\n'
    label, side, count = convert_annotation(text, 100, 50)
    assert label.decode().split() == ['0', '0.0750000000', '0.1000000000', '0.1500000000', '0.2000000000']
    assert count['training_boxes'] == 1 and count['ignored_boxes'] == 3
    assert count['clipped_boxes'] == 1 and side[0]['original'][-1] == 2
    assert len(side) == 4


@pytest.mark.parametrize('text', ['nan,0,1,1,1,1,0,0', '0,0,-1,1,1,1,0,0',
                                     '0,0,1,1,1,1.5,0,0', '0,0,1,1,2,1,0,0', '0,0,1'])
def test_bad_annotations_fail(text):
    with pytest.raises(ValueError):
        convert_annotation(text, 100, 100)


def test_empty_and_zero_area():
    assert convert_annotation('', 32, 32)[0] == b''
    labels, side, stats = convert_annotation('100,100,5,5,1,2,0,0', 32, 32)
    assert labels == b'' and stats['dropped_degenerate'] == 1
    assert side[0]['disposition'] == 'degenerate_after_clip'


def make_dataset(tmp):
    source = tmp / 'original'
    for i, split in enumerate(('train', 'val', 'test-dev')):
        folder = source / f'VisDrone2019-DET-{split}'
        (folder / 'images').mkdir(parents=True)
        (folder / 'annotations').mkdir()
        Image.new('RGB', (32, 32), color=(i * 50, 0, 0)).save(folder / 'images' / f'id_{i}.jpg')
        (folder / 'annotations' / f'id_{i}.txt').write_text('0,0,10,10,1,1,0,0\n')
    return source


def test_prepare_is_stable_preserves_originals(tmp_path):
    source = make_dataset(tmp_path)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    target = tmp_path / 'prepared'
    counts = dict.fromkeys(('train', 'val', 'test-dev'), 1)
    first = prepare(source, target, counts=counts)
    assert first == prepare(source, target, counts=counts, verify=True)
    assert before == {p: p.read_bytes() for p in before}
    assert len((target / 'samples.jsonl').read_text().splitlines()) == 3
    assert first['training_ignore_background_mask'] is False
    (target / 'labels/visdrone-train/id_0.txt').write_text('corrupt')
    with pytest.raises(ValueError):
        prepare(source, target, counts=counts, verify=True)


def test_partition_keeps_tail_context():
    records = list(range(101))
    parts = [batches(records, r, 6, 16) for r in range(6)]
    assert sorted(v for part in parts for batch in part for v in batch) == records
    assert parts[0][0] == list(range(0, 96, 6))
    assert parts[0][-1] == [96]
    assert parts == [batches(records, r, 6, 16) for r in range(6)]


def test_export_empty_and_sorted(tmp_path):
    predictions = [{'image_id': 'a', 'category_id': 9, 'score': score, 'bbox': [1,2,3,4]}
                   for score in (0.2, 0.9, 0.0001)]
    export_predictions(predictions, ['a', 'b'], tmp_path)
    assert (tmp_path / 'b.txt').read_bytes() == b''
    assert (tmp_path / 'a.txt').read_text().splitlines() == ['1,2,3,4,0.9,10,-1,-1', '1,2,3,4,0.2,10,-1,-1']


@pytest.mark.parametrize('category,score,box', [(10,.1,[0,0,1,1]), (True,.1,[0,0,1,1]),
                                             (0,float('nan'),[0,0,1,1]), (0,.1,[0,0,-1,1])])
def test_invalid_export(tmp_path, category, score, box):
    with pytest.raises(ValueError):
        export_predictions([{'image_id':'a','category_id':category,'score':score,'bbox':box}], ['a'], tmp_path)


def test_official_scale():
    report = {'backend':'official-matlab', 'toolkit_commit':TOOLKIT_COMMIT,
              'metrics_percent':dict.fromkeys(METRICS, 20), 'metrics':dict.fromkeys(METRICS, .2)}
    assert validate_official_report(report) == report
    report['metrics']['AP_all'] = 20
    with pytest.raises(ValueError):
        validate_official_report(report)


def test_same_split_duplicate_bytes_are_preserved(tmp_path):
    source = make_dataset(tmp_path)
    train = source / 'VisDrone2019-DET-train'
    (train / 'images/id_repeat.jpg').write_bytes((train / 'images/id_0.jpg').read_bytes())
    (train / 'annotations/id_repeat.txt').write_text('')
    result = prepare(source, tmp_path / 'prepared', counts={'train':2, 'val':1, 'test-dev':1})
    assert result['splits']['train']['same_split_duplicate_image_bytes'] == [['id_0', 'id_repeat']]
    assert result['splits']['train']['images'] == 2


def test_cross_split_duplicate_bytes_rejected(tmp_path):
    source = make_dataset(tmp_path)
    first = source / 'VisDrone2019-DET-train/images/id_0.jpg'
    (source / 'VisDrone2019-DET-val/images/id_1.jpg').write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match='across official splits'):
        prepare(source, tmp_path / 'prepared', counts={'train':1,'val':1,'test-dev':1})


def test_npy_conversion_preserves_source_and_rejects_corruption(tmp_path):
    import torch
    from scripts.d1.cache_visdrone import convert_preserving_source
    from ultralytics.nn.foundation.cache import FeatureCacheWriter
    from ultralytics.nn.foundation.npy_cache import NpyFeatureCacheReader
    split = 'visdrone-train'
    source, out = tmp_path / split, tmp_path / 'npy'
    contract = {'model_id':'test','teacher_weights_sha256':'a'*64,'preprocessing_sha256':'b'*64,
                'output_layers':[4,8,12],'feature_names':['block4','block8','block12'],
                'dtype':'float16','expected_shape':[384,40,40]}
    writer = FeatureCacheWriter(source, split=split, contract=contract, shard_prefix=split+'-r00')
    feature = {name:torch.full((384,40,40), float(i), dtype=torch.float16)
               for i,name in enumerate(contract['feature_names'])}
    writer.add(sample_id=split+'/original_id', split=split, image_path='images/'+split+'/original_id.jpg',
               image_sha256='c'*64, features=feature)
    writer.close()
    before = {p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    first = convert_preserving_source(source, out)
    assert first == convert_preserving_source(source, out)
    assert before == {p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    reader = NpyFeatureCacheReader(out / split)
    reader.verify_sample(split+'/original_id')
    for name, value in reader.get(split+'/original_id').items():
        assert torch.equal(value, feature[name])
    path = out / split / 'original_id.npy'
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError):
        convert_preserving_source(source, out)


@pytest.mark.parametrize('failure', [None, 'missing_rank', 'bad_identity', 'partial', 'corrupt_shard'])
def test_finalize_fails_closed(tmp_path, monkeypatch, failure):
    import types
    import torch
    from scripts.d1 import cache_visdrone as cache
    from ultralytics.nn.foundation.cache import FeatureCacheWriter, verify_feature_cache
    from scripts.d1.cache_features import write_json
    contract = {'model_id':'test','teacher_weights_sha256':'a'*64,'preprocessing_sha256':'b'*64,
                'output_layers':[4,8,12],'feature_names':['block4','block8','block12'],
                'dtype':'float16','expected_shape':[1,2,2], 'schema_version':'d1-cache-v1'}
    records = [{'sample_id':f'visdrone-train/id{i}', 'split':'visdrone-train',
                'image_path':f'images/visdrone-train/id{i}.jpg','image_sha256':str(i)*64} for i in range(4)]
    monkeypatch.setattr(cache, 'rows', lambda *args: records)
    args = types.SimpleNamespace(output=tmp_path, data=tmp_path, batch=2)
    own = {'world_size':2,'contract':contract}
    for rank in range(2):
        part = tmp_path / 'parts/visdrone-train' / f'rank{rank:02d}'
        writer = FeatureCacheWriter(part, split='visdrone-train', contract=contract,
                                   shard_prefix=f'visdrone-train-r{rank:02d}')
        for record in records[rank::2]:
            writer.add(**record, features={name:torch.ones(1,2,2,dtype=torch.float16) for name in contract['feature_names']})
        writer.close()
        report = {'status':'PASSED','identity':own,'rank':rank,'split':'visdrone-train',
                  'batch_contexts':[[r['sample_id'] for r in records[rank::2]]], 'verification':verify_feature_cache(part)}
        if failure == 'bad_identity' and rank == 1:
            report['identity'] = {'different':'run'}
        if failure != 'missing_rank' or rank != 1:
            write_json(tmp_path / 'reports' / f'visdrone-train-r{rank:02d}.json', report)
        if failure == 'partial' and rank == 1:
            (part / '.unexpected.part').touch()
        if failure == 'corrupt_shard' and rank == 1:
            next(part.glob('*.safetensors')).write_bytes(b'corrupt')
    if failure:
        with pytest.raises((ValueError, FileNotFoundError)):
            cache.finalize(args, 'train', own)
        assert not (tmp_path / 'safetensors/visdrone-train/index.json').exists()
    else:
        result = cache.finalize(args, 'train', own)
        assert result['sample_count'] == 4 and result['tensor_count'] == 12
        assert cache.finalize(args, 'train', own) == result
