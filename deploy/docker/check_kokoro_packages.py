"""Run against an isolated release installation; never installs or downloads."""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--japanese', action='store_true')
    parser.add_argument('--applications', action='store_true')
    args = parser.parse_args()
    forbidden = ['unidic', 'torch', 'spacy-curated-transformers']
    if not args.japanese:
        forbidden.extend(['pyopenjtalk', 'unidic-lite', 'fugashi', 'jaconv', 'mojimoji'])
    for name in forbidden:
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        raise AssertionError(f'Unexpected distribution: {name}')
    for dist in metadata.distributions():
        direct = json.loads(dist.read_text('direct_url.json') or '{}')
        assert not direct.get('dir_info', {}).get('editable'), dist.metadata['Name']
    # Enforce offline smoke tests: the slim model must already be installed.
    assert importlib.util.find_spec('en_core_web_sm')
    from misaki import en, espeak, zh
    result = {'phonemes': {}}
    for route, british in [('a', False), ('b', True)]:
        frontend = en.G2P(trf=False, british=british,
                          fallback=espeak.EspeakFallback(british=british))
        phonemes, _ = frontend('Hello world. The quick brown fox jumps.')
        assert phonemes and '❓' not in phonemes
        assert frontend.nlp.pipe_names == ['tok2vec', 'tagger']
        result['phonemes'][route] = phonemes
    phonemes, _ = zh.ZHG2P()('你好，世界。今天是星期四。')
    assert phonemes and '❓' not in phonemes
    result['phonemes']['z'] = phonemes
    import onnxruntime
    from rknnlite.api import RKNNLite
    assert onnxruntime.__version__ == '1.29.0'
    result['rknn_import'] = RKNNLite.__name__
    if args.japanese:
        from misaki.ja import JAG2P
        dictionary = Path(os.environ.get('KOKORO_JA_DICDIR', ''))
        if not dictionary.is_absolute():
            raise AssertionError('KOKORO_JA_DICDIR must point to the read-only mounted dictionary')
        os.environ['MECABRC'] = str(dictionary / 'mecabrc')
        assert (dictionary / 'sys.dic').is_file(), dictionary
        frontend = JAG2P()
        assert frontend.version == 'cutlet'
        # Prove the Tagger selected the explicitly provisioned lite dictionary.
        assert Path(frontend.cutlet.tagger.dictionary_info[0]['filename']).parent == dictionary
        phonemes, _ = frontend('こんにちは、世界。')
        assert phonemes and '❓' not in phonemes
        result['phonemes']['j'] = phonemes
        result['unidic_lite_dictionary'] = {
            str(p.relative_to(dictionary)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(dictionary.rglob('*')) if p.is_file()
        }
    if args.applications:
        import rkvoice_stream
        import voxedge
        assert metadata.version('rkvoice-stream') == '0.2.0'
        assert metadata.version('voxedge') == '0.0.13a0'
        result['application_paths'] = [rkvoice_stream.__file__, voxedge.__file__]
    result['versions'] = dict(sorted((d.metadata['Name'], d.version)
                                    for d in metadata.distributions()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
