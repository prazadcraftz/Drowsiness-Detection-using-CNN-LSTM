"""Benchmark correctness without fragile performance thresholds."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import benchmark as cli
from driver_monitor.metrics.benchmark import (
    ROOT, STAGES, ProcessTelemetry, SampleBook, aggregate_runs, budget_50ms,
    distribution_ms, environment, instrumentation, memory_trend, ordered_jobs,
    runtime_imports, thread_settings, worker, write_new_json,
)


def test_percentiles_and_population_std():
    stats=distribution_ms([0,10,20,30,40])
    assert stats['sample_count']==5 and stats['mean_ms']==stats['p50_ms']==20
    assert stats['p90_ms']==36 and stats['p95_ms']==38 and stats['p99_ms']==39.6
    assert stats['min_ms']==0 and stats['max_ms']==40
    assert stats['std_ms']==pytest.approx(np.std([0,10,20,30,40]))


@pytest.mark.parametrize('values',[[],[-1],[float('nan')],[float('inf')],[True],['1']])
def test_empty_or_invalid_rejected_not_discarded(values):
    with pytest.raises(ValueError):distribution_ms(values)


def test_one_sample_and_raw_order_preserved():
    values=[.123456789]
    stats=distribution_ms(values)
    assert values==[.123456789] and stats['std_ms']==0
    assert all(stats[key]==values[0] for key in ('mean_ms','p50_ms','p90_ms','p95_ms','p99_ms','min_ms','max_ms'))


def test_50ms_inclusive_boundary():
    stats=budget_50ms([49,50,51,100])
    assert stats['percent_at_or_below_50ms']==50
    assert stats['p95_to_50ms_ratio']==pytest.approx(1.853)


def test_warmup_sampled_valid_observations_not_decoded_frames():
    from driver_monitor.results import InferenceResult,ObservationResult
    book=SampleBook(2)
    for i,(face,sampled) in enumerate([(False,False),(True,True),(True,False),(True,True),(False,False),(True,True)]):
        book.frame_index=i
        book.add('frame_decode',i+1)
        book.classify(InferenceResult(ObservationResult(face,None,None,1,False,timestamp_seconds=i/20,sampled=sampled)))
    assert [r['steady'] for r in book.frames]==[False,False,False,False,True,True]
    exported=book.export()['frame_decode']
    assert exported['raw_elapsed_ns']==[1,2,3,4,5,6]
    assert exported['steady_state']['sample_count']==2 and exported['warmup_sample_count']==4
    json.dumps(book.export(),allow_nan=False)


def test_timer_uses_exact_clock_no_rounding_and_finally_records():
    ticks=iter([10,23,30,47])
    book=SampleBook(0,clock=lambda:next(ticks))
    with book.measure('frame_decode'):pass
    with pytest.raises(RuntimeError):
        with book.measure('frame_decode'):raise RuntimeError('test')
    assert book.samples['frame_decode']==[(0,13),(0,17)]


def test_stage_names_consistent_and_no_fake_empty_percentiles():
    book=SampleBook()
    assert set(book.samples)==set(STAGES) and len(STAGES)==len(set(STAGES))
    assert book.export()=={}
    with pytest.raises(ValueError):book.add('fake_inference',1)
    with pytest.raises(ValueError):book.add('frame_decode',-1)


@pytest.mark.parametrize('value',[-1,True,1.5])
def test_bad_warmup(value):
    with pytest.raises(ValueError):SampleBook(value)


def test_seeded_order_is_reproducible_and_complete():
    a=ordered_jobs(['a','b'],5,seed=12)
    assert a==ordered_jobs(['a','b'],5,seed=12)
    assert a!=ordered_jobs(['a','b'],5,seed=13)
    assert len(a)==20
    assert len({(r['source'],r['run'],r['mode']) for r in a})==20


@pytest.mark.parametrize('sources,runs,modes',[([],5,('headless',)),(['a'],0,('headless',)),(['a'],1,()),(['a'],1,('bad',))])
def test_bad_order_config(sources,runs,modes):
    with pytest.raises(ValueError):ordered_jobs(sources,runs,modes)


def test_json_exclusive_write_preserves_raw_samples(tmp_path):
    output=tmp_path/'nested'/'result.json'
    payload={'raw_elapsed_ns':[1,123456789],'stats':distribution_ms([.1,.2])}
    write_new_json(output,payload)
    assert json.loads(output.read_text())==payload
    before=output.read_bytes()
    with pytest.raises(FileExistsError):write_new_json(output,{'overwritten':True})
    assert output.read_bytes()==before
    with pytest.raises(ValueError):write_new_json(ROOT/'tests'/'fixtures'/'never.json',payload)


def test_cli_existing_output_rejected_before_runtime_or_source_loading(tmp_path,monkeypatch):
    output=tmp_path/'old.json';output.write_text('keep')
    monkeypatch.setattr(cli,'runtime_imports',lambda:pytest.fail('Existing output loaded runtime'))
    with pytest.raises(FileExistsError):cli.run_benchmarks(['missing'],output=output)
    assert output.read_text()=='keep'


def test_rss_sample_handling_and_descriptive_trend(monkeypatch):
    monkeypatch.setattr(ProcessTelemetry,'memory',staticmethod(lambda:{'rss_bytes':4096,'peak_rss_bytes':8192}))
    telemetry=ProcessTelemetry()
    telemetry.sample('startup',0);telemetry.sample('processing',100)
    result=telemetry.summary()
    assert len(result['samples'])==2 and result['samples'][0]['rss_bytes']==4096
    assert memory_trend(result['samples'])['net_change_bytes']==0
    assert memory_trend([])['assessment']=='insufficient memory samples'
    rows=[{'processed_frames':i,'rss_bytes':i*10} for i in range(1,4)]
    assert memory_trend(rows)['linear_slope_bytes_per_frame']==10
    assert memory_trend(rows)['strictly_monotonic_growth']


def test_real_environment_contains_read_only_threads_and_versions():
    runtime=runtime_imports()
    before=thread_settings(runtime)
    info=environment(runtime)
    assert thread_settings(runtime)==before
    assert info['device']=='cpu' and info['software']['torch']==torch.__version__
    assert info['thread_settings']==before and before['logical_cpu_count']>=1
    assert info['cpu_model'] and info['os'] and info['python_executable']
    assert 'background_conditions' in info


def test_instrumentation_forwards_raw_exactly_and_restores(frozen_phase2):
    from driver_monitor.inference.engine import DrowsinessInferenceEngine
    from driver_monitor.inference.decision import DecisionLayer
    _,arrays=frozen_phase2
    with DrowsinessInferenceEngine(initialize_detector=False) as engine:
        engine.detector=SimpleNamespace(detect=lambda frame:None,close=lambda:None)
        layer=DecisionLayer(engine.config)
        original_tensor=torch.tensor
        original_process=engine.process_frame
        book=SampleBook(0)
        with instrumentation(engine,layer,book):
            raw=engine.infer_sequence(arrays['sequences'][0])
            np.testing.assert_array_equal(raw.logits,arrays['logits'][0])
            np.testing.assert_array_equal(raw.probabilities,arrays['probabilities'][0])
            assert book.samples['cnn_lstm_forward'] and book.samples['softmax'] and book.samples['tensor_creation']
        assert torch.tensor is original_tensor and engine.process_frame==original_process
        with pytest.raises(RuntimeError):
            with instrumentation(engine,layer,book):raise RuntimeError('restore test')
        assert torch.tensor is original_tensor and engine.process_frame==original_process


def test_aggregate_distribution_and_between_run_variability():
    samples=[{'job':{'kind':'model_only'},'stages':{'model_only_forward':{'raw_elapsed_ns':[1_000_000,3_000_000]}}},
             {'job':{'kind':'model_only'},'stages':{'model_only_forward':{'raw_elapsed_ns':[4_000_000]}}}]
    aggregated=aggregate_runs(samples)[0]
    assert aggregated['distribution']['sample_count']==3
    assert aggregated['per_run_median_ms']==[2,4] and aggregated['between_run_median_std_ms']==1


def test_unknown_worker_kind_fails():
    with pytest.raises(ValueError):worker({'kind':'unknown'})
