"""Audit and report the complete Phase 14 development window-length experiment."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import CLASS_NAMES
from evaluation.windows import (LENGTHS,STRIDES,ARCHITECTURES,SEEDS,COUNTS,RULES,
    P12,P13,verify,load,run_directory,diagnostics,temporal,padding_stats)
from evaluation.architectures import summary,paired_summary,pareto_front,latency_statistics
from evaluation.manifest import sha256,write_json,read_manifest
from evaluation.metrics import classification_metrics,aggregate_videos
from evaluation.balancing import guard_sources
from tools.report_phase12 import csv_out,table

D = ROOT/'results/phase14'


def analyze():
    folds,old,config,seal = verify(D)
    audit = load(D/'test_access_audit.json')
    assert audit['completed_new_runs']==108 and audit['reused_controls']==36
    assert audit['phase11_test_model_inference_passes']==audit['test_cache_reads']==audit['blocked_open_attempts']==0
    entries = [json.loads(s) for s in (D/'candidate_access_audit.jsonl').read_text().splitlines()]
    assert len(entries)==len({(r['architecture'],r['window'],r['fold'],r['seed']) for r in entries})==108
    assert sha256(D/'candidate_access_audit.jsonl')==audit['candidate_access_log_sha256']
    for e in entries:
        fold = folds['folds'][e['fold']]
        assert e['training_sources']==fold['train_source_ids'] and e['validation_inference_sources']==fold['validation_source_ids']
        guard_sources(e['training_sources']+e['validation_inference_sources'],folds['development_source_ids'],folds['excluded_test_source_ids'])
        assert e['test_inference_passes']==0
    assert not list(D.rglob('failure.json')) and not list(D.rglob('*test_predictions*'))
    assert sha256(D/'window_spec.json')==load(D/'experiment_started.json')['window_spec_sha256']
    arrays = {}
    for n in LENGTHS:
        for f in range(3):
            if n==16:
                arrays[n,f] = old[f]['validation']
            else:
                with np.load(D/f'windows_{n}/fold_{f}/validation.npz',allow_pickle=False) as a:
                    arrays[n,f] = {k:a[k] for k in a.files}
    rows,classes,metrics = {},[],{}
    from sklearn.metrics import f1_score,confusion_matrix
    for name in ARCHITECTURES:
        for n in LENGTHS:
            for f in range(3):
                for seed in SEEDS:
                    path = run_directory(D,name,n,f,seed)
                    m,h,c = [load(path/k) for k in ('metrics.json','history.json','config.json')]
                    assert c['architecture']==name and c['fold']==f and c['seed']==seed
                    expected_recipe = dict(load(P13/'experiment_config.json')['recipe'],seed=seed) if n==16 else dict(
                        load(D/'window_spec.json')['recipe'],seed=seed,window_length=n,window_stride=STRIDES[n])
                    assert c['recipe']==expected_recipe
                    assert c['architecture_spec_sha256']==config['architecture_spec_sha256']
                    assert c['normalization_sha256']==sha256(P12/f'fold_{f}/train_norm.npy')
                    assert c['initial_weights_sha256']==load(P13/name/f'fold_{f}/seed_{seed}/config.json')['initial_weights_sha256']
                    assert m['checkpoint_sha256']==sha256(path/'best.pt')
                    assert m['checkpoint_bytes']==(path/'best.pt').stat().st_size and m['trainable_parameters']==COUNTS[name]
                    assert m['test_inference_passes']==0 and not h['selection_used_test']
                    # Earliest tie and stopping policy are frozen. No selection from final loss/video metrics.
                    best = max(h['epochs'],key=lambda e:e['validation_macro_f1'])
                    assert best['epoch']==h['best_epoch']==m['best_epoch']
                    assert m['window']['macro_f1']==h['best_validation_macro_f1']
                    with np.load(path/'validation_predictions.npz',allow_pickle=False) as a:
                        y,prob,sources = [a[k] for k in ('labels','probabilities','source_ids')]
                    values = arrays[n,f]
                    np.testing.assert_array_equal(y,values['labels']); np.testing.assert_array_equal(sources,values['source_ids'])
                    guard_sources(sources,folds['folds'][f]['validation_source_ids'],folds['excluded_test_source_ids'])
                    computed = classification_metrics(y,prob.argmax(1))
                    assert all(m['window'][k]==v for k,v in computed.items())
                    assert m['video']==aggregate_videos(y,prob,sources) and m['video']['support']==51
                    assert abs(f1_score(y,prob.argmax(1),labels=[0,1,2],average='macro')-computed['macro_f1'])<1e-12
                    assert confusion_matrix(y,prob.argmax(1),labels=[0,1,2]).tolist()==computed['confusion_matrix']
                    diag = diagnostics(values,prob)
                    if n!=16:
                        assert diag==m['diagnostics']
                    m = dict(m,diagnostics=diag)
                    metrics[name,n,f,seed] = m
                    row = {'architecture':name,'window':n,'fold':f,'seed':seed,'reused_control':n==16,
                        'best_epoch':h['best_epoch'],'stopping_epoch':h['stopping_epoch'],
                        'checkpoint_bytes':m['checkpoint_bytes'],'parameters':COUNTS[name],
                        'training_wall_seconds':h['training_wall_seconds']}
                    for k in ('macro_f1','accuracy','macro_precision','macro_recall','weighted_f1','worst_class_recall','support'):
                        row['window_'+k] = m['window'][k]
                    for k in ('macro_f1','accuracy','support'):
                        row['video_'+k] = m['video'][k]
                    for subset in ('non_padded','padded'):
                        for k in ('macro_f1','accuracy','support'):
                            row[subset+'_'+k] = diag[subset][k]
                    rows[name,n,f,seed] = row
                    for level in ('window','video'):
                        for label,values in m[level]['per_class'].items():
                            classes.append({'architecture':name,'window':n,'fold':f,'seed':seed,'level':level,'class':label,**values})
    aggregate,perclass,matrices = {},{},{}
    for name in ARCHITECTURES:
        for n in LENGTHS:
            key = f'{name}_{n}'
            selected = [r for r in rows.values() if r['architecture']==name and r['window']==n]
            fields = [k for k in selected[0] if k.startswith(('window_','video_','non_padded_','padded_'))]+['checkpoint_bytes']
            aggregate[key] = {field:summary([r[field] for r in selected]) for field in fields}
            perclass[key] = {level:{label:{metric:summary([c[metric] for c in classes if c['architecture']==name and c['window']==n and c['level']==level and c['class']==label])
                for metric in ('precision','recall','f1')} for label in CLASS_NAMES} for level in ('window','video')}
            matrices[key] = {level:{'per_seed':{str(s):sum(np.asarray(metrics[name,n,f,s][level]['confusion_matrix']) for f in range(3)).tolist() for s in SEEDS},
                'pooled_three_seeds':sum(np.asarray(metrics[name,n,f,s][level]['confusion_matrix']) for f in range(3) for s in SEEDS).tolist()}
                for level in ('window','video')}
    paired,paired_rows,decisions = {},[],{}
    for name in ARCHITECTURES:
        for n in (8,24,32):
            key,reference = f'{name}_{n}',f'{name}_16'
            paired[key] = {field:paired_summary({(f,s):rows[name,n,f,s][field] for f in range(3) for s in SEEDS},
                {(f,s):rows[name,16,f,s][field] for f in range(3) for s in SEEDS}) for field in ('window_macro_f1','window_worst_class_recall','video_macro_f1')}
            for f in range(3):
                for s in SEEDS:
                    paired_rows.append({'architecture':name,'window':n,'reference_window':16,'fold':f,'seed':s,
                        **{field+'_delta':rows[name,n,f,s][field]-rows[name,16,f,s][field] for field in ('window_macro_f1','window_worst_class_recall','video_macro_f1')}})
            p = paired[key]['window_macro_f1']
            class_deltas = {label:perclass[key]['window'][label]['recall']['mean']-perclass[reference]['window'][label]['recall']['mean'] for label in CLASS_NAMES}
            worst_delta = paired[key]['window_worst_class_recall']['mean']
            safety = worst_delta>=-RULES['maximum_worst_recall_regression'] and min(class_deltas.values())>=-RULES['maximum_class_recall_regression']
            quality = p['mean']>=RULES['quality_gain'] and p['positive_pairs']>=RULES['minimum_positive_pairs'] and safety
            responsive = n<16 and p['mean']>=-RULES['responsiveness_f1_margin'] and safety
            context = n>16 and quality
            reasons = []
            if worst_delta < -RULES['maximum_worst_recall_regression']:
                reasons.append('mean worst-class recall regresses >0.005')
            reasons.extend(label+' mean recall regresses >0.01' for label,v in class_deltas.items() if v < -RULES['maximum_class_recall_regression'])
            if not quality:
                if p['mean']<RULES['quality_gain']:
                    reasons.append('quality gain <0.003')
                if p['positive_pairs']<RULES['minimum_positive_pairs']:
                    reasons.append('fewer than 5 positive pairs')
            if n<16 and not responsive and p['mean']<-RULES['responsiveness_f1_margin']:
                reasons.append('F1 loss exceeds responsiveness margin 0.002')
            decisions[key] = {'quality_interesting':quality,'responsiveness_interesting':responsive,'context_interesting':context,
                'flagged_for_later_consideration':quality or responsive or context,'safety_pass':safety,'class_mean_recall_deltas':class_deltas,
                'mean_worst_recall_delta':worst_delta,'failed_route_conditions':reasons}
    padding = load(D/'padding_counts.json')
    source_counts = load(D/'source_window_counts.json')
    for row in padding:
        n,f,split,c = row['window'],row['fold'],row['split'],row['class_id']
        matching = [s for s in source_counts if (s['window'],s['fold'],s['split'],s['class_id'])==(n,f,split,c)]
        assert sum(s['windows'] for s in matching)==row['windows']
        assert sum(s['padded_windows'] for s in matching)==row['padded_windows']
        assert row['padded_percentage']==row['padded_windows']/row['windows']*100
    latency = load(D/'latency_results.json')
    assert len(latency['runs'])==48 and latency['raw_samples_sha256']==sha256(D/'latency_samples.npz')
    assert latency['seal']['phase11_test_model_inference_passes']==latency['seal']['test_cache_reads']==0
    with np.load(D/'latency_samples.npz',allow_pickle=False) as a:
        for run in latency['runs']:
            k = f"{run['architecture']}_window_{run['window']}_repeat_{run['repeat']}"
            assert run['wall']==latency_statistics(a[k+'_wall_ms'])
            assert run['process_cpu']==latency_statistics(a[k+'_cpu_ms'])
            assert len(a[k+'_wall_ms'])==1024
            with np.load(D/'latency_raw'/(k+'.npz'),allow_pickle=False) as single:
                np.testing.assert_array_equal(a[k+'_wall_ms'],single['wall_ms'])
                np.testing.assert_array_equal(a[k+'_cpu_ms'],single['process_cpu_ms'])
        for key in aggregate:
            name,n = key.rsplit('_',1)
            for clock,suffix in (('wall','wall_ms'),('process_cpu','cpu_ms')):
                # Preserve randomized acquisition/archive insertion order: mean
                # accumulation can differ by an ulp if repetitions are sorted.
                expected = latency_statistics(np.concatenate([a[k] for k in a.files
                    if k.startswith(f'{name}_window_{n}_repeat_') and k.endswith('_'+suffix)]))
                assert latency['aggregate'][key][clock]==expected
    pareto = {
        'quality_p50':pareto_front({k:(v['window_macro_f1']['mean'],latency['aggregate'][k]['wall']['p50_ms']) for k,v in aggregate.items()}),
        'quality_nominal_warmup':pareto_front({k:(v['window_macro_f1']['mean'],int(k.rsplit('_',1)[1])/20) for k,v in aggregate.items()}),
        'quality_parameters':pareto_front({k:(v['window_macro_f1']['mean'],COUNTS[k.rsplit('_',1)[0]]) for k,v in aggregate.items()})}
    return {'aggregates':aggregate,'per_class':perclass,'paired':paired,'decisions':decisions,'pareto_sets':pareto,
        'confusion_matrices':matrices,'new_runs':108,'reused_controls':36,'comparison_cells':144,
        'primary_metric':'validation all-window macro F1','sample_sd_ddof':1,'test_inference_passes':0},list(rows.values()),classes,paired_rows


def plots(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = D/'plots'; directory.mkdir(exist_ok=False)
    a,p = result['aggregates'],result['per_class']
    l = load(D/'latency_results.json')['aggregate']
    colors = dict(zip(ARCHITECTURES,('#0072B2','#D55E00','#009E73','#CC79A7')))
    def finish(fig,ax,name,title,xlabel,ylabel):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.2)
        ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(directory/(name+'.png'),dpi=170); plt.close(fig)
    for field,title,ylabel in (
        ('f1','Development quality by window length','All-window macro F1'),
        ('worst','Worst-class recall by window length','Mean worst-class recall'),
        ('yawning','Yawning recall by window length','Mean Yawning recall'),
        ('latency','Batch-1 CPU model latency by window length','Wall P50 (ms)')):
        fig,ax = plt.subplots(figsize=(8,5))
        for name in ARCHITECTURES:
            keys = [f'{name}_{n}' for n in LENGTHS]
            values = [a[k]['window_macro_f1']['mean'] if field=='f1' else a[k]['window_worst_class_recall']['mean'] if field=='worst'
                else p[k]['window']['Yawning']['recall']['mean'] if field=='yawning' else l[k]['wall']['p50_ms'] for k in keys]
            if field=='f1':
                ax.errorbar(LENGTHS,values,yerr=[a[k]['window_macro_f1']['std'] for k in keys],marker='o',capsize=3,label=name,color=colors[name])
            else:
                ax.plot(LENGTHS,values,'-o',label=name,color=colors[name])
        ax.set_xticks(LENGTHS)
        finish(fig,ax,field+'_vs_window',title+(' (mean ± sample SD)' if field=='f1' else ''),'Observations per window',ylabel)
    for kind in ('latency','coverage'):
        fig,ax = plt.subplots(figsize=(8,5))
        for name in ARCHITECTURES:
            x = [l[f'{name}_{n}']['wall']['p50_ms'] if kind=='latency' else n/20 for n in LENGTHS]
            y = [a[f'{name}_{n}']['window_macro_f1']['mean'] for n in LENGTHS]
            ax.plot(x,y,'-o',label=name,color=colors[name])
            if kind=='latency':
                for xx,yy,n in zip(x,y,LENGTHS):
                    ax.annotate(str(n),(xx,yy),xytext=(5,5),textcoords='offset points',fontsize=8)
        if kind=='coverage':
            ax.set_xticks([n/20 for n in LENGTHS],[f'{n/20:.2f}\n(N={n})' for n in LENGTHS])
        finish(fig,ax,'quality_vs_'+kind,'Development quality / '+kind+' trade-off',
            'Wall P50 model latency (ms)' if kind=='latency' else 'Nominal coverage / warm-up at 20 obs/s (s)','Mean all-window macro F1')
    fig,ax = plt.subplots(figsize=(8,5))
    padding = load(D/'padding_counts.json')
    for split in ('train','validation'):
        selected = [r for r in padding if r['architecture']=='cnn_lstm' and r['split']==split]
        percentages = [sum(r['padded_windows'] for r in selected if r['window']==n)/sum(r['windows'] for r in selected if r['window']==n)*100 for n in LENGTHS]
        ax.plot(LENGTHS,percentages,'-o',label=split+' (all architectures share arrays)')
    ax.set_xticks(LENGTHS)
    finish(fig,ax,'padding_vs_window','Deterministic repeat-last padding prevalence','Observations per window','Padded windows (%)')
    fig,ax = plt.subplots(figsize=(8,5))
    for name in ARCHITECTURES:
        entries = [result['paired'][f'{name}_{n}']['window_macro_f1'] for n in (8,24,32)]
        ax.errorbar((8,24,32),[e['mean'] for e in entries],yerr=[e['std'] for e in entries],marker='o',capsize=3,label=name,color=colors[name])
    ax.axhline(0,color='black',linewidth=.8); ax.set_xticks((8,24,32))
    finish(fig,ax,'paired_delta','Within-architecture deltas vs window16 (mean ± sample SD)','Observations per window','Paired all-window macro F1 delta')


def report(result,rows,classes,paired):
    a,p = result['aggregates'],result['per_class']
    latency = load(D/'latency_results.json')
    padding = load(D/'padding_counts.json')
    provenance = load(D/'experiment_config.json')
    before = load(ROOT/'docs/phase14_preservation.json')
    tests = ET.parse(ROOT/'docs/phase14_tests.xml').getroot().find('testsuite')
    assert tests is not None and int(tests.attrib['failures'])==int(tests.attrib['errors'])==int(tests.attrib['skipped'])==0
    test_count = int(tests.attrib['tests'])
    parity = load(ROOT/'results/phase14_production_parity/result.json')
    assert parity['exact_parity'] and parity['windows_exact']==parity['classes_exact']==699
    assert all(v==0. for s in parity['sources'] for v in s['maximum_differences'].values())
    assert before['changed_prior_files']=={} and before['missing_prior_files']==[]
    count_rows = [{k:r[k] for k in ('window','fold','split','class_id','windows','padded_windows','padded_percentage')}
                  for r in padding if r['architecture']=='cnn_lstm']
    quality = [{'architecture':name,'N':n,**{k:a[f'{name}_{n}']['window_macro_f1'][k] for k in ('mean','std','minimum','maximum')}} for name in ARCHITECTURES for n in LENGTHS]
    secondary = [{'configuration':key,**{k:values['window_'+k]['mean'] for k in ('accuracy','macro_precision','macro_recall','weighted_f1','worst_class_recall')},
                  'Yawning_recall':p[key]['window']['Yawning']['recall']['mean']} for key,values in a.items()]
    videos = [{'configuration':key,'accuracy':values['video_accuracy']['mean'],'macro_f1':values['video_macro_f1']['mean'],
               **{label+'_recall':p[key]['video'][label]['recall']['mean'] for label in CLASS_NAMES}} for key,values in a.items()]
    diagnostic = [{'configuration':key,'non_padded_F1':values['non_padded_macro_f1']['mean'],'non_padded_accuracy':values['non_padded_accuracy']['mean'],
        'non_padded_support_3seeds':int(sum(r['non_padded_support'] for r in rows if f"{r['architecture']}_{r['window']}"==key)),
        'padded_F1':values['padded_macro_f1']['mean'],'padded_accuracy':values['padded_accuracy']['mean'],
        'padded_support_3seeds':int(sum(r['padded_support'] for r in rows if f"{r['architecture']}_{r['window']}"==key))} for key,values in a.items()]
    perclass = [{'configuration':key,'class':label,**{metric:p[key]['window'][label][metric]['mean'] for metric in ('precision','recall','f1')}} for key in a for label in CLASS_NAMES]
    paired_table = [{'configuration':key,**values['window_macro_f1']} for key,values in result['paired'].items()]
    timing = [{'configuration':key,**latency['aggregate'][key]['wall'],'process_cpu_mean_ms':latency['aggregate'][key]['process_cpu']['mean_ms']} for key in a]
    sizes = [{'configuration':key,'parameters':COUNTS[key.rsplit('_',1)[0]],'checkpoint_bytes_min':v['checkpoint_bytes']['minimum'],'checkpoint_bytes_max':v['checkpoint_bytes']['maximum']} for key,v in a.items()]
    decision = [{'configuration':key,'quality':v['quality_interesting'],'responsiveness':v['responsiveness_interesting'],'context':v['context_interesting'],
        'safety':v['safety_pass'],'later_consideration':v['flagged_for_later_consideration'],
        'failed_conditions':'; '.join(v['failed_route_conditions']) or 'none'} for key,v in result['decisions'].items()]
    rawrows = read_manifest(ROOT/'results/phase11/dataset_manifest_raw.csv')
    short_id = next(r['source_id'] for r in rawrows if r['relative_path']=='Eye Close/Eye Close_51.avi')
    short = [r for r in load(D/'source_window_counts.json') if r['source_id']==short_id and r['split']=='validation']
    changed_created = ['evaluation/windows.py','tests/test_phase14_windows.py','tools/report_phase14.py','tools/run_phase14.py','tools/audit_phase14_preservation.py',
        'docs/phase14_before_hashes.json','docs/phase14_unit.xml','docs/phase14_recipe_unit.xml','docs/phase14_tests.xml',
        'docs/phase14_preservation.json','docs/phase14_validation.json','docs/PHASE14_WINDOW_LENGTH.md',
        'results/phase14/**','results/phase14_production_parity/**','results/phase14_* logs']
    lines = [
        '# PHASE 14 — Window-length development study\n\n**PHASE14 PASS.** All 108 new runs completed, with 36 immutable window16 controls reused: 144 comparison cells. No final model was selected. Work stops after Phase 14.\n',
        '## Files and experimental boundary\n\nCreated: '+', '.join('`'+s+'`' for s in changed_created)+'. Modified prior files: **none**. '
        f"All {before['prior_files_preserved']} files in the before snapshot retain their SHA256; separately verified by a file-byte preservation audit. "
        'That audit hashes sealed artifacts without decoding or inspecting their contents and is separate from candidate processes. '
        'The production model, saved weights, normalization, sampler, buffer, inference engine, state machine, events, dashboard, SQLite, main.py and legacy preprocessing are unchanged. '
        'No balancing, preprocessing correction, all-development refit, test evaluation, production promotion, full application benchmark or Phase 15 work occurred.\n',
        '## Data, integrity, and sealed test\n\nExactly 153 development videos, 51/class; unchanged three video-grouped folds, each 102 train and 51 validation, with 34/17 per class. '
        'The persisted Phase 12 fold file is reused directly; no fold file was regenerated. Canonical assignment verification does not create new folds.\n\n'
        f"CV SHA256: `{provenance['cv_sha256']}`. Frozen Phase 13 architecture-spec SHA256: `{provenance['architecture_spec_sha256']}`. "
        f"Pre-training window-spec SHA256: `{provenance['window_spec_sha256']}`.\n\n"
        'Only verified pre-normalization development caches were loaded: source IDs, raw-video SHA256, cache SHA256, finite features, observation counts and increasing original frame IDs were checked. '
        'MediaPipe was not rerun for candidates. All three existing Phase 12 feature-wise normalizers retain their SHA256 and exact 102 training IDs; validation contributions are zero. '
        'No normalizer was refitted. No Phase 11 global train_norm.npy was used.\n\n'
        'Window16 arrays were reconstructed exactly for normalized float64 values, labels, source order, frame and observation identities, repeat-last flags and starts. '
        'The original arrays omit start indices; starts were independently derived from contiguous source window ordinals ×8 and verified against raw chronology. '
        'All 36 checkpoint/config/history/metrics/probability artifacts are referenced by path and SHA256 in `reused_window16/references.json`; saved metrics were recomputed from those probabilities. '
        'The initial shell training attempt ended before its first checkpoint/metrics were written. Its config, start record and access entry remain in `interrupted_attempt_1`, with an explicit interruption record. '
        'The identical frozen matrix was restarted via `tools/run_phase14.py` in a hidden background process; no completed run was discarded. '
        'The variable-length adapter inherits frozen construction/initialization and keeps state keys and parameters; only the fixed16 input guard is relaxed. '
        'N16 training/evaluation outputs, including seeded dropout, are bit exact. A two-epoch recipe unit test reproduced frozen history and weights; it produced no control experiment checkpoint.\n\n'
        'The canonical 27 Phase 11 test sources remain sealed. Candidate source whitelists and a permanent process file-open audit reject their raw videos, raw feature caches, normalized test arrays, predictions and metrics. '
        '**Phase 11 test model inference passes during Phase 14 = 0; test-cache reads = 0; attempted blocked opens in candidate experiments = 0.** '
        'A separate subprocess unit test deliberately checks six forbidden opens and confirms rejection. No candidate test predictions or test confusion matrices exist.\n',
        '## Frozen experiment recipe and temporal contract\n\nFour models: cnn_lstm, lstm_only, cnn_gru, cnn_bilstm. CNN-only is excluded after its Phase 13 class-recall gate failure. '
        'Seeds 1201/1202/1203, folds 0/1/2. New runs: 3 lengths ×4 architectures ×3 folds ×3 seeds =108. Controls: 4×3×3=36. '
        'Unweighted CE, ordinary seeded shuffle, Adam 0.001, batch 64, no scheduler, max 60 epochs, patience 10, minimum improvement 1e-12, earliest ties, validation WINDOW macro F1 selection. '
        'CPU float32; deterministic algorithms; num_workers = 0; Torch 8/8 threads. Same-seed initial state digests match Phase 13 for every run. '
        'Every source uses range(0,valid_observations,N/2); incomplete tails repeat its final valid observation. Windows never cross sources. '
        'No source is removed for face coverage or padding.\n\n'+table([temporal(n) for n in LENGTHS])+'\n'
        'Nominal coverage N/20 and first-to-last span (N−1)/20 are distinct. Face loss can make actual wall-time span longer. '
        'Theoretical float64 payload excludes Python/deque/tensor/model overhead. Potential runtime prediction stride remains1 after warm-up, independent of training stride N/2. '
        'The production16-entry buffer remains unchanged. Whole-video labels provide no trustworthy event-onset annotation; these are temporal responsiveness characteristics, not measured drowsiness/yawn/alert response times.\n\n'
        'CNN lengths: 8→6→4→pool2; 16→14→12→pool6; 24→22→20→pool10; 32→30→28→pool14. Hard forward tests verify all 16 architecture/length combinations output finite (B,3).\n',
        '## Window counts and padding\n\nCounts below are shared by every architecture and seed; `padding_analysis.csv` also expands architecture×window×fold×split×class. '
        'Full per-source counts, decoded/valid frame counts, and padding shares are in `source_window_counts.json` and `.csv`.\n\n'+table(count_rows)+'\n'
        f"The retained short source is `{short_id}`, Eye Close/Eye Close_51.avi: **10 valid observations /135 decoded frames**. Its validation counts by length follow; train inclusion in other folds uses the same policy.\n\n"+table(short)+'\n',
        '## Primary development quality\n\nAll deterministic held-out windows remain primary. Mean, sample SD (ddof1), min and max summarize nine fold/seed cells, not nine independent datasets.\n\n'+table(quality)+'\n',
        '## Secondary window and class behavior\n\n'+table(secondary)+'\n'+table(perclass)+'\n'
        'Yawning recall is tracked descriptively; no recipe was tuned specifically to it. Raw per-class fold/seed metrics for window and video levels are in `per_class_results.csv`; summary dispersion is in aggregate_results.json.\n',
        '## Video-level development results\n\nMean window probabilities per held-out source → argmax; exactly one vote per source. Each cell has 51 decisions. Across folds, each 153-video cohort is evaluated once per seed (459 repeated-seed video decisions/configuration).\n\n'+table(videos)+'\n',
        '## Padding diagnostics\n\nNon-padded and padded metrics are secondary only; all windows remain primary. Fixed three-class order and zero-division = 0 are used even for subsets. '
        'Support totals below include three seeds of the same physical windows; raw per-cell supports and metrics are stored. Padded subsets are smaller and should not be interpreted as independent validation cohorts.\n\n'+table(diagnostic)+'\n',
        '## Paired within-architecture comparisons\n\nCandidate minus SAME architecture at16, matched fold/seed. No cross-architecture reference is substituted.\n\n'+table(paired_table)+'\n'
        '12 candidate configurations ×9 pairs =108 paired differences are retained in `paired_window_comparisons.csv`, with F1, worst recall and video F1. '
        'These overlaps and repeated seeds preclude interpreting pair counts as independent replications or formal significance tests.\n',
        '## Model-only latency and parameter/storage cost\n\nPrepared normalized float32 batch1 tensor →logits only. Fixed Torch 8/8. '
        '64 development inputs at common source/start anchors (51 first windows and13 starts at48), same identities across lengths; every architecture within a length uses exactly the same tensors. '
        'For each configuration the predeclared fold 0/seed 1201 checkpoint is measured in three repetitions, 128 warm-up calls and 1024 measured calls per repetition: 3072 raw samples/configuration, 49152 total. '
        'Job order is randomized with seed 1402. This checkpoint choice was fixed before quality results; latency is not a nine-checkpoint distribution. '
        'Wall and processCPU arrays are preserved both per job and pooled, with source/frame/start/tensor hashes. No input preparation or model loading is timed.\n\n'+table(timing)+'\n'
        'All raw samples remain in the means and quantiles; no trimming. Extreme wall stalls are predeclared as≥100ms. '
        f"Observed extreme stalls: **{len(latency['extreme_stalls'])}**. "
        'They are listed below with paired CPU time; OS/scheduler cause is not inferred solely from elapsed time. Windows process CPU accounting sums multiple threads and is coarse; zeros or values exceeding wall time are possible. It is not another wall latency estimate.\n\n'
        +(table(latency['extreme_stalls']) if latency['extreme_stalls'] else 'No measured samples exceeded the extreme-stall threshold.\n')+'\n'+table(sizes)+'\n'
        'Parameter count is invariant across lengths within each frozen model. Container/metadata differences in checkpoint bytes do not indicate a change in learned parameter count.\n',
        '## Later-consideration filters and Pareto sets\n\nRules were persisted before training: quality gain≥0.003, ≥5/9 positive pairs, mean worst-class recall regression≤0.005, and no class mean recall regression>0.01. '
        'Shorter windows are responsiveness-interesting when no more than0.002 below their own16-control and safety passes; quality gains also qualify under the persisted noninferiority-margin interpretation. '
        'Longer windows are context-interesting only when the quality rule passes. These filters describe evidence for later consideration and do not select a final architecture/window. '
        'Window16 configurations are retained reference evidence, not tested as new window candidates.\n\n'+table(decision)+'\n'
        +'\n'.join('- '+name+': '+', '.join(values) for name,values in result['pareto_sets'].items())+'\n\n'
        'Pareto dominance uses higher mean F1 and lower stated cost; no hidden composite score. The sets can differ with descriptive timing variation and are not final selection.\n',
        '## Architecture × window observations\n\n'+interaction_text(result)+'\n'
        'Changes in temporal context also change the number of overlapping training windows and padding share. This experiment measures their combined deterministic protocol effect; it does not isolate context from exposure/optimization changes. '
        'No formal interaction significance model was predeclared or run.\n',
        '## Confusions and plots\n\nAll 16 configurations have raw window and video confusion CSVs, per seed and pooled three-seed totals, rows actual and columns predicted in Normal/Yawning/Eye Close order. '
        'Pooled confusion counts repeat the development cohort across seeds; quality summaries are means of cell metrics, not F1 from the pooled matrix. '
        'The eight plots visualize macro F1, worst recall, Yawning recall, model P50, quality/P50, quality/coverage, padding shares and paired F1 deltas. RawCSV/JSON is authoritative.\n',
        f"## Regression and acceptance\n\n**{test_count} tests passed** ({test_count-495} Phase 14 tests; all 495 Phase2–13 tests retained). "
        '**Production 699/699 windows and 699/699 classes exact; every maximum numerical difference 0.0.** '
        'Frozen raw-production parity is a separate authorized regression path, using the existing production model and fixtures, never an experimental checkpoint or candidate test evaluation. '
        'All 108 new runs completed with no failure/discard; 36 controls accounted for; 153 development sources preserved; normalizer refits 0; candidate sealed-test inference 0. '
        'No final model selection, all-development refit, final test evaluation or promotion. **PHASE14 PASS.**\n',
        '## Limitations and exact reproduction\n\nDevelopment folds and seed runs overlap; reported SD is descriptive, not an independent-video confidence interval. Different lengths yield different correlated window counts; video aggregation gives each source one vote but depends on whole-video labels. '
        'Very short/padded windows are retained and disclosed; subset performance is not a new selection target. Legacy BGR/SRGB tagging, pose geometry/landmark order/angle conversion, EAR/MAR and zero-anchored normalization remain as frozen. '
        'CPU timing is specific to this host, fixed threads and representative checkpoint; wall tails may include scheduling stalls. Nominal 20 observations/s properties do not measure actual event response. '
        'Development evidence cannot establish sealed-test generalization.\n\n'
        'Commands run from repository root using a fresh output path (artifact creation is exclusive; do not overwrite existing evidence):\n\n'
        '```powershell\n.venv-baseline/Scripts/python.exe -B -m evaluation.windows prepare\n'
        '.venv-baseline/Scripts/python.exe -B -m pytest -q tests/test_phase14_windows.py\n'
        '.venv-baseline/Scripts/python.exe -B -m evaluation.windows run\n'
        '.venv-baseline/Scripts/python.exe -B -m evaluation.windows latency\n'
        '.venv-baseline/Scripts/python.exe -B -m pytest -q --junitxml=docs/phase14_tests.xml\n'
        '.venv-baseline/Scripts/python.exe -B tools/optimize_phase9.py --parity production --output results/phase14_production_parity/result.json\n'
        '.venv-baseline/Scripts/python.exe -B tools/audit_phase14_preservation.py\n'
        "$env:MPLCONFIGDIR='.cache/matplotlib'\n.venv-baseline/Scripts/python.exe -B tools/report_phase14.py\n```\n"
        'The recorded execution redirected Python stdout/stderr to exclusive line-buffered phase14 log files. '
        'The before/after preservation audit is separate from experiment execution; its result is required by the reporter. '
        'No Phase 15 work is authorized by this report.\n'
    ]
    text = '\n'.join(lines)
    with (ROOT/'docs/PHASE14_WINDOW_LENGTH.md').open('x',encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'docs/phase14_validation.json',{'status':'PHASE14_PASS','tests':test_count,'new_tests':test_count-495,
        'production_windows_exact':699,'maximum_numerical_difference':0.,'completed_new_runs':108,'reused_controls':36,
        'comparison_cells':144,'candidate_test_inference_passes':0,'test_cache_reads':0,
        'prior_files_preserved':before['prior_files_preserved'],'changed_prior_files':{},'normalization_refits':0,
        'metrics_recomputed_and_sklearn_crosschecked':True,'latency_quantiles_recomputed':True,
        'window16_reconstruction_exact':True,'final_model_selected':False,'phase15_started':False,
        'report_sha256':sha256(ROOT/'docs/PHASE14_WINDOW_LENGTH.md')})


def interaction_text(result):
    a = result['aggregates']
    parts = []
    for name in ARCHITECTURES:
        values = [(n,a[f'{name}_{n}']['window_macro_f1']['mean']) for n in LENGTHS]
        highest = max(values,key=lambda t:t[1])
        recall = result['per_class']
        deltas = ', '.join(f"{label} {recall[f'{name}_32']['window'][label]['recall']['mean']-recall[f'{name}_16']['window'][label]['recall']['mean']:+.6f}" for label in CLASS_NAMES)
        parts.append(f"{name}: largest observed mean F1 at N={highest[0]} ({highest[1]:.6f}); N8-to-N32 change {values[-1][1]-values[0][1]:+.6f}. N32-vs16 mean recall deltas: {deltas}.")
    return '\n\n'.join(parts)+'\n\nThese are descriptive maxima within the evaluated matrix, not model selection. CNN recurrent lengths are 2/6/10/14, while LSTM-only reads 8/16/24/32 raw observations; the frozen CNN representation compresses the temporal sequence.'


def main():
    result,rows,classes,paired = analyze()
    write_json(D/'aggregate_results.json',result)
    csv_out(D/'run_results.csv',rows); csv_out(D/'per_class_results.csv',classes)
    csv_out(D/'paired_window_comparisons.csv',paired)
    csv_out(D/'paired_window_summary.csv',[dict(configuration=k,**v['window_macro_f1']) for k,v in result['paired'].items()])
    csv_out(D/'padding_analysis.csv',load(D/'padding_counts.json'))
    csv_out(D/'source_window_counts.csv',load(D/'source_window_counts.json'))
    csv_out(D/'temporal_tradeoffs.csv',[temporal(n) for n in LENGTHS])
    directory = D/'confusion_matrices'; directory.mkdir(exist_ok=False)
    for configuration,levels in result['confusion_matrices'].items():
        for level,values in levels.items():
            for label,matrix in dict(values['per_seed'],pooled_three_seeds=values['pooled_three_seeds']).items():
                csv_out(directory/f'{configuration}_{level}_{label}.csv',[dict(actual=CLASS_NAMES[i],**dict(zip(CLASS_NAMES,row))) for i,row in enumerate(matrix)])
    plots(result)
    report(result,rows,classes,paired)
    print(json.dumps({'status':'PHASE14_PASS','flags':[k for k,v in result['decisions'].items() if v['flagged_for_later_consideration']]}))


if __name__=='__main__':
    main()
