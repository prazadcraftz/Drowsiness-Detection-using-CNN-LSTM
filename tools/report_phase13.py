"""Audit the frozen architecture study and report development-only evidence."""
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.architectures import (verify_protocol, latency_statistics, summary,
                                     paired_summary, pareto_front, compare_control)
from evaluation.balancing import guard_sources
from evaluation.manifest import sha256,write_json
from evaluation.metrics import classification_metrics,aggregate_videos
from evaluation.models import NAMES,build_model
from evaluation.reproducibility import seed_everything
from tools.report_phase12 import table,csv_out

D=ROOT/'results/phase13'


def load(path):
    return json.loads(Path(path).read_text())


def analyze():
    folds,_,_,_=verify_protocol(D,raw=True,enforce_seal=False)
    config,spec,latency,control=[load(D/name) for name in ('experiment_config.json','architecture_spec.json','latency_results.json','control_reproduction.json')]
    assert load(D/'experiment_started.json')['config_sha256']==sha256(D/'experiment_config.json')
    assert control['passed'] and len(control['runs'])==9
    assert all(all(r[k] for k in ('weights_bit_exact','epoch_metrics_exact','selection_exact','validation_probabilities_exact')) for r in control['runs'])
    assert sha256(D/'cv_folds.json')==sha256(ROOT/'results/phase12/cv_folds.json')
    audit=load(D/'test_access_audit.json')
    assert audit['phase11_test_model_inference_passes']==audit['test_cache_reads']==0 and audit['completed_runs']==45
    access=[json.loads(line) for line in (D/'candidate_access_audit.jsonl').read_text().splitlines()]
    assert len(access)==45 and sha256(D/'candidate_access_audit.jsonl')==audit['candidate_access_log_sha256']
    assert len({(e['architecture'],e['fold'],e['seed']) for e in access})==45
    for e in access:
        fold=folds['folds'][e['fold']]
        assert e['train_source_ids']==fold['train_source_ids'] and e['validation_source_ids']==fold['validation_source_ids']
        guard_sources(e['train_source_ids']+e['validation_source_ids'],folds['development_source_ids'],folds['excluded_test_source_ids'])
        assert e['test_inference_passes']==0
    assert not list(D.rglob('test_predictions*')) and not list(D.rglob('failure.json'))
    assert latency['test_inference_passes']==0 and len(latency['runs'])==45
    assert latency['raw_samples_sha256']==sha256(D/'latency_samples.npz')
    assert latency['input_sha256']==sha256(D/'latency_inputs.npz')
    raw={}
    with np.load(D/'latency_samples.npz',allow_pickle=False) as archive:
        for k in archive.files:
            raw[k]=archive[k]
    for row in latency['runs']:
        key=f"{row['architecture']}_fold_{row['fold']}_seed_{row['seed']}"
        assert all(row[k]==v for k,v in latency_statistics(raw[key]).items())
    for name in NAMES:
        assert latency['aggregate'][name]==latency_statistics(np.concatenate([v for k,v in raw.items() if k.startswith(name+'_fold_')]))
    confirmation=load(D/'latency_confirmation/results.json')
    confirmation_protocol=load(D/'latency_confirmation/protocol.json')
    assert confirmation['test_inference_passes']==0 and len(confirmation['runs'])==45
    assert confirmation_protocol['primary_results_sha256']==sha256(D/'latency_results.json')
    assert confirmation_protocol['input_sha256']==latency['input_sha256']
    assert confirmation_protocol['protocol']==latency['protocol']
    assert confirmation_protocol['job_order']==load(D/'latency_started.json')['job_order']
    assert confirmation['protocol_sha256']==sha256(D/'latency_confirmation/protocol.json')
    assert confirmation['samples_sha256']==sha256(D/'latency_confirmation/samples.npz')
    with np.load(D/'latency_confirmation/samples.npz',allow_pickle=False) as archive:
        for row in confirmation['runs']:
            key=f"{row['architecture']}_fold_{row['fold']}_seed_{row['seed']}"
            assert all(row[k]==v for k,v in latency_statistics(archive[key]).items())
            cpu=archive[key+'_process_cpu_ms']
            assert cpu.shape==archive[key].shape and np.isfinite(cpu).all() and (cpu>=0).all()
        for name in NAMES:
            values=np.concatenate([archive[k] for k in archive.files if k.startswith(name+'_fold_') and not k.endswith('_process_cpu_ms')])
            assert confirmation['aggregate'][name]==latency_statistics(values)
    seed_everything(1201)
    runs,classes,metrics={},[],{}
    from sklearn.metrics import f1_score,confusion_matrix
    for name in NAMES:
        for fold in range(3):
            for seed in config['training_seeds']:
                p=D/name/f'fold_{fold}'/f'seed_{seed}'
                m,h,c=[load(p/x) for x in ('metrics.json','history.json','config.json')]
                assert c['architecture']==name and c['seed']==seed and c['fold']==fold
                assert c['recipe']==dict(config['recipe'],seed=seed)
                assert not c['recipe']['class_weighting'] and not c['recipe']['weighted_sampling']
                assert c['normalization_sha256']==sha256(ROOT/f'results/phase12/fold_{fold}/train_norm.npy')
                assert m['test_inference_passes']==0 and not h['selection_used_test']
                assert m['checkpoint_sha256']==sha256(p/'best.pt')
                assert m['checkpoint_bytes']==(p/'best.pt').stat().st_size
                assert m['trainable_parameters']==spec[name]['trainable_parameters']
                assert max(h['epochs'],key=lambda e:e['validation_macro_f1'])['epoch']==h['best_epoch']==m['best_epoch']
                assert h['best_validation_macro_f1']==m['window']['macro_f1']
                with np.load(p/'validation_predictions.npz',allow_pickle=False) as archive:
                    y,prob,sources=[archive[k] for k in ('labels','probabilities','source_ids')]
                guard_sources(sources,folds['folds'][fold]['validation_source_ids'],folds['excluded_test_source_ids'])
                assert set(sources)==set(folds['folds'][fold]['validation_source_ids'])
                computed=classification_metrics(y,prob.argmax(1))
                assert all(m['window'][k]==v for k,v in computed.items())
                assert m['video']==aggregate_videos(y,prob,sources)
                assert abs(f1_score(y,prob.argmax(1),labels=[0,1,2],average='macro')-computed['macro_f1'])<1e-12
                assert confusion_matrix(y,prob.argmax(1),labels=[0,1,2]).tolist()==computed['confusion_matrix']
                if name=='cnn_lstm':
                    model=build_model(name)
                    model.load_state_dict(torch.load(p/'best.pt',map_location='cpu',weights_only=True))
                    compare_control(model,h,m,prob,ROOT/f'results/phase12/unweighted/fold_{fold}/seed_{seed}')
                metrics[(name,fold,seed)]=m
                runs[(name,fold,seed)]={'architecture':name,'fold':fold,'seed':seed,'best_epoch':m['best_epoch'],'stopping_epoch':h['stopping_epoch'],
                    **{'window_'+k:m['window'][k] for k in ('macro_f1','accuracy','macro_precision','macro_recall','worst_class_recall')},
                    'video_macro_f1':m['video']['macro_f1'],'video_accuracy':m['video']['accuracy'],
                    'window_support':m['window']['support'],'video_support':m['video']['support'],'checkpoint_bytes':m['checkpoint_bytes'],
                    'trainable_parameters':m['trainable_parameters'],'training_seconds':h['training_wall_seconds']}
                for level in ('window','video'):
                    for label,values in m[level]['per_class'].items():
                        classes.append({'architecture':name,'fold':fold,'seed':seed,'level':level,'class':label,**values})
    names=('window_macro_f1','window_accuracy','window_macro_precision','window_macro_recall','window_worst_class_recall','video_macro_f1','video_accuracy')
    aggregates={name:{metric:summary([r[metric] for r in runs.values() if r['architecture']==name]) for metric in names} for name in NAMES}
    perclass={name:{level:{label:{metric:summary([c[metric] for c in classes if c['architecture']==name and c['level']==level and c['class']==label]) for metric in ('precision','recall','f1')} for label in ('Normal','Yawning','Eye Close')} for level in ('window','video')} for name in NAMES}
    paired,paired_rows={},[]
    for name in NAMES:
        if name=='cnn_lstm':
            continue
        paired[name]={}
        for metric in ('window_macro_f1','window_worst_class_recall','video_macro_f1'):
            candidate={(f,s):runs[(name,f,s)][metric] for f in range(3) for s in config['training_seeds']}
            reference={(f,s):runs[('cnn_lstm',f,s)][metric] for f in range(3) for s in config['training_seeds']}
            paired[name][metric]=paired_summary(candidate,reference)
        for f in range(3):
            for s in config['training_seeds']:
                paired_rows.append({'architecture':name,'fold':f,'seed':s,**{metric+'_delta':runs[(name,f,s)][metric]-runs[('cnn_lstm',f,s)][metric] for metric in ('window_macro_f1','window_worst_class_recall','video_macro_f1')}})
    front=pareto_front({name:(aggregates[name]['window_macro_f1']['mean'],latency['aggregate'][name]['p50_ms']) for name in NAMES})
    guidance=config['guidance'];decisions={}
    for name in NAMES:
        if name=='cnn_lstm':
            continue
        mean_gain=paired[name]['window_macro_f1']['mean']
        noninferior=sum(r['window_macro_f1_delta']>=-guidance['efficient_quality_margin'] for r in paired_rows if r['architecture']==name)
        safety={'worst_class_recall':paired[name]['window_worst_class_recall']['mean']>=-guidance['maximum_worst_recall_regression'],
                'class_recalls':all(perclass[name]['window'][c]['recall']['mean']-perclass['cnn_lstm']['window'][c]['recall']['mean']>=-guidance['maximum_class_recall_regression'] for c in perclass[name]['window']),
                'variability':aggregates[name]['window_macro_f1']['std']-aggregates['cnn_lstm']['window_macro_f1']['std']<=guidance['maximum_macro_f1_std_increase']}
        reduction=guidance['minimum_latency_or_parameter_reduction_fraction']
        efficient=(mean_gain>=-guidance['efficient_quality_margin'] and noninferior>=guidance['minimum_noninferior_pairs'] and
                   (latency['aggregate'][name]['p50_ms']<=latency['aggregate']['cnn_lstm']['p50_ms']*(1-reduction) or spec[name]['trainable_parameters']<=spec['cnn_lstm']['trainable_parameters']*(1-reduction)))
        quality=(mean_gain>=guidance['strong_quality_gain'] and paired[name]['window_macro_f1']['positive_pairs']>=guidance['minimum_positive_pairs_for_quality_gain'])
        decisions[name]={'safety_gates':safety,'efficient_route':efficient,'strong_quality_route':quality,'noninferior_pairs':noninferior,
                         'interesting_for_later_consideration':all(safety.values()) and (efficient or quality)}
    matrices={}
    for name in NAMES:
        matrices[name]={}
        for level in ('window','video'):
            byseed={str(s):sum(np.array(metrics[(name,f,s)][level]['confusion_matrix']) for f in range(3)).tolist() for s in config['training_seeds']}
            matrices[name][level]={'per_seed':byseed,'pooled_three_seeds':sum(np.array(v) for v in byseed.values()).tolist()}
    confirmation_front=pareto_front({name:(aggregates[name]['window_macro_f1']['mean'],confirmation['aggregate'][name]['p50_ms']) for name in NAMES})
    return {'aggregates':aggregates,'per_class':perclass,'paired':paired,'decisions':decisions,'pareto_quality_latency':front,
            'confirmation_pareto_quality_latency':confirmation_front,
            'confusion_matrices':matrices,'number_of_runs':45,'test_inference_passes':0},list(runs.values()),classes,paired_rows


def plots(result,runs,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=D/'plots';p.mkdir(exist_ok=True)
    labels=[n.replace('_',' ').upper() for n in NAMES]
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for axis,metric,title in zip(axes,('window_macro_f1','window_worst_class_recall'),('Validation window macro F1','Validation worst-class recall')):
        axis.boxplot([[r[metric] for r in runs if r['architecture']==n] for n in NAMES],tick_labels=labels)
        axis.tick_params(axis='x',rotation=25);axis.set(title=title,ylabel='Score');axis.grid(axis='y',alpha=.2)
    fig.savefig(p/'quality_variability.png',dpi=150);plt.close(fig)
    latency=load(D/'latency_results.json')['aggregate'];spec=load(D/'architecture_spec.json')
    for metric,title,filename in (('latency','Model-only CPU P50 latency vs development quality','latency_quality.png'),('parameters','Parameter count vs development quality','parameter_quality.png')):
        fig,axis=plt.subplots(figsize=(10,6))
        fig.subplots_adjust(left=.15,right=.97,bottom=.15,top=.90)
        for n in NAMES:
            x=latency[n]['p50_ms'] if metric=='latency' else spec[n]['trainable_parameters']
            y=result['aggregates'][n]['window_macro_f1']
            axis.errorbar(x,y['mean'],yerr=y['std'],fmt='o',capsize=3,label=n.replace('_',' '))
        if metric=='latency':
            front=result['pareto_quality_latency'];order=sorted(front,key=lambda n:latency[n]['p50_ms'])
            axis.plot([latency[n]['p50_ms'] for n in order],[result['aggregates'][n]['window_macro_f1']['mean'] for n in order],'--',color='gray',label='Descriptive Pareto front')
        axis.set(xlabel='Batch-1 model-only CPU P50 (ms)' if metric=='latency' else 'Trainable parameters',ylabel='Mean validation macro F1 ± sample SD',title=title)
        axis.legend(fontsize=8,loc='upper left');axis.grid(alpha=.2)
        fig.savefig(p/filename,dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axis=plt.subplots(figsize=(12,5),layout='constrained')
    for n in NAMES:
        if n!='cnn_lstm':
            rows=[r for r in paired if r['architecture']==n]
            axis.plot(range(9),[r['window_macro_f1_delta'] for r in rows],'o-',label=n.replace('_',' '))
    axis.axhline(0,color='black',linewidth=1);axis.set(xticks=range(9),xticklabels=[f'f{f}/s{s}' for f in range(3) for s in (1201,1202,1203)],ylabel='Macro F1 change vs same-fold/seed CNN-LSTM',title='Paired development differences')
    axis.tick_params(axis='x',rotation=25);axis.legend();axis.grid(alpha=.2)
    fig.savefig(p/'paired_f1_delta.png',dpi=150);plt.close(fig)
    fig,axis=plt.subplots(figsize=(11,5),layout='constrained')
    for i,n in enumerate(NAMES):
        values=[result['per_class'][n]['window'][c]['recall'] for c in ('Normal','Yawning','Eye Close')]
        axis.errorbar(np.arange(3)+(i-2)*.06,[v['mean'] for v in values],yerr=[v['std'] for v in values],fmt='o',capsize=3,label=n.replace('_',' '))
    axis.set(xticks=range(3),xticklabels=['Normal','Yawning','Eye Close'],ylabel='Validation recall mean ± sample SD',title='Per-class recall across 9 fold/seed runs')
    axis.legend();axis.grid(axis='y',alpha=.2)
    fig.savefig(p/'per_class_recall.png',dpi=150);plt.close(fig)
    for level in ('window','video'):
        fig,axes=plt.subplots(2,3,figsize=(15,10))
        fig.subplots_adjust(left=.08,right=.98,bottom=.08,top=.87,wspace=.38,hspace=.40)
        for axis,n in zip(axes.flat,NAMES):
            matrix=np.array(result['confusion_matrices'][n][level]['pooled_three_seeds'])
            axis.imshow(matrix,cmap='Blues')
            for i in range(3):
                for j in range(3):
                    axis.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
            axis.set(xticks=range(3),yticks=range(3),xticklabels=['Normal','Yawning','Eye Close'],yticklabels=['Normal','Yawning','Eye Close'],xlabel='Predicted',ylabel='Actual',title=n.replace('_',' '))
        axes.flat[-1].axis('off')
        fig.suptitle(f'Development {level} confusion counts pooled across folds and 3 seeds\nRepeated predictions are not independent subjects',y=.97)
        fig.savefig(p/f'{level}_confusion.png',dpi=150,bbox_inches='tight');plt.close(fig)


def report(result,runs,tests,newtests,preserved):
    config=load(D/'experiment_config.json');spec=load(D/'architecture_spec.json');latency=load(D/'latency_results.json')
    quality=[];sizes=[]
    for n in NAMES:
        a=result['aggregates'][n];l=latency['aggregate'][n]
        quality.append({'Architecture':n,'F1 mean':a['window_macro_f1']['mean'],'F1 SD':a['window_macro_f1']['std'],
                        'Accuracy':a['window_accuracy']['mean'],'Macro precision':a['window_macro_precision']['mean'],
                        'Macro recall':a['window_macro_recall']['mean'],'Worst recall':a['window_worst_class_recall']['mean'],
                        'Video F1':a['video_macro_f1']['mean'],'Video accuracy':a['video_accuracy']['mean']})
        sizes.append({'Architecture':n,'Parameters':spec[n]['trainable_parameters'],'Checkpoint bytes min':min(r['checkpoint_bytes'] for r in runs if r['architecture']==n),
                      'Checkpoint bytes max':max(r['checkpoint_bytes'] for r in runs if r['architecture']==n),**l})
    csv_out(D/'model_sizes.csv',sizes)
    classrows=[{'Architecture':n,'Level':level,'Class':c,**{m:values[m]['mean'] for m in ('precision','recall','f1')}} for n in NAMES for level in ('window','video') for c,values in result['per_class'][n][level].items()]
    variability=[{'Architecture':n,'Metric':m,**v} for n,a in result['aggregates'].items() for m,v in a.items()]
    pairs=[{'Architecture':n,'Paired metric':m,**v} for n,a in result['paired'].items() for m,v in a.items()]
    architectures=table([{'Name':n,'Recurrent hidden':spec[n]['hidden_size'],'Layers':spec[n]['recurrent_layers'],'Readout':spec[n]['readout'],'Parameters':spec[n]['trainable_parameters']} for n in NAMES])
    recommendations=[n for n,d in result['decisions'].items() if d['interesting_for_later_consideration']]
    confirmation=load(D/'latency_confirmation/results.json')
    confirmation_table=table([{'Architecture':n,**confirmation['aggregate'][n]} for n in NAMES])
    decision='\n'.join(f'- {n}: {json.dumps(d,sort_keys=True)}.' for n,d in result['decisions'].items())
    matrix_text=''
    for n in NAMES:
        for level in ('window','video'):
            matrix_text+=f'\n**{n}, {level}, pooled 3 seeds**\n\n'+table([{'Actual / predicted':c,**dict(zip(('Normal','Yawning','Eye Close'),row))} for c,row in zip(('Normal','Yawning','Eye Close'),result['confusion_matrices'][n][level]['pooled_three_seeds'])])
    deltas={n:result['paired'][n]['window_macro_f1']['mean'] for n in result['paired']}
    gru_ratio=latency['aggregate']['cnn_gru']['p50_ms']/latency['aggregate']['cnn_lstm']['p50_ms']
    lstm_ratio=latency['aggregate']['lstm_only']['p50_ms']/latency['aggregate']['cnn_lstm']['p50_ms']
    raw=''
    for n in NAMES:
        raw+=f"\n### {n}\n\nHypothesis: {spec[n]['hypothesis']}\n\n```text\n{spec[n]['module_definition']}\n```\n"
    text=f'''# Phase 13 — architecture ablations on fixed development folds

**PHASE 13 PASS.** All 45 planned runs completed. The nine CNN-LSTM controls reproduce Phase 12 unweighted weights, epoch histories, checkpoint selection and validation probabilities exactly. Development-only Pareto set (mean window macro F1 / model-only CPU P50): **{', '.join(result['pareto_quality_latency'])}**. Additional models meeting the predeclared later-consideration rules: **{', '.join(recommendations) if recommendations else 'none; retain the CNN-LSTM control'}**. No architecture was promoted. Phase 11 test model inference passes during Phase 13: **0**. Stop after Phase 13.

## 1–4. Files, preservation and boundary

Created evaluation/models.py (central fixed registry), evaluation/architectures.py (prepare/run/latency driver), tests/test_phase13_architectures.py, tools/report_phase13.py, tools/confirm_phase13_latency.py (separate confirmation after a timing anomaly), this report, docs/phase13_before_hashes.json, unit/full-suite XML, docs/phase13_validation.json, experiment/training/latency/report logs and all new results/phase13 artifacts. New production regression evidence is in results/phase13_production_parity. Exact created-file inventory: results/phase13/created_files.json.

**Files modified from before Phase 13: none.** All {preserved} pre-existing non-environment files are byte-identical, including all Phase 11/12 evidence, raw corpus, production assets and runtime code. Existing unrelated working-tree changes are preserved. No balancing, window study, corrected preprocessing, final test evaluation, all-development refit, policy/runtime/storage change, deployment, model promotion or Phase 14+ work. No production benchmark repeated or profiling dependency added. Only the five primary requested architectures were trained.

Canonical test IDs are checked against the original split before every experiment. Guarded fold data cannot contain a sealed source; process file-open seal rejects the actual sealed arrays, raw caches/videos and test metric files. The reused guard's error message names its originating Phase 12 implementation; it enforces the same seal during Phase 13. Preservation checks only hash prior bytes and do not load candidate test tensors or infer on test sources.

## 5–8. Exact reused protocol, normalization and recipe

153 development sources, 51/class. Byte-identical reuse of Phase 12 cv_folds.json (SHA `{sha256(D/'cv_folds.json')}`), three fixed content-group-disjoint, class-stratified folds: 102 train / 51 validation videos per fold, 34 train / 17 validation per class. Fold assignment SHA `{load(D/'cv_folds.json')['assignment_sha256']}`. Each source is validation exactly once per seed. The 27 Phase 11 test IDs remain excluded. Video-independent only; unknown subject identities are not invented.

Verified raw video and PRE-normalization cache hashes for all 153 development sources against Phase 12 provenance, finite observations and increasing original frame indices. Reconstructed every fold's zero-initialized extrema from fold TRAIN only and compared train/validation windows, frame identities, padding and labels exactly with the frozen prepared arrays. Normalizers and prepared arrays are read directly from Phase 12; no normalized arrays or range files are overwritten. Input/code/spec SHA-256 hashes are locked in experiment_config.json before training. No MediaPipe extraction repeated.

Fixed feature order EAR-R, EAR-L, MAR, phi, theta; input (B,5,16), stride8, repeat-last valid tail padding and historical every-frame valid-observation compression. No BGR/SRGB, landmark, pose, angle, zero-anchor or face-gap changes. Same unweighted CE, seeded shuffled DataLoader (no weighted sampler), Adam .001, batch64, CPU float32, max60epochs, patience10, no scheduler, minimum improvement 1e-12, earliest exact ties, eval/inference_mode for validation, deterministic algorithms, Torch intra/interop8/8 and num_workers0. Only architecture changes.

## 9–13. Predeclared models, parameters, initialization and runs

{architectures}
Reference cnn_lstm directly instantiates model.py CNN_LSTM, preserving state keys, construction RNG draws, module shapes and the exact forward implementation (246595 parameters). Its front-end uses Conv1d 5→32→64, kernel3/no padding, ReLU after each, MaxPool1d2: temporal lengths16→14→12→6, Conv dropout .25; two-layer LSTM hidden128, internal recurrent dropout0, external output dropout .25, last output, dense128→64(ReLU)→3.

CNN-only chooses ONE fixed global temporal mean readout, producing64features from the same convolution stack. It retains external readout dropout .25 and dense64→64→3; no flatten/readout sweep. LSTM-only feeds exactly permuted (B,16,5) values to two-layer hidden128 LSTM, external dropout then last temporal output; dense128→64→3. CNN-GRU replaces only the recurrent block with two-layer hidden128 GRU; same convolution, external dropout, last-output readout and dense head. CNN-BiLSTM uses two layers, hidden64/direction; concatenates top-layer final forward and final reverse hidden summaries, yielding128features, external dropout .25 and the same dense head. This avoids the incomplete reverse summary in a last-position bidirectional output. It can use later positions within the already completed window and does not see future frames beyond that window.

Initialization policy is fixed across models: PyTorch module construction followed by Conv He/ReLU, dense Xavier, recurrent input He/linear, recurrent hidden orthogonal, all biases zero. Structural differences imply different weight tensors/RNG consumption; same seed/policy and data order rules are used, not an impossible identical-weight requirement across architectures. Train seeds1201/1202/1203, three folds, five models:45runs. All9reference runs completed and passed exact Phase12reproduction BEFORE the36alternative runs. Technical failures would be persisted as failure.json and stop the study; **none occurred**. Best checkpoints only are retained per run; raw histories, best/stopping epochs, selected validation metrics/probabilities and configurations remain auditable. No final model fit on all153sources.

## 14–22. Development quality, variability and class trade-offs

Equal averaging of nine fold/seed runs per architecture. Window macro F1 is primary; checkpoint epochs selected only on fold validation macro F1. Video mean window probabilities→argmax, one decision per source; 51 validation sources/fold. Per-run and per-class raw CSV/JSON is authoritative, not a single best run. SD below is sample SD (ddof=1), describing correlated folds/seeds; no independence/significance claim.

{table(quality)}
{table(classrows)}
{table(variability)}
## 23–24. Development confusion and paired differences

Rows actual, columns predicted Normal/Yawning/Eye Close. Separate raw CSV matrices sum three validation folds for each fixed training seed. Pooled tables additionally sum three seeds:8043window predictions /459video predictions, representing2681unique development windows /153videos each predicted three times. These repeated predictions are not independent subjects and are not test matrices.
{matrix_text}
Each paired delta is candidate minus CNN-LSTM at identical fold/seed. Raw9deltas/model are in paired_comparisons.csv; summaries below include mean, SD, min/max, positive/negative/zero pairs.

{table(pairs)}
## 25–27. Model-only CPU latency, sizes and Pareto trade-off

{table(sizes)}
Latency is **prepared batch-1 tensor→model logits on CPU**, excluding decode, MediaPipe, features, rendering, SQLite, audio, loader, checkpoint loading and normalization. Thread counts fixed8/8 for every architecture. Shared64development tensors from fold0validation, deterministic seed1301, source/window/frame IDs and exact tensor hashes persisted in latency_inputs.json and latency_inputs.npz. All candidates see the same tensor order. Each of45best checkpoints has128warm-up calls and1024timed calls (16cycles of64inputs). Jobs have predeclared seed1302 shuffled order; no concurrent agent training/tests/production regression during measurement. All logits evaluated in eval/inference_mode. Individual raw timing samples are in latency_samples.npz; per-checkpoint quantiles and pooled9216samples/architecture are in latency_results.json. Quantiles use NumPy percentile (linear interpolation). Checkpoint bytes reflect serialized state overhead and are separate from parameter counts. No batch64headline or FLOP estimate was used.

Descriptive non-dominated quality/P50 set: **{', '.join(result['pareto_quality_latency'])}**. A dominates B only if mean window F1≥ and pooled P50≤, with at least one strict inequality. No arbitrary composite score. Plots show variability and quality versus latency/parameters; proximity is not statistical equivalence. Tail quantiles are reported rather than hidden by a mean. Host CPU/OS scheduling, thread overhead and background system activity can affect timing; one hardware/software environment, no realtime deadline guarantee. Per-checkpoint latency distributions are also available.

### Preserved timing anomaly and separate full-design confirmation

The primary `cnn_lstm_fold_1_seed_1203` sample 29 took **423,121.8114 ms** (about 423 seconds), with neighboring calls around1–3ms. Every sample is retained: this single wall-clock stall dominates the primary pooled CNN-LSTM mean 47.614986 ms, while its pooled P50/P95/P99 are 1.736300/2.453700/4.085615 ms. A first run's extreme stall is not evidence that ordinary CNN-LSTM computation costs 47.6 ms; its underlying cause is unverified because that pass did not record process CPU time. There were two samples above 100 ms in that checkpoint. No sample was trimmed, no architecture was retrained and no recommendation threshold was changed.

After the complete regression suite/parity finished, all 45 timing jobs were repeated once with the SAME weights, inputs, seed/order, threads, warm-up and repetitions. This separate confirmation records process CPU time as an additional coarse scheduling diagnostic and preserves the primary data. Confirmation results are under latency_confirmation, including protocol, raw wall/process-CPU samples and per-checkpoint results. They are not used to substitute a favorable primary P50 or to alter the predeclared recommendation rules. The extra timer calls add a small diagnostic overhead outside the wall interval; raw wall latency remains authoritative.

{confirmation_table}
Confirmation quality/P50 Pareto set: **{', '.join(result['confirmation_pareto_quality_latency'])}**. Primary and confirmation differences describe host variability rather than a new tuning experiment. All measured means and tails are disclosed; no whole-application speedup is claimed.

## 28–33. Engineering interpretation and later recommendations

- CNN-only vs CNN-LSTM: removing recurrence changes mean window F1 by **{deltas['cnn_only']:+.6f}**. Its mean is close to the control in the predeclared practical margin and its model is much smaller/faster, but class recall and paired consistency still determine acceptance. Near means alone do not establish equivalence or justify removing recurrence; this is one fixed global-mean CNN readout.
- LSTM-only vs CNN-LSTM: removing convolution changes mean F1 by **{deltas['lstm_only']:+.6f}**. Under this fixed recipe, LSTM-only has higher mean quality, so this study does not justify the CNN front-end as necessary for classification quality. It processes16raw temporal positions rather than6convolution outputs; primary P50 is **{lstm_ratio:.3f}×** the CNN-LSTM value. Convolution affects representation and compute, and this result does not prove it useless across other recipes.
- CNN-GRU vs CNN-LSTM: mean F1 change **{deltas['cnn_gru']:+.6f}**, parameters188739 vs246595; primary P50 is **{gru_ratio:.3f}×** reference. The smaller GRU therefore does not give lower median CPU latency in this measurement, despite its quality/size benefit. Fewer parameters alone are not a speed claim.
- CNN-BiLSTM vs CNN-LSTM: mean F1 change **{deltas['cnn_bilstm']:+.6f}**, parameters181059 vs246595. Two-direction completed-window context and smaller per-direction states change the architecture; no claim of future-frame access beyond the current window or biological causality.

Recommendation guidance was locked before training: an efficient candidate requires mean F1 within0.002 of control, at least5/9paired deltas within that margin, and at least20%lower pooled P50 or parameter count. Alternatively, a strong-quality candidate requires mean F1 gain≥0.005 and at least5/9positive pairs; higher cost is disclosed for later judgment. Both routes require mean worst-class recall regression≤0.005, no class mean recall regression>0.01, and F1sampleSD increase≤0.01. This is practical engineering guidance, not statistical significance or authorization to promote.

{decision}

Reference CNN-LSTM remains the verified control. Additional architectures recommended for later Phase14/15consideration: **{', '.join(recommendations) if recommendations else 'none under the fixed guidance'}**. These recommendations and the descriptive Pareto set answer different questions; an efficient Pareto point can still fail recall/stability safeguards. **No architecture was promoted**, no production state or normalizer replaced, and no new test or resume metric claimed. Phase11clean final test metrics remain historical and are not reused for candidate selection. No Phase14study begun.

Phase9established MediaPipe at approximately83%of raw runtime. These new model-only numbers cannot establish whole-application speedup; even a faster model may produce only modest end-to-end improvement. The complete Phase9/SQLite benchmarks were not repeated.

## 34–37. Access audit, regression and limitations

**Phase11test model inference passes during Phase13=0.** test_access_audit.json and45-entry candidate_access_audit.jsonl persist exact candidate source IDs and the zero-test invariant. Latency inputs also contain development sources only. No sealed cache/video/tensor/metric is opened by candidate training, validation or latency code. No candidate test_prediction files, test confusion matrices or test-driven decisions exist. Source guards, immutable artifact hashes, file-open refusal inherited from Phase12 and explicit validation coverage checks enforce this technically. Report analysis reconstructs raw development data and recomputes saved validation metrics independently with sklearn; it does not run candidate inference on any test or validation source.

Full suite: **{tests}passed**, comprising all475Phase2–12tests and{newtests}new Phase13tests; no failures/errors/skips. Fresh production regression **699/699windows and classes exact**, max difference0 across compared fields, using existing frozen production fixtures. That authorized production-model check is separate from candidate inference on the sealed Phase11cohort. All old artifacts and runtime preserved byte-for-byte.

Limitations: video independence only, unknown driver overlap, single dataset and single hardware/software environment; validation optimizes epochs and assesses architectures, so it is not untouched final evaluation. Folds share training videos and seeds reuse validation observations; SD/paired comparisons are descriptive correlated runs. Fixed equal recipe isolates this study but does not find each architecture's optimal hyperparameters; global-mean CNN and64/direction BiLSTM are the single predeclared designs. Whole-video labels do not validate event timing, clinical safety or alert accuracy. Model-only timing excludes the dominant vision pipeline and cannot predict deployment latency. No architecture/data/window/preprocessing tuning, new subjects, all-development refit or final test occurred.

## 38–39. Reproduction and status

Keep the frozen Phase11/12artifacts and recorded software environment. Commands below target NEW output paths; canonical artifacts refuse overwrite. From repository root:

```powershell
$env:MPLCONFIGDIR = '.cache/matplotlib'
.venv-baseline/Scripts/python.exe -B -m evaluation.architectures prepare --output results/phase13_reproduction
.venv-baseline/Scripts/python.exe -B -m evaluation.architectures run --output results/phase13_reproduction
.venv-baseline/Scripts/python.exe -B -m evaluation.architectures latency --output results/phase13_reproduction
.venv-baseline/Scripts/python.exe -B -m pytest -q --junitxml=docs/phase13_tests_reproduction.xml
.venv-baseline/Scripts/python.exe -B tools/optimize_phase9.py --parity production --output results/phase13_production_parity_reproduction/result.json
```

Canonical execution used the same prepare/run/latency commands with results/phase13, full-suite docs/phase13_tests.xml, production parity results/phase13_production_parity/result.json, then tools/report_phase13.py. The reporter reads canonical paths and writes new-only outputs, so do not run it over completed report artifacts. Reproduction histories, weights and saved validation probabilities can be compared directly; byte-level checkpoint serialization hashes and latency samples need not repeat. Control reproducibility was verified across two phases on this CPU, not all environments. No test inference command exists in this workflow.

The separate anomaly investigation used `.venv-baseline/Scripts/python.exe -B tools/confirm_phase13_latency.py` after regression. It reads canonical trained checkpoints and writes a NEW latency_confirmation directory, refusing overwrite. It does not train models or modify the primary benchmark. Keep this confirmation optional when reproducing the original predeclared design; retain both passes whenever investigating the reported stall.

**PHASE13PASS. Stop after Phase13; Phase14not begun.**

## Fixed layer definitions (predeclared)
{raw}
'''
    fixes={'stride8':'stride 8','batch64':'batch 64','max60epochs':'max 60 epochs','patience10':'patience 10','improvement1e-12':'improvement 1e-12','interop8/8':'interop 8/8','num_workers0':'num_workers 0','246595':'246,595',
           'dense128':'dense 128','dense64':'dense 64','hidden128':'hidden 128','hidden64':'hidden 64','producing64':'producing 64','yielding128':'yielding 128','layers16':'layers 16','lengths16':'lengths 16','seed1301':'seed 1301','seed1302':'seed 1302',
           '64features':'64 features','dropout0':'dropout 0','seeds1201':'seeds 1201','models:45runs':'models: 45 runs','All9reference':'All 9 reference','Phase12reproduction':'Phase 12 reproduction','the36alternative':'the 36 alternative','all153sources':'all 153 sources',
           'fold0validation':'fold 0 validation','Shared64development':'Shared 64 development','of45best':'of 45 best','has128warm':'has 128 warm','and1024timed':'and 1,024 timed','(16cycles of64inputs)':'(16 cycles of 64 inputs)','pooled9216samples':'pooled 9,216 samples',
           'samples/architecture':'samples/architecture','within0.002':'within 0.002','least5/9paired':'least 5/9 paired','least20%lower':'least 20% lower','least5/9positive':'least 5/9 positive','F1sampleSD':'F1 sample SD','Phase14/15consideration':'Phase 14/15 consideration',
           'Phase11clean':'Phase 11 clean','Phase14study':'Phase 14 study','Phase9established':'Phase 9 established','approximately83%of':'approximately 83% of','Phase9/SQLite':'Phase 9/SQLite','Phase11test':'Phase 11 test','Phase13=0':'Phase 13 = 0',
           'and45-entry':'and 45-entry','Phase12 and':'Phase 12 and','{tests}passed':'{tests} passed','all475Phase2–12tests':'all 475 Phase 2–12 tests','and{newtests}new':'and {newtests} new','Phase13tests':'Phase 13 tests','699/699windows':'699/699 windows','difference0':'difference 0','Phase11cohort':'Phase 11 cohort',
           'Phase11/12artifacts':'Phase 11/12 artifacts','PHASE13PASS':'PHASE 13 PASS','after Phase13':'after Phase 13','Phase14not':'Phase 14 not','8043window':'8,043 window','/459video':' / 459 video','representing2681unique':'representing 2,681 unique','/153videos':' / 153 videos','Raw9deltas/model':'Raw 9 deltas/model',
           'parameters188739':'parameters 188,739','vs246595':'vs 246,595','parameters181059':'parameters 181,059','processes16raw':'processes 16 raw','than6convolution':'than 6 convolution','and64/direction':'and 64/direction'}
    for a,b in fixes.items():text=text.replace(a,b)
    text=text.replace(f'{tests}passed',f'{tests} passed').replace(f'and{newtests}new',f'and {newtests} new')
    with (ROOT/'docs/PHASE13_ARCHITECTURE_ABLATIONS.md').open('x',encoding='utf-8') as handle:handle.write(text)


def main():
    result,runs,classes,paired=analyze()
    xml=ET.parse(ROOT/'docs/phase13_tests.xml');cases=xml.findall('.//testcase')
    assert not any(xml.findall('.//'+t) for t in ('failure','error','skipped'))
    new=sum(c.attrib['classname']=='tests.test_phase13_architectures' for c in cases)
    assert len(cases)-new==475
    parity=load(ROOT/'results/phase13_production_parity/result.json')
    assert parity['exact_parity'] and parity['windows_exact']==parity['classes_exact']==699
    assert all(all(v==0 for v in s['maximum_differences'].values()) for s in parity['sources'])
    before=load(ROOT/'docs/phase13_before_hashes.json')
    changed={name:sha256(ROOT/name) for name,expected in before.items() if sha256(ROOT/name)!=expected}
    assert not changed,changed
    csv_out(D/'run_results.csv',runs);csv_out(D/'per_class_results.csv',classes);csv_out(D/'paired_comparisons.csv',paired)
    (D/'confusion_matrices').mkdir()
    for n,levels in result['confusion_matrices'].items():
        for level,matrices in levels.items():
            for seed,matrix in matrices['per_seed'].items():
                csv_out(D/'confusion_matrices'/f'{n}_{level}_seed_{seed}.csv',[{'actual':c,**dict(zip(('Normal','Yawning','Eye Close'),row))} for c,row in zip(('Normal','Yawning','Eye Close'),matrix)])
    write_json(D/'aggregate_results.json',result)
    plots(result,runs,paired)
    write_json(ROOT/'docs/phase13_validation.json',{'status':'PHASE13_PASS','tests':len(cases),'new_tests':new,'prior_files_preserved':len(before),'changed_prior_files':changed,
               'completed_runs':45,'reference_control_runs_exact':9,'candidate_test_inference_passes':0,'production_windows_exact':699,
               'fold_norms_and_windows_reconstructed':True,'metrics_recomputed_and_sklearn_crosschecked':True,'latency_quantiles_recomputed':True,
               'phase14_started':False,'pareto_quality_latency':result['pareto_quality_latency']})
    report(result,runs,len(cases),new,len(before))
    created=sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and not any(x.startswith('.venv') or x in ('.git','.cache','__pycache__','.pytest_cache') for x in p.relative_to(ROOT).parts) and p.relative_to(ROOT).as_posix() not in before)
    write_json(D/'created_files.json',created+['results/phase13/created_files.json'])
    print(json.dumps({'status':'PHASE13_PASS','tests':len(cases),'quality':result['aggregates'],'pareto':result['pareto_quality_latency'],'decisions':result['decisions']},indent=2))


if __name__=='__main__':main()
