"""Train only a separate copy of FAST on frozen, cached layer-16 features."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from .gate_reference_training import (Checkpoints, _atomic_torch, _graceful_interrupt,
    _learning_rate, atomic_json, cpu_state, weights_sha)
from .progress import tqdm


@dataclass(frozen=True)
class GateSettings:
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = .001
    weight_decay: float = .01
    warmup_fraction: float = .05
    clip_norm: float = 1.
    checkpoint_every: int = 25
    threshold: float = .5
    seed: int = 42

    def validate(self):
        if min(self.epochs,self.batch_size,self.checkpoint_every)<1 or self.threshold!=.5:
            raise ValueError('Invalid counts or changed preregistered threshold')
        if not 0<=self.warmup_fraction<1 or self.learning_rate<=0 or self.weight_decay<0:
            raise ValueError('Invalid optimizer settings')


@torch.inference_mode()
def predictions(head, features, indices, batch_size=256):
    return torch.cat([head(features['x'][part]).squeeze(-1).float().cpu()
                      for part in torch.as_tensor(indices).split(batch_size)])


def routing_metrics(logits, rows, threshold=.5):
    if len(logits)!=len(rows) or not rows or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite aligned routing predictions are required')
    y=torch.tensor([r['mode_target'] for r in rows],dtype=torch.bool)
    route=logits.sigmoid()>=threshold
    def score(mask):
        n=int(mask.sum()); positives=int((mask&y).sum()); negatives=n-positives
        tp=int((mask&y&route).sum()); fp=int((mask&~y&route).sum())
        return dict(cases=n,decision_cases=positives,writing_cases=negatives,
            correct=int((mask&(route==y)).sum()),accuracy=float((route[mask]==y[mask]).float().mean()),
            correct_decisions=tp,missed_decisions=positives-tp,false_fast=fp,
            decision_recall=tp/positives if positives else None,
            writing_error_rate=fp/negatives if negatives else None)
    total=score(torch.ones(len(rows),dtype=torch.bool))
    by_format={}
    for style in sorted({r.get('schema_family',r.get('style',r.get('input_format','unknown'))) for r in rows}):
        mask=torch.tensor([r.get('schema_family',r.get('style',r.get('input_format','unknown')))==style for r in rows])
        by_format[style]=score(mask)
    recalls=[m['decision_recall'] for m in by_format.values() if m['decision_recall'] is not None]
    total.update(loss=float(F.binary_cross_entropy_with_logits(logits.float(),y.float())),
                 macro_decision_recall=sum(recalls)/len(recalls) if recalls else None,
                 worst_format_recall=min(recalls) if recalls else None,by_format=by_format)
    return total


def validation_result(head, features, indices, datasets, baseline_old, batch_size=256):
    result={name:routing_metrics(predictions(head,features,indices[name],batch_size),datasets[name])
            for name in ('old_validation','new_validation')}
    old,new=result['old_validation'],result['new_validation']
    eligible=(old['accuracy']+1e-12>=baseline_old['accuracy']-.0025 and
              old['writing_error_rate']<=.01 and new['writing_error_rate']<=.01)
    return dict(**result,eligible=eligible,selection_score=new['macro_decision_recall'],
                selection_loss=(old['loss']+new['loss'])/2)


def better(candidate, current):
    return current is None or (candidate['selection_score'],-candidate['selection_loss']) > (
        current['selection_score'],-current['selection_loss'])


def fit_gate(features, train_indices, datasets, indices, initial, folder, settings, identity, stop_after=None):
    """No reference model/LoRA object is accepted by this optimizer boundary."""
    settings.validate()
    if not train_indices:
        raise ValueError('Missing training data')
    validation_indices = set(indices['old_validation']) | set(indices['new_validation'])
    if validation_indices.intersection(train_indices):
        raise ValueError('Training and validation feature rows overlap')
    if set(features['y'][train_indices].tolist()) != {0., 1.}:
        raise ValueError('Both routing labels are required in training')
    head=nn.Linear(features['x'].shape[1],1,dtype=torch.float32,device='cpu')
    head.load_state_dict(initial,strict=True)
    initial_sha=weights_sha(initial)
    baseline_old=routing_metrics(predictions(head,features,indices['old_validation']),datasets['old_validation'])
    run_identity=dict(identity=identity,settings=asdict(settings),initial_gate=initial_sha,
                      features=features['identity_key'],training_ids=[features['ids'][i] for i in train_indices],
                      validation_ids={k:[features['ids'][i] for i in indices[k]] for k in ('old_validation','new_validation')})
    store=Checkpoints(folder,run_identity,retain=3)
    total=math.ceil(len(train_indices)/settings.batch_size)*settings.epochs
    optimizer=torch.optim.AdamW(head.parameters(),lr=settings.learning_rate,weight_decay=settings.weight_decay)
    saved=store.latest()
    if saved is None:
        state=dict(epoch=0,cursor=0,updates=0,examples_seen=0,micro_in_window=0,best_epoch=None,
            completed=False,history=[],best_overall=None,best_eligible=None,epoch_loss_sum=0.,epoch_examples=0)
        best_overall_weights=best_eligible_weights=None
        print(f'NEW FAST: {folder}; initial weights {initial_sha[:12]}',flush=True)
    else:
        if saved['total_updates']!=total: raise ValueError('Training schedule changed')
        head.load_state_dict(saved['weights']);optimizer.load_state_dict(saved['optimizer'])
        state=saved['state'];best_overall_weights=saved['best_overall_weights'];best_eligible_weights=saved['best_eligible_weights']
        torch.set_rng_state(saved['torch_rng'])
        print(f'RESUME FAST: {folder}; epoch {state["epoch"]}, update {state["updates"]}, cursor {state["cursor"]}',flush=True)
    def checkpoint():
        store.save(dict(weights=cpu_state(head),optimizer=optimizer.state_dict(),state=state,
            best_overall_weights=best_overall_weights,best_eligible_weights=best_eligible_weights,
            torch_rng=torch.get_rng_state(),total_updates=total))
    if saved is None: checkpoint()
    updates_at_start=state['updates']
    with _graceful_interrupt() as interrupt:
        try:
            while not state['completed']:
                order=list(train_indices);random.Random(settings.seed+1000+state['epoch']).shuffle(order)
                with tqdm(total=len(order),initial=state['cursor'],desc=f'FAST seed {settings.seed}, epoch {state["epoch"]+1}/{settings.epochs}',unit='request') as bar:
                    while state['cursor']<len(order):
                        part=order[state['cursor']:state['cursor']+settings.batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        logits=head(features['x'][part]).squeeze(-1)
                        loss=F.binary_cross_entropy_with_logits(logits,features['y'][part])
                        if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite gate loss')
                        loss.backward()
                        norm=nn.utils.clip_grad_norm_(head.parameters(),settings.clip_norm)
                        if not bool(torch.isfinite(norm)): raise FloatingPointError('Non-finite gate gradient')
                        lr=_learning_rate(state['updates'],total,settings.learning_rate,settings.warmup_fraction)
                        for group in optimizer.param_groups: group['lr']=lr
                        optimizer.step()
                        state['cursor']+=len(part);state['updates']+=1;state['examples_seen']+=len(part)
                        state['epoch_loss_sum']+=float(loss.detach())*len(part);state['epoch_examples']+=len(part)
                        bar.update(len(part));bar.set_postfix(loss=f'{float(loss.detach()):.4f}',update=state['updates'])
                        if state['updates']%settings.checkpoint_every==0 or state['cursor']==len(order): checkpoint()
                        if interrupt['requested'] or (stop_after is not None and state['updates']-updates_at_start>=stop_after):
                            checkpoint();return None
                metrics=validation_result(head,features,indices,datasets,baseline_old,settings.batch_size)
                metrics.update(epoch=state['epoch']+1,training_loss=state['epoch_loss_sum']/state['epoch_examples'])
                if better(metrics,state['best_overall']):
                    state['best_overall']=metrics;best_overall_weights=cpu_state(head)
                if metrics['eligible'] and better(metrics,state['best_eligible']):
                    state['best_eligible']=metrics;best_eligible_weights=cpu_state(head)
                state['history'].append(metrics);state['epoch']+=1;state['cursor']=0
                state['epoch_loss_sum']=0.;state['epoch_examples']=0
                state['completed']=state['epoch']==settings.epochs
                selected=state['best_eligible'] or state['best_overall']
                state['best_epoch']=selected['epoch']
                checkpoint()
                print(f'Epoch {metrics["epoch"]}: new decision recall {metrics["new_validation"]["decision_recall"]:.1%}; '
                      f'writing errors {metrics["new_validation"]["writing_error_rate"]:.1%}; retention eligible={metrics["eligible"]}',flush=True)
        except KeyboardInterrupt:
            # The signal context normally stops between updates; preserve current state on a manual exception too.
            checkpoint();raise
    selected=state['best_eligible'] or state['best_overall']
    weights=best_eligible_weights if state['best_eligible'] is not None else best_overall_weights
    summary=dict(completed=True,selected_epoch=selected['epoch'],selected_metrics=selected,
                 eligible_on_validation=state['best_eligible'] is not None,history=state['history'],
                 selection='best eligible trained epoch' if state['best_eligible'] else 'best trained epoch; failed validation constraints',
                 initial_gate_sha256=initial_sha,selected_gate_sha256=weights_sha(weights),settings=asdict(settings),
                 updates=state['updates'],examples_seen=state['examples_seen'],baseline_old=baseline_old)
    metadata=dict(schema=1,model='Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        gate_after=16,hidden_size=head.in_features,parameters=sum(p.numel() for p in head.parameters()),
        threshold=.5,features='RMS-normalized last prompt state after frozen decoder layer 16',
        selected_gate_sha256=weights_sha(weights),eligible_on_validation=summary['eligible_on_validation'],
        seed=settings.seed,selected_epoch=selected['epoch'],identity=identity,
        status='candidate; final tests determine promotion eligibility')
    _atomic_torch(Path(folder)/'candidate.pt',{'weights':weights,'metadata':metadata})
    atomic_json(Path(folder)/'candidate.json',metadata)
    atomic_json(Path(folder)/'summary.json',summary)
    return dict(weights=weights,metadata=metadata,summary=summary)
