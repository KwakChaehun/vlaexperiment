"""Static research figure for the approach-covered feature prediction pilot."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/vla-feature-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',default='reports/direct_arrival_features_20261117')
    args=parser.parse_args();out=Path(args.report);data=json.loads((out/'aggregate.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    for method in ['stale','direct','corrected','oracle_rank32']:
        rr=sorted([r for r in data['offline'] if r['method']==method],key=lambda r:r['horizon'])
        ax[0].plot([r['horizon']/20 for r in rr],[r['feature_energy_vs_stale'] for r in rr],
                   marker='o',linestyle='--' if method.startswith('oracle') else '-',label=method)
    ax[0].set(xlabel='Forecast horizon (s)',ylabel='Feature residual energy / stale',title='249 pairs from 12 held-out episodes')
    ax[0].legend(fontsize=8);ax[0].grid(alpha=.15)
    methods=['fresh4','fresh2','fresh1','stale','direct','corrected'];x=np.arange(len(methods))
    for offset,key,label in [(-.18,'successes','Native success'),(.18,'stable_successes','Maintained for 10 more steps')]:
        rr=[next(r for r in data['closed'] if r['method']==m) for m in methods]
        bars=ax[1].bar(x+offset,[r[key] for r in rr],.36,label=label)
        ax[1].bar_label(bars,fontsize=8,padding=2)
    ax[1].set(xticks=x,xticklabels=methods,ylim=(0,12.5),ylabel='Successful conditions / 12',title='Forecast used from the FIRST arrival')
    ax[1].tick_params(axis='x',rotation=25);ax[1].legend(fontsize=8,loc='lower left')
    fig.suptitle('Direct future-feature prediction: feature gains do not ensure control gains',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg')


if __name__=='__main__':main()
