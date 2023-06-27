
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import pandas as pd
from typing import Union, Optional, List
from cycler import cycler


class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value
        
class Labels:
    
    abs_label = {
        'C4': r'${\rm C}^4_{p4mm}$',
        'C6': r'${\rm C}^6_{p6mm}$',
        'Crect': r'${\rm C}^2_{p2mm}$',
        'C3': r'${\rm C}^3_{p3m1}$',

        'freeE': r"$\rm{F} / \rm{nk_B T}$",
        'freeAB': r"$\rm{U} / \rm{nk_B T}$",
        'freeWS': r"$\rm{-TS} / \rm{nk_B T}$",
        'bridge': r"$v_B$",
        'width': r'$\bar{w}$',
        'ksi': r"$\xi$",
        'tau': r'$\tau$',
        'chiN': r'$\chi N$',
        'ly': r'$L_x/R_g$',
        'lz': r'$L_y/R_g$',
        'gamma_B': r'$\gamma _{\rm B}$',
        'phi_AB': r'$\phi _{\rm AB}$',
    }
    
    ref_label = {
        
        'C4': r'${\rm C}^4_{p4mm}$',
        'C6': r'${\rm C}^6_{p6mm}$',
        'Crect': r'${\rm C}^2_{p2mm}$',
        'C3': r'$\rm{C}^3_{p3m1}$',

        'freeE': r"$\Delta F / nk_B T$",
        'freeAB': r"$\Delta U / nk_B T$",
        'freeWS': r"$-T\Delta S / nk_B T$",

        'width': r'${\rm \Delta} \bar{w_I}$',
        'lylz': 'Lx/Ly',
        'bridge': r"$\Delta v_B$",
        'chiN': r'$\chi N$',
        'ksi': r"$\xi$",
        'tau': r"$\tau$",
        
        'gamma_B': r'$\gamma _{\rm B}$',
        'phi_AB': r'$\phi _{\rm AB}$',
        
        'freeAB1': r'$\rm A/B_1$',
        'freeAB2': r'$\rm A/B_2$',
    }


_style_ref = (cycler(color=list('rbm') + [
                'indianred', 'tomato', 'chocolate',
                'olivedrab', 'teal', 'deepskyblue', 'darkviolet']) +
          cycler(marker=list('sXvoP*D><p')))

_style_abs = (cycler(color=list('krbm') + [
                'indianred', 'tomato', 'chocolate',
                'olivedrab', 'teal', 'deepskyblue']) +
              cycler(marker=list('osXvP*D><p')))

class CompareJudger():
    
    def __init__(self, 
                 path:str,
                 div:Union[int, float]=1,
                 acc:int=3,
                 ) -> None:
        
        self.path = path
        self.div = div
        self.acc = acc
        pass


    def data(self, subset=['lx', 'ly', 'lz', 'phase', 'freeE'], **kwargs):
        df = pd.read_csv(self.path)
        df = df.dropna(axis=0)
        df = df.drop_duplicates(subset=subset)
        df['lylz'] = np.around(df['ly']/df['lz'], self.acc)
        df['lxly'] = np.around(df['lx']/df['ly'], self.acc)
        try:
            df['chiN'] = df['chiN']/self.div
        except KeyError:
            pass
        funcs = kwargs.get('funcs', None)
        if funcs is not None:
            if isinstance(funcs, list):
                for func in funcs:
                    df = func(df)
            else:
                df = funcs(df)
        return df
    
    def ref_compare(self, base, others, xlabel, ylabel:Union[List[str], str], ax:Optional[object]=None, horiline:bool=False, **kwargs):
        
        ref_labels = Labels.ref_label.copy()
        ref_labels.update(kwargs.get('labels', {}))
        
        data = self.data()
        data = data.sort_values(by=xlabel)
        
        if isinstance(ylabel, list):
            data['tmp'] = np.sum([data[i] for i in ylabel], axis=0)
            print(f'请为标签{"+".join(ylabel)}指定一个名称')
            ylabel = 'tmp'
        
        plt.figure(figsize=kwargs.get('figsize', (9, 6.5)))
        if not ax:
            ax = plt.gca()
            ax.set_prop_cycle(_style_ref)
        base_data = data[data['phase'] == base]
        
        base_xticks = base_data[xlabel].values
        base_yticks = base_data[ylabel].values
        
        others = [others] if isinstance(others, str) else others
        
        for o in others:
            o_data = data[data['phase'] == o]
            o_xticks = o_data[xlabel].values
            o_yticks = o_data[ylabel].values
            mask = np.in1d(o_xticks, base_xticks)
            o_xticks = o_xticks[mask]
            o_yticks = o_yticks[mask]
            
            inverse_mask = np.in1d(base_xticks, o_xticks)
            # base_xticks_mask = base_xticks[inverse_mask]
            base_yticks_mask = base_yticks[inverse_mask]
            ax.plot(o_xticks, o_yticks-base_yticks_mask, 
                    label=ref_labels.get(o, o),
                     lw=2.5, markersize=8, alpha=1.0)
        
        if horiline:
            rest = data[data['phase'] != base]
            rest_xticks = rest[xlabel].unique()
            mask = np.in1d(base_xticks, rest_xticks)
            base_xticks_mask = base_xticks[mask]
            ax.plot(
                base_xticks_mask, np.zeros_like(base_xticks_mask), 
                label=ref_labels.get(base, base),
                lw=2.5, c='k', marker='o', markersize=8, alpha=0.8)
        
        if trans := kwargs.get('trans'):
            for t in trans:
                ax.axvline(x=t, c='k', alpha=0.5, ls='--', lw=2.5)
        if xminor := kwargs.get('xminor', 5):
            ax.xaxis.set_minor_locator(AutoMinorLocator(xminor))
        if yminor := kwargs.get('yminor', 5):
            ax.yaxis.set_minor_locator(AutoMinorLocator(yminor))
        if xmain := kwargs.get('xmain', False):
            ax.yaxis.set_major_locator(MultipleLocator(xmain))
        if ymain := kwargs.get('ymain', False):
            ax.yaxis.set_major_locator(MultipleLocator(ymain))
        
        plt.tick_params(axis='both', labelsize=25, pad=8)
        plt.ylabel(ref_labels.get(ylabel, ylabel), fontsize=30)
        plt.xlabel(ref_labels.get(xlabel, xlabel), fontsize=30)
        if loc := kwargs.get('legend', 'in'):
            if loc == 'in':
                plt.legend(fontsize=25, loc='best')
            elif loc == 'out':
                plt.legend(fontsize=25, loc='upper left', bbox_to_anchor=(1, 1))
        plt.margins(*kwargs.get('margin',(0.15,0.15)))
        plt.tight_layout()
        if save := kwargs.get('save', False):
            plt.savefig(save, dpi=300)

    def abs_compare(self, phases: Union[list, str], xlabel, ylabel:Union[str, List[str]], **kwargs):
        
        abs_labels = Labels.abs_label.copy()
        abs_labels.update(kwargs.get('labels', {}))

        data = self.data()
        data = data.sort_values(by=xlabel)
        
        if isinstance(ylabel, list):
            data['tmp'] = np.sum([data[i] for i in ylabel], axis=0)
            print(f'请为标签{"+".join(ylabel)}指定一个名称')
            ylabel = 'tmp'
            
        
        plt.figure(figsize=kwargs.get('figsize', (9, 6.5)))
        ax = plt.gca()
        ax.set_prop_cycle(_style_abs)
        
        phases = [phases] if isinstance(phases, str) else phases
        
        for p in phases:
            tmp = data[data.phase == p]
            ax.plot(tmp[xlabel], tmp[ylabel],
                    label=abs_labels.get(p, p),
                    lw=2.5, markersize=8)
            
        plt.xlabel(abs_labels.get(xlabel, xlabel), fontsize=30)
        plt.ylabel(abs_labels.get(ylabel, ylabel), fontsize=30)
        plt.tick_params(axis='both', labelsize=25, pad=8)
        if xminor := kwargs.get('xminor', 5):
            ax.xaxis.set_minor_locator(AutoMinorLocator(xminor))
        if yminor := kwargs.get('yminor', 5):
            ax.yaxis.set_minor_locator(AutoMinorLocator(yminor))
        if xmain := kwargs.get('xmain', False):
            ax.yaxis.set_major_locator(MultipleLocator(xmain))
        if ymain := kwargs.get('ymain', False):
            ax.yaxis.set_major_locator(MultipleLocator(ymain))
        if trans := kwargs.get('trans'):
            for i in trans:
                plt.axvline(x=i, c='k', alpha=0.5, ls='--', lw=2.5)
            
        if kwargs.get('legend', True):
            plt.legend(fontsize=25, loc='upper left',bbox_to_anchor=(0.98, 0.8))

        plt.margins(*kwargs.get('margin', (0.15, 0.15)))
        plt.tight_layout()
        if save := kwargs.get('save', False):
            plt.savefig(save, dpi=300)
    
    def multi_target_ref(self, base, other, xlabel, ylabels, 
                         ylabel_name,
                         **kwargs):
        
        data = self.data
        data = data.sort_values(by=xlabel)
        
        plt.figure(figsize=kwargs.get('figsize', (9, 6.5)))
        ax = plt.gca()
        ax.set_prop_cycle(_style_ref)
        
        ylabels = ylabels if isinstance(ylabels, list) else [ylabels]
        base_data = data[data['phase'] == base]
        base_xticks = base_data[xlabel].values
        
        for yl in ylabels:
            # self.ref_compare(base, other, xlabel, yl, ax=ax, horiline=False)
            base_yticks = base_data[yl].values
            o_data = data[data['phase'] == other]
            o_xticks = o_data[xlabel].values
            o_yticks = o_data[yl].values
            
            mask = np.in1d(o_xticks, base_xticks)
            o_xticks = o_xticks[mask]
            o_yticks = o_yticks[mask]

            inverse_mask = np.in1d(base_xticks, o_xticks)
            # base_xticks_mask = base_xticks[inverse_mask]
            base_yticks_mask = base_yticks[inverse_mask]
            ax.plot(o_xticks, o_yticks-base_yticks_mask,
                    label=Labels.ref_label[yl],
                    lw=2.5, markersize=8, alpha=1.0)
            
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.axhline(y=0, c='k', ls=':', lw=4, alpha=0.5)
        ax.xaxis.set_major_locator(MultipleLocator(kwargs.get('xmain', 5)))
        plt.tick_params(axis='both', labelsize=25, pad=8)
        plt.tick_params(axis='both', labelsize=25)
        plt.ylabel(Labels.ref_label.get(ylabel_name, ylabel_name), fontsize=30)
        plt.xlabel(Labels.ref_label.get(xlabel, xlabel), fontsize=30)

        if kwargs.get('legend', True):
            plt.legend(fontsize=25)
        plt.margins(*kwargs.get('margin', (0.15, 0.15)))
        plt.tight_layout()
        if save := kwargs.get('save', False):
            plt.savefig(save, dpi=300)
    

