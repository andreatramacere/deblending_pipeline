import pandas
import os
import glob
from pandas import DataFrame
from itertools import product
import numpy as np
import seaborn as sns
import  pylab as plt
from deblending_pipeline.run_asterism import DataSetDetection
from deblending_pipeline.run_analysis import DataSetAnalysis
import scipy as sp
from scipy.stats import chi2,norm
from scipy import stats
from astropy.table import Table

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
_set_names_=['couple_skymaker_r1','couple_skymaker_r5','couple_CANDELS_r1','couple_CANDELS_r5','single_skymaker_r1','single_CANDELS_r1','couple_big_skymaker_r10','couple_big_CANDELS_r10']

def multi_table(table_list):
    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell
    '''
    return HTML(
        '<table><tr style="background-color:white;">' +
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>')

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = '#d65f5f' if val < 0 else  '#5fba7d'
    return 'background-color: %s' % color


def select_rows(df, method, mag_cut=-1, rec_sim_th=-1, contam_th=-1, h1=None, h2=None, th_name='denc_pd_ratio_th',th_val=None,
                th_val_min=None):
    # df=df[(df['mag_cut']==mag_cut) &(df['rec_sim_th']==rec_sim_th) &(df['contam_th']==contam_th)]
    s = locals()
    for v in ['mag_cut', 'rec_sim_th', 'contam_th']:
        if s[v] is not None:
            df = df[(df[v] == s[v])]

    if method != 'multi-th':
        if 'h_min' in df.columns:
            h_min_name = 'h_min'
            h_max_name = 'h_max'
        else:
            h_min_name = 'h_frac_min'
            h_max_name = 'h_frac_max'

        if th_val is not None:
            df = df[df[th_name] == th_val]

        if th_val_min is not None:
            df = df[df[th_name] >= th_val_min]

        if h1 != None:
            df = df[df[h_min_name] == h1]

        if h2 != None:
            df = df[df[h_max_name] == h2]

    df.reset_index(inplace=True)

    return df


def build_comp_table(files,
                     mag_cut=-1,
                     rec_sim_th=-1,
                     contam_th=-1,
                     text='',
                     set_names=_set_names_,
                     debl_ok_name='frac_ok_th_net',
                     h1=None,
                     h2=None,
                     th_name='denc_pd_ratio',
                     th_val=None,
                     th_val_min=None):

    col_names = ['sample', 'multi-th', 'Nthr', 'MinCnt', 'denclue', 'delta_d', 'K_d', 'h1_d', 'h2_d', 'sig_th_d','watershed', 'delta_w', 'K_w', 'h1_w', 'h2_w', 'wc', 'sig_th_w']

    comp_df = DataFrame(columns=col_names)
    comp_df['sample'] = set_names

    for f in files:
        sn = None
        for name in set_names:
            if name in f:
                sn = name

        if sn is not None:
            df = pandas.read_pickle(f)
            f = os.path.basename(f)

            if 'sextractor' in f:
                method = 'multi-th'
            elif 'denclue' in f:
                method = 'denclue'
            elif 'watershed' in f:
                method = 'watershed'
            else:
                raise RuntimeError('deblending method not uderstood')
            if method != 'multi-th':
                if 'h_min' in df.columns:
                    h_min_name = 'h_min'
                    h_max_name = 'h_max'
                else:
                    h_min_name = 'h_frac_min'
                    h_max_name = 'h_frac_max'

            df = select_rows(df,
                             method,
                             mag_cut=mag_cut,
                             rec_sim_th=rec_sim_th,
                             contam_th=contam_th,
                             h1=h1,
                             h2=h2,
                             th_name=th_name,
                             th_val=th_val,
                             th_val_min=th_val_min)

            comp_df[method][comp_df['sample'].str.match(sn)] = df[debl_ok_name].max()

            if df.size > 0:
                id_max = df[debl_ok_name].idxmax()
                if method != 'multi-th':
                    comp_df['K_%s' % method[0]][comp_df['sample'].str.match(sn)] = df['K_denclue'].iloc[id_max]
                    comp_df['h1_%s' % method[0]][comp_df['sample'].str.match(sn)] = df[h_min_name].iloc[id_max]
                    comp_df['h2_%s' % method[0]][comp_df['sample'].str.match(sn)] = df[h_max_name].iloc[id_max]
                    if 'valid_sig_th' in df.columns:
                        s = df['valid_sig_th'].iloc[id_max]
                    else:
                        s = df['sig_th'].iloc[id_max]

                    if 'denc_pd_ratio_th' in df.columns:
                        s = df['denc_pd_ratio_th'].iloc[id_max]

                    comp_df['sig_th_%s' % method[0]][comp_df['sample'].str.match(sn)] = s
                    if method == 'watershed' and 'watershed_compactness' in df.columns:
                        comp_df['wc'][comp_df['sample'].str.match(sn)] = df['watershed_compactness'].iloc[id_max]


                else:
                    comp_df['Nthr'][comp_df['sample'].str.match(sn)] = df['Nthr'].iloc[id_max]
                    comp_df['MinCnt'][comp_df['sample'].str.match(sn)] = df['MinCnt'].iloc[id_max]

    if comp_df.notnull().sum().sum() > len(set_names):
        # print('------------------------------------------------')

        text += ' , mag_cut=%s' % mag_cut
        text += ' , rec_sim_th=%s' % rec_sim_th
        text += ' , contam_th=%s' % contam_th
        comp_df['delta_d'] = (comp_df['denclue'] - comp_df['multi-th']) * 100
        comp_df['delta_w'] = (comp_df['watershed'] - comp_df['multi-th']) * 100
        comp_df['watershed'] = comp_df['watershed'] * 100
        comp_df['denclue'] = comp_df['denclue'] * 100
        comp_df['multi-th'] = comp_df['multi-th'] * 100
        comp_df = comp_df.apply(pandas.to_numeric, errors='ignore')
        comp_df = comp_df.round(4)
        comp_df = comp_df.style.applymap(color_negative_red, subset=['delta_d', 'delta_w']).apply(highlight_max, axis=1,
                                                                                                  subset=['multi-th',
                                                                                                          'denclue',
                                                                                                          'watershed']).set_caption(
            text)

        # display(comp_df)
        # print('------------------------------------------------')
    else:
        comp_df = None

    return comp_df



def compare_tables(asterism_run_list,
                   sextractor_run,
                   root_pandas_tables='results/pandas_tables',
                   mag_cut_l=[-1],
                   rec_sim_th_l=[-1],
                   contam_th_l=[-1],
                   set_names=_set_names_,
                   h1=None,
                   h2=None,
                   th_name='denc_pd_ratio',
                   th_val=None,
                   th_val_min=None):
    df_list=[]
    comb=product(asterism_run_list,rec_sim_th_l,contam_th_l,mag_cut_l)
    for c in comb:
        print(c)
        files=glob.glob('%s/%s*.pkl'%(root_pandas_tables,c[0]))
        files.extend(glob.glob('%s/%s*.pkl'%(root_pandas_tables,sextractor_run)))
        df_list.append(build_comp_table(files,
                                        mag_cut=c[3],
                                        rec_sim_th=c[1],
                                        contam_th=c[2],
                                        text=c[0],
                                        set_names=set_names,
                                        h1=h1,
                                        h2=h2,
                                        th_name=th_name,
                                        th_val=th_val,
                                        th_val_min=th_val_min))


    df_list=list(filter(None.__ne__, df_list))

    M=len(df_list)
    N_cols=1
    d=np.floor_divide(M,N_cols)
    pad=np.remainder(M,N_cols)
    print(M,N_cols,pad,d)
    for ID in range(d):
        display(multi_table(df_list[ID*N_cols:ID*N_cols+N_cols]))
        print(ID*N_cols,ID*N_cols+N_cols)
    if pad>0:
        display(multi_table(df_list[ID*N_cols+N_cols:ID*N_cols+N_cols+pad]))
        print(ID*N_cols+N_cols,ID*N_cols+N_cols+pad)

        return df_list

def efficiency_histogram_plot(sex_file,
                              ast_denc_file,
                              ast_dencw_file,
                              set_name,
                              ax,
                              debl_ok_name='frac_ok_th_net',
                              mag_cut=-1,
                              rec_sim_th=-1,
                              contam_th=-1,
                              th_name='denc_pd_ratio',
                              th_val=None,
                              th_val_min=None,
                              plot_watershed=False):
    try:
        df_ast_d = pandas.read_pickle(ast_denc_file)
    except:
        df_ast_d = None

    try:
        df_ast_w = pandas.read_pickle(ast_dencw_file)
    except:
        df_ast_w = None

    try:
        df_sex = pandas.read_pickle(sex_file)
    except:
        df_sex = None

    if df_ast_d is not None:
        ast_d = select_rows(df_ast_d, 'denclue', mag_cut=mag_cut, rec_sim_th=rec_sim_th, contam_th=contam_th,
                            th_name=th_name,th_val=th_val, th_val_min=th_val_min)
        # print(ast_d)
        sns.distplot([ast_d[debl_ok_name]], hist=True, rug=True, label='asterism d', ax=ax, norm_hist=True);

    if df_ast_w is not None and plot_watershed is True:
        ast_w=select_rows(df_ast_w,'watershed',mag_cut=mag_cut,rec_sim_th=rec_sim_th,contam_th=contam_th,th_name=th_name,th_val=th_val, th_val_min=th_val_min)
        sns.distplot([ast_w[debl_ok_name]], hist=True, rug=True,label='asterism w',ax=ax,norm_hist=True);
        #print(len(ast_w))

    if df_sex is not None:
        sex = select_rows(df_sex, 'multi-th', mag_cut=mag_cut, rec_sim_th=rec_sim_th, contam_th=contam_th)
        sns.distplot([sex[debl_ok_name]], hist=True, rug=True, label='sextractor', ax=ax, norm_hist=True);
        # print(len(sex))

    ax.set_xlabel('deblending efficiency')
    ax.set_ylabel('distribution')
    ax.set_title(set_name)
    ax.legend()
    return  ax

def efficiency_vs_th_par_plot(run_name_list,
                              set_names_list,
                              method,
                              par_name,
                              par_list,
                              root_pandas_tables='results/pandas_tables',
                              debl_ok_name='frac_ok_th_net',
                              mag_cut=-1,
                              h1=None,
                              h2=None,
                              c = 3,
                              rec_sim_th=-1,
                              contam_th=-1):
    # matplotlib inline
    # fig, (ax1) = plt.subplots(1,1,figsize=(10,10))
    N = len(par_list)
    x = np.linspace(0., 5, N)
    y = np.zeros(N)


    r = max(1, np.divmod(len(set_names_list), c)[0])
    if np.divmod(len(set_names_list), c)[1] > 0:
        r += 1
    print(len(set_names_list), c, r)
    fig, axs = plt.subplots(r, c, figsize=(10 * c, 5 * r))
    axs = axs.flatten()
    for ID_ax, set_name in enumerate(set_names_list):
        for run_name in run_name_list:
            f = '%s/%s_%s_%s.pkl' % (root_pandas_tables,run_name, set_name, method)
            for ID, p in enumerate(par_list):
                df = pandas.read_pickle(f)
                df = select_rows(df,
                                 method,
                                 contam_th=contam_th,
                                 mag_cut=mag_cut,
                                 rec_sim_th=rec_sim_th,
                                 h1=h1,
                                 h2=h2)

                d = df[(df[par_name] == p)]
                x[ID] = p
                y[ID] = d[debl_ok_name].max()
            axs[ID_ax].plot(x, y, label='%s %s' % (par_name, run_name))
        axs[ID_ax].legend()
        axs[ID_ax].grid()
        axs[ID_ax].set_title(set_name)
        axs[ID_ax].set_xlabel(par_name)
        axs[ID_ax].set_xlabel(debl_ok_name)

    plt.tight_layout()
    plt.show()

    return fig

def sex_plot(sextractor_run_name,
             set_name,
             root_pandas_tables='results/pandas_tables',
             mag_cut=25.3,
             contam_th=0.0,
             rec_det_th=-1,
             rec_sim_th=0.3,
             debl_ok_name='frac_debl_ok_th',
             plot=False,
             save_plot=False,
             wd='./'):
    df = pandas.read_pickle('%s/%s_%s_sextractor.pkl' % (root_pandas_tables,sextractor_run_name, set_name))
    df = df[(df['mag_cut'] == mag_cut) & (df['rec_sim_th'] == rec_sim_th) & (df['contam_th'] == contam_th)]

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    df['Nthr'] = np.float32(df['Nthr'].values)
    df['MinCnt'] = ['%4.4f' % float(f) for f in df['MinCnt']]

    pivot_debl = df.pivot('Nthr', 'MinCnt', debl_ok_name)
    pivot_debl.sort_values('Nthr', inplace=True, ascending=False)
    best = df.sort_values(debl_ok_name)[debl_ok_name].max()
    worst = df.sort_values(debl_ok_name)[debl_ok_name].min()

    pivot_under_debl = df.pivot('Nthr', 'MinCnt', 'frac_under_net')
    pivot_under_debl.sort_values('Nthr', inplace=True, ascending=False)
    max_under = df.sort_values('frac_under_net')['frac_under_net'].max()
    min_under = df.sort_values('frac_under_net')['frac_under_net'].min()

    pivot_overdebl_debl = df.pivot('Nthr', 'MinCnt', 'frac_overd_net')
    pivot_overdebl_debl.sort_values('Nthr', inplace=True, ascending=False)
    max_over = df.sort_values('frac_overd_net')['frac_overd_net'].max()
    min_over = df.sort_values('frac_overd_net')['frac_overd_net'].min()

    ax = axs[0]
    ax.set_title('best=%3.3f worst=%3.3f' % (best, worst))
    sns.heatmap(pivot_debl, cmap='jet', vmin=0, vmax=1, ax=ax)
    ax.yaxis.set_tick_params(rotation=0)

    ax = axs[1]
    ax.set_title('max_under=%3.3f min_under=%3.3f' % (max_under, min_under))
    sns.heatmap(pivot_under_debl, cmap='Blues', vmin=0, vmax=1, ax=ax)
    ax.yaxis.set_tick_params(rotation=0)

    ax = axs[2]
    ax.set_title('max_over=%3.3f min_over=%3.3f' % (max_over, min_over))
    sns.heatmap(pivot_overdebl_debl, cmap='Reds', vmin=0, vmax=1, ax=ax)
    ax.yaxis.set_tick_params(rotation=0)

    fig.tight_layout(rect=[0, 0, 1.0, 0.8])
    fig.suptitle(
        'SExtractor - run:%s - sample:%s - size =%d \n  contam_th=%s rec_det_th=%s rec_sim_th=%s mag_cut=%s' % (
        sextractor_run_name, set_name,
        df['n_net'].iloc[0],
        contam_th,
        rec_det_th,
        rec_sim_th, mag_cut),
        y=0.98, fontsize=15)

    fig_name = 'sex_%s_rec_sim_th_%s_contam_th_%s_mag_cut_%s' % (set_name, rec_sim_th, contam_th, mag_cut)
    print('figname', fig_name)
    if save_plot == True:
        file_name = os.path.join(wd, fig_name)
        fig.savefig('%s.pdf' % file_name)


def asterism_plot(asterism_run_name,
                  set_name,
                  root_pandas_tables='results/pandas_tables',
                  segmethod='denclue',
                  mag_cut=25.3,
                  contam_th=0,
                  rec_det_th=-1,
                  rec_sim_th=-1,
                  plot=False,
                  save_plot=False,
                  wd='./',
                  h_min_name='h_frac_min',
                  h_max_name='h_frac_max',
                  comp=None,
                  sig_th_list=[1.0, 1.2, 1.5],
                  denc_pd_ratio_th_list=[1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
                  K_denclue_list=[4],
                  debl_ok_name='frac_ok_th_net'):
    df = pandas.read_pickle('%s/%s_%s_%s.pkl' % (root_pandas_tables,asterism_run_name, set_name, segmethod))
    df = df[(df['mag_cut'] == mag_cut) & (df['rec_sim_th'] == rec_sim_th) & (df['contam_th'] == contam_th)]

    df = df.loc[df['denclue_segm_method'] == segmethod]
    if comp is not None:
        df = df.loc[df['watershed_compactness'] == comp]


    r = len(K_denclue_list) * len(sig_th_list) * len(denc_pd_ratio_th_list)
    c = 3
    fsize = (15, +5 * r)

    # fig = plt.figure(constrained_layout=True,figsize=fsize)
    # gs = fig.add_gridspec(r,c)
    # axs=np.empty(shape=(r,c),dtype=np.object)
    # for row in range(r):
    #    for col in range(3):
    #        axs[row,col]=fig.add_subplot(gs[row, col])

    fig, axs = plt.subplots(r, c, figsize=fsize, constrained_layout=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    if np.ndim(axs) == 1:
        axs = np.array([axs])

    incr_row = 0
    size = df['n_net'].iloc[0]

    for ID_sig_th, (sig_th, denc_pd_ratio_th, k) in enumerate(
            product(sig_th_list, denc_pd_ratio_th_list, K_denclue_list)):
        df_final = df.loc[
            (df['K_denclue'] == k) & (df['valid_sig_th'] == sig_th) & (df['denc_pd_ratio_th'] == denc_pd_ratio_th)]

        pivot_debl = df_final.pivot(h_min_name, h_max_name, debl_ok_name)
        df_final.sort_values(h_min_name, inplace=True, ascending=False)
        best = df_final.sort_values(debl_ok_name)[debl_ok_name].iloc[-1]
        worst = df_final.sort_values(debl_ok_name)[debl_ok_name].iloc[0]

        pivot_under_debl = df_final.pivot(h_min_name, h_max_name, 'frac_under_net')
        max_under = df_final.sort_values('frac_under_net')['frac_under_net'].iloc[-1]
        min_under = df_final.sort_values('frac_under_net')['frac_under_net'].iloc[0]

        pivot_overdebl_debl = df_final.pivot(h_min_name, h_max_name, 'frac_overd_net')
        max_over = df_final.sort_values('frac_overd_net')['frac_overd_net'].iloc[-1]
        min_over = df_final.sort_values('frac_overd_net')['frac_overd_net'].iloc[0]

        ax = axs[incr_row, 0]
        ax.set_title('K=%2.2f denc_pd_ratio_th=%2.2f  \n best=%2.2f worst=%2.2f' % (k, denc_pd_ratio_th, best, worst))
        sns.heatmap(pivot_debl, cmap='jet', vmin=0, vmax=1, ax=ax, square=True)

        ax = axs[incr_row, 1]
        ax.set_title('K=%2.2f denc_pd_ratio_th=%2.2f  \n max_under=%2.2f min_under=%2.2f' % (
        k, denc_pd_ratio_th, max_under, min_under))
        sns.heatmap(pivot_under_debl, cmap='Blues', vmin=0, vmax=1, ax=ax, square=True)

        ax = axs[incr_row, 2]
        ax.set_title('K=%2.2f denc_pd_ratio_th=%2.2f  \n max_over=%2.2f min_over=%2.2f' % (
        k, denc_pd_ratio_th, max_over, min_over))
        sns.heatmap(pivot_overdebl_debl, cmap='Reds', vmin=0, vmax=1, ax=ax, square=True)

        incr_row += 1

    fig.suptitle('ASTErIsM - run:%s  -sample:%s\n  method:%s  size =%d - contam_th=%s rec_sim_th=%s mag_cut=%s' % (
    asterism_run_name,
    set_name,
    segmethod,
    size,
    contam_th,
    rec_sim_th,
    mag_cut),
                 y=0.98,
                 fontsize=15)

    # plt.subplots_adjust(wspace=0,hspace=0)
    fig.tight_layout(rect=[0, 0, 1.0, 0.9])
    # fig.tight_layout()

    if plot == True:
        plt.show()

    fig_name = 'ast_%s_%s_segmeth_%s_rec_sim_th_%s_contam_th_%s_mag_cut_%s' % (
    asterism_run_name, set_name, segmethod, rec_sim_th, contam_th, mag_cut)
    if comp is not None:
        fig_name += '_comp_%2.2f' % comp
    print('figname', fig_name)
    if save_plot == True:
        file_name = os.path.join(wd, fig_name)
        fig.savefig('%s.pdf' % file_name)
        plt.close('all')


def get_debl_catalog(run_name,
                     set_name,
                     root_dir='lesta_analysis',
                     root_ast_detection='deblending_detection/asterism',
                     root_ast_analysis='deblending_analysis/asterism',
                     root_data_path='./',
                     method='denclue',
                     seg_method='denclue',
                     h_min=0.05,
                     h_max=0.25,
                     K_denclue=4,
                     watershed_compactness=0.,
                     validation=True,
                     valid_abs_size_th=1,
                     valid_sig_th=1.0,
                     valid_overlap_max=1.0,
                     downsampling=1,
                     valid_denc_pb_ratio_th=-1,
                     valid_denc_pd_ratio_th=-1,
                     valid_denc_prandom_ratio_th=-1,
                     mag_cut=-1,
                     rec_sim_th=-1,
                     rec_det_th=-1,
                     contam_th=0.):

    dsd = DataSetDetection.from_name_factory(set_name, root_data_path)

    ast_flag = dsd.get_run_flag(h_min,
                                h_max,
                                K_denclue,
                                watershed_compactness,
                                validation,
                                valid_abs_size_th,
                                valid_sig_th,
                                valid_overlap_max,
                                downsampling,
                                valid_denc_pb_ratio_th,
                                valid_denc_pd_ratio_th,
                                denc_prand_ratio=valid_denc_prandom_ratio_th)

    root_ast_detection = os.path.join(root_dir, run_name, root_ast_detection)
    # print('root_ast_detection',root_ast_)
    dsa = DataSetAnalysis.from_name_factory(set_name,
                                            root_data_path,
                                            root_ast_detection,
                                            debl_method=method,
                                            debl_segmethod='seg_%s' % seg_method,
                                            ast_flag=ast_flag,
                                            ast_name=dsd.name,
                                            sex_flag=None,
                                            mag_cut=mag_cut)

    debl_cat = dsa.deblended_catalog

    sim_file = dsd.pd_sim_table
    df = pandas.read_pickle(sim_file)
    if mag_cut is not None and mag_cut != -1:
        mag_filter = (df['mag'] < mag_cut) & (df['nearest_mag'] < mag_cut)
        selected = df[mag_filter]
        msk = [debl_cat['image_ID'][ID] in selected.index for ID, i in enumerate(debl_cat['image_ID'])]
        debl_cat = debl_cat[msk]
    else:
        mag_filter = None

    ast_flag = dsd.get_run_flag(h_min,
                                h_max,
                                K_denclue,
                                watershed_compactness,
                                validation,
                                valid_abs_size_th,
                                valid_sig_th,
                                valid_overlap_max,
                                downsampling,
                                valid_denc_pb_ratio_th,
                                valid_denc_pd_ratio_th,
                                valid_denc_prandom_ratio_th)

    # print('--->',ast_flag)
    # f_sky='test_asterism/deblending_analysis/asterism/couples_19_26_24.5_d10_r1/couple_skymaker_r1/denclue/seg_denclue/h_min_0.10_h_max_0.20_K_denc_04_comp_0.00_donws_1_valid_1_size_th_001_sig_th_0.00_overlap_th_1.00_pb_th_-100.00_pd_th_-100.00_rec_sim_th_0.00_rec_det_th_-1.00_contam_th_0.00_mag_cut_25.30_analysis_res.pkl'
    # f_candels='test_asterism/deblending_analysis/asterism/couples_19_26_24.5_d10_r1/couple_CANDELS_r1/denclue/seg_denclue/h_min_0.10_h_max_0.20_K_denc_04_comp_0.01_donws_1_valid_1_size_th_001_sig_th_1.00_overlap_th_1.00_pb_th_-100.00_pp_th_-100.00_rec_sim_th_0.00_rec_det_th_-1.00_contam_th_0.00_mag_cut_25.30_analysis_res.pkl'

    analysis_flag = 'rec_sim_th_%2.2f_rec_det_th_%2.2f_contam_th_%2.2f_mag_cut_%2.2f' % (
    rec_sim_th, rec_det_th, contam_th, mag_cut)
    analysis_path_ast = os.path.join(root_dir, run_name, root_ast_analysis, dsd.data_flag, set_name, method,
                                     'seg_%s' % seg_method)
    analsys_file_ast_res = os.path.join(analysis_path_ast, '%s_%s_analysis_res.pkl' % (ast_flag, analysis_flag))
    df_ast_res = pandas.read_pickle(analsys_file_ast_res)
    print('res_file', analsys_file_ast_res)
    msk_t = [debl_cat['src_ID'][ID] not in df_ast_res.iloc[image_ID]['contaminant_list'] for ID, image_ID in
             enumerate(debl_cat['image_ID'])]
    msk_c = [debl_cat['src_ID'][ID] in df_ast_res.iloc[image_ID]['contaminant_list'] for ID, image_ID in
             enumerate(debl_cat['image_ID'])]
    msk_u = [debl_cat['src_ID'][ID] not in df_ast_res.iloc[image_ID]['contaminant_list'] for ID, image_ID in
             enumerate(debl_cat['image_ID'])]

    df_debl_cat = Table(debl_cat).to_pandas()
    category = ['None'] * len(debl_cat)
    df_debl_cat['category'] = pandas.Series(category, index=df_debl_cat.index)
    df_debl_cat['category'][msk_c] = 'contaminant'
    df_debl_cat['category'][msk_t] = 'true'
    df_debl_cat['flux_ave'] = df_debl_cat['flux_tot'] / df_debl_cat['n_points'] / 0.003

    return Table(debl_cat),df_debl_cat, df_ast_res, msk_t, msk_c, msk_u, mag_filter


def scatter_plot_th_par(debl_cat, msk_t, msk_c, msk_u, log=False,par_skip_val=-1,par_skip_name='denc_pd_ratio',
                        par_names=['bkg_p', 'children_p', 'bkg_p_mean', 'signif_ave', 'denc_pd_ratio','denc_pb_ratio', 'flux_ave']):
    _cat = DataFrame.copy(debl_cat,deep=True)

    if 'denc_prandom_ratio' in debl_cat.columns:
        par_names.append('denc_prandom_ratio')
    if log == True:

        for n in par_names:
            _cat[n] = np.log10(_cat[n])

    g = sns.pairplot(_cat[_cat[par_skip_name]>par_skip_val], hue="category", hue_order=['true', 'contaminant'], vars=par_names, markers=["o", "+"],
                     plot_kws=dict(s=30, alpha=0.5), height=3)

    return g

def hist_plot_th_par(debl_cat, msk_t, msk_c, msk_u, log=False,skip_true=False,par_skip_val=-1,par_skip_name='denc_pd_ratio',par_names=['bkg_p', 'children_p', 'bkg_p_mean', 'denc_pd_ratio','denc_pb_ratio']):

    if 'denc_prandom_ratio' in debl_cat.columns:
        par_names.append('denc_prandom_ratio')

    r = max(1, np.divmod(len(par_names), 2)[0])
    if np.divmod(len(par_names), 2)[1] > 0:
        r += 1

    fig, axs = plt.subplots(r, 2, figsize=(15, 10))
    axs = axs.flatten()
    msk = debl_cat[par_skip_name] > par_skip_val
    for ID, n in enumerate(par_names):
        hist_plot(debl_cat[n], msk_t, msk_c, msk_u,msk, n, ax=axs[ID], log=log,skip_true=skip_true,par_skip_val=par_skip_val,par_skip_name=par_skip_name)


def hist_plot_par(debl_cat, msk_t, msk_c, msk_u, par_name, log=False,skip_true=False,par_skip_val=-1,par_skip_name='denc_pd_ratio',scale=1.0):
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs = [axs]
    ID = 0
    msk=debl_cat[par_skip_name]>par_skip_val
    hist_plot(debl_cat[par_name], msk_t , msk_c , msk_u , msk,par_name, log=log, ax=axs[ID],skip_true=skip_true,scale=scale)


def hist_plot(x, msk_t, msk_c, msk_u,msk, par_name, log=False, ax=None,skip_true=False,scale=1.0):
    #msk_ = x > -1
    x = x[msk]

    msk_c = np.array(msk_c)[msk]
    msk_t = np.array(msk_t)[msk]

    x_c = x[msk_c]
    x_t = x[msk_t]
    # t = stats.t.rvs(df=2, size=len(x))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # if par_name=='denc_pb_ratio':
    #    x_c=x_c[x_c<3.0]

    if log is True:
        x_c = np.log10(x_c)
        x_t = np.log10(x_t)
        # t=np.log10(t)

    ax[0].hist(x_c, density=True, cumulative=False, label='%s c' % par_name, )
    ax[1].hist(x_c, density=True, cumulative=True, label='%s c' % par_name, bins=50)
    if skip_true is False:
        ax[0].hist(x_t, density=True, cumulative=False, label='%s t' % par_name,alpha=0.5)
        ax[1].hist(x_t, density=True, cumulative=True, label='%s t' % par_name, alpha=0.5,bins=50)



    # t=t[t<10]
    # t=t[t>x_c.min()]

    if par_name == 'denc_pb_ratio' or par_name=='denc_pd_ratio':


        if log is True:
            trials = 100000

            k = norm.rvs(0, scale, trials)
            k = np.log10(k[k > 0])
            hist = np.histogram(k, bins=100)
            hist_dist = sp.stats.rv_histogram(hist)
            ratio = 1.0 / (1.0 - hist_dist.cdf(np.log10(x_c.min()),scale=scale))

            x1 = np.linspace(x_c.min()*0.5, x_c.max()*2, 100)
            y_gauss = hist_dist.pdf(x1) * ratio
        else:
            ratio = 1.0 / (1.0 - norm.cdf(x_c.min(),scale=scale))

            x1 = np.linspace(x_c.min()*0.5, x_c.max()*2, 100)
            y_gauss = norm.pdf(x1, 0, scale) * ratio

        trials = 1000
        k_stat = np.zeros(trials)
        p_val = np.zeros(trials)
        if log is True:
            x_c = 10 ** x_c
        for ID in range(trials):
            k = norm.rvs(0, scale, len(x_c))
            k = k[k > x_c.min()]
            #k = k[k < 1.5]
            k_stat[ID], p_val[ID] = stats.ks_2samp(x_c, k[k > x_c.min()])
        #ax[0].plot(x1, y_gauss, 'k-', lw=2, label='pdf Normal, ks=%f, p-val=%f' % (k_stat.mean(), p_val.mean()))

    if par_name == 'denc_prandom_ratio':

        #ratio = 1.0 / (1.0 - norm.cdf(1.0))


        if log is True:
            #trials = 100000

            #k = norm.rvs(0, 1.0, trials)
            #k = np.log10(k[k > 0])
            #hist = np.histogram(k, bins=100)
            #hist_dist = sp.stats.rv_histogram(hist)
            #ratio = 1.0 / (1.0 - hist_dist.cdf(0))

            x1 = np.linspace(-3, 3, 100)
            #y_gauss = hist_dist.pdf(x1)
        else:
            x1 = np.linspace(-3, 3, 100)

        y_gauss = norm.pdf(x1, 0, scale)

        trials = 1000
        k_stat = np.zeros(trials)
        p_val = np.zeros(trials)
        if log is True:
            x_c = 10 ** x_c
        for ID in range(trials):
            k = norm.rvs(0, scale, len(x_c))
            k_stat[ID], p_val[ID] = stats.ks_2samp(x_c, k[k >x_c.min()])
        #ax[0].plot(x1, y_gauss, 'k-', lw=2, label='pdf Normal, ks=%f, p-val=%f' % (k_stat.mean(), p_val.mean()))


    ax[0].legend()
    ax[0].grid()
    ax[1].legend()
    ax[1].grid()

def efficiency_hist_plot(set_names,
                         sextractor_run_name,
                         ast_run_name,
                         c=3,
                         root_pandas_tables='results/pandas_tables',
                         th_name='denc_pd_ratio',
                         th_val=None,
                         th_val_min=None,
                         plot_watershed=False):

    r = max(1, np.divmod(len(set_names), c)[0])
    if np.divmod(len(_set_names_), c)[1] > 0:
        r += 1

    fig, axs = plt.subplots(r, c, figsize=(5 * c, 3.5 * r))
    axs = axs.flatten()
    #fig.suptitle('%s %s' % (ast_run_name, sextractor_run_name))
    for ID, set_name in enumerate(_set_names_):
        ax = efficiency_histogram_plot('%s/%s_%s_sextractor.pkl' % (root_pandas_tables, sextractor_run_name, set_name),
                                       '%s/%s_%s_denclue.pkl' % (root_pandas_tables, ast_run_name, set_name),
                                       '%s/%s_%s_watershed.pkl' % (root_pandas_tables, ast_run_name, set_name),
                                       set_name,
                                       ax=axs.flatten()[ID],
                                       debl_ok_name='frac_ok_th_net',
                                       contam_th=0.0,
                                       mag_cut=25.3,
                                       rec_sim_th=-1,
                                       th_name=th_name,
                                       th_val=th_val,
                                       th_val_min=th_val_min,
                                       plot_watershed=plot_watershed)

        #ax.set_xlim(.2,1.0)

    plt.tight_layout()
    return fig

def eval_debl_ok(debl_cat, df_assoc, mag_filter=None, th=0.7, name='denc_pb_ratio',n_obj=2,par_skip_val=-1):
    tp_l = []
    fp_l = []
    tn_l = []
    fn_l = []




    for i, r in df_assoc.iterrows():
        _c = []
        _a = []
        skip_list=[]
        for c in r['contaminant_list']:
            m = debl_cat['image_ID'] == r['image_ID']
            m *= debl_cat['src_ID'] == c

            if debl_cat[name][m] > th:
                _c.append(c)
            if debl_cat[name][m] <=par_skip_val:
                skip_list.append(c)

        fp_size=len(_c)
        fp_l.append(fp_size)
        tn_l.append(len(r['contaminant_list'])-len(skip_list) -fp_size)
        skip_list = []
        for a in r['assoc_list']:
            m = debl_cat['image_ID'] == r['image_ID']
            m *= debl_cat['src_ID'] == a
            if debl_cat[name][m] > th:
                _a.append(a)
            if debl_cat[name][m] <= par_skip_val:
                skip_list.append(a)
        tp_size = len(_a)
        tp_l.append(tp_size)
        fn_l.append(len(r['assoc_list']) -len(skip_list) -tp_size)
    d = {'tp': tp_l, 'fp': fp_l,'tn':tn_l,'fn':fn_l}
    df_final = DataFrame(data=d)
    if mag_filter is not None:
        df_final = df_final[mag_filter]

    #df_final=df_final[debl_cat[name]>part_skip_val]
    debl_ok = np.sum(np.logical_and(df_final['tp'] == n_obj , df_final['fp'] == 0))
    #over_debl = np.sum(df_final['cont'] > 0)
    #under_debl = np.sum(df_final['assoc'] < n_obj)

    if mag_filter is not None:
        size= mag_filter.sum()
    else:
        size =len(df_final)

    debl_ok *= 1.0/size


    return debl_ok,np.sum(df_final['tp']),np.sum(df_final['fp']),np.sum(df_final['tn']),np.sum(df_final['fn']),size,df_final



def validation_analysis(debl_cat,par_name,df_ast_res,mag_filter,N=50,th_min=-1,th_max=2,n_obj=2):


    th = np.logspace(th_min, th_max, N)

    debl_ok = np.zeros(N)
    #associated = np.zeros(N)
    #contamiant = np.zeros(N)

    tp = np.zeros(N)
    fp = np.zeros(N)
    tn = np.zeros(N)
    fn = np.zeros(N)






    for ID in range(th.size):
        debl_ok[ID], tp[ID], fp[ID], tn[ID], fn[ID], df_final, size = eval_debl_ok(debl_cat,
                                                                                   df_ast_res,
                                                                                   mag_filter=mag_filter,
                                                                                   th=th[ID],
                                                                                   name=par_name,
                                                                                   n_obj=n_obj)



    recall = tpr = tp / (tp + fn)
    precision = tp / (tp + fp)
    fpr = fall_out = fp / (fp + tn)
    tnr = tn / (tn + fp)


    return th,debl_ok,tpr,precision,fpr,tnr