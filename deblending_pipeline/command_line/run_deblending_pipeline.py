#!/usr/bin/env python

import glob
import argparse
import os
import sys
import warnings
from deblending_pipeline.run_asterism import DataSetDetection
from deblending_pipeline.run_asterism import from_cl
from deblending_pipeline.candidate_table import build_candidate_df
from deblending_pipeline.run_candidate_analysis import DataSetAnalysis
from deblending_pipeline.candiate_analysis import deblending_analysis
import pandas as pd
import gzip
import shutil
import json
import pandas
from pandas import DataFrame
from inspect import signature

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from contextlib import contextmanager
@contextmanager
def stdout_redirected(new_stdout):
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout



def gzip_files(files_list):
    for in_file in files_list:
        with open(in_file, 'rb') as f_in:
            out_file=in_file+'.gz'
            with gzip.open(out_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                os.remove(in_file) 

def run_deblending_detection(dsd,
                             conf_file,
                             root_ast,
                             h_min,
                             h_max,
                             K_denclue,
                             watershed_compactness,
                             validation,
                             method,
                             denclue_segm_method,
                             downsampling,
                             valid_abs_size_th,
                             valid_sig_th,
                             valid_overlap_max,
                             denc_pb_ratio=0.0,
                             denc_pd_ratio=0.0,
                             denc_prandom_ratio=None,
                             max_image_id=None,
                             log_file=None):
    
    print('run deblending detection')

    dsd.run_detection(conf_file,
                  dsd.name,
                  root_ast,
                  h_min,
                  h_max,
                  K_denclue,
                  watershed_compactness,
                  method,
                  validation,
                  downsampling=downsampling,
                  save_products=True,
                  denclue_segm_method=denclue_segm_method,
                  max_image_id=max_image_id,
                  valid_abs_size_th=valid_abs_size_th,
                  valid_sig_th=valid_sig_th,
                  overlap_max=valid_overlap_max,
                  log_file=log_file,
                  denc_pb_ratio=denc_pb_ratio,
                  denc_pd_ratio=denc_pd_ratio,
                  denc_prandom_ratio=denc_prandom_ratio)



def run_dblending_candidate_table(dsd,dsa,max_image_id=None,sex=False):
    if sex==True:
        debl_map=dsa.debl_map_sex[:max_image_id]
        seg_map = dsa.seg_map_sex[:max_image_id]
    else:
        debl_map=dsa.debl_map_ast[:max_image_id]
        seg_map=dsa.seg_map_ast[:max_image_id]

    debl_candidate_df=build_candidate_df(dsa.cube[:max_image_id],dsa.true_map[:max_image_id],debl_map,seg_map)
    return debl_candidate_df




def run_dblending_analysis(dsa,debl_flag,debl_candidate_df,rec_sim_th,rec_det_th,contam_th,mag_cut=None,max_image_id=None,sex=False):
    print('run deblending analysis')
    if mag_cut is not None:
        debl_filter=dsa.debl_filter[:max_image_id]
    else:
        debl_filter=None
    
    if sex==True:
        debl_map=dsa.debl_map_sex[:max_image_id]
        code='sextractor'
    else:
        debl_map=dsa.debl_map_ast[:max_image_id]
        code = 'astersim'

    if mag_cut is None:
        mag_cut = -1

    #print('mag_cut',mag_cut)
    debl_analysis_table,df_ast,debl_stats=deblending_analysis(dsa.cube[:max_image_id],
                                  dsa.true_map[:max_image_id],
                                  debl_map,
                                  '%s %s'%(code,debl_flag),
                                  dsa.n_sim,
                                  debl_filter=debl_filter,
                                  rec_sim_th=rec_sim_th,
                                  rec_det_th=rec_det_th,
                                  contam_th=contam_th,
                                  mag_cut=mag_cut,
                                  verbose=False,
                                  candidate_df=debl_candidate_df)


    flag='rec_sim_th_%2.2f_rec_det_th_%2.2f_contam_th_%2.2f_mag_cut_%2.2f'%(rec_sim_th,rec_det_th,contam_th,mag_cut)
    return debl_analysis_table,flag,debl_stats


def run_asterism(set_name,
                 dsd,
                 run_detection=True,
                 run_candidate=True,
                 run_analysis=True,
                 max_image_id=1,
                 root_data_path='./',
                 root_wd='./',
                 conf_file='conf/detection.conf',
                 root_ast_detection='deblending_detection/asterism',
                 root_ast_analysis='deblending_analysis/asterism',
                 h_min=0.05,
                 h_max=0.20,
                 K_denclue=8,
                 valid_denc_pb_ratio_th=0.0,
                 valid_denc_pd_ratio_th=0.0,
                 valid_denc_prandom_ratio_th=0.0,
                 watershed_compactness=0.,
                 validation=True,
                 downsampling=True,
                 valid_abs_size_th=8,
                 valid_sig_th=1.5,
                 valid_overlap_max=0.85,
                 method='denclue',
                 # method='extrema'
                 denclue_segm_method='denclue',
                 # denclue_segm_method='watershed'
                 rec_sim_th=0.1,
                 rec_det_th=-1,
                 contam_th=-1,
                 # overlap_th=-1
                 skip_log_file=False,
                 mag_cut=None):
    sig_pars = locals()
    # root_sex_analysis=os.path.join(root_data_path,root_sex_analysis)
    root_ast_detection = os.path.join(root_wd, root_ast_detection)
    root_ast_analysis = os.path.join(root_wd, root_ast_analysis)

    # dsd=DataSetDetection.from_name_factory(set_name,root_data_path)
    # print('run asterism pipeline',locals())
    # asterism
    # if only_sex is False:
    ast_flag = dsd.get_run_flag(h_min, h_max,
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

    if run_detection is True:
        wd = dsd.get_wd(root_ast_detection, dsd.data_flag, dsd.name, method, denclue_segm_method)
        if skip_log_file is True:
            log_file = None
        else:
            log_file = os.path.join(wd, '%s.log' % ast_flag)

        print('run asterism pipeline wd', wd)
        run_deblending_detection(dsd,
                                 conf_file,
                                 root_ast_detection,
                                 h_min,
                                 h_max,
                                 K_denclue,
                                 watershed_compactness,
                                 validation,
                                 method,
                                 denclue_segm_method,
                                 downsampling,
                                 valid_abs_size_th,
                                 valid_sig_th,
                                 valid_overlap_max,
                                 denc_pb_ratio=valid_denc_pb_ratio_th,
                                 denc_pd_ratio=valid_denc_pd_ratio_th,
                                 denc_prandom_ratio=valid_denc_prandom_ratio_th,
                                 max_image_id=max_image_id,
                                 log_file=log_file)




    dsa = DataSetAnalysis.from_name_factory(set_name,
                                            root_data_path,
                                            root_ast_detection,
                                            debl_method=method,
                                            debl_segmethod='seg_%s' % denclue_segm_method,
                                            ast_flag=ast_flag,
                                            ast_name=dsd.name,
                                            sex_flag=None,
                                            mag_cut=mag_cut)

    if run_detection is True:
        out_asterims_prods = glob.glob(os.path.join(dsa.ast_path, '%s*.fits' % ast_flag))

        if log_file is not None:
            out_asterims_prods.append(log_file)

        gzip_files(out_asterims_prods)



    analysis_path_ast = os.path.join(root_ast_analysis, dsd.data_flag, set_name, method, 'seg_%s' % denclue_segm_method)
    # if only_sex is False:

    os.makedirs(analysis_path_ast,exist_ok=True)
    debl_candidate_df_file=os.path.join(analysis_path_ast,'%s_debl_candidate_table.pkl'%ast_flag)

    # detection pars
    wd = dsd.get_wd(root_ast_detection, dsd.data_flag, dsd.name,method, denclue_segm_method)
    par_file = os.path.join(wd, '%s_par.json' % ast_flag)

    keep_list=['conf_file','name','max_image_id', 'method','denclue_segm_method', 'h_frac_min', 'h_frac_max', 'valid_abs_size_th','denc_pd_ratio_th','denc_pb_ratio_th',
     'valid_sig_th', 'overlap_max', 'K_denclue', 'watershed_compactness','downsampling',
     'validate_children', 'morph_corr', 'log_file']

    with open(par_file, 'r') as fp:
        pars = json.load(fp)
    print('par_file',par_file)

    for k in list(pars.keys()):

        if k not in keep_list:
            print('->', k)
            pars.pop(k)

    print(pars)


    if run_candidate is True:
        debl_candidate_df=run_dblending_candidate_table(dsd,
                                                        dsa,
                                                        max_image_id=max_image_id)


        for kp in pars.keys():
            debl_candidate_df[kp]=pars[kp]
        debl_candidate_df.to_pickle(debl_candidate_df_file)

    if run_analysis is True:
        if run_candidate is False:
            debl_candidate_df = pd.read_pickle(debl_candidate_df_file)


        debl_analysis_table, analysis_flag,debl_stats=run_dblending_analysis(dsa,ast_flag,debl_candidate_df,rec_sim_th,rec_det_th,contam_th,mag_cut=mag_cut,max_image_id=max_image_id)

        analsys_file_ast_res = os.path.join(analysis_path_ast, '%s_%s_analysis_res.pkl' % (ast_flag,analysis_flag))
        analsys_file_ast_stat = os.path.join(analysis_path_ast, '%s_%s_analysis_stats.pkl' % (ast_flag, analysis_flag))
        analsys_file_ast_stat_txt = os.path.join(analysis_path_ast, '%s_%s_analysis_stats.txt' % (ast_flag, analysis_flag))

        df_analysis_table = DataFrame(debl_analysis_table)
        df_analysis_stats = DataFrame(debl_stats)

        print('analsys_file_ast_stat', analsys_file_ast_stat)
        #print('signature  pars',  sig_pars.keys())
        # df_pars=pandas.DataFrame(pars)
        # df_analysis_stats=pandas.concat([df_pars,df_stats], axis=1, sort=False)
        #print(sig_pars.keys())
        #print('---')
        for kp in pars.keys():
            #print(kp)
            #if kp in sig_pars.keys():
            #print('->kp',kp)
            df_analysis_stats[kp]=pars[kp]
            df_analysis_table[kp]=pars[kp]
        #print('mag_cut',mag_cut)
        #print('out file',analsys_file_ast_stat)
        #print(df_analysis_stats.keys())
        pandas.to_pickle(df_analysis_stats, analsys_file_ast_stat)
        pandas.to_pickle(df_analysis_table, analsys_file_ast_res)
        with open(analsys_file_ast_stat_txt, 'w') as f:
            df_analysis_stats.to_string(f,index=False)




def run_sextractor(set_name,
        dsd,
        run_candidate=True,
        run_analysis=True,
        max_image_id=1,
        root_data_path='./',
        root_wd='./',
        root_sex_analysis='sextractor_detection/sextractor',
        rec_sim_th=0.1,
        rec_det_th=-1,
        contam_th=-1,
        sex_path_debl='./sextractor_detection/segmap_debl_detthr_1.2_minarea_10',
        mag_cut=None,
        sex_flag='DebNthr_64_DebMin_0.0001'):

    print('run sextractor pipeline', 'flag',sex_flag)
    Nthr = sex_flag.split('_')[1]
    Min = sex_flag.split('_')[3]
    root_sex_analysis=os.path.join(root_wd,root_sex_analysis)
    # root_ast_detection=os.path.join(root_data_path,root_ast_detection)
    # root_ast_analysis=os.path.join(root_data_path,root_ast_analysis)

    #dsd=DataSetDetection.from_name_factory(set_name,root_data_path)


    #ast_flag=None

    dsa = DataSetAnalysis.from_name_factory(set_name,
                                            root_data_path,
                                            None,
                                            debl_method=None,
                                            debl_segmethod=None,
                                            sex_path_debl=sex_path_debl,
                                            ast_flag=None,
                                            ast_name=dsd.name,
                                            sex_flag=sex_flag,
                                            mag_cut=mag_cut)






    # sextractor
    analysis_path_sex = os.path.join(root_sex_analysis, dsd.data_flag, set_name)
    #if only_asterism is False:

    os.makedirs(analysis_path_sex,exist_ok=True)
    debl_candidate_df_file=os.path.join(analysis_path_sex,'%s_debl_candidate_table.pkl'%sex_flag)
    if run_candidate is True:
        debl_candidate_df=run_dblending_candidate_table(dsd,
                                                        dsa,
                                                        max_image_id=max_image_id,
                                                        sex=True)

        debl_candidate_df['Nthr'] = Nthr
        debl_candidate_df['MinCnt'] = Min
        debl_candidate_df.to_pickle(debl_candidate_df_file)


    if run_analysis is True:
        if run_candidate is False:
            debl_candidate_df = pd.read_pickle(debl_candidate_df_file)

        debl_analysis_table, analysis_flag,debl_stats=run_dblending_analysis(dsa,sex_flag,debl_candidate_df,rec_sim_th,rec_det_th,contam_th,mag_cut=mag_cut,max_image_id=max_image_id,sex=True)
        analsys_file_res_sex = os.path.join(analysis_path_sex, '%s_%s_analysis_res.pkl' % (sex_flag, analysis_flag))
        analsys_file_stat_sex = os.path.join(analysis_path_sex, '%s_%s_analysis_stats.pkl' % (sex_flag, analysis_flag))
        analsys_file_stat_sex_txt = os.path.join(analysis_path_sex,'%s_%s_analysis_stats.txt' % (sex_flag, analysis_flag))





        df = DataFrame(debl_stats)
        df['Nthr']=Nthr
        df['MinCnt']=Min

        pandas.to_pickle(df, analsys_file_stat_sex)
        with open(analsys_file_stat_sex_txt, 'w') as f:
            df.to_string(f)


        df = DataFrame(debl_analysis_table)
        df['Nthr'] = Nthr
        df['MinCnt'] = Min
        pandas.to_pickle(df, analsys_file_res_sex)




def set_datasets(set_name,root_data_path='./', method=None, denclue_segm_method=None, ast_flag=None, sex_flag=None):

    dsd=DataSetDetection.from_name_factory(set_name,root_data_path)

    dsa=DataSetAnalysis.from_name_factory(set_name,
                                         root_data_path,
                                         None,
                                         debl_method=method,
                                         debl_segmethod='seg_%s'%denclue_segm_method,
                                         ast_flag=ast_flag,
                                         ast_name=dsd.name,
                                         sex_flag=sex_flag)

    print('--------')
    print()
   
    return dsd,dsa


def test_set_datasets():
    for name in ['couple_skymaker_r1','couple_skymaker_r5','couple_CANDELS_r1','couple_CANDELS_r5','couple_big_skymaker_r10','couple_big_CANDELS_r10']:
        set_datasets(name,method='denclue',denclue_segm_method='denclue',ast_flag=None,sex_flag=None)


#def test_run():
#    run('couple_skymaker_r1',run_detection=False,run_candidate=False,max_image_id=100)


def run(set_name,
        run_detection=True,
        run_candidate=True,
        run_analysis=True,
        only_sex=False,
        only_asterism=False,
        max_image_id=1,
        root_data_path='./',
        root_wd='./',
        conf_file='conf/detection.conf',
        root_sex_analysis='deblending_analysis/sextractor',
        root_ast_detection='deblending_detection/asterism',
        root_ast_analysis='deblending_analysis/asterism',
        h_min=0.05,
        h_max=0.20,
        K_denclue=8,
        watershed_compactness=0.,
        validation=True,
        downsampling=True,
        valid_abs_size_th=8,
        valid_sig_th=1.5,
        valid_denc_pb_ratio_th=0.0,
        valid_denc_pd_ratio_th=0.0,
        valid_denc_prandom_ratio_th=0.0,
        valid_overlap_max=0.85,
        method='denclue',
        #method='extrema'
        denclue_segm_method='denclue',
        #denclue_segm_method='watershed'
        rec_sim_th=0.1,
        rec_det_th=-1,
        contam_th=-1,
        sex_path_debl='./sextractor_detection/segmap_debl_detthr_1.2_minarea_10',
        skip_log_file=False,
        mag_cut=None,
        sex_flag=None):


    pars_dict=locals()



    dsd = DataSetDetection.from_name_factory(set_name, root_data_path)

    asterism_parm_dict={}
    if only_sex is False:

        astersim_sig = signature(run_asterism)
        for param in astersim_sig.parameters.values():
            if param.name in pars_dict.keys():
                asterism_parm_dict[param.name]=pars_dict[param.name]

        asterism_parm_dict['dsd'] = dsd
        run_asterism(**asterism_parm_dict)

    sex_parm_dict={}
    if only_asterism is False:
        sex_sig = signature(run_sextractor)
        for param in sex_sig.parameters.values():
            if param.name in pars_dict.keys():
                sex_parm_dict[param.name]=pars_dict[param.name]

        dsa = DataSetAnalysis.from_name_factory(set_name,
                                                root_data_path,
                                                None,
                                                debl_method=None,
                                                debl_segmethod=None,
                                                ast_flag=None,
                                                sex_path_debl=sex_path_debl,
                                                sex_flag=sex_flag,
                                                ast_name=dsd.name,
                                                mag_cut=mag_cut)

        if sex_flag is None:

            _p = os.path.join(dsa.sex_deblm_map_path,'*segmentation_map_debl*')
            print('_p',_p)
            _fl = glob.glob(_p)


            for f in _fl:
                sex_flag=os.path.basename(f)
                print('sex file',sex_flag)
                sex_flag = sex_flag.split('DebNthr_')[1]
                print('sex_flag',sex_flag)
                sex_flag = sex_flag.split('_segmentation')[0]
                print('sex_flag',sex_flag)
                sex_flag='DebNthr_' + sex_flag
                sex_parm_dict['sex_flag']=sex_flag


                sex_parm_dict['dsd'] = dsd
                run_sextractor(**sex_parm_dict)

        else:
            sex_parm_dict['dsd'] = dsd
            run_sextractor(**sex_parm_dict)

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('set_name', type=str, default=None,help=str(['couple_skymaker_r1','couple_skymaker_r5','couple_CANDELS_r1','couple_CANDELS_r5','couple_big_skymaker_r10','couple_big_CANDELS_r10']))
    parser.add_argument('-only_sex', action='store_true')
    parser.add_argument('-only_asterism', action='store_true')
    parser.add_argument('-run_detection', action='store_true')
    parser.add_argument('-run_candidate', action='store_true')
    parser.add_argument('-run_analysis', action='store_true')
    parser.add_argument('-max_image_id', type=int, default=None)
    parser.add_argument('-root_wd', type=str, default='./')
    parser.add_argument('-root_data_path', type=str, default='./')
    parser.add_argument('-sex_path_debl', type=str, default='./sextractor_detection/segmap_debl_detthr_1.2_minarea_10')
    parser.add_argument('-conf_file', type=str, default='conf/detection.conf')
    parser.add_argument('-h_min', type=float, default=0.05)
    parser.add_argument('-h_max', type=float, default=0.20)
    parser.add_argument('-K_denclue', type=int, default=8)
    parser.add_argument('-watershed_compactness', type=float, default=0.0)
    parser.add_argument('-validation', action='store_true')
    parser.add_argument('-downsampling', action='store_true')
    parser.add_argument('-valid_abs_size_th', type=int, default=1)
    parser.add_argument('-valid_sig_th', type=float, default=1.5)
    parser.add_argument('-valid_overlap_max', type=float, default=1.00)
    parser.add_argument('-valid_denc_pb_ratio_th', type=float, default=-1.0)
    parser.add_argument('-valid_denc_pd_ratio_th', type=float, default=-1.0)
    parser.add_argument('-valid_denc_prandom_ratio_th', type=float, default=-1.0)
    parser.add_argument('-method', type=str, default='denclue',help='denclue')
    parser.add_argument('-denclue_segm_method', type=str, default='denclue', help='denclue,watershed')
    parser.add_argument('-rec_sim_th', type=float, default=-1.0, help='')
    parser.add_argument('-rec_det_th', type=float, default=-1.0, help='')
    parser.add_argument('-contam_th', type=float, default=0.0, help='')
    parser.add_argument('-mag_cut', type=float, default=None, help='mag cut')
    parser.add_argument('-sex_flag', type=str, default=None)
    parser.add_argument('-skip_log_file', action='store_true')
    args = parser.parse_args()

    run(**vars(args))


if __name__ == "__main__":
    main()
