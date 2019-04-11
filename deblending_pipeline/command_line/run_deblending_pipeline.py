#!/usr/bin/env python

import glob
import argparse
import os
import sys
import warnings
from deblending_pipeline.run_asterism import DataSetDetection
from deblending_pipeline.run_asterism import from_cl
from deblending_pipeline.table import build_candidate_df
from deblending_pipeline.run_analysis import DataSetAnalysis
from deblending_pipeline.analysis import deblending_analysis
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
                             validation,
                             method,
                             denclue_segm_method,
                             downsampling,
                             valid_abs_size_th,
                             valid_sig_th,
                             valid_overlap_max,
                             max_image_id=None,
                             log_file=None):
    
    print('run deblending detection')

    dsd.run_detection(conf_file,
                  dsd.name,
                  root_ast,
                  h_min,
                  h_max,
                  K_denclue,
                  method,
                  validation,
                  downsampling=downsampling,
                  save_products=True,
                  denclue_segm_method=denclue_segm_method,
                  max_image_id=max_image_id,
                  valid_abs_size_th=valid_abs_size_th,
                  valid_sig_th=valid_sig_th,
                  overlap_max=valid_overlap_max,
                  log_file=log_file)



def run_dblending_candidate_table(dsd,dsa,max_image_id=None,sex=False):
    if sex==True:
        debl_map=dsa.debl_map_sex[:max_image_id]
    else:
        debl_map=dsa.debl_map_ast[:max_image_id]

    debl_candidate_df=build_candidate_df(dsa.cube[:max_image_id],dsa.true_map[:max_image_id],debl_map)   
    return debl_candidate_df




def run_dblending_analysis(dsa,debl_flag,debl_candidate_df,rec_sim_th,rec_det_th,contam_th,mag=None,max_image_id=None,sex=False):
    print('run deblending analysis')
    if mag is not None:
        debl_filter=dsa.debl_filter[:max_image_id]
    else:
        debl_filter=None
    
    if sex==True:
        debl_map=dsa.debl_map_sex[:max_image_id]
        code='sextractor'
    else:
        debl_map=dsa.debl_map_ast[:max_image_id]
        code = 'astersim'

    debl_analysis_table,df_ast,debl_stats=deblending_analysis(dsa.cube[:max_image_id],
                                  dsa.true_map[:max_image_id],
                                  debl_map,
                                  '%s %s'%(code,debl_flag),
                                  dsa.n_sim,
                                  debl_filter=debl_filter,
                                  rec_sim_th=rec_sim_th,
                                  rec_det_th=rec_det_th,
                                  contam_th=contam_th,
                                  verbose=False,
                                  candidate_df=debl_candidate_df)
    
    flag='rec_sim_th_%2.2f_rec_det_th_%2.2f_contam_th_%2.2f_mag_%2.2f'%(rec_sim_th,rec_det_th,contam_th,mag)
    return debl_analysis_table,flag,debl_stats





def run_asterism(set_name,
        dsd,
        run_detection=True,
        run_candidate=True,
        run_analysis=True,
        max_image_id=1,
        root_path='./',
        conf_file='conf/detection.conf',
        root_ast_detection='deblending_detection/asterism',
        root_ast_analysis='deblending_analysis/asterism',
        h_min=0.05,
        h_max=0.20,
        K_denclue=8,
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
        mag=25.3):

    sig_pars=locals()
    # root_sex_analysis=os.path.join(root_path,root_sex_analysis)
    root_ast_detection=os.path.join(root_path,root_ast_detection)
    root_ast_analysis=os.path.join(root_path,root_ast_analysis)

    # dsd=DataSetDetection.from_name_factory(set_name,root_path)
    print('run asterism pipeline',locals())
    # asterism
    # if only_sex is False:
    ast_flag = dsd.get_run_flag(h_min, h_max, K_denclue, validation, valid_abs_size_th, valid_sig_th,
                                valid_overlap_max,
                                downsampling)
    if run_detection is True:

        wd=dsd.get_wd(root_ast_detection,dsd.data_flag,dsd.name,method,denclue_segm_method)



        log_file=os.path.join(wd,'%s.log'%ast_flag)
        print('run asterism pipeline wd',wd)
        run_deblending_detection(dsd,
                                 conf_file,
                                 root_ast_detection,
                                 h_min,
                                 h_max,
                                 K_denclue,
                                 validation,
                                 method,
                                 denclue_segm_method,
                                 downsampling,
                                 valid_abs_size_th,
                                 valid_sig_th,
                                 valid_overlap_max,
                                 max_image_id=max_image_id,
                                 log_file=log_file)




    dsa = DataSetAnalysis.from_name_factory(set_name,
                                            root_path,
                                            debl_method=method,
                                            debl_segmethod='seg_%s' % denclue_segm_method,
                                            ast_flag=ast_flag,
                                            ast_name=dsd.name,
                                            sex_flag=None,
                                            mag=mag)

    if run_detection is True:
        out_asterims_prods = glob.glob(os.path.join(dsa.ast_path, '%s*.fits' % ast_flag))
        # print (out_asterims_prods)
        out_asterims_prods.append(log_file)
        gzip_files(out_asterims_prods)



    analysis_path_ast = os.path.join(root_ast_analysis, dsd.data_flag, set_name, method, 'seg_%s' % denclue_segm_method)
    # if only_sex is False:

    os.makedirs(analysis_path_ast,exist_ok=True)
    debl_candidate_df_file=os.path.join(analysis_path_ast,'%s_debl_candidate_table.pkl'%ast_flag)

    # detection pars
    wd = dsd.get_wd(root_ast_detection, dsd.data_flag, dsd.name,method, denclue_segm_method)
    par_file = os.path.join(wd, '%s_par.json' % ast_flag)

    if run_candidate is True:
        debl_candidate_df=run_dblending_candidate_table(dsd,
                                                        dsa,
                                                        max_image_id=max_image_id)

        debl_candidate_df.to_pickle(debl_candidate_df_file)
    else:
        debl_candidate_df=pd.read_pickle(debl_candidate_df_file)


    if run_analysis is True:
        debl_analysis_table, analysis_flag,debl_stats=run_dblending_analysis(dsa,ast_flag,debl_candidate_df,rec_sim_th,rec_det_th,contam_th,mag=mag,max_image_id=max_image_id)
        analsys_file_ast_table = os.path.join(analysis_path_ast, '%s_%s_analysis_res.pkl' % (ast_flag,analysis_flag))
        analsys_file_ast_stat = os.path.join(analysis_path_ast, '%s_%s_analysis_stats.pkl' % (ast_flag, analysis_flag))

        df_analysis_table=DataFrame(debl_analysis_table)
        pandas.to_pickle(df_analysis_table,analsys_file_ast_table)

        df_analysis_stats = DataFrame(debl_stats)
        with open(par_file, 'r') as fp:
            pars = json.load(fp)
        print('detection pars', pars)
        # df_pars=pandas.DataFrame(pars)
        # df_analysis_stats=pandas.concat([df_pars,df_stats], axis=1, sort=False)
        for kp in pars.keys():
           if kp in sig_pars.keys():
               df_analysis_stats[kp]=pars[kp]
        pandas.to_pickle(df_analysis_stats  , analsys_file_ast_stat)






def run_sextractor(set_name,
        dsd,
        run_candidate=True,
        run_analysis=True,
        max_image_id=1,
        root_path='./',
        root_sex_analysis='deblending_analysis/sextractor',
        rec_sim_th=0.1,
        rec_det_th=-1,
        contam_th=-1,
        #overlap_th=-1
        mag=25.3,
        sex_flag='DebNthr_64_DebMin_0.0001'):

    print('run sextractor pipeline', 'flag',sex_flag)
    Nthr = sex_flag.split('_')[1]
    Min = sex_flag.split('_')[3]
    root_sex_analysis=os.path.join(root_path,root_sex_analysis)
    # root_ast_detection=os.path.join(root_path,root_ast_detection)
    # root_ast_analysis=os.path.join(root_path,root_ast_analysis)

    #dsd=DataSetDetection.from_name_factory(set_name,root_path)


    #ast_flag=None

    dsa = DataSetAnalysis.from_name_factory(set_name,
                                            root_path,
                                            debl_method=None,
                                            debl_segmethod=None,
                                            ast_flag=None,
                                            ast_name=dsd.name,
                                            sex_flag=sex_flag,
                                            mag=mag)






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

        debl_candidate_df.to_pickle(debl_candidate_df_file)
    else:
        debl_candidate_df=pd.read_pickle(debl_candidate_df_file)

    if run_analysis is True:

        debl_analysis_table, analysis_flag,debl_stats=run_dblending_analysis(dsa,sex_flag,debl_candidate_df,rec_sim_th,rec_det_th,contam_th,mag=mag,max_image_id=max_image_id,sex=True)
        analsys_file_sex = os.path.join(analysis_path_sex, '%s_%s_analysis_res.pkl' % (sex_flag, analysis_flag))
        analsys_file_ast_sex = os.path.join(analysis_path_sex, '%s_%s_analysis_stats.pkl' % (sex_flag, analysis_flag))
        df = DataFrame(debl_analysis_table)
        pandas.to_pickle(df, analsys_file_sex)
        df = DataFrame(debl_stats)
        df['Nthr']=Nthr
        df['Min']=Min

        pandas.to_pickle(df, analsys_file_ast_sex)

def set_datasets(set_name,root_path='./', method=None, denclue_segm_method=None, ast_flag=None, sex_flag=None):

    dsd=DataSetDetection.from_name_factory(set_name,root_path)

    dsa=DataSetAnalysis.from_name_factory(set_name,
                                         root_path,
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
        root_path='./',
        conf_file='conf/detection.conf',
        root_sex_analysis='deblending_analysis/sextractor',
        root_ast_detection='deblending_detection/asterism',
        root_ast_analysis='deblending_analysis/asterism',
        h_min=0.05,
        h_max=0.20,
        K_denclue=8,
        validation=True,
        downsampling=True,
        valid_abs_size_th=8,
        valid_sig_th=1.5,
        valid_overlap_max=0.85,
        method='denclue',
        #method='extrema'
        denclue_segm_method='denclue',
        #denclue_segm_method='watershed'
        rec_sim_th=0.1,
        rec_det_th=-1,
        contam_th=-1,
        #overlap_th=-1
        mag=25.3,
        sex_flag=None):


    pars_dict=locals()



    dsd = DataSetDetection.from_name_factory(set_name, root_path)

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

        if sex_flag is None:
            dsa = DataSetAnalysis.from_name_factory(set_name,
                                                    root_path,
                                                    debl_method=None,
                                                    debl_segmethod=None,
                                                    ast_flag=None,
                                                    sex_flag=sex_flag,
                                                    ast_name=dsd.name,
                                                    mag=mag)
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
    parser.add_argument('-max_image_id', type=int, default=-1)
    parser.add_argument('-root_path', type=str, default='./')
    parser.add_argument('-conf_file', type=str, default='conf/detection.conf')
    parser.add_argument('-h_min', type=float, default=0.05)
    parser.add_argument('-h_max', type=float, default=0.20)
    parser.add_argument('-K_denclue', type=int, default=8)
    parser.add_argument('-validation', action='store_true')
    parser.add_argument('-downsampling', action='store_true')
    parser.add_argument('-valid_abs_size_th', type=int, default=8)
    parser.add_argument('-valid_sig_th', type=float, default=1.5)
    parser.add_argument('-valid_overlap_max', type=float, default=0.85)
    parser.add_argument('-method', type=str, default='denclue',help='denclue')
    parser.add_argument('-denclue_segm_method', type=str, default='denclue', help='denclue,watershed')
    parser.add_argument('-rec_sim_th', type=float, default=-1.0, help='')
    parser.add_argument('-rec_det_th', type=float, default=-1.0, help='')
    parser.add_argument('-contam_th', type=float, default=-1.0, help='')
    parser.add_argument('-mag', type=float, default=25.3, help='mag cut')
    parser.add_argument('-sex_flag', type=str, default=None)

    args = parser.parse_args()

    run(**vars(args))


if __name__ == "__main__":
    main()
