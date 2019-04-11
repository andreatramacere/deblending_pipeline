#!/usr/bin/env python

import glob
import argparse
import numpy as np
import os
from asterism.analysis_pipelines.source_detection import SrcDetectionPipeline
import sys
import warnings
import json

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class DataSetDetection(object):

    def __init__(self,root_rel_path,data_root_path,data_flag,sample_flag,sample_flag_1,segmap_root_path,name=None):
        print("Run asterism conf:",name)
        self.name=name
        self.data_flag = data_flag
        self.sample_flag = sample_flag
        self.sample_flag_1 = sample_flag_1

        self.data_path = os.path.join(root_rel_path,data_root_path,self.data_flag,'data')

        self.pd_sim_table = os.path.join('%s' % self.data_path, 'sky_to_CANDELS.pkl')


        _path = os.path.join(self.data_path, '*%s*_cube*%s*' % (self.sample_flag, sample_flag_1))

        _f = glob.glob(_path)
        
        _cube = [n for n in _f if 'true' not in n]
        _true = [n for n in _f if 'true' in n]
        
        self.cube_file = _cube[0]
        self.rms_file = os.path.join(self.data_path,'rms_map_0.00289.fits')
        print('-> cube file', self.cube_file)
        print('-> rms file', self.rms_file)
        _path = os.path.join(root_rel_path,segmap_root_path,'%s*%s*_segmap*' % (self.data_flag,self.sample_flag))

        try:
            self.segmap_file = glob.glob(_path)[0]
        except:
            raise  RuntimeError('segmap path problem', _path)

        print('-> segmap file',  self.segmap_file)
        #print(os.path.exists(self.cube_file), self.cube_file)
        #print(os.path.exists(self.rms_file), self.rms_file)
        #print(os.path.exists(self.segmap_file), self.segmap_file)

    @classmethod
    def from_name_factory(cls, name, root_rel_path,):
        _dict = {}
        _dict['couple_skymaker_r1'] = cls.build_couple_skymaker_r1
        _dict['couple_skymaker_r5'] = cls.build_couple_skymaker_r5
        _dict['couple_CANDELS_r1'] = cls.build_couple_CANDELS_r1
        _dict['couple_CANDELS_r5'] = cls.build_couple_CANDELS_r5
        _dict['single_skymaker_r1'] = cls.build_single_skymaker_r1
        _dict['single_CANDELS_r1'] = cls.build_single_CANDELS_r1
        _dict['couple_big_skymaker_r10'] = cls.build_couple_big_skymaker_r10
        _dict['couple_big_CANDELS_r10'] = cls.build_couple_big_CANDELS_r10

        return _dict[name](root_rel_path,)


    @classmethod
    def build_couple_skymaker_r1(cls, root_rel_path,name='couple_skymaker_r1'):
        return cls(root_rel_path,'datasets', 'couples_19_26_24.5_d10_r1', 'cat_tot_vis', 'CANDELS',
                   'deblending_detection/sextractor/segmap_detthr_1.2_minarea_10',name=name)

    @classmethod
    def build_couple_CANDELS_r1(cls, root_rel_path,name='couple_CANDELS_r1'):
        return cls(root_rel_path, 'datasets', 'couples_19_26_24.5_d10_r1', 'real', 'n_obj_2',
                   'deblending_detection/sextractor/segmap_detthr_1.2_minarea_10',name=name)

    @classmethod
    def build_single_skymaker_r1(cls, root_rel_path,name='single_skymaker_r1'):
        return cls(root_rel_path, 'datasets', 'single_19_26_24.5_d10_r1', 'cat_tot_vis', 'CANDELS',
                   'deblending_detection/sextractor/segmap_detthr_1.2_minarea_10',name=name)

    @classmethod
    def build_single_CANDELS_r1(cls, root_rel_path, name='single_CANDELS_r1'):
        return cls(root_rel_path, 'datasets', 'single_19_26_24.5_d10_r1', 'real', 'n_obj_1',
                   'deblending_detection/sextractor/segmap_detthr_1.2_minarea_10', name=name)


    @classmethod
    def build_couple_skymaker_r5(cls, root_rel_path,name='couple_skymaker_r5'):
        return cls(root_rel_path, 'datasets', 'couples_19_26_24.5_d10_r5', 'cat_tot_vis', 'CANDELS',
                   'deblending_detection/sextractor/segmap_detthr_1.2_minarea_10',name=name)

    @classmethod
    def build_couple_CANDELS_r5(cls, root_rel_path,name='couple_CANDELS_r5'):
        return cls(root_rel_path, 'datasets', 'couples_19_26_24.5_d10_r5', 'real', 'n_obj_2',
                   'deblending_detection/sextractor/segmap_detthr_1.2_minarea_10',name=name)

    @classmethod
    def build_couple_big_skymaker_r10(cls, root_rel_path,name='couple_big_skymaker_r10'):
        return cls(root_rel_path, 'datasets', 'big_19_23_24.5_d50_r10', 'cat_tot_vis', 'CANDELS',
                   'deblending_detection/sextractor/segmap_detthr_1.2_minarea_10',name=name)

    @classmethod
    def build_couple_big_CANDELS_r10(cls, root_rel_path,name='couple_big_CANDELS_r10'):
        return cls(root_rel_path, 'datasets', 'big_19_23_24.5_d50_r10', 'real', 'n_obj_2',
                   'deblending_detection/sextractor/segmap_detthr_1.2_minarea_10',name=name)


    @staticmethod
    def get_run_flag(h_frac_min,
                     h_frac_max,
                     K_denclue,
                     validate_children,
                     valid_abs_size_th,
                     valid_sig_th,
                     valid_overlap_max,
                     downsampling):

        flag = 'h_min_%2.2f_h_max_%2.2f_K_denc_%2.2d_donws_%d_valid_%d' % (h_frac_min, h_frac_max,K_denclue,downsampling,validate_children)
        if validate_children is True:
            flag = flag+'_size_th_%3.3d_sig_th_%2.2f_overlap_th_%2.2f'%(valid_abs_size_th, valid_sig_th,
                                                                        valid_overlap_max)
        return flag

    @staticmethod
    def get_wd(wd,
               data_flag,
               name,
               method,
               denclue_segm_method):

        return os.path.join(wd,data_flag,name,method, 'seg_%s' % denclue_segm_method)

    def run_detection(self,
                      conf_file,
                      name,
                      wd,
                      h_frac_min,
                      h_frac_max,
                      K_denclue,
                      method,
                      validate_children,
                      denclue_segm_method='',
                      downsampling=True,
                      max_image_id=None,
                      save_products=True,
                      valid_abs_size_th=8,
                      valid_sig_th=1.5,
                      overlap_max=0.85,
                      log_file=None):

        _run_detection(cube=self.cube_file,
                       name=name,
                       K_denclue=K_denclue,
                       conf_file=conf_file,
                       rms_file=self.rms_file,
                       segm_file=self.segmap_file,
                       data_flag=self.data_flag,
                       wd=wd,
                       save_products=save_products,
                       method=method,
                       denclue_segm_method=denclue_segm_method,
                       h_frac_max=h_frac_max,
                       h_frac_min=h_frac_min,
                       morph_corr=False,
                       downsampling=downsampling,
                       validate_children=validate_children,
                       max_image_id=max_image_id,
                       valid_abs_size_th=valid_abs_size_th,
                       valid_sig_th=valid_sig_th,
                       overlap_max=overlap_max,
                       log_file=log_file)




def run_job():
    pass


def _run_detection(cube,
                   conf_file,
                   name,
                   rms_file=None,
                   segm_file=None,
                   data_flag=None,
                   image_id=None,
                   wd=None,
                   max_image_id=None,
                   plot=False,
                   save_products=True,
                   method='',
                   denclue_segm_method='',
                   h_frac_min=0.05,
                   h_frac_max=0.20,
                   valid_abs_size_th=8,
                   valid_sig_th=1.5,
                   overlap_max=0.85,
                   K_denclue=4,
                   downsampling=True,
                   validate_children=True,
                   morph_corr=False,
                   log_file=None):

    pars_dict=locals()

    print('asterism start')
    pipeline = SrcDetectionPipeline(parser=None, argv=None, conf_file=conf_file)

    # pipeline.dump_configuration_file('pipeline.conf')
    # source the configuration from the conf file
    # all the parameters are set from conf file

    pipeline.set_pars_from_conf_file()

    base_name = os.path.basename(cube)
    # pipeline.dump_configuration_file('pipeline.conf')
    # source the configuration from the conf file
    # all the parameters are set from conf file

    pipeline.set_pars_from_conf_file()

    pipeline.IO_conf_task.set_par('infile', value=cube)
    pipeline.IO_conf_task.set_par('input_rms_map_file', value=rms_file)
    pipeline.IO_conf_task.set_par('input_seg_map_file', value=segm_file)
    pipeline.IO_conf_task.set_par('save_products', value=save_products)
    pipeline.IO_conf_task.set_par('out_name', value='')
    pipeline.IO_conf_task.set_par('log_file', value=log_file)

    # To display segmentation
    pipeline.do_src_detection.image_segmentation.set_par('plot',value=plot)
    
    if image_id is not None:
        pipeline.IO_conf_task.set_par('image_id', value=image_id)
    
    if max_image_id is not None:
        pipeline.IO_conf_task.set_par('max_image_id', value=max_image_id)

    pipeline.do_src_detection.source_catalog.set_par('out_catalog', value=True)
    pipeline.do_src_detection.source_catalog.set_par('out_seg_map', value=True)
    pipeline.do_src_detection.source_catalog.set_par('get_only_central_source', value=False)

    #method = 'denclue'
    #denclue_segm_method = 'watershed'
    if segm_file is not None:
        im_seg_method = 'from_seg_map'
    else:
        im_seg_method = 'connected'

    #wd_flag = os.path.join(method, 'seg_%s' % denclue_segm_method)
    if wd is not None:
        wd = DataSetDetection.get_wd(wd,data_flag,name,method,denclue_segm_method)

    pipeline.IO_conf_task.set_par('working_dir',value=wd)

    pipeline.do_src_detection.deblending_validation.set_par('validate_children', value=validate_children)
    pipeline.do_src_detection.deblending_validation.set_par('abs_size_th', value=valid_abs_size_th)
    pipeline.do_src_detection.deblending_validation.set_par('sig_th', value=valid_sig_th)
    pipeline.do_src_detection.deblending_validation.set_par('overlap_max', value=overlap_max)

    pipeline.do_src_detection.set_deblending_method.set_par('method', value=method)

    if denclue_segm_method != '' and denclue_segm_method is not None and method == 'denclue':
        pipeline.do_src_detection.denclue_deblending.set_par('segmentation_method', value=denclue_segm_method)

    pipeline.do_src_detection.denclue_deblending.set_par('morphological_correction', value=morph_corr)
    pipeline.do_src_detection.denclue_deblending.set_par('attr_dbs_K',value=K_denclue)
    pipeline.do_src_detection.denclue_deblending.set_par('do_downsampling', value=downsampling)
    pipeline.do_src_detection.image_segmentation.set_par('method', value=im_seg_method)

    # To display segmenataion
    pipeline.do_src_detection.denclue_deblending.set_par('plot',value=plot)
    pipeline.do_src_detection.extrema_deblending.set_par('plot',value=plot)
    pipeline.do_src_detection.deblending_validation.set_par('plot',value=plot)
    
    flag = DataSetDetection.get_run_flag(h_frac_min,
                                         h_frac_max,
                                         K_denclue,
                                         validate_children,
                                         valid_abs_size_th,
                                         valid_sig_th,
                                         overlap_max,
                                         downsampling)
    #print("flag", flag)
    pipeline.IO_conf_task.set_par('flag', value=flag)

    pipeline.do_src_detection.cluster_scale_finder.set_par('h_min_frac', value=np.float(h_frac_min))
    pipeline.do_src_detection.cluster_scale_finder.set_par('h_max_frac', value=np.float(h_frac_max))
    par_file = os.path.join(wd, '%s_par.json' % flag)
    # print('par_file', par_file)

    with open(par_file, 'w') as fp:
        json.dump(pars_dict, fp)
    products_collection = pipeline.run()

    pipeline.dump_configuration_file(flag+'.conf')
    print('asterism stop')

def from_cl(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('in_file',    type=str, default=None)
    parser.add_argument('conf_file',  type=str, default=None)
    parser.add_argument('-image_id',  type=int, nargs='*' ,help='id of the image',default=None)
    parser.add_argument('-conf_file', type=str, default=None)
    parser.add_argument('-rms_file', type=str, default=None)
    parser.add_argument('-segm_file', type=str, default=None)
    parser.add_argument('-dump_conf_file', action='store_true')
    parser.add_argument('-max_image_id', type=int,default=None)
    parser.add_argument('-save_products', action='store_true')
    parser.add_argument('-wd', type=str)
    parser.add_argument('-plot', action='store_true')

    args = parser.parse_args()
    
    file_name=args.in_file
    conf_file=args.conf_file
    rms_file=args.rms_file
    segm_file=args.segm_file
    image_id=args.image_id
    max_image_id=args.max_image_id
    save_products=args.save_products
    wd=args.wd
    plot=args.plot
        
    if args.dump_conf_file is True:
        pipeline = SrcDetectionPipeline()
        pipeline.dump_configuration_file(conf_file)
    else:

        _run_detection(file_name,conf_file, rms_file=rms_file, segm_file=segm_file, image_id=image_id,max_image_id=max_image_id,plot=plot,save_products=save_products,wd=wd)


if __name__ == "__main__":
    from_cl()



