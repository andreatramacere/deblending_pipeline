from astropy.io import fits as pf
import pandas
from pandas import DataFrame
import numpy as np
import glob
import os
import pylab as plt
from .run_asterism import   DataSetDetection


__author__ = "Andrea Tramacere"


class DataSetAnalysis(object):
    
    
    
    def __init__(self,
                 root_rel_path,
                 root_data_paht,
                 data_flag,
                 sample_flag,
                 sample_flag_1,
                 ast_root_path,
                 ast_flag,
                 ast_name,
                 debl_method,
                 sex_path_seg,
                 sex_path_debl,
                 sex_path_debl_1,
                 n_sim=2,
                 sex_flag=None,
                 debl_segmethod='',
                 mag_cut=None):


        self.n_sim = n_sim

        self.data_path = os.path.join(root_rel_path ,root_data_paht,data_flag,'data')


        # pandas table
        _pd=os.path.join('%s'%self.data_path, 'sky_to_CANDELS.pkl')
        print('simulation pandas file',os.path.exists(_pd),_pd)
        # filter
        if os.path.exists(_pd) and mag_cut is not None:
            
            pd=pandas.read_pickle(_pd)
            self.debl_filter=((pd['mag']<mag_cut)&(pd['nearest_mag']<mag_cut))
        else:
            self.debl_filter=None
       
        
        #cube and true map file
        _path = os.path.join(self.data_path, '*%s*_cube*%s*' % (sample_flag, sample_flag_1))
        # print(_path)
        _f = glob.glob(_path)
        # print(_f)
        _cube = [n for n in _f if 'true' not in n]
        _true = [n for n in _f if 'true' in n]

        self.cube=pf.getdata(_cube[0])
        self.true_map=pf.getdata(_true[0])
        print('-> cube file', _cube[0])
        print('-> true map file', _true[0])

        if ast_flag is not None:
            ast_path=os.path.join(ast_root_path,data_flag,ast_name,debl_method,debl_segmethod)

            print('ast_path', os.path.join(ast_path,ast_flag))
            self.ast_path=ast_path
            try:
                _debl_map_ast = glob.glob('%s/%s*segmentation_map_debl_overlap.fits*'%(ast_path,ast_flag))[0]
                _seg_map_ast = glob.glob('%s/%s*segmentation_map.fits*'%(ast_path,ast_flag))[0]
                _deblended_catalog = glob.glob('%s/%s*_deblended_catalog.fits*'%(ast_path,ast_flag))[0]
                _segment_catalog = glob.glob('%s/%s*_segmentation_catalog.fits*'%(ast_path,ast_flag))[0]
            except:
                raise RuntimeError('ast path problem', os.path.join(ast_path,ast_flag))
            print('ast debl_map',_debl_map_ast)
            print('ast seg_map',_seg_map_ast)
            print('ast deblended_catalog',_deblended_catalog)
            print('ast segment_catalog',_segment_catalog)
            self.debl_map_ast=pf.getdata(_debl_map_ast)
            self.seg_map_ast= pf.getdata(_seg_map_ast)
            self.deblended_catalog=pf.getdata(_deblended_catalog)
            self.segment_catalog=pf.getdata(_segment_catalog)

        # print('%s/segmap_debl_detthr_1.2_minarea_10/%s_%s*DebNthr_64_DebMin_0.002_segmentation_map_debl.fits.gz'%(sex_path,data_flag,sex_flag))


        self.sex_deblm_map_path=os.path.join(root_rel_path, sex_path_debl,data_flag,sex_path_debl_1)
        _sex_seg_map_path =os.path.join(root_rel_path,sex_path_seg,'%s*%s*_segmap*' % (data_flag,sample_flag))
        if sex_flag is not None:

            _path_sex_debl_map = os.path.join( self.sex_deblm_map_path,'*%s*segmentation_map_debl*'%sex_flag)
            try:
                segmap_debl_file = glob.glob(_path_sex_debl_map)[0]
                seg_map_file= glob.glob(_sex_seg_map_path)[0]
            except:
                raise RuntimeError('sex path problem', _path)
            print('-> sex debl_map', segmap_debl_file)
            print('-> sex seg map', seg_map_file)
            self.debl_map_sex = pf.getdata(segmap_debl_file)
            self.seg_map_sex = pf.getdata(seg_map_file)
    @classmethod
    def from_name_factory(cls,name,
                          root_rel_path,
                          ast_root_path,
                          debl_method,
                          debl_segmethod,
                          ast_flag,
                          sex_flag,
                          ast_name,
                          mag_cut=None,
                          sex_path_debl='sextractor_detection/segmap_debl_detthr_1.2_minarea_10'):
        _dict={}
        _dict['couple_skymaker_r1'] = cls.build_couple_skymaker_r1
        _dict['couple_skymaker_r5'] = cls.build_couple_skymaker_r5
        _dict['single_skymaker_r1'] = cls.build_single_skymaker_r1
        _dict['single_CANDELS_r1'] = cls.build_single_CANDELS_r1
        _dict['couple_CANDELS_r1'] = cls.build_couple_CANDELS_r1
        _dict['couple_CANDELS_r5'] = cls.build_couple_CANDELS_r5
        _dict['couple_big_skymaker_r10'] = cls.build_couple_big_skymaker_r10
        _dict['couple_big_CANDELS_r10'] = cls.build_couple_big_CANDELS_r10

        return _dict[name](root_rel_path,
                           ast_root_path,
                           debl_method,
                           debl_segmethod,
                           ast_flag,
                           ast_name,
                           sex_flag,
                           mag_cut=mag_cut,
                           sex_path_debl=sex_path_debl)



    @classmethod
    def build_couple_skymaker_r1(cls,
                                 root_rel_path,
                                 ast_root_path,
                                 debl_method,
                                 debl_segmethod,
                                 ast_flag,
                                 ast_name,
                                 sex_flag,
                                 mag_cut=None,
                                 sex_path_debl='sextractor_detection/segmap_debl_detthr_1.2_minarea_10'):
        d = cls(root_rel_path,
                'datasets',
                data_flag='couples_19_26_24.5_d10_r1',
                sample_flag='cat_tot_vis',
                sample_flag_1='CANDELS',
                ast_root_path=ast_root_path,
                ast_flag=ast_flag,
                debl_method=debl_method,
                debl_segmethod=debl_segmethod,
                sex_path_seg='sextractor_detection/segmap_detthr_1.2_minarea_10',
                sex_path_debl=sex_path_debl,
                sex_path_debl_1='r1_skymaker',
                sex_flag=sex_flag,
                ast_name=ast_name,
                mag_cut=mag_cut)

        return d

    @classmethod
    def build_couple_skymaker_r5(cls,
                                 root_rel_path,
                                 ast_root_path,
                                 debl_method,
                                 debl_segmethod,
                                 ast_flag,
                                 ast_name,
                                 sex_flag,
                                 mag_cut=None,
                                 sex_path_debl='sextractor_detection/segmap_debl_detthr_1.2_minarea_10'):
        d = cls(root_rel_path,
                'datasets',
                data_flag='couples_19_26_24.5_d10_r5',
                sample_flag='cat_tot_vis',
                sample_flag_1='CANDELS',
                ast_root_path=ast_root_path,
                ast_flag=ast_flag,
                debl_method=debl_method,
                debl_segmethod=debl_segmethod,
                sex_path_seg='sextractor_detection/segmap_detthr_1.2_minarea_10',
                sex_path_debl=sex_path_debl,
                sex_path_debl_1='r5_skymaker',
                sex_flag=sex_flag,
                ast_name=ast_name,
                mag_cut=mag_cut)

        return d

    @classmethod
    def build_couple_CANDELS_r1(cls,
                                root_rel_path,
                                ast_root_path,
                                debl_method,
                                debl_segmethod,
                                ast_flag,
                                ast_name,
                                sex_flag,
                                mag_cut=None,
                                sex_path_debl='sextractor_detection/segmap_debl_detthr_1.2_minarea_10'):
        d = cls(root_rel_path,
                'datasets',
                data_flag='couples_19_26_24.5_d10_r1',
                sample_flag='real',
                sample_flag_1='n_obj_2',
                ast_root_path=ast_root_path,
                ast_flag=ast_flag,
                debl_method=debl_method,
                debl_segmethod=debl_segmethod,
                sex_path_seg='sextractor_detection/segmap_detthr_1.2_minarea_10',
                sex_path_debl=sex_path_debl,
                sex_path_debl_1='r1_candels',
                sex_flag=sex_flag,
                ast_name=ast_name,
                mag_cut=mag_cut)

        return d

    @classmethod
    def build_couple_CANDELS_r5(cls,
                                root_rel_path,
                                ast_root_path,
                                debl_method,
                                debl_segmethod,
                                ast_flag,
                                ast_name,
                                sex_flag,
                                mag_cut=None,
                                sex_path_debl='sextractor_detection/segmap_debl_detthr_1.2_minarea_10'):
        d = cls(root_rel_path,
                'datasets',
                data_flag='couples_19_26_24.5_d10_r5',
                sample_flag='real',
                sample_flag_1='n_obj_2',
                ast_root_path=ast_root_path,
                ast_flag=ast_flag,
                debl_method=debl_method,
                debl_segmethod=debl_segmethod,
                sex_path_seg='sextractor_detection/segmap_detthr_1.2_minarea_10',
                sex_path_debl=sex_path_debl,
                sex_path_debl_1='r5_candels',
                sex_flag=sex_flag,
                ast_name=ast_name,
                mag_cut=mag_cut)

        return d

    @classmethod
    def build_single_skymaker_r1(cls,
                                 root_rel_path,
                                 ast_root_path,
                                 debl_method,
                                 debl_segmethod,
                                 ast_flag,
                                 ast_name,
                                 sex_flag,
                                 mag_cut=None,
                                 sex_path_debl='sextractor_detection/segmap_debl_detthr_1.2_minarea_10'):
        d = cls(root_rel_path,
                'datasets',
                data_flag='single_19_26_24.5_d10_r1',
                sample_flag='cat_tot_vis',
                sample_flag_1='CANDELS',
                ast_root_path=ast_root_path,
                ast_flag=ast_flag,
                debl_method=debl_method,
                debl_segmethod=debl_segmethod,
                sex_path_seg='sextractor_detection/segmap_detthr_1.2_minarea_10',
                sex_path_debl=sex_path_debl,
                sex_path_debl_1='r1_skymaker',
                sex_flag=sex_flag,
                ast_name=ast_name,
                mag_cut=mag_cut,
                n_sim=1)
        return d

    @classmethod
    def build_single_CANDELS_r1(cls,
                                root_rel_path,
                                ast_root_path,
                                debl_method,
                                debl_segmethod,
                                ast_flag,
                                ast_name,
                                sex_flag,
                                mag_cut=None,
                                sex_path_debl='sextractor_detection/segmap_debl_detthr_1.2_minarea_10'):
        d = cls(root_rel_path,
                'datasets',
                data_flag='single_19_26_24.5_d10_r1',
                sample_flag='real',
                sample_flag_1='n_obj_1',
                ast_root_path=ast_root_path,
                ast_flag=ast_flag,
                debl_method=debl_method,
                debl_segmethod=debl_segmethod,
                sex_path_seg='sextractor_detection/segmap_detthr_1.2_minarea_10',
                sex_path_debl=sex_path_debl,
                sex_path_debl_1='r1_candels',
                sex_flag=sex_flag,
                ast_name=ast_name,
                mag_cut=mag_cut,
                n_sim=1)

        return d


    @classmethod
    def build_couple_big_skymaker_r10(cls,
                                      root_rel_path,
                                      ast_root_path,
                                      debl_method,
                                      debl_segmethod,
                                      ast_flag,
                                      ast_name,
                                      sex_flag,
                                      mag_cut=None,
                                      sex_path_debl='sextractor_detection/segmap_debl_detthr_1.2_minarea_10'):
        d = cls(root_rel_path,
                'datasets',
                data_flag='big_19_23_24.5_d50_r10',
                sample_flag='cat_tot_vis',
                sample_flag_1='CANDELS',
                ast_root_path=ast_root_path,
                ast_flag=ast_flag,
                debl_method=debl_method,
                debl_segmethod=debl_segmethod,
                sex_path_seg='sextractor_detection/segmap_detthr_1.2_minarea_10',
                sex_path_debl=sex_path_debl,
                sex_path_debl_1='big_r10_skymaker',
                sex_flag=sex_flag,
                ast_name=ast_name,
                mag_cut=mag_cut)

        return d

    @classmethod
    def build_couple_big_CANDELS_r10(cls,
                                     root_rel_path,
                                     ast_root_path,
                                     debl_method,
                                     debl_segmethod,
                                     ast_flag,
                                     ast_name,
                                     sex_flag,
                                     mag_cut=None,
                                     sex_path_debl='sextractor_detection/segmap_debl_detthr_1.2_minarea_10'):
        d = cls(root_rel_path,
                'datasets',
                data_flag='big_19_23_24.5_d50_r10',
                sample_flag='real',
                sample_flag_1='n_obj_2',
                ast_root_path=ast_root_path,
                ast_flag=ast_flag,
                debl_method=debl_method,
                debl_segmethod=debl_segmethod,
                sex_path_seg='sextractor_detection/segmap_detthr_1.2_minarea_10',
                sex_path_debl=sex_path_debl,
                sex_path_debl_1='big_r10_candels',
                sex_flag=sex_flag,
                ast_name=ast_name,
                mag_cut=mag_cut)

        return d

def select_catalog_par(debl_rep,catalog,p,msk_sel,true_map,seg_map):
    x=[]
    p_sel=[]
    
    for ID,img_ID in enumerate(debl_rep['image_ID'][msk_sel]):
        id_parent=np.unique(seg_map[img_ID][np.logical_and(true_map[img_ID],seg_map[img_ID])])
        id_parent=id_parent[id_parent>0]
        msk=np.logical_and(np.isin(catalog['src_ID'],id_parent),catalog['image_ID']==img_ID)
        #print(id_parent)
        if msk.sum()<1:
            pass
        else:
            #print (catalog['image_ID'][msk])
            x.extend(p[msk])
           
            
    for ID,img_ID in enumerate(debl_rep['image_ID']):
        id_parent=np.unique(seg_map[img_ID][np.logical_and(true_map[img_ID],seg_map[img_ID])])
        id_parent=id_parent[id_parent>0]
        msk=np.logical_and(np.isin(catalog['src_ID'],id_parent),catalog['image_ID']==img_ID)
        #print(id_parent)
       
       
        if msk.sum()<1:
            pass
        else:
            #print (catalog['image_ID'][msk])
            
            p_sel.extend(p[msk])
     
    return np.array(x),np.array(p_sel)


def anlysis(debl_rep,segment_catalog,deblended_catalog,n_sim,true_map,seg_map):
    
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(12,4))
    p=segment_catalog['r_max']/segment_catalog['r_cluster']
    
    h=np.zeros(p.size)
    for ID,im_id in enumerate(segment_catalog['image_ID']):
        _h=deblended_catalog['h'][deblended_catalog['image_ID']==im_id]
        if len(_h)==0:
            h[ID]=None
        else:
            h[ID]=_h[0]

    print(h[~np.isnan(h)].min(),h[~np.isnan(h)].max())
    p=h/segment_catalog['r_max']

    msk_sel=debl_rep['overlap']>n_sim
    x,p_sel=select_catalog_par(debl_rep,segment_catalog,p,msk_sel,true_map,seg_map)
    ax1.hist(p_sel,density=True)
    ax1.hist(x,fill=False,density=True,edgecolor='red',alpha=0.5)

    msk_sel=debl_rep['overlap']<n_sim
    x,p_sel=select_catalog_par(debl_rep,segment_catalog,p,msk_sel,true_map,seg_map)
    ax2.hist(p_sel,density=True)
    ax2.hist(x,fill=False,density=True,edgecolor='blue',alpha=0.5)

    msk_sel=debl_rep['overlap']==n_sim
    x,p_sel=select_catalog_par(debl_rep,segment_catalog,p,msk_sel,true_map,seg_map)
    ax3.hist(p_sel,density=True)
    ax3.hist(x,fill=False,density=True,edgecolor='black',alpha=0.5)






