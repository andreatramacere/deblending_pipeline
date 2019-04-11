from pandas import DataFrame
import numpy as np
import  pylab as plt

from asterism.core.geometry.distance import dist_eval
from asterism.analysis_tasks.source_detection.image_segmentation.image_segmentation import do_image_segmentation,build_seg_map
from asterism.core.image_manager.image import Image

__author__ = "Andrea Tramacere"


def get_cluster_distance_to_clusters_list(cluster,clusters_list):
    """[summary]
    
    Parameters
    ----------
    cluster : astersim clusters
        [description]
    clusters_list : [type]
        [description]
    
    Returns
    -------
    distance array : 1d np.arra
        
    lis of ID : list of integers

    """

    center_pos_array = np.zeros((len(clusters_list), 2))
    cl_ID_list = []

    for ID, cl in enumerate(clusters_list):
        center_pos_array[ID] = (cl.x_c, cl.y_c)
        cl_ID_list.append(cl.ID)
    dist = dist_eval(center_pos_array,x_c=cluster.x_c,y_c=cluster.y_c,metric=cluster._metric)
    #_d = sorted(zip(dist, cl_ID_list))
    #dist, cl_ID_list = zip(*_d)
    return dist, cl_ID_list





def detect_cluster_with_overlap(image,input_seg_map_overlap,ID_target,image_ID):
    """This function detects a cluster with ID==ID_target strating from a segmap with overlappin ID (-1)
    Pixels with  ID==-1 and continguos to pixels with ID==ID_target are merged together 
    and the corresponding cluster is detected
    
    Parameters
    ----------
    image : asterims Image
        [description]
    input_seg_map_overlap : 2dim numpy.array
        [description]
    ID_target : int
        [description]
    
    Raises
    ------
    RuntimeError
        if the cluster with ID==ID_target is not detected
    RuntimeError
        if the cluster with ID==ID_target is not detected, or more then one are detected
    
    Returns
    -------
    seg_map : 2dim numpy.array
        the segmap with -1 pixes properly aggregated to adiacent ID_target pixels
    _cl_list : list of asterism clusters
        
    """
    #we clean the input_seg_map_overlap keeping only 
    #pixels with ID==-1 or ID==ID_target
    #there might be region with non contiguos to ID_target, coming from overlapping 
    #from other sources with  ID!=ID_target
    input_seg_map_overlap=input_seg_map_overlap.astype(np.int)
    input_seg_map=np.copy(input_seg_map_overlap)
    
    input_seg_map[input_seg_map == -1] = ID_target
    input_seg_map[input_seg_map!=ID_target]=0
    #image.masked=np.logical_not(input_seg_map)

    #now we detect all the conneted regions in the input_seg_map
    #in this way all the ID=-1 in the original segmap are merged 
    #with contigous ID==ID_target pixels
    _image=Image(input_seg_map)
    _cl_list, K, selected_coords = do_image_segmentation(image=_image,
                                                        bkg_threshold=0.5,
                                                        method='connected')

    seg_map_contig,_foo=build_seg_map(image, _cl_list, 0, out_seg_map_overlap=False)

    #now we select in the seg_map_contig, the region with the largest overlap 
    #with  input_seg_map_overlap. Of course only the cluster with the contiguos
    #attached -1 pixels has increased in size. In this way we remove the -1 islands
    #due to overlappin with with  ID!=ID_target in the original input_seg_map
    if len(_cl_list)>=1:
        _ov=[]
        for _cl in _cl_list:
            msk=seg_map_contig==_cl.ID
            msk=np.logical_and(msk, input_seg_map_overlap!=1)
            _ov.append(msk.sum())

        
        _selected_ID=np.argmax(_ov)
        _cl_list=[_cl_list[_selected_ID]]
        slected_cl=_cl_list[0]
    else:
        raise RuntimeError('the segmap connected islands  have not been detected',len(_cl_list),'target ID',ID_target,'image_ID',image_ID)
    
    #we update the seg_map_contig keeping only pixels from slected_cl
    seg_map_contig[seg_map_contig!=slected_cl.ID]=0
    seg_map_contig[seg_map_contig==slected_cl.ID]=ID_target
        
   
    
    #now we extact the cluster using the cleaned contiguos segmap
    _cl_list, K, selected_coords= do_image_segmentation(image=image,
                                                        seg_map_bkg_val=0,
                                                        bkg_threshold=0,
                                                        input_seg_map=seg_map_contig,                 
                                                        method='from_seg_map')
    
    #this is an extra control and should never happen!
    if len(_cl_list)==0 or len(_cl_list)>1:
        raise RuntimeError('the cluster has not been detected or more than one have been detected',len(_cl_list),ID_target)
    else:
        pass
    
    #seg_map_contig,_foo=build_seg_map(image, _cl_list, 0, out_seg_map_overlap=False)
    #seg_map_contig[seg_map_contig!=_cl_list[0].ID]=0
    #seg_map_contig[seg_map_contig==_cl_list[0].ID]=ID_target   
    #_cl_list[0].ID=ID_target
   
    return seg_map_contig,_cl_list

def build_candidate_list(image,true_map,deblended_map,ID_sim,ID_det_overlap_list,verbose=False,plot=False,image_ID=None):
    """[summary]
    
    Parameters
    ----------
    image : [type]
        [description]
    true_map : [type]
        [description]
    deblended_map : [type]
        [description]
    ID_sim : [type]
        [description]
    ID_det_overlap_list : [type]
        [description]
    verbose : bool, optional
        [description] (the default is False, which [default_description])
    plot : bool, optional
        [description] (the default is False, which [default_description])
    
    Raises
    ------
    RuntimeError
        [description]
    RuntimeError
        [description]
    
    Returns
    -------
    [type]
        [description]
    """

    img=Image(image)
    
    
    
    #sim_cl_list=[]  
    #for ID_sim in ID_sim_list:
    seg_map_overlapped,sim_cl_list= detect_cluster_with_overlap(img,true_map,ID_sim,image_ID)
    #sim_cl_list.extend(_cl_list)
    if len(sim_cl_list)==0 or len(sim_cl_list)>1:
        raise RuntimeError('the cluster has not been detected or more than one have been detected',len(sim_cl_list))
    cl_sim=sim_cl_list[0]
    
    det_cl_list=[]
    for ID_det in ID_det_overlap_list:
          
        seg_map_overlapped,_cl_list= detect_cluster_with_overlap(img,deblended_map,ID_det,image_ID)
        if len(_cl_list)!=1:
            raise  RuntimeError('the cluster has not been detected or more than one have been detected',len(_cl_list),ID_det)
        cl_det=_cl_list[0]
        
        
        rec_dict={}
        
        #if verbose==True:
        #    print('--> ID_det',ID_det,'ID_segmap',np.unique(seg_map_overlapped),'cl_sel ID',cl.ID,'ID_det_overlap_list',ID_det_overlap_list)
        #for cl_sim in sim_cl_list:
        msk_det=seg_map_overlapped==ID_det
        qual_dict={}
        msk_sim=np.logical_or(true_map==ID_sim,true_map==-1)
        roi=np.logical_and(msk_det,msk_sim)

        #msk=np.zeros(seg_map.shape)
        #roi=np.logical_and(data.true_map[image_ID]==ID_sim,seg_map==ID_det)
        #roi=np.logical_or(roi,data.true_map[image_ID]==-1)

        #msk_sim=np.logical_or(true_map==ID_sim,true_map==-1)
        #msk_det=seg_map_overlapped==ID_det
        #roi= np.logical_and(msk_det,msk_sim).sum()
        qual_dict['det_ID']=cl_det.ID
        qual_dict['rec_det_frac']=msk_det.sum()/msk_sim.sum()
        qual_dict['rec_sim_frac']=roi.sum()/msk_sim.sum()
        qual_dict['dist']=dist_eval([(cl_sim.x_c,cl_sim.y_c)],x_c=cl_det.x_c,y_c=cl_det.y_c,metric=cl_det._metric)[0]
        rec_dict[cl_sim.ID]=qual_dict
        if plot==True:
            print('   test with ID_sim',ID_sim,'roi',roi.sum(),'det',msk_det.sum(),'sim',msk_sim.sum(),'len',cl_det.cl_len)
            fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(12,4))
            ax1.imshow(msk_det)
            ax2.imshow(msk_sim)
            ax3.imshow(roi)
            ax3.set_title("sim=%f, det=%f"%(roi.sum()/msk_sim.sum(),roi.sum()/msk_det.sum()))
            ax1.set_title('%d'%ID_det)
            plt.show()
               
        cl_det.rec_dict=rec_dict
        det_cl_list.append(cl_det)
            
        
  
    return det_cl_list
    
        



def build_candidate_df(cube,true_map,deblended_map,overlap_th=-1,verbose=False):
    """
    
    """
    
    print('build_candidate_df','true map shape',true_map.shape,)
    
    
    Table=[]
    for ID_img,tm in enumerate(true_map):
        
        #msk of simulated
        #you mus keep -1 for overlap so the condition is !=0
        msk_sim=true_map[ID_img]!=0
        
        #msk of detected
        msk_det=np.logical_and(msk_sim,deblended_map[ID_img]>0)
        
        
        #sim list
        #in this case you don't want to keep -1 in the id list
        ID_sim_list=np.unique(true_map[ID_img][true_map[ID_img]>0]).astype(np.int)
        
        #detected in the full stamp
        n_found=len(np.unique(deblended_map[ID_img][deblended_map[ID_img]>0]).astype(np.int))
        
        #detected list overlapping the true map
        #these will be splitted in associated and contaminant
        ID_det_overlap_list=np.unique(deblended_map[ID_img][msk_det]).astype(np.int)
        ID_det_overlap_list=ID_det_overlap_list[ID_det_overlap_list>0]
        
        if verbose==True:
            print('---->  build candidate IMAGE',ID_img)
        
        for ID,ID_sim in enumerate(ID_sim_list):
            try:
                canditate_list=build_candidate_list(cube[ID_img],true_map[ID_img],deblended_map[ID_img],ID_sim,ID_det_overlap_list,verbose=verbose,image_ID=ID_img)
                tab_row=[ID_img,ID_sim,ID_det_overlap_list,[_c.rec_dict[ID_sim] for _c in canditate_list]]
            except:
                tab_row=[ID_img,ID_sim,[],[]]
            Table.append(tab_row)
        
        df=DataFrame(Table,columns=['image_ID', 'sim_ID','ID_det_list', 'rec_dict_list'])
    return df

