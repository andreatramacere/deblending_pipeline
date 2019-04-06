import numpy as np
from .table import build_candidate_df

__author__ = "Andrea Tramacere"


def get_associated_and_contaminant(candidate_df, image_ID, ID_sim_list, verbose=False):
    assoc_dict = {}
    assoc_list = []
    sel_row = np.argwhere(np.logical_and(candidate_df['sim_ID']==ID_sim_list[0] ,candidate_df['image_ID']==image_ID))[0][0]
    sim_row = candidate_df.loc[sel_row]
    
    contaminant_list=sim_row['ID_det_list'].tolist()
    # if verbose==True:
    #    print('gett assoc cont for image ->',image_ID,'sim_list',ID_sim_list,contaminant_list)
    # print(type(sim_row))
    for sim_ID in ID_sim_list:
        sel_row = np.argwhere(np.logical_and(candidate_df['sim_ID']==sim_ID ,candidate_df['image_ID']==image_ID))[0][0]
        sim_row = candidate_df.loc[sel_row]
        # sim_row= df.loc[df['sim_ID']==sim_ID]
        # if verbose==True:
        #   print('sim_row ->',sim_row['rec_dict_list'])
        #    for rec_dict in sim_row['rec_dict_list']:
        #        print('rec_dict',rec_dict['dist'])
        
        # print('ID sim',sim_ID)
        dist=[rec_dict['dist'] for rec_dict in sim_row['rec_dict_list'] ]
        
        # dist, cl_ID_list=get_cluster_distance_to_clusters_list(sim_cl,det_cl_list)
        #
        if len(dist)>0:
            _selected=np.argmin(dist)
            # print('dist',dist,_selected,sim_row['rec_dict_list'][_selected]['det_ID'])
            assoc_dict[sim_ID]=sim_row['rec_dict_list'][_selected]
            # contaminant_list.append(sim_row['rec_dict_list'][_selected]['det_ID'])
            if sim_row['rec_dict_list'][_selected]['det_ID'] in contaminant_list:
                contaminant_list.remove(sim_row['rec_dict_list'][_selected]['det_ID'])
                assoc_list.append(sim_row['rec_dict_list'][_selected]['det_ID'])
    
    contaminant_dict = {}
    for sim_ID in ID_sim_list:
        sel_row = np.argwhere(np.logical_and(candidate_df['sim_ID']==sim_ID ,candidate_df['image_ID']==image_ID))[0][0]
        sim_row = candidate_df.loc[sel_row]
        _l = []
        # print(sim_row.keys())
        for rec_dict in sim_row['rec_dict_list']:
            if rec_dict['det_ID'] in contaminant_list:
                    _l.append(rec_dict)
        contaminant_dict[sim_ID] =_l
                        
    if verbose is True:
        print('assoc_dict', assoc_dict)
        print('contaminant_dict', contaminant_dict)
        print('contaminant_list', contaminant_list)
    return assoc_dict, contaminant_dict, contaminant_list, assoc_list


def debl_quality_analysis(true_map, candidate_df, rec_det_th=-1, rec_sim_th=-1, contam_th=-1, verbose=False):
    print('debl_quality_analysis', 'true map shape', true_map.shape)
    out=np.zeros(true_map.shape[0],dtype=[('image_ID', '>i4'), ('success_n', 'bool'), ('success_qual', 'bool'), ('overlap', '>i4'), ('assoc', '>i4'), ('contaminant', '>i4')])
    for ID_img,tm in enumerate(true_map):
        
        ID_sim_list = np.unique(true_map[ID_img][true_map[ID_img] > 0]).astype(np.int)
        if verbose is True:
            print('---->  quality IMAGE',ID_img)
        
        assoc_dict,contaminant_dict,contaminant_list,assoc_list=get_associated_and_contaminant(candidate_df,ID_img,ID_sim_list,verbose=verbose)
        
        n_sim = len(ID_sim_list)
        n_assoc = len(assoc_list)
        n_overlap = n_assoc+len(contaminant_list)
        
        rec_det = np.zeros(len(ID_sim_list))
        rec_sim = np.zeros(len(ID_sim_list))
        # cont_frac=np.zeros(n_assoc)
        
        contaminant_list=[]
        # print(assoc_dict,contaminant_dict)
       
        for ID, ID_sim in enumerate(assoc_dict):
            # tab_row=[ID_img,ID_sim,[_c.rec_dict[ID_sim] for _c in canditate_list]]
            # Table.append(tab_row)
            assoc_rec_dict=assoc_dict[ID_sim]
            
            if verbose is True:
                print('---> ID_sim ', ID_sim, 'associated to det', assoc_rec_dict['det_ID'], 'n_sim', n_sim, 'assoc_list', assoc_list)
            
            rec_det[ID]=assoc_rec_dict['rec_det_frac']
            rec_sim[ID]=assoc_rec_dict['rec_sim_frac']
            if ID_sim in contaminant_dict:
                if verbose is True:
                    print([v['rec_sim_frac'] for v in contaminant_dict[ID_sim] if v['rec_sim_frac']>contam_th])
                contaminant_list.extend([v['rec_sim_frac'] for v in contaminant_dict[ID_sim] if v['rec_sim_frac']>contam_th])
       
        success_n = n_assoc == n_sim and len(contaminant_list) == 0
        success_qual = success_n and np.sum(rec_sim > rec_sim_th) == n_sim
        success_qual = success_qual and np.sum(rec_det > rec_det_th) == n_sim
        if verbose is True:
            print('IMG', ID_img, 'success_n', success_n, 'n_assoc', n_assoc, 'n_sim', n_sim, 'len cont', len(contaminant_list), 'n_overlap', n_overlap, 'sim', rec_sim, 'det', rec_det)
            if success_n is True and len(contaminant_list)>0:
                for ID,ID_sim in enumerate(assoc_dict):
                    print('IMG', ID_img, 'len cont', len(contaminant_list), 'n_overlap', n_overlap, 'sim', rec_sim, 'det', rec_det)
                    assoc_rec_dict = assoc_dict[ID_sim]
                    print('sim ID', assoc_rec_dict)
                print('----------')
            
        out[ID_img] = (ID_img+1,success_n,success_qual,n_overlap,n_assoc,len(contaminant_list))
        if verbose is True:
            print('----> <-----')
        # df=DataFrame(Table,columns=['image_ID', 'sim_ID', 'rec_dict_list'])
    return out


def deblending_analysis(cube, true_map, debl_map, name, n_sim, debl_filter=None, rec_sim_th=-1, rec_det_th=-1, overlap_th=1, contam_th=-1, verbose=False,candidate_df=None):

    print('------------------------------------------------')
    print('%s'%name)

    if candidate_df is None:
        candidate_df=build_candidate_df(cube,true_map,debl_map,overlap_th=-1,verbose=verbose)

    # df=pandas.read_pickle('df.pd')
    debl_rep = debl_quality_analysis(true_map,candidate_df,rec_sim_th=rec_sim_th,rec_det_th=rec_det_th,contam_th=contam_th,verbose=verbose)
    
    if debl_filter is not None:
            debl_rep=debl_rep[debl_filter]
    
    print('filtered size',debl_rep.size)
    over=debl_rep['contaminant']>0
    under=debl_rep['assoc']<n_sim
    non_det=debl_rep['overlap']<1
    
    det_ok_frac=debl_rep['success_n'].sum()/debl_rep['image_ID'].size

    frac_ok_th=debl_rep['success_qual'].sum()/debl_rep['image_ID'].size  
    frac_ok_th_real=debl_rep['success_qual'].sum()/(debl_rep['image_ID'].size-non_det.sum())
    
    over_frac=over.sum()/debl_rep['image_ID'].size
    under_frac=under.sum()/debl_rep['image_ID'].size
    
    over_frac_real=over.sum()/(debl_rep['image_ID'].size-non_det.sum())
    under_frac_real=under.sum()/(debl_rep['image_ID'].size-non_det.sum())
    if n_sim==1:
        under_frac=None
        under_frac_real=None
        
    print( 'fraction of debl OK n_det  ', det_ok_frac,
           '\nfraction of debl OK>th     ',frac_ok_th,
           '\nfraction of underdebl      ',under_frac,
           '\nfraction of overdebl       ',over_frac,
           '\nfraction of non-detected   ',non_det.sum()/debl_rep['image_ID'].size,
           # '\nspurious (not in true map) ',debl_rep['found'].sum()-debl_rep['overlap'].sum(),
           '\nfraction of debl OK>th     (excluding non-detected)',frac_ok_th_real,
           '\nfraction of underdebl      (excluding non-detected)',under_frac_real,
           '\nfraction of overdebl       (excluding non-detected)',over_frac_real)
           
    print()
    ID_list_KO_over_ast = debl_rep['image_ID'][over]-1
    print('len over list',len(ID_list_KO_over_ast))
    print('over list',ID_list_KO_over_ast)

    ID_list_KO_under_ast = debl_rep['image_ID'][under]-1
    if n_sim == 1:
        print('len non_det list', len(ID_list_KO_under_ast))
        print('non_det', ID_list_KO_under_ast)
    else:
        print('len under list', len(ID_list_KO_under_ast))
        print('under list', ID_list_KO_under_ast)
    print('------------------------------------------------')
    print()
    return debl_rep,candidate_df
