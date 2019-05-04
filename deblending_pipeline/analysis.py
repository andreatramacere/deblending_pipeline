import numpy as np
from pandas import DataFrame

from .table import build_candidate_df

__author__ = "Andrea Tramacere"


def get_associated_and_contaminant(candidate_df, image_ID, ID_sim_list, verbose=False):
    #print('---> image_ID',image_ID)
    #print(ID_sim_list[0],
    #      np.sum(candidate_df['sim_ID']==ID_sim_list[0]),
    #      np.sum(candidate_df['image_ID']==image_ID),
    #      np.sum(np.logical_and(candidate_df['sim_ID']==ID_sim_list[0] ,candidate_df['image_ID']==image_ID)),
    #      np.argwhere(np.logical_and(candidate_df['sim_ID'] == ID_sim_list[0], candidate_df['image_ID'] == image_ID))
    #      )

    assoc_dict = {}
    contaminant_dict = {}
    assoc_list = []
    contaminant_list = []
    sel_row=candidate_df.loc[candidate_df['image_ID']==image_ID]
    failed=np.sum(sel_row['failed'])>0
    if len(ID_sim_list)>0 and ~ failed:
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


        for sim_ID in ID_sim_list:
            sel_row = np.argwhere(np.logical_and(candidate_df['sim_ID']==sim_ID ,candidate_df['image_ID']==image_ID))[0][0]
            sim_row = candidate_df.loc[sel_row]
            _l = []
            # print(sim_row.keys())
            for rec_dict in sim_row['rec_dict_list']:
                if rec_dict['det_ID'] in contaminant_list:
                        _l.append(rec_dict)
            contaminant_dict[sim_ID] =_l
    else:
        failed=True

    #print('failed->', failed)
    if verbose is True:
        print('assoc_dict', assoc_dict)
        print('contaminant_dict', contaminant_dict)
        print('contaminant_list', contaminant_list)
    return assoc_dict, contaminant_dict, contaminant_list, assoc_list,failed


def debl_quality_analysis(true_map, candidate_df, rec_det_th=-1, rec_sim_th=-1, contam_th=-1, verbose=False,mag_cut=-1):
    print('debl_quality_analysis', 'true map shape', true_map.shape)
    #out=np.zeros(true_map.shape[0],dtype=[('image_ID', '>i4'),
    #                                      ('failed','bool'),
    #                                      ('success_n', 'bool'),
    #                                      ('success_qual', 'bool'),
    #                                      ('overlap', 'i4'),
    #                                      ('assoc', 'i4'),
    #                                      ('contaminant', 'i4')])
    out=[]
    for ID_img,tm in enumerate(true_map):
        
        ID_sim_list = np.unique(true_map[ID_img][true_map[ID_img] > 0]).astype(np.int)
        if verbose is True:
            print('---->  quality IMAGE',ID_img)
        
        assoc_dict,contaminant_dict,contaminant_list,assoc_list,failed=get_associated_and_contaminant(candidate_df,ID_img,ID_sim_list,verbose=verbose)
        

        if ~failed :
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

            out.append([ID_img, failed,success_n,success_qual,n_overlap,n_assoc,len(contaminant_list),rec_det_th,rec_sim_th,contam_th,mag_cut])
            if verbose is True:
                print('----> <-----')
        else:
            out.append([ID_img, failed, -1, -1, -1, -1, 0,rec_det_th,rec_sim_th,contam_th,mag_cut])
    return DataFrame(out,columns=['image_ID','failed', 'success_n','success_qual', 'overlap','assoc','contaminant','rec_det_th', 'rec_sim_th', 'contam_th','mag_cut'])

def eval_stats(debl_analysis_table,n_sim,debl_filter=None,rec_det_th=-1, rec_sim_th=-1, contam_th=-1,mag_cut=-1):

    if debl_filter is not None:
        debl_analysis_table=debl_analysis_table.loc[debl_filter.values]

    print('filtered size for debl filter', debl_analysis_table.shape)
    failed = (debl_analysis_table['failed']==True)
    print('failed      ', failed.sum())
    if failed.sum()>0:
        print('failed list', debl_analysis_table['image_ID'][failed])
    print()
    debl_analysis_table=debl_analysis_table.loc[~failed.values]
    print('filtered size for failed', debl_analysis_table.shape)
    over = debl_analysis_table['contaminant'] > 0
    under = debl_analysis_table['assoc'] < n_sim
    # non det<1 and not failed
    # need to remove failed from det<1, because det=-1 for failed
    non_det =  debl_analysis_table['overlap'] < 1



    net_count = debl_analysis_table['image_ID'].size - non_det.sum()
    det_ok_frac = debl_analysis_table['success_n'].sum() / debl_analysis_table['image_ID'].size

    frac_ok_th = debl_analysis_table['success_qual'].sum() / debl_analysis_table['image_ID'].size
    frac_ok_th_real = debl_analysis_table['success_qual'].sum() / (net_count)

    over_frac = over.sum() / debl_analysis_table['image_ID'].size
    under_frac = under.sum() / debl_analysis_table['image_ID'].size

    over_frac_real = over.sum() / (net_count)
    under_frac_real = under.sum() / (net_count)
    if n_sim == 1:
        under_frac = None
        under_frac_real = None

    print('stats below are net of failed')
    print('fraction of debl OK n_det  ', det_ok_frac,
          '\nfraction of debl OK>th     ', frac_ok_th,
          '\nfraction of underdebl      ', under_frac,
          '\nfraction of overdebl       ', over_frac,
          '\nfraction of non-detected   ', 1.0 - net_count / debl_analysis_table['image_ID'].size,
          # '\nspurious (not in true map) ',debl_analysis_table['found'].sum()-debl_analysis_table['overlap'].sum(),
          '\nfraction of debl OK>th     (excluding non-detected/failed)', frac_ok_th_real,
          '\nfraction of underdebl      (excluding non-detected/failed)', under_frac_real,
          '\nfraction of overdebl       (excluding non-detected/failed)', over_frac_real)

    print()
    print('non detected', non_det.sum())

    ID_list_KO_over_ast = list(debl_analysis_table['image_ID'][over].values )
    print('len over list', len(ID_list_KO_over_ast))
    print('over list', ID_list_KO_over_ast)

    ID_list_KO_under_ast = list(debl_analysis_table['image_ID'][under].values )
    if n_sim == 1:
        print('len non_det list', len(ID_list_KO_under_ast))
        print('non_det/failed', ID_list_KO_under_ast)
    else:
        print('len under list', len(ID_list_KO_under_ast))
        print('under list', ID_list_KO_under_ast)
    print('------------------------------------------------')
    print()
    debl_stats = np.zeros(1, dtype=[('frac_debl_ok', 'f4'),
                                    ('frac_debl_ok_th', 'f4'),
                                    ('frac_under', 'f4'),
                                    ('frac_over', 'f4'),
                                    ('frac_ok_th_net', 'f4'),
                                    ('frac_under_net', 'f4'),
                                    ('frac_overd_net', 'f4'),
                                    ('n_tot', 'i4'),
                                    ('n_net', 'i4'),
                                    ('rec_det_th', 'f4'),
                                    ('rec_sim_th', 'f4'),
                                    ('contam_th', 'f4',),
                                    ('mag_cut', 'f4',)])

    debl_stats[0] = (det_ok_frac,
                     frac_ok_th,
                     under_frac,
                     over_frac,
                     frac_ok_th_real,
                     under_frac_real,
                     over_frac_real,
                     debl_analysis_table['image_ID'].size,
                     net_count,
                     rec_det_th,
                     rec_sim_th,
                     contam_th,
                     mag_cut)

    return debl_stats

def deblending_analysis(cube, true_map, debl_map, name, n_sim, debl_filter=None, rec_sim_th=-1, rec_det_th=-1, overlap_th=1,mag_cut=-1, contam_th=-1, verbose=False,candidate_df=None):

    print('------------------------------------------------')
    print('name ',name)
    #print('candidate_df ',candidate_df)

    if candidate_df is None:
        candidate_df=build_candidate_df(cube,true_map,debl_map,overlap_th=-1,verbose=verbose)
    print('mag_cut',mag_cut)
    # df=pandas.read_pickle('df.pd')
    debl_analysis_table= debl_quality_analysis(true_map,candidate_df,rec_sim_th=rec_sim_th,rec_det_th=rec_det_th,contam_th=contam_th,verbose=verbose,mag_cut=mag_cut)
    




    debl_stats = eval_stats(debl_analysis_table, n_sim, debl_filter=debl_filter,rec_sim_th=rec_sim_th,rec_det_th=rec_det_th,mag_cut=mag_cut,contam_th=contam_th)

    #debl_stats=[-1,-1,-1,-1,-1,-1,-1]

    return debl_analysis_table,candidate_df,debl_stats
