import numpy as np
import matplotlib.pyplot as plt
import pickle

months={}
months['length']=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
months['name']=['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
months['start']=np.cumsum(months['length'][:-1])
months['start']=np.insert(months['start'],0,0)


######################################################################################
#### ROUTINES TO AGGREGATE INTO MONTHLY VALUES OR PERIODS OF N DAYS (e.g. 7 days) ####
######################################################################################


def divide_into_months(integrand, mode='sum'):
    '''
    Dimension input: N x 544/365 x nch
    Dimension output: N x (nmonths x nch), where nmonts=17/12 according to input dimension
    '''
    nch=integrand.shape[-1]
    n_years=integrand.shape[0]
    if integrand.shape[1]==544:
        l1=months['start']-365
        l1=l1[7:]
        l2=months['start']
        l=list(l1)+list(l2)
        l=np.array(l)+(543-364)
        l=list(l)
        l.append(544)
        starts=np.array(l)
    elif integrand.shape[1]==365:
        starts=list(months['start'])
        starts.append(365)
    else:
        raise NotImplementedError('Shape of input not supported')
    n_months=len(starts)-1   
    integrand_divided=np.zeros((n_years,n_months*nch))
    for im in range(n_months):
        for feat in np.arange(nch):
            if mode=='sum':
                integrand_divided[:,feat*n_months+im]=np.sum(integrand[:,starts[im]:starts[im+1],feat],axis=1)  
            elif mode=='std':
                integrand_divided[:,feat*n_months+im]=np.std(integrand[:,starts[im]:starts[im+1],feat],axis=1)  
            elif mode=='abs':
                integrand_divided[:,feat*n_months+im]=np.sum( np.abs(integrand[:,starts[im]:starts[im+1],feat]),axis=1)  
            elif mode=='abs_sum':
                integrand_divided[:,feat*n_months+im]=np.abs(np.sum(integrand[:,starts[im]:starts[im+1],feat],axis=1))
            else:
                raise NotImplemented('Mode not supported')
    return integrand_divided


def divide_into_periods(data, window, mode='sum', output_shape='nn'):
    '''
    Dimension input: N x 544/365 x nch
    Dimension output: N x nmonths x nch, where nmonts is recomputed according to the period
    '''
    n_ch=data.shape[-1]
    n_years=data.shape[0]
    n_days=data.shape[1]
    n_months=n_days//window
    data_1=data[:,-n_months*window:,:]
    data_1=np.transpose(data_1,(0,2,1))
    data_1=data_1.reshape((n_years,n_ch, n_months, window))
    if mode=='sum':
        data_1=np.sum(data_1,axis=-1)
    elif mode=='mean':
        data_1=np.mean(data_1,axis=-1)
    else:
        raise NotImplemented('Mode not supported')
    data_1=np.transpose(data_1,(0,2,1))
    if output_shape=='pca':
        data_1=np.transpose(data_1,(0,1,2))
        data_1=data_1.reshape(data_1.shape[0],-1)
    print('(divide_into_periods) Output shape:',data_1.shape)
    return data_1


##################################################################
#### ROUTINES TO LOAD DATA DOWNLOADED FROM ZENODO INTO MEMORY ####
##################################################################


def get_data(return_dict_format=False, basefolder='.', plot=True):
    drivers_dict={}
    drivers={}
    ig={}
    for site in ['tropical','boreal','temperate']:
        if plot:
            plt.figure()
        data={}
        data['ppt']=np.load(f'{basefolder}/{site}/PPT_all.npy')
        data['at']=np.load(f'{basefolder}/{site}/AT_all.npy')
        data['par']=np.load(f'{basefolder}/{site}/PAR_all.npy')
        data['gpp']=np.load(f'{basefolder}/{site}/GPPsum_all.npy')
        drivers_dict[site]={}
        for typ in ['ppt','at','par','gpp']:
            temp_0=data[typ].reshape(-1,365)
            st=3
            temp_now=temp_0[st:st+10000,:]
            temp_bef=temp_0[st-1:st-1+10000,:]
            drivers_dict[site][typ]=np.concatenate([temp_bef,temp_now],axis=1)
        labels=np.sum(drivers_dict[site]['gpp'][:,-365:],axis=1)
        ig[site]=pickle.load( open( f"{basefolder}/{site}/integrated_gradients.p", "rb" ) )
        res=np.sum(np.sum(ig[site], axis=1), axis=1)
        if plot:
            plt.title(site)
            plt.plot(labels,res,'o', markersize=2)
            plt.ylabel('Sum of integrated gradient analysis over all features')
            plt.xlabel('Yearly GPP')
            plt.plot(res,res,'--')
        drivers[site]=np.zeros(ig[site].shape)
        drivers[site][:,:,0]=drivers_dict[site]['ppt'][:,-544:]
        drivers[site][:,:,1]=drivers_dict[site]['at'][:,-544:]
        drivers[site][:,:,2]=drivers_dict[site]['par'][:,-544:]
    if return_dict_format:
        return drivers, ig, drivers_dict
    else:
        return drivers, ig
    

def load_gpp_and_mask(percentile=0.1, lower=True, 
                      sites=['tropical','boreal','temperate'], 
                      basefolder_data='./datasets'):
    
    data_reshaped={}
    total_gpp={}
    mask={}
    for site in sites:
        data={}
        data['gpp']=np.load(f'{basefolder_data}/{site}/GPPsum_all.npy')
        data_reshaped[site]={}
        for it,typ in enumerate(['gpp']):
            data_reshaped[site][typ]=data[typ].reshape(-1,365)
            st=3
            total_gpp[site]=np.sum(data_reshaped[site][typ][st:st+10000,:],axis=1)
            quant=np.quantile(total_gpp[site],percentile)
            if lower:
                mask[site]=total_gpp[site]<quant
            else:
                mask[site]=total_gpp[site]>quant
    return total_gpp, mask


