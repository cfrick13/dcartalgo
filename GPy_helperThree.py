#GPy_helper

import GPy
import re
import numpy as np
import matplotlib.pyplot as plt

def catchImportantWords(ss20):
    m = re.findall(r'SMAD|GENE|doubt', ss20, re.IGNORECASE)
    ss21 = m[0]

    if bool(re.search('GENE',ss21)):
        m = re.findall(r'caga|iffl|median|total|doubt', ss20, re.IGNORECASE)
        ss22 = str(m[0])
        ss2 = ss22
        newstrsub = ss2
        if bool(re.findall(r'median|total', ss22, re.IGNORECASE)):
            ss2 = ss22+' $\itsnail$:mCherry' 
            newstrsub = '$\itsnail$:mCherry' 
        if bool(re.findall(r'SYST1|doubt', ss20, re.IGNORECASE)):
            ss2 = ss22 + ', f(Smad complex)' 
            newstrsub = ss22
        elif bool(re.findall(r'SYST2|doubt', ss20, re.IGNORECASE)):
            ss2 = ss22 + ', f(Smad complex, X)' 
            newstrsub = ss22
        elif bool(re.findall(r'SYST3|doubt', ss20, re.IGNORECASE)):
            ss2 = ss22 + ', f(Smad complex, many X)' 
            newstrsub = ss22


    else:
        m = re.findall(r'rsmad|complex|median|total|doubt', ss20, re.IGNORECASE)
        ss22 = str(m[0])
        ss2 = ss22
        newstrsub = 'R-Smad'
        if bool(re.findall(r'median|total', ss22, re.IGNORECASE)):
            newstrsub = 'NG-Smad3'
            ss2 = ss22 +' NG-Smad3' 
        elif bool(re.findall(r'complex|doubt', ss22, re.IGNORECASE)):
            ss2 = 'Smad ' + ss22
            newstrsub = 'Smad complex'

    newstr = ss2
    return newstr,newstrsub

def msefunc(xp,Y):
    mseeach={}
    Yscale = np.zeros(Y.shape)
    xpscale = np.zeros(Y.shape)
    for i in range(0,Y.shape[1]):
        Yscale[:,i] = (Y[:,i]-np.mean(Y[:,i]))/np.std(Y[:,i])
        xpscale[:,i] = (xp[:,i]-np.mean(xp[:,i]))/np.std(xp[:,i])
        mseeach[i]=np.mean((Yscale[:,i]-xpscale[:,i])**2)
    scalex = xpscale
    scaley = Yscale
    mse = np.mean((Yscale-xpscale)**2)
    return scalex,scaley, mse, mseeach



def GPymadness(X,Y,messages,max_f_eval,inputstr,predtstr,titlestr):
    kerndim = X.shape[1]
    ker = GPy.kern.Matern52(kerndim,ARD=True) + GPy.kern.White(kerndim)

    # create simple GP model
    m = GPy.models.GPRegression(X,Y,ker)
    # optimize and plot
    m.optimize(messages=messages,max_f_eval = max_f_eval)
    return m

def Gpy_mse_plotOLD(m,X,Y,inputstr,predtstr,titlestr):
    msize=20
    xp,xstd = m.predict(X)
    x1,y1, mse, mseeach = msefunc(xp,Y)
    msev = np.asarray(list(mseeach.values())).reshape(-1,)
    for i in range(0,x1.shape[1]):
        x = x1[:,i]
        y = y1[:,i]
        xy = np.hstack((x[:],y[:]))
        xyp = [np.min(xy[:]),np.max(xy[:])]
        plt.plot(xyp,xyp,'k-',linewidth=0.5)
        if x1.shape[1]==1:
            plt.scatter(x,y,c='k',marker='.',s = msize)
            plt.xlabel('original')
            plt.ylabel('predicted')  
        else:
            ax = fig.add_subplot(1,x1s,i+1)
            ax.scatter(x,y,c='k',marker='.',s = msize)
            plt.xlabel('original')
            plt.ylabel('predicted')                    
        if x1.shape[1]>1:
            plt.title(titlestr + ' dim-' + str(i+1))
        else:
            plt.title(titlestr)
        ax = plt.gca()
        mseval = np.format_float_scientific(msev[i], unique=False, precision=1,exp_digits=1)
        t = ax.text(0.01,0.99,'rescaled mse = ',transform=ax.transAxes, horizontalalignment='left',verticalalignment='top')
        t = ax.text(0.01,0.90,str(mseval),transform=ax.transAxes, horizontalalignment='left',verticalalignment='top')
    return mseeach



def predictSELF_BasedOnDMAP(time_data,ss10,ss20,X0,X2,sdat1,sdat2,ev_in,savepath,fontsize):
#             predSpecificBasedOnDMAP(time_data,ss10,ss20,X,X2,sdat1,sdat2)
    import scipy.signal as scifp
    import GPy

    ss1,ss1sub = catchImportantWords(ss10)
    #first make a plot with multiple panels of the dmaps trying to predict specific values

    suptitlestr = 'predict ' + ss1sub + ' trajectories using ' + ss1sub + ' manifold, '+ str(ev_in) + ', (' + ss1 + ')'

    tvec = time_data[1,:]
    basal = (np.where((tvec<=0) & (tvec>=-30)))[0]

    fc = np.zeros(sdat1.shape)
    ss = sdat1.copy()
    for k in range(sdat1.shape[0]): fc[k,:] = ss[k,:]/np.median(ss[k,basal])
    diff = np.zeros(sdat1.shape)
    ss = sdat1.copy()
    for k in range(sdat1.shape[0]): diff[k,:] = ss[k,:]-np.median(ss[k,basal])
    relrate = np.gradient(fc.copy(),tvec,axis=1)
    integral = np.cumsum(sdat1.copy(),axis=1)



    max_f_eval = 1000
    messages=False    
    inputstr = ss10 + ' DMAP'
    Yarray = [sdat1,fc,diff,relrate,integral,relrate,integral]
    titlestrarray = ['level, t=', 'foldchange, t=', 'difference, t=','rate/basal,t=','integral, t=', 'max(rate/basal)','max(integral)']
    tvalarray = np.zeros(len(Yarray))
#     tvalarray = [30, 30, 30, 0, 0]
#     print(tvalarray)
    fig = plt.figure(figsize=(3,2))
    tvec = time_data[1,:]
    for i in range(len(Yarray)):
        titlestr = titlestrarray[i]
#         fig.add_subplot(1,len(Yarray),i+1)
        if bool(re.search(r'max',titlestr)):
            tval = 0
        else:
            Y = Yarray[i]
            ym = np.median(Y.copy(),axis=0)
            ymm = np.max(ym[:])
            ym = (ym-np.min(ym))/(ymm-np.min(ym))
            peaks,_ =scifp.find_peaks(ym, height=0.8, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
            if not len(peaks)>0:
                tt = (np.where(ym>0.95))[0]
                tv = tt[0]
            else:
                tv = peaks[0]
            plt.plot(tvec,ym,'-',label=titlestrarray[i])
            plt.plot(tvec[tv],(ym[tv]),'ro')
            tval = np.floor(tvec[tv])
            
        tvalarray[i] = tval 
    ax = plt.gca()
    ax.legend(bbox_to_anchor=(1.2, 1))
    plt.show()


    nop = len(titlestrarray) #number of panels
    ws = 0.4
    fig = plt.figure(figsize=((1.6 + (ws*(nop-1)))*nop,2.5))
    msekeeper={}
    for i in range(len(Yarray)):
        print(i)
        tval = tvalarray[i]
        titlestr = titlestrarray[i]
        Y = Yarray[i]

        if not tval==0:
            titlestr = titlestrarray[i] +''+str(tval)+'min'
            predtstr = ss1sub + ', t=' +str(tval)+'min'
            peak = ((np.where(tvec<=tval))[0])[-1]
            Y=Y[:,peak]
        else:
            Y = np.max(Y,axis=1)
            predtstr = ss1sub

        Y = Y.reshape(-1,1)
        X=X0.copy()

        m = GPymadness(X,Y,messages,max_f_eval,inputstr,predtstr,titlestr)
        ax = fig.add_subplot(1,nop,i+1)
        mseeach = Gpy_mse_plot(m,X,Y,inputstr,predtstr,titlestr)
        msekeeper[predtstr]=mseeach

    plt.suptitle(t=suptitlestr,x=0.5,y=1.2,fontsize=fontsize*1.2,fontweight = 'bold')
    fig.subplots_adjust(wspace=ws)
    savestr = savepath+ ss10 + 'predictions.png'
    plt.savefig(savestr,bbox_inches='tight')
    fig.add_axes([0,0,1,1]).axis("off")
    plt.show()
    return msekeeper

def getCombos2(ev_in,i_start):
    from itertools import combinations
    ll1 = [list(combinations(ev_in,x+1)) for x in range(i_start,len(ev_in))]  

    newlist=list()
    for x in range(0,len(ll1)): 
        ll1sub = ll1[x]
        for i in range(0,len(ll1sub)):
            newlist.append(ll1sub[i])
    return newlist


def Gpy_mse_plot_OTHER(m,X,Y,inputstr,predtstr,titlestr,fig,nop,i_P):
#     fig = plt.figure(figsize=(8,3))
    msize=20
    xp,xstd = m.predict(X)
    x1,y1, mse, mseeach = msefunc(xp,Y)
    x1s = x1.shape[1]
    msev = np.asarray(list(mseeach.values())).reshape(-1,)
    for i in range(0,x1s):
        x = x1[:,i]
        y = y1[:,i]
        xy = np.hstack((x[:],y[:]))
        xyp = [np.min(xy[:]),np.max(xy[:])]
        xyp = [np.min(xy[:]),np.max(xy[:])]
        ax = fig.add_subplot(x1s,nop,i_P + nop*(i))
        plt.plot(xyp,xyp,'k-',linewidth=0.5)

#         if x1.shape[1]==1:
#             plt.scatter(x,y,c='k',marker='.',s = msize)
#             plt.xlabel('original dim=' + str(i+1))
#             plt.ylabel('predicted dim=' + str(i+1))  
#         else:
#             ax.scatter(x,y,c='k',marker='.',s = msize)
#             plt.xlabel('original dim=' + str(i+1))
#             plt.ylabel('predicted dim=' + str(i+1))                    
       
#         if x1.shape[1]>1:
#             titlestr2 = inputstr + ', dims=' + titlestr
#         else:
#             titlestr2 = titlestr +'when does this happen'
        xlabelstr='original'
        if i<x1s:
            xlabelstr=''
        tst1 = str([i+1])
        tst2 = titlestr
        
        if x1.shape[1]==1:
            plt.scatter(x,y,c='k',marker='.',s = msize)
            plt.xlabel(xlabelstr)
            plt.ylabel('predicted')  
        else:
            ax.scatter(x,y,c='k',marker='.',s = msize)
            plt.xlabel(xlabelstr)
            plt.ylabel('predicted')                    
       
        if x1.shape[1]>1:
            titlestr2 = 'pred. '+ predtstr + tst1 + ',  use ' + inputstr + tst2
        else:
            titlestr2 = titlestr +'when does this happen'
            
        titlestr2 = titlestr2.replace('],','],\n')
#         print(titlestr2,'for reals')
        ax.set_title(titlestr2,pad=10)
        ax = plt.gca()
        xgap = np.max(x)-np.min(x)
        ygap = np.max(y)-np.min(y)
        ax.set_xlim(np.min(x) - xgap*0.1,np.max(x) + xgap*0.1)
        ax.set_ylim(np.min(y)-ygap*0.1,np.max(y)+ygap*0.1)
        if msev[i]<0.01:
            mseval = np.format_float_scientific(msev[i], unique=False, precision=1,exp_digits=1)
        else:
            mseval = np.round(msev[i],2)
        t = ax.text(0.01,0.99,'rescaled mse = ',transform=ax.transAxes, horizontalalignment='left',verticalalignment='top')
        t = ax.text(0.01,0.90,str(mseval),transform=ax.transAxes, horizontalalignment='left',verticalalignment='top')
    return mseeach


def makeFigureWithDefinedSubplotAxes(axW,axH,gapW,gapH,numX,numY):
#     axW = 2 #axwidth
#     axH = 2 #axheight
#     gapW = 1.0 #inches
#     gapH = 0.4 #inches

    
#     #using these values (relative values) as defualt
#     left = 0.125 #values defined as defaults for subplots_adjust
#     right = 0.9
#     bottom = 0.1
#     top = 0.9

#     fW = fW*L +fW*R +...
#     fW-fW*L-fW*R = ...
#     fW*(1-L-R)=...
#     fW = (axW*nop + gapW*(nop-1))/(1-left-(1-right))
#     fH = (axH*xs1 + gapH*(xs1-1))/(1-(1-top)-bottom)  #figureHeight
    
    wL = 0.5 #leftval in inches
    wR = 0.5 #rightval in inches
    hB = 0.5 #bottomval in inches
    hT = 0.5 #topval inches
    
    fW = wL + wR + axW*numX + gapW*(numX-1)
    fH = hB + hT + axH*numY + gapH*(numY-1)
    
    left = wL/fW
    right = 1- (wR/fW)
    bottom = hB/fH
    top = 1-(hT/fH)
    wspace = gapW/axW
    hspace = gapH/axH
    

    return fW,fH,left,right,top,bottom,wspace,hspace




def predictOTHER_DMAP_BasedOnDMAP_hiddenvar(time_data,ss10,ss20,X0,X2,sdat1,sdat2,ev_in,ev_out,savepath,fontsize,hvarName,hvarvec):
#             predSpecificBasedOnDMAP(time_data,ss10,ss20,X,X2,sdat1,sdat2)
    import scipy.signal as scifp
    import GPy

    ss1,ss1sub = catchImportantWords(ss10)
    ss2,ss2sub = catchImportantWords(ss20)
    #first make a plot with multiple panels of the dmaps trying to predict specific values
    ss3 = 'experimental data'
    if bool(re.search('SYST',ss10)):
        ss3 = ss10
    elif bool(re.search('SYST',ss20)):
        ss3 = ss20
    elif bool(re.search('exp3',ss20)):
        ss3 = 'experiment= exp3'
    elif bool(re.search('exp4',ss20)):
        ss3 = 'experiment= exp4'
    elif bool(re.search('exp3',ss10)):
        ss3 = 'experiment= exp3'
    elif bool(re.search('exp4',ss10)):
        ss3 = 'experiment= exp4'
        
        
    evinstr = str(ev_in)
    evinstr = evinstr.replace(']', ','+hvarName+']')
    suptitlestr = 'predict ' + ss2 + ' DMAP, '+str(ev_out)+' using ' + ss1 + ' DMAP, '+ evinstr + ', (' + ss3 + ' + '+hvarName + ' )' 

    max_f_eval = 1000
    messages=False    
    inputstr = ss1sub + ' DMAP'
          
    ev_combo = [int(x) for x in range(len(ev_in)+1)]
    dmapcombos = getCombos2(ev_combo,len(ev_combo)-2) 
#     dmapcombos = getCombos2(ev_combo,0) 
    
    nop = len(dmapcombos) #number of panels across
    xs1 = X2.shape[1] #number of panels down
    

    axW = 2
    axH = 2
    gapW = 2.2
    gapH = 1
    fW,fH,left,right,top,bottom,wspace,hspace = makeFigureWithDefinedSubplotAxes(axW,axH,gapW,gapH,numX=nop,numY=xs1)


    fsize = (fW,fH)
    fig = plt.figure(figsize=fsize)
         
    msekeeper={}
    for i in range(nop):
        dmap_ev_in = dmapcombos[i]
        
        X1 = X0[:,dmap_ev_in]
#         X1 = X0[:,dmap_ev_in] 
        if X1.shape[1]>1:
            X = X1
        else:
            print(X1.shape)
            X = X1.reshape(-1,1)
        

        if X2.shape[1]>1:
            Y = X2
        else:
            print(X2.shape)
            Y = X2.reshape(-1,1)
        
        predtstr = ss2sub
        inputstr = ss1sub
#         titlestr = str(np.asarray(dmap_ev_in))
#         titlestr = str(list(dmap_ev_in))
        titlestr = str([x+1 for x in dmap_ev_in])
        titlestr = titlestr.replace(str(len(ev_combo)),hvarName)
#             titlestr = titlestr[0:-2] + ',' hvarName ']'
        m = GPymadness(X,Y,messages,max_f_eval,inputstr,predtstr,titlestr)
        mseeach = Gpy_mse_plot_OTHER(m,X,Y,inputstr,predtstr,titlestr,fig,nop,i+1)
        msekeeper[predtstr]=mseeach

    fig.suptitle(t=suptitlestr,x=0.5,y=1.05,fontsize=fontsize*1.2,fontweight = 'bold')
    fig.subplots_adjust(wspace=wspace,hspace=hspace,left=left,right=right,bottom=bottom,top=top)
    savestr = savepath+ ss10 + 'predictions of ' + ss20 +' '+ hvarName + '.png'
    plt.savefig(savestr,bbox_inches='tight')
    fig.add_axes([0,0,1,1]).axis("off")
    plt.show()
    return msekeeper


def predictOTHER_DMAP_BasedOnDMAP_hiddenvarSIMPLER(time_data,ss10,ss20,X0,X2,sdat1,sdat2,ev_in,ev_out,savepath,fontsize,hvarName,hvarvec):
#             predSpecificBasedOnDMAP(time_data,ss10,ss20,X,X2,sdat1,sdat2)
    import scipy.signal as scifp
    import GPy

    ss1,ss1sub = catchImportantWords(ss10)
    ss2,ss2sub = catchImportantWords(ss20)
    #first make a plot with multiple panels of the dmaps trying to predict specific values
    ss3 = 'experimental data'
    if bool(re.search('SYST',ss10)):
        ss3 = ss10
    elif bool(re.search('SYST',ss20)):
        ss3 = ss20
    elif bool(re.search('exp3',ss20)):
        ss3 = 'experiment= exp3'
    elif bool(re.search('exp4',ss20)):
        ss3 = 'experiment= exp4'
    elif bool(re.search('exp3',ss10)):
        ss3 = 'experiment= exp3'
    elif bool(re.search('exp4',ss10)):
        ss3 = 'experiment= exp4'
        
        
    evinstr = str(ev_in)
    evinstr = evinstr.replace(']', ','+hvarName[0] +']')
    suptitlestr = 'predict ' + ss2 + ' DMAP, '+str(ev_out)+' using ' + ss1 + ' DMAP, '+ evinstr + ', (' + ss3 + ' + '+hvarName[0] + ' )' 

    max_f_eval = 1000
    messages=False    
    inputstr = ss1sub + ' DMAP'
          
    ev_combo = [int(x) for x in range(len(ev_in))]
    dmapcombos = getCombos2(ev_combo,len(ev_combo)-2) 
#     dmapcombos = getCombos2(ev_combo,0) 
    
    dmap1 = ev_combo
    dmap2 = ev_combo.copy()
    dmap3 = ev_combo.copy()
    dmap2.append(len(ev_combo))
    dmap3.append(len(ev_combo)+1)
#     [(0, 1, 2),(0, 1, 2, 3),(0, 1, 2, 4)]
    dmapcombos = [dmap1, dmap2, dmap3]
    print(dmapcombos)
    nop = len(dmapcombos) #number of panels across
    xs1 = X2.shape[1] #number of panels down
    

    axW = 2
    axH = 2
    gapW = 2.2
    gapH = 1
    fW,fH,left,right,top,bottom,wspace,hspace = makeFigureWithDefinedSubplotAxes(axW,axH,gapW,gapH,numX=nop,numY=xs1)


    fsize = (fW,fH)
    fig = plt.figure(figsize=fsize)
         
    msekeeper={}
    for i in range(nop):
        dmap_ev_in = dmapcombos[i]
        
        X1 = X0[:,dmap_ev_in]
#         X1 = X0[:,dmap_ev_in] 
        if X1.shape[1]>1:
            X = X1
        else:
            print(X1.shape)
            X = X1.reshape(-1,1)
        

        if X2.shape[1]>1:
            Y = X2
        else:
            print(X2.shape)
            Y = X2.reshape(-1,1)
        
        predtstr = ss2sub
        inputstr = ss1sub
#         titlestr = str(np.asarray(dmap_ev_in))
#         titlestr = str(list(dmap_ev_in))
        titlestr = str([x+1 for x in dmap_ev_in])
        titlestr = titlestr.replace(str(len(ev_combo)+1),hvarName[0])
        titlestr = titlestr.replace(str(len(ev_combo)+2),hvarName[1])
#         titlestr = titlestr.replace('],','],\n')
#             titlestr = titlestr[0:-2] + ',' hvarName ']'
        m = GPymadness(X,Y,messages,max_f_eval,inputstr,predtstr,titlestr)
        mseeach = Gpy_mse_plot_OTHER(m,X,Y,inputstr,predtstr,titlestr,fig,nop,i+1)
        msekeeper[predtstr]=mseeach

    fig.suptitle(t=suptitlestr,x=0.5,y=1.05,fontsize=fontsize*1.2,fontweight = 'bold')
    fig.subplots_adjust(wspace=wspace,hspace=hspace,left=left,right=right,bottom=bottom,top=top)
    savestr = savepath+ ss10 + 'predictions of ' + ss20 +' '+ hvarName[0] + '.png'
    plt.savefig(savestr,bbox_inches='tight')
    fig.add_axes([0,0,1,1]).axis("off")
    plt.show()
    return msekeeper
