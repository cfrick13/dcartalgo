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

def Gpy_mse_plot(m,X,Y,inputstr,predtstr,titlestr):
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

def predictSELF_BasedOnDMAP(time_data,ss10,X0,sdat1,ev_in,savepath,fontsize):
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
    axW = 2
    axH = 2
    gapW = 1
    gapH = 1
    
    fW,fH,left,right,top,bottom,wspace,hspace = makeFigureWithDefinedSubplotAxes(axW,axH,gapW,gapH,numX=nop,numY=1)


    fsize = (fW,fH)
    fig = plt.figure(figsize=fsize)
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
    fig.subplots_adjust(wspace=wspace,hspace=hspace,left=left,right=right,bottom=bottom,top=top)
    savestr = savepath+ ss10 + 'predictions.png'
    plt.savefig(savestr,bbox_inches='tight')
    fig.add_axes([0,0,1,1]).axis("off")
    plt.show()
    return msekeeper




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
