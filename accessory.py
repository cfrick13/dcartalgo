import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
import numpy as np

import matplotlib.tri as mtri

from scipy.integrate import odeint
from scipy.integrate import complex_ode
import scipy.spatial.distance
import scipy
import scipy.interpolate
import scipy.signal

import re

seedval=20



from dmap_sp_chrisNEW import *

import dmap_sp_chrisNEW
import importlib
importlib.reload(dmap_sp_chrisNEW)


# max = diameter of data set
# min = max(vector of distances to 10th nearest neighbor for all points). 
# d = ||x-y||
# delta = 10e-10
# keps(d) = exp(((-d)**2)/epsilon)

# (A)  kemax(Dmax) ≥ delta
# (B)  kemin(Dmin) ≥ delta

# (1) keps(d)≥detta
# (2) exp(((-d)**2)/epsilon) ≥ delta
# (3) ((-d)**2)/epsilon ≥ ln(delta)
# (4) (-d)**2)/ln(delta) ≥ epsilon

# epsilon_min ≤ (-d_min)**2)/ln(delta)
# epsilon_max ≤ (-d_max)**2)/ln(delta)

def epsminmax(datain):
    distMatrix = scipy.spatial.distance.pdist(datain,'euclidean')
    distMatrix = scipy.spatial.distance.squareform(distMatrix)
    dist_true = distMatrix.copy()
    delta = 1e-10
    # print(delta)
    dist_true[dist_true<delta]=np.nan

    sort_dist_true = np.sort(dist_true,axis=1)
    distbest = sort_dist_true[:,10]
    d_min = np.max(distbest)

    nansum = np.sum(sort_dist_true,axis=0)
    nan0 = np.where(np.isnan(nansum))
    nanny = nan0[0]
    maxdistidx = nanny[0]-1
    # print(maxdistidx)
    # print('nanny',nanny)
    # print(nansum)

    distmax = sort_dist_true[:,maxdistidx]
    d_max = np.max(distmax)

    epsilon_min = (-(d_min)**2)/np.log(delta)
    epsilon_max = (-(d_max)**2)/np.log(delta)

    print('eps_min=',epsilon_min)
    print('eps_max=',epsilon_max)
    return epsilon_min,epsilon_max



def funky(x, p1,p2,p3,p4):
    return p1*(x**3) + p2*(x**2) + p3*(x) + p4
#   return p1*np.cos(p2*x) + p2*np.sin(p1*x)

def iqr_outlier(x):
    q1 = np.percentile(x,25)
    q3 = np.percentile(x,75)
    iqr = q3-q1
    outidx = (x < (q1 - (1.5 * iqr))) | (x > (q3 + (1.5 * iqr)))
    return outidx

def residuals_of_poly4_fit(x0,y0):
    xidx = np.argsort(x0)
    xix = x0[xidx]
    yix = y0[xidx]
    poptx, pcovx = scipy.optimize.curve_fit(funky, xix, yix,p0=(1.0,0.2,0.1,0.1),maxfev=100000)
    px1 = poptx[0]
    px2 = poptx[1]
    px3 = poptx[2]
    px4 = poptx[3]
    fy = funky(xix,px1,px2,px3,px4)
    residuals1 = (yix - fy)

    xidx = np.argsort(y0)
    xiy = x0[xidx]
    yiy = y0[xidx]
    popty, pcovy = scipy.optimize.curve_fit(funky, yiy, xiy,p0=(1.0,0.2,0.1,0.1),maxfev=100000)
    py1 = popty[0]
    py2 = popty[1]
    py3 = popty[2]
    py4 = popty[3]
    fx = funky(yiy,py1,py2,py3,py4)
    residuals2 = (xiy - fx)

#     residx1 = iqr_outlier(residuals1**2)
#     residx2 = iqr_outlier(residuals2**2)
    
    residx1 = numpy.zeros(np.shape(residuals1), dtype=bool)
    residx2 = numpy.zeros(np.shape(residuals2), dtype=bool)
    
    if sum(residuals1[~residx1]**2)<sum(residuals2[~residx2]**2):
        fres = sum(residuals1[~residx1]**2)
        popt = poptx
        xout = xix
        yout = funky(xix,popt[0],popt[1],popt[2],popt[3])
        iout = 0
    else:
        fres = sum(residuals2[~residx2]**2)
        popt = popty
        xout = funky(yiy,popt[0],popt[1],popt[2],popt[3])
        yout = yiy
        iout = 1

    return fres,popt,xout,yout,iout

def rescaleevecs(evecs,evals,eps_eps):
    showev = evecs
    showevl = evals

#     showevl = (np.abs(showevl))**(np.median(eps_eps)/4)
    
    shave = showev.copy()
    for i in range(showevl.shape[0]):
        shave[:,i] = showevl[i]*showev[:,i]
        
    return shave

def autodparam(evecs,evals,eps_eps,inoutstr,stdcut,taurcut):
    print('working on '+inoutstr)
    PTS = 5
    w1 = evecs[:,0]

    showev = evecs
    showevl = evals
    showevl = (np.abs(showevl))**(np.median(eps_eps)/4)
    shave = showev.copy()
    for i in range(showevl.shape[0]):
        shave[:,i] = showevl[i]*showev[:,i]

#     yl1 = np.percentile(shave,0.1)
#     yl2 = np.percentile(shave,99.9)
    yl1 = np.percentile(evecs,0.1)
    yl2 = np.percentile(evecs,99.9)

    std = np.zeros(showevl.shape)
    
    # plot scatter of each dparam
    # determine standard deviation with outliers removed
    fig,ax = plt.subplots(PTS,PTS,figsize=(8,8),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    fig.suptitle(inoutstr+'_dparams', fontsize=16)
    for i in range(PTS):
        for k in range(PTS):
            eigenindex = i*PTS+k
            x = range(0,len(showev[:,0]))
            y = shave[:,eigenindex]
#             y = evecs[:,eigenindex]
            sc1 = ax[i,k].scatter(x,y, 5, w1, cmap='jet')
            outy = iqr_outlier(y)
#             std[eigenindex] = np.absolute(np.std(y[~outy]))  
            std[eigenindex] = np.absolute(np.std(y))  
            ax[i,k].set_title([i,k,eigenindex])
            ax[i,k].set_ylim((yl1 - np.abs(yl1*0.5)),(yl2 + yl2))

    stdcut0 = np.max(std)/stdcut
    std = std-np.min(std[1:-1])
    stdcop = std.copy()
    
    #plot standard deviation of each dparam coordinate
    fig = plt.figure()
    plt.plot(np.abs(std),'k.-')
    plt.xlabel('dparam')
    plt.ylabel('stdev')

    
    #make plot showing stdev vals for each dparam
    xx = np.linspace(0,len(std)-1,len(std))
    yy = np.ones([len(std),1])*stdcut0
    plt.plot(xx,yy)
    
    keeperstdidx = np.abs(std)>stdcut0
    boolidx = numpy.array(keeperstdidx, dtype=bool)
    boolidx[0]=False
    keeprange = np.linspace(0,len(keeperstdidx)-1,len(keeperstdidx))
    keeperstd = keeprange[boolidx]
    
    #######

    #now that we know that eval[1] and evec[1] have significant variability plot other things with respect
    #to them. If they are functions of [1] then they are not important

    
    # sort the chosen eigenvectors based on who has max STDdev
    stdabs = np.abs(stdcop)
    keepvec =keeperstd.astype(int)
    stdval = stdabs[keepvec]
    cvec0 = np.argsort(stdval)
    cvec = cvec0[::-1]

    
    keepint =keeperstd.astype(int)
    keepint = keepint[cvec]
    shavesub = shave[:,keepint]
    tauscale = np.percentile(shavesub,95)-np.percentile(shavesub,5)
    tauscale=1
    fig,ax = plt.subplots(len(keeperstd),len(keeperstd),figsize=(8,8),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    fig.suptitle(inoutstr+'_dparams', fontsize=16)
    
    tauraall = np.zeros([len(showevl),len(showevl)])
    for i in range(len(showevl)):
        for j in range(len(showevl)):
            eigy1=i
            eigy2=j
            x = showevl[eigy1]*showev[:,eigy1]
            y = showevl[eigy2]*showev[:,eigy2]

            outx = iqr_outlier(x)
            outy = iqr_outlier(y)
            outidx = outx|outy

            x0 = x[~outidx]
            y0 = y[~outidx]
            #             print(x0.shape,x.shape)
            xidx = np.argsort(x0)

#             fres,popt,xout,yout,iout = residuals_of_poly4_fit(x0,y0)
#             tauraall[i,j]=fres/np.sum([np.std(x0),np.std(y0)])
            
    #determine the correlation between different eigenvectors to determine unique dparams
    taura = np.zeros([len(keepint),len(keepint)])
    for i in range(len(keepint)):
        for j in range(len(keepint)):
            eigy1=keepint[i]
            eigy2=keepint[j]
#             x = showevl[eigy1]*showev[:,eigy1]
#             y = showevl[eigy2]*showev[:,eigy2]
            x = showev[:,eigy1]
            y = showev[:,eigy2]
            #using eigenvector without eigenvalue multiplication makes it easier to compare residuals values. 
        
#             tau, p_value = scipy.stats.kendalltau(x, y)
            
            outx = iqr_outlier(x)
            outy = iqr_outlier(y)
            outidx = outx|outy
 
            x0 = x[~outidx]
            y0 = y[~outidx]
#             print(x0.shape,x.shape)
            xidx = np.argsort(x0)
            
            fres,popt,xout,yout,iout = residuals_of_poly4_fit(x0,y0)
             
#             tauscale = np.sum(np.abs([np.std(x0),np.std(y0)]))   
            tauscale=1
            taur = np.round(fres,decimals=3)/tauscale
            taura[i,j]=np.absolute(taur)

            std = np.std(y)
#             print("i=",i,"j=",j)

            if len(keepint)>1:
                sc1 = ax[i,j].scatter(x,y, 5, w1, cmap='jet')
                ax[i,j].plot(xout,yout)
                ax[i,j].set_title([eigy1,eigy2, '%1.2f' % taur])
            else:
                sc1 = ax.scatter(x,y, 5, w1, cmap='jet')
                ax.plot(xout,yout)
                ax.set_title([eigy1,eigy2, '%1.2f' % taur])
            
            
#             if iout==0:
#                 ax[i,j].plot(xout,funky(xout,popt[0],popt[1],popt[2],popt[3]))
#             else:
#                 ax[i,j].plot(funky(yout,popt[0],popt[1],popt[2],popt[3]),yout)

                
           


    fig,ax = plt.subplots(1,1,sharey=True)
    plt.imshow( np.absolute(taura), cmap='viridis')
    plt.colorbar(extend='both')
#     plt.clim(0, 0.01);

#     tcut = np.max(tauraall[:])/taurcut
    tcut = taurcut

    fig,ax = plt.subplots(1,taura.shape[1],sharex=True,sharey=True,figsize=(12,3))
    for j in range(taura.shape[1]):
        xvec = np.linspace(0,taura.shape[1]-1,taura.shape[1])
        taurvec = taura[:,j]
        boolbool = np.ones(taurvec.shape,dtype=bool)
#         boolbool[j] = False
        yvec = np.ones(taurvec.shape)*(tcut)
        xv = xvec[boolbool]
        yv = taurvec[boolbool]
        yy = yvec[boolbool]
        ax[j].plot(xv[j+1:len(xv)],yv[j+1:len(xv)],'*-')
#         plt.ylim([0,cutoff*8])
        plt.xlim([0,len(boolbool)])
        ax[j].plot(xvec,yvec)
        ax[j].text(1, tcut*0.5, 'is a function of %g' % j)
#         ax[j].legend(xx[0:taura.shape[1]])



    #############

    print(tcut)
    removeme=[]
    for i in range(taura.shape[0]):
    #         if i not in removeme:
        tidx = taura[i,:]<(tcut) #find anything that is tightly correlated
        tidx[i] = False #set self correlation = false
        nvec = np.linspace(0,len(tidx)-1,len(tidx))
        if i>=i:
            tidx[0:i]=False
        nvec = np.linspace(0,len(tidx)-1,len(tidx))
        rm = nvec[tidx] 
        removeme = np.append(removeme,rm.astype(int))


      
    chuckz = np.unique(removeme.astype(int))
    tbidx = np.ones((keeperstd.shape), dtype=bool)
    tbidx[chuckz]=False
    keepdparams = keepint[tbidx]
    print('good '+inoutstr+' dparams are:', keepdparams.astype(int))
    return keepdparams


def plot_trajectories_based_on_dparam(tvec,traj,evecs_0,ev_0,strstr):
    #plot the Smad trajectories colored by info
    x = np.transpose(tvec)
    y = np.transpose(traj)
    xs = x[:,1]
    xs0 = numpy.where((xs<0)&(xs>-30))
    tv0 = xs0[0]
    
    #fold change
    fc = y.copy()
    for i in range(y.shape[1]):
        fc[:,i] = y[:,i]/np.median(y[tv0,i])
        
    #difference
    diff = y.copy()
    for i in range(y.shape[1]):
        diff[:,i] = y[:,i]-np.median(y[tv0,i])
        
    #gradient of fc
    grfc = np.gradient(fc.copy(),axis=0)


    namesj = ['abundance','foldchange','difference','rate']
    numberOfTransforms = 4;
    
    fig,ax = plt.subplots(numberOfTransforms,len(ev_0),figsize=(len(ev_0)*3,2*numberOfTransforms),sharey=False)
    fig.subplots_adjust(wspace=0.2,hspace=0.2)
    for j in range(numberOfTransforms):
        print(j)
        for i in range(len(ev_0)):
            lp = np.linspace(0,1,y.shape[1])
            ev_in_v = ev_0[i]
            hh = np.argsort(evecs_0[:,ev_in_v])
            cmvec = lp[hh]
            color=iter(plt.cm.viridis(lp))
            for k in range(y.shape[1]):
                xs = x[:,hh[k]]
                
                if j==0: #level
                    ys2 = y[:,hh[k]]
                elif j==1: #fold change
                    ys2 = fc[:,hh[k]]
                elif j==2: #difference
                    ys2 = diff[:,hh[k]]
                elif j==3:
                    ys2 = grfc[:,hh[k]]              
                else:
                    ys2 = y[:,hh[k]]

                c=next(color)
                if len(ev_0)==1:
                    ax[j].plot(ys2,c=c,linewidth=0.5)
                else:
                    ax[j,i].plot(ys2,c=c,linewidth=0.5)

            if len(ev_0)==1:
                ax[j].set_ylabel(namesj[j])
                ax[j].set_title(strstr+' dim %g' % (i+1))
            else:
                ax[j,i].set_ylabel(namesj[j])
                ax[j,i].set_title(strstr+' dim %g' % (i+1))
#             plt.show()
            
            
                    
def plot_trajectories_based_on_dparamOLD(tvec,traj,evecs_0,ev_0,strstr):
    #plot the Smad trajectories colored by info
    fig,ax = plt.subplots(2,len(ev_0),figsize=(len(ev_0)*3,3),sharey=False)

    x = np.transpose(tvec)
    y = np.transpose(traj)
    for j in range(2):
        for i in range(len(ev_0)):
#             print(ev_0)
            lp = np.linspace(0,1,y.shape[1])
            ev_in_v = ev_0[i]
            hh = np.argsort(evecs_0[:,ev_in_v])
            cmvec = lp[hh]
            color=iter(plt.cm.viridis(lp))
            for k in range(y.shape[1]):
                xs = x[:,hh[k]]
                xs0 = numpy.where(xs<0)
                tv0 = xs0[0]
                ys = y[:,hh[k]]

                if j==1:
                    ys2 = np.divide(ys,np.median(ys[tv0[:]]))
                else:
                    ys2 = ys

                c=next(color)
                if len(ev_0)==1:
                    ax[j].plot(ys2,c=c,linewidth=0.5)
                    ax[j].set_title(strstr+' dim %g' % (i+1))
                else:
                    ax[j,i].plot(ys2,c=c,linewidth=0.5)
                    ax[j,i].set_title(strstr+' dim %g' % (i+1))

                
def shaver(evecs,evals,eps_eps):
    showev = evecs
    showevl = evals
    showevl = (np.abs(showevl))**(np.median(eps_eps)/4)
    shave = showev.copy()
    for i in range(showevl.shape[0]):
        shave[:,i] = showevl[i]*showev[:,i]
    return shave

def shaverscale(evecs,evals,eps_eps):
    showev = evecs
    showevl = evals
    showevl = np.reciprocal(evals)
    shave = showev.copy()
    for i in range(showevl.shape[0]):
        shave[:,i] = showevl[i]*showev[:,i]
    return shave


def autodparam2(evecs,evals,eps_eps,inoutstr):
    print('working on '+inoutstr)
    PTS = 5
    w1 = evecs[:,0]

    showev = evecs
    showevl = evals

    showevl = (np.abs(showevl))**(np.median(eps_eps)/4)
    
    shave = showev.copy()
    for i in range(showevl.shape[0]):
        shave[:,i] = showevl[i]*showev[:,i]

    yl1 = np.percentile(shave,0.1)
    yl2 = np.percentile(shave,99.9)

    std = np.zeros(showevl.shape)
    
    # plot scatter of each dparam
    fig,ax = plt.subplots(PTS,PTS,figsize=(8,8),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    fig.suptitle(inoutstr+'_dparams', fontsize=16)
    for i in range(PTS):
        for k in range(PTS):
            eigenindex = i*PTS+k
            x = range(0,len(showev[:,1]))
            y = showevl[eigenindex]*showev[:,eigenindex]
            sc1 = ax[i,k].scatter(x,y, 5, w1, cmap='jet')
            outy = iqr_outlier(y)
            std[eigenindex] = np.absolute(np.std(y[~outy]))  
            ax[i,k].set_title([i,k,eigenindex])
            plt.ylim((yl1 - np.abs(yl1*0.5)),(yl2 + yl2))

    stdcut0 = np.max(std)/stdcut
    std = std-np.min(std[1:-1])
    stdcop = std.copy()
    #plot standard deviation of each dparam coordinate
    fig = plt.figure()
    plt.plot(np.abs(std),'k.-')
    plt.xlabel('dparam')
    plt.ylabel('stdev')

    
    xx = np.linspace(0,len(std)-1,len(std))
    yy = np.ones([len(std),1])*stdcut0
    plt.plot(xx,yy)
    keeperstdidx = np.abs(std)>stdcut0
    boolidx = numpy.array(keeperstdidx, dtype=bool)
    boolidx[0]=False
    keeprange = np.linspace(0,len(keeperstdidx)-1,len(keeperstdidx))
    keeperstd = keeprange[boolidx]
    keepvec =keeperstd.astype(int)
    return keepvec
    
    print('done')



def accbar1(k,datatest,evecs_test,smadsnail):
    from itertools import combinations
    listcomb = [[x,x] for x in range(evecs_test.shape[1])]
    shave_i = evecs_test
    testvec = [0,1]

    kout_in1i, koutDPrnd_in1i,koutDPbest_in1i =  accuracytester7(listcomb,shave_i,k,'on',testvec,smadsnail,seedval,datatest)

        
    listcomb1 = listcomb
    return listcomb1, kout_in1i, koutDPrnd_in1i, koutDPbest_in1i



def accbar2(k,datatest,evecs_test,smadsnail):
    from itertools import combinations
#     listcomb = [[x,x] for x in range(evecs_test.shape[1])]
    comb = combinations(range(evecs_test.shape[1]),2)
    listcomb = list(comb)
    shave_i = evecs_test
    testvec = [0,1]
    print(len(listcomb))
    kout_in1i, koutDPrnd_in1i,koutDPbest_in1i =  accuracytester7(listcomb,shave_i,k,'on',testvec,smadsnail,seedval,datatest)
        
    listcomb1 = listcomb
    return listcomb1, kout_in1i, koutDPrnd_in1i, koutDPbest_in1i

def accbar3(k,datatest,evecs_test,smadsnail):
    from itertools import combinations
    comb = combinations(range(evecs_test.shape[1]),3)
    listcomb = list(comb)
    shave_i = evecs_test
    testvec = [0,1,2]
    print(len(listcomb))
    kout_in1i, koutDPrnd_in1i,koutDPbest_in1i =  accuracytester7(listcomb,shave_i,k,'on',testvec,smadsnail,seedval,datatest)
        
    listcomb1 = listcomb
    return listcomb1, kout_in1i, koutDPrnd_in1i, koutDPbest_in1i

def accbar1_2(k,goodin,datatest,evecs_test,smadsnail):
    from itertools import combinations
    comb = combinations(range(evecs_test.shape[1]),1)
    listcomb = list(comb)
    for i in range(len(listcomb)):
        tmat = listcomb[i]
        t1 = goodin[0]
        t2 = int(tmat[0])
        ev_i = [t1,t2]
        listcomb[i] = ev_i

    shave_i = evecs_test
    testvec = [0,1,2]
#     print(len(listcomb))
    kout_in1i, koutDPrnd_in1i,koutDPbest_in1i =  accuracytester7(listcomb,shave_i,k,'on',testvec,smadsnail,seedval,datatest)
        
    listcomb1 = listcomb
    return listcomb1, kout_in1i, koutDPrnd_in1i, koutDPbest_in1i

def accbar12_3(k,goodin,datatest,evecs_test,smadsnail):
    from itertools import combinations
    comb = combinations(range(evecs_test.shape[1]),1)
    listcomb = list(comb)
    for i in range(len(listcomb)):
        tmat = listcomb[i]
        t1 = int(goodin[0])
        t2 = int(goodin[1])
        t3 = int(tmat[0])
        ev_i = [t1,t2,t3]
        listcomb[i] = ev_i

    shave_i = evecs_test
    testvec = [0,1,2]
#     print(len(listcomb))
    kout_in1i, koutDPrnd_in1i,koutDPbest_in1i =  accuracytester7(listcomb,shave_i,k,'on',testvec,smadsnail,seedval,datatest)
        
    listcomb1 = listcomb
    return listcomb1, kout_in1i, koutDPrnd_in1i, koutDPbest_in1i

def accbar123_4(k,goodin,datatest,evecs_test,smadsnail):
    from itertools import combinations
    comb = combinations(range(evecs_test.shape[1]),1)
    listcomb = list(comb)
    for i in range(len(listcomb)):
        tmat = listcomb[i]
        t1 = int(goodin[0])
        t2 = int(goodin[1])
        t3 = int(goodin[2])
        t4 = int(tmat[0])
        ev_i = [t1,t2,t3,t4]
        listcomb[i] = ev_i

    shave_i = evecs_test
    testvec = [0,1,2,3]
#     print(len(listcomb))
    kout_in1i, koutDPrnd_in1i,koutDPbest_in1i =  accuracytester7(listcomb,shave_i,k,'on',testvec,smadsnail,seedval,datatest)
        
    listcomb1 = listcomb
    return listcomb1, kout_in1i, koutDPrnd_in1i, koutDPbest_in1i

def accbar_multiDEF(k,goodin,datatest,evecs_test,smadsnail):

    ev_i = [int(x) for x in goodin]
    listcomb = [ev_i]

    shave_i = evecs_test
    testvec = [int(x) for x in range(0,len(goodin))]
    kout_in1i, koutDPrnd_in1i,koutDPbest_in1i =  accuracytester7(listcomb,shave_i,k,'on',testvec,smadsnail,seedval,datatest)
        
    listcomb1 = listcomb
    return listcomb1, kout_in1i, koutDPrnd_in1i, koutDPbest_in1i


def accbar3_21(k,datatest,evecs_test,smadsnail):
    from itertools import combinations
    comb = combinations(range(evecs_test.shape[1]),2)
    listcomb = list(comb)
    for i in range(len(listcomb)):
        tmat = listcomb[i]
        t1 = int(1)
        t2 = int(tmat[0])
        t3 = int(tmat[1])
        ev_i = [t1,t2,t3]
        listcomb[i] = ev_i

    shave_i = evecs_test
    testvec = [0,1,2]
    print(len(listcomb))
    kout_in1i, koutDPrnd_in1i,koutDPbest_in1i =  accuracytester7(listcomb,shave_i,k,'on',testvec,smadsnail,seedval,datatest)
        
    listcomb1 = listcomb
    return listcomb1, kout_in1i, koutDPrnd_in1i, koutDPbest_in1i
def accbar3_21(k,datatest,evecs_test,smadsnail):
    from itertools import combinations
    comb = combinations(range(evecs_test.shape[1]),2)
    listcomb = list(comb)
    for i in range(len(listcomb)):
        tmat = listcomb[i]
        t1 = int(1)
        t2 = int(tmat[0])
        t3 = int(tmat[1])
        ev_i = [t1,t2,t3]
        listcomb[i] = ev_i

    shave_i = evecs_test
    testvec = [0,1,2]
    print(len(listcomb))
    kout_in1i, koutDPrnd_in1i,koutDPbest_in1i =  accuracytester7(listcomb,shave_i,k,'on',testvec,smadsnail,seedval,datatest)
        
    listcomb1 = listcomb
    return listcomb1, kout_in1i, koutDPrnd_in1i, koutDPbest_in1i


def accbar1plot(kout_in1i,koutDPbest_in1i,koutDPrnd_in1i,listcomb1,smadsnail):
    #determine accuracy of DParams (1 param)
    ratiovec = np.reciprocal(kout_in1i)/np.reciprocal(koutDPbest_in1i)
    # plt.plot(ratiovec)

    newarg = np.argsort(ratiovec)
    ratiosort = ratiovec[newarg]

    a = np.argmax(ratiovec)
    newin = -1

    vv = listcomb1[newarg[newin]]

    top5 = np.asarray([str(listcomb1[newarg[newin]]) for newin in [-5,-4,-3,-2,-1]])
    bot5 = np.asarray([str(listcomb1[newarg[newin]]) for newin in [0,1,2,3,4]])
    top5val = [ratiosort[newin] for newin in [-5,-4,-3,-2,-1]]
    bot5val = [ratiosort[newin] for newin in [0,1,2,3,4]]

    xticks = np.hstack([bot5,top5])

    fig = plt.figure(figsize=(3,3))
    plt.bar(range(10),np.hstack([bot5val,top5val]))

    fsize=12
    plt.title('Accuracy: '+smadsnail+ ' combinations',fontsize=fsize)
    plt.xticks(range(10),xticks,fontsize=fsize,rotation=90)
    plt.ylabel('accuracy',fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.ylim(0,1)
    plt.show()
    print(top5val[-1])
    return top5val[-1],top5[-1]

    
def accbar2plot(kout_in2i,koutDPbest_in2i,koutDPrnd_in2i,listcomb2,smadsnail):
    #determine accuracy of DParams (2 param)
    ratiovec = np.reciprocal(kout_in2i)/np.reciprocal(koutDPbest_in2i)
    # plt.plot(ratiovec)


    newarg = np.argsort(ratiovec)
    ratiosort = ratiovec[newarg]

    a = np.argmax(ratiovec)


    newin = -1

    vv = listcomb2[newarg[newin]]

    top5 = np.asarray([str(listcomb2[newarg[newin]]) for newin in [-5,-4,-3,-2,-1]])
    bot5 = np.asarray([str(listcomb2[newarg[newin]]) for newin in [0,1,2,3,4]])
    top5val = [ratiosort[newin] for newin in [-5,-4,-3,-2,-1]]
    bot5val = [ratiosort[newin] for newin in [0,1,2,3,4]]

    xticks = np.hstack([bot5,top5])

    fig = plt.figure(figsize=(3,3))
    plt.bar(range(10),np.hstack([bot5val,top5val]))

    fsize=12
    plt.title('Accuracy: '+smadsnail+ ' combinations',fontsize=fsize)
    plt.xticks(range(10),xticks,fontsize=fsize,rotation=90)
    plt.ylabel('accuracy',fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.ylim(0,1)
    plt.show()
    print(top5val[-1])
    return top5val[-1],top5[-1]

def accbar1get(kout_in1i,koutDPbest_in1i,koutDPrnd_in1i,listcomb1,smadsnail):
    #determine accuracy of DParams (1 param)
    ratiovec = np.reciprocal(kout_in1i)/np.reciprocal(koutDPbest_in1i)
    # plt.plot(ratiovec)

    newarg = np.argsort(ratiovec)
    ratiosort = ratiovec[newarg]

    a = np.argmax(ratiovec)
    newin = -1

    vv = listcomb1[newarg[newin]]

    top5 = np.asarray([str(listcomb1[newarg[newin]]) for newin in [-2,-1]])
    bot5 = np.asarray([str(listcomb1[newarg[newin]]) for newin in [0,1]])
    top5val = [ratiosort[newin] for newin in [-2,-1]]
    bot5val = [ratiosort[newin] for newin in [0,1]]

    xticks = np.hstack([bot5,top5])


    return top5val[-1],listcomb1[newarg[-1]]

def accbar2get(kout_in2i,koutDPbest_in2i,koutDPrnd_in2i,listcomb2,smadsnail):
    #determine accuracy of DParams (2 param)
    ratiovec = np.reciprocal(kout_in2i)/np.reciprocal(koutDPbest_in2i)
    # plt.plot(ratiovec)


    newarg = np.argsort(ratiovec)
    ratiosort = ratiovec[newarg]

    a = np.argmax(ratiovec)


    newin = -1

    vv = listcomb2[newarg[newin]]

    top5 = np.asarray([str(listcomb2[newarg[newin]]) for newin in [-2,-1]])
    bot5 = np.asarray([str(listcomb2[newarg[newin]]) for newin in [0,1]])
    top5val = [ratiosort[newin] for newin in [-2,-1]]
    bot5val = [ratiosort[newin] for newin in [0,1]]

    xticks = np.hstack([bot5,top5])
    top5array = listcomb2[newarg[-1]]

    return top5val[-1],listcomb2[newarg[-1]]

def accbar_multiGET(kout_in2i,koutDPbest_in2i,koutDPrnd_in2i,listcomb2,smadsnail):
    #determine accuracy of DParams (2 param)
    ratiovec = np.reciprocal(kout_in2i)/np.reciprocal(koutDPbest_in2i)
    # plt.plot(ratiovec)


    newarg = np.argsort(ratiovec)
    ratiosort = ratiovec[newarg]

    return ratiosort[-1]


### quantify the accuracy of diffusion map coordinates (knn) relative to traces at random
#validate accuracy (test2), using evecs only in 3d
def accuracytester7(listcomb,evecs_in,k,ploton,testvec,smadsnail,seedval,datatrajectories):

    
#     distmetric = 'mahalanobis'
    distmetric = 'euclidean'
    from time import time
    lt=time()
    data = datatrajectories
    dvec = data[1,:]

#     import sklearn.metrics as sm
    
    numpy.random.seed(seed=None)
    distMatrix = scipy.spatial.distance.pdist(data, distmetric)
    distMatrix = scipy.spatial.distance.squareform(distMatrix)
    dist_true = distMatrix.copy()
#     dist_true = sm.pairwise_distances(data,metric = distmetric)
    sort_dist_true = np.sort(dist_true,axis=1)
    arg_dist_true = np.argsort(dist_true,axis=1)

    closestbest = arg_dist_true[:,1:k+1]
    distbest = sort_dist_true[:,1:k+1]
    nbsbest = closestbest[0]


    kout_in1i = np.zeros(len(listcomb))
    koutDPrnd_in1i = np.zeros(len(listcomb))
    koutDPbest_in1i = np.zeros(len(listcomb))
    for i in range(len(listcomb)):
        tmat = listcomb[i]    
        xyz=evecs_in[:,tmat]

#         dist = sm.pairwise_distances(xyz,metric = distmetric)
#         distmetric = 'euclidean'
        distMatrix = scipy.spatial.distance.pdist(xyz, distmetric)
        distMatrix = scipy.spatial.distance.squareform(distMatrix)
        dist = distMatrix.copy()
        arg_dist = np.argsort(dist)
        closest = arg_dist[:,1:k+1]
        
    
        kvecsum = np.zeros(data.shape[0])
        kvecsumrnd = np.zeros(data.shape[0])
        kvecsumbest = np.zeros(data.shape[0])
        numpy.random.seed(seed=None)
        for j in range(data.shape[0]):
            #find knn in 49, compute distance in 49 dim space
            #find knn in DMap, compute distance in 49 dim space

            cp1 = j
            dmap_knn = closest[cp1,0:k]
            ndim_knn = closestbest[cp1,0:k]
            rand_knn = np.random.randint(0,data.shape[0],k);

            dmap_dist = dist_true[cp1,dmap_knn]
            ndim_dist = dist_true[cp1,ndim_knn]
            rand_dist = dist_true[cp1,rand_knn]

            kvecsum[j] = np.sum(dmap_dist)
            kvecsumrnd[j] = np.sum(rand_dist)
            kvecsumbest[j] = np.sum(ndim_dist)

        kout_in1i[i] = np.sum(kvecsum)
        koutDPrnd_in1i[i] = np.sum(kvecsumrnd)
        koutDPbest_in1i[i] = np.sum(kvecsumbest)
        
        if((time()-lt)>5):
            lt = time()
#             print(np.round((i/len(listcomb))*100))
#             print(np.round((i/len(listcomb))*100), end='\r')

#     print('done')
        
    return kout_in1i, koutDPrnd_in1i, koutDPbest_in1i