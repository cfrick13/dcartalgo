    def resampledata(data_in,time_in,gradcuts,samprates):
        from itertools import groupby
        # import scipy.ndimage.morphology.binary_dilation as dilate
        import scipy.ndimage as ndimage
#         data_in = ogdin.copy()
#         time_in = time_data.copy()
#         gradcuts = [25,50,75]
#         samprates = [60,30,15,5]
        tvec = time_in[1,:]

        pt = np.gradient(data_in.copy(),tvec,axis=1)
        pt = np.gradient(pt,tvec,axis=1)
        pt = np.abs(pt)
        pt = np.median(pt,axis=0)
        pt = (pt-np.min(pt))/(np.percentile(pt,95)-np.min(pt))
#         plt.figure(figsize=(6,3))
#         plt.semilogy(pt.T)
#         for gcc in gradcuts:
#             gc = np.percentile(pt,gcc)
#             print(gc)
#             plt.semilogy([0,len(pt)],[gc,gc],linewidth=2)
#         plt.ylim(1e-9,10)
#         plt.show()

        emptyvec = np.zeros(len(pt))
        for x in range(len(gradcuts)):
            xx = x+1
            gcc = gradcuts[x]
            gc = np.percentile(pt,gcc)
            idx = (pt>=gc)
#             plt.plot(idx,'^-')
            emptyvec[idx] =xx   
#             plt.plot(idx,'*-') 
#         plt.show()

#         plt.plot(emptyvec)
#         plt.show()


        uvec = np.unique(emptyvec)
        suvec =np.sort(uvec)
        asuvec = suvec[::-1]
        for x in np.unique(emptyvec):
#         for x in asuvec:
            idx = emptyvec==x
            islandvec = [list(g) for k, g in groupby(emptyvec)]
            while np.min([len(x) for x in islandvec])<4:    
                idx = ndimage.morphology.binary_dilation(idx,iterations=2)
                emptyvec[idx] = xx
                islandvec = [list(g) for k, g in groupby(idx)] 
                print([len(x) for x in islandvec])
            emptyvec[idx] = x 

#         plt.plot(emptyvec)
#         plt.show()

        islandvec = [list(g) for k, g in groupby(emptyvec)] #--> AAAA BBB CC D
        # for x in islandvec:
        tidx = 0
        tvecnew = list()
        for i, (j) in enumerate(islandvec):
        #     print(i,j)
            sr = int(j[0])
            ilen = len(j)
            srate = samprates[sr]

            treg = np.ceil((tvec[tidx+ilen-1]-tvec[tidx])/srate)
            tinterp1 = np.linspace(tvec[tidx],tvec[tidx+ilen-1],treg)
            print(len(tinterp1))
            tvecnew= tvecnew + list(tinterp1)
            tidx = tidx+ilen

        keep_in = data_in.copy()
        x = np.asarray(tvecnew)
        time_datanew = np.zeros((keep_in.shape[0],len(x)))
        obsdnew = np.zeros((keep_in.shape[0],len(x)))
        for k in range(0,data_in.shape[0]): 
            xp = time_in[k,:]
            fp = keep_in[k,:]
            yp = numpy.interp(x, xp, fp, left=None, right=None, period=None)
            obsdnew[k,:] = yp
            time_datanew[k,:] = x

        data_out = obsdnew
        time_out = time_datanew


        return data_out, time_out