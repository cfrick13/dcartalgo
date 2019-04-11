def analysis_parameters(uniqueid='agnostic-AGI-zto_numeps'):

    #epsilon search parameters
    num_eps = 50      #number of epsilon values to screen
    eL = 1.5          #divide lower estimated min epsilon by this factor
    eH = 4            #multiply upper estimated max epsilon by this factor

    evecs_cut = True  #exclude eigenfunctions with low eigenvalues
    num_evecs=30      #number of eigen vectors to compute using DMAPS

    sampiter = 10     #number of times to iterate through Local linear regression
    subsampfactor = 10 #scale at which to subsample data for LLR # subsampsize = np.ceil(datasize/subsampfactor)
    subsampmax = 500  # never run LLR for more than 500 data pts; it will fail

    reducedimval = 50 #max dimensionality of data that DMAPs will be used to predict 
    promval = 0.01    #defines prominence of peaks at which to threshold for Local Linear Regression analysis of residuals

    specid = uniqueid+str(num_eps)+'-'+str(eL)+'-'+str(eH)+'_rdimval'+str(reducedimval)+'_n-evecs'+str(num_evecs)+'cut-'+str(evecs_cut)+'_promval-'+str(promval)+'_SI-'+str(sampiter)+'_subsampF'+str(subsampfactor)+'_sampmax'+str(subsampmax)  

    return num_eps,eL,eH,evecs_cut,num_evecs,sampiter,subsampfactor,subsampmax,reducedimval,promval,specid


def runsToRun():
    import itertools
    a = ['exp3','exp4','exp5']
    b = ['SMAD-median','SMAD-total']
    c = ['SNAIL-median','SNAIL-total']
    bc = b+c
    d1 = list(itertools.product(a, bc))

#     a = ['expSYST1_PC','expSYST2_PC','expSYST3_PC']
    a1 = ['expSYST1_PC']
    b1 = ['SMAD-rsmad','SMAD-complex']
    a2 = ['expSYST1_PC','expSYST2_PC','expSYST5_PC']
#     b2 = ['GENE-iffl']
    b2 = ['GENE-iffl','GENE-caga','RFP-iffl','RFP-caga']
    dab1 = list(itertools.product(a1,b1))
    dab2 = list(itertools.product(a2,b2))
    d2 = dab1+dab2

    

    # dall = d1+d2
    dall = d2+d1
    print(len(dall), ' runs to perform')
    return dall



def catchImportantWords(ss20):
    import re
    m = re.findall(r'SMAD|SNAIL|GENE|RFP|doubt', ss20, re.IGNORECASE)
    ss21 = m[0]

    if bool(re.search('GENE|RFP|SNAIL',ss21,re.IGNORECASE)):
        m = re.findall(r'caga|iffl|median|total|doubt', ss20, re.IGNORECASE)
        ss22 = str(m[0])
        if re.search('RFP|GENE',ss21,re.IGNORECASE):
            ss22 = ss21+ss22
        
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
        elif bool(re.findall(r'SYST5|doubt', ss20, re.IGNORECASE)):
            ss2 = ss22 + ', f(Smad complex, X and Y)' 
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


def runsToRun_old():
    import itertools
    a = ['exp5']
    b = ['median','total']
    c = ['median','total']
    d1 = list(itertools.product(a, b, c))

    # # a = ['expSYST1_PC','expSYST2_PC','expSYST3_PC']
    a = ['expSYST1_PC','expSYST2_PC']
    # # a = ['expSYST3_PC']
    b = ['complex']
    c = ['iffl','caga']
    d2 = list(itertools.product(a, b, c))


    # dall = d1+d2
    dall = d2
    print(len(dall), ' runs to perform')
    return dall