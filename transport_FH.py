'''
Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
'''

import numpy as np
import numpy.linalg as lin
import scipy.sparse as sps
import scipy.sparse.linalg as slin
import scipy.linalg as scilin
import scipy.optimize as opt
import scipy.fftpack as sft
import scipy.integrate as sint
import scipy.special as spc
import datetime


'''
Below are the Pauli matrices, to be used in a Jordan-Wigner encoding of the fermionic operators.
'''

matX = sps.csr_matrix(np.array([[0.,1.],[1.,0.]],dtype=np.complex128))
matY = sps.csr_matrix(np.array([[0.,-1.j],[1.j,0.]],dtype=np.complex128))
matZ = sps.csr_matrix(np.array([[1.,0.],[0.,-1.]],dtype=np.complex128))
matPl = 0.5*(matX+1.j*matY)
matMi = 0.5*(matX-1.j*matY)
matId = sps.csr_matrix(np.array([[1.,0.],[0.,1.]],dtype=np.complex128))

def gmatCreation(nthSite,nSites,zVerbose=0):
    '''
    This function generates the matrix for a creation operator on the `nthSite` of `nSites`
    fermionic modes. Note that the encoding here doesn't keep track of spin indices - 
    that has to be done by either higher level code or bookkeeping on the part of the coder.
    The creation operator is encoded by a Jordan-Wigner transform:
    \begin{equation}
    \hat{c}_ j^\dagger=\left(\prod_{k=1}^{j-1}\hat{\sigma}_ k^{(z)}\right)\hat{\sigma}_ j^{(+)}
    \end{equation}
    '''
    
    matCr = sps.csr_matrix(np.array([[1.]],dtype=np.complex128))
    for jIdx in np.arange(nSites):
        if jIdx<nthSite:
            matCr = sps.kron(matCr,matZ,format='csr')
        elif jIdx==nthSite:
            matCr = sps.kron(matCr,matPl,format='csr')
        else:
            matCr = sps.kron(matCr,matId,format='csr')
    return matCr


def gmatAnnihilation(nthSite,nSites,zVerbose=0):
    '''
    This function generates the matrix for a annihilation operator on the `nthSite` of `nSites`
    fermionic modes. Note that the encoding here doesn't keep track of spin indices - 
    that has to be done by either higher level code or bookkeeping on the part of the coder.
    The annihilation operator is encoded by a Jordan-Wigner transform:
    \begin{equation}
    \hat{c}_ j=\left(\prod_{k=1}^{j-1}\hat{\sigma}_ k^{(z)}\right)\hat{\sigma}_ j^{(-)}
    \end{equation}
    '''
    
    matAn = np.array([[1.]],dtype=np.complex128)
    for jIdx in np.arange(nSites):
        if jIdx<nthSite:
            matAn = sps.kron(matAn,matZ,format='csr')
        elif jIdx==nthSite:
            matAn = sps.kron(matAn,matMi,format='csr')
        else:
            matAn = sps.kron(matAn,matId,format='csr')
    return matAn


def gmatNumber(nthSite,nSites,zVerbose=0):
    '''
    This function generates the matrix for a number operator on the `nthSite` of `nSites`
    fermionic modes. Note that the encoding here doesn't keep track of spin indices - 
    that has to be done by either higher level code or bookkeeping on the part of the coder.
    The number operator is defined directly as the product of the creation and annihilation
    operators as defined above.
    '''
    
    matN = gmatCreation(nthSite,nSites).dot(gmatAnnihilation(nthSite,nSites))
    return matN


def fFullFourierTransform(raXs,raTs):
    '''
    This code takes Scipy's FFT pack to return the Fourier transform of `raXs` being a
    function of `raTs`, with Fourier variable `ftTs` and the output function being `ftXs`.
    The specific Fourier transform being performed is a non-unitary one,
    \begin{equation}
    \tilde{x}(\omega)=\int_\mathbb{R}e^{-i\omega t}dtx(t),
    \end{equation}
    so that the spectral functions we get with this will satisfy sum rules. This means that the
    inverse Fourier transform, if needed, would include a factor of $1/2\pi$.
    '''
    
    ftX1 = sft.fft(raXs)
    ftXs = 0.5*(np.amax(raTs)-np.amin(raTs))*sft.fftshift(ftX1)/len(raXs)
    ftT1 = sft.fftfreq(len(raXs),raTs[1]-raTs[0])
    ftTs = sft.fftshift(ftT1)
    return ftXs,-ftTs

def fFullFourierTransform_unitary(raXs,raTs):
    '''
    This code takes Scipy's FFT pack to return the Fourier transform of `raXs` being a
    function of `raTs`, with Fourier variable `ftTs` and the output function being `ftXs`.
    The specific Fourier transform being performed is the *unitary* one this time,
    \begin{equation}
    \tilde{x}(\omega)=\int_\mathbb{R}e^{2\pi i\omega t}dtx(t),
    \end{equation}
    for preserved normalization of functions.
    '''
    
    ftX1 = sft.fft(raXs)
    ftXs = 0.5*(np.amax(raTs)-np.amin(raTs))*sft.fftshift(ftX1)/len(raXs)
    ftT1 = sft.fftfreq(len(raXs),raTs[1]-raTs[0])
    ftTs = sft.fftshift(ftT1)
    return ftXs*np.sqrt(2./np.pi),-ftTs*2.*np.pi
    
def gvecTimeEvolved(vec,matUdt_approx,tTol=1.0e-10,nRestart=25):
    '''
    This function uses the aforementioned Crank-Nicolson method for time propagation of an input vector `vec`
    under time evolution `matUdt_approx`.
    '''
    
    vecOut = slin.gmres(matUdt_approx.T.conj(),matUdt_approx.dot(vec),x0=vec,restart=nRestart,maxiter=100,\
                        M=matUdt_approx,tol=tTol)[0]
    return vecOut


def gGroundSystem(matH):
    '''
    This function is just a shorthand for getting the ground state `vecGS` and energy `rEg`
    of an input Hamiltonian `matH`.
    '''
    
    vecGuess = np.random.rand(matH.shape[0],1)
    rEg,vecGS = slin.lobpcg(matH,vecGuess,maxiter=100,largest=False)
    return vecGS.T[0],rEg[0]

def gtenGreensOperatorsTtoF(tenGRts,raTimes,rTol):
    '''
    This function takes the time-domain calculations of the Green's functions, `tenGRts` at times
    `raTimes`, and outputs the frequency domain Green's functions. In order to get something easier
    to work with, these Green's functions are "windowed", i.e. multiplied by a factor of $e^{-\eta t}$
    for some small parameter $\eta$ (decided by `rTol`), in order to broaden the peaks of delta
    functions for computational ease.
    '''
    
    # Here's where $eta$ is decided - we take the last point in the grid of time points and impose
    # that all the Green's functions are no larger than `rTol` in magnitude at the end of this time.
    # The helps broaden peaks as well as avoid spurious high frequencies from being included in the
    # Fourier transform.
    rEta = -np.log(rTol)/(np.amax(raTimes)-np.amin(raTimes))
    raWindow = np.exp(-rEta*raTimes)
    
    # Here we allocate space for the frequency domain Green's functions and loop through all of them
    # to do the Fourier transforms.
    tenGRw = np.zeros(tenGRts.shape,dtype=np.complex128)
    for jIdx in np.arange(tenGRts.shape[0]):
        for kIdx in np.arange(tenGRts.shape[0]):
            ftGR,ftTR = fFullFourierTransform(tenGRts[jIdx,kIdx,:]*raWindow,raTimes)
            tenGRw[jIdx,kIdx,:] = ftGR
            
    # And output!
    return tenGRw,ftTR

def fFermi(rOmega,rBeta,rMu):
    '''
    This calculates the Fermi-Dirac distribution at energy `rOmega`, temperature `rBeta`,
    and chemical potential `rMu`:
    \begin{equation}
    f(\omega)=\frac{1}{1+e^{\beta(\omega-\mu)}}.
    \end{equation}
    '''
    
    return 1./(1.+np.exp(rBeta*(rOmega-rMu)))

def gmatTotalNumber(nSites):
    '''
    This generates the total number operator for a system of `nSites`, assuming that
    each individual site has both a spin up and spin down orbital.
    '''
    
    matNtotal = gmatNumber(0,nSites*2)+gmatNumber(1,nSites*2)
    for jIdx in np.arange(1,nSites):
        matNtotal += gmatNumber(2*jIdx,nSites*2)+gmatNumber(2*jIdx+1,nSites*2)
    return matNtotal

def gmatCharge(nthSite,nSites):
    '''
    This generates an operator that counts the number of electrons on the `nthSite` of
    `nSites`, assuming that each individual site has both a spin up and spin down orbital.
    '''
    
    matOut = gmatNumber(2*nthSite,2*nSites)+gmatNumber(2*nthSite+1,2*nSites)
    return matOut

def gmatSpin(nthSite,nSites):
    '''
    This generates an operator that finds the average spin on. the `nthSite` of `nSites`.
    '''
    
    matOut = 0.5*(gmatNumber(2*nthSite,2*nSites)-gmatNumber(2*nthSite+1,2*nSites))
    return matOut

def fMaximallyMixedExpectation(vecs,matA):
    '''
    This takes the set of states in `vecs` and outputs the expectation value of the
    operator `matA` assuming that you have the maximally mixed state of the subspace
    defined by the set of states in `vecs`.
    '''
    
    nStates = vecs.shape[1]
    rOut = 0.
    for jIdx in np.arange(nStates):
        rOut += np.sum(vecs[:,jIdx].T.conj()*matA.dot(vecs[:,jIdx]))/nStates
    return rOut

def fMaximallyMixedCorrelator(vecs,matA,matB):
    '''
    This calculates the correlation function <matA.matB>-<matA><matB>, assuming that you
    have the maximally mixed state of the subspace defined by the set of states in `vecs`.
    '''
    
    rOut = fMaximallyMixedExpectation(vecs,matA.dot(matB))-\
           fMaximallyMixedExpectation(vecs,matA)*fMaximallyMixedExpectation(vecs,matB)
    return rOut

def glstHalfFillingIndices(nSites):
    '''
    This generates a list of the indices associated with states that are half filled,
    using the basis defined by the set of creation and annihilation operators being
    used in this code. Note that here, `nSites` is the total number of sites *after*
    counting both the spin up and spin down orbitals.
    '''
    
    lstOut = []
    for jIdx in np.arange(2**nSites):
        b = bin(jIdx)[2:]
        n = 0
        for el in b:
            n += int(el)
        if n==int(nSites/2) or n==int(nSites/2)-1 or n==int(nSites/2)+1:
            lstOut.append(jIdx)
    return lstOut
def glstGeneralFillingIndices(nSites,nFilling):
    '''
    This generates a list of the indices associated with states that have `nFilling` particles,
    using the basis defined by the set of creation and annihilation operators being
    used in this code. Note that here, `nSites` is the total number of sites *after*
    counting both the spin up and spin down orbitals.
    '''
    
    lstOut = []
    for jIdx in np.arange(2**nSites):
        b = bin(jIdx)[2:]
        n = 0
        for el in b:
            n += int(el)
        n = nSites-n
        if n==nFilling or n==nFilling-1 or n==nFilling+1:
            lstOut.append(jIdx)
    return lstOut
def gmatHalfFillingProjector_spindegenerate(nSites):
    '''
    This function creates a projector down to the half filled space. Note that here,
    `nSites` is the total number of sites *after* counting both the spin up and spin
    down orbitals.
    '''
    
    lstHalf = glstHalfFillingIndices(nSites)
    intended = int(spc.binom(nSites,int(nSites/2)-1)+\
                               spc.binom(nSites,int(nSites/2))+\
                               spc.binom(nSites,int(nSites/2)+1))
    if len(lstHalf)==intended:
        print('Looks like the dimension\'s at least right!')
        print(len(lstHalf),intended)
    else:
        print('ERROR: something doesn\'t add up!')
        print(len(lstHalf),intended)
        stop
    matOut = sps.csr_matrix((intended,2**nSites),dtype=np.complex128)
    for jIdx in np.arange(len(lstHalf)):
        matOut[jIdx,lstHalf[jIdx]] = 1.
    return matOut

def gmatGeneralFillingProjector_spindegenerate(nSites,nFilling,bCheck=False):
    '''
    This function creates a projector down to the space with `nFilling` particles. Note
    that here, `nSites` is the total number of sites *after* counting both the spin up
    and spin down orbitals.
    '''
    
    lstGen = glstGeneralFillingIndices(nSites,nFilling)
    intended = int(spc.binom(nSites,nFilling-1)+\
                               spc.binom(nSites,nFilling)+\
                               spc.binom(nSites,nFilling+1))
    if len(lstGen)==intended and bCheck:
        print('Looks like the dimension\'s at least right!')
        print(len(lstGen),intended)
    elif len(lstGen)!=intended:
        print('ERROR: something doesn\'t add up!')
        print(len(lstGen),intended)
        stop
    matOut = sps.csr_matrix((intended,2**nSites),dtype=np.complex128)
    for jIdx in np.arange(len(lstGen)):
        matOut[jIdx,lstGen[jIdx]] = 1.
    return matOut

def glstGeneralFillingIndices_iso(nSites,nFilling):
    lstOut = []
    for jIdx in np.arange(2**nSites):
        b = bin(jIdx)[2:]
        n = 0
        for el in b:
            n += int(el)
        if n==nFilling:
            lstOut.append(jIdx)
    return lstOut
def gmatGeneralFillingProjector_iso(nSites,nFilling):
    lstGen = glstGeneralFillingIndices_iso(nSites,nFilling)
    matOut = sps.csr_matrix((len(lstGen),2**nSites),dtype=np.complex128)
    for jIdx in np.arange(len(lstGen)):
        matOut[jIdx,lstGen[jIdx]] = 1.
    return matOut

def lst2str(lst):
    '''
    This function turns a list into a string.
    '''
    s = ''
    for elt in lst:
        s += str(elt)
    return s
def gmatRemoveHoppings(matTs,lstRmv):
    '''
    This takes a matrix of tunnel couplings on some lattice, `matTs`, and
    removes the indices from `lstRmv`.
    '''
    
    matNewTs = np.delete(np.delete(matTs,lstRmv,0),lstRmv,1)
    return matNewTs

def glstlstSublists(lst): 
    '''
    This function takes in some list of elements, `lst`, and outputs the
    power set of that original list.
    '''
    
    lstEmpty = []   
    lstSet = [lstEmpty] 
    for jIdx in np.arange(len(lst)): 
        lstOrig = lstSet[:] 
        lstNew = lst[jIdx] 
        for kIdx in np.arange(len(lstSet)): 
            lstSet[kIdx] = lstSet[kIdx]+[lstNew] 
        lstSet = lstOrig+lstSet
    return lstSet

def gmatFermiHubbard_manyHoppings(matTs,rE,rU):
    '''
    This function outputs the Hamiltonian for a Fermi-Hubbard model with tunnel couplings
    defined in `matTs`, on-site energy `rE` (which includes the chemical potential), and
    on-site repulsion `rU`.
    '''
    
    nSites = matTs.shape[0]
    matHout = rE*(gmatNumber(0,2*nSites)+gmatNumber(1,2*nSites))+\
              rU*gmatNumber(0,2*nSites).dot(gmatNumber(1,2*nSites))
    for jIdx in np.arange(1,nSites):
        matHout += rE*(gmatNumber(2*jIdx,2*nSites)+gmatNumber(2*jIdx+1,2*nSites))+\
                   rU*gmatNumber(2*jIdx,2*nSites).dot(gmatNumber(2*jIdx+1,2*nSites))
    for jIdx in np.arange(nSites):
        for kIdx in np.arange(nSites):
            if jIdx!=kIdx and matTs[jIdx,kIdx]!=0.:
                rtNow = matTs[jIdx,kIdx]
                matHout +=  rtNow*gmatCreation(2*jIdx,2*nSites).dot(gmatAnnihilation(2*kIdx,2*nSites))+\
                            rtNow*gmatCreation(2*jIdx+1,2*nSites).dot(gmatAnnihilation(2*kIdx+1,2*nSites))+\
                            rtNow*gmatCreation(2*kIdx,2*nSites).dot(gmatAnnihilation(2*jIdx,2*nSites))+\
                            rtNow*gmatCreation(2*kIdx+1,2*nSites).dot(gmatAnnihilation(2*jIdx+1,2*nSites))
    return matHout

def fCalculateOccupancies(raGRws,raWs,rBeta):
    '''
    This function takes the frequency domain Green's function, `raGRws` at the frequency
    points `raWs` and inverse temperature `rBeta`, and calculates the occupancies of each
    site.
    '''
    
    raOut = np.zeros([raGRws.shape[0]],dtype=np.complex128)
    for jIdx in np.arange(raGRws.shape[0]):
        raGRjIdx = raGRws[jIdx,jIdx,:]-raGRws[jIdx,jIdx,0]
        raGLjIdx = fFermi(raWs,rBeta,0.)*(raGRjIdx-raGRjIdx.conj())
        raOut[jIdx] = 2.*sint.trapz(raGLjIdx,raWs)
    return raOut

def gtenLehmanFrequency_semithermal(matH,raws,nSites,rEta,rBeta,nTh,bLimit=True,nMax=256):
    '''
    Here we calculate the Green's functions in the frequency domain using the Lehman representation.
    In light of this being done in the Lehman representation, it uses full, exact diagonalization of
    the Hamiltonian in the input `matH`, so this shouldn't be used with especially large systems.
    The Green's function is calculated at the frequency points in `raws` and assumesthat there are
    `nSites` orbitals in the system (after accounting for spin degeneracies). These functions are
    broadened by `rEta`, and a thermal distribution is taken, including the `nTh` lowest states in
    energy to construct a thermal state with inverse temperature `rBeta`.
    
    On account of this using full, exact diagonalization, there are a few flags included to warn the
    user from. doing this with too large of a system - see the print statement after
    `if bLimit and matH.shape[0]>nMax:` for information on how to get around this.
    
    The output is a 3-index array, the first two indices being associated with the site index and
    the last one  being associated with frequency points.
    '''
    
    # Stop/warn people from using too large of a system
    if bLimit and matH.shape[0]>nMax:
        print('WARNING: this is a brute force calculation that fully diagonalizes the input Hamiltonian.')
        print('Hamiltonians with dimension greater than',nMax,'might be especially inefficient to work with.')
        print('If you still want to do this, set `bLimit=False` in the inputs, or increase `nMax`.')
    
    # Do the full diagonalization
    raEs,vecs = slin.lobpcg(matH,np.eye(matH.shape[0]),\
                                    tol=1.0e-6,maxiter=150,largest=False)
    raEs,vecs = slin.lobpcg(matH,vecs,tol=1.0e-8,maxiter=150,largest=False)
    
    # Allocate space for the Green's functions
    tenGRws = np.zeros([nSites,nSites,len(raws)],dtype=np.complex128)
    
    # Calculate the thermal weights for the state
    raWeights = np.exp(-rBeta*(raEs[:nTh]-raEs[0]))/np.sum(np.exp(-rBeta*(raEs[:nTh]-raEs[0])))
    
    # Loop through all orbitals and frequency points to do the Lehman calculation of the Green's function
    for tIdx,rW in enumerate(raWeights):
        matPartWs = np.zeros([nSites,len(raEs)],dtype=np.complex128)
        matHoleWs = np.zeros([nSites,len(raEs)],dtype=np.complex128)
        for jIdx in np.arange(nSites):
            matCj = gmatAnnihilation(jIdx,nSites)
            for kIdx in np.arange(len(raEs)):
                matPartWs[jIdx,kIdx] = np.sum(vecs[:,kIdx].conj()*matCj.T.conj().dot(vecs[:,tIdx]))
                matHoleWs[jIdx,kIdx] = np.sum(vecs[:,kIdx].conj()*matCj.dot(vecs[:,tIdx]))
        for jIdx in np.arange(nSites):
            for kIdx in np.arange(nSites):
                for eIdx,rEn in enumerate(raEs):
                    tenGRws[jIdx,kIdx,:] += (matPartWs[jIdx,eIdx].conj()*matPartWs[kIdx,eIdx]/(raws+1.j*rEta-\
                                  (rEn-raEs[tIdx]))+\
                                  matHoleWs[jIdx,eIdx]*matHoleWs[kIdx,eIdx].conj()/(raws+1.j*rEta+\
                                  (rEn-raEs[tIdx])))*rW
    # Output!
    return tenGRws

def gtenRetardedGreensOperatorTime_ground(nsites,matH,rTmax,ntimes,bTrack=False,nTrack=100):
    '''
    This function calculates the full (single-particle) Green's function/operator on Hamiltonian
    `matH` with `nsites` fermionic modes. Specifically, it fills out
    \begin{equation}
    G_{jk}^R(t)=-i\theta(t)\left\langle\left\left\{\hat{U}(t)^\dagger\hat{c}_ j\hat{U}(t),
    \hat{c}_ k^\dagger\right\}\right\rangle
    \end{equation}
    for all $j$ and $k$ indexing the possible fermionic modes, with the expectation value being
    relative to the ground state of the Hamiltonian. The values of $G_{jk}^R(t)$ are saved for
    $t$ from $0$ to `rTmax` on a grid with `ntimes` points in time. The boolean `bTrack` is for
    when we want to see the progress of the calculation, being updated every `nTrack` time steps.
    
    Note that this particular Green's function, being in the time domain, isn't necessarily the
    best for calculating things in the frequency domain - while you of course can take this and
    Fourier transform it, there's a possibility that you'll end up with numerical artifacts from
    windowing if you don't have a *lot* of data points in time. In order to resolve low energy
    phenomena, you need a long calculation in the time domain, but in order to make sure the
    calculation is still accurate you need small time steps to account for high energy phenomena.
    That's why the actual transport calculations resort to frequency domain calculations.
    '''
    
    np.random.seed(0)
    raEsFull,vecsFull = slin.lobpcg(matH,np.random.rand(2**nsites,3),tol=1.0e-6,maxiter=150,largest=False)
    raEsFull,vecsFull = slin.lobpcg(matH,vecsFull,tol=1.0e-8,maxiter=150,largest=False)
    matN_full = gmatTotalNumber(int(nsites/2))
    rN_now = np.sum(vecsFull[:,0].T.conj()*matN_full.dot(vecsFull[:,0].T.conj()))
    matProj = gmatGeneralFillingProjector_spindegenerate(nsites,int(np.abs(np.round(rN_now,3))))
    matH_hf = matProj.dot(matH.dot(matProj.T.conj()))
    
    # First calculate the ground state and energy of the system.
    vecGS,rEg = gGroundSystem(matH_hf)
    vecGS_full,rEg_full = gGroundSystem(matH)
    
    # Here we allocate space for the vectors representing the action of the (time-evolved)
    # creation and annihilation operators on the ground state, in order to avoid calculating
    # large, dense matrices for the time evolution operators. Those ending in `0s` are static
    # in time, and those ending in `ts` are time evolved, `d` denoting "dagger" for creation
    # operators and the others being annihilation operators.
    teng0s = np.zeros([nsites,len(vecGS)],dtype=np.complex128)
    tengts = np.zeros([nsites,len(vecGS)],dtype=np.complex128)
    tengd0s = np.zeros([nsites,len(vecGS)],dtype=np.complex128)
    tengdts = np.zeros([nsites,len(vecGS)],dtype=np.complex128)
    
    # This is where the vectors are filled out.
    for jIdx in np.arange(nsites):
        matA0 = matProj.dot(gmatAnnihilation(jIdx,nsites).dot(matProj.T.conj()))
        matC0 = matProj.dot(gmatCreation(jIdx,nsites).dot(matProj.T.conj()))
        teng0s[jIdx,:] = matA0.dot(vecGS)
        tengts[jIdx,:] = matA0.dot(vecGS)
        tengd0s[jIdx,:] = matC0.dot(vecGS)
        tengdts[jIdx,:] = matC0.dot(vecGS)
        
    # Here we fill in the grid points in time and get the time step size.
    raTimes = np.linspace(0.,rTmax,ntimes)
    rdT = raTimes[1]-raTimes[0]
    
    # Below is the approximation being used for the Crank-Nicolson time evolution. This is
    # currently set to be first order, but if better accuracy is needed this can be adjusted
    # to a higher order of approximation.
    matUdt = sps.eye(len(vecGS),format='csr')-0.5j*rdT*matH_hf
    
    # Here we allocate space for $G_{jk}^R(t)$, the first index in the object being for $j$, then
    # $k$, then an index for the time.
    tenGRt_vals = np.zeros([nsites,nsites,ntimes],dtype=np.complex128)
    
    # Below is the loop that fills out `tenGRt_vals` at each time step, updating `tengts` and
    # `tengdts` as it goes along.
    for lIdx,rT in enumerate(raTimes):
        if lIdx%nTrack==0 and bTrack:
            print(datetime.datetime.now(),lIdx)
        for jIdx in np.arange(nsites):
            for kIdx in np.arange(nsites):
                tenGRt_vals[jIdx,kIdx,lIdx] = -1.j*(np.exp(-1.j*rEg*rT)*np.sum(tengts[kIdx,:].conj()*\
                                                                               teng0s[jIdx,:])+\
                                                    np.exp(1.j*rEg*rT)*np.sum(tengd0s[jIdx,:].conj()*\
                                                                              tengdts[kIdx,:]))
        for jIdx in np.arange(nsites):
            tengts[jIdx,:] = gvecTimeEvolved(tengts[jIdx,:],matUdt)
            tengdts[jIdx,:] = gvecTimeEvolved(tengdts[jIdx,:],matUdt)
            
    if bGS:
        plot(np.abs(vecGS),label=str(np.round(rEg,3)))
        legend()
    
    # Output the Green's operator and time grid.
    return tenGRt_vals,raTimes

def gtenMWparts_Wideband(tenGRs,matGammaL,matGammaR):
    '''
    This defines a few tensors that expidite the calculation of the MW formula by
    making sure a bunch of matrix calculations only get done once.
    '''
    tenP1 = 0.*tenGRs
    tenP2 = 0.*tenGRs
    tenP3 = 0.*tenGRs
    for jIdx in np.arange(tenGRs.shape[2]):
        matGRf = tenGRs[:,:,jIdx]
        matGAf = matGRf.T.conj()
        tenP1[:,:,jIdx] = matGammaL.dot(matGRf.dot((matGammaL+matGammaR).dot(matGAf)))
        tenP2[:,:,jIdx] = matGammaL.dot(matGRf.dot((matGammaL).dot(matGAf)))
        tenP3[:,:,jIdx] = matGammaL.dot(matGRf.dot((matGammaR).dot(matGAf)))
    return tenP1,tenP2,tenP3

def fMeirWingreen_WideBand_detailed(raOmegas,tenP1,tenP2,tenP3,rmuL,rmuR,rBeta):
    '''
    This is meant to calculate the function being integrated in the Meir-Wingreen formula for the current,
    under the assumption that the connected leads/electron baths fit the wide-band approximation. This
    function, output as `raMW`, is calculated at independent variable values `raOmegas` assuming a left
    lead at chemical potential `muL`, the right being at `muR`. `rBeta` is the inverse temperature and
    `matGammaL` is the coupling to/from the left lead, `matGammaR` to/from the right. The boolean `bWB`
    should be input as `True` if `tenGRws`, the single-particle Green's operator for the system, already
    includes the embedding self-energy from the leads, `False` otherwise. As opposed to previous functions,
    here we allow the hopping from the each lead to differ for each individual site, allowing a more
    detailed and potentially accurate description of the system.
    '''
    
    # The space for the function output is allocated here, then calculated in the loop.
    raMW = -np.trace(fFermi(raOmegas,rBeta,rmuL)*tenP1-\
                    (fFermi(raOmegas,rBeta,rmuL)*tenP2+fFermi(raOmegas,rBeta,rmuR)*tenP3))
    return raMW

def gcalcBiasSweep(matTs,matGammaL,matGammaR,rE,rmu,rU,raBias,ftTimes,rEta,rBeta,nTh,bLimit=True,nMax=256):
    '''
    This function takes in a matrix of tunnel couplings, `matTs` between interacting sites and `matGammaL`/
    `matGammaR` for tunneling to the leads, the on-site energy `rE`, chemical potential `rmu`, on-site
    repulsion `rU`, with the Green's functions to be calculated at the frequency points `ftTimes` for a
    thermal state consisting of the `nTh` lowest energy states with inverse temperature `rBeta`, with the
    Green's function to be broadened by `rEta`, and outputs the current calculated at the biases defined
    in `raBias`,  to be output in `raCurrents`.
    '''
    
    # Allocate space for the output
    raCurrents = np.zeros([raBias.shape[0]])
    # Define the Hamiltonian
    matH = gmatFermiHubbard_manyHoppings(matTs,rE-rmu,rU)
    # Calculate the Green's functions
    ftGRs = gtenLehmanFrequency_semithermal(matH,ftTimes,2*matTs.shape[0],rEta,rBeta,nTh,bLimit,nMax)
    # Define a few tensors that expidite the calculation of the current
    tenP1,tenP2,tenP3 = gtenMWparts_Wideband(ftGRs,matGammaL,matGammaR)
    # Calculate the current for each desired bias
    for jIdx,rBias in enumerate(raBias):
        raMW = fMeirWingreen_WideBand_detailed(ftTimes,tenP1,tenP2,tenP3,0.5*rBias,-0.5*rBias,rBeta)
        raCurrents[jIdx] = np.real(sint.trapz(raMW,ftTimes))
    return raCurrents

def gcalcDoubleBiasSweep(matTs,matGammaL,matGammaR,raErange,rmu,rU,raBias,ftTimes,rEta,rBeta,nTh,\
                         bLimit=True,nMax=256,nTrack=None):
    '''
    This function takes in a matrix of tunnel couplings, `matTs` between interacting sites and `matGammaL`/
    `matGammaR` for tunneling to the leads, chemical potential `rmu`, on-site
    repulsion `rU`, with the Green's functions to be calculated at the frequency points `ftTimes` for a
    thermal state consisting of the `nTh` lowest energy states with inverse temperature `rBeta`, with the
    Green's function to be broadened by `rEta`, and outputs the current calculated at the biases defined
    in `raBias` and for the set of on-site energies defined in `raErange`,  to be output in `raCurrents2D`.
    '''
    raCurrents2D = np.zeros([raBias.shape[0],raErange.shape[0]])
    for jIdx,rE_now in enumerate(raErange):
        if nTrack!=None:
            if jIdx%nTrack==0:
                print(jIdx,datetime.datetime.now())
        raCurrents2D[:,jIdx] = gcalcBiasSweep(matTs,matGammaL,matGammaR,\
                                              rE_now,rmu,rU,raBias,ftTimes,rEta,rBeta,nTh,bLimit,nMax)
    return raCurrents2D
