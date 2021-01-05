import numpy as np

class LorentzVector(object):
    def __init__(self, fourVec, Px=None, Py=None, Pz=None, E=None):
        if Px is not None:
            self.Px = Px
            self.Py = Py
            self.Pz = Pz
            self.E  = E
        else:
            self.Px = fourVec.fP.fX
            self.Py = fourVec.fP.fY
            self.Pz = fourVec.fP.fZ
            self.E  = fourVec.fE
        self.setVar()
        
    def setVar(self):
        self.P2 = self.Px*self.Px + self.Py*self.Py + self.Pz*self.Pz
        self.P  = np.sqrt(self.P2)
        self.M2 = self.E*self.E - self.P2
        self.M  = np.sqrt(np.abs(self.M2))*np.sign(self.M2)
        self.Pt = np.sqrt(self.Px*self.Px + self.Py*self.Py)
        self.cosTheta = self.Pz/self.P
        self.Eta = -0.5*np.log( (1.0-self.cosTheta)/(1.0+self.cosTheta) )
        self.Phi = np.arctan2(self.Py, self.Px)

    def __add__(self, other):
        Px = np.concatenate([self.Px, other.Px])
        Py = np.concatenate([self.Py, other.Py])
        Pz = np.concatenate([self.Pz, other.Pz])
        E  = np.concatenate([self.E,  other.E])
        return LorentzVector(None, Px, Py, Pz, E)

    def count(self):
        return len(self.P)
    
    def cuts(self, pt, eta):
        cut = np.ones(len(self.P),dtype=bool)
        cut = self.Pt > pt and np.abs(self.Eta) < eta
        self.Px = self.Px[cut]
        self.Py = self.Py[cut]
        self.Pz = self.Pz[cut]
        self.E  = self.E[cut]
        self.setVar()        

    def boost(self, bx, by, bz):
        b2 = bx*bx + by*by + bz*bz
        gamma = 1.0 / np.sqrt(1.0 - b2)
        bp = bx*self.Px + by*self.Py + bz*self.Pz
        gamma2 = (gamma - 1.0)/b2
  
        self.Px = self.Px + gamma2*bp*bx + gamma*bx*self.E
        self.Py = self.Py + gamma2*bp*by + gamma*by*self.E
        self.Pz = self.Pz + gamma2*bp*bz + gamma*bz*self.E
        self.E  = gamma*(self.E + bp)
        self.setVar()

    def sortByP(self, cut=7):
        argsort = np.argsort(self.P)[::-1]
        self.Px = self.Px[argsort][:cut]
        self.Py = self.Py[argsort][:cut]
        self.Pz = self.Pz[argsort][:cut]
        self.E  = self.E[argsort][:cut]
        self.setVar()

    def getFWM(self):
        fwmom_ = list(0.0 for i in range(0,5+1))
        esum_total = sum(self.P)
        esum_total_sq = esum_total * esum_total 
        for i in range(0, len(self.P)):
            p_i = self.P[i]
            for j in range(0, len(self.P)):
                p_j = self.P[j]
                cosTheta = (self.Px[i]*self.Px[j] + self.Py[i]*self.Py[j] + self.Pz[i]*self.Pz[j]) / (p_i*p_j)                
                pi_pj_over_etot2 = p_i * p_j / esum_total_sq
                
                fwmom_[0] += pi_pj_over_etot2 
                fwmom_[1] += pi_pj_over_etot2 * cosTheta 
                fwmom_[2] += pi_pj_over_etot2 * 0.5 * ( 3. * pow( cosTheta, 2. ) - 1. ) 
                fwmom_[3] += pi_pj_over_etot2 * 0.5 * ( 5. * pow( cosTheta, 3. ) - 3. * cosTheta ) 
                fwmom_[4] += pi_pj_over_etot2 * 0.125 * ( 35. * pow( cosTheta, 4. ) - 30. * pow( cosTheta, 2. ) + 3. ) 
                fwmom_[5] += pi_pj_over_etot2 * 0.125 * ( 63. * pow( cosTheta, 5. ) - 70. * pow( cosTheta, 3. ) + 15. * cosTheta ) 
        return np.array(fwmom_)

    def getJMT(self):
        momentumTensor = np.zeros((3, 3))
        norm = 0.0        
        for i in range(0, len(self.P)):
            norm += self.P2[i]
            momentumTensor[0][0] += self.Px[i]*self.Px[i]
            momentumTensor[0][1] += self.Px[i]*self.Py[i]
            momentumTensor[0][2] += self.Px[i]*self.Pz[i]
            momentumTensor[1][0] += self.Py[i]*self.Px[i]
            momentumTensor[1][1] += self.Py[i]*self.Py[i]
            momentumTensor[1][2] += self.Py[i]*self.Pz[i]
            momentumTensor[2][0] += self.Pz[i]*self.Px[i]
            momentumTensor[2][1] += self.Pz[i]*self.Py[i]
            momentumTensor[2][2] += self.Pz[i]*self.Pz[i]            
        momentumTensor = (1./norm)*momentumTensor
        e, v = np.linalg.eig(momentumTensor)
        return e

