import numpy as np
import uproot
from LorentzVector import LorentzVector
from Model import Model
    
def getDataSets(samplesToRun, treename):
    dsets = []
    if len(samplesToRun) == 0:
        raise IndexError("No sample in samplesToRun")
    for filename in samplesToRun:
        try:
            f = uproot.open(filename)
            dsets.append( f[treename] )
        except Exception as e:
            print("Warning: \"%s\" has issues" % filename, e)
            continue
    return dsets

def makeInputVars(events, config, nEvents = 100):
    inputVars = []
    nJetsVec = []
    for event in events:
        if len(inputVars) == nEvents: break
        
        jets = LorentzVector(event["Jets"])
        beta = sum(jets.Pz) / sum(jets.E)
        jets.cuts(30.0, 2.4)
        nJets = jets.count()
        if nJets < 7: continue
        nJetsVec.append(nJets)                
        jets.boost(0.0, 0.0, -beta)
        jets.sortByP(7)
        fwm = jets.getFWM()
        jmte = jets.getJMT()

        muons = LorentzVector(event["Muons"])
        electrons = LorentzVector(event["Electrons"])
        leptons = muons + electrons
        nLeptons = leptons.count()
        if nLeptons < 1: continue
        leptons.boost(0.0, 0.0, -beta)
        leptons.sortByP(1)
        
        inputVar = np.concatenate([jets.Pt, jets.Eta, jets.Phi, jets.M, leptons.Pt, leptons.Eta, leptons.Phi, leptons.M, fwm[2:], jmte])
        inputVars.append(inputVar)        
    return np.array(inputVars), np.array(nJetsVec)

def getNNBin(discriminators, config, nJets):
    binNumVec = []
    for i in range(0, len(discriminators)): 
        discriminator = discriminators[i]
        NGoodJets_pt30 = nJets[i]        
        nMVABin = (len(config["binEdges"]) / (int(config["maxNJet"]) - int(config["minNJet"]) + 1)) - 1
        nJetBinning = None
        if(NGoodJets_pt30 < int(config["minNJet"])):
            nJetBinning = 0
        elif(int(config["minNJet"]) <= NGoodJets_pt30 and NGoodJets_pt30 <= int(config["maxNJet"])):
            nJetBinning = NGoodJets_pt30-int(config["minNJet"])
        elif(int(config["maxNJet"]) < NGoodJets_pt30):
            nJetBinning = int(config["maxNJet"])-int(config["minNJet"])
        
        for j in range((nMVABin+1)*nJetBinning + 1, (nMVABin+1)*(nJetBinning+1)):
            passDeepESMBin = discriminator > float(config["binEdges"][j-1]) and discriminator <= float(config["binEdges"][j])
            binNum = j - (nMVABin+1)*nJetBinning
            if passDeepESMBin: binNumVec.append(binNum)
    return np.array(binNumVec)

def main():
    #Setup input data (only an example just need a array of data corresponding to the correct input variables for each event)
    treename = "TreeMaker2/PreSelection"
    samplesToRun = ["test.root"]
    datasets = getDataSets(samplesToRun, treename)
    events = datasets[0].arrays(["Jets", "Muons", "Electrons"])
    
    #Make the NN model object that will output the NN score
    model = Model(model_filepath = "keras_frozen_2017.pb", cfg_filepath = "DeepEventShape_2017.cfg")
    inputVars, nJets = makeInputVars(events, model.config, 100)

    #Run the NN and compute which MVA bin each event belongs to (Two main output)
    discriminator = model.predict(inputVars)[:,0]
    nnBinValues = getNNBin(discriminator, model.config, nJets)
    for i in range(0, len(discriminator)):
        print "MVA bin:", nnBinValues[i], "     NN Score:", discriminator[i]
    
if __name__ == '__main__':
    main()
    
