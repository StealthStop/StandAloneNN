import tensorflow as tf
import numpy as np
import uproot

class Model(object):
    def __init__(self, model_filepath, cfg_filepath):
        self.model_filepath = model_filepath
        self.cfg_filepath = cfg_filepath
        self.load_graph(model_filepath=self.model_filepath, cfg_filepath=self.cfg_filepath)

    def readCfgFile(self, cfg_filepath):
        print('Reading config file')
        f = open(cfg_filepath, 'r')
        lines = f.read().splitlines()
        f.close()
        lines = [l.replace(' ','') for l in lines if "=" in l]        

        dic = {}
        for l in lines:
            a,b = l.split('=')
            c = a.split('[')
            d = b.replace('"','')
            
            if not c[0] in dic:
                dic[c[0]] = d
            elif type(dic[c[0]]) is not type([]):
                dic[c[0]] = [dic[c[0]], d]
            else:
                dic[c[0]].append(d)
        return dic
            
    def load_graph(self, model_filepath, cfg_filepath):
        print('Loading model...')
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.config = self.readCfgFile(cfg_filepath)
        numInputVar = len(self.config['mvaVar'])
        
        with self.graph.as_default():
            self.input = tf.placeholder(np.float32, shape = [None, numInputVar], name=self.config['inputOp'])
            self.dropout_rate = tf.placeholder(tf.float32, shape = [], name = self.config['outputOp'])
            tf.import_graph_def(graph_def, {self.config['inputOp']: self.input, self.config['outputOp']: self.dropout_rate})

        self.graph.finalize()
        self.sess = tf.Session(graph = self.graph)

    def predict(self, data):
        output_tensor = self.graph.get_tensor_by_name("import/"+self.config['outputOp']+":0")
        output = self.sess.run(output_tensor, feed_dict = {self.input: data, self.dropout_rate: 0})
        return output

class Jets(object):
    def __init__(self, fourVec):
        self.Px = fourVec.fP.fX
        self.Py = fourVec.fP.fY
        self.Pz = fourVec.fP.fZ
        self.E  = fourVec.fE
        self.P2 = self.Px*self.Px + self.Py*self.Py + self.Pz*self.Pz
        self.P  = np.sqrt(self.P2)
        self.M  = self.mass() 
        
    def mass(self):
        return np.sqrt(self.E*self.E - self.P2)
    
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
    
def main():
    treename = "TreeMaker2/PreSelection" #"myMiniTree"
    samplesToRun = ["Fall17.RPV_2t6j_mStop-1000_mN1-100_TuneCP2_13TeV-madgraphMLM-pythia8_0_RA2AnalysisTree.root"]
    datasets = getDataSets(samplesToRun, treename)
    events_Jets = datasets[0]['Jets'].array()

    for event_Jets in events_Jets:
        #jets = JaggedCandidateArray.candidatesfromcounts(
        #    df['Jets'].counts,
        #    px=df['Jets'].fP.fX.flatten(),
        #    py=df['Jets'].fP.fY.flatten(),
        #    pz=df['Jets'].fP.fZ.flatten(),
        #    energy=df['Jets'].fE.flatten(),
        #)

        jets = Jets(event_Jets)
        print len(event_Jets), jets.Px[0], jets.Py[0], jets.Pz[0], jets.E[0], jets.P[0], jets.M[0] 

    
    #model = Model(model_filepath = "keras_frozen_2017.pb", cfg_filepath = "DeepEventShape_2017.cfg")
    #rand_array = np.random.rand(1, 39)
    #print rand_array
    #print model.predict(rand_array)

if __name__ == '__main__':
    main()
    
