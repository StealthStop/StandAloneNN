import tensorflow as tf
import numpy as np

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

