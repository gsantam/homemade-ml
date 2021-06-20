import numpy as np
import math
from collections import Counter


class HomemadeDecisionTree():
    def __init__(self,X,y,max_depth,depth = 0,mean_y = None):
        self.X = X
        self.y = y
        self.leaf = False
        self.depth = depth
        self.max_depth = max_depth
        self.mean_y = mean_y
        
    def fit(self):
        if not (self.X.shape[0]>0 and self.X.shape[1]>0 and self.depth<self.max_depth):
            self.leaf = True
            return 
            
        best_cut = self.get_best_predictor_with_cut()
        self.predictor_idx = best_cut[0]
        self.cut = best_cut[1]
        self.mean_y_under_cut = best_cut[2]
        self.mean_y_over_cut = best_cut[3]
        predictor = self.X[:,self.predictor_idx]
        
        X_left = self.X[predictor<self.cut,:]
        y_left = self.y[predictor<self.cut]
        #X_left = np.delete(X_left,self.predictor_idx,axis = 1)  
        self.left_node = self.__class__(X_left,y_left,self.max_depth,self.depth + 1,self.mean_y_under_cut)
        self.left_node.fit()
    
        X_right = self.X[predictor>=self.cut,:]
        y_right = self.y[predictor>=self.cut]
        #X_right = np.delete(X_right,self.predictor_idx,axis = 1)  
        self.right_node = self.__class__(X_right,y_right,self.max_depth,self.depth + 1,self.mean_y_over_cut)
        self.right_node.fit()
            
    def predict(self,X):
        y_s = []
        for x in X:
            y_s.append(self.predict_(x))
        return np.asarray(y_s)
        
    def predict_(self,x):
        if self.leaf:
            return self.mean_y
        else:
            value_at_predictor = x[self.predictor_idx]
            #x_without_predictor = np.delete(x,self.predictor_idx,axis = 0)
            x_without_predictor = x
            if value_at_predictor < self.cut:
                return self.left_node.predict_(x_without_predictor)
            else:
                return self.right_node.predict_(x_without_predictor)
        
    def get_best_predictor_with_cut(self):

        best_key_metric = 10**10
        for predictor_idx in range(self.X.shape[1]):
            best_key_metric_of_predictor = 10**10
            predictor = self.X[:,predictor_idx]
            for cut in predictor:
                key_metric,value_y_under_cut,value_y_over_cut = self.evaluate_key_metric(predictor,cut)
                if key_metric<=best_key_metric_of_predictor:
                    best_key_metric_of_predictor = key_metric
                    best_cut_of_predictor = [predictor_idx,cut,value_y_under_cut,value_y_over_cut]
                    
            if best_key_metric_of_predictor<best_key_metric:
                best_key_metric = best_key_metric_of_predictor 
                best_cut = best_cut_of_predictor
                    
        return best_cut
    
    def get_error(self):
        if self.leaf:
            return np.sum((self.mean_y - self.y)**2)
        else:
            return self.left_node.get_error() + self.right_node.get_error()

class HomemadeDecisionTreeRegressor(HomemadeDecisionTree):
    def evaluate_key_metric(self,predictor,cut):
        mean_y_under_cut = np.mean(self.y[predictor<cut])
        mean_y_over_cut = np.mean(self.y[predictor>=cut])
        sse = np.sum((self.y[predictor<cut] - mean_y_under_cut)**2) + np.sum((self.y[predictor>=cut] - mean_y_over_cut)**2)
        mse = (1/len(self.y))*sse
        return mse,mean_y_under_cut,mean_y_over_cut

class HomemadeDecisionTreeClassifier(HomemadeDecisionTree):
    def compute_entropy(self,props):
        entropy = 0
        for prop in props:
            entropy-=props[prop]*math.log(props[prop])
        return entropy

    def evaluate_key_metric(self,predictor,cut):
        prop_y_under_cut = Counter(self.y[predictor<cut])
        prop_y_over_cut = Counter(self.y[predictor>=cut])
        prop_y_under_cut = {x:prop_y_under_cut[x]/len(self.y[predictor<cut]) for x in prop_y_under_cut}
        prop_y_over_cut = {x:prop_y_over_cut[x]/len(self.y[predictor>=cut]) for x in prop_y_over_cut}
        entropy = 0
        if len(self.y[predictor<cut])>0:
            entropy += self.compute_entropy(prop_y_under_cut)*(len(self.y[predictor<cut])/len(self.y) ) 
        if len(self.y[predictor>=cut])>0:
            entropy += self.compute_entropy(prop_y_over_cut)*(len(self.y[predictor>=cut])/len(self.y) ) 

        return entropy,prop_y_under_cut,prop_y_over_cut
                

if __name__ == "__main__":
    pass