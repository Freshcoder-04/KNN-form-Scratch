import numpy as np

class KNN:
    def __init__(self,k,metric):
        self.k = k
        self.metric = metric

    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def EuclideanDistance(self,X_test):
        X_train_sq = np.sum(np.array(self._X_train)**2,axis=1)
        X_test_sq = np.sum(np.array(X_test)**2,axis=1)
        distances = np.sqrt(X_train_sq[np.newaxis,:] + X_test_sq[:,np.newaxis] - 2 * np.dot(X_test, self._X_train.T))
        return distances

    def ManhattanDistance(self,X_test):
        distances = np.abs(X_test.values[:, np.newaxis, :] - self._X_train.values[np.newaxis, :, :]).sum(axis=2)
        return distances

    def CosineDistance(self,X_test):
        Train_Norm = np.sqrt(np.sum(self._X_train.values**2,axis=1))
        Test_Norm = np.sqrt(np.sum(X_test.values**2,axis=1))
        Distances = 1-np.dot(X_test/Test_Norm[:,np.newaxis],self._X_train.T/Train_Norm)
        return Distances

    def predict(self):
        k_nearest_ind = np.argpartition(self.AllDist, self.k, axis=1)[:, :self.k]
        # print(k_nearest_ind.shape)
        k_nearest_labels = np.array((k_nearest_ind.shape[0],k_nearest_ind.shape[1]),dtype=int)
        k_nearest_labels = np.array([self.Y_train.iloc[ind]['genre_index'] for ind in k_nearest_ind])
        k_nearest_labels = k_nearest_labels.astype(int)
        Y_predicted = np.array([np.argmax(np.bincount(k_nearest_labels[i])) for i in range(k_nearest_labels.shape[0])])
        return Y_predicted
    
    def HyperParameterTuning(self,X_test,Y_test):
        met = Metrics()
        perf = {}
        for m in range (1,4):
            self.metric = m
            self.knn(X_test)
            for k in range(3,40,2):
                self.k = k
                self.Y_predicted = self.predict()
                acc = met.Accuracy(Y_test,self.Y_predicted)
                perf[(self.k,self.metric)] = acc
        # Reference: geeksforgeeks
        # Start
        keys = list(perf.keys())
        values = list(perf.values())
        sorted_value_ind = np.argsort(values)
        perfSorted = {keys[i]:values[i] for i in sorted_value_ind}
        self.SortedHP = perfSorted
        # End
        return {k: v for k, v in list(perfSorted.items())[-1:-11:-1]}

    def knn(self,X_test):
        batchSize = 1000
        numberBatches = (int)(self.X_train.shape[0]/batchSize)
        tempInd = 0
        result = np.empty((0,0))
        print(numberBatches)
        for i in range(1,numberBatches+1):
            print(i)
            self._X_train = self.X_train[tempInd:tempInd+batchSize]
            tempInd+=batchSize
            if self.metric==1:
                DistancesCalc = self.CosineDistance(X_test)
            elif self.metric==2:
                DistancesCalc = self.EuclideanDistance(X_test)
            elif self.metric==3:
                DistancesCalc = self.ManhattanDistance(X_test)
            if result.shape == (0,0):
                result = DistancesCalc
            else:
                result = np.hstack((result,DistancesCalc))


        self._X_train = self.X_train[tempInd:]
        if self.metric==1:
            DistancesCalc = self.CosineDistance(X_test)
        elif self.metric==2:
            DistancesCalc = self.EuclideanDistance(X_test)
        elif self.metric==3:
            DistancesCalc = self.ManhattanDistance(X_test)
        if result.shape == (0,0):
                result = DistancesCalc
        else:
            result = np.hstack((result,DistancesCalc)) # Chatgpt

        print("All batches executed successfully")
        self.AllDist = result

class Metrics:
    def __init__(self):
        pass
    def Accuracy(self,Y_test,Y_predicted):
        # Implement Accuracy, Recall, f1-score
        Correct = []
        for i in range(0,Y_predicted.shape[0]):
            if (Y_predicted[i] == Y_test.iloc[i]['genre_index']):
                Correct.append(1)
            else:
                Correct.append(0)
        accuracy = sum(Correct)/len(Correct)*100
        return accuracy