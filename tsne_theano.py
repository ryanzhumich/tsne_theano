'''
Implementation of t-SNE in Theano

Code is modified from lvdmaaten.github.io/tsne/

Created by Rui Zhang
Dec, 2015
'''

import cPickle
import gzip
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
import pylab
from vis import visual_digits

numpy.random.seed(1234)

class tSNE(object):
    def __init__(self, X, no_dims, initial_dims, perplexity):
        # Take down inputs
        self.X = X
        self.no_dims = no_dims
        self.initial_dims = initial_dims
        self.perplexity = perplexity
        self.n,self.d = self.X.shape
        
        # Reduce dimension of X to initial_dims using PCA

        self.X = self.pca()
        print 'mean pca X', numpy.mean(self.X)
        print 'var pca X', numpy.var(self.X)
        self.n,self.d = self.X.shape
 
        # Compute P matrix
        P = self.x2p()
        self.P = theano.shared(numpy.asarray(P,dtype=theano.config.floatX),borrow=True)
        # Initialize Y
        Y_init = numpy.random.randn(self.n,self.no_dims).astype(theano.config.floatX)

        print 'mean Y init', numpy.mean(Y_init)
        print 'var Y init', numpy.var(Y_init)

        self.Y = theano.shared(value=Y_init, name='Y', borrow=True)

        # Construct computational graphs to compute Q matrix
        sum_Y = T.sum(T.sqr(self.Y),1)
        num = 1 / (1 + T.add(T.add(-2 * T.dot(self.Y,self.Y.T), sum_Y).T, sum_Y))
        num = T.extra_ops.fill_diagonal(num,0)
        Q_ = num / T.sum(num)
        self.num = num
        Q_ = T.maximum(Q_,1e-12)
        self.Q = Q_

        # parameters of the model
        self.params = [self.Y]
        
    def KL_cost(self):
        # return KL divergence between P and Q matrix
        return T.sum(self.P * T.log(self.P / self.Q))

    def pca(self):
        print 'Reducing dimension to',self.initial_dims,'...'
        #X = self.X - numpy.mean(self.X,0)
        #u,s,v = numpy.linalg.svd(numpy.dot(X.T,X))
        X = self.X - numpy.tile(numpy.mean(self.X,0),(self.n,1)) 
        (l, M) = numpy.linalg.eig(numpy.dot(X.T,X))
        Y = numpy.dot(X, M[:,0:self.initial_dims])
        return Y.real
        #return numpy.dot(X,u[:,:self.initial_dims]).real
 
    
    def x2p(self,tol=1e-5):
        # Search for \sigma for each x_i given a fixed perplexity
        print 'Searching for \sigma...' 
        (n,d) = self.X.shape
        sum_X = numpy.sum(numpy.square(self.X),1)
        D = numpy.add(numpy.add(-2 * numpy.dot(self.X, self.X.T), sum_X).T, sum_X) # D is the matrix where entries are pair-wise Euclidean distance of X
        P = numpy.zeros((n,n))
        beta = numpy.ones((n,1))           # \beta_i = 2 * \sigma_i^2 
        logU = numpy.log(self.perplexity)


        print 'mean D',numpy.mean(D)
        print 'var D',numpy.var(D)
        # loop through each x_i
        for i in range(n):
            if i % 500 == 0:
                print i,'/',n
            # binary search over beta[i]
            betamin = -numpy.inf
            betamax = numpy.inf
            Di = D[i, numpy.concatenate((numpy.r_[0:i], numpy.r_[i+1:n]))] # elimate D[i,i]
            (H, thisP) = self.Hbeta(Di,beta[i])
            Hdiff = H - logU
            tries = 0
            while numpy.abs(Hdiff) > tol and tries < 50:
                if Hdiff > 0:
                #perplexity is too big, need a smaller \sigma_i(larger \beta_i)
                    betamin = beta[i].copy()
                    if betamax == numpy.inf or betamax == -numpy.inf:
                        beta[i] *= 2
                    else:
                        beta[i] = (beta[i] + betamax) / 2
                else:
                    betamax = beta[i].copy()
                    if betamin == numpy.inf or betamin == -numpy.inf:
                        beta[i] /= 2
                    else:
                        beta[i] = (beta[i] + betamin) / 2
                (H,thisP) = self.Hbeta(Di,beta[i])
                Hdiff = H - logU
                tries += 1
            P[i, numpy.concatenate((numpy.r_[0:i],numpy.r_[i+1:n]))] = thisP;
        
        # Return the symmetric version of P matrix
        # Note that p_{j|i} and p_{i|j} is transpose of each other
        P = P + numpy.transpose(P)
        P = P / numpy.sum(P)
        # early exaggeration
        P = P * 4
        P = numpy.maximum(P,1e-12)

        print 'Mean value of sigma:', numpy.mean(numpy.sqrt(1/beta))
        print  'Var of P:', numpy.var(P)
        return P
    
    def Hbeta(self,D = numpy.array([]), beta = 1.0):
        # input: 
        #     D:    pair-wise distance array of x_i
        #     beta: 2 * \sigma_i^2
        # output:
        #     P:    conditional probability P_i 
        #     H:    Shannon Entropy of P_i

        P = numpy.exp(-D.copy() * beta)
        sum_P = sum(P)
        # calculation of Shannon entropy of P
        H = numpy.log(sum_P) + beta * numpy.sum(D * P) / sum_P;
        P = P / sum_P
        return H, P
         
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #dataset = 'mnist_train'
    if 'mnist' in dataset:
        #print 'Loading MNIST2500...'
        #X = numpy.loadtxt("../data/mnist2500_X.txt")
        #labels = numpy.loadtxt("../data/mnist2500_labels.txt")


        datafile = gzip.open('./data/mnist.pkl.gz','rb')
        train_set,valid_set,test_set = cPickle.load(datafile)
    
        if 'train' in dataset:
            print 'Loading MNIST Train'
            x = train_set[0]
            y = train_set[1]

            x = x[:10000,:]
            y = y[:10000]

            for l in range(11):
                print l,sum(1 for a in y if a == l)

        elif 'test' in dataset:
            print 'Loading MNIST Test'
            x = test_set[0]
            y = test_set[1]


        #print X.shape,labels.shape
        #print x.shape,y.shape
        #no shared variable of X for now
        #X is used only once for computing P matrix 
        #X = theano.shared(numpy.asarray(X,dtype=theano.config.floatX),borrow=True)

        print numpy.max(x), numpy.min(x) 
    #return X,labels
    elif 'yale' in dataset:
        print 'Loading Yale Face...'
        faces = cPickle.load(open('./data/yalefaces.pkl'))
        print str(faces.shape)
        print numpy.max(faces),numpy.min(faces)

        x = faces / numpy.max(faces)
        y = numpy.array([])
    else:
        print 'unknown dataset'
        sys.exit() 
    print 'Dataset shape:', x.shape, y.shape
    return x,y
def gradient_updates(cost,params,lr,momentum,gains,iY,classifier):
    min_gain = 0.01

    updates=[]

    #dY = T.grad(cost,classifier.Y)
    n = classifier.n
    no_dims = classifier.no_dims
    n2 = n * n

    L = (classifier.P - classifier.Q) * classifier.num
    L_re = T.tile(L.reshape((n2,1)), (1,no_dims)).reshape((no_dims*n2,))

    at1 = T.tile(classifier.Y, (1,n)).reshape((no_dims*n2,))
    at2 = T.tile(classifier.Y, (n,1)).reshape((no_dims*n2,))
    dY  = L_re * (at1 - at2)
    dY  = T.sum(dY.reshape((n,n,no_dims)),1)

    a = T.gt(dY,0)
    b = T.gt(iY,0)
        
    gains_new = (gains + 0.2) * T.neq(a,b) + (gains * 0.8) * T.eq(a,b)
    gains_new = T.maximum(gains_new,min_gain)

    iY_new = momentum*iY - lr*(gains_new*dY)
    updates.append((gains,gains_new))
    updates.append((iY, iY_new))
    updates.append((classifier.Y, classifier.Y + iY_new))
    #updates.append((iY, momentum*iY - lr*(gains_new*dY)))
    #updates.append((classifier.Y, classifier.Y + iY))
    return updates

def tsne(dataset='mnist_test',no_dims = 2, initial_dims = 50, perplexity = 20.0):
    """
    :type dataset: strin, classifierg
    :param dataset: 

    :type no_dims: int
    :param no_dims: number of dimension in low-dimensional space

    :type initial_dims: int
    :param initial_dims: dimension of data input to tsne after PCA preprocessing
    
    :type perplexity: float
    :param perplexity: perplexity of P distribution    

    """
    X,labels = load_data(dataset)

    print 'Initializing classifier...'
    classifier = tSNE(X,no_dims,initial_dims,perplexity)

    print 'Building model...'
    cost = classifier.KL_cost()
 
    lr = theano.shared(numpy.cast[theano.config.floatX](500.0))
    momentum = theano.shared(numpy.cast[theano.config.floatX](0.5))
     
    gains = theano.shared(numpy.ones((classifier.n,no_dims),dtype=theano.config.floatX))
    iY = theano.shared(classifier.Y.get_value()*0)

    train_model = theano.function(inputs=[],outputs=cost,updates=gradient_updates(cost,classifier.params,lr,momentum,gains,iY,classifier))

    cost_model = theano.function(inputs=[],outputs=cost)
    dY_model   = theano.function(inputs=[],outputs=T.grad(cost,classifier.Y))

    cost_start = cost_model()
    print "Start with Cost", cost_start
    
    dY_start = dY_model()
    print "var dY", numpy.var(dY_start / 4)

    print 'Training... ...'
    start_time = timeit.default_timer()

    epoch = 0
    n_epochs = 1000
    while (epoch < n_epochs):
        
        # change momentum
        if epoch == 20:
            momentum.set_value(numpy.cast[theano.config.floatX](0.8))
       
        cost_e = train_model()
        # normalize Y to center at origion of the low dimensional space
        Y_e = classifier.Y.get_value()
        Y_e = Y_e - numpy.mean(Y_e,0)
        classifier.Y.set_value(Y_e)

        if (epoch + 1) % 10 == 0:
            print 'Epoch:', epoch+1, 'Cost: ', cost_e
       
        # Stop early exaggeration 
        if epoch == 100:
            P_e = classifier.P.get_value()
            P_e = P_e / 4
            classifier.P.set_value(P_e)

        epoch += 1

    end_time = timeit.default_timer()

    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    
    Y = classifier.Y.get_value()

    fig = pylab.figure()
    pylab.scatter(Y[:,0], Y[:,1], 20, labels)
    pylab.show()
    fig.savefig(dataset+'.png')
    
    #results=[X,Y,labels]
    #cPickle.dump(results,open('../results/'+dataset+str(no_dims)+'.pkl','wb')) 
    
if __name__ == '__main__':
    tsne()
