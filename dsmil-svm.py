
from __future__ import print_function, division

import time

import numpy as np
from misvm.quadprog import quadprog
from misvm.kernel import by_name as kernel_by_name
from misvm.util import spdiag
from misvm.svm import SVM
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing


def normalization(m):
    mu = np.min(m)
    sigma = np.max(m)
    return (m - mu) / (sigma - mu)


def hausdorff(listA, listB):
    ha = []
    hb = []
    ain = np.array(listA).shape[1]  
    bin = np.array(listB).shape[1]

    for j in range(len(listA)):
        for i in range(ain):
            A = np.array(listA[j][i])
            ha.append([])
            for n in range(len(listB)):
                for m in range(bin):
                    B = np.array(listB[n][m])
                    min1 = np.linalg.norm(A - B, ord=2)
                    ha[i].append(min1)
                ha[i] = np.min(ha[i])
    hAB = np.mean(ha)

    return hAB


def tanimoto_coefficient(listA, listB):
    """
    This method implements the cosine tanimoto coefficient metric
    :param p_vec: vector one
    :param q_vec: vector two
    :return: the tanimoto coefficient between vector one and two
    """

    ha = []
    ain = np.array(listA).shape[1]  
    bin = np.array(listB).shape[1]

    for j in range(len(listA)):
        for i in range(ain):
            p_vec = np.array(listA[j][i])
            ha.append([])
            for n in range(len(listB)):
                for m in range(bin):
                    q_vec = np.array(listB[n][m]).T
                    pq = np.dot(p_vec, q_vec)
                    p_square = np.linalg.norm(p_vec)
                    q_square = np.linalg.norm(q_vec)
                    min1 = pq / (p_square + q_square - pq)
                    ha[i].append(min1)
                ha[i] = np.min(ha[i])
    hAB = np.mean(ha)
    return hAB


def ins_to_bag_sim(bags):
    """
    AP Clustering
    """
    bags2 = []
    d = []  
    bcc = []  
    bs = []  

    for m in range(len(bags)):
        a = bags[m]  
        bcc.append([])
        d.append([])
        # print('a_len', len(a))  
        ap = AffinityPropagation().fit(a)
        inss = ap.affinity_matrix_  
        cn = ap.cluster_centers_indices_  
        cc = ap.cluster_centers_  

        if len(cn) != 0:
            bcc[m].append(cc.tolist())  
        else:
            bcc[m].append(a.tolist())

        ins_b = -1 * np.mean(inss, axis=1)  
        if len(set(ins_b)) == 1:  
            ins_to_bag = np.array([1 for v in range(len(ins_b))])
        elif len(ins_b) == 1:  
            ins_to_bag = np.array([1])
        else:
            ins_to_bag = normalization(ins_b)
        # print('ins_to_bag', ins_to_bag)
        d[m] = ins_to_bag
        a = np.array(np.transpose(np.array([d[m], ] * a.shape[1]))) * np.array(a)
        a = np.asmatrix(a)
        bags2.append(a)

    
    for n in range(len(bcc)):
        bs.append([])
        for z in range(len(bcc)):
            if n != z:
                bs1 = hausdorff(bcc[n], bcc[z])
                #bs2 = tanimoto_coefficient(bcc[n], bcc[z])
            else:
                bs1 = 1
                #bs2 = 1
            bs[n] = bs1
            #bs[n] = bs2

    return bags2, bcc, bs

class DSMIL(SVM):
    """
    Normalized set kernel of Gaertner, et al. (2002)
    """

    def __init__(self, *args, **kwargs):
        """
        @param kernel : the desired kernel function; can be linear, quadratic,
                        polynomial, or rbf [default: linear]
                        (by default, no normalization is used; to use averaging
                        or feature space normalization, append either '_av' or
                        '_fs' to the kernel name, as in 'rbf_av')
        @param C : the loss/regularization tradeoff constant [default: 1.0]
        @param scale_C : if True [default], scale C by the number of examples
        @param p : polynomial degree when a 'polynomial' kernel is used
                   [default: 3]
        @param gamma : RBF scale parameter when an 'rbf' kernel is used
                      [default: 1.0]
        @param verbose : print optimization status messages [default: True]
        @param sv_cutoff : the numerical cutoff for an example to be considered
                           a support vector [default: 1e-7]
        """
        # super(NSK, self).__init__(*args, **kwargs)
        super(DSMIL, self).__init__(*args, **kwargs)
        self._bags = None
        self._sv_bags = None
        self._bag_predictions = None

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """

        bags1 = list(map(np.asmatrix, bags))  
        self._y = np.asmatrix(y).reshape((-1, 1))  
        bags2, bcc, bgsim = ins_to_bag_sim(bags1)  
        bagsim = normalization(bgsim)
        S = np.sqrt(np.mat(bagsim)).tolist()
        bags3 = list(np.array(S[0]) * np.array(bags2))
        self._bags = bags3


        if self.scale_C:
            C = self.C / float(len(self._bags))
        else:
            C = self.C

        if self.verbose:
            print('Setup QP...')
        K, H, f, A, b, lb, ub = self._setup_svm(self._bags, self._y, C)

        # Solve QP
        if self.verbose:
            print('Solving QP...')
        self._alphas, self._objective = quadprog(H, f, A, b, lb, ub,
                                                 self.verbose)
        self._compute_separator(K)

    def _compute_separator(self, K):

        self._sv = np.nonzero(self._alphas.flat > self.sv_cutoff)
        self._sv_alphas = self._alphas[self._sv]
        self._sv_bags = [self._bags[i] for i in self._sv[0]]
        self._sv_y = self._y[self._sv]

        n = len(self._sv_bags)
        if n == 0:
            self._b = 0.0
            self._bag_predictions = np.zeros(len(self._bags))
        else:
            _sv_all_K = K[self._sv]
            _sv_K = _sv_all_K.T[self._sv].T
            e = np.matrix(np.ones((n, 1)))
            D = spdiag(self._sv_y)
            self._b = float(e.T * D * e - self._sv_alphas.T * D * _sv_K * e) / n
            self._bag_predictions = np.array(self._b
                                             + self._sv_alphas.T * D * _sv_all_K).reshape((-1,))

    def predict(self, bags):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @return : an array of length n containing real-valued label predictions
                  (threshold at zero to produce binary predictions)
        """
        if self._sv_bags is None or len(self._sv_bags) == 0:
            return np.zeros(len(bags))
        else:
            kernel = kernel_by_name(self.kernel, p=self.p, gamma=self.gamma)
            predict_bags = list(map(np.asmatrix, bags))
            start_sim = time.clock()
            Pbags, bcc, bgsim = ins_to_bag_sim(predict_bags)
            end_sim = time.clock()
            bagsim = normalization(bgsim)
            S = np.sqrt(np.mat(bagsim)).tolist()
            bags3 = list(np.array(S[0]) * np.array(Pbags))
            K = kernel(bags3, self._sv_bags)
            sim_time = end_sim - start_sim
            return np.array(self._b + K * spdiag(self._sv_y) * self._sv_alphas).reshape((-1,)), sim_time
