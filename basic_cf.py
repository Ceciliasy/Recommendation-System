#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic Collaborate Filtering: basic_cf.py
"""
import numpy as np
import scipy.sparse as sparse
import time
from scipy.sparse.linalg import spsolve
import argparse

def load_tuples(filename):
    t0 = time.time()
    rating_tuple = []
    for i, line in enumerate(open(filename, 'r')):
        user, item, count = line.strip().split(' ')
        user = int(float(user))
        item = int(float(item))
        count = float(count)
        rating_tuple.append([user,item,count])
        if i % 100000 == 0:
            print('loaded %i counts...' % i)
    t1 = time.time()
    print('Finished loading tuples in %f seconds' % (t1 - t0))
    return rating_tuple
def load_matrix(rating_tuple, num_users, num_items):
    t0 = time.time()
    counts = sparse.dok_matrix((num_users, num_items), dtype=float)
    for i, r_tuple in enumerate(rating_tuple):
        user, item, count = r_tuple
        if user >= num_users:
            continue
        if item >= num_items:
            continue
        if count != 0:
            counts[user, item] = count
        if i % 100000 == 0:
            print('loaded %i counts...' % i)
    counts = counts.tocsr()
    t1 = time.time()
    print('Finished loading matrix in %f seconds' % (t1 - t0))
    return counts

class CF():
    def __init__(self,ratings, K=200, lambda_u=0.01, lambda_v=0.01, n_iter=1, alpha = 40, checkpoint = None):
        self.K = K
        self.n_iter = n_iter
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.alpha = alpha
        
        self.num_users = int(np.max(np.array(ratings)[:,0]) + 1)
        self.num_products = int(np.max(np.array(ratings)[:,1]) + 1)
        self.ratings = self.load_matrix(ratings)
        if(checkpoint):
            self.U = np.load(checkpoint+"/matrix_u.npy")
            self.V = np.load(checkpoint+"/matrix_v.npy")
        else:
            self.U = np.sqrt(1.0/lambda_u) * np.random.randn(self.num_users,K)
            self.V = np.sqrt(1.0/lambda_v) * np.random.randn(self.num_products,K)
        
    def fit(self):
        for i in range(self.n_iter):
            # Update user vector
            self.U = self.update("user")
            #  Update product vector
            self.V = self.update("product")
    def update(self,target_type):
        t0 = time.time()
        if target_type == "user":
            num = self.num_users
            Y = sparse.csr_matrix(self.V)
            lambda_ = self.lambda_u
        else:
            num = self.num_products
            Y = sparse.csr_matrix(self.U)
            lambda_ = self.lambda_v
        num_fixed = Y.shape[0]
        YTY = Y.T.dot(Y) 
        # accumulate YtCuY + regularization*I in A
        A = YTY + lambda_ * sparse.eye(self.K)
        # accumulate YtCuPu in b
        b = np.zeros(self.K)
        # placeholder for solution
        X = np.zeros((num, self.K))
        for i in range(num):
            Ri = self.ratings[i].T if target_type =="user" else self.ratings[:i].T
            X[i] = self.solve_equation(A,b,Y,Ri)
        t1 = time.time()
        print('Finished update '+target_type+' in %f seconds' % (t1 - t0))
        return X
            
    def solve_equation(self,A,b,Y,Ri):
        # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
        # YtCuY + regularization * I = YtY + regularization * I + Yt(Cu-I)

        # accumulate YtCuY + regularization*I in A
        # accumulate YtCuPu in b
        #import ipdb;ipdb.set_trace()
        for j in Ri.indices:
            factor = Y[j]
            r = Ri[j].data[0]
            confidence = 1+self.alpha*r
            if r > 0:     
                b += confidence * factor
                assert (factor.T*factor).shape != (1,1)
                A += (confidence - 1) * factor.T*factor 
        x = spsolve(A, b.T)
        return x
    
    def predict_rate(self,user_id, product_id):
        if(user_id >= self.num_users):
            u_vector = np.zeros_like(self.U[0])
        else:
            u_vector = self.U[user_id]
        if(product_id >= self.num_products):
            v_vector = np.zeros_like(self.V[0])
        else:
            v_vector = self.V[product_id]
        return u_vector.dot(v_vector)
    
    def MSE(self,test):
        error = []
        for user,product,target_rate in test:
            error.append((self.predict_rate(user,product)-target_rate)**2)
        return np.mean(error)
    
    def load_matrix(self,rating_tuple):
        t0 = time.time()
        counts = sparse.dok_matrix((self.num_users, self.num_products), dtype=float)
        for i, r_tuple in enumerate(rating_tuple):
            user, item, count = r_tuple
            if user >= self.num_users:
                continue
            if item >= self.num_products:
                continue
            if count != 0:
                counts[user, item] = count
            if i % 100000 == 0:
                print('loaded %i counts...' % i)
        counts = counts.tocsr()
        t1 = time.time()
        print('Finished loading matrix in %f seconds' % (t1 - t0))
        return counts
    def save_UV(self, dir_path):
        file_u = dir_path + "/matrix_u.npy"
        file_v = dir_path + "/matrix_v.npy"
        np.save(file_u, self.U)
        np.save(file_v, self.V)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training a Collaborate Filtering model')
    parser.add_argument('--source_dir', type=str, action='store', help='source directory', default="./data")
    parser.add_argument('--n_iter', type=int, action='store', help='number of iterations', default=10)
    parser.add_argument('--save_dir', type=str, action='store', help='save directory', default="./cf_result")

    args = parser.parse_args()
    print(args)

    train_file = args.source_dir+"/train_tuples.txt"
    val_file = args.source_dir+"/val_tuples.txt"
    test_file = args.source_dir+"/test_tuples.txt"

    train_tuple = load_tuples(train_file)
    test_tuple = load_tuples(test_file)
    val_tuple = load_tuples(val_file)

    num_users = int(np.max(np.array(train_tuple)[:,0]))
    num_items = int(np.max(np.array(train_tuple)[:,1]))
    # train_counts = load_matrix(train_tuple,num_users,num_items)
    # test_counts = load_matrix(test_tuple,num_users,num_items)
    # val_counts = load_matrix(val_tuple,num_users,num_items)

    n_iter = 10
    cf = CF(train_tuple)
    for i in range(n_iter):
        cf.fit()
        train_error = cf.MSE(train_tuples)
        val_error = cf.MSE(val_tuples)
        print("iter: {}, train error: {}, validation error: {} ".format(i,train_error,val_error))
        cf.save_UV(".")
        print("save UV to {}".format(save_dir))