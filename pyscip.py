# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:01:09 2022

@author: Leo
"""
import sys
import numpy as np
from pyscipopt import Model,quicksum
from pyscipopt.scip import Expr, Term
import time
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool
import itertools


class MIOCP:
   def __init__(self,A,B,c,R0,c0,Rf,cf,Q0,q0,Qf,qf,Lu,u_min,u_max,NLrhs=None,constraint=None):
      self.A = np.array(A, dtype=np.float64)
      self.B = np.array(B, dtype=np.float64)
      self.c = np.array(c, dtype=np.float64)
      self.R0 = np.array(R0, dtype=np.float64)
      self.c0 = np.array(c0, dtype=np.float64)
      self.Rf = np.array(Rf, dtype=np.float64)
      self.cf = np.array(cf, dtype=np.float64)
      self.Q0 = np.array(Q0, dtype=np.float64)
      self.q0 = np.array(q0, dtype=np.float64)
      self.Qf = np.array(Qf, dtype=np.float64)
      self.qf = np.array(qf, dtype=np.float64)
      self.Lu = np.array(Lu)
      self.u_min = np.array(u_min, dtype=np.float64)
      self.u_max = np.array(u_max, dtype=np.float64)
      self.NLrhs = NLrhs
      self.constraint = constraint

plot_iterations = True
compute_obj_val = True
plot_result = True
plot_err = True

#plots of convergence rate
plot_objvals=True
#plot convergence rate only after full cycle of solving VCPs
plot_objvalsbyfulldoms=False
#plot ratio of value change
divsofsteps=False

#writedoc needs certain folders!
writedoc=False
multiprocessed=True

#Optionally allow for controls to be continuous
mixedintegercontrols=True
#Optionally use additional variables to implement restrictions
usingadditionalvars=False

#Stopping Criteria
threshold_x = 1e-2
threshold_lam =1e-2

solver_time=1500
overall_time=2000

number_of_domains = 1   

#weights
gamma=1
epsilon = 0.5

#precision
number_of_time_steps=50

testcase=1

#strictness on the restrictions
feas=1e-6 #1e-6 standard


#set up virtual control problem
def get_vcp(miocp, xk, uk, delta_t,k,K,fix_u=False):
    n = xk.shape[0]
    m = uk.shape[0]
    number_of_steps = xk.shape[1] - 1
    model=Model('vcp in %d' %k)
    x = np.empty((n, number_of_steps+1), dtype=object)
    for i in range(n):
        for j in range(number_of_steps+1):
            x[i, j] = model.addVar(name="x(%s,%s)" %(i,j), lb=None,ub=None,obj=xk[i,j])

    u = np.empty((m, number_of_steps), dtype=object)
    if mixedintegercontrols:
        if fix_u:
            for i in range(m):
                for j in range(number_of_steps):
                    u[i,j] = model.addVar(vtype='I',name="u(%s,%s)" %(i,j),lb=uk[i,j],ub=uk[i,j],obj=uk[i,j])
        else:
            for i in range(m):
                for j in range(number_of_steps):       
                    u[i,j] = model.addVar(vtype='I',name="u(%s,%s)" %(i,j),lb=miocp.u_min[i],ub=miocp.u_max[i],obj=uk[i,j])
    else:
        if fix_u:
            for i in range(m):
                for j in range(number_of_steps):
                    u[i,j] = model.addVar(vtype='C',name="u(%s,%s)" %(i,j),lb=uk[i,j],ub=uk[i,j],obj=uk[i,j])
        else:
            for i in range(m):
                for j in range(number_of_steps):       
                    u[i,j] = model.addVar(vtype='C',name="u(%s,%s)" %(i,j),lb=miocp.u_min[i],ub=miocp.u_max[i],obj=uk[i,j])
    
    if miocp.constraint is not None:
        miocp.constraint(model,x,u)
    
    if miocp.NLrhs is None:
        a, b = get_butcher_tableau()
        number_of_stages = len(b)
        stages = np.empty((number_of_stages, number_of_steps),object)
        for j in range(number_of_stages):
            for step in range(number_of_steps):             
                expr_1=np.zeros((n,1))
                for  l in range(j):
                    expr_1 = expr_1 + expr_sum(expr_1,  multiply_matrix_with_array_of_expr(model, a[j][l], stages[l][step]))
                expr_2_factorexpr1 = delta_t*expr_1
                expr_2_xplusexpr1= x[:n, step].reshape(-1, 1) + expr_2_factorexpr1
                expr_2_Atimessum=multiply_matrix_with_array_of_expr(model, miocp.A, expr_2_xplusexpr1)
                expr_2_Btimesum = multiply_matrix_with_array_of_expr(model, miocp.B, u[:m,step].reshape(-1, 1))
                expr_2_sum_of_AandB = expr_sum(expr_2_Atimessum, expr_2_Btimesum)
                expr_2=expr_sum(expr_2_sum_of_AandB, miocp.c)
                stages[j][step] = expr_2
        for step in range(number_of_steps):      
            expr_3 = sum (b[j] * stages[j][step] for j in range(number_of_stages))
            for i in range(n):
                model.addCons(expr_3[i,0] == (x[i, step + 1] - x[i, step]) / delta_t)
    else:
        if miocp.NLrhs is not None:
            a, b = get_butcher_tableau()
            number_of_stages = len(b)
            stages = [None] * number_of_stages
            x_tmp=np.empty((number_of_stages,n,number_of_steps+1),dtype=object)
            
            if usingadditionalvars:
                for j in range(number_of_stages):
                    for i in range(n):
                        for step in range(number_of_steps):
                            sum1=0
                            for l in range(j):
                                sum1 = expr_sum(sum1,multiply_matrix_with_array_of_expr(model, a[j][l], stages[l][i][step]))
                            x_tmp[j][i,step]=x[i][step] + delta_t * sum1
                    stages[j] = miocp.NLrhs(model, x_tmp[j], u)
                for i in range(n):
                    for step in range(number_of_steps):
                        sumbtimesstages=0
                        for j in range(number_of_stages):
                            sumbtimesstages = expr_sum(sumbtimesstages,  multiply_matrix_with_array_of_expr(model, b[j,0], stages[j][i][step]))
                        model.addCons((x[i][step + 1] - x[i][step])  == sumbtimesstages * delta_t)
            else:
                for j in range(number_of_stages):
                    x_tmp[j] = np.empty((n, number_of_steps+1), dtype=object)
                    for i in range(n):
                        for step in range(number_of_steps+1):
                            x_tmp[j,i, step] = model.addVar(vtype = "C", name = f"rhs_tmp_{j}_{i}_{step}",lb=None)
                    for i in range(n):
                        for step in range(number_of_steps):
                            sum1=0
                            for l in range(j):
                                sum1 = expr_sum(sum1,multiply_matrix_with_array_of_expr(model, a[j][l], stages[l][i][step]))
                            Cons1=x[i][step] + delta_t * sum1
                            model.addCons(x_tmp[j][i][step] == Cons1)
                    stages[j] = miocp.NLrhs(model, x_tmp[j], u)
                for i in range(n):
                    for step in range(number_of_steps):
                        sumbtimesstages=0
                        for j in range(number_of_stages):
                            sumbtimesstages = expr_sum(sumbtimesstages,  multiply_matrix_with_array_of_expr(model, b[j,0], stages[j][i][step]))
                        model.addCons((x[i][step + 1] - x[i][step])  == sumbtimesstages * delta_t)  
    
    if k == 0:
        product=multiply_matrix_with_array_of_expr(model, miocp.R0, x[:n, 0].reshape(-1,1))
        for i in range(len(miocp.c0)):
            model.addCons(product[i,0] == miocp.c0[i])  
    if k == K:
        product=multiply_matrix_with_array_of_expr(model, miocp.Rf, x[:n, 0].reshape(-1,1))
        for i in range(len(miocp.cf)):
            model.addCons(product[i,0] == miocp.cf[i])    
    model.setRealParam("limits/time", solver_time)
    model.setIntParam('display/verblevel',5)
    model.setParam('numerics/feastol', feas) #genuinely helps
    
    #some SCIP parameters to play around with
    # model.setRealParam('memory/savefac', 0.5) #standard 0.81
    # model.setBoolParam('misc/avoidmemout',False)
    # model.setParam('presolving/maxrounds', 5)
    # model.setRealParam('separating/cgmip/memorylimit',1.79769313486232e+308)
        # model.setBoolParam('branching/relpscost/filtercandssym',True)
    # if testcase==2 or testcase==3:
    #     model.setCharParam('constraints/nonlinear/linearizeheursol','i')
    # model.setCharParam('constraints/nonlinear/checkvarlocks','b')
    # model.setBoolParam('constraints/setppc/cliquelifting',True)
    # model.setBoolParam('reoptimization/storevarhistory',True)
    # model.setIntParam('misc/usesymmetry',7)
    # model.setBoolParam('constraints/and/delaysepa',False)
    # model.setBoolParam('history/allowtransfer',True)
    # model.setBoolParam('heuristics/completesol/beforepresol',False)
    # model.setBoolParam('constraints/indicator/addcouplingcons',True)
    # model.setIntParam('heuristics/crossover/nusedsols',10)
    return model

#solver on VCP
def solve_vcp(model,miocp,gamma,phi,xk,uk,lam,delta_t,k):
    K=phi.shape[1]
    n = xk.shape[0]
    m = uk.shape[0]
    number_of_steps = xk.shape[1] - 1
    variable_list=model.getVars()
    x = np.empty((n, number_of_steps+1), dtype=object)
    u = np.empty((m, number_of_steps), dtype=object)
    for var in variable_list:
        if 'u' in var.name:
            u[int(var.name.split('(')[1].split(',')[0]), int(var.name.split('(')[1].split(',')[1][:-1])]=var
    for var in variable_list:
        if 'x' in var.name:
            x[int(var.name.split('(')[1].split(',')[0]), int(var.name.split('(')[1].split(',')[1][:-1])]=var
    objvar=np.empty((K+1),dtype=object)
    if K == 0:
        sum1=quicksum(miocp.Q0[i,j] * x[i,0] * x[j,0] for i in range(n) for j in range(n)) / 2
        sum2=miocp.q0.T @ x[:n,0]
        sum3=quicksum(miocp.Qf[i,j] * x[i,number_of_steps] * x[j,number_of_steps] for i in range(n) for j in range(n)) / 2
        sum4=miocp.qf.T @ x[:n,number_of_steps]
        sum5=0
        if len(miocp.Lu)==1:
            for step in range(number_of_steps):
                for j in range(m):
                    for i in range(m):
                        sum5= sum5 + miocp.Lu * u[i,step] * u[j,step]
        else:
            for step in range(number_of_steps):
                for j in range(m):
                    for i in range(m):
                        sum5= sum5 +miocp.Lu[i,j] * u[i,step] * u[j,step]        
        if isinstance(sum5, Expr):
            sum5=sum5 * (delta_t/2)
        else:
            sum5=sum5[0] * (delta_t/2)
        obj = sum1 + sum2 + sum3 + sum4 + sum5
        objvar[k] = model.addVar(name="objvar%d"%k, vtype= "C",lb=None, ub=None)
        model.setObjective(objvar[k], "minimize")
        model.addCons(objvar[k] >= obj)
        
    elif k == 0:
        sum1=quicksum(miocp.Q0[i,j] * x[i,0] * x[j,0] for i in range(n) for j in range(n)) / 2
        sum2=miocp.q0.T @ x[:n,0]
        sum3=quicksum((x[i,number_of_steps] - phi[1,k,i])**2 for i in range(n)) / (2 * gamma)        
        sum4=0
        if len(miocp.Lu)==1:
            for step in range(number_of_steps):
                for j in range(m):
                    for i in range(m):
                        sum4 = sum4 + miocp.Lu * u[i,step] * u[j,step]
        else:
            for step in range(number_of_steps):
                for j in range(m):
                    for i in range(m):
                        sum4= sum4 + miocp.Lu[i,j] * u[i,step] * u[j,step]
        if isinstance(sum4, Expr):
            sum4=sum4 * (delta_t/2)
        else:
            sum4=sum4[0] * (delta_t/2)
        obj=sum1 + sum2 + sum3 + sum4
        objvar[k] = model.addVar(name="objvar%d"%k , vtype= "C",lb=None, ub=None)
        model.setObjective(objvar[k], "minimize")
        model.addCons(objvar[k] >= obj)
        
    elif k == K:
        sum1=quicksum((x[i,0] - phi[0,k-1,i])**2 for i in range(n)) / (2 * gamma)
        sum2=quicksum(miocp.Qf[i,j] * x[i,number_of_steps] * x[j,number_of_steps] for i in range(n) for j in range(n)) / 2
        sum3=miocp.qf.T @ x[:n,number_of_steps]
        sum4=0
        if len(miocp.Lu)==1:
            for step in range(number_of_steps):
                for j in range(m):
                    for i in range(m):
                        sum4 = sum4 +miocp.Lu * u[i,step] * u[j,step]
        else:
            for step in range(number_of_steps):
                for j in range(m):
                    for i in range(m):
                        sum4= sum4 + miocp.Lu[i,j] * u[i,step] * u[j,step]
        if isinstance(sum4, Expr):
            sum4=sum4 * (delta_t/2)
        else:
            sum4=sum4[0] * (delta_t/2)
        obj=sum1 + sum2 + sum3 + sum4
        objvar[k] = model.addVar(name="objvar%s" %k, vtype= "C",lb=None,ub=None)
        model.setObjective(objvar[k], "minimize")
        model.addCons(objvar[k] >= obj)
    
    else:
        sum1=quicksum((x[i,0] - phi[0,k-1,i])**2 for i in range(n)) / (2 * gamma)
        sum2=quicksum((x[i,number_of_steps] - phi[1,k,i])**2 for i in range(n)) / (2 * gamma)
        sum3=0
        if len(miocp.Lu)==1:
            for step in range(number_of_steps):
                for j in range(m):
                    for i in range(m):
                        sum3+=miocp.Lu * u[i,step] * u[j,step]
        else:
            for step in range(number_of_steps):
                for j in range(m):
                    for i in range(m):
                        sum3+=miocp.Lu[i,j] * u[i,step] * u[j,step]
        if isinstance(sum3, Expr):
            sum3=sum3 * (delta_t/2)
        else:
            sum3=sum3[0] * (delta_t/2)       
        obj=sum1 + sum2 + sum3 
        objvar[k] = model.addVar(name="objvar%s" %k, vtype= "C", ub=None) #lb=0
        model.setObjective(objvar[k], "minimize")
        model.addCons(objvar[k] >= obj)
  
    # model.writeProblem("model_%s.cip" %k)
    model.hideOutput()
    model.optimize()

    if model.getStatus() != "optimal":
        print('NOT OPTIMAL')
        print(model.getStatus())
        raise ValueError("Could not solve virtual control problem")
    xk = np.empty((n, number_of_steps+1), dtype=object)
    for i in range(n):
        for j in range(number_of_steps+1):
            xk[i, j] = model.getVal(x[i, j])
    uk = np.empty((m, number_of_steps), dtype=object)        
    for i in range(m):
        for j in range(number_of_steps):
            uk[i, j] = model.getVal(u[i, j])
    
    if k!=0:
        lam[0, k-1, :] = (xk[:, 0] - phi[0, k-1, :]) / gamma
    if k !=K:
        lam[1, k, :] = -(xk[:, -1] - phi[1, k, :]) / gamma
    
    # print(model.getObjVal())
    return xk,uk,lam

def add_iteration_errors(x_errors, lam_errors, x, lam):
    K = lam.shape[1]
    n = lam.shape[2]
    x_error = np.zeros((K, n))
    for k in range(K):
        x_error[k,:] = np.abs(x[k][:,-1] - x[k+1][:,0])
    lam_error = np.abs(lam[1,:,:] - lam[0,:,:])
    x_errors.append(x_error)
    lam_errors.append(lam_error)

def get_max_errors(x, lam):
    K = lam.shape[1]
    max_error_x = 0
    #there are only K-1 points in which the states coincide
    for k in range(K-1):
        max_error_x = max(max_error_x, np.max(np.abs(x[k][:, -1] - x[k+1][:, 0])))
    max_error_lam = np.max(np.abs(lam[1,:,:] - lam[0,:,:])) if K > 0 else 0    
    return (max_error_x, max_error_lam)

def plot_errors(x_errors, lam_errors):
    number_of_iterations = len(x_errors)
    K = x_errors[0].shape[0]
    if K == 0:
        return None
    p_iter = range(number_of_iterations)
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration")
    p_x = [np.max(x) for x in x_errors]
    ax.plot(p_iter, p_x, label='x')
    p_lam = [np.max(lam) for lam in lam_errors]
    ax.plot(p_iter, p_lam, label='λ')
    ax.set_title('Errors')
    ax.legend(loc='upper right')
    return fig

def plot_iteration(ts, delta_t, x, u):
    plt.style.use('ggplot')
    n = x[0].shape[0]
    m = u[0].shape[0]
    K = len(ts) - 2
    fig, ax = plt.subplots()
    for k in range(K+1):
        number_of_steps = x[k].shape[1] - 1
        p_t = [ts[k] + delta_t * s for s in range(number_of_steps+1)]
        for i in range(n):
            p_x = x[k][i,:]
            label = f"x{i}" if k == 0 else ""
            ax.plot(p_t, p_x, label=label,ls='--', color=f'C{i}')
        for i in range(m):
            p_u = np.zeros(number_of_steps+1)
            p_u[:number_of_steps] = u[k][i,:]
            p_u[number_of_steps] = u[k][i,number_of_steps-1]
            if k==0:
                if m==1:
                    label='u'
                else:
                    label='u'+str(i)
            else:
                label=''
            ax.step(p_t, p_u, label=label,ls='solid' ,color=f'C{n+i}')
    ax.legend(loc='right')
    plt.gcf()
    plt.show()
    return fig

def plotobjvals(iter,objval):
    if iter==1:
        return
    p_iter=list(range(1, iter + 1))
    #plot ratio of change of value?
    if divsofsteps:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the values against the iteration numbers in the first subplot (normal plot)
    axs[0].plot(p_iter, objval)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Normal plot')
    
    # Plot the values against the iteration numbers in the second subplot (logarithmic plot)
    axs[1].semilogy(p_iter, objval)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Value (log scale)')
    axs[1].set_title('Logarithmic plot')
    
    if divsofsteps:
        if iter>2:
            #Plot the ratios
            ratios = [(objval[i] - objval[i+1]) / (objval[i-1] - objval[i]) for i in range(1, len(objval)-1)]
            axs[2].plot(p_iter[1:-1], ratios)
            axs[2].set_xlabel('Iteration')
            axs[2].set_ylabel('Ratio')
            axs[2].set_title('Visual plot of convergence')
    plt.show()
    return
    
    
def algo(miocp, ts, gamma: float, epsilon: float, delta_t: float, phi: np.ndarray = None):
    n = len(miocp.q0)
    m = len(miocp.u_min) 
    K = len(ts) - 2
    x = np.empty(K+1, dtype=object)
    u = np.empty(K+1, dtype=object)
    for i in range(K+1):
      x[i] = np.empty(n, dtype=np.float64)
    x = np.empty(K+1, dtype=object)
    for i in range(K+1):
        u[i] = np.empty(m, dtype=np.float64)
    lam = np.empty((2, K, n))
    x_errors = []
    lam_errors = []
    objvals = []
    for k in range(K+1):
        tk = ts[k]   
        tkplusone = ts[k+1]
        number_of_steps = int(np.ceil((tkplusone - tk) / delta_t))
        ts[k+1] = tk + number_of_steps * delta_t
        x[k] = np.zeros((n, number_of_steps+1))
        u[k] = np.zeros((m, number_of_steps))
    if phi.shape == (0,0,0):
        phi = np.zeros((2, K, n))
    models = [None]*(K+1)
    for k in range(K+1):
        models[k] = get_vcp(miocp, x[k], u[k], delta_t, k, K)
    
    #iter
    print("{:<5s} {:<8s} {:<10s} {:<10s}".format("Iter", "Time", "Error x", "Error lambda"))
    iter = 1
    time_all = 0
    
    while True:
        start_time = time.time()
        if multiprocessed:    
            # Make the Pool of workers
            pool = Pool(4) #number of physical cores
            res = pool.starmap(solve_vcp, zip(models, itertools.repeat(miocp), itertools.repeat(gamma),itertools.repeat(phi), x, u, itertools.repeat(lam),itertools.repeat(delta_t) ,range(K+1)))
            # # rearrange the x and u and lambda
            resarr=np.array(res,dtype=object).reshape(K+1,3)
            x=resarr[:,0]
            u=resarr[:,1]
            lam=resarr[K,2]
           
        else:
            for k in range(K+1):
                x[k],u[k],lam=solve_vcp(models[k], miocp, gamma, phi, x[k], u[k], lam, delta_t, k)

        for k in range(K):
            phi[0,k,:] = (1 - epsilon) * (x[k][:,-1] - gamma * lam[1,k,:]) + epsilon * (x[k+1][:,0] - gamma * lam[0,k,:])
            phi[1,k,:] = (1 - epsilon) * (x[k+1][:,0] + gamma * lam[0,k,:]) + epsilon * (x[k][:,-1] + gamma * lam[1,k,:])
        
        add_iteration_errors(x_errors, lam_errors, x, lam)
        max_error_x, max_error_lam = get_max_errors(x, lam)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("{:<5.0f} {:<8.3f} {:<10.5f} {:<10.5f}".format(iter, elapsed_time, max_error_x, max_error_lam))
        if plot_iterations:
            plot_iteration(ts, delta_t, x, u)
        if plot_objvals:
            objvals.append(get_objective_value(miocp,x,u,delta_t)[2])
            print('Value in Iteration %s: ' %iter,objvals[iter-1])
            plotobjvals(iter,objvals)
        if plot_objvalsbyfulldoms:
            if iter%number_of_domains == 0:
                objvals.append(get_objective_value(miocp,x,u,delta_t)[2])
                print('Value in Iteration %s: ' %(iter//number_of_domains),objvals[iter//number_of_domains-1])
                plotobjvals(iter//number_of_domains,objvals)
            
        if max_error_lam <= threshold_lam and max_error_x <= threshold_x:
            print("Continuous solution found!")
            print(f'Iterations: {iter}')
            print()
            return x, u, x_errors, lam_errors,objvals

    
        for k in range(K+1):
            models[k].freeTransform()
            for var in models[k].getVars():
                if 'objvar' in var.name:
                    models[k].delVar(var)
            conlist=models[k].getConss()
            conlen=len(conlist)
            models[k].delCons(conlist[conlen-1])        
            # models[k].writeProblem('freetransformed iteration %d in k = %d' %(iter,k))
        iter += 1
        time_all += elapsed_time
        if time_all >= overall_time:
            raise Exception("OVERALL TIME LIMIT")
            

#expansion of variable class from PySCIPOpt
def multiply_matrix_with_array_of_expr(model,matr,exprarr):
    if isinstance(matr, float) or isinstance(matr, int) or isinstance(exprarr, Expr) :
        expr=matr*exprarr
        return expr
    m= len(matr)
    n = len(matr[0])
    n_ = len(exprarr)
    l = n_//n
    if  not isinstance(n_ % n, int):
        raise ValueError("The number of columns in A must match the number of rows in B")
    expr=np.empty((m, l), dtype=object)
    for i in range(m):
        for j in range(l):
            expr[i, j] = Expr({Term():0.0})
    if l==1:
        for i in range(m):
            for j in range(n):
                summand = matr[i,j] * exprarr[j]
                expr[i]= expr[i]+ summand
        return expr
    for i in range(m):
        for j in range(l):
            for k in range(n):
                summand = matr[i,k] * exprarr[k,j]
                expr[i, j] = expr[i, j]+ summand
    return expr

#expansion of Expr class from PySCIPOpt
def expr_sum(expr1,expr2):
    if isinstance(expr1, int) or isinstance(expr1, float) or (isinstance(expr1, Expr) and isinstance(expr2, Expr)):
        return expr1+expr2
    n_o_rows1 = len(expr1)
    n_o_rows2 = len(expr2)
    if n_o_rows1 is not n_o_rows2:
        raise ValueError('The arrays are of different dimensions')
    n_o_columns1 = len(expr1[0])
    expr=np.empty((n_o_rows1, n_o_columns1), dtype=object)
    if n_o_columns1 != 1 and n_o_rows1 !=1:
        for i in range(n_o_rows1):
            for j in range(n_o_rows1):
                expr[i,j]=expr1[i,j]+expr2[i,j]
        return expr   
    elif n_o_columns1 == 1:
        for j in range(n_o_rows1):
            expr[j]=expr1[j]+expr2[j]
        return expr
    elif n_o_rows1 == 1:
        for j in range(n_o_columns1):
            expr[j]=expr1[j]+expr2[j]
        return expr
              
def get_objective_value(miocp,x,u,delta_t):
    K = len(x) - 1
    n = x[0].shape[0]
    if compute_obj_val:
        u_all = np.hstack(u)
        number_of_steps = u_all.shape[1]
        x_all = np.zeros((n, number_of_steps+1))
        index = number_of_steps+1
        for k in range(K, -1, -1):
            s = x[k].shape[1]
            x_all[:, (index-s):index] = x[k]
            index -= s - 1
        model_all = get_vcp(miocp, x_all, u_all, delta_t, 0, 0, True)
        x_all,u_all,lam = solve_vcp(model_all, miocp, 1, np.empty((0, 0, 0)), x_all, u_all, np.empty((0, 0, 0)), delta_t, 0)
        val=model_all.getObjVal()
        if testcase == 2 or testcase == 3:
            val += 0.01 * 0.01

    return x_all, u_all, val

def get_butcher_tableau():
    a = np.array([[0, 0, 0, 0], [1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1, 0]])
    b = np.array([[1/6], [1/3], [1/3], [1/6]])
    return (a,b)

def save_data_file(filename, t0, delta_t, x, u):
    n = x.shape[0]
    m = u.shape[0]
    number_of_steps = x.shape[1] - 1
    data = np.empty((number_of_steps+2, 1+n+m),dtype=object)
    data[0, 0] = "t"
    data[1:, 0] = t0 + delta_t * np.arange(number_of_steps + 1)
    for i in range(n):
        data[0, 1 + i] = f"x{i + 1}"
        data[1:, 1 + i] = x[i, :]
    for i in range(m):
        data[0, 1 + n + i] = f"u{i + 1}"
        data[1:, 1 + n + i] = np.concatenate((u[i, :], [u[i, -1]]))
    np.savetxt(f"{filename}.dat", data, delimiter=" ", fmt="%s")


def save_error_data_file(filename, x_errors, lam_errors):
    number_of_iterations = len(x_errors)
    data = np.empty((number_of_iterations+1, 3), dtype=object)
    data[0, 0] = "Iteration"
    data[1:, 0] = np.arange(1, number_of_iterations+1)
    data[0, 1] = "x"
    data[1:, 1] = np.array([np.max(x_error) for x_error in x_errors])
    data[0, 2] = "lambda"
    data[1:, 2] = np.array([np.max(lam_error) for lam_error in lam_errors])
    np.savetxt(filename + ".dat", data, delimiter=' ', fmt='%s')

def run_miocp(miocp, t_max, phi=np.empty((0, 0, 0))):
    try:
        #ts is an array of the split points between the domains, incl start and end
        ts = np.linspace(0, t_max, number_of_domains+1)
        delta_t = t_max / number_of_time_steps
        start = time.time()
        x, u, x_errors, lam_errors,objvals = algo(miocp, ts, gamma, epsilon, delta_t, phi)
        end = time.time()
        print("Iteration time: ", end - start)
        if compute_obj_val:
            start = time.time()
            x_all,u_all, val = get_objective_value(miocp, x, u, delta_t)
            print('objective value =',val)
            end = time.time()
            if plot_result:
                if testcase == 3:
                    u_all = u_all[0, :] / 3 + 2 * u_all[1, :] / 3 + u_all[2, :]
                u_all=np.reshape(u_all,(1, number_of_time_steps)) 
                fig= plot_iteration([ts[0], ts[-1]], delta_t, [x_all], [u_all])
                filename = 'testcase'+str(testcase)+'/plot_res/'+'n_of_doms='+str(number_of_domains)+'_gamma='+str(gamma)+'_eps ='+str(epsilon)+'_thrshld_x='+str(threshold_x)+'_thrshld_lam='+str(threshold_lam)
                save_data_file(filename, ts[0], delta_t, x_all, u_all)
                fig.savefig(filename + ".pdf")
            print("Overall time: ", end - start)
        if plot_err:
            fig=plot_errors(x_errors, lam_errors)
            filename = 'testcase'+str(testcase)+'/plot_err/'+'n_of_doms='+str(number_of_domains)+'_gamma='+str(gamma)+'_eps ='+str(epsilon)+'_thrshld_x='+str(threshold_x)+'_thrshld_lam='+str(threshold_lam)
            save_error_data_file(filename, x_errors, lam_errors)
            fig.savefig(filename + ".pdf")
        if plot_objvals:
            None
    except Exception as e:
        print(e)
        print("")

def test1():
   t_max=1
   A = np.array([[0, 2], [-1, 1]], dtype=np.float64)
   B = np.array([[0],[-1]] , dtype = np.float64)
   c = np.array([0, 0], dtype=np.float64)
   R0 = np.array([[1, 0], [0, 1]], dtype=np.float64)
   c0 = np.array([-2, 1], dtype=np.float64)
   Rf = np.array([[0, 0], [0, 0]], dtype=np.float64)
   cf = np.array([0, 0], dtype=np.float64)
   Q0 = np.array([[0, 0], [0, 0]], dtype=np.float64)
   q0 = np.array([0, 0], dtype=np.float64)
   Qf = np.array([[2, 0], [0, 2]], dtype=np.float64)
   qf = np.array([0, 0], dtype=np.float64)
   Lu = np.array([1e-2], dtype=float)
   u_min = np.array([0], dtype=np.float64)
   u_max = np.array([4], dtype=np.float64)
   NLrhs = None
   constraint = None
   miocp = MIOCP(A, B, c, R0, c0, Rf, cf, Q0, q0, Qf, qf, Lu, u_min, u_max, NLrhs, constraint)
   run_miocp(miocp, t_max)

#Fuller's IVP
def test2():
    t_max=1
    A = np.array([[0, 0], [0, 0]], dtype=np.float64)
    B = np.array([[0],[0]] , dtype = np.float64)
    c = np.array([0, 0], dtype=np.float64)
    R0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    c0 = np.array([0.01, 0, 0], dtype=np.float64)
    Rf = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    cf = np.array([0, 0, 0], dtype=np.float64)
    Q0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    q0 = np.array([0, 0, 0], dtype=np.float64)
    Qf = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.float64)
    qf = np.array([-0.02, 0, 1], dtype=np.float64)
    Lu = np.array([0], dtype=float)
    u_min = np.array([0], dtype=np.float64)
    u_max = np.array([1], dtype=np.float64)

    def NLrhs(model, x, u):
        n = len(x)
        number_of_steps = np.shape(u)[1]
        rhs = np.empty((n,number_of_steps), dtype=object)  
        #define x dot respectively, prob via constraints
        rhs[0] = x[1, :number_of_steps]
        # rhs[1] = np.ones(number_of_steps) - 2 * u[0,:]
        rhs[1] = np.ones(number_of_steps) - 2 * u[0,:]
        # print('landed in nlrhs',rhs[1])
        rhs[2] = x[0, :number_of_steps]**2
        return rhs
    constraint = None
    miocp = MIOCP(A, B, c, R0, c0, Rf, cf, Q0, q0, Qf, qf, Lu, u_min, u_max, NLrhs, constraint)
    run_miocp(miocp, t_max)

#Fuller's Multi-Mode IVP
def test3():
    t_max=1
    A = np.array([[0, 0], [0, 0]], dtype=np.float64)
    B = np.array([[0],[0]] , dtype = np.float64)
    c = np.array([0, 0], dtype=np.float64)
    R0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    c0 = np.array([0.01, 0, 0], dtype=np.float64)
    Rf = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    cf = np.array([0, 0, 0], dtype=np.float64)
    Q0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    q0 = np.array([0, 0, 0], dtype=np.float64)
    Qf = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.float64)
    qf = np.array([-0.02, 0, 1], dtype=np.float64)
    Lu = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
    u_min = np.array([0, 0, 0, 0], dtype=np.float64)
    u_max = np.array([1, 1, 1, 1], dtype=np.float64)
    def NLrhs(model, x, u):
        n = 3
        number_of_steps = u.shape[1]
        rhs = np.empty((n,number_of_steps), dtype=object)
        rhs[0] = x[1, :number_of_steps]
        rhs[1,:number_of_steps] = 1 - 2 * u[0, :] - 0.5 * u[1, :] - 3 * u[2, :]
        rhs[2] = x[0, :number_of_steps]**2
        return rhs
    def constraint(model, x, u):
        number_of_steps = u.shape[1]
        for step in range(number_of_steps):
            model.addCons(quicksum(u[i,step] for i in range(len(u))) == 1)
    miocp = MIOCP(A, B, c, R0, c0, Rf, cf, Q0, q0, Qf, qf, Lu, u_min, u_max, NLrhs, constraint)
    run_miocp(miocp, t_max)

#F-8 Engine
def test4():
    t_max = 1
    A = np.array([[0, 0], [0, 0]], dtype=np.float64)
    B = np.array([[0],[0]] , dtype = np.float64)
    c = np.array([0, 0], dtype=np.float64)
    R0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.float64)
    c0 = np.array([0.4655, 0, 0, 0], dtype=np.float64)
    Rf = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.float64)
    cf = np.array([0, 0, 0, 0], dtype=np.float64)
    Q0 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
    q0 = np.array([0, 0, 0, 0], dtype=np.float64)
    Qf = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 100]], dtype=np.float64)
    qf = np.array([0, 0, 0, 0], dtype=np.float64)
    Lu = np.array([0])
    u_min = np.array([0], dtype=np.float64)
    u_max = np.array([1], dtype=np.float64)
    def NLrhs(model, x, u):
        n = 4
        xi = 0.05236
        number_of_steps = u.shape[1]
        rhs = np.empty((n,number_of_steps), dtype=object)
        rhs[0] = x[3, :number_of_steps] * (-0.877 * x[0, :number_of_steps] + x[2, :number_of_steps] - 0.088 * x[0, :number_of_steps] * x[2, :number_of_steps] + 0.47 * x[0, :number_of_steps]**2 - 0.019 * x[1, :number_of_steps]**2 - x[0, :number_of_steps]**2 * x[2, :number_of_steps] + 3.846 * x[0, :number_of_steps]**3 + 0.215 * xi - 0.28 * x[0, :number_of_steps]**2 * xi + 0.47 * x[0, :number_of_steps] * xi**2 - 0.63 * xi**3 - (0.215 * xi - 0.28 * x[0, :number_of_steps]**2 * xi - 0.63 * xi**3) * 2 * u[0, :])
        rhs[1] = x[3, :number_of_steps] * x[2, :number_of_steps]
        rhs[2] = x[3, :number_of_steps] * (-4.208 * x[0, :number_of_steps] - 0.396 * x[2, :number_of_steps] - 0.47 * x[0, :number_of_steps]**2 - 3.564 * x[0, :number_of_steps]**3 + 20.967 * xi - 6.265 * x[0, :number_of_steps]**2 * xi + 46 * x[0, :number_of_steps] * xi**2 - 61.4 * xi**3 - (20.967 * xi - 6.265 * x[0, :number_of_steps]**2 * xi - 61.4 * xi**3) * 2 * u[0, :])
        rhs[3] = np.zeros((1,number_of_steps))#x[3, :number_of_steps] * 0
        return rhs

        return rhs
    def constraint(model, x, u):
        number_of_steps = u.shape[1]
        for step in range( number_of_steps + 1):
            model.addCons(x[3, step] >= 0)
    miocp = MIOCP(A, B, c, R0, c0, Rf, cf, Q0, q0, Qf, qf, Lu, u_min, u_max, NLrhs, constraint)
    run_miocp(miocp, t_max)
    
# def test5():
#     #https://mintoc.de/index.php/Electric_Car
    
#     t_max=10
#     # Problem parameters
#     R_bat = 0.05
#     V_alim = 150
#     R_m = 0.03
#     K_m = 0.27
#     L_m = 0.05
#     r = 0.33
#     K_r = 10
#     M = 250
#     g = 9.81
#     K_f = 0.03
#     rho = 1.293
#     S = 2
#     C_x = 0.4
#     imax=150
    
#     A = np.array([[- R_m / L_m,-K_m / L_m,0,0], [0,0,0,0], [0,r / K_r,0,0],[0,0,0,0]],dtype=np.float64)
#     B = np.array([V_alim / L_m,0,0,0],  dtype = np.float64)
#     c = np.array([0,0,0,0], dtype=np.float64)
#     R0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
#     c0 = np.array([0,0,0,0], dtype=np.float64)
#     Rf = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
#     cf = np.array([0,0,0,0], dtype=np.float64)
#     Q0 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  dtype=np.float64)
#     q0 = np.array([0,0,0,0], dtype=np.float64)
#     Qf = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
#     qf = np.array([0,0,0,1], dtype=np.float64)
#     Lu = np.array([0], dtype=np.float64)
#     u_min = np.array([-1], dtype=np.float64)
#     u_max = np.array([1],dtype=np.float64)
    
#     def NLrhs(model, x, u):
#         n=4
#         number_of_steps = u.shape[1]
#         rhs = np.empty((n,number_of_steps), dtype=object)
#         rhs[0] =
#         rhs[1] =
#         rhs[2] =
#         rhs[3] =
#         return rhs
#     def constraint(model, x, u):
#         number_of_steps = u.shape[1]
#         for step in range(number_of_steps+1):
#             x[0,step].chgVarLb(-imax)
#             x[0,step].chgVarUb(imax)
#         for step in range(number_of_steps):
#             model.addCons(u[0,step] **2 == 1)
            
#     miocp = MIOCP(A, B, c, R0, c0, Rf, cf, Q0, q0, Qf, qf, Lu, u_min, u_max, NLrhs, constraint)
#     run_miocp(miocp, t_max) 


def run_test():
    print('testcase:'+str(testcase)+'_n_of_doms='+str(number_of_domains)+'_gamma='+str(gamma)+'_eps='+str(epsilon)+'_thrshld_x='+str(threshold_x)+'_thrshld_lam='+str(threshold_lam))
    [test1,test2,test3,test4][testcase-1]()

if __name__ == '__main__':
    if writedoc:
        with open('testcase'+str(testcase)+'/n_of_doms='+str(number_of_domains)+'_gamma='+str(gamma)+'_eps='+str(epsilon)+'_thrshld_x='+str(threshold_x)+'_thrshld_lam='+str(threshold_lam)+'_feastol'+str(feas)+'.txt', 'w') as f:
            # Redirect standard output to file
            sys.stdout = f
            # Print statements will now be written to file
            run_test()
        
            # Reset standard output to console
            sys.stdout = sys.__stdout__
    else:
        run_test()

