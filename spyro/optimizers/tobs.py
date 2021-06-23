########## TOBS solver ##########
import numpy as np
import cplex

def TOBS(dfdx, dgdx, gbar, gi, epsilons, beta, x):
    # Truncation error (for flip limits constraint)
    C1 =  (x.flatten(order='F')==0).astype('int')
    C2 = -(x.flatten(order='F')==1).astype('int')
    truncation = C1 + C2
    # Normalization of sensitivities
    norm = np.maximum(np.abs(dgdx.flatten(order='F')).max(),
                      np.finfo('float').eps)
    dgdx /= norm
    # Constraints sensitivities
    dgdx = np.vstack((dgdx.reshape((1,dgdx.size), order='F'),
                      truncation.reshape((1, truncation.size), order='F')))
    # Constraint relaxation (move limit)
    target = (gbar - gi)/norm
    deltag = epsilons*np.abs(gi)/norm
    A = (target > deltag).astype('int')
    B = (target < -deltag).astype('int')
    constlimits = (A-B)*deltag + (1-(A+B))*target
    constlimits = np.vstack((constlimits, beta*x.size))
    # Variable limits
    lower_limits = -(np.abs(x.flatten(order='F') - 1) < 0.001).astype('int')
    upper_limits = (np.abs(x.flatten(order='F')) < 0.001).astype('int')
    # prepare linear constrains
    ctype = dgdx.shape[1]*'I'
    rows = []
    for row in dgdx:
        rows.append([np.arange(dgdx.shape[1]).tolist(), row])
    # Update CPLEX
    slip_problem = cplex.Cplex()
    slip_problem.set_results_stream(None)
    slip_problem.objective.set_sense(slip_problem.objective.sense.minimize)
    slip_problem.variables.add(obj=np.array(dfdx.flatten(order='F')).squeeze(),
                               lb = lower_limits.tolist(),
                               ub = upper_limits.tolist(),
                               types = ctype)
    slip_problem.linear_constraints.add(lin_expr=rows,
                                        senses=dgdx.shape[0]*'L',
                                        rhs=constlimits.flatten())
    slip_problem.solve()
    
    # import IPython; IPython.embed(); exit()

    deltax = slip_problem.solution.get_values()
    x += np.array(deltax).reshape(x.shape, order='F')
    return x

