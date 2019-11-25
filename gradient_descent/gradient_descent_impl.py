def next_step(gamma,d_function,x):
    eval_d_function = d_function(*x)
    return [x[i] - gamma*(eval_d_function[i]) for i in range(len(x))]
    

def gradient_descent(gamma,function,d_function,x_0 ,max_steps = 500):
    x_iters = []
    x_n = x_0
    x_n_1 = next_step(gamma,d_function, x_n )
    tol = 10**-4
    i = 0 
    while abs(function(*x_n_1)-function(*x_n))>tol and i<max_steps and function(*x_n_1)!=float("inf") :

        x_iters.append(x_n)
        x_n = x_n_1
        x_n_1 = next_step(gamma,d_function, x_n )
        i+=1
        
    if i>=max_steps or function(*x_n_1)==float("inf"):
        print(i)
        print("The algorithm didn't converge")
        
    return x_iters
