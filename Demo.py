import numpy as np
import matplotlib.pyplot as plt 

#####################################################################
#Scroll to the end for examples, uncomment the function as needed
#####################################################################

#####################################################################
#Code for the multiquadric
#####################################################################
def MQ(point,c):
    return np.sqrt(c**2+np.linalg.norm(point)**2)

#####################################################################
#The first derivative of the function sqrt(x+1)
#Note that we work with the MQ defined as 
#f(a) = sqrt(a+1)
#then input a = x^2
#####################################################################
def f_dash(x):
    denom = np.sqrt(x+1)
    return 0.5/denom

#####################################################################
#Code returning the RBF matrix
#Returns the nxn RBF matrix
#####################################################################
def phi(x,n,c):
    phi = np.ones((n,n))
    for i, point in enumerate(x):
        for j in range(i,n):
            MQ_dist = MQ((point-x[j]),c)
            phi[i,j] = MQ_dist
            phi[j,i] = MQ_dist
    return phi

#####################################################################
# Marchenko-Pastur function
# x = x value
# g = gamma value which should be equal to n/d
# sigma = variance of underlying distribution
#####################################################################
def marchpast(x, g, sig):
    "Marchenko-Pastur distribution"
    def max_0(a):
        "Element wise maximum of (a,0)"
        return np.maximum(a, np.zeros_like(a))
    g_plus = sig*(1+g**0.5)**2
    g_minus = sig*(1-g**0.5)**2
    return np.sqrt(max_0(g_plus - x) * max_0(x- g_minus)) /(2*np.pi*g*x*sig)


######################################################################
# Generating the MP distribution for the RBF matrix for normally distributed points
# Returns a histogram of eigenvalues and the MP distribution plotted on top
# n = number of points
# d = dimension of points
# c = shape parameter
# k is the distance between points, squared.
# for normally distributed points it's 2*d-1
# sigma is the variance. Here it's 1 (of course)
# number = number of RBF matrices generated
# usuallly I use small values for n and d, and ~500 iterations.
#####################################################################
def MP_plotter_normal(n,d,c,number):

    #Generating the RBF matrices
    matrices=  []
    for i in range(number):
        points = np.random.normal(0,1,size=(n,d))
        matrices.append(phi(points,n,c))
    
    #constants for transforming the MP distribution
    sigma = 1
    k = 2*d-1
    Const1 = c - MQ(np.sqrt(k),c) + k*f_dash(k)
    Const2 = 2*f_dash(k)*d

    #Calculating and plotting histogram and transformed MP distribution
    eigvals = np.sort(np.linalg.eigvals(matrices).real.ravel())

    #We do not plot the outlying eigenvalue from each RBF matrix
    #otherwise the histogram looks terrible.
    eigvals = eigvals[:-number]

    #plotting the histogram
    nn, dummy1, dummy2=plt.hist(eigvals.ravel(), 
                                bins="auto", 
                                density=True,color = 'coral')
    
    #MP dist plotting
    gamma = n/d
    x=np.arange(eigvals[0]-1, eigvals[-1]+1, 0.003)
    plt.plot(x, (1/Const2)*marchpast(-(x-Const1)/(Const2), gamma,sigma),color='indigo')
    plt.ylim(top=nn[1:].max() * 1.4)
    plt.show()


######################################################################
# Generating the MP distribution for the RBF matrix for normally distributed points
# Here we 'normalise' the distribution first by dividing by \sqrt{2d-1}
# Note that this changes the value of sig and we can now use k=1
# This also changes sigma to now be 1/(2*d-1)
# Returns a histogram of eigenvalues and the MP distribution plotted on top
# n = number of points
# d = dimension of points
# c = shape parameter
# number = number of RBF matrices generated
# usuallly I use small values for n and d, and ~500 iterations.
#####################################################################
def Normalised_MP_plotter_normal(n,d,c,number):

    #Generating the RBF matrices
    matrices=  []
    for i in range(number):
        points = np.random.normal(0,1,size=(n,d))/np.sqrt(2*d-1)
        matrices.append(phi(points,n,c))
    
    #constants
    k = 1
    sigma = 1/(2*d-1)
    Const1 = c - MQ(np.sqrt(k),c) + k*f_dash(k)
    Const2 = 2*f_dash(k)*d

    #Calculating and plotting histogram and transformed MP distribution
    #We do not plot the outlying eigenvalue from each RBF matrix
    eigvals = np.sort(np.linalg.eigvals(matrices).real.ravel())

    #We do not plot the outlying eigenvalue from each RBF matrix
    eigvals = eigvals[:-number]

    #plotting the histogram
    nn, dummy1, dummy2=plt.hist(eigvals.ravel(), 
                                bins="auto", 
                                density=True,color = 'coral')
    
    #MP dist plotting
    gamma = n/d
    x=np.arange(eigvals[0]-1, eigvals[-1]+1, 0.003)
    plt.plot(x, (1/Const2)*marchpast(-(x-Const1)/Const2, gamma,sigma),color='indigo')
    plt.ylim(top=nn[1:].max() * 1.4)
    plt.show()



######################################################################
# Generating the MP distribution for the RBF matrix for the uniform distribution
# in the unit ball
# Returns a histogram of eigenvalues and the MP distribution plotted on top
# n = number of points
# d = dimension of points
# c = shape parameter
# number = number of RBF matrices generated
# usuallly I use small values for n and d, and ~500 iterations.
# k=2 here and sigma = 1/(d+2), we can just use 1 if you like.
#####################################################################
def MP_plotter_ball(n,d,c,number):

    #Generating the RBF matrices
    matrices=  []
    for i in range(number):
        #This bit of code generates an nxd array
        #where each row is a point in the unit ball.
        #data here is distributed uniformly.
        cube = np.random.standard_normal(size=(n, d))
        norms = np.linalg.norm(cube,axis=1)
        surface_sphere = cube/norms[:,np.newaxis]
        scales = np.random.uniform(0,1, size= n)
        points = surface_sphere* (scales[:, np.newaxis])**(1/d)
        matrices.append(phi(points,n,c))
    
    #constants
    k = 2
    sigma = 1/(d+2)
    Const1 = c - MQ(np.sqrt(k),c) + k*f_dash(k)
    Const2 = 2*f_dash(k)*d
    
    print(-Const2*sigma*(1+np.sqrt(n/d))**2+Const1)
    print(-Const2*sigma*(1-np.sqrt(n/d))**2+Const1)

    
    #print(1/(-Const2*sigma*(1-np.sqrt(n/d))**2+Const1))

    #Calculating and plotting histogram and transformed MP distribution
    #We do not plot the outlying eigenvalue from each RBF matrix
    eigvals = np.sort(np.linalg.eigvals(matrices).real.ravel())

    #We do not plot the outlying eigenvalue from each RBF matrix
    eigvals = eigvals[:-number]

    #plotting the histogram
    nn, dummy1, dummy2=plt.hist(eigvals.ravel(), 
                                bins="auto", 
                                density=True,color = 'coral')
    
    #MP dist plotting
    gamma = n/d
    x=np.arange(eigvals[0]-1, eigvals[-1]+1, 0.003)
    plt.plot(x, (1/Const2)*marchpast(-(x-Const1)/Const2, gamma,sigma),color='indigo')
    plt.ylim(top=nn[1:].max() * 1.4)
    plt.show()


############################################################################
#Generating scatter plots of eigenvalues of RBF matrix & approximant
#This time we want to use a larger value of n and d
#We also include the outlying eigenvalue
#We do this for the unit ball
############################################################################
def eigvals_plotter(n,d,c):
    #Generating the eigenvalues of the RBF matrix
    cube = np.random.standard_normal(size=(n, d))
    norms = np.linalg.norm(cube,axis=1)
    surface_sphere = cube/norms[:,np.newaxis]
    scales = np.random.uniform(0,1, size= n)
    points = surface_sphere* (scales[:, np.newaxis])**(1/d)

    matrix = phi(points,n,c)
    actual_eigvals = np.sort(np.linalg.eigvals(matrix))    

    #Generating the eigenvalues of the approximant matrix
    #constants
    k = 2
    Const1 = c - MQ(np.sqrt(k),c) + k*f_dash(k)
    Const2 = 2*f_dash(k)
    Const3 = MQ(np.sqrt(k),c)

    approx_matrix = Const1*np.eye(n) - Const2*points@points.T + Const3*np.ones((n,n))
    approx_eigvals = np.sort(np.linalg.eigvals(approx_matrix))

    #Plotting
    xes = range(len(approx_eigvals))
    plt.scatter(xes, actual_eigvals, label='Actual eigenvalues')
    plt.scatter(xes, approx_eigvals,marker='.', label ='Approx eigenvalues')
    plt.legend()
    plt.show()

############################################################################
#EXAMPLE 1 - normal distribution using MP_plotter_normal(n,d,c,number)
#n = 30, d=100, c=1 and we generate number = 300 matrices for their eigenvalues.
#this gives us gamma = 0.3
############################################################################
#MP_plotter_normal(15,15,1,300)

############################################################################
#EXAMPLE 2 - 'normalised' normal distribution using Normalised_MP_plotter_normal(n,d,c,number)
#n = 30, d=100, c=1 and we generate number = 300 matrices for their eigenvalues.
#this gives us gamma = 0.3
############################################################################
#Normalised_MP_plotter_normal(10,1000,1,300)

############################################################################
#EXAMPLE 3 - unit d-ball using MP_plotter_ball(n,d,c,number)
#n = 30, d=100, c=1 and we generate number = 300 matrices for their eigenvalues.
#this gives us gamma = 0.3
############################################################################
#MP_plotter_ball(10,2000,1,300)

############################################################################
#EXAMPLE 4 - unit d-ball, plotting the whole spectrum, using eigvals_plotter(n,d,c)
#Here we use a larger value of n as we only generate one matrix
############################################################################
#eigvals_plotter(200,300,1)
