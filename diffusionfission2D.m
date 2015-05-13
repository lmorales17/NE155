% defining necessary physical constants, mesh spacing, and the size of the nxm matrix 
n = 10;
m = 10;
D = ones(n+1,m+1);
sigA = 0.6*ones(n+1,m+1);
nuF = 0.7;
delt = 0.1*ones(1,n+1);
eps = 0.1*ones(1,m+1);

% setting up discretized equations for finite volume matrix for iteration
aL = zeros(n,m);
for i = 1:n
    for j = 1:m
        aL(i,j) = -(D(i,j)*eps(j) + D(i,j+1)*eps(j+1))/(2*delt(i));
    end
end
aR = zeros(n,m);
for i = 1:n
    for j = 1:m
        aR(i,j) = -(D(i,j)*eps(j) + D(i+1,j+1)*eps(j+1))/(2*delt(i+1));
    end
end
aB = zeros(n,m);
for i = 1:n
    for j = 1:m
        aB(i,j) = -(D(i,j)*delt(i) + D(i+1,j)*delt(i+1))/(2*eps(j));
    end 
end
aT = zeros(n,m);
for i = 1:n
    for j = 1:m
        aT(i,j) = -(D(i,j+1)*delt(i) + D(i+1,j+1)*delt(i+1))/(2*eps(j+1));
    end
end
aC = zeros(n,m);
for i = 1:n
    for j = 1:m
        aC(i,j) = sigA(i,j) - (aL(i,j) + aR(i,j) + aB(i,j) + aT(i,j));
    end
end

% defining Q for power iteration as the fission source term
x = ones(n,1);
Q = cell(n,1);   
for k = 1:n
    Q{k} = nuF*x;
end
% left boundary conditions: vacuum fluxL = 0
for j = 1:m
    Q{j}(1) = 0;
    aR(1,j) = 0;
    aT(1,j) = 0;
    aB(1,j) = 0;
    aC(1,j) = 1;
end
% bottom boundary condition: vacuum fluxB = 0
for i = 1:n
    Q{1}(i) = 0;
    aR(i,1) = 0;
    aL(i,1) = 0;
    aT(i,1) = 0;
    aC(i,1) = 1;
end
% right boundary condition: reflecting 
for j = 2:m
    aL(n,j) = -(D(n-1,j)*eps(j) + D(n-1,j-1)*eps(j-1))/(2*delt(n-1));
    aB(n,j) = -D(n-1,j)*delt(j)/(2*eps(j));
    aT(n,j) = -D(n-1,j-1)*delt(j)/(2*eps(j-1));
    aC(n,j) = sigA(n,j) - (aL(n,j) + aB(n,j) + aT(n,j));
end
% top boundary condition: reflecting 
for i = 1:n-1
    aR(i,m) = -D(i,m)*eps(m)/(2*delt(i));
    aB(i,m) = -(D(i,m)*delt(i) + D(i+1,m)*delt(i+1))/(2*eps(m));
    aL(i,m) = -D(i+1,m)*eps(m)/(2*delt(i+1));
    aC(i,m) = sigA(i,m) - (aR(i,m) + aB(i,m) + aL(i,m));
end

% setting up center matrix which will be the diagonals of A
C = cell(n,1);
for k = 1:n
    for i = 2:n-1
        for j = 2:n-1
            C{k}(i,i) = aC(i,k);
            C{k}(i,i+1) = aR(i,k);
            C{k}(i,i-1) = aL(i,k);
        end
    end
    C{k}(1,1) = aC(1,k);
    C{k}(1,2) = aR(1,k);
    C{k}(n,n) = aC(n,k);
    C{k}(n,n-1) = aL(n,k);
end
% the top matrix which will be the upper triangular diagonals of A
T = cell(n,1);
for i = 1:n
    for j = 1:n
        for k = 1:n
            T{k}(i,i) = aT(i,k);
        end
    end
end
% bottom matrix for the lower triagonal diagonals of A
B = cell(n,1);
for i = 1:n
    for j = 1:n
        for k = 1:n
            B{k}(i,i) = aB(i,k);
        end
    end
end
% final matrix setup for A with C,T, and B
A = cell(n,n);
for k = 1:n
    for m = 1:n
        A{k,m} = zeros(n);
        A{k,k} = C{k};
    end
end
for k = 1:n-1
    A{k,k+1} = T{k};
end
for k = 2:n
    A{k,k-1} = B{k};
end

A = cell2mat(A);
Q = cell2mat(Q);

% solve using gauss seidel iteration and defining necessary terms 
Q0 = Q;
k0 = 1;
b = 1/k0*Q0;
m = 1;
N = n*n;
x = ones(N,1);
tol = 10^-5;
h = delt(1);

for m = 1:100000;
    for i= 1:N
        x(i) = 1/A(i,i)*(b(i) - A(i,1:i-1)*x(1:i-1) - A(i,i+1:N)*x(i+1:N));
        Q = nuF*x;
        sumQ0 = 0;
        sumQ = 0;
        for j = 1:N-1;
            sumQ0 = sumQ0 + Q0(j)*h/2;
            sumQ = sumQ + Q(j)*h/2;
        end
        k = k0*(Q(1)*h/2 + sumQ)/(Q0(1)*h/2 + sumQ0);
        b = 1/k*Q;     
    end
    if max(abs(A*x-b))<tol
        iter = m;
        break;
    end
    k0 = k;
end
% converts flux from vector to matrix so it can be graphed as a 3d surface
flux = reshape(x,n,n)
surf(flux)
