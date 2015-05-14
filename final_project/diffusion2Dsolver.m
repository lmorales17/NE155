function [flux,iterations,T] = diffusion2Dsolver(a,sigtr,siga,source,method)
tic;

% checking input values for correctness. no values can be negative 
if a <= 0
    error('Error: Invalid a')
elseif any(sigtr < 0)
    error('Error: Invalid sigtr')
elseif any(siga < 0) 
    error('Error: Invalid siga')
elseif source < 0 
    error('Error: Invalid nuf')
else
    fprintf('Input check successful')
end

a = a
sigTr = sigtr
sigA = siga 
Source = source 

% sigTr and sigA are input vectors containing cross section information about the
% materials
n = ceil(a);
D = 1/(3*sum(sigtr))*ones(n+1,n+1);
sigA = sum(siga)*ones(n+1,n+1);
delt = 0.1*ones(1,n+1);
eps = 0.1*ones(1,n+1);


% setting up discretized equations for finite volume matrix for iteration
aL = zeros(n);
aR = zeros(n);
aB = zeros(n);
aT = zeros(n);
aC = zeros(n);
for i = 1:n
    for j = 1:n
        aL(i,j) = -(D(i,j)*eps(j) + D(i,j+1)*eps(j+1))/(2*delt(i));
        aR(i,j) = -(D(i,j)*eps(j) + D(i+1,j+1)*eps(j+1))/(2*delt(i+1));
        aB(i,j) = -(D(i,j)*delt(i) + D(i+1,j)*delt(i+1))/(2*eps(j));
        aT(i,j) = -(D(i,j+1)*delt(i) + D(i+1,j+1)*delt(i+1))/(2*eps(j+1));
        aC(i,j) = sigA(i,j) - (aL(i,j) + aR(i,j) + aB(i,j) + aT(i,j));
    end
end

% defining the source vector 
S = cell(n,1);   
for k = 1:n
    S{k} = source*ones(n,1);
end

% left boundary conditions: vacuum fluxL = 0
for j = 2:n
    S{j}(1) = 0;
    aR(1,j) = 0;
    aT(1,j) = 0;
    aB(1,j) = 0;
    aC(1,j) = 1;
end

% bottom boundary condition: vacuum fluxB = 0
for i = 1:n-1
    S{1}(i) = 0;
    aR(i,1) = 0;
    aL(i,1) = 0;
    aT(i,1) = 0;
    aC(i,1) = 1;
end

% right boundary condition: reflecting 
for j = 2:n
    aL(n,j) = -(D(n-1,j)*eps(j) + D(n-1,j-1)*eps(j-1))/(2*delt(n-1));
    aB(n,j) = -D(n-1,j)*delt(j)/(2*eps(j));
    aT(n,j) = -D(n-1,j-1)*delt(j)/(2*eps(j+1));
    aC(n,j) = sigA(n,j) - (aL(n,j) + aB(n,j) + aT(n,j));
end
    
% top boundary condition: reflecting 
for i = 1:n-1
    aR(i,n) = -D(i,n)*eps(n)/(2*delt(i));
    aB(i,n) = -(D(i,n)*delt(i) + D(i+1,n)*delt(i+1))/(2*eps(n));
    aL(i,n) = -D(i+1,n)*eps(n)/(2*delt(i+1));
    aC(i,n) = sigA(i,n) - (aR(i,n) + aB(i,n) + aL(i,n));
end

% Center matrix that will go on the diagonals 
C = cell(n,1);
for k = 1:n
    for i = 2:n-1
        C{k}(i,i) = aC(i,k);
        C{k}(i,i+1) = aR(i,k);
        C{k}(i,i-1) = aL(i,k);
    end
    C{k}(1,1) = aC(1,k);
    C{k}(1,2) = aR(1,k);
    C{k}(n,n) = aC(n,k);
    C{k}(n,n-1) = aL(n,k);
end

% top matrix for upper diagonal
T = cell(n,1);
for i = 1:n
    for k = 1:n
        T{k}(i,i) = aT(i,k);
    end
end

% bottom matrix for lower diagonal
B = cell(n,1);
for i = 1:n
    for k = 1:n
        B{k}(i,i) = aB(i,k);
    end
end

% building matrix A
A = cell(n,n);
for i = 1:n
    for j = 1:n;
        A{i,j} = zeros(n);
    end
end
for k = 1:n
    A{k,k} = C{k};
end
for k = 1:n-1
    A{k,k+1} = T{k};
end
for k = 2:n
    A{k,k-1} = B{k};
end

A = cell2mat(A);
S = cell2mat(S);


% solve using gauss seidel iteration
b = S;
x = zeros(size(b));
N = n*n;
tol = 10^-5;

switch lower(method)
    case 'gauss seidel'
        for k = 1:10000000;
            for i= 1:N
                x(i) = 1/A(i,i)*(b(i) - A(i,1:i-1)*x(1:i-1) - A(i,i+1:N)*x(i+1:N));
            end
            if any(isnan(x))
                error('Undefined Flux')
            end
            M{k} = reshape(x,n,n);
            if max(abs(A*x-b))<tol
                iter = k;
                break;
            end 
        end
    case 'sor'
        w = 1.2;
        for k = 1:100000;
            for i= 1:N
                x(i) =  (1-w)*x(i) + w/A(i,i)*(b(i) - A(i,1:i-1)*x(1:i-1) - A(i,i+1:N)*x(i+1:N));
            end
            if any(isnan(x))
                error('Undefined Flux')
            end
            M{k} = reshape(x,n,n);
            if max(abs(A*x-b))<tol
                iter = k;
                break;
            end
        end
    case 'jacobi'        
        for k = 1:100000;
            for i = 1:N
                y(i)=1/A(i,i)*(b(i)-A(i,1:i-1)*x(1:i-1)-A(i,i+1:N)*x(i+1:N));
            end
            if any(isnan(y))
                error('Undefined Flux')
            end
            M{k} = reshape(x,n,n);
            if max(abs(A*y'-b))<tol
                iter = k;
                break;
            end
            x = y';
        end
    otherwise
        error('Unknown iterative method.')
end

flux = reshape(x,n,n);
iterations = iter;
figure(1)
surf(flux)
title('Flux for Fixed Source Diffusion Equation','fontsize',18)
zlabel('Magnitude of Flux','fontsize',18)

% solves for flux directly 
fluxd = A\b;
fluxd = reshape(fluxd,n,n);

% calculates and plots the difference between the iterative solution and
% the direct solution 
Error = fluxd-flux;
figure(2)
plot(Error)
for i=1:length(Error)
   leg(i)={['\Phi ' num2str(i)]};
end
legend(leg,'location','northwest');
title('Error between Iterative and Direct Solution','fontsize',18)
xlabel('Length of Core','fontsize',18)
ylabel('Error','fontsize',18)

% creates and saves a video of the flux converging and saves as an avi file
writerObj = VideoWriter('flux.avi');
open(writerObj);

figure(3)
surf(flux)
axis tight
set(gca,'nextplot','replacechildren');
set(gcf,'Renderer','zbuffer');
for k =1:floor(length(M)/4)
    surf(M{k})
    frame = getframe;
    writeVideo(writerObj,frame);
end

close(writerObj);

T = toc;

end 




    
