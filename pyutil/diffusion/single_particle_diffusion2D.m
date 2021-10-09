% This is to illustrate diffusion of particles (i.e., Brownian motion)
clear

nd = 2; % number of dimensions
N = 1000;  %number of time steps
K = 100;    %number of particles
D = 1/2;   % diffusion coefficient

dt = .025;  % time step
diff = sqrt(2*D*dt);

% initialize positions of particles
x = zeros(K,nd);
X(:,:,1) = x;

% output vectors
var(1,:) = zeros(1,2);
T(1) = 0;

% vectors for plots
y = [-100:100]/5;
binc = [-20:20];
ksk = 20;
est1 = 0;
est2 = 0;
%L = 5;
%k = 1;

t = 0;
for j = 2:N %loop through time steps
    
    x = x + diff*randn(K,nd);
    X(:,:,j) = x;
    var(j,:) = std(x,0,1).^2;
      t = t +dt;
      %est1 = est1 + t^2;
      %est2 = est2 + t*var(j);
    T(j) = t;   

    
if(fix(j/ksk)*ksk==j)
    figure(1)
    subplot(2,1,1)
    hist(x(:,1),binc)
    title('Histogram of particle x positions','fontsize',20)
    hold on
  p = K*exp(-y.^2/(4*D*t))/sqrt(4*pi*D*t);
plot(y,p,'r','linewidth',2)
hold off
axis([-20 20 0 K/2])

subplot(2,1,2)
    hist(x(:,2),binc)
    title('Histogram of particle y positions','fontsize',20)
    hold on
  p = K*exp(-y.^2/(4*D*t))/sqrt(4*pi*D*t);
plot(y,p,'r','linewidth',2)
hold off
axis([-20 20 0 K/2])
end

    
    %axis([-20 20 0 250])
     pause(0.01)
end
 
% plot positions in 2D
 figure(2)
 hold on
 for k=1:10
    xplot = squeeze(X(k,1,:));
    yplot = squeeze(X(k,2,:));
    plot(xplot,yplot,'linewidth',2)
    plot(xplot(end),yplot(end),'*k')
 end
xlabel('x','fontsize',20)
ylabel('y','fontsize',20)
title('Sample walker trajectories')
hold off

figure(3)
hold on
 for k=1:K
    xplot = squeeze(X(k,1,end));
    yplot = squeeze(X(k,2,end));
    plot(xplot(end),yplot(end),'*k')
 end
xlabel('x','fontsize',20)
ylabel('y','fontsize',20)
title('Endpoints of walkers')
hold off

% figure(3)
% sl = est2/est1;
% plot(T,var,T,sl*T,'--', 'linewidth',2 )
% hold off
% xlabel('t','fontsize',20)
% ylabel('Variance','fontsize',20)

