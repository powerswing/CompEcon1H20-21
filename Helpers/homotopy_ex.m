% plot homotopy function
clear all; close all;
clc
%% Frame: limits and zero line
x = linspace(0,3,101)';
e1 = linspace(-5,3,101)';
fig = figure;
plot(x, zeros(size(x)), 'k', 'Linewidth', 2); grid on; hold on; xlim([min(x),max(x)]); ylim([min(e1),max(e1)]);
xlabel('x'); ylabel('f(x)');

%% Function H1 and H0:
h0 = x;
h1 =(2*x-4+sin(pi*x));
%% t=0, t=1
hs = plot(x,h0,x,h1, 'Linewidth', 2);


%% H fixed-point with initial value = 0
nn = 11; %a few plots
for j = 1:nn
    t = (j-1)/(nn-1);
    f0 = @(x0) (1-t)*x0 + t*(2*x0-4+sin(pi*x0));
    y0 = f0(x);
 if rem(t,0.2) == 0 %&& t > 0 && t < 1 plots only between 0 and 1. 0.2 is delta t. try different values
       plot(x, y0, 'k--', 'Linewidth', 2); %add plots to current figure
  end
end
%%
legend(hs,{'t = 0','t = 1'},'Location','NorthWest')
% add arrow
xa = [0.7 0.59];
ya = [0.4 0.61];
ta = annotation('textarrow',xa,ya,'String','t=0.6');
ta.FontSize = 11;
ta.TextLineWidth=0.8;
figname = 'homotopy1.jpg';
saveas(gcf,figname); %save current figure


%% Solution path: foreach t all x such H=0
% In this example we create a new loop, but it is more efficient to include
% everyting in one loop
nn = 101;
xini = 0; xsolvec = zeros(nn,2);
for j = 1:nn
    t = (j-1)/(nn-1);
    f0 = @(x0) (1-t)*x0 + t*(2*x0-4+sin(pi*x0));  
    [xsol] = fsolve(f0,xini); %solve system of nonlinear equations
%     [xsol,fvsol,efsol] = fsolve(f0,xini); 
    xsolvec(j,:) = [xsol,t];
    xini = xsol;    
end
% plot solution path:
h = figure;
plot(xsolvec(:,1), xsolvec(:,2), 'r', 'Linewidth', 2); grid on; hold on; xlim([min(x),max(x)]); ylim([0,1]);
xlabel('x'); ylabel('t');

figname = 'homotopy1_path.jpg';
saveas(gcf,figname); %save current figure