%%
clear,clc,close all
% Parameters
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Accuracy = 8;        %Degree of accuracy (integer, higher means more time more accurate)
N = 2^Accuracy + 1;   %Resolution
lamda = 0.1;          %Wavelength (um)
TH = 0.4;             %Threshold intensity (normalized)
NA = 0.85;            %Numerical aperature
L  = 0.09;             %Smallest possible length in the technology
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Reading mask shape from file
[File,Path] = uigetfile('*.txt');
tic
Y = importdata([Path File]);
X = Y(:,1); Y = Y(:,2);
num_points = length(X)-1;
sides = zeros(num_points,1);
for i=1:num_points
    sides(i) = sqrt((X(i)-X(i+1))^2 + (Y(i)-Y(i+1))^2);
end
scale = L/min(sides);
sides = sides*scale;
X = scale*X; Y = scale*Y;
X = X-min(X); Y = Y-min(Y);
Lx = 1.4*max(X);   %Mask real dimensions (um)
Ly = 1.4*max(Y);   %Mask real dimensions (um)
% Center the polygon
x = linspace(-0.2*max(X),1.2*max(X),N);
y = linspace(-0.2*max(Y),1.2*max(Y),N);
% Create mask matrix
[x,y] = meshgrid(x,y);
x=reshape(x,[],1);
y=reshape(y,[],1);
mask = inpolygon(x,y,X,Y);
mask = reshape(mask,N,N);
mask = double(mask);

%% Calculations
% Creating different domains
dx = Lx/(N-1); dy = Ly/(N-1);
[nx,ny] = meshgrid(-(N-1)/2:(N-1)/2 , -(N-1)/2:(N-1)/2); % Create discritized domain
fx=(1/dx)*(1/N)*nx; fy=(1/dx)*(1/N)*ny;  % Discrete frequency domain (1/um)
I = fft2( mask ); I = fftshift(I);
% objective lens pupil function 
P = sqrt((fx.^2)+(fy.^2));
P = double(P < (NA/lamda));
I = ifft2(P.*I);
I = real( I .* conj( I ) );
I = I/max(max(I));
aerial = double(I > TH) ; 

%% Calculate error
mError=sum(abs(mask-aerial));
mError=100*(mError/sum(mask));
disp(['error = ' num2str(mError) '%']);

%% Plotting
% figure(),imagesc(mask);
% axis( 'equal' ); title( 'Mask image' );
% figure(),imagesc(aerial);
% axis( 'equal' ); title( 'Aerial image' );
figure(),mesh(nx,ny,I)
title( 'Normalized light intensity' );
%Draw boudary
[r,c] = find(mask,1,'first');
Bm = bwtraceboundary(mask,[r,c],'S');
[r,c] = find(aerial,1,'first');
Ba_M = bwtraceboundary(aerial,[r,c],'S');
figure();
plot(Bm(:,1),Bm(:,2),'k',...
     Ba_M(:,1),Ba_M(:,2),'r--')
axis( 'equal' ); legend('Mask','Aerial image')
toc