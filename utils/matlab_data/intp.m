% File name: intp.m
% Author: Anton Lukashchuk <anton.lukashchuk@epfl.ch>
%
% This file is part of REDE project (https://github.com/akarazeev/REDE)

% interpolation by myself

w = data(:,1);
m = data(:,3);
sz = max(size(data(:,1)));

c = 2.99792458e8;

T0 = 282e12;

N = 20000;

m_total = linspace(min(m),max(m),N);%round(min(m_sample)):1:round(max(m_sample)-10);
omega_total = interpn(m,w,m_total,'spline');

h = (max(m)-min(m))/(N-1);
D1_total = diff(omega_total)/h;
D2_total = diff(D1_total)/h;


m_total = m_total(1:end-1);
omega_total = omega_total(1:end-1);

[~ , ind] = min(abs(T0 - omega_total) );
w0 = omega_total(ind);
m0 = m_total(ind);
D1 = D1_total(ind);

omega_grid = w0 + D1.*(m_total-m0);

delta = mod(omega_total - omega_grid,D1);
delta_omega_total = delta - sign(delta-D1/2).*D1.*(abs(delta)>D1/2);

% delta_omega_total = omega_total - omega_grid;

lambda_grid = c./omega_total;


figure(4);
hold on
plot(omega_total*1e-12,D1_total)%,'.')
ylabel('D_1/2\pi (Hz)')
xlabel('Frequency (THz)')
title('MFree spectral range in SiN resonator')

figure(5);
hold on
plot(omega_total(2:end)*1e-12,D2_total)%,'.')
ylabel('D_2/2\pi (Hz)')
xlabel('Frequency (THz)')
title('Modal dispersion in SiN resonator')

figure(6)
hold on
plot(omega_total*1e-12,delta_omega_total*1e-9,'.');
ylabel('Mode deviation (GHz)')
xlabel('Frequency (THz)')
title('Modal spectral deviation in SiN resonator')

% plot(freq_grid*1e-12,delta_omega_total/(2*pi))

% figure(7)
% hold on
% plot(c./(omega_total/2/pi*f0*1e-12)*1e-6,delta_omega_total/(2*pi)*1e-9)%,'.')
% ylabel('Mode deviation (Hz)')
% xlabel('Wavelength (um)')
% title('Modal spectral deviation in SiN resonator')
